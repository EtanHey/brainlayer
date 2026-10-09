import io
import json
import sqlite3

import pytest

from brainlayer.pipeline.relation_inference import local_caller, restrict_to_conversations

MODEL = "explicit-local-model"
PROMPT = 'rules\nINPUT: [{"chunk_id":"long-source-uuid","content":"Atlas uses SQLite.","entities":[{"id":"project-uuid","name":"Atlas","type":"project"},{"id":"tool-uuid","name":"SQLite","type":"technology"}]}]'


def envelope(*, finish="stop", model=MODEL, source="Atlas"):
    content = {
        "relations": [
            {
                "source_name": source,
                "target_name": "SQLite",
                "type": "uses",
                "temporal_status": "current",
                "quote": "Atlas uses SQLite.",
            }
        ]
    }
    return io.BytesIO(
        json.dumps(
            {"model": model, "choices": [{"finish_reason": finish, "message": {"content": json.dumps(content)}}]}
        ).encode()
    )


def test_wire_uses_names_only_but_restores_real_provenance(monkeypatch):
    def post(request, timeout):
        assert request.full_url == "http://127.0.0.1:8183/v1/chat/completions"
        payload = json.loads(request.data)
        assert payload["model"] == MODEL
        assert payload["messages"][0]["role"] == "system"
        chunk = json.loads(payload["messages"][1]["content"])
        assert chunk == {
            "source_text": "Atlas uses SQLite.",
            "entities": [
                {"name": "Atlas", "type": "project"},
                {"name": "SQLite", "type": "technology"},
            ],
        }
        assert "long-source-uuid" not in request.data.decode()
        assert "project-uuid" not in request.data.decode()
        return envelope()

    monkeypatch.setattr("brainlayer.pipeline.relation_inference._open_local", post)
    result = json.loads(local_caller("http://127.0.0.1:8183", MODEL)(PROMPT))["chunks"][0]
    assert result["chunk_id"] == "long-source-uuid"
    assert result["relations"][0]["source_id"] == "project-uuid"
    assert result["relations"][0]["target_id"] == "tool-uuid"


@pytest.mark.parametrize(
    "kwargs,error",
    [({"finish": "length"}, RuntimeError), ({"model": "wrong"}, RuntimeError), ({"source": "invented"}, ValueError)],
)
def test_incomplete_wrong_model_or_invented_alias_fails_closed(monkeypatch, kwargs, error):
    monkeypatch.setattr("brainlayer.pipeline.relation_inference._open_local", lambda *a, **k: envelope(**kwargs))
    with pytest.raises(error):
        local_caller("http://127.0.0.1:8183", MODEL)(PROMPT)


def test_invalid_output_gets_one_model_correction_never_a_synthetic_empty(monkeypatch):
    calls = []

    def post(request, timeout):
        calls.append(json.loads(request.data))
        return envelope(source="invented") if len(calls) == 1 else envelope()

    monkeypatch.setattr("brainlayer.pipeline.relation_inference._open_local", post)
    result = local_caller("http://127.0.0.1:8183", MODEL)(PROMPT)
    assert json.loads(result)["chunks"][0]["relations"]
    assert len(calls) == 2
    assert "Validation failed" in calls[1]["messages"][-1]["content"]


def test_raw_trace_keeps_rejected_proposals_before_correction(monkeypatch):
    monkeypatch.setattr(
        "brainlayer.pipeline.relation_inference._open_local", lambda *a, **k: envelope(source="invented")
    )
    events = []
    with pytest.raises(ValueError):
        local_caller("http://127.0.0.1:8183", MODEL, on_response=events.append)(PROMPT)
    assert [event["attempt"] for event in events] == [1, 2]
    assert all('"source_name": "invented"' in event["response"]["choices"][0]["message"]["content"] for event in events)
    assert len(events[0]["request"]["messages"]) == 2
    assert len(events[1]["request"]["messages"]) == 4
    assert all(event["chunk_id"] == "long-source-uuid" and len(event["window_sha256"]) == 64 for event in events)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "http://127.0.0.1:8178",
        "https://example.com:8183",
        "http://127.0.0.1",
        "http://127.0.0.1:8183?",
        "http://127.0.0.1:8183#",
    ],
)
def test_shared_or_remote_endpoints_refused_before_io(endpoint):
    with pytest.raises(ValueError):
        local_caller(endpoint, MODEL)


@pytest.mark.parametrize("userinfo", [":fake-secret@", "@", "user:fake-secret@"])
def test_cli_refuses_userinfo_before_database_or_output(monkeypatch, capsys, tmp_path, userinfo):
    from brainlayer.pipeline.relation_inference import main

    endpoint = f"http://{userinfo}127.0.0.1:8183"
    monkeypatch.setattr(
        "sys.argv",
        ["relation_inference", "--db", str(tmp_path / "absent.db"), "--model", MODEL, "--endpoint", endpoint],
    )
    monkeypatch.setattr("sqlite3.connect", lambda *a, **k: pytest.fail("credential endpoint reached DB"))
    monkeypatch.setattr(
        "brainlayer.pipeline.relation_inference._open_local",
        lambda *a, **k: pytest.fail("credential endpoint reached I/O"),
    )
    with pytest.raises(ValueError) as error:
        main()
    output = capsys.readouterr()
    assert "fake-secret" not in str(error.value) + output.out + output.err


def test_conversation_filter_is_connection_local_and_preserves_source_rows():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (id TEXT, source TEXT, content_type TEXT, source_class TEXT)")
    rows = [
        ("chat", "claude_code", "user_message", "cli-agent"),
        ("video", "digest", "user_message", None),
        ("code", "codex_cli", "ai_code", "cli-agent"),
        ("reply", "codex_cli", "assistant_text", "cli-agent"),
    ]
    conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", rows)
    restrict_to_conversations(conn)
    assert conn.execute("SELECT id FROM chunks ORDER BY id").fetchall() == [("chat",), ("reply",)]
    assert conn.execute("SELECT * FROM main.chunks").fetchall() == rows
    conn.close()


def test_unambiguous_casefold_name_resolves_without_fuzzy_matching(monkeypatch):
    monkeypatch.setattr("brainlayer.pipeline.relation_inference._open_local", lambda *a, **k: envelope(source="atlas"))
    result = json.loads(local_caller("http://127.0.0.1:8183", MODEL)(PROMPT))
    assert result["chunks"][0]["relations"][0]["source_id"] == "project-uuid"


def test_duplicate_normalized_names_abstain_instead_of_selecting_an_id(monkeypatch):
    monkeypatch.setattr("brainlayer.pipeline.relation_inference._open_local", lambda *a, **k: envelope())
    chunk = json.loads(PROMPT.split("INPUT: ")[1])[0]
    chunk["entities"].append(dict(id="other", name="atlas", type="project"))
    with pytest.raises(ValueError, match="ambiguous"):
        local_caller("http://127.0.0.1:8183", MODEL)("INPUT: " + json.dumps([chunk]))


def test_name_resolution_does_not_bypass_original_evidence_validation(monkeypatch):
    monkeypatch.setattr("brainlayer.pipeline.relation_inference._open_local", lambda *a, **k: envelope())
    with pytest.raises(ValueError, match="exact evidence"):
        local_caller("http://127.0.0.1:8183", MODEL)(PROMPT.replace("Atlas uses SQLite.", "Atlas does not use SQLite."))


def test_input_delimiter_inside_source_is_preserved(monkeypatch):
    def post(request, timeout):
        source = json.loads(json.loads(request.data)["messages"][1]["content"])["source_text"]
        assert source == "INPUT: Atlas uses SQLite."
        return envelope()

    monkeypatch.setattr("brainlayer.pipeline.relation_inference._open_local", post)
    result = local_caller("http://127.0.0.1:8183", MODEL)(
        PROMPT.replace("Atlas uses SQLite.", "INPUT: Atlas uses SQLite.")
    )
    assert json.loads(result)["chunks"][0]["relations"]


@pytest.mark.parametrize("body", [b"{", b'{"choices": []}', b'{"choices": null}'])
def test_bad_http_envelope_is_not_a_semantic_rejection(monkeypatch, body):
    monkeypatch.setattr("brainlayer.pipeline.relation_inference._open_local", lambda *a, **k: io.BytesIO(body))
    with pytest.raises(RuntimeError, match="HTTP envelope"):
        local_caller("http://127.0.0.1:8183", MODEL)(PROMPT)


@pytest.fixture
def local_http_server(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    monkeypatch.setattr("urllib.request._opener", None)
    servers = []

    def start(reply):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.do_POST()

            def do_POST(self):
                requests.append(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                status, headers, body = reply()
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}", requests

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_source_body_never_reaches_environment_proxy(monkeypatch, local_http_server):
    proxy, leaked = local_http_server(lambda: (502, {}, b"proxy must not receive source"))
    endpoint, received = local_http_server(lambda: (200, {}, envelope().getvalue()))
    monkeypatch.setenv("http_proxy", proxy)
    monkeypatch.setattr("urllib.request.proxy_bypass", lambda host: False)
    result = local_caller(endpoint, MODEL)(PROMPT)
    assert json.loads(result)["chunks"][0]["relations"]
    assert len(received) == 1
    assert leaked == []


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_local_source_body_never_follows_redirect(status, local_http_server):
    sink, leaked = local_http_server(lambda: (200, {}, envelope().getvalue()))
    endpoint, received = local_http_server(lambda: (status, {"Location": sink}, b""))
    with pytest.raises(RuntimeError, match="Redirect"):
        local_caller(endpoint, MODEL)(PROMPT)
    assert len(received) == 1
    assert leaked == []


@pytest.mark.parametrize("conversations", [False, True])
def test_hidden_classes_never_feed_default_graph(conversations):
    from brainlayer.pipeline.relation_inference import restrict_sources

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (id TEXT, source TEXT, content_type TEXT, source_class TEXT)")
    rows = [
        ("chat", "claude_code", "user_message", "cli-agent"),
        ("brain", "realtime_watcher", "assistant_text", "brain-worker"),
        ("desktop", "claude_code", "user_message", "desktop"),
        ("subagent", "claude_code", "assistant_text", "subagent"),
        ("manual", "mcp", "user_message", None),
    ]
    conn.executemany("INSERT INTO chunks VALUES (?,?,?,?)", rows)
    restrict_sources(conn, conversations=conversations)
    expected = [("chat",), ("subagent",)] if conversations else [("chat",), ("manual",), ("subagent",)]
    assert conn.execute("SELECT id FROM chunks ORDER BY id").fetchall() == expected
    assert conn.execute("SELECT * FROM main.chunks").fetchall() == rows
    conn.close()
