"""Evolution model calls stay on a selected numeric loopback endpoint."""

import ipaddress
import json
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture(autouse=True)
def prevent_legacy_sdk_send(monkeypatch):
    import urllib3.util.connection

    from brainlayer.pipeline import longitudinal_analyzer

    attempts, remote_attempts = [], []
    connect = urllib3.util.connection.create_connection

    def refuse_remote_connection(address, *args, **kwargs):
        try:
            local = ipaddress.ip_address(address[0]).is_loopback
        except ValueError:
            local = False
        if not local:
            remote_attempts.append(address)
            raise AssertionError(f"remote connection attempted: {address}")
        return connect(address, *args, **kwargs)

    monkeypatch.setattr(urllib3.util.connection, "create_connection", refuse_remote_connection)
    monkeypatch.setattr(
        longitudinal_analyzer,
        "ollama",
        types.SimpleNamespace(
            generate=lambda **kwargs: attempts.append(kwargs) or {"response": "legacy SDK"},
        ),
        raising=False,
    )
    yield attempts
    assert remote_attempts == []


@pytest.mark.parametrize(
    "host",
    [
        "https://remote.example.invalid",
        "http://192.0.2.1:11434",
        "0.0.0.0:11434",
        "http://[::]:11434",
        "http://[2001:db8::1]:11434",
        "http://localhost.evil.invalid",
        "http://user@localhost",
        "http://localhost/path",
        "http://localhost?host=remote",
        "http://localhost#remote",
        "ftp://localhost",
        "",
        "http://localhost:99999",
        "127.1",
    ],
)
def test_remote_or_ambiguous_host_refused_before_send(monkeypatch, host, prevent_legacy_sdk_send):
    from brainlayer.pipeline import longitudinal_analyzer

    monkeypatch.setenv("OLLAMA_HOST", host)
    with pytest.raises(ValueError, match="loopback"):
        longitudinal_analyzer._ollama_generate(model="fixture", prompt="Synthetic prompt")
    assert prevent_legacy_sdk_send == []


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        (None, "http://127.0.0.1:11434"),
        ("localhost:11435", "http://127.0.0.1:11435"),
        ("http://localhost/", "http://127.0.0.1:11434"),
        ("https://127.0.0.1:443", "https://127.0.0.1:443"),
        ("http://[::1]:11434", "http://[::1]:11434"),
        ("127.0.0.2:11435", "http://127.0.0.2:11435"),
    ],
)
def test_host_is_normalized_to_numeric_loopback(monkeypatch, host, expected):
    from brainlayer.pipeline.longitudinal_analyzer import loopback_ollama_url

    if host is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", host)
    assert loopback_ollama_url() == expected


@pytest.fixture
def model_endpoint(monkeypatch):
    import urllib3.util.connection

    payloads, connections, state = [], [], {"status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payloads.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            if state["status"] != 200:
                self.send_header("Location", "http://remote.example.invalid:9999/api/generate")
            self.end_headers()
            self.wfile.write(b'{"response":"synthetic local analysis"}')

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    target = ("127.0.0.1", server.server_port)
    connect = urllib3.util.connection.create_connection

    def only_model_endpoint(address, *args, **kwargs):
        connections.append(address)
        if address != target:
            raise AssertionError(f"non-model connection attempted: {address}")
        return connect(address, *args, **kwargs)

    monkeypatch.setattr(urllib3.util.connection, "create_connection", only_model_endpoint)
    monkeypatch.setenv("OLLAMA_HOST", f"http://localhost:{server.server_port}")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    try:
        yield target, payloads, connections, state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_real_local_request_ignores_proxy_env_and_scrubs_prompt(model_endpoint):
    from brainlayer.pipeline.longitudinal_analyzer import _ollama_generate

    target, payloads, connections, _state = model_endpoint
    secret = "gsk_" + "A1b2C3d4" * 7
    result = _ollama_generate(model="fixture", prompt=f"Synthetic {secret}", options={"temperature": 0.1})
    assert result["response"] == "synthetic local analysis"
    assert connections == [target]
    assert payloads[0]["model"] == "fixture"
    assert payloads[0]["stream"] is False
    assert secret not in payloads[0]["prompt"]


@pytest.mark.parametrize("status", [302, 307])
def test_redirect_refused_without_second_connection(model_endpoint, status):
    from brainlayer.pipeline.longitudinal_analyzer import _ollama_generate

    target, payloads, connections, state = model_endpoint
    state["status"] = status
    with pytest.raises(RuntimeError, match="redirect"):
        _ollama_generate(model="fixture", prompt="Synthetic prompt")
    assert connections == [target]
    assert len(payloads) == 1


def test_host_revalidated_after_previous_local_send(model_endpoint, monkeypatch, prevent_legacy_sdk_send):
    from brainlayer.pipeline.longitudinal_analyzer import _ollama_generate

    target, _payloads, connections, _state = model_endpoint
    _ollama_generate(model="fixture", prompt="First synthetic prompt")
    monkeypatch.setenv("OLLAMA_HOST", "http://remote.example.invalid")
    with pytest.raises(ValueError, match="loopback"):
        _ollama_generate(model="fixture", prompt="Second synthetic prompt")
    assert connections == [target]
    assert prevent_legacy_sdk_send == []


def test_cli_refuses_remote_host_before_loading_timeline(monkeypatch):
    from typer.testing import CliRunner

    from brainlayer.cli import app
    from brainlayer.pipeline import unified_timeline

    attempts = []
    monkeypatch.setenv("OLLAMA_HOST", "http://remote.example.invalid")
    monkeypatch.setattr(unified_timeline.UnifiedTimeline, "load_whatsapp", lambda self: attempts.append("load"))
    result = CliRunner().invoke(app, ["analyze-evolution", "--yes"])
    assert result.exit_code == 1
    assert "loopback" in result.output
    assert attempts == []


def test_all_four_analysis_sends_use_the_local_transport(model_endpoint):
    from dataclasses import replace
    from datetime import datetime

    from brainlayer.pipeline.longitudinal_analyzer import (
        analyze_batch_with_llm,
        analyze_evolution,
        generate_weighted_master_guide,
    )
    from brainlayer.pipeline.time_batcher import TimeBatch

    target, payloads, connections, _state = model_endpoint
    message = types.SimpleNamespace(text="Synthetic writing sample.", language="english", relationship_tag=None)
    batch = TimeBatch("2025-H1", datetime(2025, 1, 1), datetime(2025, 6, 30), [message])
    analysis = analyze_batch_with_llm(batch, language="english", model="fixture")
    assert analysis.message_count == 1
    analyses = [analysis, replace(analysis, period="2025-H2")]
    evolution = analyze_evolution(analyses, model="fixture")
    assert evolution == "synthetic local analysis"
    assert generate_weighted_master_guide(analyses, evolution, model="fixture") == "synthetic local analysis"
    assert len(payloads) == 4
    assert connections == [target] * 4
