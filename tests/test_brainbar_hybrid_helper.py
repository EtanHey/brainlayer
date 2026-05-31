import threading
import time
import uuid
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent

from brainlayer.brainbar_hybrid_helper import HybridSearchHelper


def test_warm_preloads_embedding_without_running_hybrid_search(monkeypatch, tmp_path):
    calls = []

    class FakeModel:
        def embed_query(self, text):
            calls.append(("embed_query", text))
            return [0.1, 0.2, 0.3]

    class FakeStore:
        def hybrid_search(self, **kwargs):
            calls.append(("hybrid_search", kwargs))
            raise AssertionError("warmup must not run a full hybrid search")

    monkeypatch.setattr("brainlayer.mcp._shared._get_embedding_model", lambda: FakeModel())
    monkeypatch.setattr("brainlayer.mcp._shared._get_search_vector_store", lambda: FakeStore())

    helper = HybridSearchHelper(socket_path=tmp_path / "helper.sock", db_path=tmp_path / "test.db")
    helper.warm()

    assert calls == [("embed_query", "brainbar hybrid helper warmup")]


def test_status_reports_helper_readiness(tmp_path):
    helper = HybridSearchHelper(socket_path=tmp_path / "helper.sock", db_path=tmp_path / "brain.db")

    assert helper._handle_request({"method": "status"}) == {"ok": True, "ready": False}

    helper._ready = True

    assert helper._handle_request({"method": "status"}) == {"ok": True, "ready": True}


def test_helper_does_not_bind_socket_until_warm_finishes(tmp_path):
    socket_path = Path(f"/tmp/bl-hh-{uuid.uuid4().hex[:8]}.sock")
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    helper = HybridSearchHelper(socket_path=socket_path, db_path=tmp_path / "brain.db")
    warm_started = threading.Event()
    allow_warm_to_finish = threading.Event()

    def slow_warm():
        warm_started.set()
        assert not socket_path.exists()
        allow_warm_to_finish.wait(timeout=2)
        helper.stop()

    helper.warm = slow_warm  # type: ignore[method-assign]
    thread = threading.Thread(target=helper.serve_forever)
    thread.start()
    try:
        assert warm_started.wait(timeout=1)
        time.sleep(0.05)
        assert not socket_path.exists()
    finally:
        allow_warm_to_finish.set()
        thread.join(timeout=2)
        helper.stop()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def test_warm_does_not_sleep_for_hybrid_search_retries(monkeypatch, tmp_path):
    calls = []

    class FakeModel:
        def embed_query(self, text):
            calls.append(("embed_query", text))
            return [0.1, 0.2, 0.3]

    class FakeStore:
        def hybrid_search(self, **kwargs):
            calls.append(kwargs)
            raise AssertionError("warmup must not retry a full hybrid search")

    monkeypatch.setattr("brainlayer.mcp._shared._get_embedding_model", lambda: FakeModel())
    monkeypatch.setattr("brainlayer.mcp._shared._get_search_vector_store", lambda: FakeStore())

    helper = HybridSearchHelper(socket_path=tmp_path / "helper.sock", db_path=tmp_path / "test.db")
    helper.warm()

    assert calls == [("embed_query", "brainbar hybrid helper warmup")]


def test_helper_routes_brain_search_to_python_mcp_with_source_all_default(monkeypatch, tmp_path):
    calls = []

    async def fake_brain_search(**kwargs):
        calls.append(kwargs)
        return (
            [TextContent(type="text", text="hybrid result manual-a0b8a")],
            {"query": kwargs["query"], "results": [{"chunk_id": "manual-a0b8a"}]},
        )

    monkeypatch.setattr("brainlayer.mcp.search_handler._brain_search", fake_brain_search)

    helper = HybridSearchHelper(socket_path=tmp_path / "helper.sock", db_path=tmp_path / "test.db")
    response = helper._handle_request(
        {
            "method": "brain_search",
            "arguments": {
                "query": "techgym speakers workshop",
                "num_results": 3,
                "project": "brainlayer",
                "tag": "speakers-workshop",
                "importance_min": 8,
                "detail": "compact",
            },
        }
    )

    assert response == {
        "ok": True,
        "text": "hybrid result manual-a0b8a",
        "metadata": {
            "structuredContent": {
                "query": "techgym speakers workshop",
                "results": [{"chunk_id": "manual-a0b8a"}],
            }
        },
    }
    assert calls == [
        {
            "query": "techgym speakers workshop",
            "project": "brainlayer",
            "source": "all",
            "tag": "speakers-workshop",
            "importance_min": 8,
            "agent_id": None,
            "num_results": 3,
            "max_results": 10,
            "detail": "compact",
            "allow_helper_route": False,
            "brainbar_helper_fast_profile": True,
        }
    ]


def test_helper_preserves_agent_id_for_brain_search(monkeypatch, tmp_path):
    calls = []

    async def fake_brain_search(**kwargs):
        calls.append(kwargs)
        return (
            [TextContent(type="text", text="hybrid result")],
            {"query": kwargs["query"], "results": []},
        )

    monkeypatch.setattr("brainlayer.mcp.search_handler._brain_search", fake_brain_search)

    helper = HybridSearchHelper(socket_path=tmp_path / "helper.sock", db_path=tmp_path / "test.db")
    response = helper._handle_request(
        {
            "method": "brain_search",
            "arguments": {
                "query": "agent scoped query",
                "agent_id": "codex-test-agent",
            },
        }
    )

    assert response["ok"] is True
    assert len(calls) == 1
    assert calls[0]["agent_id"] == "codex-test-agent"


def test_helper_preserves_brain_search_mcp_error(monkeypatch, tmp_path):
    async def fake_brain_search(**_kwargs):
        return CallToolResult(content=[TextContent(type="text", text="Invalid detail='verbose'")], isError=True)

    monkeypatch.setattr("brainlayer.mcp.search_handler._brain_search", fake_brain_search)

    helper = HybridSearchHelper(socket_path=tmp_path / "helper.sock", db_path=tmp_path / "test.db")
    response = helper._handle_request({"method": "brain_search", "arguments": {"query": "x"}})

    assert response == {
        "ok": True,
        "text": "Invalid detail='verbose'",
        "metadata": {},
        "isError": True,
    }


def test_content_text_extracts_single_dict_text():
    assert HybridSearchHelper._content_text({"type": "text", "text": "dict text"}) == "dict text"


def test_read_line_rejects_oversized_chunk_before_newline():
    class FakeSocket:
        def __init__(self):
            self.chunks = [b"x" * 1_000_001 + b"\n"]

        def recv(self, _size):
            return self.chunks.pop(0) if self.chunks else b""

    with pytest.raises(ValueError, match="request exceeds 1MB"):
        HybridSearchHelper._read_line(FakeSocket())


def test_handle_connection_ignores_client_disconnect_on_response(monkeypatch, tmp_path):
    helper = HybridSearchHelper(socket_path=tmp_path / "helper.sock", db_path=tmp_path / "test.db")
    monkeypatch.setattr(helper, "_read_line", lambda _conn: b'{"method":"brain_search","arguments":{"query":"x"}}')
    monkeypatch.setattr(helper, "_handle_request", lambda _request: {"ok": True, "text": "result", "metadata": {}})

    class ClosedSocket:
        def sendall(self, _payload):
            raise BrokenPipeError("client disconnected")

    helper._handle_connection(ClosedSocket())
