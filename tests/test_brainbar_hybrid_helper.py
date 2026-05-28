import sqlite3

import pytest
from mcp.types import CallToolResult, TextContent

from brainlayer.brainbar_hybrid_helper import HybridSearchHelper


def test_warm_preloads_embedding_then_hybrid_search(monkeypatch, tmp_path):
    calls = []

    class FakeModel:
        def embed_query(self, text):
            calls.append(("embed_query", text))
            return [0.1, 0.2, 0.3]

    class FakeStore:
        def hybrid_search(self, **kwargs):
            calls.append(("hybrid_search", kwargs))
            return [{"chunk_id": "warm-1"}]

    monkeypatch.setattr("brainlayer.mcp._shared._get_embedding_model", lambda: FakeModel())
    monkeypatch.setattr("brainlayer.mcp._shared._get_search_vector_store", lambda: FakeStore())

    helper = HybridSearchHelper(socket_path=tmp_path / "helper.sock", db_path=tmp_path / "test.db")
    helper.warm()

    assert calls == [
        ("embed_query", "brainbar hybrid helper warmup"),
        (
            "hybrid_search",
            {
                "query_embedding": [0.1, 0.2, 0.3],
                "query_text": "brainbar hybrid helper warmup",
                "n_results": 5,
            },
        ),
    ]


def test_warm_retries_locked_hybrid_search_and_continues(monkeypatch, tmp_path):
    calls = []
    sleeps = []

    class FakeModel:
        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

    class FakeStore:
        def hybrid_search(self, **kwargs):
            calls.append(kwargs)
            if len(calls) < 3:
                raise sqlite3.OperationalError("database is locked")
            return [{"chunk_id": "warm-1"}]

    monkeypatch.setattr("brainlayer.mcp._shared._get_embedding_model", lambda: FakeModel())
    monkeypatch.setattr("brainlayer.mcp._shared._get_search_vector_store", lambda: FakeStore())
    monkeypatch.setattr("brainlayer.brainbar_hybrid_helper.time.sleep", lambda delay: sleeps.append(delay))

    helper = HybridSearchHelper(socket_path=tmp_path / "helper.sock", db_path=tmp_path / "test.db")
    helper.warm()

    assert len(calls) == 3
    assert sleeps == [0.05, 0.1]


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
