import pytest
from mcp.types import TextContent

from brainlayer.brainbar_hybrid_helper import HybridSearchHelper


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
            "num_results": 3,
            "detail": "compact",
        }
    ]


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
