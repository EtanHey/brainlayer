from pathlib import Path

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

    helper = HybridSearchHelper(socket_path=tmp_path / "helper.sock", db_path=Path("/tmp/test.db"))
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
