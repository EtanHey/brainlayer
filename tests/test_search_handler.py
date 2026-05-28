import pytest

from brainlayer.mcp import call_tool


class FakeEmbeddingModel:
    def embed_query(self, _query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class RecordingSearchStore:
    def __init__(self) -> None:
        self.hybrid_kwargs = None

    def count(self) -> int:
        return 1

    def hybrid_search(self, **kwargs):
        self.hybrid_kwargs = kwargs
        return {
            "ids": [["chunk-agent-1"]],
            "documents": [["auth implementation result"]],
            "metadatas": [[{"source_file": "test.md", "project": "brainlayer"}]],
            "distances": [[0.25]],
        }

    def enrich_results_with_session_context(self, results):
        return results


@pytest.mark.asyncio
async def test_brain_search_mcp_threads_agent_id_to_hybrid_search(monkeypatch):
    store = RecordingSearchStore()

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)

    await call_tool(
        "brain_search",
        {
            "query": "auth implementation",
            "source": "all",
            "agent_id": "codex-test-agent",
        },
    )

    assert store.hybrid_kwargs is not None
    assert store.hybrid_kwargs["agent_id"] == "codex-test-agent"
