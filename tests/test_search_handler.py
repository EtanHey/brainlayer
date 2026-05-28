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


class RecordingKgSearchStore(RecordingSearchStore):
    def __init__(self) -> None:
        super().__init__()
        self.kg_hybrid_kwargs = None

    def kg_hybrid_search(self, **kwargs):
        self.kg_hybrid_kwargs = kwargs
        return {
            "chunks": {
                "ids": [["chunk-kg-agent-1"]],
                "documents": [["auth implementation entity result"]],
                "metadatas": [[{"source_file": "kg.md", "project": "brainlayer"}]],
                "distances": [[0.25]],
            },
            "facts": [],
        }


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


@pytest.mark.asyncio
async def test_brain_search_entity_route_threads_agent_id_to_kg_hybrid_search(monkeypatch):
    store = RecordingKgSearchStore()

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._exact_chunk_lookup_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._kg_facts_sql", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "brainlayer.mcp.search_handler._detect_entities",
        lambda *_args, **_kwargs: [{"name": "Alice"}],
    )
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)

    await call_tool(
        "brain_search",
        {
            "query": "what do we know about Alice",
            "source": "all",
            "agent_id": "codex-test-agent",
        },
    )

    assert store.kg_hybrid_kwargs is not None
    assert store.kg_hybrid_kwargs["agent_id"] == "codex-test-agent"
