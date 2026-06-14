"""Consumer-role scoping guards for retrieval-time BrainLayer search."""

import json

import pytest

from brainlayer._helpers import serialize_f32
from brainlayer.chunk_origin import CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT
from brainlayer.scoping import ConsumerScope, resolve_consumer_scope
from brainlayer.search_repo import _hybrid_cache
from brainlayer.vector_store import VectorStore


@pytest.fixture(autouse=True)
def clear_hybrid_cache():
    _hybrid_cache.clear()
    yield
    _hybrid_cache.clear()


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "consumer-scope.db")
    s._binary_index_available = False
    yield s
    s.close()


def _embed(text: str) -> list[float]:
    seed = (sum(ord(c) for c in text[:40]) % 97) / 1000.0
    return [seed + (i / 10000.0) for i in range(1024)]


def _insert_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    content: str,
    embedding: list[float],
    project: str | None,
    created_at: str = "2026-06-01T00:00:00Z",
    chunk_origin: str | None = None,
) -> None:
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, created_at, chunk_origin
        ) VALUES (?, ?, ?, 'scope-test.jsonl', ?, 'assistant_text', ?, 'claude_code', ?, ?)""",
        (
            chunk_id,
            content,
            json.dumps({}),
            project,
            len(content),
            created_at,
            chunk_origin,
        ),
    )
    cursor.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_f32(embedding)),
    )


def _ids(results: dict) -> list[str]:
    return results["ids"][0]


def _projects(results: dict) -> list[str | None]:
    return [meta.get("project") for meta in results["metadatas"][0]]


class _ScopedEmbeddingModel:
    def embed_query(self, query: str) -> list[float]:
        return _embed(query)


def test_resolve_consumer_scope_defaults_to_worker_fail_closed(monkeypatch):
    monkeypatch.delenv("BRAINLAYER_CONSUMER", raising=False)

    scope = resolve_consumer_scope(project="brainlayer")

    assert scope.role == "worker"
    assert scope.project_filter == "brainlayer"
    assert scope.allow_null_project is False
    assert scope.deny_all is False


def test_worker_without_project_denies_all(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_CONSUMER", "worker")

    scope = resolve_consumer_scope(project=None)

    assert scope.role == "worker"
    assert scope.deny_all is True
    assert scope.allow_null_project is False


def test_worker_consumer_scope_blocks_foreign_and_null_projects_across_rrf(store):
    query_embedding = _embed("repo b semantic target")
    _insert_chunk(
        store,
        chunk_id="repo-a-allowed",
        content="repo scoped allowed memory",
        embedding=_embed("distant repo a"),
        project="repo-a",
    )
    _insert_chunk(
        store,
        chunk_id="repo-b-vector-and-fts",
        content="scope leak sentinel foreign repo",
        embedding=query_embedding,
        project="repo-b",
    )
    _insert_chunk(
        store,
        chunk_id="null-vector-and-fts",
        content="scope leak sentinel user local",
        embedding=[value + 0.00001 for value in query_embedding],
        project=None,
    )

    results = store.hybrid_search(
        query_embedding=query_embedding,
        query_text="scope leak sentinel",
        n_results=10,
        consumer_scope=ConsumerScope.for_worker("repo-a"),
    )

    assert "repo-b-vector-and-fts" not in _ids(results)
    assert "null-vector-and-fts" not in _ids(results)
    assert set(_projects(results)) <= {"repo-a"}


def test_worker_consumer_scope_closes_null_project_fts_only_leak(store):
    _insert_chunk(
        store,
        chunk_id="null-fts-leak",
        content="fts only leak sentinel",
        embedding=_embed("unrelated null vector"),
        project=None,
    )
    store.conn.cursor().execute("DELETE FROM chunk_vectors")

    results = store.hybrid_search(
        query_embedding=_embed("nothing close"),
        query_text="fts only leak sentinel",
        n_results=5,
        consumer_scope=ConsumerScope.for_worker("repo-a"),
    )

    assert _ids(results) == []


def test_orchestrator_consumer_scope_preserves_cross_repo_and_null_visibility(store):
    _insert_chunk(
        store,
        chunk_id="repo-a",
        content="orchestrator full context sentinel repo a",
        embedding=_embed("repo a"),
        project="repo-a",
    )
    _insert_chunk(
        store,
        chunk_id="repo-b",
        content="orchestrator full context sentinel repo b",
        embedding=_embed("repo b"),
        project="repo-b",
    )
    _insert_chunk(
        store,
        chunk_id="null-user-local",
        content="orchestrator full context sentinel user local",
        embedding=_embed("user local"),
        project=None,
    )

    results = store.hybrid_search(
        query_embedding=_embed("repo a"),
        query_text="orchestrator full context sentinel",
        n_results=10,
        consumer_scope=ConsumerScope.for_orchestrator(),
    )

    assert {"repo-a", "repo-b", "null-user-local"} <= set(_ids(results))
    assert {"repo-a", "repo-b", None} <= set(_projects(results))


def test_coach_consumer_scope_uses_personal_lane_and_recency_checkpoints(store):
    _insert_chunk(
        store,
        chunk_id="old-personal-checkpoint",
        content="[precompact checkpoint] coach personal checkpoint status",
        embedding=_embed("coach checkpoint"),
        project="personal",
        created_at="2026-05-01T00:00:00Z",
        chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
    )
    _insert_chunk(
        store,
        chunk_id="new-personal-checkpoint",
        content="[precompact checkpoint] coach personal checkpoint status",
        embedding=_embed("coach checkpoint"),
        project="personal",
        created_at="2026-06-10T00:00:00Z",
        chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
    )
    _insert_chunk(
        store,
        chunk_id="foreign-code-checkpoint",
        content="[precompact checkpoint] coach personal checkpoint status",
        embedding=_embed("coach checkpoint"),
        project="brainlayer",
        created_at="2026-06-12T00:00:00Z",
        chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
    )

    results = store.hybrid_search(
        query_embedding=_embed("coach checkpoint"),
        query_text="latest coach personal checkpoint",
        n_results=5,
        consumer_scope=ConsumerScope.for_coach(),
    )

    assert _ids(results)[:2] == ["new-personal-checkpoint", "old-personal-checkpoint"]
    assert "foreign-code-checkpoint" not in _ids(results)
    assert set(_projects(results)) <= {"personal", None}


@pytest.mark.asyncio
async def test_mcp_roundtrip_applies_worker_and_orchestrator_scopes_on_seeded_store(store, monkeypatch):
    from brainlayer.mcp.search_handler import _brain_search

    _insert_chunk(
        store,
        chunk_id="repo-a-handler",
        content="handler scope sentinel repo a",
        embedding=_embed("handler scope sentinel"),
        project="repo-a",
    )
    _insert_chunk(
        store,
        chunk_id="repo-b-handler",
        content="handler scope sentinel repo b",
        embedding=_embed("handler scope sentinel"),
        project="repo-b",
    )
    _insert_chunk(
        store,
        chunk_id="null-handler",
        content="handler scope sentinel user local",
        embedding=_embed("handler scope sentinel"),
        project=None,
    )

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: _ScopedEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)
    monkeypatch.setattr("brainlayer.scoping.resolve_project_scope", lambda: "repo-a")

    monkeypatch.setenv("BRAINLAYER_CONSUMER", "worker")
    _content, worker_structured = await _brain_search(
        query="handler scope sentinel",
        source="all",
        num_results=10,
        allow_helper_route=False,
    )

    worker_projects = {result["project"] for result in worker_structured["results"]}
    assert worker_projects == {"repo-a"}

    _content, worker_exact_structured = await _brain_search(
        query="null-handler",
        source="all",
        num_results=10,
        allow_helper_route=False,
    )
    assert worker_exact_structured["results"] == []

    monkeypatch.setenv("BRAINLAYER_CONSUMER", "orchestrator")
    _content, orchestrator_structured = await _brain_search(
        query="handler scope sentinel",
        source="all",
        num_results=10,
        allow_helper_route=False,
    )

    orchestrator_ids = {result["chunk_id"] for result in orchestrator_structured["results"]}
    orchestrator_projects = {result.get("project") for result in orchestrator_structured["results"]}
    assert {"repo-a-handler", "repo-b-handler", "null-handler"} <= orchestrator_ids
    assert {"repo-a", "repo-b", None} <= orchestrator_projects

    _content, orchestrator_exact_structured = await _brain_search(
        query="null-handler",
        source="all",
        num_results=10,
        allow_helper_route=False,
    )
    assert orchestrator_exact_structured["results"][0]["chunk_id"] == "null-handler"
