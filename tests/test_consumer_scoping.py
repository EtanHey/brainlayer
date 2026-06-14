"""Consumer-role scoping guards for retrieval-time BrainLayer search."""

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

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


def _insert_context_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    content: str,
    project: str | None,
    conversation_id: str,
    position: int,
) -> None:
    store.conn.cursor().execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, conversation_id, position
        ) VALUES (?, ?, ?, 'scope-context.jsonl', ?, 'assistant_text', ?, 'claude_code', ?, ?)""",
        (
            chunk_id,
            content,
            json.dumps({}),
            project,
            len(content),
            conversation_id,
            position,
        ),
    )


def _ids(results: dict) -> list[str]:
    return results["ids"][0]


def _projects(results: dict) -> list[str | None]:
    return [meta.get("project") for meta in results["metadatas"][0]]


def _text_parts(parts: list[Any]) -> str:
    return "\n".join(getattr(part, "text", str(part)) for part in parts)


class _ScopedEmbeddingModel:
    def embed_query(self, query: str) -> list[float]:
        return _embed(query)


@dataclass
class _FakeRecallResult:
    target: str = "src/app.py"
    file_history: list[dict[str, Any]] = field(default_factory=list)
    related_chunks: list[dict[str, Any]] = field(default_factory=list)
    session_summaries: list[dict[str, Any]] = field(default_factory=list)

    def format(self) -> str:
        parts = [f"## Recall: {self.target}"]
        for row in self.file_history:
            parts.append(f"{row.get('project')}:{row.get('action')}:{row.get('session_id')}")
        for row in self.related_chunks:
            parts.append(f"{row.get('project')}:{row.get('content')}")
        return "\n".join(parts)


@dataclass
class _FakeThinkResult:
    query: str = "scope"
    decisions: list[dict[str, Any]] = field(default_factory=list)
    patterns: list[dict[str, Any]] = field(default_factory=list)
    bugs: list[dict[str, Any]] = field(default_factory=list)
    context: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0

    def format(self) -> str:
        rows = [*self.decisions, *self.patterns, *self.bugs, *self.context]
        return "\n".join(f"{row.get('project')}:{row.get('content')}" for row in rows)


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


def test_lead_scope_expands_repo_to_all_configured_worktrees(monkeypatch):
    monkeypatch.setattr(
        "brainlayer.scoping._load_scopes",
        lambda: {
            "worktrees": {
                "repo-a.worktree.feature-x": "repo-a",
                "repo-a.worktree.feature-y": "repo-a",
                "repo-b.worktree.other": "repo-b",
            }
        },
    )

    scope = resolve_consumer_scope(project="repo-a", consumer="lead")

    assert scope.role == "lead"
    assert scope.project_filter == "repo-a"
    assert scope.project_filters == (
        "repo-a",
        "repo-a.worktree.feature-x",
        "repo-a.worktree.feature-y",
    )
    assert scope.allow_null_project is False


def test_lead_scope_includes_user_configured_parallel_repos(monkeypatch):
    monkeypatch.setattr(
        "brainlayer.scoping._load_scopes",
        lambda: {
            "worktrees": {
                "repo-a.worktree.feature-x": "repo-a",
                "repo-b.worktree.other": "repo-b",
            },
            "lead_parallel_projects": {"repo-a": ["repo-b"]},
        },
    )

    scope = resolve_consumer_scope(project="repo-a.worktree.feature-x", consumer="lead")

    assert scope.project_filter == "repo-a"
    assert scope.project_filters == (
        "repo-a",
        "repo-a.worktree.feature-x",
        "repo-b",
        "repo-b.worktree.other",
    )


def test_worker_on_worktree_sees_worktree_and_main(monkeypatch):
    monkeypatch.setattr(
        "brainlayer.scoping._load_scopes",
        lambda: {"worktrees": {"repo-a.worktree.feature-x": "repo-a"}},
    )

    scope = resolve_consumer_scope(project="repo-a.worktree.feature-x", consumer="worker")

    assert scope.role == "worker"
    assert scope.project_filter == "repo-a.worktree.feature-x"
    assert scope.project_filters == ("repo-a.worktree.feature-x", "repo-a")


def test_worker_on_main_sees_main_only(monkeypatch):
    monkeypatch.setattr(
        "brainlayer.scoping._load_scopes",
        lambda: {"worktrees": {"repo-a.worktree.feature-x": "repo-a"}},
    )

    scope = resolve_consumer_scope(project="repo-a", consumer="worker")

    assert scope.project_filter == "repo-a"
    assert scope.project_filters == ("repo-a",)


def test_prefilter_project_skips_when_scope_allows_null_project():
    from brainlayer.mcp.search_handler import _prefilter_project_for_consumer_scope

    assert _prefilter_project_for_consumer_scope("personal", ConsumerScope.for_coach()) is None


def test_engine_think_and_recall_pass_consumer_scope_to_hybrid_search():
    from brainlayer.engine import recall, think

    calls = []

    class FakeStore:
        def hybrid_search(self, **kwargs):
            calls.append(kwargs)
            return {
                "documents": [["allowed memory"]],
                "metadatas": [[{"project": "repo-a", "intent": "implementing"}]],
            }

    scope = ConsumerScope.for_worker("repo-a")

    think("scoped work", store=FakeStore(), embed_fn=_embed, project=None, consumer_scope=scope)
    recall(store=FakeStore(), embed_fn=_embed, topic="scoped work", project=None, consumer_scope=scope)

    assert calls[0]["consumer_scope"] == scope
    assert calls[1]["consumer_scope"] == scope


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


def test_lead_consumer_scope_includes_repo_worktrees_and_parallel_repos(store, monkeypatch):
    monkeypatch.setattr(
        "brainlayer.scoping._load_scopes",
        lambda: {
            "worktrees": {
                "repo-a.worktree.feature-x": "repo-a",
                "repo-b.worktree.parallel": "repo-b",
            },
            "lead_parallel_projects": {"repo-a": ["repo-b"]},
        },
    )
    for chunk_id, project in [
        ("repo-a-main", "repo-a"),
        ("repo-a-worktree", "repo-a.worktree.feature-x"),
        ("repo-b-main", "repo-b"),
        ("repo-b-worktree", "repo-b.worktree.parallel"),
        ("repo-c-foreign", "repo-c"),
        ("null-user-local", None),
    ]:
        _insert_chunk(
            store,
            chunk_id=chunk_id,
            content="lead worktree visibility sentinel",
            embedding=_embed("lead worktree visibility sentinel"),
            project=project,
        )

    scope = resolve_consumer_scope(project="repo-a", consumer="lead")
    results = store.hybrid_search(
        query_embedding=_embed("lead worktree visibility sentinel"),
        query_text="lead worktree visibility sentinel",
        n_results=10,
        consumer_scope=scope,
    )

    assert set(_ids(results)) == {
        "repo-a-main",
        "repo-a-worktree",
        "repo-b-main",
        "repo-b-worktree",
    }


def test_worker_worktree_consumer_scope_includes_worktree_and_main_only(store, monkeypatch):
    monkeypatch.setattr(
        "brainlayer.scoping._load_scopes",
        lambda: {"worktrees": {"repo-a.worktree.feature-x": "repo-a"}},
    )
    for chunk_id, project in [
        ("repo-a-main", "repo-a"),
        ("repo-a-worktree", "repo-a.worktree.feature-x"),
        ("repo-a-other-worktree", "repo-a.worktree.other"),
        ("repo-b-main", "repo-b"),
        ("null-user-local", None),
    ]:
        _insert_chunk(
            store,
            chunk_id=chunk_id,
            content="worker worktree visibility sentinel",
            embedding=_embed("worker worktree visibility sentinel"),
            project=project,
        )

    scope = resolve_consumer_scope(project="repo-a.worktree.feature-x", consumer="worker")
    results = store.hybrid_search(
        query_embedding=_embed("worker worktree visibility sentinel"),
        query_text="worker worktree visibility sentinel",
        n_results=10,
        consumer_scope=scope,
    )

    assert set(_ids(results)) == {"repo-a-main", "repo-a-worktree"}


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


def test_worker_direct_context_filters_foreign_and_null_neighbors(store):
    conversation_id = "mixed-session"
    _insert_context_chunk(
        store,
        chunk_id="repo-b-before",
        content="foreign repo neighbor must not leak",
        project="repo-b",
        conversation_id=conversation_id,
        position=1,
    )
    _insert_context_chunk(
        store,
        chunk_id="repo-a-target",
        content="allowed repo target",
        project="repo-a",
        conversation_id=conversation_id,
        position=2,
    )
    _insert_context_chunk(
        store,
        chunk_id="null-after",
        content="user-local neighbor must not leak",
        project=None,
        conversation_id=conversation_id,
        position=3,
    )
    _insert_context_chunk(
        store,
        chunk_id="repo-a-after",
        content="allowed repo neighbor",
        project="repo-a",
        conversation_id=conversation_id,
        position=4,
    )

    worker_context = store.get_context(
        "repo-a-target",
        before=3,
        after=3,
        consumer_scope=ConsumerScope.for_worker("repo-a"),
    )

    assert worker_context["target"]["id"] == "repo-a-target"
    assert [chunk["id"] for chunk in worker_context["context"]] == ["repo-a-target", "repo-a-after"]


def test_orchestrator_direct_context_preserves_mixed_project_neighbors(store):
    conversation_id = "mixed-session"
    _insert_context_chunk(
        store,
        chunk_id="repo-b-before",
        content="foreign repo neighbor is visible to orchestrator",
        project="repo-b",
        conversation_id=conversation_id,
        position=1,
    )
    _insert_context_chunk(
        store,
        chunk_id="repo-a-target",
        content="allowed repo target",
        project="repo-a",
        conversation_id=conversation_id,
        position=2,
    )
    _insert_context_chunk(
        store,
        chunk_id="null-after",
        content="user-local neighbor is visible to orchestrator",
        project=None,
        conversation_id=conversation_id,
        position=3,
    )

    orchestrator_context = store.get_context(
        "repo-a-target",
        before=3,
        after=3,
        consumer_scope=ConsumerScope.for_orchestrator(),
    )

    assert [chunk["id"] for chunk in orchestrator_context["context"]] == [
        "repo-b-before",
        "repo-a-target",
        "null-after",
    ]


@pytest.mark.asyncio
async def test_worker_file_timeline_filters_foreign_and_null_rows(monkeypatch):
    from brainlayer.mcp.search_handler import _file_timeline

    class FakeStore:
        def get_file_timeline(self, file_path, project=None, limit=50):  # noqa: ANN001, ARG002
            return [
                {
                    "file_path": "src/app.py",
                    "timestamp": "2026-06-01T00:00:00Z",
                    "session_id": "repo-a-session",
                    "action": "edit",
                    "project": "repo-a",
                },
                {
                    "file_path": "src/app.py",
                    "timestamp": "2026-06-01T01:00:00Z",
                    "session_id": "repo-b-session",
                    "action": "edit",
                    "project": "repo-b",
                },
                {
                    "file_path": "src/app.py",
                    "timestamp": "2026-06-01T02:00:00Z",
                    "session_id": "null-session",
                    "action": "read",
                    "project": None,
                },
            ]

    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: FakeStore())

    parts = await _file_timeline(
        file_path="src/app.py",
        project="repo-a",
        consumer_scope=ConsumerScope.for_worker("repo-a"),
    )

    text = _text_parts(parts)
    assert "Found 1 interactions" in text
    assert "repo-a-session"[:8] in text
    assert "repo-b" not in text
    assert "null-session"[:8] not in text


@pytest.mark.asyncio
async def test_worker_regression_filters_foreign_and_null_rows(monkeypatch):
    from brainlayer.mcp.search_handler import _regression

    class FakeStore:
        def get_file_regression(self, file_path, project=None):  # noqa: ANN001, ARG002
            return {
                "timeline": [
                    {
                        "file_path": "src/app.py",
                        "timestamp": "2026-06-01T00:00:00Z",
                        "session_id": "repo-a-session",
                        "action": "test-pass",
                        "project": "repo-a",
                        "branch": "main",
                    },
                    {
                        "file_path": "src/app.py",
                        "timestamp": "2026-06-01T01:00:00Z",
                        "session_id": "repo-b-session",
                        "action": "edit",
                        "project": "repo-b",
                        "branch": "foreign",
                    },
                    {
                        "file_path": "src/app.py",
                        "timestamp": "2026-06-01T02:00:00Z",
                        "session_id": "null-session",
                        "action": "edit",
                        "project": None,
                        "branch": "local",
                    },
                ],
                "last_success": {
                    "timestamp": "2026-06-01T00:00:00Z",
                    "session_id": "repo-a-session",
                    "branch": "main",
                    "project": "repo-a",
                },
                "changes_after": [
                    {
                        "timestamp": "2026-06-01T01:00:00Z",
                        "session_id": "repo-b-session",
                        "action": "edit",
                        "project": "repo-b",
                        "branch": "foreign",
                    },
                    {
                        "timestamp": "2026-06-01T02:00:00Z",
                        "session_id": "null-session",
                        "action": "edit",
                        "project": None,
                        "branch": "local",
                    },
                ],
            }

    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: FakeStore())

    parts = await _regression(
        file_path="src/app.py",
        project="repo-a",
        consumer_scope=ConsumerScope.for_worker("repo-a"),
    )

    text = _text_parts(parts)
    assert "Timeline: 1 interactions" in text
    assert "repo-a-s" in text
    assert "repo-b" not in text
    assert "null-session"[:8] not in text


@pytest.mark.asyncio
async def test_worker_recall_filters_foreign_and_null_rows(monkeypatch):
    from brainlayer.mcp.search_handler import _recall

    def fake_recall(**_kwargs):
        return _FakeRecallResult(
            file_history=[
                {
                    "timestamp": "2026-06-01T00:00:00Z",
                    "session_id": "repo-a-session",
                    "action": "edit",
                    "project": "repo-a",
                },
                {
                    "timestamp": "2026-06-01T01:00:00Z",
                    "session_id": "repo-b-session",
                    "action": "edit",
                    "project": "repo-b",
                },
                {"timestamp": "2026-06-01T02:00:00Z", "session_id": "null-session", "action": "read", "project": None},
            ],
            related_chunks=[
                {"content": "allowed related", "project": "repo-a"},
                {"content": "foreign related", "project": "repo-b"},
                {"content": "null related", "project": None},
            ],
        )

    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: object())
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: _ScopedEmbeddingModel())
    monkeypatch.setattr("brainlayer.engine.recall", fake_recall)

    parts, structured = await _recall(
        file_path="src/app.py",
        project="repo-a",
        consumer_scope=ConsumerScope.for_worker("repo-a"),
    )

    text = _text_parts(parts)
    assert structured["file_history"] == [
        {
            "timestamp": "2026-06-01T00:00:00",
            "action": "edit",
            "session_id": "repo-a-session",
            "file_path": "",
        }
    ]
    assert [chunk["project"] for chunk in structured["related_chunks"]] == ["repo-a"]
    assert "repo-b" not in text
    assert "null related" not in text


@pytest.mark.asyncio
async def test_recall_signal_route_passes_consumer_scope(monkeypatch):
    from brainlayer.mcp.search_handler import _brain_search

    calls = []

    async def fake_recall(**kwargs):
        calls.append(kwargs)
        return ([], {"target": kwargs.get("topic")})

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._query_signals_recall", lambda _query: True)
    monkeypatch.setattr("brainlayer.mcp.search_handler._recall", fake_recall)

    await _brain_search(
        query="recall scoped work",
        project="repo-a",
        consumer="worker",
        allow_helper_route=False,
    )

    assert calls[0]["consumer_scope"].project_filter == "repo-a"
    assert calls[0]["consumer_scope"].allow_null_project is False


@pytest.mark.asyncio
async def test_think_filters_with_expanded_lead_scope(monkeypatch):
    from brainlayer.mcp.search_handler import _think

    captured = {}

    def fake_think(**kwargs):
        captured["project"] = kwargs.get("project")
        return _FakeThinkResult(
            decisions=[
                {"content": "main allowed", "project": "repo-a"},
                {"content": "worktree allowed", "project": "repo-a.worktree.feature-x"},
                {"content": "foreign blocked", "project": "repo-c"},
                {"content": "null blocked", "project": None},
            ],
            total=4,
        )

    monkeypatch.setattr(
        "brainlayer.scoping._load_scopes",
        lambda: {"worktrees": {"repo-a.worktree.feature-x": "repo-a"}},
    )
    consumer_scope = resolve_consumer_scope(project="repo-a", consumer="lead")
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: object())
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: _ScopedEmbeddingModel())
    monkeypatch.setattr("brainlayer.engine.think", fake_think)

    parts, structured = await _think(
        context="think scoped work",
        project="repo-a",
        consumer_scope=consumer_scope,
    )

    assert captured["project"] is None
    assert [item["project"] for item in structured["decisions"]] == ["repo-a", "repo-a.worktree.feature-x"]
    assert "foreign blocked" not in _text_parts(parts)


@pytest.mark.asyncio
async def test_lead_file_timeline_skips_single_project_prefilter(monkeypatch):
    from brainlayer.mcp.search_handler import _file_timeline

    class FakeStore:
        def get_file_timeline(self, file_path, project=None, limit=50):  # noqa: ANN001, ARG002
            assert project is None
            return [
                {
                    "file_path": "src/app.py",
                    "timestamp": "2026-06-01T00:00:00Z",
                    "session_id": "main-session",
                    "action": "edit",
                    "project": "repo-a",
                },
                {
                    "file_path": "src/app.py",
                    "timestamp": "2026-06-01T01:00:00Z",
                    "session_id": "worktree-session",
                    "action": "edit",
                    "project": "repo-a.worktree.feature-x",
                },
                {
                    "file_path": "src/app.py",
                    "timestamp": "2026-06-01T02:00:00Z",
                    "session_id": "foreign-session",
                    "action": "read",
                    "project": "repo-c",
                },
            ]

    monkeypatch.setattr(
        "brainlayer.scoping._load_scopes",
        lambda: {"worktrees": {"repo-a.worktree.feature-x": "repo-a"}},
    )
    consumer_scope = resolve_consumer_scope(project="repo-a", consumer="lead")
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: FakeStore())

    parts = await _file_timeline(
        file_path="src/app.py",
        project="repo-a",
        consumer_scope=consumer_scope,
    )

    text = _text_parts(parts)
    assert "Found 2 interactions" in text
    assert "main-ses" in text
    assert "worktree" in text
    assert "foreign" not in text


@pytest.mark.asyncio
async def test_current_context_filters_files_and_plan_from_scoped_sessions(monkeypatch):
    from brainlayer.engine import SessionInfo
    from brainlayer.mcp.search_handler import _current_context

    def fake_current_context(**_kwargs):
        return SimpleNamespace(
            active_projects=["repo-a", "repo-b"],
            active_branches=["main", "foreign"],
            active_plan="foreign-plan",
            recent_files=["foreign.py"],
            recent_sessions=[
                SessionInfo(
                    session_id="repo-a-session",
                    project="repo-a",
                    branch="main",
                    plan_name="allowed-plan",
                    files_changed=["allowed.py"],
                ),
                SessionInfo(
                    session_id="repo-b-session",
                    project="repo-b",
                    branch="foreign",
                    plan_name="foreign-plan",
                    files_changed=["foreign.py"],
                ),
            ],
            format=lambda: "context",
        )

    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: object())
    monkeypatch.setattr("brainlayer.engine.current_context", fake_current_context)

    _parts, structured = await _current_context(
        project="repo-a",
        consumer_scope=ConsumerScope.for_worker("repo-a"),
    )

    assert structured["active_projects"] == ["repo-a"]
    assert structured["active_branches"] == ["main"]
    assert structured["active_plan"] == "allowed-plan"
    assert structured["recent_files"] == ["allowed.py"]
    assert [session["project"] for session in structured["recent_sessions"]] == ["repo-a"]


@pytest.mark.asyncio
async def test_sessions_route_honors_deny_all_and_expanded_scope(monkeypatch):
    from brainlayer.engine import SessionInfo
    from brainlayer.mcp.search_handler import _sessions

    captured_projects = []

    def fake_sessions(**kwargs):
        captured_projects.append(kwargs.get("project"))
        return [
            SessionInfo(session_id="main-session", project="repo-a", branch="main"),
            SessionInfo(session_id="worktree-session", project="repo-a.worktree.feature-x", branch="feature-x"),
            SessionInfo(session_id="foreign-session", project="repo-c", branch="foreign"),
        ]

    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: object())
    monkeypatch.setattr("brainlayer.engine.sessions", fake_sessions)

    monkeypatch.setattr(
        "brainlayer.scoping._load_scopes",
        lambda: {"worktrees": {"repo-a.worktree.feature-x": "repo-a"}},
    )
    lead_scope = resolve_consumer_scope(project="repo-a", consumer="lead")

    parts = await _sessions(project="repo-a", consumer_scope=lead_scope)
    text = _text_parts(parts)

    assert captured_projects[-1] is None
    assert "main-ses" in text
    assert "worktree" in text
    assert "repo-c" not in text

    parts = await _sessions(project=None, consumer_scope=ConsumerScope.for_worker(None))
    assert "No sessions found" in _text_parts(parts)


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


@pytest.mark.asyncio
async def test_mcp_search_consumer_argument_overrides_shared_server_env(store, monkeypatch):
    from brainlayer.mcp.search_handler import _brain_search

    _insert_chunk(
        store,
        chunk_id="repo-a-explicit-consumer",
        content="explicit consumer scope sentinel repo a",
        embedding=_embed("explicit consumer"),
        project="repo-a",
    )
    _insert_chunk(
        store,
        chunk_id="repo-b-explicit-consumer",
        content="explicit consumer scope sentinel repo b",
        embedding=_embed("explicit consumer"),
        project="repo-b",
    )
    _insert_chunk(
        store,
        chunk_id="null-explicit-consumer",
        content="explicit consumer scope sentinel user local",
        embedding=_embed("explicit consumer"),
        project=None,
    )

    monkeypatch.setenv("BRAINLAYER_CONSUMER", "worker")
    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: _ScopedEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("brainlayer.mcp.search_handler._normalize_project_name", lambda project: project)
    monkeypatch.setattr("brainlayer.scoping.resolve_project_scope", lambda: "repo-a")

    _content, structured = await _brain_search(
        query="explicit consumer scope sentinel",
        source="all",
        num_results=10,
        consumer="orchestrator",
        allow_helper_route=False,
    )

    assert {result["chunk_id"] for result in structured["results"]} == {
        "repo-a-explicit-consumer",
        "repo-b-explicit-consumer",
        "null-explicit-consumer",
    }


@pytest.mark.asyncio
async def test_entity_id_search_resolves_project_before_worker_scope(monkeypatch):
    from brainlayer.mcp.search_handler import _brain_search

    async def fake_search(**kwargs):
        return ([], kwargs)

    monkeypatch.setenv("BRAINLAYER_CONSUMER", "worker")
    monkeypatch.setattr("brainlayer.scoping.resolve_project_scope", lambda: "repo-a")
    monkeypatch.setattr("brainlayer.mcp.search_handler._search", fake_search)

    _content, kwargs = await _brain_search(query="entity lookup", entity_id="entity-1")

    assert kwargs["project"] == "repo-a"
    assert kwargs["consumer_scope"].project_filter == "repo-a"
    assert kwargs["consumer_scope"].deny_all is False


@pytest.mark.asyncio
async def test_recall_resolves_consumer_scope_with_source_filter(monkeypatch):
    from brainlayer.mcp.search_handler import _brain_recall

    calls = []

    async def fake_current_context(**kwargs):
        calls.append(kwargs)
        return ([], {"context": "ok"})

    monkeypatch.setattr("brainlayer.mcp.search_handler._current_context", fake_current_context)

    await _brain_recall(
        mode="context",
        project="repo-a",
        consumer="worker",
        source_filter="%youtube%",
    )

    assert calls[0]["consumer_scope"].source_filter == "%youtube%"


@pytest.mark.asyncio
async def test_entity_kg_search_receives_consumer_scope(monkeypatch):
    from brainlayer.mcp.search_handler import _brain_search

    calls = []

    class FakeStore:
        def kg_hybrid_search(self, **kwargs):
            calls.append(kwargs)
            return {"chunks": {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}}

        def hybrid_search(self, **_kwargs):
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_route_enabled", lambda: False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: FakeStore())
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_embedding_model", lambda: _ScopedEmbeddingModel())
    monkeypatch.setattr("brainlayer.mcp.search_handler._expanded_fts_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._detect_entities", lambda *_args, **_kwargs: [{"name": "RepoA"}])
    monkeypatch.setattr("brainlayer.mcp.search_handler._kg_facts_sql", lambda *_args, **_kwargs: [])

    await _brain_search(
        query="RepoA",
        project="repo-a",
        consumer="worker",
        allow_helper_route=False,
    )

    assert calls[0]["consumer_scope"].project_filter == "repo-a"
    assert calls[0]["consumer_scope"].allow_null_project is False
