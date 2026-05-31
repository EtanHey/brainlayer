"""Tests for content_class write classification and default search filtering."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from brainlayer._helpers import serialize_f32
from brainlayer.mcp import call_tool
from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore


def _embed(seed: float) -> list[float]:
    return [seed + (i / 10000.0) for i in range(1024)]


def _insert_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    content: str,
    content_class: str,
    embedding: list[float] | None = None,
    content_type: str = "note",
    created_at: str = "2026-05-01T00:00:00Z",
) -> None:
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, importance, created_at, content_class
        ) VALUES (?, ?, '{}', 'content-class-test.jsonl', 'content-class-test',
            ?, ?, 'manual', 5, ?, ?)""",
        (chunk_id, content, content_type, len(content), created_at, content_class),
    )
    if embedding is not None:
        cursor.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, serialize_f32(embedding)),
        )


def test_content_class_schema_defaults_to_knowledge(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "content-class-schema.db")
    try:
        cols = {row[1]: row[2] for row in store.conn.cursor().execute("PRAGMA table_info(chunks)")}
        indexes = {row[1] for row in store.conn.cursor().execute("PRAGMA index_list(chunks)")}

        assert cols["content_class"] == "TEXT"
        assert "idx_chunks_content_class" in indexes

        cursor = store.conn.cursor()
        cursor.execute(
            """INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type, char_count
            ) VALUES ('default-class', 'default class content', '{}', 'test', 'brainlayer', 'note', 21)"""
        )

        row = cursor.execute("SELECT content_class FROM chunks WHERE id = 'default-class'").fetchone()
    finally:
        store.close()

    assert row == ("knowledge",)


@pytest.mark.parametrize(
    ("content", "content_type", "expected"),
    [
        (
            "[BL-LEAD DECISION: chose sqlite-vec over Chroma because local ownership matters]",
            "note",
            "decision",
        ),
        ("Decided to keep the drain as the single writer because lock safety matters", "note", "decision"),
        ("Always use WAL mode for BrainLayer writes", "note", "decision"),
        ("A learning about WAL checkpoints and search freshness", "learning", "knowledge"),
        ("Explicit stored decision", "decision", "decision"),
        ("[BL-LEAD tick] helper check CLAUDE_COUNTER: 4", "note", "operational"),
        ("[CLAUDE_COUNTER 7]", "note", "operational"),
        ("<task-notification><result>worker done</result></task-notification>", "note", "operational"),
        ("watcher heartbeat alive sessions=4", "note", "operational"),
        ("bare status: PR checks still pending on surface:4", "note", "operational"),
        ("[BL-LEAD tick] שלום heartbeat status", "note", "knowledge"),
        ("[BL-LEAD tick] Etan heartbeat status", "note", "knowledge"),
        ("[BL-LEAD tick] health finance heartbeat status", "note", "knowledge"),
        ("ad-hoc eval test query for search ranking", "note", "test"),
        ("ambiguous coordination note with useful context", "note", "knowledge"),
    ],
)
def test_classify_content_class_decision_first(content: str, content_type: str, expected: str) -> None:
    from brainlayer.content_class import classify_content_class

    assert classify_content_class(content, content_type=content_type) == expected


def test_store_memory_persists_content_class_decision_first(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "content-class-store.db")
    try:
        decision = store_memory(
            store,
            embed_fn=None,
            content="[BL-LEAD DECISION: chose X over Y because it preserves durable recall]",
            memory_type="note",
            project="brainlayer",
        )
        operational = store_memory(
            store,
            embed_fn=None,
            content="[BL-LEAD tick] CLAUDE_COUNTER: 8 status only",
            memory_type="note",
            project="brainlayer",
        )

        rows = dict(
            store.conn.cursor().execute(
                "SELECT id, content_class FROM chunks WHERE id IN (?, ?)",
                (decision["id"], operational["id"]),
            )
        )
    finally:
        store.close()

    assert rows[decision["id"]] == "decision"
    assert rows[operational["id"]] == "operational"


def test_hybrid_search_excludes_operational_and_test_by_default(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "content-class-search.db")
    query_embedding = _embed(0.1)
    try:
        _insert_chunk(
            store,
            chunk_id="operational-semantic",
            content="[BL-LEAD tick] status-only coordination memory exactmatch",
            content_class="operational",
            embedding=query_embedding,
        )
        _insert_chunk(
            store,
            chunk_id="test-semantic",
            content="ad-hoc eval test query exactmatch",
            content_class="test",
            embedding=query_embedding,
        )
        _insert_chunk(
            store,
            chunk_id="knowledge-survivor",
            content="durable knowledge memory exactmatch",
            content_class="knowledge",
            embedding=_embed(0.2),
        )
        store.build_binary_index()
        store._trigram_fts_available = False

        default_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="exactmatch",
            n_results=5,
        )
        opt_in_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="exactmatch",
            n_results=5,
            include_operational=True,
        )
        class_filter_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="exactmatch",
            n_results=5,
            content_class_filter="operational",
        )
    finally:
        store.close()

    assert default_results["ids"][0] == ["knowledge-survivor"]
    assert "operational-semantic" in opt_in_results["ids"][0]
    assert "test-semantic" in opt_in_results["ids"][0]
    assert class_filter_results["ids"][0] == ["operational-semantic"]


def test_hybrid_search_auto_includes_operational_for_status_intent(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "content-class-status-intent.db")
    query_embedding = _embed(0.3)
    try:
        _insert_chunk(
            store,
            chunk_id="operational-status",
            content="[BL-LEAD tick] operational status heartbeat exactstatus",
            content_class="operational",
            embedding=query_embedding,
        )
        store.build_binary_index()
        store._trigram_fts_available = False

        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="operational status heartbeat exactstatus",
            n_results=3,
        )
    finally:
        store.close()

    assert results["ids"][0] == ["operational-status"]


class RecordingSearchStore:
    def __init__(self) -> None:
        self.hybrid_kwargs = None

    def count(self) -> int:
        return 1

    def hybrid_search(self, **kwargs):
        self.hybrid_kwargs = kwargs
        return {
            "ids": [["chunk-1"]],
            "documents": [["durable result"]],
            "metadatas": [[{"source_file": "test.md", "project": "brainlayer"}]],
            "distances": [[0.2]],
        }

    def enrich_results_with_session_context(self, results):
        return results


class FakeEmbeddingModel:
    def embed_query(self, _query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_brain_search_threads_content_class_controls_to_hybrid_search(monkeypatch) -> None:
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
            "query": "status heartbeat",
            "source": "all",
            "include_operational": True,
            "content_class": "operational",
        },
    )

    assert store.hybrid_kwargs is not None
    assert store.hybrid_kwargs["include_operational"] is True
    assert store.hybrid_kwargs["content_class_filter"] == "operational"


def test_dry_run_backfill_reports_counts_and_samples_without_updating(tmp_path: Path) -> None:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "backfill_content_class.py"
    spec = importlib.util.spec_from_file_location("backfill_content_class", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    store = VectorStore(tmp_path / "content-class-backfill.db")
    try:
        cursor = store.conn.cursor()
        cursor.execute(
            """INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type, char_count
            ) VALUES
                ('decision-visible', '[BL-LEAD DECISION: chose X over Y because durable]', '{}', 'test', 'brainlayer', 'note', 57),
                ('operational-hidden', '[BL-LEAD tick] CLAUDE_COUNTER: 4', '{}', 'test', 'brainlayer', 'note', 33),
                ('test-hidden', 'ad-hoc eval test query', '{}', 'test', 'brainlayer', 'note', 22),
                ('hebrew-visible', '[BL-LEAD tick] שלום heartbeat status', '{}', 'test', 'brainlayer', 'note', 38),
                ('person-visible', '[BL-LEAD tick] Etan heartbeat status', '{}', 'test', 'brainlayer', 'note', 37),
                ('personal-visible', '[BL-LEAD tick] health finance heartbeat status', '{}', 'test', 'brainlayer', 'note', 47)"""
        )

        report = module.build_backfill_report(store, sample_limit=5)
        rows_after = dict(cursor.execute("SELECT id, content_class FROM chunks"))
    finally:
        store.close()

    assert report["counts"] == {"decision": 1, "knowledge": 3, "operational": 1, "test": 1}
    assert report["keep_visible_override_total"] == 3
    assert report["personal_hidden"] == 0
    assert report["hidden_decision_or_personal_risk_total"] == 0
    rescued_ids = {row["chunk_id"] for row in report["keep_visible_override_samples"]}
    assert rescued_ids == {"hebrew-visible", "person-visible", "personal-visible"}
    assert [row["chunk_id"] for row in report["samples"]["operational"]] == ["operational-hidden"]
    assert [row["chunk_id"] for row in report["samples"]["decision"]] == ["decision-visible"]
    assert rows_after == {
        "decision-visible": "knowledge",
        "operational-hidden": "knowledge",
        "test-hidden": "knowledge",
        "hebrew-visible": "knowledge",
        "person-visible": "knowledge",
        "personal-visible": "knowledge",
    }
