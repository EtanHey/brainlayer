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


def _fts_ids(store: VectorStore, table_name: str) -> list[str]:
    return [
        row[0]
        for row in store.conn.cursor().execute(
            f"SELECT chunk_id FROM {table_name} ORDER BY chunk_id"  # noqa: S608 - test-controlled table names
        )
    ]


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


def test_fts_routes_operational_out_of_knowledge_index_and_skips_cold_classes(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "content-class-fts-routing.db")
    try:
        _insert_chunk(
            store,
            chunk_id="knowledge-doc",
            content="durable knowledge routing sentinel",
            content_class="knowledge",
        )
        _insert_chunk(
            store,
            chunk_id="operational-doc",
            content="[BL-LEAD tick] status-only coordination routing sentinel",
            content_class="operational",
        )
        _insert_chunk(
            store,
            chunk_id="test-doc",
            content="ad-hoc eval test query routing sentinel",
            content_class="test",
        )
        _insert_chunk(
            store,
            chunk_id="benchmark-doc",
            content="BrainLayer Search Benchmark diagnostic routing sentinel",
            content_class="benchmark",
        )

        knowledge_fts_ids = _fts_ids(store, "chunks_fts")
        operational_fts_ids = _fts_ids(store, "chunks_fts_operational")
    finally:
        store.close()

    assert knowledge_fts_ids == ["knowledge-doc"]
    assert operational_fts_ids == ["operational-doc"]


def test_knowledge_fts_rows_match_knowledge_only_build_when_operational_and_cold_rows_exist(tmp_path: Path) -> None:
    mixed_store = VectorStore(tmp_path / "mixed-fts-routing.db")
    isolated_store = VectorStore(tmp_path / "knowledge-only-fts-routing.db")
    try:
        _insert_chunk(
            mixed_store,
            chunk_id="knowledge-doc",
            content="durable avgdl sentinel exactmatch",
            content_class="knowledge",
        )
        _insert_chunk(
            mixed_store,
            chunk_id="operational-doc",
            content="[BL-LEAD tick] status-only exactmatch",
            content_class="operational",
        )
        _insert_chunk(
            mixed_store,
            chunk_id="benchmark-doc",
            content="BrainLayer Search Benchmark diagnostic exactmatch",
            content_class="benchmark",
        )
        _insert_chunk(
            isolated_store,
            chunk_id="knowledge-doc",
            content="durable avgdl sentinel exactmatch",
            content_class="knowledge",
        )

        mixed_rows = list(
            mixed_store.conn.cursor().execute(
                "SELECT chunk_id, content, summary, tags, resolved_query, key_facts, resolved_queries "
                "FROM chunks_fts ORDER BY chunk_id"
            )
        )
        isolated_rows = list(
            isolated_store.conn.cursor().execute(
                "SELECT chunk_id, content, summary, tags, resolved_query, key_facts, resolved_queries "
                "FROM chunks_fts ORDER BY chunk_id"
            )
        )
        mixed_avg_len = sum(len(row[1] or "") for row in mixed_rows) / len(mixed_rows)
        isolated_avg_len = sum(len(row[1] or "") for row in isolated_rows) / len(isolated_rows)
    finally:
        mixed_store.close()
        isolated_store.close()

    assert mixed_rows == isolated_rows
    assert mixed_avg_len - isolated_avg_len == 0


def test_operational_fts_rows_require_explicit_include_operational(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "operational-fts-search.db")
    try:
        _insert_chunk(
            store,
            chunk_id="operational-doc",
            content="[BL-LEAD tick] status-only coordination explicitoperational",
            content_class="operational",
        )
        store._trigram_fts_available = False

        default_results = store.hybrid_search(
            query_embedding=None,
            query_text="explicitoperational",
            n_results=5,
        )
        intent_only_results = store.hybrid_search(
            query_embedding=None,
            query_text="operational status explicitoperational",
            n_results=5,
        )
        included_results = store.hybrid_search(
            query_embedding=None,
            query_text="explicitoperational",
            n_results=5,
            include_operational=True,
        )
    finally:
        store.close()

    assert default_results["ids"][0] == []
    assert intent_only_results["ids"][0] == []
    assert included_results["ids"][0] == ["operational-doc"]


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
        (
            "BrainLayer Search Benchmark diagnostic prompt: evaluate conceptual queries and rank recall results",
            "note",
            "benchmark",
        ),
        ("ambiguous coordination note with useful context", "note", "knowledge"),
    ],
)
def test_classify_content_class_decision_first(content: str, content_type: str, expected: str) -> None:
    from brainlayer.content_class import classify_content_class

    assert classify_content_class(content, content_type=content_type) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("CLAUDE_COUNTER: 4", "operational"),
        (
            "CLAUDE_COUNTER: 7\n\n"
            "Fixed the BrainBar helper readiness path and added coverage for persistent helper lifecycle "
            "state, socket startup, and search request forwarding.",
            "knowledge",
        ),
        ("Done.\n\nCLAUDE_COUNTER: 3", "operational"),
        ("<task-notification><result>worker done</result></task-notification>", "operational"),
        (
            "<task-notification><result>worker done</result></task-notification>\n\n"
            "The worker result means the schema migration needs a second verification pass before rollout.",
            "knowledge",
        ),
        (
            "Created two stories in update.json and documented the offline sync route, MCP socket path, "
            "screen owners, rollout checklist, and fixture coverage.\n\nheartbeat",
            "knowledge",
        ),
    ],
)
def test_operational_markers_must_dominate_content(content: str, expected: str) -> None:
    from brainlayer.content_class import classify_content_class

    assert classify_content_class(content, content_type="note") == expected


def test_audit_recursion_benchmark_class_does_not_require_trivial_residual() -> None:
    from brainlayer.content_class import classify_content_class

    content = (
        "BrainLayer Search Benchmark\n\n"
        "Diagnostic prompt: run conceptual query coverage for memory retrieval, compare ranked results, "
        "and explain why the benchmark prompt ranked itself first."
    )

    assert classify_content_class(content, content_type="note", tags='["audit", "r02"]') == "benchmark"


def test_audit_tagged_substantive_analysis_that_mentions_search_stays_visible() -> None:
    from brainlayer.content_class import classify_content_class

    content = (
        "BrainLayer MCP implementation audit found a real handler issue. "
        "The fix is to preserve structured errors in search_handler.py and add coverage."
    )

    assert classify_content_class(content, content_type="note", tags='["audit", "r02"]') == "knowledge"


def test_audit_tagged_project_search_prompt_stays_visible_without_benchmark_signals() -> None:
    from brainlayer.content_class import classify_content_class

    content = (
        "Audit story: search the codebase for localStorage usage and report mismatches between visitorId "
        "and email usage in SongScript."
    )

    assert classify_content_class(content, content_type="note", tags='["audit"]') == "knowledge"


def test_metric_reference_without_eval_table_stays_visible() -> None:
    from brainlayer.content_class import classify_content_class

    content = (
        "The demo framing says k=60 was a flat optimum and MAP moved only 0.0023, "
        "which affects BrainLayer top-K design."
    )

    assert classify_content_class(content, content_type="note", tags='["audit", "r02"]') == "knowledge"


def test_eval_results_table_is_benchmark() -> None:
    from brainlayer.content_class import classify_content_class

    content = (
        "EVAL FINAL RESULTS with pooled qrels:\n"
        "| Metric | FTS5 | Hybrid |\n"
        "| ndcg@10 | 0.910 | 0.930 |\n"
        "| recall@20 | 0.671 | 0.700 |"
    )

    assert classify_content_class(content, content_type="note") == "benchmark"


def test_leading_brain_search_dump_is_benchmark() -> None:
    from brainlayer.content_class import classify_content_class

    content = '┌─ brain_search: "overnight sprint" ─ 3 results\n│ result dump'

    assert classify_content_class(content, content_type="note") == "benchmark"


def test_embedded_brain_search_reference_in_analysis_stays_visible() -> None:
    from brainlayer.content_class import classify_content_class, content_class_is_default_hidden

    content = (
        "Results are there with non-zero scores. The issue is that some chunks contain text like "
        "`┌─ brain_search: ...` because those chunks are search output; the fix is an ingest guard."
    )

    content_class = classify_content_class(content, content_type="note")

    assert content_class != "benchmark"
    assert content_class_is_default_hidden(content_class) is False


def test_bundled_counter_substance_is_never_default_hidden() -> None:
    from brainlayer.content_class import classify_content_class, content_class_is_default_hidden

    content = (
        "CLAUDE_COUNTER: 11\n\n"
        "Created two stories in update.json for the new onboarding flow and updated the fixtures, "
        "screen-state notes, animation timing checklist, and release checklist."
    )
    content_class = classify_content_class(content, content_type="note")

    assert content_class == "knowledge"
    assert content_class_is_default_hidden(content_class) is False


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
            chunk_id="benchmark-semantic",
            content="BrainLayer Search Benchmark diagnostic exactmatch",
            content_class="benchmark",
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
            include_operational=True,
            content_class_filter="operational",
        )
    finally:
        store.close()

    assert default_results["ids"][0] == ["knowledge-survivor"]
    assert "operational-semantic" in opt_in_results["ids"][0]
    assert "test-semantic" in opt_in_results["ids"][0]
    assert "benchmark-semantic" in opt_in_results["ids"][0]
    assert class_filter_results["ids"][0] == ["operational-semantic"]


def test_hybrid_search_requires_explicit_operational_include_for_status_intent(tmp_path: Path) -> None:
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

        default_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="operational status heartbeat exactstatus",
            n_results=3,
        )
        included_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="operational status heartbeat exactstatus",
            n_results=3,
            include_operational=True,
        )
    finally:
        store.close()

    assert default_results["ids"][0] == []
    assert included_results["ids"][0] == ["operational-status"]


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
                ('benchmark-hidden', 'BrainLayer Search Benchmark diagnostic exactmatch', '{}', 'test', 'brainlayer', 'note', 46),
                ('bundled-visible', 'CLAUDE_COUNTER: 8\n\nCreated two update.json stories and refreshed fixture coverage for onboarding screens.', '{}', 'test', 'brainlayer', 'note', 94),
                ('hebrew-visible', '[BL-LEAD tick] שלום heartbeat status', '{}', 'test', 'brainlayer', 'note', 38),
                ('person-visible', '[BL-LEAD tick] Etan heartbeat status', '{}', 'test', 'brainlayer', 'note', 37),
                ('personal-visible', '[BL-LEAD tick] health finance heartbeat status', '{}', 'test', 'brainlayer', 'note', 47)"""
        )

        report = module.build_backfill_report(store, sample_limit=5)
        rows_after = dict(cursor.execute("SELECT id, content_class FROM chunks"))
    finally:
        store.close()

    assert report["counts"] == {"decision": 1, "knowledge": 4, "operational": 1, "test": 1, "benchmark": 1}
    assert report["keep_visible_override_total"] == 3
    assert report["operational_marker_kept_total"] == 4
    assert report["personal_hidden"] == 0
    assert report["hidden_decision_or_personal_risk_total"] == 0
    rescued_ids = {row["chunk_id"] for row in report["keep_visible_override_samples"]}
    assert rescued_ids == {"hebrew-visible", "person-visible", "personal-visible"}
    marker_kept_ids = {row["chunk_id"] for row in report["operational_marker_kept_samples"]}
    assert "bundled-visible" in marker_kept_ids
    assert [row["chunk_id"] for row in report["samples"]["operational"]] == ["operational-hidden"]
    assert [row["chunk_id"] for row in report["samples"]["benchmark"]] == ["benchmark-hidden"]
    assert [row["chunk_id"] for row in report["samples"]["decision"]] == ["decision-visible"]
    assert rows_after == {
        "decision-visible": "knowledge",
        "bundled-visible": "knowledge",
        "operational-hidden": "knowledge",
        "test-hidden": "knowledge",
        "benchmark-hidden": "knowledge",
        "hebrew-visible": "knowledge",
        "person-visible": "knowledge",
        "personal-visible": "knowledge",
    }


def test_migrate_fts_isolation_dry_run_and_apply_moves_existing_mixed_rows(tmp_path: Path) -> None:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "migrate_fts_operational_isolation.py"
    spec = importlib.util.spec_from_file_location("migrate_fts_operational_isolation", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    db_path = tmp_path / "mixed-existing-fts.db"
    store = VectorStore(db_path)
    try:
        _insert_chunk(
            store,
            chunk_id="knowledge-doc",
            content="durable migration knowledge exactmigrate",
            content_class="knowledge",
        )
        _insert_chunk(
            store,
            chunk_id="operational-doc",
            content="[BL-LEAD tick] migration operational exactmigrate",
            content_class="operational",
        )
        _insert_chunk(
            store,
            chunk_id="benchmark-doc",
            content="BrainLayer Search Benchmark diagnostic exactmigrate",
            content_class="benchmark",
        )
        cursor = store.conn.cursor()
        cursor.execute("DELETE FROM chunks_fts")
        cursor.execute("DELETE FROM chunks_fts_operational")
        cursor.execute("DELETE FROM chunks_fts_trigram")
        cursor.execute("DELETE FROM chunk_fts_rowids")
        cursor.execute("""
            INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
            SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id FROM chunks
        """)
    finally:
        store.close()

    dry_run = module.migrate_fts_isolation(db_path, dry_run=True, batch_size=2)
    assert dry_run["dry_run"] is True
    assert dry_run["before"]["knowledge_fts_operational_rows"] == 1
    assert dry_run["after"] is None

    apply_report = module.migrate_fts_isolation(db_path, dry_run=False, batch_size=2)
    assert apply_report["after"]["knowledge_fts_ids"] == ["knowledge-doc"]
    assert apply_report["after"]["operational_fts_ids"] == ["operational-doc"]
    assert apply_report["after"]["cold_fts_ids"] == []
