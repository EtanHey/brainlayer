"""Tests for the unified Gemini enrichment controller.

Covers realtime/batch routing, content-hash dedup, retry logic, rate limiting,
telemetry, MCP handler, stats, error handling, idempotency, LaunchAgent plists,
and CLI integration.

Target: 35+ tests per A-R2 acceptance criteria.
"""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _candidate(chunk_id: str = "c1", content: str = "x" * 120) -> dict:
    return {
        "id": chunk_id,
        "content": content,
        "project": "brainlayer",
        "content_type": "assistant_text",
        "source": "claude_code",
    }


def _fake_gemini_client(response_text='{"summary":"sum","tags":["python"]}'):
    """Create a fake Gemini client that returns the given response text."""

    class FakeClient:
        class _Models:
            def generate_content(self, **kwargs):
                return SimpleNamespace(text=response_text)

        def __init__(self):
            self.models = self._Models()

    return FakeClient()


@pytest.fixture(autouse=True)
def _isolate_enrich_cost_counter(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAINLAYER_ENRICH_COST_DIR", str(tmp_path / "enrich-cost"))


def _patch_realtime_deps(monkeypatch, controller, store, response_text=None):
    """Common monkeypatching for realtime enrichment tests."""
    monkeypatch.setattr(controller, "build_external_prompt", MagicMock(return_value=("prompt", SimpleNamespace())))
    monkeypatch.setattr(controller, "parse_enrichment", MagicMock(return_value={"summary": "sum", "tags": ["python"]}))
    monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: SimpleNamespace()))
    monkeypatch.setattr(controller, "_sleep", lambda _: None)
    monkeypatch.setattr(
        controller,
        "_get_gemini_client",
        lambda: _fake_gemini_client(response_text or '{"summary":"sum","tags":["python"]}'),
    )


# ── Existing realtime tests ──────────────────────────────────────────────────


def test_enrichment_provenance_columns_are_audit_queryable_but_not_normal_search_payload(tmp_path):
    from brainlayer.store import store_memory
    from brainlayer.vector_store import VectorStore

    store = VectorStore(tmp_path / "provenance.db")
    try:
        columns = {row[1] for row in store.conn.cursor().execute("PRAGMA table_info(chunks)")}
        assert {"enrichment_model", "enrichment_backend"}.issubset(columns)

        result = store_memory(
            store=store,
            embed_fn=None,
            content="Track B provenance stamps enrichment model and backend for audit grading.",
            memory_type="decision",
            project="brainlayer",
        )
        store.update_enrichment(
            result["id"],
            summary="Track B stamps model/backend provenance.",
            tags=["project/brainlayer"],
            enrichment_model="gemini-2.5-flash-lite",
            enrichment_backend="gemini-flex",
        )

        row = (
            store.conn.cursor()
            .execute(
                "SELECT enrichment_model, enrichment_backend FROM chunks WHERE id = ?",
                (result["id"],),
            )
            .fetchone()
        )
        assert row == ("gemini-2.5-flash-lite", "gemini-flex")

        normal_payload = store.get_chunk(result["id"])
        assert normal_payload is not None
        assert "enrichment_model" not in normal_payload
        assert "enrichment_backend" not in normal_payload
    finally:
        store.close()


# ── Content-hash dedup tests ─────────────────────────────────────────────────


def test_content_hash_deterministic():
    from brainlayer.enrichment_replay import _content_hash

    h1 = _content_hash("hello world")
    h2 = _content_hash("hello world")
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex


def test_content_hash_strips_whitespace():
    from brainlayer.enrichment_replay import _content_hash

    h1 = _content_hash("  hello world  ")
    h2 = _content_hash("hello world")
    assert h1 == h2


def test_content_hash_differs_for_different_content():
    from brainlayer.enrichment_replay import _content_hash

    h1 = _content_hash("hello")
    h2 = _content_hash("world")
    assert h1 != h2


def test_is_duplicate_returns_false_when_column_missing():
    from brainlayer.enrichment_replay import _is_duplicate_content

    store = MagicMock()
    store._read_cursor.side_effect = Exception("no such column: content_hash")
    assert _is_duplicate_content(store, "content") is False


def test_is_duplicate_returns_true_when_hash_exists_enriched():
    from brainlayer.enrichment_replay import _is_duplicate_content

    store = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value.fetchone.return_value = (1,)
    store._read_cursor.return_value = cursor
    assert _is_duplicate_content(store, "content") is True


def test_is_duplicate_returns_false_when_summary_cleared():
    from brainlayer.enrichment_replay import _is_duplicate_content

    store = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value.fetchone.return_value = (0,)
    store._read_cursor.return_value = cursor

    assert _is_duplicate_content(store, "content") is False
    query, params = cursor.execute.call_args[0]
    assert "summary IS NOT NULL" in query
    assert params == (_is_duplicate_content.__globals__["_content_hash"]("content"),)


def test_is_duplicate_returns_false_when_hash_not_found():
    from brainlayer.enrichment_replay import _is_duplicate_content

    store = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value.fetchone.return_value = (0,)
    store._read_cursor.return_value = cursor
    assert _is_duplicate_content(store, "content") is False


def test_ensure_content_hash_column_creates_if_missing():
    from brainlayer.enrichment_replay import _ensure_content_hash_column

    store = MagicMock()
    cursor = MagicMock()
    cursor.execute.side_effect = [Exception("no such column"), None]
    store.conn.cursor.return_value = cursor
    assert _ensure_content_hash_column(store) is True


def test_ensure_content_hash_column_noop_if_exists():
    from brainlayer.enrichment_replay import _ensure_content_hash_column

    store = MagicMock()
    cursor = MagicMock()
    store.conn.cursor.return_value = cursor
    assert _ensure_content_hash_column(store) is True


def test_ensure_content_hash_column_drops_legacy_unique_index(tmp_path):
    from brainlayer.enrichment_replay import _ensure_content_hash_column

    db_path = tmp_path / "legacy-content-hash.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT NOT NULL, content_hash TEXT)")
    conn.execute("CREATE UNIQUE INDEX idx_content_hash_unique ON chunks(content_hash)")
    conn.commit()

    store = MagicMock()
    store.conn = conn

    assert _ensure_content_hash_column(store) is True

    indexes = {row[1]: bool(row[2]) for row in conn.execute("PRAGMA index_list(chunks)")}
    assert "idx_content_hash_unique" not in indexes
    assert indexes.get("idx_content_hash") is False


def test_backfill_content_hashes_processes_null_rows():
    from brainlayer.enrichment_replay import _backfill_content_hashes

    store = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value = [("id1", "content1"), ("id2", "content2")]
    store.conn.cursor.return_value = cursor

    count = _backfill_content_hashes(store, limit=10)
    assert count == 2


def test_backfill_content_hashes_skips_empty_content():
    from brainlayer.enrichment_replay import _backfill_content_hashes

    store = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value = [("id1", ""), ("id2", None)]
    store.conn.cursor.return_value = cursor

    count = _backfill_content_hashes(store, limit=10)
    assert count == 0


def test_backfill_content_hashes_handles_legacy_unique_index(tmp_path):
    from brainlayer.enrichment_replay import _backfill_content_hashes, _content_hash, _ensure_content_hash_column

    db_path = tmp_path / "legacy-backfill.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT NOT NULL, content_hash TEXT)")
    conn.execute("CREATE UNIQUE INDEX idx_content_hash_unique ON chunks(content_hash)")
    conn.executemany(
        "INSERT INTO chunks (id, content, content_hash) VALUES (?, ?, NULL)",
        [("c1", "duplicate content"), ("c2", "duplicate content")],
    )
    conn.commit()

    store = MagicMock()
    store.conn = conn

    assert _ensure_content_hash_column(store) is True

    count = _backfill_content_hashes(store, limit=10)

    expected_hash = _content_hash("duplicate content")
    rows = conn.execute("SELECT id, content_hash FROM chunks ORDER BY id").fetchall()
    assert count == 2
    assert rows == [("c1", expected_hash), ("c2", expected_hash)]


# ── EnrichmentResult dataclass tests ─────────────────────────────────────────


# ── Gemini config tests ──────────────────────────────────────────────────────


# ── Rate limiting tests ──────────────────────────────────────────────────────


# ── Retry logic tests ────────────────────────────────────────────────────────


# ── Error handling tests ──────────────────────────────────────────────────────


# ── Meta-research filter tests ───────────────────────────────────────────────


def test_meta_research_filter_detects_common_patterns():
    from brainlayer.enrichment_replay import is_meta_research

    samples = [
        "brain_search(query='crypto trading bot')",
        'brain_search query="crypto trading bot"',
        "Search results for 'crypto trading bot Ofir strategy'",
        "Query 3 for 'crypto trading bot Ofir strategy' degraded from 2.4/5 to 2.2/5",
        "Eval score: 2.6/5 after ingestion",
        "Grade: 3/5",
        "[BrainLayer auto] Memories matching: 5",
        '{"hookEventName":"search","additionalContext":"tool payload"}',
    ]

    assert all(is_meta_research(sample) for sample in samples)


def test_meta_research_filter_preserves_real_content():
    from brainlayer.enrichment_replay import is_meta_research

    samples = [
        "We decided to keep the enrichment controller in a single file until the batch path is stabilized.",
        "def build_index(query: str) -> list[str]:\n    return [query.strip()]",
        "Ofir said the strategy should defer position sizing until volatility normalizes.",
        "Conversation note: Noa wants the daemon restart deferred until after the migration lands.",
    ]

    assert all(not is_meta_research(sample) for sample in samples)


# ── Apply enrichment tests ───────────────────────────────────────────────────


def test_apply_enrichment_calls_update_enrichment_with_all_fields():
    from brainlayer.enrichment_replay import _apply_enrichment

    store = MagicMock()
    chunk = _candidate("c1")
    enrichment = {
        "summary": "test summary",
        "tags": ["python", "test"],
        "importance": 7,
        "intent": "implementation",
        "primary_symbols": ["func_a"],
        "resolved_query": None,
        "key_facts": ["PR #1722"],
        "resolved_queries": [
            "What changed in enrichment v2?",
            "enrichment v2 key_facts resolved_queries",
            "Enrichment v2 added key_facts and resolved_queries.",
        ],
        "epistemic_level": "certain",
        "version_scope": "v1.0",
        "debt_impact": "low",
        "external_deps": ["pytest"],
        "sentiment_label": "frustration",
        "sentiment_score": -0.6,
        "sentiment_signals": ["damn", "broken"],
    }

    _apply_enrichment(store, chunk, enrichment)

    store.update_enrichment.assert_called_once_with(
        chunk_id="c1",
        summary="test summary",
        tags=["python", "test"],
        importance=7,
        intent="implementation",
        primary_symbols=["func_a"],
        resolved_query="What changed in enrichment v2?",
        key_facts=["PR #1722"],
        resolved_queries=[
            "What changed in enrichment v2?",
            "enrichment v2 key_facts resolved_queries",
            "Enrichment v2 added key_facts and resolved_queries.",
        ],
        epistemic_level="certain",
        version_scope="v1.0",
        debt_impact="low",
        external_deps=["pytest"],
        sentiment_label="frustration",
        sentiment_score=-0.6,
        sentiment_signals=["damn", "broken"],
        enrichment_model="gemini-2.5-flash-lite",
        enrichment_backend="gemini-flex",
    )


def test_apply_enrichment_sets_content_hash():
    from brainlayer.enrichment_replay import _apply_enrichment, _content_hash

    store = MagicMock()
    cursor = MagicMock()
    store.conn.cursor.return_value = cursor
    chunk = _candidate("c1", "test content")

    _apply_enrichment(store, chunk, {"summary": "s"})

    expected_hash = _content_hash("test content")
    # _apply_enrichment issues several UPDATEs (raw_entities, content_hash,
    # provenance_class); assert on the content_hash write specifically rather
    # than relying on it being the last execute call.
    content_hash_calls = [
        call.args for call in cursor.execute.call_args_list if call.args and "content_hash" in call.args[0]
    ]
    assert content_hash_calls, "expected a content_hash UPDATE"
    assert content_hash_calls[-1][1] == (expected_hash, "c1")


def test_apply_enrichment_persists_raw_entities():
    from brainlayer.enrichment_replay import _apply_enrichment

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, raw_entities_json TEXT, content_hash TEXT)")
    conn.execute("INSERT INTO chunks (id, raw_entities_json, content_hash) VALUES (?, NULL, NULL)", ("c1",))

    store = MagicMock()
    store.conn = conn
    chunk = _candidate("c1", "test content")
    entities = [
        {"name": "Ofir", "type": "person", "relation": "described the strategy"},
        {"name": "BrainLayer", "type": "project", "relation": "stores the chunk"},
    ]

    _apply_enrichment(store, chunk, {"summary": "s", "entities": entities})

    row = conn.execute("SELECT raw_entities_json FROM chunks WHERE id = ?", ("c1",)).fetchone()
    assert row == (json.dumps(entities),)


def test_apply_enrichment_triggers_raw_entity_promotion(tmp_path, monkeypatch):
    from brainlayer import kg_promotion
    from brainlayer.enrichment_replay import _apply_enrichment
    from brainlayer.vector_store import VectorStore

    monkeypatch.setattr(kg_promotion, "_KNOWN_GIVEN_NAME_ALIASES", {"alex": {"אלכס"}})
    store = VectorStore(tmp_path / "apply-promotion.db")
    try:
        tag = "alex-sample-identification"
        cursor = store.conn.cursor()
        cursor.execute(
            """INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type,
                char_count, source, raw_entities_json, tags
            ) VALUES (?, ?, '{}', 'test.jsonl', 'brainlayer', 'assistant_text',
                      ?, 'test', ?, ?)""",
            (
                "existing",
                "Alex Sample coached Noa.",
                len("Alex Sample coached Noa."),
                json.dumps([{"name": "Alex Sample", "type": "person", "relation": "coach"}]),
                json.dumps([tag]),
            ),
        )
        cursor.execute("INSERT OR IGNORE INTO chunk_tags (chunk_id, tag) VALUES (?, ?)", ("existing", tag))
        cursor.execute(
            """INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type,
                char_count, source, tags
            ) VALUES (?, ?, '{}', 'test.jsonl', 'brainlayer', 'assistant_text',
                      ?, 'test', ?)""",
            ("new", "היי אלכס", len("היי אלכס"), json.dumps([tag])),
        )
        cursor.execute("INSERT OR IGNORE INTO chunk_tags (chunk_id, tag) VALUES (?, ?)", ("new", tag))

        _apply_enrichment(
            store,
            _candidate("new", "היי אלכס"),
            {"summary": "s", "entities": [{"name": "אלכס", "type": "person", "relation": "recipient"}]},
        )

        entity = store.resolve_entity("Alex Sample")
        assert entity is not None
        hebrew_entity = store.resolve_entity("אלכס")
        assert hebrew_entity is not None
        assert hebrew_entity["id"] == entity["id"]
    finally:
        store.close()


# ── Telemetry tests ──────────────────────────────────────────────────────────


def test_emit_enrichment_start_swallows_oserror_and_logs_debug(monkeypatch):
    from brainlayer import enrichment_controller as controller

    events = []
    debug_logs = []
    monkeypatch.setattr(controller, "_emit_enrichment_event", lambda e: events.append(e) or True)
    monkeypatch.setattr(controller.os, "write", lambda *_args: (_ for _ in ()).throw(OSError("pipe closed")))
    monkeypatch.setattr(controller.logger, "debug", lambda msg, *args: debug_logs.append(msg % args if args else msg))

    controller._emit_enrichment_start("realtime", 25)

    assert len(events) == 1
    assert events[0]["_type"] == "start"
    assert any("ENRICHMENT_RUNTIME_LOADED" in entry for entry in debug_logs)


def test_emit_enrichment_complete_fires(monkeypatch):
    from brainlayer import enrichment_controller as controller
    from brainlayer.enrichment_controller import EnrichmentResult

    events = []
    monkeypatch.setattr(controller, "_emit_enrichment_event", lambda e: events.append(e) or True)

    result = EnrichmentResult(mode="local", attempted=10, enriched=8, skipped=1, failed=1)
    controller._emit_enrichment_complete(result, 1500.0)

    assert len(events) == 1
    assert events[0]["_type"] == "complete"
    assert events[0]["enriched"] == 8
    assert events[0]["duration_ms"] == 1500.0


def test_emit_enrichment_error_truncates_long_errors(monkeypatch):
    from brainlayer import enrichment_controller as controller

    events = []
    monkeypatch.setattr(controller, "_emit_enrichment_event", lambda e: events.append(e) or True)

    long_error = "x" * 500
    controller._emit_enrichment_error("realtime", "chunk123", long_error)

    assert len(events[0]["error"]) == 300


def test_realtime_emits_start_and_complete_events(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = []

    events = []
    monkeypatch.setattr(controller, "_emit_enrichment_event", lambda e: events.append(e) or True)

    controller.enrich_realtime(store, limit=5)

    types = [e["_type"] for e in events]
    assert "start" in types
    assert "complete" in types


# ── Telemetry module tests ───────────────────────────────────────────────────


def test_telemetry_emit_returns_false_without_axiom_token(monkeypatch):
    import brainlayer.telemetry as telemetry

    monkeypatch.delenv("AXIOM_TOKEN", raising=False)
    telemetry._client = None
    telemetry._client_failed = False

    result = telemetry.emit("test-dataset", {"key": "value"})
    assert result is False


def test_telemetry_emit_many_returns_true_for_empty_list():
    import brainlayer.telemetry as telemetry

    result = telemetry.emit_many("test-dataset", [])
    assert result is True


def test_telemetry_enrichment_helpers_exist():
    from brainlayer.telemetry import (
        emit_enrichment_complete,
        emit_enrichment_error,
        emit_enrichment_start,
    )

    assert callable(emit_enrichment_start)
    assert callable(emit_enrichment_complete)
    assert callable(emit_enrichment_error)


# ── MCP handler tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["unknown", "realtime", "batch"])
async def test_brain_enrich_handler_is_retired_before_database_access(monkeypatch, mode):
    from brainlayer.mcp.enrich_handler import _brain_enrich

    forbidden_db = MagicMock(side_effect=AssertionError("retired handler opened DB"))
    monkeypatch.setattr("brainlayer.mcp.enrich_handler._get_vector_store", forbidden_db, raising=False)
    result = await _brain_enrich(mode=mode, phase="submit", chunk_ids=["synthetic"])
    assert result.is_error is True
    assert "Enrichment has been retired" in result.content[0].text
    forbidden_db.assert_not_called()


@pytest.mark.asyncio
async def test_brain_enrich_handler_stats_is_retired_without_reading_metadata(monkeypatch):
    from brainlayer.mcp.enrich_handler import _brain_enrich

    forbidden_db = MagicMock(side_effect=AssertionError("retired stats opened DB"))
    monkeypatch.setattr("brainlayer.mcp.enrich_handler._get_vector_store", forbidden_db, raising=False)
    result = await _brain_enrich(stats=True)
    assert result.is_error is True
    assert "Enrichment has been retired" in result.content[0].text
    forbidden_db.assert_not_called()


@pytest.mark.asyncio
async def test_enrich_stats_returns_correct_structure():
    from brainlayer.mcp.enrich_handler import _enrich_stats

    store = MagicMock()
    cursor = MagicMock()
    # Simulate: total=1000, enriched=600, unenriched=350, skipped=50, recent=20
    cursor.execute.return_value.fetchone.side_effect = [(1000,), (600,), (350,), (50,), (20,)]
    store._read_cursor.return_value = cursor

    result = await _enrich_stats(store)
    text = result.content[0].text

    # _enrich_stats returns formatted text lines, not JSON
    assert "Total: 1,000" in text
    assert "Enriched: 600" in text
    assert "(60.0%)" in text
    assert "Remaining: 350" in text
    assert "Skipped: 50" in text
    assert "Last 24h: 20" in text


# ── Batch mode tests ─────────────────────────────────────────────────────────


# ── Realtime chunk_ids filter test ────────────────────────────────────────────


def test_realtime_passes_chunk_ids_to_candidates(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = []

    controller.enrich_realtime(store, chunk_ids=["a", "b"])

    store.get_enrichment_candidates.assert_called_once_with(limit=500, since_hours=8760, chunk_ids=["a", "b"])


# ── LaunchAgent plist validation ──────────────────────────────────────────────


def test_launchd_installer_supports_explicit_load_and_unload():

    install_script = (Path(__file__).parent.parent / "scripts" / "launchd" / "install.sh").read_text()
    assert 'LAUNCH_DIR="$HOME/Library/LaunchAgents"' in install_script
    assert "load)" in install_script
    assert "unload)" in install_script
    assert "install_plist decay" in install_script


def test_launchd_installer_uses_standard_env_file_instead_of_embedding_google_key():

    install_script = (Path(__file__).parent.parent / "scripts" / "launchd" / "install.sh").read_text()
    assert ".zshrc" not in install_script
    assert "__GOOGLE_API_KEY__" not in install_script
    assert 'BRAINLAYER_ENV_FILE="${BRAINLAYER_ENV_FILE:-$HOME/.config/brainlayer/brainlayer.env}"' in install_script
    assert "__BRAINLAYER_ENV_RUN__" in install_script


# ── Gemini model constant test ────────────────────────────────────────────────


def test_gemini_realtime_model_default():
    from brainlayer.enrichment_controller import GEMINI_REALTIME_MODEL

    assert "flash-lite" in GEMINI_REALTIME_MODEL
    assert "2.5" in GEMINI_REALTIME_MODEL


# ── Empty candidates handling ─────────────────────────────────────────────────


def test_realtime_returns_zero_counts_for_no_candidates(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = []

    result = controller.enrich_realtime(store)

    assert result.attempted == 0
    assert result.enriched == 0
    assert result.skipped == 0
    assert result.failed == 0


def test_decay_plist_invokes_cli_decay_entrypoint():

    plist_path = Path(__file__).parent.parent / "scripts" / "launchd" / "com.brainlayer.decay.plist"
    content = plist_path.read_text()
    assert "__BRAINLAYER_BIN__" in content
    assert "<string>decay</string>" in content
    assert "<string>--json</string>" in content


def test_wal_checkpoint_plist_invokes_cli_checkpoint_entrypoint():

    plist_path = Path(__file__).parent.parent / "scripts" / "launchd" / "com.brainlayer.wal-checkpoint.plist"
    content = plist_path.read_text()
    assert "__BRAINLAYER_BIN__" in content
    assert "<string>wal-checkpoint</string>" in content
