"""Tests for PreCompact checkpoint tagging and retrieval."""

from __future__ import annotations

import asyncio
import json
import os

import apsw

from brainlayer._helpers import serialize_f32
from brainlayer.chunk_origin import (
    CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
    CHUNK_ORIGIN_UNKNOWN,
    detect_chunk_origin,
)
from brainlayer.dedupe import compute_dedupe_fields
from brainlayer.drain import _apply_watcher
from brainlayer.mcp.search_handler import _brain_resume, _kg_facts_sql
from brainlayer.search_repo import _hybrid_cache
from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore
from brainlayer.watcher_bridge import create_flush_callback


def _embed(seed: str) -> list[float]:
    base = (sum(ord(char) for char in seed) % 97) / 1000.0
    return [base + (index / 10000.0) for index in range(1024)]


def _insert_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    content: str,
    embedding: list[float] | None = None,
    chunk_origin: str | None = None,
    created_at: str = "2026-05-16T10:00:00+00:00",
    conversation_id: str = "session-a",
) -> None:
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, created_at, conversation_id, chunk_origin
        ) VALUES (?, ?, '{}', 'test.jsonl', 'brainlayer', 'assistant_text', ?, 'claude_code', ?, ?, ?)""",
        (chunk_id, content, len(content), created_at, conversation_id, chunk_origin),
    )
    cursor.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_f32(embedding or _embed(content))),
    )


def test_detect_chunk_origin_matches_direct_and_watcher_checkpoint_forms():
    assert detect_chunk_origin("[PreCompact checkpoint]\ntimestamp: 2026-05-16") == CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT
    assert detect_chunk_origin("# PreCompact Checkpoint\nsession_id: abc") == CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT
    assert (
        detect_chunk_origin('Call mcp__brainlayer__brain_store exactly once with content="[PreCompact checkpoint]\\n"')
        == CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT
    )
    assert detect_chunk_origin("Enabled Claude Code PreCompact checkpoint storage in ~/.claude") == CHUNK_ORIGIN_UNKNOWN


def test_detect_chunk_origin_auto_detects_when_explicit_origin_is_unknown():
    assert (
        detect_chunk_origin("[PreCompact checkpoint]\ntimestamp: 2026-05-16", CHUNK_ORIGIN_UNKNOWN)
        == CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT
    )
    assert detect_chunk_origin("normal assistant memory", CHUNK_ORIGIN_UNKNOWN) == CHUNK_ORIGIN_UNKNOWN


def test_vector_store_migration_adds_chunk_origin_and_backfills_legacy_rows(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = apsw.Connection(str(db_path))
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            source_file TEXT NOT NULL,
            project TEXT,
            content_type TEXT,
            value_type TEXT,
            char_count INTEGER,
            source TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO chunks (id, content, metadata, source_file, source) VALUES (?, ?, '{}', 'watcher.jsonl', ?)",
        ("direct", "[PreCompact checkpoint]\ntimestamp: 2026-05-16", "mcp"),
    )
    conn.execute(
        "INSERT INTO chunks (id, content, metadata, source_file, source) VALUES (?, ?, '{}', 'watcher.jsonl', ?)",
        ("leading-whitespace", "\t\n[PreCompact checkpoint]\ntimestamp: 2026-05-16", "mcp"),
    )
    conn.execute(
        "INSERT INTO chunks (id, content, metadata, source_file, source) VALUES (?, ?, '{}', 'watcher.jsonl', ?)",
        (
            "wrapped",
            'Call mcp__brainlayer__brain_store exactly once with content="[PreCompact checkpoint]\\ntimestamp"',
            "realtime_watcher",
        ),
    )
    conn.execute(
        "INSERT INTO chunks (id, content, metadata, source_file, source) VALUES (?, ?, '{}', 'watcher.jsonl', ?)",
        (
            "wrapped-leading-whitespace",
            (" \n\t" * 400)
            + 'Call mcp__brainlayer__brain_store exactly once with content="[PreCompact checkpoint]\\ntimestamp"',
            "realtime_watcher",
        ),
    )
    conn.execute(
        "INSERT INTO chunks (id, content, metadata, source_file, source) VALUES (?, ?, '{}', 'manual.md', ?)",
        ("manual", "Enabled Claude Code PreCompact checkpoint storage in ~/.claude", "manual"),
    )
    conn.close()

    store = VectorStore(db_path)
    rows = dict(store.conn.cursor().execute("SELECT id, chunk_origin FROM chunks"))
    migration = (
        store.conn.cursor()
        .execute("SELECT details FROM schema_migrations WHERE name = '2026_05_16_fm6_chunk_origin'")
        .fetchone()
    )
    store.close()

    assert rows == {
        "direct": CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        "leading-whitespace": CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        "wrapped": CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        "wrapped-leading-whitespace": CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        "manual": CHUNK_ORIGIN_UNKNOWN,
    }
    assert migration is not None
    assert json.loads(migration[0])["backfilled_precompact_checkpoints"] == 4


def test_upsert_chunks_tags_precompact_origin(tmp_path):
    store = VectorStore(tmp_path / "upsert.db")
    store.upsert_chunks(
        [
            {
                "id": "checkpoint-upsert",
                "content": "[PreCompact checkpoint]\ntimestamp: 2026-05-16",
                "metadata": {},
                "source_file": "mcp",
                "project": "brainlayer",
                "content_type": "assistant_text",
                "char_count": 45,
                "source": "mcp",
                "created_at": "2026-05-16T10:00:00+00:00",
            }
        ],
        [_embed("checkpoint-upsert")],
    )

    row = store.conn.cursor().execute("SELECT chunk_origin FROM chunks WHERE id = 'checkpoint-upsert'").fetchone()
    store.close()

    assert row == (CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,)


def test_upsert_chunks_recomputes_origin_when_stable_chunk_content_changes(tmp_path):
    store = VectorStore(tmp_path / "upsert-recompute.db")
    store.upsert_chunks(
        [
            {
                "id": "stable-source-file-0",
                "content": "[PreCompact checkpoint]\nold compacted state",
                "metadata": {},
                "source_file": "session.jsonl",
                "project": "brainlayer",
                "content_type": "assistant_text",
                "char_count": 41,
                "source": "mcp",
                "created_at": "2026-05-16T10:00:00+00:00",
            }
        ],
        [_embed("old checkpoint")],
    )
    store.upsert_chunks(
        [
            {
                "id": "stable-source-file-0",
                "content": "normal assistant memory after source file was reprocessed",
                "metadata": {},
                "source_file": "session.jsonl",
                "project": "brainlayer",
                "content_type": "assistant_text",
                "char_count": 55,
                "source": "mcp",
                "created_at": "2026-05-16T10:05:00+00:00",
            }
        ],
        [_embed("normal reprocessed content")],
    )

    row = store.conn.cursor().execute("SELECT chunk_origin FROM chunks WHERE id = 'stable-source-file-0'").fetchone()
    store.close()

    assert row == (CHUNK_ORIGIN_UNKNOWN,)


def test_update_chunk_recomputes_origin_when_content_changes(tmp_path):
    store = VectorStore(tmp_path / "update-origin.db")
    _insert_chunk(
        store,
        chunk_id="updated-checkpoint",
        content="normal assistant memory",
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
    )

    updated = store.update_chunk("updated-checkpoint", content="[PreCompact checkpoint]\nrestored task state")

    row = store.conn.cursor().execute("SELECT chunk_origin FROM chunks WHERE id = 'updated-checkpoint'").fetchone()
    store.close()

    assert updated is True
    assert row == (CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,)


def test_update_chunk_recomputes_dedupe_fingerprints_when_content_changes(tmp_path):
    store = VectorStore(tmp_path / "update-dedupe.db")
    created_at = "2026-05-16T10:00:00+00:00"
    original = "Original update fingerprint memory"
    updated_content = "Edited update fingerprint memory with a different dedupe identity"
    _insert_chunk(
        store,
        chunk_id="updated-fingerprint",
        content=original,
        created_at=created_at,
    )

    updated = store.update_chunk("updated-fingerprint", content=updated_content)

    row = (
        store.conn.cursor()
        .execute(
            """
            SELECT dedupe_hash, simhash, simhash_band_0, simhash_band_1, simhash_band_2, simhash_band_3
            FROM chunks
            WHERE id = 'updated-fingerprint'
            """
        )
        .fetchone()
    )
    expected = compute_dedupe_fields(updated_content, created_at)
    store.close()

    assert updated is True
    assert row == (expected.dedupe_hash, expected.simhash, *expected.bands)


def test_watcher_and_drain_tag_precompact_origin(tmp_path, monkeypatch):
    db_path = tmp_path / "watcher.db"
    source_file = tmp_path / "session.jsonl"
    source_file.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-16T10:00:00+00:00",
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": 'Call mcp__brainlayer__brain_store exactly once with content="[PreCompact checkpoint]\\ntimestamp: 2026-05-16"',
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("BRAINLAYER_WRITER_ARBITRATION", raising=False)

    flush = create_flush_callback(db_path)
    flush([json.loads(source_file.read_text(encoding="utf-8")) | {"_source_file": str(source_file)}])
    store = VectorStore(db_path)
    row = store.conn.cursor().execute("SELECT chunk_origin FROM chunks WHERE source = 'realtime_watcher'").fetchone()

    _apply_watcher(
        store.conn,
        {
            "chunk_id": "queued-checkpoint",
            "content": "[PreCompact checkpoint]\ntimestamp: 2026-05-16",
            "metadata": {},
            "source_file": "queue.jsonl",
            "created_at": "2026-05-16T10:00:00+00:00",
        },
    )
    queued = store.conn.cursor().execute("SELECT chunk_origin FROM chunks WHERE id = 'queued-checkpoint'").fetchone()
    store.close()

    assert row == (CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,)
    assert queued == (CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,)


def test_search_excludes_checkpoints_by_default_and_can_include_them(tmp_path):
    _hybrid_cache.clear()
    store = VectorStore(tmp_path / "search.db")
    _insert_chunk(
        store,
        chunk_id="checkpoint",
        content="Recoverable focus token alpha beta",
        chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
    )
    _insert_chunk(
        store,
        chunk_id="normal",
        content="Recoverable focus token alpha beta",
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
    )
    store.build_binary_index()

    text_default = store.search(query_text="Recoverable focus token", n_results=10)
    text_with_checkpoints = store.search(query_text="Recoverable focus token", n_results=10, include_checkpoints=True)
    hybrid_default = store.hybrid_search(
        query_embedding=_embed("Recoverable focus token"),
        query_text="Recoverable focus token",
        n_results=10,
    )
    hybrid_with_checkpoints = store.hybrid_search(
        query_embedding=_embed("Recoverable focus token"),
        query_text="Recoverable focus token",
        n_results=10,
        include_checkpoints=True,
    )
    store.close()

    assert text_default["ids"][0] == ["normal"]
    assert set(text_with_checkpoints["ids"][0]) == {"checkpoint", "normal"}
    assert "checkpoint" not in hybrid_default["ids"][0]
    assert {"checkpoint", "normal"}.issubset(set(hybrid_with_checkpoints["ids"][0]))


def test_search_excludes_checkpoint_content_even_when_origin_backfill_missed_it(tmp_path):
    _hybrid_cache.clear()
    store = VectorStore(tmp_path / "search-missed-origin.db")
    _insert_chunk(
        store,
        chunk_id="missed-origin-checkpoint",
        content="\t\n[PreCompact checkpoint]\nCurrent task: should not leak through default search",
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
    )
    _insert_chunk(
        store,
        chunk_id="normal-missed-origin-control",
        content="Current task memory that should remain visible in default search",
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
    )
    store.build_binary_index()

    text_default = store.search(query_text="Current task", n_results=10)
    hybrid_default = store.hybrid_search(
        query_embedding=_embed("Current task"),
        query_text="Current task",
        n_results=10,
    )
    text_with_checkpoints = store.search(query_text="Current task", n_results=10, include_checkpoints=True)
    hybrid_with_checkpoints = store.hybrid_search(
        query_embedding=_embed("Current task"),
        query_text="Current task",
        n_results=10,
        include_checkpoints=True,
    )
    store.close()

    assert text_default["ids"][0] == ["normal-missed-origin-control"]
    assert "missed-origin-checkpoint" not in hybrid_default["ids"][0]
    assert {"missed-origin-checkpoint", "normal-missed-origin-control"}.issubset(set(text_with_checkpoints["ids"][0]))
    assert {"missed-origin-checkpoint", "normal-missed-origin-control"}.issubset(set(hybrid_with_checkpoints["ids"][0]))


def test_vector_search_overfetches_when_checkpoint_filter_discards_nearest_neighbors(tmp_path):
    store = VectorStore(tmp_path / "vector-overfetch.db")
    query_embedding = [0.0] * 1024
    for index in range(5):
        _insert_chunk(
            store,
            chunk_id=f"checkpoint-{index}",
            content=f"[PreCompact checkpoint]\nnoise {index}",
            embedding=query_embedding,
            chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        )
    _insert_chunk(
        store,
        chunk_id="normal-just-outside-knn-window",
        content="normal memory just outside the checkpoint nearest-neighbor window",
        embedding=[0.001] + ([0.0] * 1023),
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
    )

    results = store.search(query_embedding=query_embedding, n_results=2)
    store.close()

    assert results["ids"][0] == ["normal-just-outside-knn-window"]


def test_vector_search_does_not_starve_normal_results_after_many_checkpoints(tmp_path):
    store = VectorStore(tmp_path / "vector-many-checkpoints.db")
    query_embedding = [0.0] * 1024
    for index in range(2001):
        _insert_chunk(
            store,
            chunk_id=f"checkpoint-{index}",
            content=f"[PreCompact checkpoint]\nnoise {index}",
            embedding=query_embedding,
            chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        )
    _insert_chunk(
        store,
        chunk_id="normal-after-many-checkpoints",
        content="normal memory after many checkpoint nearest neighbors",
        embedding=[0.001] + ([0.0] * 1023),
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
    )

    results = store.search(query_embedding=query_embedding, n_results=1)
    store.close()

    assert results["ids"][0] == ["normal-after-many-checkpoints"]


def test_vector_search_does_not_overfetch_for_checkpoint_filter_when_no_checkpoints_exist(tmp_path, monkeypatch):
    store = VectorStore(tmp_path / "vector-no-checkpoint-overfetch.db")
    query_embedding = [0.0] * 1024
    for index in range(5):
        _insert_chunk(
            store,
            chunk_id=f"normal-{index}",
            content=f"normal memory {index}",
            embedding=[index / 1000.0] + ([0.0] * 1023),
            chunk_origin=CHUNK_ORIGIN_UNKNOWN,
        )

    captured_k = []
    real_read_cursor = store._read_cursor

    class CapturingCursor:
        def __init__(self, cursor):
            self.cursor = cursor

        def execute(self, sql, params=None):
            if "FROM chunk_vectors v" in sql and params:
                captured_k.append(params[1])
            return self.cursor.execute(sql, params)

    monkeypatch.setattr(store, "_read_cursor", lambda: CapturingCursor(real_read_cursor()))

    store.search(query_embedding=query_embedding, n_results=2)
    store.close()

    assert captured_k == [2]


def test_store_memory_checkpoint_invalidates_checkpoint_count_cache(tmp_path):
    store = VectorStore(tmp_path / "store-memory-cache.db")
    query_embedding = [0.0] * 1024
    _insert_chunk(
        store,
        chunk_id="normal-after-checkpoint",
        content="normal memory reachable after checkpoint overfetch",
        embedding=[0.001] + ([0.0] * 1023),
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
    )
    assert store.search(query_embedding=query_embedding, n_results=1)["ids"][0] == ["normal-after-checkpoint"]
    assert store._checkpoint_count_cache == 0

    store_memory(
        store=store,
        embed_fn=lambda _content: query_embedding,
        content="[PreCompact checkpoint]\nmanual recovery state",
        memory_type="note",
        project="brainlayer",
    )

    results = store.search(query_embedding=query_embedding, n_results=1)
    store.close()

    assert results["ids"][0] == ["normal-after-checkpoint"]


def test_external_checkpoint_write_invalidates_cached_checkpoint_count(tmp_path):
    db_path = tmp_path / "external-checkpoint-cache.db"
    reader = VectorStore(db_path)
    writer = VectorStore(db_path)
    query_embedding = [0.0] * 1024
    try:
        _insert_chunk(
            reader,
            chunk_id="normal-after-external-checkpoint",
            content="normal memory reachable after external checkpoint",
            embedding=[0.001] + ([0.0] * 1023),
            chunk_origin=CHUNK_ORIGIN_UNKNOWN,
        )
        assert reader.search(query_embedding=query_embedding, n_results=1)["ids"][0] == [
            "normal-after-external-checkpoint"
        ]
        assert reader._checkpoint_count_cache == 0

        _insert_chunk(
            writer,
            chunk_id="external-checkpoint",
            content="[PreCompact checkpoint]\nexternal writer state",
            embedding=query_embedding,
            chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        )

        results = reader.search(query_embedding=query_embedding, n_results=1)
    finally:
        reader.close()
        writer.close()

    assert results["ids"][0] == ["normal-after-external-checkpoint"]


def test_checkpoint_overfetch_composes_with_entity_filter_overfetch(tmp_path):
    store = VectorStore(tmp_path / "checkpoint-entity-overfetch.db")
    query_embedding = [0.0] * 1024
    try:
        for index in range(5):
            _insert_chunk(
                store,
                chunk_id=f"checkpoint-near-{index}",
                content=f"[PreCompact checkpoint]\nnear checkpoint {index}",
                embedding=[index / 10000.0] + ([0.0] * 1023),
                chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
            )
        for index in range(5):
            _insert_chunk(
                store,
                chunk_id=f"normal-unlinked-{index}",
                content=f"normal unlinked memory {index}",
                embedding=[0.01 + (index / 10000.0)] + ([0.0] * 1023),
                chunk_origin=CHUNK_ORIGIN_UNKNOWN,
            )
        _insert_chunk(
            store,
            chunk_id="normal-linked-target",
            content="normal entity-linked memory after checkpoint and entity filtering",
            embedding=[0.02] + ([0.0] * 1023),
            chunk_origin=CHUNK_ORIGIN_UNKNOWN,
        )
        store.upsert_entity("entity-target", "project", "Target Project")
        store.conn.cursor().execute(
            "INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance) VALUES (?, ?, ?)",
            ("entity-target", "normal-linked-target", 0.95),
        )

        results = store.search(query_embedding=query_embedding, n_results=1, entity_id="entity-target")
    finally:
        store.close()

    assert results["ids"][0] == ["normal-linked-target"]


def test_vector_search_caps_results_after_checkpoint_overfetch(tmp_path):
    store = VectorStore(tmp_path / "vector-result-cap.db")
    query_embedding = [0.0] * 1024
    for index in range(5):
        _insert_chunk(
            store,
            chunk_id=f"normal-near-{index}",
            content=f"normal nearby memory {index}",
            embedding=[index / 10000.0] + ([0.0] * 1023),
            chunk_origin=CHUNK_ORIGIN_UNKNOWN,
        )
        _insert_chunk(
            store,
            chunk_id=f"checkpoint-far-{index}",
            content=f"[PreCompact checkpoint]\nfar checkpoint {index}",
            embedding=[1.0 + index] + ([0.0] * 1023),
            chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        )

    results = store.search(query_embedding=query_embedding, n_results=2)
    store.close()

    assert results["ids"][0] == ["normal-near-0", "normal-near-1"]


def test_readonly_legacy_db_without_chunk_origin_searches_as_unknown(tmp_path):
    db_path = tmp_path / "legacy-readonly.db"
    store = VectorStore(db_path)
    _insert_chunk(
        store,
        chunk_id="legacy-normal",
        content="legacy searchable memory",
        embedding=_embed("legacy searchable memory"),
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
    )
    store.close()

    conn = apsw.Connection(str(db_path))
    conn.execute("DROP INDEX IF EXISTS idx_chunks_chunk_origin")
    conn.execute("ALTER TABLE chunks DROP COLUMN chunk_origin")
    conn.close()
    readonly_store = None
    os.chmod(db_path, 0o444)
    try:
        readonly_store = VectorStore(db_path)
        text_results = readonly_store.search(query_text="legacy searchable", n_results=1)
        vector_results = readonly_store.search(query_embedding=_embed("legacy searchable memory"), n_results=1)
        legacy_chunk = readonly_store.get_chunk("legacy-normal")
    finally:
        if readonly_store is not None:
            readonly_store.close()
        os.chmod(db_path, 0o644)

    assert text_results["ids"][0] == ["legacy-normal"]
    assert vector_results["ids"][0] == ["legacy-normal"]
    assert legacy_chunk is not None
    assert legacy_chunk["chunk_origin"] == CHUNK_ORIGIN_UNKNOWN


def test_brain_resume_returns_empty_on_readonly_legacy_db_without_chunk_origin(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy-resume-readonly.db"
    store = VectorStore(db_path)
    _insert_chunk(
        store,
        chunk_id="legacy-checkpoint",
        content="[PreCompact checkpoint]\nlegacy state",
        chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
    )
    store.close()

    conn = apsw.Connection(str(db_path))
    conn.execute("DROP INDEX IF EXISTS idx_chunks_chunk_origin")
    conn.execute("ALTER TABLE chunks DROP COLUMN chunk_origin")
    conn.close()
    readonly_store = None
    os.chmod(db_path, 0o444)
    try:
        readonly_store = VectorStore(db_path)
        monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: readonly_store)
        monkeypatch.setattr("brainlayer.mcp.search_handler._utcnow_iso", lambda: "2026-05-16T12:00:00+00:00")

        content, structured = asyncio.run(_brain_resume())
    finally:
        if readonly_store is not None:
            readonly_store.close()
        os.chmod(db_path, 0o644)

    assert structured == {"session_id": None, "lookback_days": 7, "total": 0, "results": []}
    assert content[0].text == "No PreCompact checkpoints found."


def test_kg_sql_facts_excludes_checkpoint_sourced_relations_by_default(tmp_path):
    store = VectorStore(tmp_path / "kg-sql-facts.db")
    _insert_chunk(
        store,
        chunk_id="checkpoint-fact-chunk",
        content="[PreCompact checkpoint]\nEtan builds checkpoint-only project",
        chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
    )
    _insert_chunk(
        store,
        chunk_id="normal-fact-chunk",
        content="Etan builds durable project",
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
    )
    store.upsert_entity("person-etan", "person", "Etan")
    store.upsert_entity("project-checkpoint", "project", "Checkpoint Project")
    store.upsert_entity("project-normal", "project", "Normal Project")
    store.add_relation(
        "rel-checkpoint",
        "person-etan",
        "project-checkpoint",
        "builds",
        properties={"description": "checkpoint-only fact"},
        source_chunk_id="checkpoint-fact-chunk",
    )
    store.add_relation(
        "rel-normal",
        "person-etan",
        "project-normal",
        "maintains",
        properties={"description": "normal fact"},
        source_chunk_id="normal-fact-chunk",
    )

    default_facts = _kg_facts_sql(store, "Etan")
    checkpoint_facts = _kg_facts_sql(store, "Etan", include_checkpoints=True)
    store.close()

    assert {fact["target"] for fact in default_facts} == {"Normal Project"}
    assert {fact["target"] for fact in checkpoint_facts} == {"Checkpoint Project", "Normal Project"}


def test_kg_hybrid_search_facts_excludes_checkpoint_sourced_relations_by_default(tmp_path):
    store = VectorStore(tmp_path / "kg-hybrid-facts.db")
    _insert_chunk(
        store,
        chunk_id="checkpoint-fact-chunk",
        content="[PreCompact checkpoint]\nEtan builds checkpoint-only project",
        chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
    )
    _insert_chunk(
        store,
        chunk_id="normal-fact-chunk",
        content="Etan builds durable project",
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
    )
    store.upsert_entity("person-etan", "person", "Etan")
    store.upsert_entity("project-checkpoint", "project", "Checkpoint Project")
    store.upsert_entity("project-normal", "project", "Normal Project")
    store.add_relation(
        "rel-checkpoint",
        "person-etan",
        "project-checkpoint",
        "builds",
        fact="checkpoint-only fact",
        source_chunk_id="checkpoint-fact-chunk",
    )
    store.add_relation(
        "rel-normal",
        "person-etan",
        "project-normal",
        "maintains",
        fact="normal fact",
        source_chunk_id="normal-fact-chunk",
    )

    default_results = store.kg_hybrid_search(
        query_embedding=_embed("Etan"),
        query_text="Etan",
        n_results=10,
        entity_name="Etan",
    )
    checkpoint_results = store.kg_hybrid_search(
        query_embedding=_embed("Etan"),
        query_text="Etan",
        n_results=10,
        entity_name="Etan",
        include_checkpoints=True,
    )
    store.close()

    assert {fact["target_entity"]["name"] for fact in default_results["facts"]} == {"Normal Project"}
    assert {fact["target_entity"]["name"] for fact in checkpoint_results["facts"]} == {
        "Checkpoint Project",
        "Normal Project",
    }


def test_kg_facts_exclude_legacy_checkpoint_content_when_origin_unknown(tmp_path):
    store = VectorStore(tmp_path / "kg-legacy-checkpoint-content-facts.db")
    _insert_chunk(
        store,
        chunk_id="legacy-checkpoint-content-fact-chunk",
        content="[PreCompact checkpoint]\nEtan builds legacy checkpoint-only project",
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
    )
    _insert_chunk(
        store,
        chunk_id="normal-legacy-control-fact-chunk",
        content="Etan builds durable legacy control project",
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
    )
    store.upsert_entity("person-etan", "person", "Etan")
    store.upsert_entity("project-legacy-checkpoint", "project", "Legacy Checkpoint Project")
    store.upsert_entity("project-legacy-normal", "project", "Legacy Normal Project")
    store.add_relation(
        "rel-legacy-checkpoint-content",
        "person-etan",
        "project-legacy-checkpoint",
        "builds",
        fact="legacy checkpoint-only fact",
        source_chunk_id="legacy-checkpoint-content-fact-chunk",
    )
    store.add_relation(
        "rel-legacy-normal",
        "person-etan",
        "project-legacy-normal",
        "maintains",
        fact="legacy normal fact",
        source_chunk_id="normal-legacy-control-fact-chunk",
    )

    sql_default_facts = _kg_facts_sql(store, "Etan")
    sql_checkpoint_facts = _kg_facts_sql(store, "Etan", include_checkpoints=True)
    hybrid_default = store.kg_hybrid_search(
        query_embedding=_embed("Etan"),
        query_text="Etan",
        n_results=10,
        entity_name="Etan",
    )
    hybrid_with_checkpoints = store.kg_hybrid_search(
        query_embedding=_embed("Etan"),
        query_text="Etan",
        n_results=10,
        entity_name="Etan",
        include_checkpoints=True,
    )
    store.close()

    assert {fact["target"] for fact in sql_default_facts} == {"Legacy Normal Project"}
    assert {fact["target"] for fact in sql_checkpoint_facts} == {
        "Legacy Checkpoint Project",
        "Legacy Normal Project",
    }
    assert {fact["target_entity"]["name"] for fact in hybrid_default["facts"]} == {"Legacy Normal Project"}
    assert {fact["target_entity"]["name"] for fact in hybrid_with_checkpoints["facts"]} == {
        "Legacy Checkpoint Project",
        "Legacy Normal Project",
    }


def test_binary_search_overfetches_when_checkpoint_filter_discards_nearest_neighbors(tmp_path):
    store = VectorStore(tmp_path / "binary-overfetch.db")
    query_embedding = [1.0] + ([0.0] * 1023)
    for index in range(5):
        _insert_chunk(
            store,
            chunk_id=f"checkpoint-{index}",
            content=f"[PreCompact checkpoint]\nnoise {index}",
            embedding=query_embedding,
            chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        )
    _insert_chunk(
        store,
        chunk_id="normal-just-outside-binary-window",
        content="normal binary memory just outside the checkpoint nearest-neighbor window",
        embedding=query_embedding,
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
    )
    store.build_binary_index()

    results = store._binary_search(query_embedding=query_embedding, n_results=2)
    store.close()

    assert results["ids"][0] == ["normal-just-outside-binary-window"]


def test_brain_resume_returns_recent_checkpoint_partition(tmp_path, monkeypatch):
    store = VectorStore(tmp_path / "resume.db")
    _insert_chunk(
        store,
        chunk_id="recent-checkpoint",
        content="[PreCompact checkpoint]\nsession_id: session-a\nCurrent task: PR A",
        chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        created_at="2026-05-16T10:00:00+00:00",
        conversation_id="session-a",
    )
    _insert_chunk(
        store,
        chunk_id="normal-note",
        content="Current task: PR A",
        chunk_origin=CHUNK_ORIGIN_UNKNOWN,
        created_at="2026-05-16T10:00:00+00:00",
        conversation_id="session-a",
    )
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._utcnow_iso", lambda: "2026-05-16T12:00:00+00:00")

    content, structured = asyncio.run(_brain_resume(session_id="session-a", lookback_days=7))
    store.close()

    assert structured["total"] == 1
    assert structured["results"][0]["chunk_id"] == "recent-checkpoint"
    assert "PR A" in content[0].text


def test_brain_resume_excludes_archived_checkpoints(tmp_path, monkeypatch):
    store = VectorStore(tmp_path / "resume-lifecycle.db")
    _insert_chunk(
        store,
        chunk_id="active-checkpoint",
        content="[PreCompact checkpoint]\nCurrent task: active",
        chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        created_at="2026-05-16T10:00:00+00:00",
        conversation_id="session-a",
    )
    _insert_chunk(
        store,
        chunk_id="archived-checkpoint",
        content="[PreCompact checkpoint]\nCurrent task: archived",
        chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        created_at="2026-05-16T11:00:00+00:00",
        conversation_id="session-a",
    )
    store.conn.cursor().execute(
        "UPDATE chunks SET archived_at = '2026-05-16T11:30:00+00:00' WHERE id = 'archived-checkpoint'"
    )
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._utcnow_iso", lambda: "2026-05-16T12:00:00+00:00")

    _content, structured = asyncio.run(_brain_resume(session_id="session-a", lookback_days=7))
    store.close()

    assert structured["total"] == 1
    assert structured["results"][0]["chunk_id"] == "active-checkpoint"


def test_brain_resume_retries_busy_read_once(tmp_path, monkeypatch):
    class BusyOnceCursor:
        def __init__(self):
            self.calls = 0

        def execute(self, _sql, _params):
            self.calls += 1
            if self.calls == 1:
                raise apsw.BusyError("database is locked")
            return []

    class BusyOnceStore:
        def __init__(self):
            self.cursor = BusyOnceCursor()

        def _read_cursor(self):
            return self.cursor

    store = BusyOnceStore()
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._utcnow_iso", lambda: "2026-05-16T12:00:00+00:00")
    monkeypatch.setattr("brainlayer.mcp.search_handler.asyncio.sleep", fake_sleep)

    content, structured = asyncio.run(_brain_resume())

    assert structured["total"] == 0
    assert content[0].text == "No PreCompact checkpoints found."
    assert store.cursor.calls == 2
    assert sleep_calls == [0.1]


def test_brain_resume_error_path_preserves_mcp_error_result(monkeypatch):
    class BrokenStore:
        def _read_cursor(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: BrokenStore())
    monkeypatch.setattr("brainlayer.mcp.search_handler._utcnow_iso", lambda: "2026-05-16T12:00:00+00:00")

    result = asyncio.run(_brain_resume())

    assert result.isError is True
    assert result.content[0].text == "Resume error: boom"


def test_brain_resume_escapes_session_id_like_wildcards(tmp_path, monkeypatch):
    store = VectorStore(tmp_path / "resume-like-escape.db")
    _insert_chunk(
        store,
        chunk_id="literal-session",
        content="[PreCompact checkpoint]\nsession_id: abc_def\nCurrent task: literal",
        chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        created_at="2026-05-16T10:00:00+00:00",
        conversation_id="other-literal",
    )
    _insert_chunk(
        store,
        chunk_id="wildcard-false-positive",
        content="[PreCompact checkpoint]\nsession_id: abcXdef\nCurrent task: false positive",
        chunk_origin=CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        created_at="2026-05-16T10:00:00+00:00",
        conversation_id="other-false-positive",
    )
    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", lambda: store)
    monkeypatch.setattr("brainlayer.mcp.search_handler._utcnow_iso", lambda: "2026-05-16T12:00:00+00:00")

    _content, structured = asyncio.run(_brain_resume(session_id="abc_def", lookback_days=7))
    store.close()

    assert structured["total"] == 1
    assert structured["results"][0]["chunk_id"] == "literal-session"


def test_brain_resume_store_creation_error_preserves_mcp_error_result(monkeypatch):
    def raise_store_error():
        raise RuntimeError("boom")

    monkeypatch.setattr("brainlayer.mcp.search_handler._get_vector_store", raise_store_error)

    result = asyncio.run(_brain_resume())

    assert result.isError is True
    assert result.content[0].text == "Resume error: boom"
