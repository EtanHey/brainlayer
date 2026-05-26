from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from brainlayer.pipeline.enrichment import mark_unenrichable
from brainlayer.vector_store import VectorStore


def test_enrich_status_migration_backfills_polluted_enriched_at_rows(tmp_path):
    db_path = tmp_path / "legacy-polluted.db"
    store = VectorStore(db_path)
    store.close()

    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO chunks (id, content, metadata, source_file, char_count, summary, enriched_at) VALUES (?, ?, '{}', 'test.jsonl', ?, ?, ?)",
            [
                ("success", "success content", 120, "summary", "2026-05-26T10:00:00+00:00"),
                ("duplicate", "duplicate content", 120, None, "skipped:duplicate"),
                ("too-short", "tiny", 4, None, "skipped:too_short"),
                ("blank", "blank content", 120, None, ""),
                ("pending", "pending content", 120, None, None),
            ],
        )
        conn.execute("UPDATE chunks SET enrich_status = NULL")
        conn.execute("DELETE FROM schema_migrations WHERE name = '2026_05_26_enrich_status_v1'")
        conn.commit()
    finally:
        conn.close()

    store = VectorStore(db_path)
    try:
        rows = {
            row[0]: (row[1], row[2])
            for row in store.conn.cursor().execute("SELECT id, enriched_at, enrich_status FROM chunks ORDER BY id")
        }
    finally:
        store.close()

    assert rows["success"] == ("2026-05-26T10:00:00+00:00", "success")
    assert rows["duplicate"] == (None, "duplicate")
    assert rows["too-short"] == (None, "too_short")
    assert rows["blank"] == (None, None)
    assert rows["pending"] == (None, None)


def test_enriched_at_values_are_parseable_datetime_or_null_after_migration(tmp_path):
    db_path = tmp_path / "datetime-integrity.db"
    store = VectorStore(db_path)
    store.close()

    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO chunks (id, content, metadata, source_file, enriched_at) VALUES (?, ?, '{}', 'test.jsonl', ?)",
            [
                ("polluted", "polluted content", "skipped:duplicate"),
                ("valid", "valid content", "2026-05-26T10:00:00+00:00"),
                ("blank", "blank content", ""),
                ("pending", "pending content", None),
            ],
        )
        conn.execute("UPDATE chunks SET enrich_status = NULL")
        conn.execute("DELETE FROM schema_migrations WHERE name = '2026_05_26_enrich_status_v1'")
        conn.commit()
    finally:
        conn.close()

    store = VectorStore(db_path)
    try:
        enriched_values = [row[0] for row in store.conn.cursor().execute("SELECT enriched_at FROM chunks")]
    finally:
        store.close()

    for enriched_at in enriched_values:
        if enriched_at is not None:
            datetime.fromisoformat(enriched_at.replace("Z", "+00:00"))


def test_mark_unenrichable_writes_status_not_enriched_at(tmp_path):
    store = VectorStore(tmp_path / "unenrichable.db")
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type, char_count, source
        ) VALUES (
            'short', 'tiny', '{}', 'test.jsonl', 'brainlayer', 'assistant_text', 4, 'claude_code'
        )
        """
    )

    try:
        tagged = mark_unenrichable(store)
        row = cursor.execute("SELECT enriched_at, enrich_status FROM chunks WHERE id = 'short'").fetchone()
    finally:
        store.close()

    assert tagged == 1
    assert row == (None, "too_short")


def test_update_enrichment_sets_success_status(tmp_path):
    store = VectorStore(tmp_path / "success-status.db")
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type, char_count, source
        ) VALUES (
            'chunk-1', 'content that should be enriched', '{}', 'test.jsonl',
            'brainlayer', 'assistant_text', 31, 'claude_code'
        )
        """
    )

    try:
        store.update_enrichment("chunk-1", summary="summary")
        enriched_at, enrich_status = cursor.execute(
            "SELECT enriched_at, enrich_status FROM chunks WHERE id = 'chunk-1'"
        ).fetchone()
    finally:
        store.close()

    assert enrich_status == "success"
    assert enriched_at is not None
    datetime.fromisoformat(enriched_at.replace("Z", "+00:00"))


def test_enrich_status_migration_sql_applies_cleanly(tmp_path):
    database_path = tmp_path / "migration.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            enriched_at TEXT
        )
        """
    )
    connection.executemany(
        "INSERT INTO chunks (id, enriched_at) VALUES (?, ?)",
        [("dupe", "skipped:duplicate"), ("blank", "")],
    )
    migration_sql = (Path(__file__).parent.parent / "migrations/004_enrich_status.sql").read_text()

    connection.executescript(migration_sql)

    rows = dict(connection.execute("SELECT id, enriched_at FROM chunks ORDER BY id").fetchall())
    row = connection.execute("SELECT enriched_at, enrich_status FROM chunks WHERE id = 'dupe'").fetchone()
    assert row == (None, "duplicate")
    assert rows["blank"] is None
