"""Tests for non-destructive legacy re-enrichment setup."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from brainlayer.vector_store import VectorStore


def test_vector_store_migrates_summary_v2_columns_on_existing_db(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    try:
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
                source TEXT,
                sender TEXT,
                language TEXT,
                conversation_id TEXT,
                position INTEGER,
                context_summary TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    store = VectorStore(db_path)
    try:
        columns = {row[1] for row in store.conn.cursor().execute("PRAGMA table_info(chunks)")}
        assert "summary_v2" in columns
        assert "enrichment_version" in columns
    finally:
        store.close()


def test_update_reenrichment_preview_preserves_existing_summary(tmp_path):
    store = VectorStore(tmp_path / "preview.db")
    created_at = datetime.now(timezone.utc).isoformat()
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type, char_count,
            summary, enriched_at, created_at, source
        ) VALUES (?, ?, '{}', 'test.jsonl', 'brainlayer', 'assistant_text', ?, ?, ?, ?, 'claude_code')
        """,
        ("chunk-1", "x" * 120, 120, "existing summary", created_at, created_at),
    )

    store.update_reenrichment_preview(
        chunk_id="chunk-1",
        summary_v2="improved summary",
        enrichment_version="2.0",
    )

    row = cursor.execute(
        "SELECT summary, summary_v2, enrichment_version, enriched_at FROM chunks WHERE id = ?",
        ("chunk-1",),
    ).fetchone()
    store.close()

    assert row == ("existing summary", "improved summary", "2.0", created_at)
