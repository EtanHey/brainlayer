"""SessionMixin.update_enrichment must not overwrite chunk_origin."""

from __future__ import annotations

import json
import sqlite3


def test_update_enrichment_stamps_enriched_by_without_changing_origin():
    from brainlayer.session_repo import SessionMixin

    class Store(SessionMixin):
        _has_chunk_origin = True

    store = Store()
    store.db_path = None
    store.conn = sqlite3.connect(":memory:")
    store.conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            enriched_at TEXT,
            enrich_status TEXT,
            summary TEXT,
            chunk_origin TEXT DEFAULT 'unknown',
            enrichment_model TEXT,
            metadata TEXT DEFAULT '{}'
        )
        """
    )
    store.conn.execute("INSERT INTO chunks (id, chunk_origin) VALUES (?, ?)", ("c1", "unknown"))
    store.update_enrichment("c1", summary="s", enrichment_model="gemini-2.5-flash-lite")

    origin, metadata_raw = store.conn.execute("SELECT chunk_origin, metadata FROM chunks WHERE id = 'c1'").fetchone()
    assert origin == "unknown"
    assert json.loads(metadata_raw)["enriched_by"] == "gemini-2.5-flash-lite"
