"""SessionMixin.update_enrichment stamps enriched_by atomically without clobbering metadata."""

from __future__ import annotations

import json
import sqlite3


class _SpyCursor:
    def __init__(self, inner: sqlite3.Cursor, statements: list[str]) -> None:
        self._inner = inner
        self._statements = statements

    def execute(self, sql, parameters=()):
        self._statements.append(sql)
        return self._inner.execute(sql, parameters)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _SpyConnection:
    def __init__(self, inner: sqlite3.Connection) -> None:
        self._inner = inner
        self.statements: list[str] = []

    def cursor(self):
        return _SpyCursor(self._inner.cursor(), self.statements)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_update_enrichment_preserves_existing_metadata_keys_via_json_set():
    from brainlayer.session_repo import SessionMixin

    class Store(SessionMixin):
        _has_chunk_origin = True

    store = Store()
    store.db_path = None
    real_conn = sqlite3.connect(":memory:")
    real_conn.execute(
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
    real_conn.execute(
        "INSERT INTO chunks (id, chunk_origin, metadata) VALUES (?, ?, ?)",
        ("c1", "unknown", json.dumps({"existing": True, "source_class": "cli_agent"})),
    )
    store.conn = _SpyConnection(real_conn)

    store.update_enrichment("c1", summary="s", enrichment_model="gemini-2.5-flash-lite")

    assert any("json_set" in sql.lower() for sql in store.conn.statements)
    origin, metadata_raw = real_conn.execute("SELECT chunk_origin, metadata FROM chunks WHERE id = 'c1'").fetchone()
    metadata = json.loads(metadata_raw)
    assert origin == "unknown"
    assert metadata["existing"] is True
    assert metadata["source_class"] == "cli_agent"
    assert metadata["enriched_by"] == "gemini-2.5-flash-lite"
    assert "_previous_metadata_raw" not in metadata
