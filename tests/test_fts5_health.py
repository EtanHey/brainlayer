"""Tests for FTS5 health monitoring and self-healing."""

import json
from unittest.mock import patch

import pytest

from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore backed by a temp DB."""
    db_path = tmp_path / "fts5-health.db"
    s = VectorStore(db_path)
    yield s
    s.close()


def _insert_chunk(store: VectorStore, chunk_id: str, content: str | None = None) -> None:
    """Insert a chunk and let triggers populate FTS rows."""
    text = content or f"content for {chunk_id}"
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type, char_count, source, tags, summary
        ) VALUES (?, ?, ?, 'test.jsonl', 'brainlayer', 'assistant_text', ?, 'test', ?, ?)
        """,
        (
            chunk_id,
            text,
            json.dumps({"test": True}),
            len(text),
            json.dumps(["fts5", "health"]),
            text[:50],
        ),
    )


def _insert_chunks(store: VectorStore, total: int) -> None:
    """Insert N chunks into the temp DB."""
    for idx in range(total):
        _insert_chunk(store, f"chunk-{idx}")


class TestFTS5HealthMonitoring:
    def test_health_events_table_created(self, store):
        """health_events table exists after VectorStore init."""
        tables = {
            row[0]
            for row in store.conn.cursor().execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='health_events'"
            )
        }
        assert "health_events" in tables

    def test_check_fts5_health_synced(self, store):
        """Matching chunk and FTS counts report synced health."""
        _insert_chunks(store, 5)

        result = store.check_fts5_health(cache_ttl_seconds=0)

        assert result["synced"] is True
        assert result["chunk_count"] == 5
        assert result["fts_count"] == 5
        assert result["desync_pct"] == 0.0
        assert result["severity"] == "info"

    def test_check_fts5_health_desync_warning(self, store):
        """A 2% desync should surface as warning."""
        _insert_chunks(store, 100)
        store.conn.cursor().execute("DELETE FROM chunks_fts WHERE chunk_id IN ('chunk-0', 'chunk-1')")

        result = store.check_fts5_health(cache_ttl_seconds=0)

        assert result["synced"] is False
        assert result["chunk_count"] == 100
        assert result["fts_count"] == 98
        assert result["desync_pct"] == pytest.approx(2.0)
        assert result["severity"] == "warning"

    def test_check_fts5_health_desync_critical(self, store):
        """A 10% desync should surface as critical."""
        _insert_chunks(store, 10)
        store.conn.cursor().execute("DELETE FROM chunks_fts WHERE chunk_id = 'chunk-0'")

        result = store.check_fts5_health(cache_ttl_seconds=0)

        assert result["synced"] is False
        assert result["chunk_count"] == 10
        assert result["fts_count"] == 9
        assert result["desync_pct"] == pytest.approx(10.0)
        assert result["severity"] == "critical"

    def test_check_fts5_health_desync_emergency(self, store):
        """A desync above 20% should trigger an emergency rebuild."""
        _insert_chunks(store, 4)
        store.conn.cursor().execute("DELETE FROM chunks_fts WHERE chunk_id = 'chunk-0'")

        result = store.check_fts5_health(cache_ttl_seconds=0)

        assert result["severity"] == "emergency"
        assert result["rebuild_triggered"] is True
        assert result["synced"] is True
        assert result["chunk_count"] == 4
        assert result["fts_count"] == 4
        assert result["desync_pct"] == 0.0

    def test_check_fts5_health_cache(self, store):
        """Repeated checks within TTL should avoid a second DB read."""
        _insert_chunks(store, 3)

        with patch.object(store, "_read_cursor", wraps=store._read_cursor) as read_cursor:
            first = store.check_fts5_health(cache_ttl_seconds=60)
            second = store.check_fts5_health(cache_ttl_seconds=60)

        assert first == second
        assert read_cursor.call_count == 1

    def test_check_wal_health(self, store):
        """WAL health exposes size and checkpoint fields."""
        _insert_chunks(store, 3)

        result = store.check_wal_health()

        assert set(result) >= {
            "wal_path",
            "wal_exists",
            "wal_size_bytes",
            "wal_size_mb",
            "checkpoint_status",
            "severity",
        }
        assert isinstance(result["checkpoint_status"], dict)

    def test_deep_integrity_check(self, store):
        """Deep integrity check should pass on a healthy index."""
        _insert_chunks(store, 12)

        result = store.deep_integrity_check()

        assert result["ok"] is True
        assert result["fts_integrity"] == "ok"
        assert result["missing_chunk_ids"] == []
        assert result["spot_check_count"] == 12

    def test_rebuild_fts5(self, store):
        """Explicit rebuild should restore missing FTS rows."""
        _insert_chunks(store, 6)
        store.conn.cursor().execute("DELETE FROM chunks_fts WHERE chunk_id IN ('chunk-0', 'chunk-1')")

        result = store.rebuild_fts5()

        assert result["success"] is True
        assert result["chunk_count"] == 6
        assert result["fts_count"] == 6
        assert result["desync_pct"] == 0.0

    def test_health_events_logged(self, store):
        """Non-OK checks should append rows to health_events."""
        _insert_chunks(store, 100)
        store.conn.cursor().execute("DELETE FROM chunks_fts WHERE chunk_id IN ('chunk-0', 'chunk-1')")

        store.check_fts5_health(cache_ttl_seconds=0)

        rows = list(
            store.conn.cursor().execute(
                "SELECT event_type, severity FROM health_events ORDER BY id"
            )
        )
        assert ("fts5_desync_warning", "warning") in rows
