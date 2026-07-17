import sqlite3
import time

from brainlayer.cli import _RewindArchiveBatcher
from brainlayer.vector_store import VectorStore


class _CountingCursor:
    def __init__(self, cursor: sqlite3.Cursor, counters: dict[str, int]):
        self._cursor = cursor
        self._counters = counters

    def execute(self, sql: str, parameters=()):
        if sql.lstrip().upper().startswith("UPDATE"):
            self._counters["updates"] += 1
        if parameters:
            return self._cursor.execute(sql, parameters)
        return self._cursor.execute(sql)

    def __getattr__(self, item):
        return getattr(self._cursor, item)


class _CountingConnection:
    def __init__(self, path, counters: dict[str, int]):
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._counters = counters
        self._change_anchor = 0

    def cursor(self):
        return _CountingCursor(self._conn.cursor(), self._counters)

    def close(self) -> None:
        self._conn.close()

    def changes(self) -> int:
        total_changes = self._conn.total_changes
        delta = total_changes - self._change_anchor
        self._change_anchor = total_changes
        return delta

    def __getattr__(self, item):
        return getattr(self._conn, item)


class _CountingVectorStore:
    def __init__(self, db_path, counters: dict[str, int]):
        counters["opens"] += 1
        self.conn = _CountingConnection(db_path, counters)

    def close(self) -> None:
        self.conn.close()


def _prepare_rewind_archive_db(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    rows = [
        ("s1-a", "s1", "realtime_watcher"),
        ("s1-b", "s1", "realtime_watcher"),
        ("s2-a", "s2", "realtime_watcher"),
        ("s2-b", "s2", "realtime_watcher"),
        ("s3-a", "s3", "realtime_watcher"),
        ("other", "s1", "manual"),
    ]
    for chunk_id, session_id, source in rows:
        store.conn.execute(
            """
            INSERT INTO chunks (
              id, content, metadata, source_file, source, conversation_id, value_type, content_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, "test", "{}", "/tmp/session.jsonl", source, session_id, "KNOWLEDGE", "assistant_text"),
        )
    store.conn.execute(
        "INSERT INTO chunks (id, content, metadata, source_file, source, conversation_id, value_type, content_type, archived_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "pre",
            "already",
            "{}",
            "/tmp/session.jsonl",
            "realtime_watcher",
            "s3",
            "KNOWLEDGE",
            "assistant_text",
            "2026-01-01T00:00:00Z",
        ),
    )
    store.close()
    return db_path


def test_rewind_archiver_batches_multiple_sessions(tmp_path):
    db_path = _prepare_rewind_archive_db(tmp_path)
    counters = {"opens": 0, "updates": 0}

    archiver = _RewindArchiveBatcher(
        db_path=db_path,
        batch_size=10_000,
        flush_interval_ms=60_000,
        vector_store_factory=lambda p: _CountingVectorStore(p, counters),
    )

    archiver.add("s1")
    archiver.maybe_flush("interval")
    archiver.add("s2")
    archiver.maybe_flush("interval")
    archiver.add("s3")
    archiver.maybe_flush("interval")

    assert archiver.pending_count == 3
    assert archiver.flush("shutdown") == 5

    assert counters["opens"] == 1
    assert counters["updates"] == 1

    conn = sqlite3.connect(db_path)
    archived = conn.execute("SELECT conversation_id, source FROM chunks WHERE archived_at IS NOT NULL").fetchall()
    assert ("s1", "realtime_watcher") in archived
    assert ("s2", "realtime_watcher") in archived
    assert ("s3", "realtime_watcher") in archived  # pre-archived baseline
    assert ("s1", "manual") not in archived
    conn.close()
    archiver.close()


def test_rewind_archiver_flushes_with_production_vector_store(tmp_path):
    db_path = _prepare_rewind_archive_db(tmp_path)
    archiver = _RewindArchiveBatcher(
        db_path=db_path,
        batch_size=10_000,
        flush_interval_ms=60_000,
    )
    archiver.add("s1")

    try:
        assert archiver.flush("rewind") == 2
        assert archiver.pending_count == 0
    finally:
        archiver.close()

    conn = sqlite3.connect(db_path)
    archived = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE conversation_id = 's1' AND source = 'realtime_watcher' AND archived_at IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    assert archived == 2


def test_rewind_archiver_flushes_on_threshold(tmp_path):
    db_path = _prepare_rewind_archive_db(tmp_path)
    counters = {"opens": 0, "updates": 0}
    archiver = _RewindArchiveBatcher(
        db_path=db_path,
        batch_size=2,
        flush_interval_ms=60_000,
        vector_store_factory=lambda p: _CountingVectorStore(p, counters),
    )

    archiver.add("s1")
    assert archiver.maybe_flush("interval") == 0
    archiver.add("s2")
    assert archiver.maybe_flush("interval") == 4

    assert archiver.pending_count == 0
    assert counters["opens"] == 1
    assert counters["updates"] == 1

    conn = sqlite3.connect(db_path)
    archived = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE source='realtime_watcher' AND archived_at IS NOT NULL AND conversation_id IN ('s1', 's2')"
    ).fetchone()[0]
    assert archived == 4
    conn.close()
    archiver.close()


def test_rewind_archiver_rewind_interval_shutdown_flush(tmp_path):
    db_path = _prepare_rewind_archive_db(tmp_path)
    counters = {"opens": 0, "updates": 0}
    archiver = _RewindArchiveBatcher(
        db_path=db_path,
        batch_size=10_000,
        flush_interval_ms=5,
        vector_store_factory=lambda p: _CountingVectorStore(p, counters),
    )

    archiver.add("s1")
    assert archiver.pending_count == 1
    assert archiver.maybe_flush("interval") == 0
    time.sleep(0.02)
    assert archiver.maybe_flush("interval") == 2
    assert counters["updates"] == 1
    assert archiver.pending_count == 0

    archiver.add("s2")
    archiver.flush("shutdown")
    assert archiver.pending_count == 0
    assert counters["updates"] == 2

    conn = sqlite3.connect(db_path)
    archived = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE source='realtime_watcher' AND archived_at IS NOT NULL AND conversation_id IN ('s1', 's2')"
    ).fetchone()[0]
    assert archived == 4
    conn.close()
    archiver.close()


def test_rewind_archiver_idempotent_no_duplicate_archival(tmp_path):
    db_path = _prepare_rewind_archive_db(tmp_path)
    counters = {"opens": 0, "updates": 0}
    archiver = _RewindArchiveBatcher(
        db_path=db_path,
        batch_size=1,
        flush_interval_ms=60_000,
        vector_store_factory=lambda p: _CountingVectorStore(p, counters),
    )

    archiver.add("s1")
    first_flush = archiver.flush("test")
    archiver.add("s1")
    second_flush = archiver.flush("test")

    assert first_flush == 2
    assert second_flush == 0
    assert archiver.archived_total == 2
    assert archiver.pending_count == 0

    conn = sqlite3.connect(db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE conversation_id = 's1' AND source='realtime_watcher' AND archived_at IS NOT NULL"
    ).fetchone()[0]
    assert count == 2
    conn.close()
    assert counters["opens"] == 1
