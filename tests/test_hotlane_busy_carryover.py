"""S17: the hotlane writer never throws away embedded vectors on BusyError.

Live failure (M4, 2026-11-02): 1,264 of 1,430 logged `hotlane adapter cycle failed`
lines died in `apsw.bestpractice.connection_optimize`, inside the hotlane's
`apsw.Connection()` constructor. APSW's own `connection_busy_timeout` hook had reset
the timeout to 100 ms first, so `PRAGMA optimize` lost to any concurrent writer and
the cycle's already-embedded batch was discarded.
"""

from __future__ import annotations

import functools
import importlib
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import apsw
import apsw.bestpractice
import pytest


def _load_hotlane_module():
    importlib.invalidate_caches()
    sys.modules.pop("scripts.hotlane_brainbar_daemon", None)
    return importlib.import_module("scripts.hotlane_brainbar_daemon")


@pytest.fixture
def hook_calls(monkeypatch):
    """Spy on APSW's connection best-practice hooks without changing what they do."""
    calls: list[tuple[str, int]] = []

    def spy(hook):
        @functools.wraps(hook)
        def wrapper(connection):
            busy_ms = connection.cursor().execute("PRAGMA busy_timeout").fetchone()[0]
            calls.append((hook.__name__, busy_ms))
            return hook(connection)

        return wrapper

    wrapped = [spy(hook) if hook.__name__.startswith("connection_") else hook for hook in apsw.bestpractice.recommended]
    monkeypatch.setattr(apsw.bestpractice, "recommended", wrapped)
    return calls


def _seed_db(db_path: Path, chunk_ids: list[str]) -> None:
    from brainlayer.vector_store import VectorStore

    store = VectorStore(db_path)
    try:
        cursor = store.conn.cursor()
        for chunk_id in chunk_ids:
            cursor.execute(
                "INSERT INTO chunks (id, content, metadata, source_file, source) VALUES (?, ?, '{}', 'brainbar-store', 'mcp')",
                (chunk_id, f"content {chunk_id}"),
            )
    finally:
        store.close()


def _vector_count(db_path: Path, chunk_ids: list[str]) -> int:
    from brainlayer.vector_store import VectorStore

    store = VectorStore(db_path, readonly=True)
    try:
        placeholders = ",".join("?" for _ in chunk_ids)
        return (
            store.conn.cursor()
            .execute(f"SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id IN ({placeholders})", chunk_ids)
            .fetchone()[0]
        )
    finally:
        store.close()


def test_hotlane_writer_skips_optimize_and_keeps_its_busy_timeout_in_constructor(tmp_path, monkeypatch, hook_calls):
    hotlane = _load_hotlane_module()
    monkeypatch.setenv("BRAINLAYER_HOTLANE_WRITE_BUSY_TIMEOUT_MS", "1234")
    db_path = tmp_path / "brainlayer.db"
    _seed_db(db_path, [])

    hook_calls.clear()
    conn = hotlane._open_hotlane_write_connection(db_path)
    try:
        hotlane_names = [name for name, _busy_ms in hook_calls]
        assert "connection_optimize" not in hotlane_names
        assert "connection_busy_timeout" not in hotlane_names
        # The non-maintenance hooks still run, and they see the hotlane budget, not APSW's 100 ms.
        assert "connection_wal" in hotlane_names
        assert {busy_ms for _name, busy_ms in hook_calls} == {1234}
        assert conn.cursor().execute("PRAGMA busy_timeout").fetchone()[0] == 1234
        assert conn.cursor().execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()

    # A normal read-write connection on the same thread still gets every best-practice hook.
    hook_calls.clear()
    normal = apsw.Connection(str(db_path))
    try:
        normal_names = [name for name, _busy_ms in hook_calls]
        assert "connection_optimize" in normal_names
        assert "connection_busy_timeout" in normal_names
    finally:
        normal.close()


def test_skip_connection_hooks_restores_previous_state():
    from brainlayer import vector_store

    with vector_store.skip_connection_hooks("connection_optimize"):
        with vector_store.skip_connection_hooks("connection_busy_timeout"):
            assert vector_store._CONNECTION_HOOK_STATE.skip_hook_names == {
                "connection_optimize",
                "connection_busy_timeout",
            }
        assert vector_store._CONNECTION_HOOK_STATE.skip_hook_names == {"connection_optimize"}
    assert not hasattr(vector_store._CONNECTION_HOOK_STATE, "skip_hook_names")


def test_hotlane_write_waits_out_a_two_second_lock_with_zero_vectors_lost(tmp_path, monkeypatch, hook_calls):
    """The S17 invariant, on a real temp DB with a lock held by another process."""
    hotlane = _load_hotlane_module()
    monkeypatch.delenv("BRAINLAYER_HOTLANE_WRITE_BUSY_TIMEOUT_MS", raising=False)
    monkeypatch.setattr("brainlayer.search_repo.clear_hybrid_search_cache", lambda _db_path: None)
    db_path = tmp_path / "brainlayer.db"
    chunk_ids = [f"chunk-{index}" for index in range(16)]
    _seed_db(db_path, chunk_ids)
    vectors = [hotlane.EmbeddedVector(chunk_id, f"content {chunk_id}", [0.25] * 1024) for chunk_id in chunk_ids]
    hold_seconds = 2.0

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import sqlite3, sys, time
                conn = sqlite3.connect({str(db_path)!r}, isolation_level=None)
                conn.execute("BEGIN IMMEDIATE")
                print("locked", flush=True)
                time.sleep({hold_seconds})
                conn.execute("COMMIT")
                conn.close()
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        locked_at = time.monotonic()
        hook_calls.clear()
        count = hotlane._write_embedded_vectors(db_path, vectors)
        finished_at = time.monotonic()
        hotlane_hooks = [name for name, _busy_ms in hook_calls]
    finally:
        holder.wait(timeout=30)

    assert holder.returncode == 0
    assert finished_at - locked_at >= hold_seconds - 0.1, "the write must have waited for the lock"
    assert count == len(vectors)
    assert _vector_count(db_path, chunk_ids) == len(vectors)
    # At least one busy attempt then a successful one: two hotlane opens, neither ran optimize.
    assert hotlane_hooks.count("connection_wal") >= 2
    assert "connection_optimize" not in hotlane_hooks


def test_write_retries_busy_begin_then_defers_only_unwritten_vectors(monkeypatch):
    hotlane = _load_hotlane_module()
    sleeps = []
    monkeypatch.setattr(hotlane, "_sleep", sleeps.append)
    monkeypatch.setattr("brainlayer.search_repo.clear_hybrid_search_cache", lambda _db_path: None)
    upserted = []
    begins = []

    class FakeCursor:
        def execute(self, sql, params=()):
            statement = " ".join(sql.split())
            if statement == "BEGIN IMMEDIATE":
                begins.append(statement)
                # The first vector commits; every BEGIN after it hits a held lock.
                if len(begins) > 1:
                    raise apsw.BusyError("database is locked")
            if statement.startswith("SELECT 1"):
                return SimpleNamespace(fetchone=lambda: (1,))
            return []

    class FakeStore:
        db_path = Path("/nonexistent/brainlayer.db")
        conn = SimpleNamespace(cursor=FakeCursor)

        def _upsert_chunk_vector(self, _cursor, chunk_id, _embedding):
            upserted.append(chunk_id)

    vectors = [hotlane.EmbeddedVector(f"chunk-{index}", f"content {index}", [0.1]) for index in range(3)]
    with pytest.raises(hotlane.HotlaneVectorWriteDeferred) as raised:
        hotlane._write_embedded_vectors(FakeStore(), vectors)

    assert isinstance(raised.value, apsw.BusyError)
    assert raised.value.written == 1
    assert raised.value.attempts == hotlane.HOTLANE_WRITE_ATTEMPTS == 3
    assert [vector.chunk_id for vector in raised.value.remaining] == ["chunk-1", "chunk-2"]
    assert isinstance(raised.value.cause, apsw.BusyError)
    assert upserted == ["chunk-0"]
    assert len(begins) == 1 + hotlane.HOTLANE_WRITE_ATTEMPTS
    backoff = [seconds for seconds in sleeps if seconds != hotlane.VECTOR_WRITE_YIELD_SECONDS]
    assert backoff == [hotlane.HOTLANE_WRITE_RETRY_BACKOFF_SECONDS, hotlane.HOTLANE_WRITE_RETRY_BACKOFF_SECONDS * 2]


def test_split_cycle_keeps_carried_vectors_when_the_candidate_scan_fails(tmp_path):
    hotlane = _load_hotlane_module()
    db_path = tmp_path / "brainlayer.db"
    db_path.touch()
    carryover = hotlane.VectorCarryover()
    carried = hotlane.EmbeddedVector("carried", "carried content", [0.5])
    carryover.put([carried])

    def failing_scan(_store, *, limit):
        raise apsw.BusyError("database is locked")

    with pytest.raises(apsw.BusyError):
        hotlane._run_split_cycle(
            db_path=db_path,
            vector_store_cls=lambda _path, readonly=False: SimpleNamespace(close=lambda: None),
            embed_fn=lambda _text: [0.0],
            recent_limit=1,
            backlog_batch=0,
            enrich_limit=0,
            enrich_since_hours=8760,
            candidate_rows_fn=failing_scan,
            pending_rows_fn=lambda _store, *, limit: [],
            write_vectors_fn=lambda *_args: pytest.fail("no write after a failed scan"),
            carryover=carryover,
        )

    assert carryover.peek() == [carried]


def test_split_cycle_writes_partial_progress_and_carries_the_rest(tmp_path):
    hotlane = _load_hotlane_module()
    db_path = tmp_path / "brainlayer.db"
    db_path.touch()
    carryover = hotlane.VectorCarryover()
    candidates = [hotlane.EmbedCandidate(f"hot-{index}", f"content {index}") for index in range(3)]
    written_batches = []

    def write_vectors(_path, vectors):
        written_batches.append([vector.chunk_id for vector in vectors])
        if len(written_batches) == 1:
            raise hotlane.HotlaneVectorWriteDeferred(
                written=1, remaining=vectors[1:], attempts=3, cause=apsw.BusyError("database is locked")
            )
        return len(vectors)

    scan = list(candidates)
    embedded_ids = []

    def embed(text):
        embedded_ids.append(text)
        return [0.0]

    def cycle():
        return hotlane._run_split_cycle(
            db_path=db_path,
            vector_store_cls=lambda _path, readonly=False: SimpleNamespace(close=lambda: None),
            embed_fn=embed,
            recent_limit=3,
            backlog_batch=0,
            enrich_limit=0,
            enrich_since_hours=8760,
            candidate_rows_fn=lambda _store, *, limit: list(scan),
            pending_rows_fn=lambda _store, *, limit: [],
            write_vectors_fn=write_vectors,
            carryover=carryover,
        )

    first = cycle()
    assert (first.embedded, first.deferred) == (1, 2)
    assert [vector.chunk_id for vector in carryover.peek()] == ["hot-1", "hot-2"]

    # Next cycle the scan still finds the two unvectored chunks; they are written from the
    # carry-over, not embedded again.
    scan[:] = candidates[1:]
    second = cycle()
    assert (second.embedded, second.deferred) == (2, 0)
    assert written_batches == [["hot-0", "hot-1", "hot-2"], ["hot-1", "hot-2"]]
    assert embedded_ids == ["content 0", "content 1", "content 2"]
    assert len(carryover) == 0


def test_carryover_is_bounded_and_counts_what_it_drops(caplog):
    hotlane = _load_hotlane_module()
    carryover = hotlane.VectorCarryover(capacity=2)
    vectors = [hotlane.EmbeddedVector(f"chunk-{index}", "content", [0.0]) for index in range(3)]

    with caplog.at_level("WARNING", logger="brainlayer.hotlane_brainbar"):
        assert carryover.put(vectors) == 1

    assert [vector.chunk_id for vector in carryover.peek()] == ["chunk-1", "chunk-2"]
    assert carryover.dropped_total == 1
    assert carryover.carried_total == 3
    assert "dropped 1 oldest computed vectors" in caplog.text
    assert "dropped_total=1" in caplog.text
