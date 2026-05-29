"""Write-resilience under load — RED-first regression for the WAL-starvation fix.

Root cause (confirmed live, BL-LEAD gen-10 2026-05-29): brain_store intermittently
queued ("DB busy") because the WAL file grew unbounded — `drain.py` used a PASSIVE
checkpoint (cannot reclaim WAL under reader load) and `journal_size_limit` was -1
(the WAL file never truncated back after a checkpoint). Observed multi-GB WAL in
production logs (2.4-3.9GB); a real store queued with WAL at only 33MB.

The fix:
  1. vector_store.py sets `PRAGMA journal_size_limit` (env BRAINLAYER_WAL_SIZE_LIMIT_BYTES,
     default 256MB) on every connection so the WAL truncates back after a checkpoint.
  2. drain.py checkpoints with TRUNCATE (not PASSIVE) so each drained batch reclaims WAL.

These tests fail on origin/main (journal_size_limit == -1, WAL unbounded) and pass
after the fix. They use isolated tmp DBs only — never the real ~/.local/share DB.
"""

from __future__ import annotations

import os
import threading
import time

import apsw
import pytest

from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore


def _wal_path(db_path) -> str:
    return str(db_path) + "-wal"


def _wal_bytes(db_path) -> int:
    try:
        return os.path.getsize(_wal_path(db_path))
    except OSError:
        return 0


@pytest.fixture
def small_wal_env(monkeypatch):
    """Cap the WAL at 256KB and disable autocheckpoint so the test controls timing."""
    limit = 256_000
    monkeypatch.setenv("BRAINLAYER_WAL_SIZE_LIMIT_BYTES", str(limit))
    monkeypatch.setenv("BRAINLAYER_WAL_AUTOCHECKPOINT", "0")
    return limit


def _write_chunks(store: VectorStore, n: int, prefix: str) -> list[str]:
    ids: list[str] = []
    for i in range(n):
        res = store_memory(
            store=store,
            embed_fn=None,
            content=f"{prefix}-{i:04d}: " + ("write-resilience load payload " * 40),
            memory_type="note",
            project="brainlayer-write-resilience",
            tags=["write-resilience", "load-test", prefix],
            importance=3,
        )
        cid = res.get("id") if isinstance(res, dict) else None
        if cid:
            ids.append(cid)
    return ids


def test_journal_size_limit_is_set(tmp_path, small_wal_env):
    """RED on origin/main: journal_size_limit defaults to -1 (unbounded WAL file)."""
    db = tmp_path / "bl.db"
    store = VectorStore(db)
    try:
        limit = store.conn.cursor().execute("PRAGMA journal_size_limit").fetchone()[0]
    finally:
        store.close()
    assert limit == small_wal_env, (
        f"journal_size_limit={limit} (expected {small_wal_env}). On origin/main this is -1, "
        "so the WAL file never truncates back after a checkpoint and grows unbounded."
    )


def test_wal_file_bounded_after_checkpoint(tmp_path, small_wal_env):
    """RED on origin/main: PASSIVE checkpoint + journal_size_limit=-1 leaves a large WAL file.

    Reproduces the production mechanism: WAL grows under a batch of writes, a PASSIVE
    checkpoint runs (the old drain behaviour), and the WAL FILE must shrink back to the
    limit. With journal_size_limit=-1 the file stays at its multi-MB high-water mark.
    """
    db = tmp_path / "bl.db"
    store = VectorStore(db)
    try:
        # Grow the WAL well past the 256KB limit (autocheckpoint disabled).
        _write_chunks(store, 400, "grow")
        assert _wal_bytes(db) > small_wal_env, "precondition: WAL should exceed the limit before checkpoint"
        # Old drain behaviour: PASSIVE checkpoint, then a tiny write to force the WAL reset
        # at which point journal_size_limit truncation applies.
        store.conn.cursor().execute("PRAGMA wal_checkpoint(PASSIVE)")
        _write_chunks(store, 1, "reset")
        store.conn.cursor().execute("PRAGMA wal_checkpoint(PASSIVE)")
        wal_now = _wal_bytes(db)
    finally:
        store.close()
    assert wal_now <= small_wal_env * 1.5, (
        f"WAL file is {wal_now} bytes after checkpoint (limit {small_wal_env}). "
        "On origin/main (journal_size_limit=-1) it stays at the multi-MB high-water mark — "
        "the unbounded-WAL starvation root cause."
    )


def test_all_stores_land_under_concurrent_writer(tmp_path, small_wal_env):
    """Acceptance guard: every store lands (read-back) even with a competing writer.

    A background thread repeatedly takes the write lock (BEGIN IMMEDIATE + slow write),
    reproducing the enrichment/drain contention that made MCP stores queue. All N stores
    must be retrievable from the DB afterwards. Stays green before/after — regression net.
    """
    db = tmp_path / "bl.db"
    # Initialise schema once before the contender opens its connection.
    VectorStore(db).close()

    stop = threading.Event()
    acquired = {"n": 0}

    def contender() -> None:
        # Hold the write lock in short bursts to contend with the main thread's
        # stores. BEGIN IMMEDIATE acquires the write lock immediately — no INSERT
        # is needed (and a schema-specific INSERT here is brittle: chunks has
        # NOT NULL columns like source_file, so a partial INSERT silently fails
        # and exercises no contention at all). The held lock forces the main
        # thread's store_memory to wait on its busy_timeout.
        c = apsw.Connection(str(db))
        c.setbusytimeout(5_000)
        cur = c.cursor()
        while not stop.is_set():
            try:
                cur.execute("BEGIN IMMEDIATE")
                acquired["n"] += 1
                time.sleep(0.02)
                cur.execute("COMMIT")
            except apsw.Error:
                try:
                    cur.execute("ROLLBACK")
                except apsw.Error:
                    pass
            time.sleep(0.005)
        c.close()

    t = threading.Thread(target=contender, daemon=True)
    t.start()
    try:
        store = VectorStore(db)
        n = 50
        ids = _write_chunks(store, n, "land")
        store.close()
    finally:
        stop.set()
        t.join(timeout=5)

    assert acquired["n"] > 0, "contender never took the write lock — no contention was exercised"
    assert len(ids) == n, f"only {len(ids)}/{n} stores returned an id under contention"
    reader = apsw.Connection(str(db), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        landed = sum(
            1 for cid in ids if list(reader.cursor().execute("SELECT 1 FROM chunks WHERE id=? LIMIT 1", (cid,)))
        )
    finally:
        reader.close()
    assert landed == n, f"only {landed}/{n} stores landed (read-back) under writer contention"
