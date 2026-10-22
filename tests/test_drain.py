"""Rewind archive writes archived_at only. Behavioral coverage: test_rewind_batch_archival.py."""

import json
import threading
import time
from contextlib import closing

import apsw
import pytest

from brainlayer import drain
from brainlayer.drain import _apply_rewind_archive
from brainlayer.vector_store import VectorStore


def test_drain_rewind_archive_helper_exists():
    assert callable(_apply_rewind_archive)


def _queued_events(tmp_path, monkeypatch, budget_ms=10):
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    monkeypatch.setenv("BRAINLAYER_DRAIN_TRANSACTION_BUDGET_MS", str(budget_ms))
    db = tmp_path / "brainlayer.db"
    VectorStore(db).close()
    queue = tmp_path / "queue"
    queue.mkdir()
    spool = queue / "batch.jsonl"
    events = [
        {
            "kind": "watcher_chunk",
            "chunk_id": f"event-{i}",
            "content": "A stable shared memory for replay testing",
            "created_at": "2026-10-22T00:00:00Z",
        }
        for i in range(3)
    ]
    spool.write_text("".join(json.dumps(event) + "\n" for event in events))
    return db, queue, spool


def test_drain_yields_writer_lock_between_events_when_budget_exceeded(tmp_path, monkeypatch):
    db, queue, spool = _queued_events(tmp_path, monkeypatch, budget_ms=250)
    first_written = threading.Event()
    writer_finished = threading.Event()
    observed = []
    errors = []
    original = drain._apply_event
    calls = 0

    def apply(conn, event):
        nonlocal calls
        calls += 1
        if calls > 1:
            # Causal barrier: drain cannot advance until the waiting writer got in.
            assert writer_finished.wait(0.25), "drain kept the writer lock across events"
            assert observed, "second writer could not BEGIN IMMEDIATE within 250ms"
        result = original(conn, event)
        if calls == 1:
            time.sleep(0.26)  # consume the 250ms event budget before the second writer requests admission
            first_written.set()
        return result

    def other_writer():
        conn = apsw.Connection(str(db))
        conn.setbusytimeout(250)
        try:
            assert first_written.wait(2)
            conn.execute("BEGIN IMMEDIATE")
            observed.append((spool.exists(), conn.execute("SELECT seen_count FROM chunks").fetchone()[0]))
            conn.execute("COMMIT")
        except BaseException as exc:
            errors.append(exc)
        finally:
            conn.close()
            writer_finished.set()

    monkeypatch.setattr(drain, "_apply_event", apply)
    writer = threading.Thread(target=other_writer)
    writer.start()
    try:
        assert drain.drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log") == 1
    finally:
        writer.join(3)
    assert not writer.is_alive()
    assert not errors
    assert observed == [(True, 1)], "writer must acquire while two events remain queued"


@pytest.mark.parametrize("crash_before_cleanup", [False, True])
def test_drain_crash_between_transactions_has_no_loss_or_duplicate(tmp_path, monkeypatch, crash_before_cleanup):
    db, queue, spool = _queued_events(tmp_path, monkeypatch)
    original = drain._apply_event
    original_spool = (spool.read_bytes(), spool.stat().st_mtime_ns)

    def slow_apply(conn, event):
        result = original(conn, event)
        time.sleep(0.02)
        return result

    monkeypatch.setattr(drain, "_apply_event", slow_apply)
    original_checkpoint = drain._post_commit_checkpoint
    if crash_before_cleanup:

        def crash(*_args):
            raise SystemExit("crash after COMMIT, before queue cleanup")

        monkeypatch.setattr(drain, "_post_commit_checkpoint", crash)
        with pytest.raises(SystemExit):
            drain.drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log")
    else:
        assert drain.drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log") == 1
    assert (spool.read_bytes(), spool.stat().st_mtime_ns) == original_spool
    assert drain.burn_drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log").failed_files == 1
    monkeypatch.setattr(drain, "_post_commit_checkpoint", original_checkpoint)
    # Reopen connections on every cycle, as after a process restart.
    for _ in range(4):
        drain.drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log")
    assert not spool.exists()
    with apsw.Connection(str(db)) as conn:
        assert conn.execute("SELECT COUNT(*), MAX(seen_count) FROM chunks").fetchone() == (1, 3)
        assert conn.execute("SELECT COUNT(*) FROM watcher_liveness_events").fetchone() == (3,)
        assert conn.execute("SELECT COUNT(*) FROM drain_event_receipts").fetchone() == (0,)


@pytest.mark.parametrize("kind", ["store_memory", "watcher_chunk", "hook_chunk"])
def test_drain_prepares_chunk_values_without_writer_lock(tmp_path, monkeypatch, kind):
    db, queue, spool = _queued_events(tmp_path, monkeypatch)
    payload = {"kind": kind, "content": "Prepare this memory before taking the writer lock"}
    payload.update(chunk_id="prepared", session_id="session")
    spool.write_text(json.dumps(payload) + "\n")
    original = drain._content_hash
    checked = []

    def hash_without_lock(content):
        with closing(apsw.Connection(str(db))) as other:
            other.setbusytimeout(0)
            other.execute("BEGIN IMMEDIATE")
            other.execute("ROLLBACK")
        checked.append(content)
        return original(content)

    monkeypatch.setattr(drain, "_content_hash", hash_without_lock)
    assert drain.drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log") == 1
    assert checked
    assert not spool.exists()
