import json
import sqlite3
import threading
from pathlib import Path

import pytest

from brainlayer import backup_run_once


def _source_db(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, content TEXT)")
        db.execute("INSERT INTO chunks(content) VALUES ('synthetic fixture')")
        db.execute("CREATE TABLE padding (payload BLOB)")
        db.executemany("INSERT INTO padding(payload) VALUES(?)", ((b"x" * 3000,) for _ in range(1500)))


def _transcript_root(path: Path) -> Path:
    root = path / "transcripts"
    root.mkdir()
    (root / "synthetic.jsonl").write_text('{"synthetic":true}\n')
    return root


def _hold_exclusive(path: Path, ready: threading.Event, release: threading.Event) -> None:
    connection = sqlite3.connect(path, timeout=1)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        ready.set()
        release.wait(timeout=5)
        connection.rollback()
    finally:
        connection.close()


def _start_exclusive_holder(path: Path) -> tuple[threading.Event, threading.Thread]:
    ready = threading.Event()
    release = threading.Event()
    thread = threading.Thread(target=_hold_exclusive, args=(path, ready, release), daemon=True)
    thread.start()
    assert ready.wait(timeout=1)
    return release, thread


def test_offline_snapshot_succeeds_when_real_writer_lock_releases(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _source_db(source)
    release, holder = _start_exclusive_holder(source)
    timer = threading.Timer(0.1, release.set)
    timer.start()
    try:
        chunks = backup_run_once._offline_snapshot(
            source,
            target,
            total_deadline_seconds=2,
            stalled_progress_seconds=0.5,
        )
    finally:
        release.set()
        timer.cancel()
        holder.join(timeout=1)
    assert not holder.is_alive()
    assert chunks == 1
    with sqlite3.connect(f"file:{target}?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_run_once_real_writer_lock_deadline_records_compact_failed_receipt(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    _source_db(source)
    destination = tmp_path / "backups"
    destination.mkdir()
    root = _transcript_root(tmp_path)
    release, holder = _start_exclusive_holder(source)
    original = backup_run_once._offline_snapshot

    def short_deadline(source_path: Path, target_path: Path) -> int:
        return original(
            source_path,
            target_path,
            total_deadline_seconds=0.3,
            stalled_progress_seconds=0.1,
        )

    monkeypatch.setattr(backup_run_once, "_offline_snapshot", short_deadline)
    try:
        with pytest.raises(backup_run_once.BackupRunOnceError, match="offline SQLite backup stalled"):
            backup_run_once.run_backup_once(
                destination=destination,
                transcript_roots=[root],
                offline_db_path=source,
                expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
                generation_id="writer-lock",
            )
    finally:
        release.set()
        holder.join(timeout=1)
    assert not holder.is_alive()

    generation = next(destination.glob(".brainlayer-backup-*-writer-lock"))
    receipt = json.loads((generation / "receipt.incomplete.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["error_type"] == "BackupRunOnceError"
    assert "status=SQLITE_BUSY" in receipt["error"]
    assert "busy=" in receipt["error"]
    assert "restarts=0" in receipt["error"]
    assert len(receipt["error"]) < 300
    assert not (generation / "receipt.complete.json").exists()
    with sqlite3.connect(generation / "brainlayer.db", timeout=0) as target_connection:
        target_connection.execute("BEGIN EXCLUSIVE")
        target_connection.rollback()


def test_offline_snapshot_source_restarts_still_hit_total_deadline_and_close_connections(tmp_path, monkeypatch):
    class FakeConnection:
        def __init__(self, source: bool) -> None:
            self.source = source
            self.closed = False

        def backup(self, _target, *, pages, sleep, progress):
            assert pages == 1000 and sleep == 0.01
            progress(sqlite3.SQLITE_OK, 5000, 6000)
            progress(sqlite3.SQLITE_OK, 4000, 6000)
            progress(sqlite3.SQLITE_OK, 5000, 6000)
            progress(sqlite3.SQLITE_OK, 4000, 6000)

        def close(self):
            self.closed = True

    source_connection = FakeConnection(source=True)
    target_connection = FakeConnection(source=False)
    connections = iter((source_connection, target_connection))
    monkeypatch.setattr(backup_run_once.sqlite3, "connect", lambda *args, **kwargs: next(connections))
    clock = iter((0.0, 1.0, 2.0, 3.0, 4.0))

    with pytest.raises(backup_run_once.BackupRunOnceError) as error:
        backup_run_once._offline_snapshot(
            tmp_path / "source.db",
            tmp_path / "target.db",
            total_deadline_seconds=4,
            stalled_progress_seconds=3,
            monotonic=lambda: next(clock),
        )

    message = str(error.value)
    assert "total deadline exceeded" in message
    assert "status=SQLITE_OK" in message
    assert "restarts=1" in message
    assert "busy=0 locked=0" in message
    assert source_connection.closed and target_connection.closed
