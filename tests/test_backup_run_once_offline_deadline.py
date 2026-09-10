import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
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


def test_run_once_real_writer_lock_records_snapshot_acquisition_failure(tmp_path, monkeypatch):
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
        with pytest.raises(backup_run_once.BackupRunOnceError, match="read snapshot acquisition failed"):
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
    assert "read snapshot acquisition failed" in receipt["error"]
    assert "backup stalled" not in receipt["error"]
    assert len(receipt["error"]) < 300
    assert not (generation / "receipt.complete.json").exists()
    assert not (generation / "brainlayer.db").exists()
    with sqlite3.connect(source, timeout=0) as source_connection:
        source_connection.execute("BEGIN EXCLUSIVE")
        source_connection.rollback()


def test_offline_snapshot_source_restarts_still_hit_total_deadline_and_close_connections(tmp_path, monkeypatch):
    class FakeConnection:
        def __init__(self, source: bool) -> None:
            self.source = source
            self.closed = False

        def execute(self, _sql):
            return self

        def fetchone(self):
            return (2,)

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _readonly_wal_source(root: Path) -> tuple[Path, dict[str, str]]:
    source_dir = root / "readonly-wal"
    source_dir.mkdir()
    source = source_dir / "source.db"
    connection = sqlite3.connect(source)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        connection.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, content TEXT)")
        connection.execute("INSERT INTO chunks(content) VALUES ('synthetic fixture')")
        connection.execute("CREATE TABLE padding(payload BLOB)")
        connection.executemany("INSERT INTO padding(payload) VALUES(?)", ((b"x" * 3000,) for _ in range(3000)))
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert connection.execute("PRAGMA page_count").fetchone()[0] > 2000
        wal = Path(str(source) + "-wal")
        shm = Path(str(source) + "-shm")
        wal_bytes = wal.read_bytes()
        shm_bytes = shm.read_bytes()
    finally:
        connection.close()
    wal.write_bytes(wal_bytes)
    shm.write_bytes(shm_bytes)
    files = (source, wal, shm)
    for item in files:
        item.chmod(0o444)
    source_dir.chmod(0o555)
    return source, {item.name: _sha256(item) for item in files}


def test_offline_snapshot_pins_readonly_wal_source_before_backup_steps(tmp_path):
    source, before = _readonly_wal_source(tmp_path)
    source_dir = source.parent
    old_target = tmp_path / "old-unpinned.db"
    progress_path = tmp_path / "old-progress.json"
    old_code = r"""
import json, os, sqlite3, sys
from pathlib import Path
source, target, progress_path = map(Path, sys.argv[1:])
state = {"calls": 0, "first_remaining": None, "last_remaining": None, "statuses": {}}
def progress(status, remaining, total):
    state["calls"] += 1
    state["first_remaining"] = remaining if state["first_remaining"] is None else state["first_remaining"]
    state["last_remaining"] = remaining
    name = {0: "SQLITE_OK", 5: "SQLITE_BUSY", 6: "SQLITE_LOCKED", 101: "SQLITE_DONE"}.get(status, str(status))
    state["statuses"][name] = state["statuses"].get(name, 0) + 1
    if state["calls"] % 100 == 0:
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
        os.replace(temporary, progress_path)
source_db = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=0.05)
target_db = sqlite3.connect(target, timeout=0.05)
source_db.backup(target_db, pages=1000, sleep=0.01, progress=progress)
"""
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", old_code, str(source), str(old_target), str(progress_path)],
        start_new_session=True,
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=1)
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1)
        old_progress = json.loads(progress_path.read_text())
        assert process.returncode == -signal.SIGTERM
        assert old_progress["calls"] >= 100
        assert old_progress["statuses"] == {"SQLITE_OK": old_progress["calls"]}
        assert old_progress["first_remaining"] == old_progress["last_remaining"]

        target = tmp_path / "pinned.db"
        assert (
            backup_run_once._offline_snapshot(
                source,
                target,
                total_deadline_seconds=2,
                stalled_progress_seconds=0.5,
            )
            == 1
        )
        with sqlite3.connect(f"file:{target}?mode=ro", uri=True) as target_connection:
            assert target_connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        after = {item.name: _sha256(item) for item in (source, Path(str(source) + "-wal"), Path(str(source) + "-shm"))}
        assert after == before
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1)
        source_dir.chmod(0o755)
