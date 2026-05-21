from __future__ import annotations

import hashlib
import os
import plistlib
import subprocess
import sys
import threading
from pathlib import Path

import apsw
import pytest

from brainlayer import drain
from brainlayer.vector_store import VectorStore


def _expected_pidfile(pidfile_dir: Path, db_path: Path) -> Path:
    resolved_path = db_path.resolve()
    path_hash = hashlib.sha256(str(resolved_path).encode("utf-8")).hexdigest()[:16]
    return pidfile_dir / f"brainlayer-writer-{path_hash}-{resolved_path.name}.pid"


def test_pidfile_created_on_rw_init(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    store = VectorStore(db_path)
    try:
        pidfile = _expected_pidfile(pidfile_dir, db_path)
        assert pidfile.exists()
        assert pidfile.read_text(encoding="utf-8").strip() == str(os.getpid())
    finally:
        store.close()


def test_pidfile_blocks_second_writer(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    store = VectorStore(db_path)
    try:
        script = """
from pathlib import Path
from brainlayer.vector_store import VectorStore, WriterInUseError

try:
    VectorStore(Path(__import__("os").environ["DB_PATH"]))
except WriterInUseError as exc:
    print(type(exc).__name__)
    print(str(exc))
    raise SystemExit(0)
raise SystemExit("second writer unexpectedly acquired the pidfile")
"""
        env = {
            **os.environ,
            "BRAINLAYER_WRITER_PIDFILE_DIR": str(pidfile_dir),
            "DB_PATH": str(db_path),
            "PYTHONPATH": f"{Path(__file__).resolve().parents[1] / 'src'}:{os.environ.get('PYTHONPATH', '')}",
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "WriterInUseError" in result.stdout
        assert f"another writer is using {db_path} (pid {os.getpid()})" in result.stdout
    finally:
        store.close()


def test_pidfile_allows_different_directories_with_same_db_basename(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    first_db = tmp_path / "first" / "writer.db"
    second_db = tmp_path / "second" / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    store = VectorStore(first_db)
    try:
        script = """
from pathlib import Path
from brainlayer.vector_store import VectorStore

store = VectorStore(Path(__import__("os").environ["DB_PATH"]))
store.close()
"""
        env = {
            **os.environ,
            "BRAINLAYER_WRITER_PIDFILE_DIR": str(pidfile_dir),
            "DB_PATH": str(second_db),
            "PYTHONPATH": f"{Path(__file__).resolve().parents[1] / 'src'}:{os.environ.get('PYTHONPATH', '')}",
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
    finally:
        store.close()


def test_pidfile_blocks_symlink_alias_to_same_database(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    real_db = tmp_path / "real" / "writer.db"
    symlink_db = tmp_path / "alias.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    store = VectorStore(real_db)
    try:
        symlink_db.symlink_to(real_db)
        script = """
from pathlib import Path
from brainlayer.vector_store import VectorStore, WriterInUseError

try:
    VectorStore(Path(__import__("os").environ["DB_PATH"]))
except WriterInUseError as exc:
    print(type(exc).__name__)
    print(str(exc))
    raise SystemExit(0)
raise SystemExit("symlink writer unexpectedly acquired the pidfile")
"""
        env = {
            **os.environ,
            "BRAINLAYER_WRITER_PIDFILE_DIR": str(pidfile_dir),
            "DB_PATH": str(symlink_db),
            "PYTHONPATH": f"{Path(__file__).resolve().parents[1] / 'src'}:{os.environ.get('PYTHONPATH', '')}",
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "WriterInUseError" in result.stdout
    finally:
        store.close()


def test_same_process_pidfile_reuse_registers_atexit_release(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    registrations = []
    monkeypatch.setattr("atexit.register", registrations.append)

    first = VectorStore(db_path)
    second = VectorStore(db_path)
    try:
        assert len(registrations) == 2
    finally:
        second.close()
        first.close()


def test_release_keeps_pidfile_ref_until_unlink_finishes(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    first = VectorStore(db_path)
    pidfile = _expected_pidfile(pidfile_dir, db_path)
    release_started = threading.Event()
    opener_done = threading.Event()
    opened: list[VectorStore] = []
    errors: list[BaseException] = []
    original_read = VectorStore._read_writer_pidfile

    def delayed_pidfile_read(path: Path) -> int | None:
        result = original_read(path)
        if path == pidfile and not release_started.is_set():
            release_started.set()
            opener_done.wait(0.5)
        return result

    def open_second_writer() -> None:
        release_started.wait(5)
        try:
            opened.append(VectorStore(db_path))
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            opener_done.set()

    monkeypatch.setattr(VectorStore, "_read_writer_pidfile", staticmethod(delayed_pidfile_read))
    opener_thread = threading.Thread(target=open_second_writer)
    release_thread = threading.Thread(target=first.close)

    opener_thread.start()
    release_thread.start()
    release_thread.join(5)
    opener_thread.join(5)

    assert not release_thread.is_alive()
    assert not opener_thread.is_alive()
    assert not errors
    assert opened
    try:
        assert pidfile.exists()
        assert pidfile.read_text(encoding="utf-8").strip() == str(os.getpid())
    finally:
        opened[0].close()


def test_stale_pidfile_cleaned_up(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    pidfile_dir.mkdir()
    pidfile = _expected_pidfile(pidfile_dir, db_path)
    pidfile.write_text("999999", encoding="utf-8")

    store = VectorStore(db_path)
    try:
        assert pidfile.exists()
        assert pidfile.read_text(encoding="utf-8").strip() == str(os.getpid())
    finally:
        store.close()


def test_stale_pidfile_unlink_missing_race_is_ignored(tmp_path, monkeypatch):
    pidfile = tmp_path / "writer.pid"
    pidfile.write_text("999999", encoding="utf-8")
    store = object.__new__(VectorStore)
    store.db_path = tmp_path / "writer.db"
    monkeypatch.setattr(store, "_pid_is_alive", lambda _pid: False)

    original_unlink = Path.unlink

    def unlink_race(path: Path, *args, **kwargs):
        if path == pidfile:
            raise FileNotFoundError
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink_race)

    assert store._handle_existing_writer_pidfile(pidfile, os.getpid()) is False


def test_pidfile_removed_on_close(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    store = VectorStore(db_path)
    pidfile = _expected_pidfile(pidfile_dir, db_path)
    assert pidfile.exists()

    store.close()

    assert not pidfile.exists()


def test_readonly_init_skips_pidfile(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    writer = VectorStore(db_path)
    writer.close()

    reader = VectorStore(db_path, readonly=True)
    try:
        pidfile = _expected_pidfile(pidfile_dir, db_path)
        assert not pidfile.exists()
        assert reader._writer_pidfile_acquired is False
        assert not hasattr(reader, "_writer_pidfile_path_value")
    finally:
        reader.close()


def test_init_retries_10_with_extended_backoff():
    assert VectorStore._INIT_MAX_RETRIES == 10
    delays = [
        min(VectorStore._INIT_BASE_DELAY * (2**attempt), VectorStore._INIT_MAX_DELAY)
        for attempt in range(VectorStore._INIT_MAX_RETRIES)
    ]
    assert delays == [0.5, 1, 2, 4, 8, 16, 30, 30, 30, 30]
    assert sum(delays) <= 600


def test_init_retry_zero_budget_reraises_original_busy(monkeypatch):
    store = object.__new__(VectorStore)
    store._INIT_MAX_RETRIES = 0
    store._INIT_BASE_DELAY = 0
    store._INIT_MAX_DELAY = 0
    calls = 0

    def busy_init():
        nonlocal calls
        calls += 1
        raise apsw.BusyError("locked")

    monkeypatch.setattr(store, "_init_db", busy_init)
    monkeypatch.setattr("time.sleep", lambda _delay: None)

    with pytest.raises(apsw.BusyError, match="locked"):
        store._init_db_with_retry()

    assert calls == 1


def test_drain_busy_timeout_30s(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_DRAIN_BUSY_TIMEOUT_MS", "30000")

    conn = drain._open_connection(tmp_path / "drain.db")
    try:
        busy_timeout_ms = conn.cursor().execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy_timeout_ms >= 30000
    finally:
        conn.close()


def test_drain_busy_timeout_invalid_env_falls_back_to_30s(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_DRAIN_BUSY_TIMEOUT_MS", "not-an-int")

    conn = drain._open_connection(tmp_path / "drain.db")
    try:
        busy_timeout_ms = conn.cursor().execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy_timeout_ms >= 30000
    finally:
        conn.close()


def test_enrichment_plist_throttle_interval_at_least_60s():
    plist_path = Path(__file__).resolve().parents[1] / "scripts" / "launchd" / "com.brainlayer.enrichment.plist"
    plist = plistlib.loads(plist_path.read_bytes())

    assert plist["ThrottleInterval"] >= 60
