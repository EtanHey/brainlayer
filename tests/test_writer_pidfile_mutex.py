from __future__ import annotations

import fcntl
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
from brainlayer.vector_store import VectorStore, WriterInUseError


def _expected_pidfile(pidfile_dir: Path, db_path: Path) -> Path:
    resolved_path = db_path.resolve()
    path_hash = hashlib.sha256(str(resolved_path).encode("utf-8")).hexdigest()[:16]
    return pidfile_dir / f"brainlayer-writer-{path_hash}-{resolved_path.name}.pid"


def _sibling_pidfile(pidfile_dir: Path, db_path: Path, suffix: str) -> Path:
    resolved_path = db_path.resolve()
    return pidfile_dir / f"brainlayer-writer-{suffix}-{resolved_path.name}.pid"


def _pidfile_pid(pidfile: Path) -> str:
    return pidfile.read_text(encoding="utf-8").splitlines()[0].strip()


def test_pidfile_created_on_rw_init(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    store = VectorStore(db_path)
    try:
        pidfile = _expected_pidfile(pidfile_dir, db_path)
        assert pidfile.exists()
        assert _pidfile_pid(pidfile) == str(os.getpid())
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


def test_inherited_pidfile_ref_must_match_current_pid(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    pidfile_dir.mkdir()
    pidfile = _expected_pidfile(pidfile_dir, db_path)
    pidfile.write_text("1\n", encoding="utf-8")
    store = object.__new__(VectorStore)
    store.db_path = db_path
    store._writer_pidfile_acquired = False

    with VectorStore._PIDFILE_REFS_LOCK:
        VectorStore._PIDFILE_REFS[pidfile] = 1
        VectorStore._PIDFILE_REF_PIDS[pidfile] = 1

    try:
        with pytest.raises(WriterInUseError, match="another writer is using"):
            store._acquire_writer_pidfile()
        assert not store._writer_pidfile_acquired
    finally:
        with VectorStore._PIDFILE_REFS_LOCK:
            VectorStore._PIDFILE_REFS.pop(pidfile, None)
            VectorStore._PIDFILE_REF_PIDS.pop(pidfile, None)


def test_pidfile_ref_mismatch_does_not_clear_existing_refs(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    pidfile_dir.mkdir()
    pidfile = _expected_pidfile(pidfile_dir, db_path)
    pidfile.write_text("999999\n", encoding="utf-8")
    store = object.__new__(VectorStore)
    store.db_path = db_path
    store._writer_pidfile_acquired = False

    with VectorStore._PIDFILE_REFS_LOCK:
        VectorStore._PIDFILE_REFS[pidfile] = 2
        VectorStore._PIDFILE_REF_PIDS[pidfile] = os.getpid()

    try:
        with pytest.raises(WriterInUseError, match="pidfile ref mismatch"):
            store._acquire_writer_pidfile()
        with VectorStore._PIDFILE_REFS_LOCK:
            assert VectorStore._PIDFILE_REFS[pidfile] == 2
        assert _pidfile_pid(pidfile) == "999999"
    finally:
        with VectorStore._PIDFILE_REFS_LOCK:
            VectorStore._PIDFILE_REFS.pop(pidfile, None)
            VectorStore._PIDFILE_REF_PIDS.pop(pidfile, None)


def test_same_process_reuse_accepts_pid_only_pidfile_when_owner_is_alive(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    monkeypatch.setattr(VectorStore, "_pid_start_time", staticmethod(lambda _pid: "current-process-start"))
    pidfile_dir.mkdir()
    pidfile = _expected_pidfile(pidfile_dir, db_path)
    pidfile.write_text(f"{os.getpid()}\n", encoding="utf-8")
    store = object.__new__(VectorStore)
    store.db_path = db_path
    store._writer_pidfile_acquired = False

    with VectorStore._PIDFILE_REFS_LOCK:
        VectorStore._PIDFILE_REFS[pidfile] = 1
        VectorStore._PIDFILE_REF_PIDS[pidfile] = os.getpid()

    try:
        store._acquire_writer_pidfile()
        assert store._writer_pidfile_acquired
        with VectorStore._PIDFILE_REFS_LOCK:
            assert VectorStore._PIDFILE_REFS[pidfile] == 2
    finally:
        store._release_writer_pidfile()
        with VectorStore._PIDFILE_REFS_LOCK:
            VectorStore._PIDFILE_REFS.pop(pidfile, None)
            VectorStore._PIDFILE_REF_PIDS.pop(pidfile, None)


def test_pidfile_reused_pid_requires_matching_start_time(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    monkeypatch.setattr(
        VectorStore,
        "_pid_start_time",
        staticmethod(lambda _pid: "current-process-start"),
        raising=False,
    )
    pidfile_dir.mkdir()
    pidfile = _expected_pidfile(pidfile_dir, db_path)
    pidfile.write_text(f"{os.getpid()}\nstart_time=previous-process-start\n", encoding="utf-8")

    store = VectorStore(db_path)
    try:
        contents = pidfile.read_text(encoding="utf-8")
        assert contents.startswith(f"{os.getpid()}\n")
        assert "start_time=current-process-start" in contents
    finally:
        store.close()


def test_relative_pidfile_dir_is_normalized_to_tmp(tmp_path, monkeypatch):
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", "brainlayer-relative-locks")
    store = object.__new__(VectorStore)
    store.db_path = db_path

    pidfile = store._writer_pidfile_path()

    assert pidfile.is_absolute()
    assert pidfile.parent == (Path("/tmp") / "brainlayer-relative-locks").resolve()


def test_pidfile_create_locks_before_writing_pid(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    events: list[str] = []
    original_flock = fcntl.flock
    original_write = os.write

    def record_flock(fd: int, operation: int) -> None:
        if operation == fcntl.LOCK_EX:
            events.append("flock")
        original_flock(fd, operation)

    def record_write(fd: int, data: bytes) -> int:
        events.append("write")
        return original_write(fd, data)

    monkeypatch.setattr("fcntl.flock", record_flock)
    monkeypatch.setattr("os.write", record_write)

    store = VectorStore(db_path)
    try:
        assert events[:2] == ["flock", "write"]
    finally:
        store.close()


def test_pidfile_acquire_retries_transient_file_exists_races(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    store = object.__new__(VectorStore)
    store.db_path = db_path
    store._writer_pidfile_acquired = False
    original_open = os.open
    create_attempts = 0

    def flaky_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal create_attempts
        if Path(path).name.startswith("brainlayer-writer-") and flags & os.O_CREAT:
            create_attempts += 1
            if create_attempts < 3:
                raise FileExistsError(17, "File exists", str(path))
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("os.open", flaky_open)
    monkeypatch.setattr(store, "_handle_existing_writer_pidfile", lambda _pidfile, _pid: False)

    store._acquire_writer_pidfile()

    try:
        assert store._writer_pidfile_acquired
        assert create_attempts == 3
    finally:
        store._release_writer_pidfile()


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
        assert _pidfile_pid(pidfile) == str(os.getpid())
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
        assert _pidfile_pid(pidfile) == str(os.getpid())
    finally:
        store.close()


def test_acquire_sweeps_all_stale_pidfiles_before_live_conflict(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    pidfile_dir.mkdir()
    live_pid = 424242
    live_pidfile = _expected_pidfile(pidfile_dir, db_path)
    live_pidfile.write_text(f"{live_pid}\n", encoding="utf-8")
    stale_pidfiles = [
        _sibling_pidfile(pidfile_dir, db_path, "stale-one"),
        _sibling_pidfile(pidfile_dir, db_path, "stale-two"),
        _sibling_pidfile(pidfile_dir, db_path, "stale-three"),
    ]
    for stale_pidfile in stale_pidfiles:
        stale_pidfile.write_text("999999\n", encoding="utf-8")

    monkeypatch.setattr(VectorStore, "_pid_is_alive", staticmethod(lambda pid: pid == live_pid))

    with pytest.raises(WriterInUseError, match=f"another writer is using {db_path} \\(pid {live_pid}\\)"):
        VectorStore(db_path)

    assert live_pidfile.exists()
    assert _pidfile_pid(live_pidfile) == str(live_pid)
    assert all(not stale_pidfile.exists() for stale_pidfile in stale_pidfiles)


def test_reclaim_skips_stale_pidfile_for_different_recorded_db_path(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    other_db_path = tmp_path / "other" / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    pidfile_dir.mkdir()
    other_pidfile = _sibling_pidfile(pidfile_dir, db_path, "other-db")
    other_pidfile.write_text(f"999999\ndb_path={other_db_path.resolve()}\n", encoding="utf-8")

    store = VectorStore(db_path)
    try:
        assert other_pidfile.exists()
    finally:
        store.close()


def test_reclaim_skips_malformed_recorded_db_path_without_blocking_open(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    pidfile_dir.mkdir()
    malformed_pidfile = _sibling_pidfile(pidfile_dir, db_path, "malformed-db-path")
    malformed_pidfile.write_text("999999\ndb_path=bad\x00path\n", encoding="utf-8")

    store = VectorStore(db_path)
    try:
        assert malformed_pidfile.exists()
    finally:
        store.close()


def test_reclaim_escapes_db_basename_glob_metacharacters(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer[abc].db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    pidfile_dir.mkdir()
    stale_pidfile = _sibling_pidfile(pidfile_dir, db_path, "stale")
    stale_pidfile.write_text("999999\n", encoding="utf-8")

    store = VectorStore(db_path)
    try:
        assert not stale_pidfile.exists()
    finally:
        store.close()


def test_reclaim_skips_unreadable_sibling_pidfile_without_blocking_open(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    pidfile_dir.mkdir()
    unreadable_pidfile = _sibling_pidfile(pidfile_dir, db_path, "unreadable")
    unreadable_pidfile.write_text("999999\n", encoding="utf-8")
    original_open = os.open

    def unreadable_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None and Path(path) == unreadable_pidfile and flags & os.O_ACCMODE == os.O_RDONLY:
            raise PermissionError(13, "Permission denied", str(path))
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("os.open", unreadable_open)

    store = VectorStore(db_path)
    try:
        assert unreadable_pidfile.exists()
    finally:
        store.close()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX fifos")
def test_reclaim_skips_fifo_sibling_pidfile_without_opening(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    pidfile_dir.mkdir()
    fifo_pidfile = _sibling_pidfile(pidfile_dir, db_path, "fifo")
    os.mkfifo(fifo_pidfile)
    original_open = os.open

    def no_fifo_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None and Path(path) == fifo_pidfile:
            raise AssertionError("fifo pidfile candidates must be skipped before open")
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("os.open", no_fifo_open)

    store = VectorStore(db_path)
    try:
        assert fifo_pidfile.exists()
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


def test_locked_stale_pidfile_is_retryable_not_active_writer(tmp_path, monkeypatch):
    pidfile = tmp_path / "writer.pid"
    pidfile.write_text("999999\nstart_time=stale-start\n", encoding="utf-8")
    store = object.__new__(VectorStore)
    store.db_path = tmp_path / "writer.db"
    monkeypatch.setattr(store, "_pid_is_alive", lambda _pid: False)

    def locked_flock(_fd: int, operation: int) -> None:
        if operation == (fcntl.LOCK_EX | fcntl.LOCK_NB):
            raise BlockingIOError

    monkeypatch.setattr(fcntl, "flock", locked_flock)

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


def test_pidfile_removed_by_atexit_without_explicit_close(tmp_path):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    pidfile = _expected_pidfile(pidfile_dir, db_path)
    script = """
from pathlib import Path
from brainlayer.vector_store import VectorStore

VectorStore(Path(__import__("os").environ["DB_PATH"]))
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
    assert not pidfile.exists()


def test_close_releases_pidfile_when_connection_close_raises(tmp_path, monkeypatch):
    pidfile_dir = tmp_path / "pidfiles"
    db_path = tmp_path / "writer.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))

    store = VectorStore(db_path)
    pidfile = _expected_pidfile(pidfile_dir, db_path)

    class BrokenConnection:
        def close(self):
            raise RuntimeError("close failed")

    store.conn = BrokenConnection()

    with pytest.raises(RuntimeError, match="close failed"):
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


def test_init_retries_10_with_extended_backoff(monkeypatch):
    monkeypatch.setattr(VectorStore, "_INIT_MAX_RETRIES", 10)
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


def test_drain_open_retries_busy_error_before_connection_exists(tmp_path, monkeypatch):
    attempts = 0
    sleep_calls: list[float] = []

    class FakeConnection:
        def __init__(self, path: str):
            self.path = path
            self.busy_timeout_ms = None
            self.extension_enabled: list[bool] = []
            self.loaded_extensions: list[str] = []

        def setbusytimeout(self, timeout_ms: int) -> None:
            self.busy_timeout_ms = timeout_ms

        def enableloadextension(self, enabled: bool) -> None:
            self.extension_enabled.append(enabled)

        def loadextension(self, path: str) -> None:
            self.loaded_extensions.append(path)

    def flaky_connection(path: str):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise apsw.BusyError("database is locked")
        return FakeConnection(path)

    monkeypatch.setenv("BRAINLAYER_DRAIN_OPEN_MAX_RETRIES", "4")
    monkeypatch.setenv("BRAINLAYER_DRAIN_OPEN_RETRY_BASE_DELAY_MS", "25")
    monkeypatch.setenv("BRAINLAYER_DRAIN_OPEN_RETRY_MAX_DELAY_MS", "100")
    monkeypatch.setattr(drain.apsw, "Connection", flaky_connection)
    monkeypatch.setattr(drain.sqlite_vec, "loadable_path", lambda: "sqlite_vec")
    monkeypatch.setattr(drain.time, "sleep", lambda delay: sleep_calls.append(delay))

    conn = drain._open_connection(tmp_path / "drain.db")

    assert attempts == 3
    assert sleep_calls == [0.025, 0.05]
    assert conn.busy_timeout_ms >= 30000
    assert conn.loaded_extensions == ["sqlite_vec"]
    assert conn.extension_enabled == [True, False]


def test_drain_busy_timeout_invalid_env_falls_back_to_30s(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_DRAIN_BUSY_TIMEOUT_MS", "not-an-int")

    conn = drain._open_connection(tmp_path / "drain.db")
    try:
        busy_timeout_ms = conn.cursor().execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy_timeout_ms >= 30000
    finally:
        conn.close()


def test_drain_busy_timeout_non_positive_env_falls_back_to_30s(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_DRAIN_BUSY_TIMEOUT_MS", "0")

    conn = drain._open_connection(tmp_path / "drain.db")
    try:
        busy_timeout_ms = conn.cursor().execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy_timeout_ms >= 30000
    finally:
        conn.close()


def test_drain_busy_timeout_overflow_env_falls_back_to_30s(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_DRAIN_BUSY_TIMEOUT_MS", "3000000000")

    conn = drain._open_connection(tmp_path / "drain.db")
    try:
        busy_timeout_ms = conn.cursor().execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy_timeout_ms == 30000
    finally:
        conn.close()


def test_enrichment_plist_throttle_interval_at_least_60s():
    plist_path = Path(__file__).resolve().parents[1] / "scripts" / "launchd" / "com.brainlayer.enrichment.plist"
    plist = plistlib.loads(plist_path.read_bytes())

    assert plist["ThrottleInterval"] >= 60
