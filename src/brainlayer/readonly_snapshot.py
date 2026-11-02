"""Bounded snapshots on a dedicated reader; never acquire BrainBar's writer gate."""

import ctypes
import math
import os
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import apsw
import sqlite_vec

from brainlayer.disk_space import measure_disk_space


@dataclass(frozen=True)
class SnapshotResult:
    duration_s: float
    bytes_written: int
    peak_wal_bytes: int
    reason: str | None = None
    detail: str = ""


def _identity(path):
    s = os.stat(path, follow_symlinks=False)
    return s.st_dev, s.st_ino


def _is_owned(path, identity):
    try:
        return _identity(path) == identity
    except FileNotFoundError:
        return False


def _open_reader(path):
    conn = apsw.Connection(path.as_uri() + "?mode=ro", flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI)
    try:
        conn.enable_load_extension(True)
        conn.load_extension(sqlite_vec.loadable_path())
        conn.enable_load_extension(False)
        return conn
    except BaseException:
        conn.close()
        raise


def _source_matches(conn, path, identity):
    moved = ctypes.c_int()
    # SQLite compares the actual open file with its pathname, closing the stat/open race.
    supported = conn.file_control("main", apsw.SQLITE_FCNTL_HAS_MOVED, ctypes.addressof(moved))
    try:
        return supported and not moved.value and _identity(path) == identity and not path.is_symlink()
    except FileNotFoundError:
        return False


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class _Abort(Exception):
    pass


def take_snapshot(db_path, target_path, *, deadline_s, wal_budget_bytes, min_free_bytes) -> SnapshotResult:
    """Publish exclusively; close the reader before returning any outcome.
    deadline_s bounds elapsed time; progress/interrupt handlers cancel SQLite work.
    WAL budget bounds absolute size, sampled every 50 ms; free space excludes purgeable.
    Cancellation must wait for a blocked filesystem syscall to return.
    """
    if not math.isfinite(deadline_s) or deadline_s < 0 or wal_budget_bytes < 0 or min_free_bytes < 0:
        raise ValueError("snapshot budgets must be finite and nonnegative")
    started = time.monotonic()
    source, target = Path(db_path).absolute(), Path(target_path).absolute()
    stop = threading.Event()
    conn = monitor = attempt = staging = owned = None
    fd, written, peak = -1, 0, 0
    reason, detail, published = None, "", False

    def cancel(why):
        nonlocal reason
        if reason is None:
            reason = why

    def progress():
        if time.monotonic() - started >= deadline_s:
            cancel("deadline")
        return reason is not None

    def sample():
        nonlocal peak
        try:
            size = Path(str(source) + "-wal").stat().st_size
        except FileNotFoundError:
            size = 0
        peak = max(peak, size)
        if size > wal_budget_bytes:
            cancel("wal_budget")
        for path, why in ((source.parent, "source_free_space"), (target.parent, "target_free_space")):
            if measure_disk_space(path, include_available=False).free_now_bytes < min_free_bytes:
                cancel(why)
        progress()

    def watch():
        nonlocal detail
        while not stop.wait(0.05):
            try:
                sample()
            except OSError as exc:
                detail = str(exc)
                cancel("monitor_error")
            if reason:
                conn.interrupt()
                return

    try:
        if os.path.lexists(target):
            cancel("target_exists")
            raise _Abort
        identity = _identity(source)
        conn = _open_reader(source)
        if not _source_matches(conn, source, identity):
            cancel("source_identity")
            raise _Abort
        sample()
        if reason:
            raise _Abort
        staging = Path(tempfile.mkdtemp(prefix=".readonly-snapshot-", dir=target.parent))
        attempt = staging / "snapshot.db"
        fd = os.open(attempt, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        owned = _identity(attempt)
        conn.set_progress_handler(progress, 1000)
        monitor = threading.Thread(target=watch, name="snapshot-budget", daemon=True)
        monitor.start()
        conn.execute("VACUUM INTO ?", (str(attempt),))
        sample()
        if not _source_matches(conn, source, identity):
            cancel("source_identity")
        if reason:
            raise _Abort
        written = os.fstat(fd).st_size
        if _identity(attempt) != owned or not stat.S_ISREG(os.lstat(attempt).st_mode) or written == 0:
            cancel("output_identity")
            raise _Abort
        os.fsync(fd)
        _sync_directory(staging)
        _sync_directory(target.parent)
        sample()
        if not _source_matches(conn, source, identity):
            cancel("source_identity")
        if _identity(attempt) != owned:
            cancel("output_identity")
        if reason:
            raise _Abort
        os.link(attempt, target, follow_symlinks=False)
        published = True
        _sync_directory(target.parent)
    except _Abort:
        pass
    except FileExistsError as exc:
        cancel("target_exists")
        detail = str(exc)
    except (OSError, apsw.Error) as exc:
        cancel("io_error" if isinstance(exc, OSError) else "sqlite_error")
        detail = str(exc)
    finally:
        stop.set()
        if monitor is not None:
            monitor.join()
        if conn is not None:
            conn.close()
        if fd >= 0:
            written = os.fstat(fd).st_size
            os.close(fd)
        try:
            if attempt is not None and _is_owned(attempt, owned):
                attempt.unlink()
            if staging is not None:
                staging.rmdir()
                _sync_directory(target.parent)
        except OSError as exc:
            cancel("cleanup_error")
            detail += f"; cleanup retained unconfirmed files: {exc}"
        if reason and published and _is_owned(target, owned):
            target.unlink()
            _sync_directory(target.parent)
    return SnapshotResult(time.monotonic() - started, written, peak, reason, detail)


def snapshot_counts(db_path, *, quick_check=False):
    """Count in one transaction with optional quick_check; VACUUM cannot share it.
    Compare stable tables only; changing tables need paired commit markers, never cross-time equality.
    """
    conn = _open_reader(Path(db_path).absolute())
    try:
        conn.execute("BEGIN")
        if quick_check and list(conn.execute("PRAGMA quick_check")) != [("ok",)]:
            raise ValueError("snapshot quick_check failed")
        return {
            table: conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
            for table in ("chunks", "chunk_vectors", "chunks_fts")
        }
    finally:
        conn.close()
