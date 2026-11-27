"""Bounded APFS file-pair snapshot under SQLite's WAL writer barrier.

Only SQLite opens the source in this process. cp runs in a DIFFERENT process:
closing a raw source descriptor here could release SQLite's POSIX locks.
"""

from __future__ import annotations

import math
import os
import plistlib
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WAL_GROWTH_SLACK_BYTES = 64 * 1024**2
# M4 read-only 180 s sample (2026-11-22): 1,733,624 B/s mean logical/disk
# process-write increments; round up to 2 MiB/s for the planning allowance.
DEFAULT_WRITE_BYTES_PER_SECOND = 2 * 1024**2


class CloneUnavailable(RuntimeError):
    """Stop this clone attempt; clean owned files before a capacity-checked fallback."""


class CloneIntegrityError(CloneUnavailable):
    """Private SQLite recovery found corruption; the caller owns its diagnostic alert."""


def divergence_allowance_bytes() -> int:
    from .backup_heavy_gate import _number

    rate = float(os.environ.get("BRAINLAYER_BACKUP_CLONE_WRITE_BYTES_PER_SECOND", DEFAULT_WRITE_BYTES_PER_SECOND))
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("BRAINLAYER_BACKUP_CLONE_WRITE_BYTES_PER_SECOND must be finite and nonnegative")
    return math.ceil(rate * _number("EXPECTED_SNAPSHOT_SECONDS", 2400))


def budget(source: Path) -> dict[str, int]:
    wal = Path(f"{source}-wal")
    return {
        "clone_wal_bytes": (wal.stat().st_size if wal.exists() else 0) + WAL_GROWTH_SLACK_BYTES,
        "clone_divergence_bytes": divergence_allowance_bytes(),
    }


def host_reserve_bytes() -> int:
    from .backup_heavy_gate import _host

    reserve = float(os.environ.get("BRAINLAYER_BACKUP_HEAVY_HOST_FREE_GIB", 40 if _host() == "m4" else 0))
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("BRAINLAYER_BACKUP_HEAVY_HOST_FREE_GIB must be finite and nonnegative")
    return math.ceil(reserve * 1024**3)


class SpaceGuard:
    """Protect an absolute reserve; volume deltas are observations, never a cutoff."""

    def __init__(self, directory: Path, archive: Path, *, minimum_free: int):
        from .disk_space import measure_disk_space

        self.directory, self.archive, self.minimum_free = directory, archive, minimum_free
        self.initial_free = measure_disk_space(directory, include_available=False).free_now_bytes
        self.peak_bytes = 0
        self.last_check = float("-inf")
        self(force=True)

    def __call__(self, *, force=False):
        from .disk_space import measure_disk_space

        now = time.monotonic()
        if not force and now - self.last_check < 0.25:
            return
        self.last_check = now
        free = measure_disk_space(self.directory, include_available=False).free_now_bytes
        used = max(0, self.initial_free - free)
        self.peak_bytes = max(self.peak_bytes, used)
        if free < self.minimum_free:
            raise CloneUnavailable(f"clone host free-space reserve breached: has {free}, needs {self.minimum_free}")


def supported(source: Path, directory: Path) -> bool:
    method = os.environ.get("BRAINLAYER_BACKUP_SNAPSHOT_METHOD", "auto")
    if method not in {"auto", "vacuum"}:
        raise ValueError("BRAINLAYER_BACKUP_SNAPSHOT_METHOD must be auto or vacuum")
    if method == "vacuum":
        return False
    if sys.platform != "darwin":
        return False
    try:
        directory = directory.resolve()
        while not directory.exists():
            directory = directory.parent
        if source.stat().st_dev != directory.stat().st_dev:
            return False
        mounted = (
            subprocess.run(["/bin/df", "-P", str(directory)], capture_output=True, check=True, timeout=5)
            .stdout.decode()
            .splitlines()
        )
        device = mounted[1].split()[0] if len(mounted) == 2 and mounted[1].split() else ""
        if not device.startswith("/dev/"):
            return False
        result = subprocess.run(
            ["/usr/sbin/diskutil", "info", "-plist", device],
            capture_output=True,
            check=True,
            timeout=5,
        )
        metadata = plistlib.loads(result.stdout)
        return isinstance(metadata, dict) and metadata.get("FilesystemType") == "apfs"
    except TimeoutError:
        raise
    except (OSError, ValueError, subprocess.SubprocessError, plistlib.InvalidFileException):
        return False


def _identity(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _copy(source: Path, destination: Path) -> None:
    subprocess.run(["/bin/cp", "-c", str(source), str(destination)], check=True, capture_output=True, timeout=5)


def snapshot(source: Path, target: Path, *, validate=None, max_recovery_bytes: int | None = None) -> os.stat_result:
    """Publish a standalone clone, or nothing. Never checkpoint/change the SOURCE.

    BEGIN IMMEDIATE prevents commits and WAL restart throughout both clonefile
    calls. PASSIVE checkpoints may still copy stable WAL frames into the source
    main file; the frozen WAL overlays those pages during recovery on our clone.
    Full/RESTART/TRUNCATE checkpoints require the writer lock we hold.
    No SHM is copied: SQLite rebuilds it for the private inode pair.
    """
    source, target = source.resolve(), target.absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if not supported(source, target.parent):
        raise CloneUnavailable("source/staging are not on one APFS volume")
    initial = _identity(source)
    wal = Path(f"{source}-wal")
    try:
        with tempfile.TemporaryDirectory(prefix=".brainlayer-clone-", dir=target.parent) as temporary:
            private = Path(temporary) / "snapshot.db"
            (Path(temporary) / ".owner").write_text(str(os.getpid()), encoding="ascii")
            # mode=rw refuses a missing source instead of creating an empty database.
            connection = sqlite3.connect(f"{source.as_uri()}?mode=rw", uri=True, timeout=0.25, isolation_level=None)
            try:
                if not hasattr(connection, "setconfig") or not hasattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE"):
                    raise CloneUnavailable("SQLite cannot disable source checkpoint on close")
                connection.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)
                if connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                    raise CloneUnavailable("source is not in WAL mode")
                connection.execute("BEGIN IMMEDIATE")
                if _identity(source)[:2] != initial[:2]:
                    raise CloneUnavailable("source inode changed before writer admission")
                wal_before = _identity(wal) if wal.exists() else None
                if wal_before is not None and max_recovery_bytes is not None and wal_before[2] > max_recovery_bytes:
                    raise CloneUnavailable("WAL outgrew admitted clone recovery budget")
                _copy(source, private)
                if wal_before is not None:
                    _copy(wal, Path(f"{private}-wal"))
                wal_after = _identity(wal) if wal.exists() else None
                if _identity(source)[:2] != initial[:2] or wal_after != wal_before:
                    raise CloneUnavailable("source DB/WAL identity changed during clone")
            finally:
                connection.rollback()
                connection.close()
            # Replay only the PRIVATE WAL into the PRIVATE inode. Its bounded
            # WAL is the maximum page divergence this recovery can allocate.
            recovered = sqlite3.connect(private)
            try:
                result = recovered.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if result != (0, 0, 0):
                    raise CloneUnavailable(f"private WAL recovery incomplete: {result}")
                if recovered.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
                    raise CloneUnavailable("private clone did not become standalone")
            except sqlite3.DatabaseError as exc:
                code = getattr(exc, "sqlite_errorcode", None)
                # Extended result codes retain the primary code in the low byte.
                corrupt = (
                    code & 0xFF in {sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB}
                    if code is not None
                    else getattr(exc, "sqlite_errorname", None) in {"SQLITE_CORRUPT", "SQLITE_NOTADB"}
                )
                if not corrupt:
                    raise  # Ordinary recovery failures use the outer unavailable/fallback path.
                raise CloneIntegrityError(f"private clone recovery failed: {type(exc).__name__}: {exc}") from exc
            finally:
                recovered.close()
            if validate is not None:
                validate(private)
            # Create-exclusive: a peer's dated snapshot can never be replaced.
            owned = private.stat()
            try:
                os.link(private, target)
                return owned
            except BaseException:
                # A stop between link and return must not leave a pinning clone.
                if target.exists() and _identity(target)[:2] == (owned.st_dev, owned.st_ino):
                    target.unlink()
                raise
    except FileExistsError:
        raise
    except TimeoutError:
        raise
    except (OSError, sqlite3.Error, subprocess.SubprocessError) as exc:
        raise CloneUnavailable(f"APFS clone unavailable: {type(exc).__name__}: {exc}") from exc
