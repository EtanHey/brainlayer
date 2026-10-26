"""Daily BrainLayer database backups.

The backup path intentionally uses SQLite's online backup API instead of copying
the database file directly, so live WAL writes are folded into a consistent
snapshot without stopping BrainBar or the enrichment jobs.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import functools
import gzip
import hashlib
import inspect
import json
import math
import os
import queue
import re
import shutil
import signal
import socket
import sqlite3
import ssl
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from . import drive_credentials
from .paths import get_db_path, resolve_db_path
from .socket_hygiene import refuse_production_brainbar_socket

_sleep = time.sleep

DEFAULT_TOKEN_PATH = drive_credentials.DEFAULT_TOKEN_PATH
DEFAULT_CLIENT_PATH = drive_credentials.DEFAULT_CLIENT_PATH
CANONICAL_MACHINE_ID = "MacBook-Pro"
CANONICAL_FOLDER_PARTS = ["Brain Drive", "06_ARCHIVE", "backups", "brainlayer-db"]
# Compatibility alias for callers that explicitly target the legacy M4 folder.
DEFAULT_FOLDER_PARTS = CANONICAL_FOLDER_PARTS
BACKUP_MACHINE_ID_ENV = "BRAINLAYER_MACHINE_ID"
DRIVE_MACHINE_PROPERTY = "brainlayer_machine"
# Verified prior IDs only (2026-10-17 lane-ops receipts). Never infer aliases from
# a router hostname or adopt another machine's legacy folder by prefix matching.
PRIOR_MACHINE_IDS = {"Locals-MacBook-Pro": ("Locals-MBP.lan",)}
DEFAULT_STAGING_DIR = Path.home() / ".local" / "share" / "brainlayer" / "backups"
MIN_RAW_FREE_BYTES = 2 * 1024 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
RECLAIM_CHUNK_BYTES = 64 * 1024 * 1024
FRAME_OVERHEAD_BYTES = 64 * 1024
POST_PURGE_POLL_SECONDS = 0.5  # 60 polls give APFS up to 30s to release reclaimed bytes.
DEFAULT_LOG_PATH = Path.home() / ".local" / "share" / "brainlayer" / "logs" / "backup-daily.log"
DEFAULT_BRAINBAR_SOCKET_PATH = "/tmp/brainbar.sock"
BACKUP_TIMEOUT_ENV = "BRAINLAYER_BACKUP_TIMEOUT_SECONDS"
BACKUP_CLIENT_TIMEOUT_ENV = "BRAINLAYER_BACKUP_CLIENT_TIMEOUT_SECONDS"
BACKUP_ATTEMPT_MAX_AGE_ENV = "BRAINLAYER_BACKUP_ATTEMPT_MAX_AGE_SECONDS"
BACKUP_FULL_VERIFY_ENV = "BRAINLAYER_BACKUP_FULL_VERIFY"
BACKUP_LOG_PATH_ENV = "BRAINLAYER_BACKUP_LOG_PATH"
BACKUP_LOG_PROVENANCE_ENV = "BRAINLAYER_BACKUP_LOG_PROVENANCE"
BACKUP_RESUME_MAX_AGE_HOURS_ENV = "BRAINLAYER_BACKUP_RESUME_MAX_AGE_HOURS"
# The snapshot budget scales with the database. MBP 2026-10-20, 20.75 GB raw, killed at 2700.3 s:
# VACUUM INTO 54 s + full integrity_check 538 s + live gzip -6 >=2107 s (still running at 9.68 GB of a
# 8.9-9.8 GB output) + live gunzip verify ~144 s (36 s standalone x ~4) = >=2843 s, i.e. >=137 s/GB.
# 200 s/GB gives >=46% headroom over that lower bound: 4200 s for 21 GB.
SNAPSHOT_SECONDS_PER_GB = 200
# The old flat budget (3x a measured 858.5 s copy) stays the floor for small databases.
SNAPSHOT_TIMEOUT_FLOOR_SECONDS = 2700
# Started at 03:45 after a full index, a 5400 s snapshot ends 05:15 and still leaves weekly
# maintenance a 2100 s hold before its 05:50 reserve (a healthy weekly took 815 s on 10-17).
SNAPSHOT_TIMEOUT_CEILING_SECONDS = 5400
DEFAULT_SNAPSHOT_TIMEOUT_SECONDS = SNAPSHOT_TIMEOUT_CEILING_SECONDS
# The calendar kills at its budget; the backup's own deadline must fire this much earlier so its
# supervisor (SIGALRM, 10 s grace, SIGKILL) can still write the BackupTimeoutError receipt.
SNAPSHOT_RECEIPT_RESERVE_SECONDS = 60
BACKUP_LOCAL_MAX_AGE_HOURS_ENV = "BRAINLAYER_BACKUP_LOCAL_MAX_AGE_HOURS"
BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS_ENV = "BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS"
BACKUP_SUPERVISED_CHILD_ENV = "BRAINLAYER_BACKUP_SUPERVISED_CHILD"
BACKUP_INVOCATION_ENV = "BRAINLAYER_BACKUP_INVOCATION_ID"
BACKUP_SQLITE_CHECK_TIMEOUT_ENV = "BRAINLAYER_BACKUP_SQLITE_CHECK_TIMEOUT_SECONDS"
BACKUP_DRIVE_RETENTION_ENV = "BRAINLAYER_BACKUP_DRIVE_RETENTION"
DRIVE_UPLOAD_DEADLINE_FLOOR_ENV = "BRAINLAYER_DRIVE_UPLOAD_DEADLINE_FLOOR_SECONDS"
DRIVE_UPLOAD_MIN_BYTES_PER_SECOND_ENV = "BRAINLAYER_DRIVE_UPLOAD_MIN_BYTES_PER_SECOND"
DRIVE_UPLOAD_STALL_MAX_ATTEMPTS_ENV = "BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS"
DRIVE_UPLOAD_RETRY_BUDGET_ENV = "BRAINLAYER_DRIVE_UPLOAD_RETRY_BUDGET_SECONDS"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DRIVE_SCOPES = drive_credentials.DRIVE_SCOPES
DEFAULT_LOCAL_COMPRESSED_KEEP = 3
DEFAULT_LOCAL_UNCOMPRESSED_KEEP = 1
DEFAULT_DRIVE_KEEP = 7
DRIVE_RETENTION_ENABLED = False
DEFAULT_BACKUP_CLIENT_TIMEOUT_SECONDS = 0
DEFAULT_BACKUP_TIMEOUT_SECONDS = 8 * 60 * 60
DEFAULT_BACKUP_ATTEMPT_MAX_AGE_SECONDS = 24 * 60 * 60
DEFAULT_BACKUP_SQLITE_CHECK_TIMEOUT_SECONDS = 0
# Two minutes tolerates ordinary request latency; the throughput term stretches
# the deadline for chunks larger than the default 8 MiB without allowing a
# zero-window connection to occupy the eight-hour backup budget.
DEFAULT_DRIVE_UPLOAD_DEADLINE_FLOOR_SECONDS = 120.0
DEFAULT_DRIVE_UPLOAD_MIN_BYTES_PER_SECOND = 256 * 1024
DEFAULT_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS = 3
# DB snapshot compression. zstd -T0 -3 measured 2.5-3.7 GB/s at a 0.473 mean ratio on
# 2 GiB windows of the real 20.7 GB raw snapshot, against 24-99 MB/s and 0.491 for
# Python gzip -6 (bl-lead backup-rc124-measure.md section 4). gzip stays selectable
# for rollback, and every reader accepts both formats: existing .db.gz backups,
# local and on Drive, remain valid inputs forever.
BACKUP_COMPRESSOR_ENV = "BRAINLAYER_BACKUP_COMPRESSOR"
BACKUP_ZSTD_PATH_ENV = "BRAINLAYER_BACKUP_ZSTD_PATH"
ZSTD_SNAPSHOT_SUFFIX = ".db.zst"
GZIP_SNAPSHOT_SUFFIX = ".db.gz"
DB_SNAPSHOT_SUFFIXES = (ZSTD_SNAPSHOT_SUFFIX, GZIP_SNAPSHOT_SUFFIX)
ZSTD_LEVEL = 3
# launchd hands jobs PATH=/usr/bin:/bin, which has no zstd; name the Homebrew prefixes.
ZSTD_CANDIDATE_PATHS = ("/opt/homebrew/bin/zstd", "/usr/local/bin/zstd")
ZSTD_POLL_SECONDS = 0.25


@dataclass(frozen=True)
class DriveRetentionPolicy:
    keep_latest: int
    filename_prefix: str = ""
    filename_suffix: str | tuple[str, ...] = DB_SNAPSHOT_SUFFIXES

    def __post_init__(self) -> None:
        if self.keep_latest < 1:
            raise ValueError("keep_latest must be at least 1")


DAILY_RETENTION = DriveRetentionPolicy(keep_latest=DEFAULT_DRIVE_KEEP)
# Daily and weekly jobs share one Drive folder, so both use the same opt-in Drive cap.
WEEKLY_RETENTION = DriveRetentionPolicy(keep_latest=DEFAULT_DRIVE_KEEP)


class BackupAlreadyRunningError(RuntimeError):
    """The shared backup staging lock is held by another backup."""


@dataclass(frozen=True)
class SQLiteBackupArtifact:
    # Historical field name: the compressed snapshot, either .db.zst or .db.gz.
    gzip_path: Path
    uncompressed_path: Path | None
    sentinel_chunks: int
    local_retention_deleted: list[str]
    stale_attempts_deleted: list[str]
    surviving_attempts: list[str]
    surviving_attempt_bytes: int
    surviving_attempt_growth_reserve_bytes: int
    attempt_reclamation: str
    writer_probe_error: str | None
    snapshot_created_at: str | None = None
    local_integrity: str | None = None
    local_integrity_pragma: str | None = None
    snapshot_duration_seconds: float | None = None
    integrity_duration_seconds: float | None = None
    compressor: str | None = None
    compressor_fallback: str | None = None
    compress_duration_seconds: float | None = None
    raw_sha256: str | None = None

    @property
    def archive_path(self) -> Path:
        return self.gzip_path


def local_backup_date(now: dt.datetime | None = None) -> dt.date:
    """Use the machine's local day for snapshot names, weekly pins and eligibility."""
    return (now if now is not None else dt.datetime.now(dt.UTC)).astimezone().date()


def _today() -> str:
    return local_backup_date().isoformat()


class InvalidMachineIdError(ValueError):
    error_code = "invalid_machine_id"


def _validate_machine_id(machine_id: str) -> str:
    value = machine_id.strip()
    if not value or any(not (character.isalnum() or character in "._-") for character in value):
        raise InvalidMachineIdError(f"{BACKUP_MACHINE_ID_ENV} must contain only letters, numbers, '.', '_', or '-'")
    return value


def resolve_machine_id(env: Mapping[str, str] | None = None) -> str:
    """Resolve a stable host identifier, preferring an explicit deployment value."""
    source = env if env is not None else os.environ
    configured = source.get(BACKUP_MACHINE_ID_ENV)
    if configured is not None:
        return _validate_machine_id(configured)

    try:
        completed = subprocess.run(
            ["/usr/sbin/scutil" if sys.platform == "darwin" else "scutil", "--get", "LocalHostName"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None
    if completed is not None and completed.returncode == 0 and completed.stdout.strip():
        return _validate_machine_id(completed.stdout)

    if sys.platform == "darwin":
        raise InvalidMachineIdError(
            f"Cannot resolve stable LocalHostName via /usr/sbin/scutil; set {BACKUP_MACHINE_ID_ENV} explicitly"
        )

    # Non-macOS development and CI hosts do not ship scutil. gethostname is a
    # portability fallback, not the production identity source.
    return _validate_machine_id(socket.gethostname().removesuffix(".local"))


def default_drive_folder_parts(machine_id: str) -> list[str]:
    """Return the per-machine Drive folder, preserving the M4's legacy folder."""
    resolved = _validate_machine_id(machine_id)
    if resolved == CANONICAL_MACHINE_ID:
        return list(CANONICAL_FOLDER_PARTS)
    return [*CANONICAL_FOLDER_PARTS[:-1], f"brainlayer-db-{resolved}"]


def _redact_upload_secrets(value: str) -> str:
    def redact_url(match: re.Match[str]) -> str:
        url = match.group(0)
        return "[redacted-session-url]" if "upload_id" in url.lower() or "/upload/drive/" in url else url

    value = re.sub(r"https?://[^\s\"'<>]+", redact_url, value)
    return re.sub(r"(?i)(upload_id(?:=|%3d))[^&\s\"'<>;]+", r"\1[REDACTED]", value)


def _redact_upload_exception(exc: BaseException) -> None:
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        exc.args = tuple(_redact_upload_secrets(arg) if isinstance(arg, str) else arg for arg in exc.args)
        exc = exc.__cause__ or exc.__context__


def _receipt_append_prefix(fd: int, read_fd: int) -> str:
    """Repair only an incomplete final append while holding the log's exclusive lock."""
    size = os.fstat(fd).st_size
    if not size or os.pread(read_fd, 1, size - 1) == b"\n":
        return ""
    start = size
    tail = b""
    while start:
        previous = max(0, start - 8192)
        tail = os.pread(read_fd, start - previous, previous) + tail
        start = previous
        if b"\n" in tail:
            break
    boundary = tail.rfind(b"\n") + 1
    try:
        json.loads(tail[boundary:])
    except (ValueError, UnicodeDecodeError):
        os.ftruncate(fd, start + boundary)
        return ""
    return "\n"  # Preserve a complete legacy record lacking a line terminator.


class BackupReceiptUnreadableError(ValueError):
    """The receipt journal cannot prove which snapshots are pending; nothing is deleted or repaired."""

    error_type = "BackupReceiptUnreadableError"

    def __init__(self, path: Path, *, error_code: str, detail: str):
        self.error_code = error_code
        self.path = Path(path)
        super().__init__(
            f"Backup receipt journal {path} {detail}; cannot prove which local snapshots are pending upload; "
            "journal and all local snapshots retained for recovery"
        )


# The one stdout shape known to sit in real journals (printed by upload_file_to_drive_raw).
_KNOWN_PROGRESS_NOISE = re.compile(rb"drive upload progress: \d+/\d+ bytes")


def _scan_backup_receipts(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    """Receipts, known progress-noise lines, and other (uncertain) non-receipt lines.

    Until 2026-05-18 the job's stdout was redirected into this journal (ec4a544a, moved by
    3dca26a2), so real journals open with ``drive upload progress: N/M bytes`` lines. A line
    that is exactly that shape is known noise. Any other line that is not a JSON object is
    uncertainty: it could have been a receipt. A torn final record is a killed writer and
    is neither.
    """
    raw = path.read_bytes()
    if not raw:
        # _append_json_log never leaves a journal empty; only a lost or interrupted rewrite does.
        raise BackupReceiptUnreadableError(path, error_code="backup_receipt_empty", detail="is empty (0 bytes)")
    raw_lines = raw.splitlines(keepends=True)
    receipts = []
    noise = uncertain = 0
    for index, line in enumerate(raw_lines):
        if not line.strip():
            continue
        try:
            receipt = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            receipt = None
        if isinstance(receipt, dict):
            receipts.append(receipt)
        elif index == len(raw_lines) - 1 and not line.endswith(b"\n"):
            continue  # A killed writer cannot invalidate the preceding durable session.
        elif _KNOWN_PROGRESS_NOISE.fullmatch(line.removesuffix(b"\n")):
            noise += 1
        else:
            uncertain += 1
    if not receipts:
        # Never quote a line: a receipt can carry a resumable-upload capability URL.
        raise BackupReceiptUnreadableError(
            path,
            error_code="backup_receipt_unreadable",
            detail=(
                f"holds no JSON receipt object ({noise} progress-noise lines, "
                f"{uncertain} other non-receipt lines skipped)"
            ),
        )
    return receipts, noise, uncertain


def _read_backup_receipts(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    """``strict`` is for readers that delete, reclaim or overwrite: an uncertain line could have
    been the receipt that superseded what remains, so any keeps those readers fail-closed.
    Known progress noise is not uncertainty."""
    receipts, _noise, uncertain = _scan_backup_receipts(path)
    if strict and uncertain:
        raise BackupReceiptUnreadableError(
            path,
            error_code="backup_receipt_unreadable",
            detail=f"has {uncertain} non-receipt lines; reclamation stays disabled",
        )
    return receipts


def _append_json_log(path: Path, payload: dict[str, Any]) -> None:
    payload = {
        key: _redact_upload_secrets(value) if "error" in key and isinstance(value, str) else value
        for key, value in payload.items()
    }
    if "uploaded" in payload:
        payload.setdefault("snapshot_created_at", None)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a", encoding="utf-8") as handle:
        if payload.get("resumable_upload"):
            os.fchmod(handle.fileno(), 0o600)  # The session URL is a private upload capability.
        fcntl.flock(handle, fcntl.LOCK_EX)
        with path.open("rb") as reader:
            prefix = _receipt_append_prefix(handle.fileno(), reader.fileno())
        handle.write(prefix + json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())  # Persist before close releases the lock shared with log_cap.


def _backup_log_path(
    log_path: Path | None, *, db_path: Path | None = None, env: Mapping[str, str] | None = None
) -> Path:
    if log_path is not None:
        return Path(log_path)
    source = env if env is not None else os.environ
    configured = source.get(BACKUP_LOG_PATH_ENV)
    if configured:
        return Path(configured)
    resolved_db_path = db_path or get_db_path()
    return resolved_db_path.parent / "logs" / "backup-daily.log"


def _backup_log_provenance() -> str:
    return os.environ.get(BACKUP_LOG_PROVENANCE_ENV, "real")


def _escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


class BackupTimeoutError(TimeoutError):
    pass


class BrainBarSnapshotBusyError(RuntimeError):
    """BrainBar refused a snapshot request because another VACUUM INTO is still running."""


# BrainBarServer.handleBackupToolCallAsync sends this before admission; the running writer continues.
BRAINBAR_SNAPSHOT_BUSY_MESSAGE = "A BrainBar backup snapshot is already in progress."


class BrainBarDatabaseIdentityError(RuntimeError):
    """BrainBar serves a different database than the one this backup was aimed at."""


# MCPRouter.handleBrainBackupVacuumInto refuses with this before any VACUUM INTO when the
# request's expected_db_path does not identify the database BrainBar serves.
BRAINBAR_DB_IDENTITY_MISMATCH = "BrainBar database identity mismatch"


class BackupTargetConflictError(FileExistsError):
    """A peer's dated raw appeared while this run's snapshot ran; it is never replaced."""


def _file_identity(path: str | os.PathLike[str]) -> tuple[int, int] | None:
    try:
        info = os.stat(path)
    except OSError:
        return None
    return info.st_dev, info.st_ino


def _same_database_file(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool:
    """Both paths exist and name the same dev/inode; a missing path never matches."""
    identity = _file_identity(a)
    return identity is not None and identity == _file_identity(b)


def _is_brainbar_snapshot_busy(response: dict[str, Any]) -> bool:
    error = response.get("error")
    return isinstance(error, dict) and error.get("message") == BRAINBAR_SNAPSHOT_BUSY_MESSAGE


class BackupStoppedError(BaseException):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"backup stopped by {signal.Signals(signum).name}")


class DriveUploadStalledError(RuntimeError):
    error_code = "drive_upload_stalled"

    def __init__(self, *, bytes_confirmed: int, total_bytes: int, attempts: int, session_url: str | None = None):
        self.bytes_confirmed = bytes_confirmed
        self.session_url = session_url
        super().__init__(
            f"{self.error_code}: confirmed {bytes_confirmed}/{total_bytes} bytes after {attempts} stalled attempts"
        )


class DriveFolderOwnedByOtherMachineError(RuntimeError):
    error_code = "drive_folder_owned_by_other_machine"

    def __init__(self, *, folder_id: str, machine_id: str, other_machine_ids: set[str]):
        self.folder_id = folder_id
        self.machine_id = machine_id
        self.other_machine_ids = sorted(other_machine_ids)
        super().__init__(
            f"{self.error_code}: folder {folder_id!r} contains snapshots owned by "
            f"{', '.join(self.other_machine_ids)}; current machine is {machine_id!r}"
        )


def _configured_backup_timeout_seconds() -> int:
    raw = os.environ.get(BACKUP_TIMEOUT_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_BACKUP_TIMEOUT_SECONDS
    try:
        seconds = int(raw)
    except ValueError as exc:
        raise ValueError(f"{BACKUP_TIMEOUT_ENV} must be an integer number of seconds") from exc
    if seconds < 1:
        raise ValueError(f"{BACKUP_TIMEOUT_ENV} must be at least 1 second")
    return seconds


def default_snapshot_timeout_seconds(db_path: Path | None = None) -> int:
    """Size the snapshot budget from the database file (stat only; never opens it)."""
    try:
        path = Path(db_path) if db_path is not None else get_db_path()
        size = path.stat().st_size
        wal = Path(f"{path}-wal")
        size += wal.stat().st_size if wal.exists() else 0
    except OSError:
        return SNAPSHOT_TIMEOUT_FLOOR_SECONDS
    seconds = math.ceil(size / 1_000_000_000 * SNAPSHOT_SECONDS_PER_GB / 300) * 300  # whole five minutes
    return int(min(SNAPSHOT_TIMEOUT_CEILING_SECONDS, max(SNAPSHOT_TIMEOUT_FLOOR_SECONDS, seconds)))


def _configured_backup_client_timeout_seconds() -> int | None:
    raw = os.environ.get(BACKUP_CLIENT_TIMEOUT_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_BACKUP_CLIENT_TIMEOUT_SECONDS or None
    try:
        seconds = int(raw)
    except ValueError as exc:
        raise ValueError(f"{BACKUP_CLIENT_TIMEOUT_ENV} must be an integer number of seconds") from exc
    if seconds < 0:
        raise ValueError(f"{BACKUP_CLIENT_TIMEOUT_ENV} must be zero or a positive number of seconds")
    return seconds or None


def _configured_backup_attempt_max_age_seconds() -> int:
    raw = os.environ.get(BACKUP_ATTEMPT_MAX_AGE_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_BACKUP_ATTEMPT_MAX_AGE_SECONDS
    try:
        seconds = int(raw)
    except ValueError as exc:
        raise ValueError(f"{BACKUP_ATTEMPT_MAX_AGE_ENV} must be an integer number of seconds") from exc
    if seconds < 1:
        raise ValueError(f"{BACKUP_ATTEMPT_MAX_AGE_ENV} must be at least 1 second")
    return seconds


def _configured_sqlite_check_timeout_seconds() -> int:
    raw = os.environ.get(BACKUP_SQLITE_CHECK_TIMEOUT_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_BACKUP_SQLITE_CHECK_TIMEOUT_SECONDS
    try:
        seconds = int(raw)
    except ValueError as exc:
        raise ValueError(f"{BACKUP_SQLITE_CHECK_TIMEOUT_ENV} must be an integer number of seconds") from exc
    if seconds < 0:
        raise ValueError(f"{BACKUP_SQLITE_CHECK_TIMEOUT_ENV} must be zero or a positive number of seconds")
    return seconds


def _configured_positive_number(name: str, default: float, *, maximum: float | None = None) -> float:
    raw = os.environ.get(name)
    try:
        value = default if raw is None or raw.strip() == "" else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be a positive number")
    return value


def local_backup_max_age_hours() -> float:
    return _configured_positive_number(BACKUP_LOCAL_MAX_AGE_HOURS_ENV, 6)


def _drive_upload_stall_max_attempts() -> int:
    value = _configured_positive_number(DRIVE_UPLOAD_STALL_MAX_ATTEMPTS_ENV, DEFAULT_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS)
    if value != int(value):
        raise ValueError(f"{DRIVE_UPLOAD_STALL_MAX_ATTEMPTS_ENV} must be an integer")
    return int(value)


class _DriveRequestDeadlineExceeded(TimeoutError):
    pass


# Network failures that say nothing about whether Drive kept the bytes: the answer is to ask
# Drive (a status probe, or a re-read), never to assume failure. HTTP errors are not here.
_DRIVE_TRANSIENT_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    ssl.SSLEOFError,
    requests.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)


def _retry_transient_drive_call(call: Callable[[], Any], *, what: str) -> Any:
    """Run a Drive call that is safe to repeat, retrying transient network errors (bounded)."""
    # google-auth wraps a token refresh that failed on the network; RefreshError (a revoked or
    # invalid grant) is a different class and is never retried.
    from google.auth.exceptions import TransportError

    limit = _drive_upload_stall_max_attempts()
    for attempt in range(1, limit + 1):
        try:
            return call()
        except BackupTimeoutError:
            raise  # the whole-run deadline is not a network blip
        except (*_DRIVE_TRANSIENT_NETWORK_ERRORS, TransportError) as exc:
            if attempt >= limit:
                raise RuntimeError(
                    f"{what} failed after {attempt} attempts: {type(exc).__name__}: {_redact_upload_secrets(str(exc))}"
                ) from exc
            sleep_seconds = min(60, 2**attempt)
            print(
                f"{what} retry attempt={attempt}/{limit}: {type(exc).__name__}: {_redact_upload_secrets(str(exc))}; sleeping {sleep_seconds}s",
                flush=True,
            )
            _sleep(sleep_seconds)
    raise AssertionError("unreachable")


def _execute_drive_read(request: Any, *, what: str) -> Any:
    """Execute a read-only Drive API request, retrying transient network errors (bounded)."""
    return _retry_transient_drive_call(request.execute, what=what)


def _drive_put_with_deadline(session: Any, url: str, *, headers: dict[str, str], data: bytes, deadline: float):
    outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def perform() -> None:
        try:
            outcome.put((True, session.put(url, headers=headers, data=data, timeout=deadline + 5)))
        except Exception as exc:
            outcome.put((False, exc))

    threading.Thread(target=perform, daemon=True, name="brainlayer-drive-put").start()
    try:
        succeeded, value = outcome.get(timeout=deadline)
    except queue.Empty as exc:
        session.close()
        raise _DriveRequestDeadlineExceeded(f"Drive PUT exceeded {deadline:.3f}s total deadline") from exc
    if succeeded:
        return value
    if isinstance(value, requests.Timeout):
        raise _DriveRequestDeadlineExceeded(f"Drive PUT exceeded {deadline:.3f}s total deadline") from value
    raise value


def _raise_backup_timeout(signum, frame) -> None:  # noqa: ARG001
    raise BackupTimeoutError("backup exceeded configured wall-clock timeout")


def _raise_backup_stopped(signum, frame) -> None:  # noqa: ARG001
    raise BackupStoppedError(signum)


def _sqlite_pragma_check(db_path: Path, pragma_name: str) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # Bound integrity-check memory while avoiding tiny-cache index thrashing.
        conn.execute("PRAGMA cache_size=-131072")
        rows = conn.execute(f"PRAGMA {pragma_name}").fetchall()
        row = rows[0] if len(rows) == 1 else None
    finally:
        conn.close()
    return str(row[0]) if row else ""


def _optional_sqlite_pragma_check(db_path: Path, pragma_name: str) -> str:
    timeout_seconds = _configured_sqlite_check_timeout_seconds()
    if timeout_seconds == 0:
        return "skipped"
    if pragma_name not in {"quick_check", "integrity_check"}:
        raise ValueError(f"unsupported SQLite check: {pragma_name}")
    check_script = (
        "import sqlite3,sys; "
        "path,check=sys.argv[1:3]; "
        "conn=sqlite3.connect('file:'+path+'?mode=ro', uri=True); "
        "row=conn.execute('PRAGMA '+check).fetchone(); "
        "conn.close(); "
        "print(str(row[0]) if row else '')"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", check_script, str(Path(db_path).expanduser().resolve()), pragma_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    if completed.returncode != 0:
        raise RuntimeError(f"Backup {pragma_name} subprocess failed: {completed.stderr.strip()}")
    result = completed.stdout.strip()
    if result != "ok":
        raise RuntimeError(f"Backup {pragma_name} failed: {result!r}")
    return result


def _count_chunks(db_path: Path) -> int:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else 0


def _validate_backup_target(db_path: Path) -> int:
    db_path = Path(db_path).expanduser().resolve()
    conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        page_count_row = conn.execute("PRAGMA page_count").fetchone()
        page_count = int(page_count_row[0]) if page_count_row else 0
        if page_count < 1:
            raise RuntimeError("Backup target has zero SQLite pages")
        chunks_row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        chunks = int(chunks_row[0]) if chunks_row else 0
        if chunks < 1:
            raise RuntimeError("Backup target has zero chunks rows")
        return chunks
    finally:
        conn.close()


def _parse_uncompressed_snapshot_date(name: str) -> dt.date | None:
    if not name.endswith(".db") or len(name) != len("YYYY-MM-DD.db"):
        return None
    try:
        return dt.date.fromisoformat(name[:10])
    except ValueError:
        return None


def prune_local_uncompressed_snapshots(
    output_dir: Path, *, keep_latest: int = DEFAULT_LOCAL_UNCOMPRESSED_KEEP
) -> list[str]:
    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")
    output_dir = Path(output_dir).expanduser()
    dated: list[tuple[dt.date, Path]] = []
    if not output_dir.exists():
        return []
    for path in output_dir.iterdir():
        parsed = _parse_uncompressed_snapshot_date(path.name)
        if parsed and path.is_file():
            dated.append((parsed, path))

    dated.sort(key=lambda pair: pair[0], reverse=True)
    deleted: list[str] = []
    for _, path in dated[keep_latest:]:
        path.unlink(missing_ok=True)
        deleted.append(path.name)
    return deleted


def _sweep_stale_backup_attempts(
    output_dir: Path,
    *,
    max_age_seconds: int | None = None,
    now: float | None = None,
    writer_started_at: float | None = None,
) -> tuple[list[str], list[Path]]:
    """Remove old, daemon-confirmed attempts while preserving unproven or recent entries."""
    output_dir = Path(output_dir)
    cutoff = (time.time() if now is None else now) - (
        _configured_backup_attempt_max_age_seconds() if max_age_seconds is None else max_age_seconds
    )
    deleted: list[str] = []
    surviving: list[Path] = []
    for path in sorted(output_dir.glob(".*.db.attempt-*")):
        try:
            if path.name.endswith(".complete"):
                continue
            if path.is_symlink() or not path.is_file():
                surviving.append(path)
                continue
            path_mtime = path.stat().st_mtime
            completion_marker = _backup_attempt_completion_marker(path)
            if completion_marker.is_symlink() or not completion_marker.is_file():
                if path_mtime <= cutoff and writer_started_at is not None and path_mtime < writer_started_at:
                    path.unlink()
                    deleted.append(path.name)
                else:
                    surviving.append(path)
                continue
            if max(path_mtime, completion_marker.stat().st_mtime) <= cutoff:
                path.unlink()
                completion_marker.unlink(missing_ok=True)
                deleted.append(path.name)
            else:
                surviving.append(path)
        except FileNotFoundError:
            continue
    return deleted, surviving


_OWNED_TEMP_PREFIXES = ("brainlayer-backup-", "brainlayer-restore-verify-")


def _owned_temp_inventory(output_dir: Path) -> list[tuple[str, int]]:
    """Measure only this job's abandoned temporary directories; never follow links."""
    inventory: list[tuple[str, int]] = []
    for path in sorted(output_dir.iterdir()):
        if not path.name.startswith(_OWNED_TEMP_PREFIXES) or path.is_symlink() or not path.is_dir():
            continue
        size = 0
        for item in path.rglob("*"):
            if item.is_file() and not item.is_symlink():
                try:
                    size += item.stat().st_size
                except FileNotFoundError:
                    pass
        inventory.append((path.name, size))
    return inventory


def _cleanup_consumed_raw_remnants(output_dir: Path) -> list[str]:
    """Remove only inode-bound consumed raws from this job's abandoned attempts.

    A consumed raw is unrestorable, unlike an unconsumed raw or any archive. The caller
    holds the job's staging lock; links, unmarked files and foreign inodes fail closed.
    """
    removed = []
    if not output_dir.exists() or output_dir.is_symlink():
        return removed
    for directory in sorted(output_dir.glob("brainlayer-backup-*")):
        try:
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError:
            continue
        try:
            if os.fstat(directory_fd).st_uid != os.getuid():
                continue
            with os.fdopen(os.open(".consumed", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)) as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
                    continue
                marker = json.loads(handle.read(4096))
            if not isinstance(marker, dict) or marker.get("consumed") is not True:
                continue
            name = marker.get("raw")
            if not isinstance(name, str) or _parse_uncompressed_snapshot_date(name) is None:
                continue
            with os.fdopen(os.open(".backup-date", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)) as handle:
                if handle.read(32).strip() != name[:10]:
                    continue
            raw_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            try:
                info = os.fstat(raw_fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_uid != os.getuid()
                    or (info.st_dev, info.st_ino) != (marker.get("dev"), marker.get("ino"))
                ):
                    continue
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                    continue
                os.unlink(name, dir_fd=directory_fd)
                os.fsync(directory_fd)
                removed.append(f"{directory.name}/{name}")
                print(f"Backup removed unrestorable consumed crash raw: {directory.name}/{name}", flush=True)
            finally:
                os.close(raw_fd)
        except (OSError, UnicodeError, ValueError):
            continue
        finally:
            os.close(directory_fd)
    return removed


def _consume_raw_in_attempt(raw: Path, directory: Path) -> Path:
    info = raw.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Cannot consume a linked or non-regular raw snapshot")
    marker = {"raw": raw.name, "dev": info.st_dev, "ino": info.st_ino, "consumed": True}
    with (directory / ".consumed").open("x", encoding="ascii") as handle:
        json.dump(marker, handle)
        handle.flush()
        os.fsync(handle.fileno())
    consumed = directory / raw.name
    raw.rename(consumed)
    # Persist both sides of the rename before any destructive punch. A power loss must
    # never resurrect the consumed inode at the published raw path.
    for path in (directory, raw.parent):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return consumed


def _prune_owned_temps_after_verified(output_dir: Path, verified_snapshot: Path) -> list[str]:
    """Remove old job-owned crash remnants only after a newer Drive backup verified."""
    verified_date = _parse_snapshot_date(verified_snapshot.name)
    if verified_date is None:
        return []
    verified_mtime = verified_snapshot.stat().st_mtime
    removed: list[str] = []
    for directory in sorted(output_dir.iterdir()):
        if not directory.name.startswith(_OWNED_TEMP_PREFIXES) or directory.is_symlink() or not directory.is_dir():
            continue
        marker = directory / ".backup-date"
        if not marker.is_file() or marker.is_symlink():
            continue
        try:
            marked_date = dt.date.fromisoformat(marker.read_text(encoding="ascii").strip())
        except (OSError, UnicodeError, ValueError):
            continue
        if marked_date > verified_date or directory.stat().st_mtime > verified_mtime:
            continue
        contents = list(directory.iterdir())
        if any(
            item.is_symlink()
            or not item.is_file()
            or item.name
            not in {
                ".backup-date",
                ".consumed",
                "restored.db",
                f"{marked_date.isoformat()}.db",
                *(f"{marked_date.isoformat()}{suffix}" for suffix in DB_SNAPSHOT_SUFFIXES),
            }
            for item in contents
        ):
            continue
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            print(
                f"Backup temp cleanup deferred: name={directory.name} error={_redact_upload_secrets(str(exc))}",
                flush=True,
            )
        else:
            removed.append(directory.name)
    return removed


def _backup_attempt_completion_marker(attempt_path: Path) -> Path:
    return attempt_path.with_name(f"{attempt_path.name}.complete")


def _brainbar_writer_started_at(socket_path: Path | str | None = None) -> float:
    request = {
        "jsonrpc": "2.0",
        "id": "backup-writer-status",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "brainlayer-backup", "version": "1.0"},
        },
    }
    response = _send_brainbar_json_request(_brainbar_socket_path(socket_path), request, timeout_seconds=5)
    result = response.get("result")
    server_info = result.get("serverInfo") if isinstance(result, dict) else None
    started_at = server_info.get("backupWriterStartedAtUnix") if isinstance(server_info, dict) else None
    if not isinstance(started_at, (int, float)) or started_at <= 0:
        raise RuntimeError("BrainBar initialize response missing backupWriterStartedAtUnix")
    return float(started_at)


def _database_logical_size_bytes(db_path: Path) -> int:
    db_path = Path(db_path).expanduser().resolve()
    main_file_size = db_path.stat().st_size
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        page_count_row = conn.execute("PRAGMA page_count").fetchone()
        page_size_row = conn.execute("PRAGMA page_size").fetchone()
    except sqlite3.Error:
        wal_path = Path(f"{db_path}-wal")
        wal_size = wal_path.stat().st_size if wal_path.exists() else 0
        return main_file_size + wal_size
    finally:
        if conn is not None:
            conn.close()
    page_count = int(page_count_row[0]) if page_count_row else 0
    page_size = int(page_size_row[0]) if page_size_row else 0
    return max(main_file_size, page_count * page_size)


def _important_usage_capacity_bytes(path: Path) -> int | None:
    """Ask macOS for capacity including purgeable space; None means use statfs."""
    if sys.platform != "darwin":
        return None
    script = (
        'function run(argv) { ObjC.import("Foundation"); '
        "var url = $.NSURL.fileURLWithPath(argv[0]); "
        "var key = $.NSURLVolumeAvailableCapacityForImportantUsageKey; "
        "var values = url.resourceValuesForKeysError($.NSArray.arrayWithObject(key), null); "
        'if (!values) return "unavailable"; '
        "var number = values.objectForKey(key); "
        'return number ? ObjC.unwrap(number).toString() : "unavailable"; }'
    )
    try:
        output = subprocess.run(
            ["/usr/bin/osascript", "-l", "JavaScript", "-e", script, str(path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        return int(output) if output.isdecimal() else None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _request_macos_purge(path: Path, amount_bytes: int) -> None:
    """Ask macOS CacheDelete to reclaim its own purgeable content before VACUUM INTO."""
    if sys.platform != "darwin":
        raise RuntimeError("macOS CacheDelete is unavailable on this platform")
    helper = Path(__file__).with_name("cachedelete_purge.swift")
    try:
        completed = subprocess.run(
            ["/usr/bin/swift", str(helper), str(path), str(amount_bytes)],
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except BackupTimeoutError:
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"macOS CacheDelete request unavailable: {_redact_upload_secrets(str(exc))}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"macOS CacheDelete request failed: {detail or f'exit {completed.returncode}'}")


def _estimated_gzip_bytes(output_dir: Path, db_size: int) -> int:
    dated = [
        (_parse_snapshot_date(path.name), path)
        for path in output_dir.iterdir()
        if path.is_file() and not path.is_symlink() and _parse_snapshot_date(path.name) is not None
    ]
    if not dated:
        return db_size
    newest = max(dated, key=lambda item: item[0])[1]
    return max(1, newest.stat().st_size)


def _raw_bytes_pruned_before_verify(output_dir: Path, date_stamp: str, keep_latest: int) -> int:
    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")
    current = output_dir / f"{date_stamp}.db"
    dated = [
        (_parse_uncompressed_snapshot_date(path.name), path)
        for path in output_dir.iterdir()
        if path.is_file() and not path.is_symlink() and _parse_uncompressed_snapshot_date(path.name) is not None
    ]
    future = [(dt.date.fromisoformat(date_stamp), current)] + [(date, path) for date, path in dated if path != current]
    kept = {path for _, path in sorted(future, key=lambda item: item[0], reverse=True)[:keep_latest]}
    return sum(path.stat().st_size for _, path in dated if path not in kept or path == current)


def _backup_phase_peaks(db_size: int, gzip_size: int, *, full: bool, reclaim_raw: bool = False) -> dict[str, int]:
    """New bytes each backup phase holds at once; existing files are already in measured free space.

    The verified raw is unlinked before restore verification allocates restored.db, so the raw
    and the restored copy never coexist (2026-10-19: summing them failed a run with ~16 GB slack).
    """
    if reclaim_raw:
        # Each frame is capped before writing. Budget worst-case expansion, even when the
        # historical compression estimate is too small, plus one frame before its raw is freed.
        expansion = math.ceil(db_size / RECLAIM_CHUNK_BYTES) * FRAME_OVERHEAD_BYTES
        phases = {"raw_copy": db_size, "gzip": db_size + expansion + RECLAIM_CHUNK_BYTES}
        if full:
            phases["drive_download"] = gzip_size
        phases["restore"] = gzip_size  # Hash verification allocates no second file.
        return phases
    phases = {
        "raw_copy": db_size,  # VACUUM INTO writes the raw snapshot.
        "gzip": db_size + gzip_size,  # The raw is still present while the gzip is written.
    }
    if full:
        phases["drive_download"] = 2 * gzip_size  # Local gzip plus the downloaded Drive copy.
    phases["restore"] = gzip_size + (gzip_size if full else 0) + db_size  # Raw already removed.
    return phases


def _peak_backup_required_bytes(db_size: int, gzip_size: int, *, full: bool, reclaim_raw: bool = False) -> int:
    return (
        max(_backup_phase_peaks(db_size, gzip_size, full=full, reclaim_raw=reclaim_raw).values())
        + MIN_RAW_FREE_BYTES
        + COPY_CHUNK_BYTES
    )


def _stream_sha256(source: Any) -> bytes:
    digest = hashlib.sha256()
    while chunk := source.read(COPY_CHUNK_BYTES):
        digest.update(chunk)
    return digest.digest()


def _raw_archive_name(output_dir: Path, raw: Path) -> str:
    """The compressed sibling that may replace `raw`: .db.zst first, then legacy .db.gz."""
    for suffix in (".zst", ".gz"):
        if os.path.lexists(output_dir / f"{raw.name}{suffix}"):
            return f"{raw.name}{suffix}"
    raise FileNotFoundError(f"No compressed snapshot beside raw snapshot: {raw.name}")


def _remove_raw_with_verified_archive(
    output_dir: Path, raw: Path, *, own_run: bool = False, archive_name: str | None = None
) -> None:
    """Prove the raw/archive pair is bound and durable before unlinking its singly owned raw.

    Accepts a .db.zst or legacy .db.gz sibling; the raw survives any proof failure.
    """
    if raw.parent != output_dir or _parse_uncompressed_snapshot_date(raw.name) is None:
        raise ValueError("Raw snapshot is outside the backup directory or has an invalid name")
    archive_name = archive_name or _raw_archive_name(output_dir, raw)
    if archive_name not in {f"{raw.name}.zst", f"{raw.name}.gz"}:
        raise ValueError(f"Archive {archive_name!r} is not the compressed sibling of {raw.name}")
    if archive_name.endswith(".zst"):
        return _remove_raw_with_verified_zstd(output_dir, raw, own_run=own_run)
    return _remove_raw_with_verified_gzip(output_dir, raw, own_run=own_run)


def _remove_raw_with_verified_zstd(output_dir: Path, raw: Path, *, own_run: bool = False) -> None:
    """zstd twin of the gzip proof: frame checksum, exact size, SQLite header, and SHA-256 if not ours."""
    directory = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:

        def file_identity(info: os.stat_result) -> tuple[int, ...]:
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f"Retaining non-regular or linked snapshot: {raw.name}")
            return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

        def identity(name: str) -> tuple[int, ...]:
            return file_identity(os.stat(name, dir_fd=directory, follow_symlinks=False))

        archive_name = f"{raw.name}.zst"
        raw_identity, archive_identity = identity(raw.name), identity(archive_name)

        def require_unchanged() -> None:
            if raw_identity != identity(raw.name) or archive_identity != identity(archive_name):
                raise ValueError(f"Snapshot changed during integrity check: {raw.name}")

        raw_hash: bytes | None = None
        if not own_run:
            # A same-date archive may predate this raw. Prove every byte, not just its name/size.
            with os.fdopen(os.open(raw.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory), "rb") as original:
                if file_identity(os.fstat(original.fileno())) != raw_identity:
                    raise ValueError(f"Snapshot changed before binding check: {raw.name}")
                raw_hash = _stream_sha256(original)
        with os.fdopen(os.open(archive_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory), "rb") as compressed:
            if file_identity(os.fstat(compressed.fileno())) != archive_identity:
                raise ValueError(f"Snapshot changed before binding check: {raw.name}")
            header, size, restored_hash = _zstd_stream_digest(compressed, want_sha256=raw_hash is not None)
            if size != raw_identity[2]:
                raise ValueError(f"Raw/zstd snapshot binding failed (size): {raw.name}")
            if header != b"SQLite format 3\x00":
                raise ValueError(f"Raw/zstd snapshot binding failed (SQLite header): {raw.name}")
            if raw_hash is not None and restored_hash != raw_hash:
                raise ValueError(f"Raw/zstd snapshot binding failed (SHA-256): {raw.name}")
            if sys.platform == "darwin":
                fcntl.fcntl(compressed.fileno(), fcntl.F_FULLFSYNC)
            else:
                os.fsync(compressed.fileno())
            os.fsync(directory)
            require_unchanged()
            os.unlink(raw.name, dir_fd=directory)
    finally:
        os.close(directory)


def _remove_raw_with_verified_gzip(output_dir: Path, raw: Path, *, own_run: bool = False) -> None:
    """Prove the pair is bound and durable before unlinking its singly owned raw."""
    if raw.parent != output_dir or _parse_uncompressed_snapshot_date(raw.name) is None:
        raise ValueError("Raw snapshot is outside the backup directory or has an invalid name")
    directory = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:

        def file_identity(info: os.stat_result) -> tuple[int, ...]:
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f"Retaining non-regular or linked snapshot: {raw.name}")
            return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

        def identity(name: str) -> tuple[int, ...]:
            return file_identity(os.stat(name, dir_fd=directory, follow_symlinks=False))

        gzip_name = f"{raw.name}.gz"
        raw_identity, gzip_identity = identity(raw.name), identity(gzip_name)

        def require_unchanged() -> None:
            if raw_identity != identity(raw.name) or gzip_identity != identity(gzip_name):
                raise ValueError(f"Snapshot changed during integrity check: {raw.name}")

        _gunzip_test(output_dir / gzip_name)
        require_unchanged()
        with os.fdopen(os.open(gzip_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory), "rb") as compressed:
            if file_identity(os.fstat(compressed.fileno())) != gzip_identity:
                raise ValueError(f"Snapshot changed before binding check: {raw.name}")
            with gzip.GzipFile(fileobj=compressed, mode="rb") as restored:
                header = restored.read(16)
                if header != b"SQLite format 3\x00":
                    raise ValueError(f"Raw/gzip snapshot binding failed (SQLite header): {raw.name}")
                # A concatenated gzip's final ISIZE covers only its last member.
                size, digest = len(header), hashlib.sha256(header)
                while chunk := restored.read(COPY_CHUNK_BYTES):
                    size += len(chunk)
                    digest.update(chunk)
                if size != raw_identity[2]:
                    raise ValueError(f"Raw/gzip snapshot binding failed (size): {raw.name}")
                if not own_run:
                    # A same-date gzip may predate this raw. Prove every byte, not just its name/size.
                    with os.fdopen(os.open(raw.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory), "rb") as original:
                        if file_identity(os.fstat(original.fileno())) != raw_identity:
                            raise ValueError(f"Snapshot changed before binding check: {raw.name}")
                        raw_hash = _stream_sha256(original)
                    if digest.digest() != raw_hash:
                        raise ValueError(f"Raw/gzip snapshot binding failed (SHA-256): {raw.name}")
            if sys.platform == "darwin":
                fcntl.fcntl(compressed.fileno(), fcntl.F_FULLFSYNC)
            else:
                os.fsync(compressed.fileno())
            os.fsync(directory)
            require_unchanged()
            os.unlink(raw.name, dir_fd=directory)
    finally:
        os.close(directory)


def _compressed_snapshot_names(output_dir: Path) -> list[str]:
    return sorted(path.name for path in output_dir.iterdir() if _parse_snapshot_date(path.name) is not None)


def _retained_raw_snapshot_names(output_dir: Path) -> list[str]:
    return sorted(path.name for path in output_dir.glob("*.db") if _parse_uncompressed_snapshot_date(path.name))


def _cleanup_raw_snapshots(output_dir: Path) -> tuple[list[str], list[str]]:
    deleted, retained_raw = [], []
    for raw in sorted(output_dir.glob("*.db")):
        if _parse_uncompressed_snapshot_date(raw.name) is None:
            continue
        try:
            _remove_raw_with_verified_archive(output_dir, raw)
            deleted.append(raw.name)
        except BackupTimeoutError:
            raise
        except (OSError, EOFError, ValueError, RuntimeError, subprocess.CalledProcessError):
            retained_raw.append(raw.name)
    return deleted, retained_raw


def _require_raw_floor(path: Path, *, forthcoming: int = 0) -> None:
    free = shutil.disk_usage(path).free
    if free < MIN_RAW_FREE_BYTES + forthcoming:
        raise RuntimeError(
            f"raw free space floor breached: raw={free} floor={MIN_RAW_FREE_BYTES} forthcoming={forthcoming}"
        )


def _copy_with_raw_floor(source: Any, destination: Any, output_dir: Path) -> None:
    while chunk := source.read(COPY_CHUNK_BYTES):
        _require_raw_floor(output_dir, forthcoming=COPY_CHUNK_BYTES)
        destination.write(chunk)
        _require_raw_floor(output_dir)


def _refuse_foreign_db_on_production_socket(db_path: Path, socket_path: Path | str | None) -> None:
    """The production BrainBar serves the configured DB; never aim it at any other one.

    2026-10-19: an ad-hoc probe outside pytest passed a temp db_path with no socket, fell
    through to /tmp/brainbar.sock, and received a 20 GB production copy. Pass an isolated
    socket_path (or BRAINBAR_SOCKET_PATH) to back up a different database.
    """
    resolved_socket = _brainbar_socket_path(socket_path)
    if os.path.realpath(resolved_socket) != os.path.realpath(DEFAULT_BRAINBAR_SOCKET_PATH):
        return
    configured_db = resolve_db_path()
    if not _same_database_file(db_path, configured_db):
        raise BrainBarDatabaseIdentityError(
            f"{BRAINBAR_DB_IDENTITY_MISMATCH}: refusing to snapshot {db_path} through the production BrainBar "
            f"socket {resolved_socket}, which serves the configured database {configured_db}; "
            "pass an isolated socket_path to back up any other database"
        )


def _remove_owned_raw(raw: Path, owned: tuple[int, int]) -> None:
    """Remove raw only if it is still the file this run promoted.

    The rename aside is atomic, so the inode checked is the inode removed. A peer's file that
    took the name is linked back create-exclusive, or left aside and reported, never deleted.
    """
    aside = raw.with_name(f".{raw.name}.orphan-{uuid.uuid4().hex}")
    try:
        os.rename(raw, aside)
    except FileNotFoundError:
        return
    captured = os.lstat(aside)
    if (captured.st_dev, captured.st_ino) == owned:
        aside.unlink()
        print(f"Backup removed unverified raw from non-canonical staging: {raw}", flush=True)
        return
    try:
        os.link(aside, raw)
        aside.unlink()
    except FileExistsError:
        print(f"Backup left a peer's raw aside, its name was retaken: {aside}", flush=True)


def _is_canonical_staging(output_dir: Path) -> bool:
    return os.path.realpath(output_dir) == os.path.realpath(Path(DEFAULT_STAGING_DIR).expanduser())


def _snapshot_suffix(name: str) -> str | None:
    """The compressed-snapshot suffix of `name`, or None for anything else."""
    return next((suffix for suffix in DB_SNAPSHOT_SUFFIXES if name.endswith(suffix)), None)


def _resolve_zstd() -> str | None:
    """Absolute zstd path, or None. A set-but-unusable override fails loudly, never falls through."""
    override = os.environ.get(BACKUP_ZSTD_PATH_ENV)
    if override is not None:
        candidate = Path(override)
        if not candidate.is_absolute() or not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise ValueError(f"{BACKUP_ZSTD_PATH_ENV} must name an absolute executable zstd: {override!r}")
        return str(candidate)
    for candidate in ZSTD_CANDIDATE_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("zstd")


def _require_zstd() -> str:
    binary = _resolve_zstd()
    if binary is None:
        raise RuntimeError(
            f"zstd binary not found (checked {BACKUP_ZSTD_PATH_ENV}, {', '.join(ZSTD_CANDIDATE_PATHS)}, PATH)"
        )
    return binary


def _selected_compressor() -> tuple[str, str | None, str | None]:
    """(compressor, zstd binary, fallback reason). Missing zstd degrades to gzip loudly, not silently."""
    requested = os.environ.get(BACKUP_COMPRESSOR_ENV, "zstd").strip().lower() or "zstd"
    if requested == "gzip":
        return "gzip", None, None
    if requested != "zstd":
        raise ValueError(f"{BACKUP_COMPRESSOR_ENV} must be zstd or gzip, not {requested!r}")
    binary = _resolve_zstd()
    if binary is None:
        reason = "zstd binary not found; compressed with gzip instead"
        print(f"Backup compressor fallback: {reason}", flush=True)
        return "gzip", None, reason
    return "zstd", binary, None


def _run_zstd(binary: str, args: list[str], *, output: Path, floor_dir: Path, stdin: Any = None) -> None:
    """Run zstd writing to `output`, enforcing the raw free-space floor while it runs.

    The child stays in this process group (no new session), so the calendar's
    process-group stop reaches it directly. Any exception here, including the
    SIGALRM/SIGTERM handlers' BackupTimeoutError/BackupStoppedError raised inside
    wait(), kills and reaps it, so it can never outlive the backup as an orphan.
    """
    with tempfile.TemporaryFile() as errors:
        with output.open("wb") as sink:
            process = subprocess.Popen(
                [binary, *args],
                stdin=stdin if stdin is not None else subprocess.DEVNULL,
                stdout=sink,
                stderr=errors,
            )
        try:
            while True:
                try:
                    returncode = process.wait(timeout=ZSTD_POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    _require_raw_floor(floor_dir)
            _require_raw_floor(floor_dir)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        if returncode != 0:
            errors.seek(0)
            detail = errors.read(4096).decode("utf-8", "replace").strip()
            raise RuntimeError(f"zstd {' '.join(args[:2])} exited {returncode}: {detail}")


def _compress_zstd(binary: str, raw: Path, destination: Path, output_dir: Path) -> None:
    _run_zstd(
        binary,
        [f"-{ZSTD_LEVEL}", "-T0", "--check", "-q", "-c", str(raw)],
        output=destination,
        floor_dir=output_dir,
    )


def _zstd_test(path: Path) -> None:
    subprocess.run([_require_zstd(), "-t", "-q", str(path)], check=True, capture_output=True, text=True)


def _zstd_stream_digest(compressed: Any, *, want_sha256: bool) -> tuple[bytes, int, bytes | None]:
    """Decompress an already-open archive fd: (first 16 bytes, total size, sha256 or None).

    Feeding the identity-checked fd on stdin binds the check to those exact bytes;
    zstd's frame checksum (written with --check) is verified, so a nonzero exit means corruption.
    """
    process = subprocess.Popen(
        [_require_zstd(), "-d", "-q", "-c"],
        stdin=compressed,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert process.stdout is not None
        digest = hashlib.sha256() if want_sha256 else None
        header = b""
        size = 0
        while chunk := process.stdout.read(COPY_CHUNK_BYTES):
            if len(header) < 16:
                header += chunk[: 16 - len(header)]
            size += len(chunk)
            if digest is not None:
                digest.update(chunk)
        returncode = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()
    if returncode != 0:
        raise ValueError(f"zstd archive failed its frame checksum (exit {returncode})")
    return header, size, digest.digest() if digest is not None else None


def _archive_test(path: Path) -> None:
    """Integrity-test a compressed snapshot of either format."""
    if path.name.endswith(ZSTD_SNAPSHOT_SUFFIX):
        _zstd_test(path)
    else:
        _gunzip_test(path)


def _decompress_archive_to(archive: Path, destination: Path) -> None:
    """Restore a compressed snapshot of either format to `destination`."""
    if archive.name.endswith(ZSTD_SNAPSHOT_SUFFIX):
        _run_zstd(_require_zstd(), ["-d", "-q", "-c", str(archive)], output=destination, floor_dir=destination.parent)
    else:
        _decompress_gzip_to(archive, destination)


def _punch_raw_hole(fd: int, offset: int, length: int) -> None:
    if sys.platform != "darwin":
        raise OSError("F_PUNCHHOLE requires macOS")
    # Darwin fpunchhole_t: flags, reserved, off_t offset, off_t length (sys/fcntl.h).
    fcntl.fcntl(fd, 99, struct.pack("=IIqq", 0, 0, offset, length))


def _probe_raw_reclamation(output_dir: Path) -> tuple[bool, str | None]:
    """Test the actual staging filesystem, using only a new, disposable file."""
    try:
        with tempfile.TemporaryFile(dir=output_dir) as probe:
            probe.write(b"x" * COPY_CHUNK_BYTES)
            probe.flush()
            os.fsync(probe.fileno())
            before = os.fstat(probe.fileno()).st_blocks
            _punch_raw_hole(probe.fileno(), 0, COPY_CHUNK_BYTES)
            after = os.fstat(probe.fileno()).st_blocks
            probe.seek(0)
            if before <= after or probe.read() != b"\0" * COPY_CHUNK_BYTES:
                raise OSError("hole punch did not release allocated blocks")
        return True, None
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _archive_stream_sha256(source: Any, *, zstd: bool) -> str:
    if zstd:
        return _zstd_stream_digest(source, want_sha256=True)[2].hex()
    with gzip.GzipFile(fileobj=source, mode="rb") as restored:
        return _stream_sha256(restored).hex()


def _compress_reclaiming_raw(raw: Path, archive: Path, *, zstd_binary: str | None) -> str:
    """Write durable independent frames, then release their raw extents; hash before punching.

    Concatenated gzip members and zstd frames decode to the original byte stream. A failed
    punch after selection aborts: switching to an unbudgeted full-copy path would be unsafe.
    The caller moves this run's raw into its attempt directory before destructive consumption.
    """
    digest = hashlib.sha256()
    with os.fdopen(os.open(raw, os.O_RDWR | os.O_NOFOLLOW), "r+b") as source, archive.open("xb") as sink:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Cannot consume a linked or non-regular raw snapshot")
        offset = 0
        while chunk := source.read(RECLAIM_CHUNK_BYTES):
            digest.update(chunk)
            if zstd_binary is None:
                frame = gzip.compress(chunk, compresslevel=6)
            else:
                frame = subprocess.run(
                    [zstd_binary, f"-{ZSTD_LEVEL}", "-T1", "--check", "-q", "-c"],
                    input=chunk,
                    capture_output=True,
                    check=True,
                ).stdout
            if len(frame) > len(chunk) + FRAME_OVERHEAD_BYTES:
                raise ValueError("Compressed frame exceeds budgeted expansion")
            _require_raw_floor(archive.parent, forthcoming=len(frame))
            sink.write(frame)
            sink.flush()
            if sys.platform == "darwin":
                fcntl.fcntl(sink.fileno(), fcntl.F_FULLFSYNC)
            else:
                os.fsync(sink.fileno())
            before_blocks = os.fstat(source.fileno()).st_blocks
            # Full chunks are allocation-aligned; punching the short final extent is safe
            # because no unread bytes follow it. APFS may retain its final partial block.
            before_free = shutil.disk_usage(archive.parent).free
            _punch_raw_hole(source.fileno(), offset, len(chunk))
            freed = shutil.disk_usage(archive.parent).free - before_free
            after_blocks = os.fstat(source.fileno()).st_blocks
            if after_blocks >= before_blocks:
                raise RuntimeError("punched raw extents not reclaimed (filesystem allocation pin)")
            # Bracket only the punch: unrelated allocations during compression must not
            # accumulate against reclaim. Allow half a chunk for concurrent volume activity.
            tolerance = len(chunk) // 2
            if freed < len(chunk) - tolerance:
                raise RuntimeError(
                    "punched raw extents not reclaimed (possible APFS snapshot pin or delayed volume reclaim)"
                )
            offset += len(chunk)
            _require_raw_floor(archive.parent)
    raw_hash = digest.hexdigest()
    with archive.open("rb") as compressed:
        if _archive_stream_sha256(compressed, zstd=zstd_binary is not None) != raw_hash:
            raise ValueError("Raw/archive snapshot binding failed (SHA-256)")
    return raw_hash


def create_sqlite_backup_artifact(
    db_path: Path,
    output_dir: Path,
    date_stamp: str | None = None,
    socket_path: Path | str | None = None,
    keep_uncompressed: bool = True,
    local_uncompressed_keep: int = DEFAULT_LOCAL_UNCOMPRESSED_KEEP,
    reclamation_status_callback: Callable[[str, str | None], None] | None = None,
    capacity_status_callback: Callable[[int, int | None, int, int], None] | None = None,
    integrity_full: bool | None = None,
    integrity_status_callback: Callable[[str, str, float], None] | None = None,
) -> SQLiteBackupArtifact:
    """Create an integrity-tested archive (.db.zst by default, .db.gz with
    BRAINLAYER_BACKUP_COMPRESSOR=gzip). Unsupported filesystems retain the legacy raw on failure;
    reclamation consumes this run's raw after integrity_check, so failure needs a fresh snapshot.

    The legacy keep_uncompressed/local_uncompressed_keep arguments are accepted for
    caller compatibility; they cannot authorize retaining duplicates or pruning only copies.
    """
    snapshot_started = time.monotonic()
    db_path = Path(db_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    date_stamp = date_stamp or _today()

    if not db_path.exists():
        raise FileNotFoundError(f"BrainLayer database not found: {db_path}")
    _refuse_foreign_db_on_production_socket(db_path, socket_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    if _parse_uncompressed_snapshot_date(f"{date_stamp}.db") is None or output_dir.is_symlink():
        raise ValueError("Backup requires a dated snapshot in a non-symlink backup directory")
    _cleanup_consumed_raw_remnants(output_dir)
    compressor, zstd_binary, compressor_fallback = _selected_compressor()
    reclaim_raw, reclaim_fallback = _probe_raw_reclamation(output_dir)
    print(
        f"Backup raw reclamation: enabled={reclaim_raw} "
        f"fallback={reclaim_fallback or 'none'}; {'streaming' if reclaim_raw else 'legacy'} peak budget",
        flush=True,
    )
    deleted, retained_raw = _cleanup_raw_snapshots(output_dir)
    for name, size in _owned_temp_inventory(output_dir):
        print(f"Backup crash remnant: name={name} bytes={size}; retained until a newer backup verifies", flush=True)
    writer_probe_error: str | None = None
    try:
        writer_started_at = _brainbar_writer_started_at(socket_path)
    except BackupTimeoutError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        writer_started_at = None
        writer_probe_error = f"{type(exc).__name__}: {_redact_upload_secrets(str(exc))}"
        print(f"BrainBar attempt reclamation degraded: {writer_probe_error}", flush=True)
    attempt_reclamation = "armed" if writer_started_at is not None else "degraded"
    if reclamation_status_callback is not None:
        reclamation_status_callback(attempt_reclamation, writer_probe_error)
    stale_attempts_deleted, surviving_attempt_paths = _sweep_stale_backup_attempts(
        output_dir,
        writer_started_at=writer_started_at,
    )
    db_size = _database_logical_size_bytes(db_path)
    surviving_attempt_bytes = 0
    surviving_attempt_growth_reserve_bytes = 0
    for attempt_path in surviving_attempt_paths:
        try:
            attempt_size = attempt_path.stat().st_size
        except (FileNotFoundError, OSError):
            continue
        surviving_attempt_bytes += attempt_size
        surviving_attempt_growth_reserve_bytes += max(0, db_size - attempt_size)
    estimated_gzip = _estimated_gzip_bytes(output_dir, db_size)
    # Only proven duplicates were reclaimed above; retained raws stay in measured free space and
    # are never budgeted as forthcoming room.
    full_verify = _should_run_full_verify(date_stamp)
    phase_peaks = _backup_phase_peaks(db_size, estimated_gzip, full=full_verify, reclaim_raw=reclaim_raw)

    def describe_phases(peaks: dict[str, int]) -> tuple[int, str]:
        phase = max(peaks, key=peaks.__getitem__)
        detail = f"peak_phase={phase} peak={peaks[phase]} phases={','.join(f'{n}={s}' for n, s in peaks.items())}"
        return peaks[phase], detail

    _, capacity_detail = describe_phases(phase_peaks)
    required_bytes = (
        _peak_backup_required_bytes(db_size, estimated_gzip, full=full_verify, reclaim_raw=reclaim_raw)
        + surviving_attempt_growth_reserve_bytes
    )
    raw_before_purge = shutil.disk_usage(output_dir).free
    try:
        important_free_bytes = _important_usage_capacity_bytes(output_dir)
    except (OSError, RuntimeError, ValueError):
        important_free_bytes = None
    purge_error: str | None = None
    raw_free_bytes = raw_before_purge
    if (
        raw_before_purge < required_bytes
        and important_free_bytes is not None
        and important_free_bytes >= required_bytes
    ):
        request_bytes = required_bytes - raw_before_purge + COPY_CHUNK_BYTES
        try:
            _request_macos_purge(output_dir, request_bytes)
        except RuntimeError as exc:
            purge_error = _redact_upload_secrets(str(exc))
        raw_free_bytes = shutil.disk_usage(output_dir).free
        for _ in range(60):
            if raw_free_bytes >= required_bytes:
                break
            time.sleep(POST_PURGE_POLL_SECONDS)
            raw_free_bytes = shutil.disk_usage(output_dir).free
    free_bytes = raw_free_bytes
    if capacity_status_callback is not None:
        capacity_status_callback(raw_free_bytes, important_free_bytes, free_bytes, required_bytes)
    print(
        f"Backup capacity: raw_before={raw_before_purge} raw_after={raw_free_bytes} "
        f"important_usage={important_free_bytes} selected={free_bytes} required={required_bytes} "
        f"{capacity_detail} purge_error={purge_error}",
        flush=True,
    )
    if raw_free_bytes < required_bytes:
        reason = purge_error or (
            "important-usage capacity unavailable or below required"
            if important_free_bytes is None or important_free_bytes < required_bytes
            else "macOS reclaimed too little purgeable space"
        )
        raise RuntimeError(
            f"Insufficient free space for backup in {output_dir}: "
            f"raw_before={raw_before_purge} raw_after={raw_free_bytes} "
            f"important_usage={important_free_bytes} required={required_bytes} floor={MIN_RAW_FREE_BYTES} "
            f"{capacity_detail}; {reason}; "
            f"{len(surviving_attempt_paths)} recent attempts reserve "
            f"{surviving_attempt_growth_reserve_bytes} growth bytes; retained raw snapshots: {retained_raw}"
        )

    def require_retry_capacity(unconfirmed: list[Path]) -> None:
        growth = 0
        for path in unconfirmed:
            try:
                growth += max(0, db_size - path.stat().st_size)
            except OSError:
                growth += db_size  # Not visible yet: the request may still write all of D.
        peak, detail = describe_phases({**phase_peaks, "raw_copy": growth + db_size})
        retry_required = peak + MIN_RAW_FREE_BYTES + COPY_CHUNK_BYTES + surviving_attempt_growth_reserve_bytes
        raw = shutil.disk_usage(output_dir).free
        if raw < retry_required:
            raise RuntimeError(
                f"Insufficient free space for backup retry in {output_dir}: raw={raw} required={retry_required} "
                f"floor={MIN_RAW_FREE_BYTES} {detail}; "
                f"{len(unconfirmed)} unconfirmed attempts reserve {growth} growth bytes"
            )

    final_archive = output_dir / f"{date_stamp}{ZSTD_SNAPSHOT_SUFFIX if compressor == 'zstd' else GZIP_SNAPSHOT_SUFFIX}"
    final_raw = output_dir / f"{date_stamp}.db"
    sentinel_chunks = 0
    if final_raw.exists() or final_raw.is_symlink():
        raise FileExistsError(f"Retaining existing raw snapshot without a verified archive: {final_raw}")

    # A raw this run promoted into non-canonical (temp/test/harness) staging is removed if the run
    # exits before verification: nothing else ever reclaims it there (2026-10-19, 19 GiB in $TMPDIR).
    # Ownership is the promoted dev/inode, never the dated name: a peer's raw is never touched.
    orphan_cleanup = not _is_canonical_staging(output_dir)
    raw_settled = False
    owned_raw: tuple[int, int] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="brainlayer-backup-", dir=output_dir) as tmp:
            (Path(tmp) / ".backup-date").write_text(date_stamp, encoding="ascii")
            raw_snapshot = final_raw
            snapshot_created_at = dt.datetime.now(dt.UTC).isoformat()
            promoted = request_brainbar_vacuum_into(
                raw_snapshot,
                socket_path=socket_path,
                attempt_dir=output_dir,
                retry_capacity_check=require_retry_capacity,
                expected_db_path=db_path,
            )
            if isinstance(promoted, os.stat_result):
                owned_raw = (promoted.st_dev, promoted.st_ino)
            _require_raw_floor(output_dir)
            sentinel_chunks = _validate_backup_target(raw_snapshot)
            pragma = (
                "integrity_check"
                if reclaim_raw
                or (integrity_full if integrity_full is not None else _should_run_full_verify(date_stamp))
                else "quick_check"
            )
            integrity_started = time.monotonic()
            try:
                integrity = _sqlite_pragma_check(raw_snapshot, pragma)
            except sqlite3.Error as exc:
                integrity = f"{type(exc).__name__}: {exc}"
            integrity_duration = time.monotonic() - integrity_started
            if integrity_status_callback is not None:
                integrity_status_callback(integrity, pragma, integrity_duration)
            # Past this point a raw is either verified or retained on purpose; the existing
            # retained-raw rules own it, never the orphan cleanup below.
            raw_settled = True
            if integrity != "ok":
                raise RuntimeError(f"Raw snapshot {pragma} failed: {integrity}; retaining raw snapshot")

            temp_archive = Path(tmp) / final_archive.name
            compress_started = time.monotonic()
            raw_sha256 = None
            if reclaim_raw:
                # Keep consumed extents inside the owned attempt, never at the published raw path.
                consumed_raw = _consume_raw_in_attempt(raw_snapshot, Path(tmp))
                print("Backup consuming verified raw snapshot; a failed attempt requires a fresh snapshot", flush=True)
                raw_sha256 = _compress_reclaiming_raw(consumed_raw, temp_archive, zstd_binary=zstd_binary)
            elif zstd_binary is not None:
                _compress_zstd(zstd_binary, raw_snapshot, temp_archive, output_dir)
            else:
                with raw_snapshot.open("rb") as src, gzip.open(temp_archive, "wb", compresslevel=6) as dst:
                    _copy_with_raw_floor(src, dst, output_dir)
            compress_duration = time.monotonic() - compress_started
            shutil.move(str(temp_archive), final_archive)
            if reclaim_raw:
                directory = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            print(f"Backup compressed: compressor={compressor} seconds={compress_duration:.1f}", flush=True)

            if not reclaim_raw:
                _remove_raw_with_verified_archive(output_dir, final_raw, own_run=True, archive_name=final_archive.name)
            deleted.append(final_raw.name)
    except BaseException:
        if orphan_cleanup and not raw_settled and owned_raw is not None:
            _remove_owned_raw(final_raw, owned_raw)
        raise

    return SQLiteBackupArtifact(
        gzip_path=final_archive,
        uncompressed_path=None,
        sentinel_chunks=sentinel_chunks,
        local_retention_deleted=deleted,
        stale_attempts_deleted=stale_attempts_deleted,
        surviving_attempts=[path.name for path in surviving_attempt_paths],
        surviving_attempt_bytes=surviving_attempt_bytes,
        surviving_attempt_growth_reserve_bytes=surviving_attempt_growth_reserve_bytes,
        attempt_reclamation=attempt_reclamation,
        writer_probe_error=writer_probe_error,
        snapshot_created_at=snapshot_created_at,
        local_integrity=integrity,
        local_integrity_pragma=pragma,
        snapshot_duration_seconds=time.monotonic() - snapshot_started,
        integrity_duration_seconds=integrity_duration,
        compressor=compressor,
        compressor_fallback=compressor_fallback,
        compress_duration_seconds=compress_duration,
        raw_sha256=raw_sha256,
    )


def create_sqlite_backup_gzip(
    db_path: Path,
    output_dir: Path,
    date_stamp: str | None = None,
    socket_path: Path | str | None = None,
) -> Path:
    artifact = create_sqlite_backup_artifact(db_path, output_dir, date_stamp=date_stamp, socket_path=socket_path)
    return artifact.gzip_path


def _brainbar_socket_path(socket_path: Path | str | None = None) -> Path:
    if socket_path is not None:
        return Path(socket_path).expanduser()
    return Path(os.environ.get("BRAINBAR_SOCKET_PATH", DEFAULT_BRAINBAR_SOCKET_PATH)).expanduser()


def _completed_attempt_ownership(path: Path) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Accept only a writer receipt binding this attempt name to its created raw inode."""
    marker = _backup_attempt_completion_marker(path)
    try:
        descriptor = os.open(marker, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            marker_info = os.fstat(stream.fileno())
            if not stat.S_ISREG(marker_info.st_mode) or marker_info.st_size > 4096:
                return None
            receipt = json.loads(stream.read(4097))
        raw_info = path.lstat()
        if (
            marker.is_symlink()
            or not stat.S_ISREG(raw_info.st_mode)
            or not isinstance(receipt, dict)
            or type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] != 1
            or receipt.get("target_path") != os.path.abspath(path)
            or any(type(receipt.get(key)) is not int for key in ("target_dev", "target_ino"))
            or (receipt["target_dev"], receipt["target_ino"]) != (raw_info.st_dev, raw_info.st_ino)
        ):
            return None
        return (raw_info.st_dev, raw_info.st_ino), (marker_info.st_dev, marker_info.st_ino)
    except (OSError, ValueError, UnicodeError):
        return None


def request_brainbar_vacuum_into(
    target_path: Path,
    socket_path: Path | str | None = None,
    timeout_seconds: int | None = None,
    max_attempts: int = 3,
    retry_backoff_seconds: int = 60,
    attempt_dir: Path | None = None,
    retry_capacity_check: Callable[[list[Path]], None] | None = None,
    expected_db_path: Path | str | None = None,
) -> os.stat_result:
    """Ask BrainBar to VACUUM INTO an attempt path, then promote it to target_path.

    Promotion is create-exclusive and returns the stat of the promoted file, which is how a
    caller proves later that a raw at target_path is still the one this run created.

    With expected_db_path, BrainBar must serve exactly that database: it refuses a mismatch
    before any VACUUM, and a response that does not confirm the same file (an older BrainBar
    ignores the argument) is deleted and refused. Identity refusals never retry.
    """
    target_path = Path(target_path).expanduser()
    resolved_expected_db = (
        os.path.realpath(Path(expected_db_path).expanduser()) if expected_db_path is not None else None
    )
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if target_path.exists():
        raise FileExistsError(f"Backup target already exists: {target_path}")
    resolved_socket_path = _brainbar_socket_path(socket_path)
    resolved_timeout_seconds: int | None = (
        _configured_backup_client_timeout_seconds() if timeout_seconds is None else timeout_seconds
    )
    if resolved_timeout_seconds is not None and resolved_timeout_seconds < 1:
        raise ValueError("timeout_seconds must be at least 1 or None")
    resolved_attempt_dir = Path(attempt_dir).expanduser() if attempt_dir is not None else target_path.parent
    resolved_attempt_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    unconfirmed_attempt_paths: list[Path] = []
    completed_attempts: dict[Path, tuple[tuple[int, int], tuple[int, int]]] = {}
    for attempt in range(1, max_attempts + 1):
        for prior_path in unconfirmed_attempt_paths:
            if prior_path in completed_attempts:
                continue
            ownership = _completed_attempt_ownership(prior_path)
            if ownership is not None:
                completed_attempts[prior_path] = ownership
            else:
                print(f"Backup retaining attempt without writer-bound ownership: {prior_path}", flush=True)
        if attempt > 1 and retry_capacity_check is not None:
            # Re-measure before every retry: an unresolved request may still grow to full size
            # beside the next one, even if its target has not appeared yet.
            retry_capacity_check(list(unconfirmed_attempt_paths))
        attempt_path = resolved_attempt_dir / f".{target_path.name}.attempt-{attempt}-{uuid.uuid4().hex}"
        request = {
            "jsonrpc": "2.0",
            "id": attempt,
            "method": "tools/call",
            "params": {
                "name": "brain_backup_vacuum_into",
                "arguments": {
                    "target_path": str(attempt_path),
                    **({"expected_db_path": resolved_expected_db} if resolved_expected_db else {}),
                },
            },
        }
        terminal_response_received = False
        try:
            response = _send_brainbar_json_request(
                resolved_socket_path,
                request,
                timeout_seconds=resolved_timeout_seconds,
            )
            if _is_brainbar_snapshot_busy(response):
                # Rejected before admission, so this request never wrote, and the writer it
                # names may be one of ours. Release nothing: no response proves it stopped.
                if unconfirmed_attempt_paths:
                    preserved = ", ".join(str(path) for path in unconfirmed_attempt_paths)
                    raise BrainBarSnapshotBusyError(
                        f"{BRAINBAR_SNAPSHOT_BUSY_MESSAGE} Not retrying beside an unresolved earlier request; "
                        f"preserving {len(unconfirmed_attempt_paths)} attempt targets and their reservations: "
                        f"{preserved}"
                    )
                raise BrainBarSnapshotBusyError(BRAINBAR_SNAPSHOT_BUSY_MESSAGE)
            terminal_response_received = True
            if response.get("error"):
                raise RuntimeError(f"BrainBar backup request failed: {response['error']}")
            result = response.get("result") or {}
            content = result.get("content") or []
            text = content[0].get("text") if content and isinstance(content[0], dict) else None
            if result.get("isError"):
                if isinstance(text, str) and BRAINBAR_DB_IDENTITY_MISMATCH in text:
                    raise BrainBarDatabaseIdentityError(f"BrainBar refused the backup before VACUUM: {text}")
                raise RuntimeError(f"BrainBar backup request failed: {text if text is not None else result}")
            if resolved_expected_db is not None:
                # BrainBar echoes the dev/inode of the file its connection has OPEN; a pathname
                # cannot prove that (the path may have been replaced under the live connection).
                try:
                    echo = json.loads(text) if isinstance(text, str) else None
                except ValueError:
                    echo = None
                echo = echo if isinstance(echo, dict) else {}
                served_identity = (echo.get("db_dev"), echo.get("db_ino"))
                expected_identity = _file_identity(resolved_expected_db)
                if not all(isinstance(v, int) for v in served_identity) or served_identity != expected_identity:
                    # Terminal response: BrainBar has stopped writing this attempt. Whatever it
                    # vacuumed is not the configured database, so it must not survive here.
                    attempt_path.unlink(missing_ok=True)
                    _backup_attempt_completion_marker(attempt_path).unlink(missing_ok=True)
                    raise BrainBarDatabaseIdentityError(
                        f"{BRAINBAR_DB_IDENTITY_MISMATCH}: expected {resolved_expected_db} {expected_identity}, "
                        f"BrainBar reported {echo.get('db_path')!r} {served_identity}; deleted the unconfirmed "
                        "snapshot. A BrainBar without open-database identity support must be upgraded."
                    )
            if not attempt_path.exists():
                raise RuntimeError(f"BrainBar backup did not create snapshot: {attempt_path}")
            try:
                _validate_backup_target(attempt_path)
            except Exception:
                # A tool response is terminal, so this attempt is no longer being written.
                attempt_path.unlink(missing_ok=True)
                _backup_attempt_completion_marker(attempt_path).unlink(missing_ok=True)
                raise
            promoted = os.stat(attempt_path)
            try:
                # Create-exclusive: a peer's dated raw that appeared meanwhile is never replaced.
                os.link(attempt_path, target_path)
            except FileExistsError:
                attempt_path.unlink(missing_ok=True)
                _backup_attempt_completion_marker(attempt_path).unlink(missing_ok=True)
                raise BackupTargetConflictError(
                    f"Backup target appeared while this snapshot ran; not replacing it: {target_path}"
                ) from None
            attempt_path.unlink()
            _backup_attempt_completion_marker(attempt_path).unlink(missing_ok=True)
            for prior_path in unconfirmed_attempt_paths:
                prior_path.unlink(missing_ok=True)
                _backup_attempt_completion_marker(prior_path).unlink(missing_ok=True)
            return promoted
        except BrainBarDatabaseIdentityError:
            for prior_path, (owned_raw, owned_marker) in completed_attempts.items():
                _remove_owned_raw(prior_path, owned_raw)
                if not prior_path.exists() and not prior_path.is_symlink():
                    _remove_owned_raw(_backup_attempt_completion_marker(prior_path), owned_marker)
            raise
        except (BackupTimeoutError, BackupTargetConflictError):
            raise
        except BrainBarSnapshotBusyError as exc:
            if unconfirmed_attempt_paths:
                raise
            # Someone else's snapshot is running and this run owns no writer: back off and retry.
            last_error = exc
            existing_target_note = ""
        except Exception as exc:
            last_error = exc
            existing_target_note = ""
            if terminal_response_received:
                # BrainBar admits one snapshot at a time and answers only after it finishes, so an
                # admitted request's terminal response proves it and every earlier request stopped.
                for completed_path in [*unconfirmed_attempt_paths, attempt_path]:
                    completed_path.unlink(missing_ok=True)
                    _backup_attempt_completion_marker(completed_path).unlink(missing_ok=True)
                unconfirmed_attempt_paths.clear()
            else:
                # A lost response does not prove VACUUM INTO is finished, or even started:
                # BrainBar keeps working after a disconnect. Never inspect, unlink, or promote
                # a path that BrainBar may still be writing, and keep its capacity reserved.
                unconfirmed_attempt_paths.append(attempt_path)
                if attempt_path.exists():
                    existing_target_note = f"; preserving isolated attempt target: {attempt_path}"
        failure = _redact_upload_secrets(str(last_error))
        if attempt >= max_attempts:
            print(
                f"BrainBar vacuum snapshot attempt {attempt}/{max_attempts} failed: {failure}{existing_target_note}",
                flush=True,
            )
            break
        print(
            f"BrainBar vacuum snapshot attempt {attempt}/{max_attempts} failed: {failure}{existing_target_note}; "
            f"retrying in {retry_backoff_seconds}s",
            flush=True,
        )
        _sleep(retry_backoff_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("BrainBar vacuum snapshot made no attempt")


def _send_brainbar_json_request(
    socket_path: Path,
    request: dict[str, Any],
    timeout_seconds: int | None,
) -> dict[str, Any]:
    payload = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout_seconds)
        refuse_production_brainbar_socket(socket_path)
        client.connect(str(socket_path))
        client.sendall(payload)
        data = b""
        while not data.endswith(b"\n"):
            chunk = client.recv(65_536)
            if not chunk:
                break
            data += chunk
    if not data:
        raise RuntimeError(f"BrainBar socket closed without response: {socket_path}")
    return json.loads(data.decode("utf-8"))


def get_drive_credentials(token_path: Path | None = None, client_path: Path | None = None):
    """Use BrainLayer-owned credentials for both database and JSONL backups."""
    return drive_credentials.load_credentials(token_path=token_path, client_path=client_path)


def build_drive_service(token_path: Path | None = None, client_path: Path | None = None):
    from googleapiclient.discovery import build

    return build("drive", "v3", credentials=get_drive_credentials(token_path, client_path))


def ensure_drive_folder(service: Any, name: str, parent_id: str | None = None) -> str:
    escaped = _escape_drive_query_value(name)
    clauses = [
        f"name = '{escaped}'",
        f"mimeType = '{DRIVE_FOLDER_MIME}'",
        "trashed = false",
    ]
    if parent_id:
        clauses.append(f"'{parent_id}' in parents")
    query = " and ".join(clauses)

    result = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id,name)", pageSize=10, supportsAllDrives=True)
        .execute()
    )
    files = result.get("files", [])
    if files:
        return files[0]["id"]

    metadata: dict[str, Any] = {"name": name, "mimeType": DRIVE_FOLDER_MIME}
    if parent_id:
        metadata["parents"] = [parent_id]
    created = service.files().create(body=metadata, fields="id", supportsAllDrives=True).execute()
    return created["id"]


def ensure_drive_folder_chain(service: Any, folder_parts: list[str]) -> str:
    parent_id = None
    for part in folder_parts:
        parent_id = ensure_drive_folder(service, part, parent_id)
    if parent_id is None:
        raise ValueError("folder_parts must not be empty")
    return parent_id


def _machine_ids(machine_id: str) -> set[str]:
    resolved = _validate_machine_id(machine_id)
    return {resolved, *PRIOR_MACHINE_IDS.get(resolved, ())}


def _find_drive_folder_ids(service: Any, folder_parts: list[str], *, parent_id: str | None = None) -> list[str]:
    """Resolve existing chains only, including duplicate names; never create one."""
    parents: list[str | None] = [parent_id]
    for name in folder_parts:
        found: list[str] = []
        for parent in parents:
            query = (
                f"name = '{_escape_drive_query_value(name)}' and mimeType = '{DRIVE_FOLDER_MIME}' and trashed = false"
            )
            if parent is not None:
                query += f" and '{_escape_drive_query_value(parent)}' in parents"
            page_token = None
            while True:
                result = (
                    service.files()
                    .list(
                        q=query,
                        spaces="drive",
                        fields="nextPageToken,files(id)",
                        pageSize=1000,
                        pageToken=page_token,
                        supportsAllDrives=True,
                    )
                    .execute()
                )
                found.extend(item["id"] for item in result.get("files", []))
                page_token = result.get("nextPageToken")
                if not page_token:
                    break
        parents = list(dict.fromkeys(found))
    return [parent for parent in parents if parent is not None]


def _list_drive_backup_files(service: Any, folder_id: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    page_token = None
    while True:
        result = (
            service.files()
            .list(
                q=f"'{_escape_drive_query_value(folder_id)}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken,files(id,name,appProperties)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
            )
            .execute()
        )
        files.extend(result.get("files", []))
        page_token = result.get("nextPageToken")
        if not page_token:
            return files


def list_machine_drive_backups(service: Any, *, machine_id: str) -> list[dict[str, Any]]:
    """Read-only recovery/retention inventory across this machine's known IDs."""
    inventory: dict[str, dict[str, Any]] = {}
    for owner in sorted(_machine_ids(machine_id)):
        for folder_id in _find_drive_folder_ids(service, default_drive_folder_parts(owner)):
            files = _list_drive_backup_files(service, folder_id)
            _assert_backup_owners(files, folder_id=folder_id, machine_id=machine_id)
            inventory.update((item["id"], item) for item in files if _parse_snapshot_date(item.get("name", "")))
    return list(inventory.values())


def _list_machine_retention_backups(service: Any, *, folder_id: str, machine_id: str) -> list[dict[str, Any]]:
    """Bind mutation to the uploaded folder and known prior-ID siblings only."""
    folder_ids = [folder_id]
    prior_ids = _machine_ids(machine_id) - {_validate_machine_id(machine_id)}
    if prior_ids:
        metadata = _execute_drive_read(
            service.files().get(fileId=folder_id, fields="parents", supportsAllDrives=True),
            what="Drive retention parent lookup",
        )
        parents = metadata.get("parents", [])
        if not isinstance(parents, list) or len(parents) != 1 or not isinstance(parents[0], str) or not parents[0]:
            raise ValueError("Cannot establish an unambiguous Drive retention parent")
        for prior_id in sorted(prior_ids):
            folder_ids.extend(
                _find_drive_folder_ids(service, [default_drive_folder_parts(prior_id)[-1]], parent_id=parents[0])
            )
    inventory: dict[str, dict[str, Any]] = {}
    for candidate in dict.fromkeys(folder_ids):
        files = _list_drive_backup_files(service, candidate)
        _assert_backup_owners(files, folder_id=candidate, machine_id=machine_id)
        inventory.update((item["id"], item) for item in files if _parse_snapshot_date(item.get("name", "")))
    return list(inventory.values())


def assert_drive_folder_owned_by_machine(service: Any, *, folder_id: str, machine_id: str) -> None:
    """Refuse a target folder containing snapshots owned by another host."""
    _assert_backup_owners(_list_drive_backup_files(service, folder_id), folder_id=folder_id, machine_id=machine_id)


def _assert_backup_owners(files: list[dict[str, Any]], *, folder_id: str, machine_id: str) -> None:
    other_machine_ids: set[str] = set()
    for item in files:
        if _parse_snapshot_date(item.get("name", "")) is None:
            continue
        properties = item.get("appProperties")
        owner = properties.get(DRIVE_MACHINE_PROPERTY) if isinstance(properties, dict) else None
        # Untagged legacy snapshots belong only to the canonical M4.
        resolved_owner = owner if isinstance(owner, str) and owner else CANONICAL_MACHINE_ID
        if resolved_owner not in _machine_ids(machine_id):
            other_machine_ids.add(resolved_owner)
    if other_machine_ids:
        raise DriveFolderOwnedByOtherMachineError(
            folder_id=folder_id,
            machine_id=machine_id,
            other_machine_ids=other_machine_ids,
        )


class RetainedStalledBackupCapacityError(RuntimeError):
    error_code = "retained_stalled_backup_capacity"

    def __init__(self, names: list[str], *, reason: str | None = None):
        self.error_type = f"RetainedStalledBackupCapacityError[{','.join(names)}]"
        detail = f"; {reason}" if reason else ""
        super().__init__(f"Insufficient capacity; retaining stalled backups: {', '.join(names)}{detail}")


class RetainedStalledBackupConflictError(RuntimeError):
    error_code = "retained_stalled_backup_conflict"

    def __init__(self, name: str):
        self.error_type = f"RetainedStalledBackupConflictError[{name}]"
        super().__init__(f"Same-day snapshot collision; retaining stalled backup {name}")


class DriveSessionExpiredError(RuntimeError):
    """Drive no longer recognizes the saved upload session (404/410)."""


class _DriveRetryableHTTPError(RuntimeError):
    pass


def upload_file_to_drive_raw(
    file_path: Path,
    folder_id: str,
    credentials: Any,
    *,
    machine_id: str,
    chunk_size: int = 8 * 1024 * 1024,
    max_attempts: int = 30,
    resume_session_url: str | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
    run_deadline: float | None = None,
) -> dict[str, Any]:
    """Upload unchanged bytes; probe ambiguous writes before retransmitting them."""
    file_path = Path(file_path)
    total = file_path.stat().st_size
    upload_url = resume_session_url

    deadline_floor = _configured_positive_number(
        DRIVE_UPLOAD_DEADLINE_FLOOR_ENV,
        DEFAULT_DRIVE_UPLOAD_DEADLINE_FLOOR_SECONDS,
        maximum=threading.TIMEOUT_MAX,
    )
    minimum_rate = _configured_positive_number(
        DRIVE_UPLOAD_MIN_BYTES_PER_SECOND_ENV, DEFAULT_DRIVE_UPLOAD_MIN_BYTES_PER_SECOND
    )
    budget = _configured_positive_number(DRIVE_UPLOAD_RETRY_BUDGET_ENV, 20 * 60)
    run_deadline = run_deadline if run_deadline is not None else time.monotonic() + _configured_backup_timeout_seconds()

    def request_deadline(desired: float) -> float:
        remaining = run_deadline - time.monotonic() - 5  # reserve receipt flush/fsync before supervisor/ALRM
        if remaining <= 0:
            raise BackupTimeoutError("Drive recovery stopped before run deadline to persist receipt")
        return min(desired, remaining)

    # Preserve the explicit operator attempt cap, without changing metadata-read retries.
    stall_limit = (
        _drive_upload_stall_max_attempts() if os.environ.get(DRIVE_UPLOAD_STALL_MAX_ATTEMPTS_ENV) else max_attempts
    )
    sent = stalled_attempts = http_attempts = 0
    slept = 0.0
    stall_started: float | None = None
    probe = resume_session_url is not None
    recovering = False
    session = requests.Session()

    def backoff() -> None:
        nonlocal slept
        elapsed = max(slept, time.monotonic() - stall_started) if stall_started is not None else slept
        if stalled_attempts >= stall_limit or elapsed >= budget:
            raise DriveUploadStalledError(
                bytes_confirmed=sent, total_bytes=total, attempts=stalled_attempts, session_url=upload_url
            )
        delay = min(300, 30 * 2 ** min(max(0, stalled_attempts - 1), 4), budget - elapsed, request_deadline(budget))
        print(
            f"drive upload stalled; confirmed={sent}/{total} attempt={stalled_attempts}/{stall_limit}; "
            f"sleeping {delay}s",
            flush=True,
        )
        _sleep(delay)
        slept += delay

    try:
        while upload_url is None:
            try:
                init = requests.post(
                    "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&supportsAllDrives=true"
                    "&fields=id,name,size,md5Checksum,appProperties",
                    headers={
                        "Authorization": f"Bearer {credentials.token}",
                        "Content-Type": "application/json; charset=UTF-8",
                        "X-Upload-Content-Type": (
                            "application/zstd" if file_path.name.endswith(".zst") else "application/gzip"
                        ),
                        "X-Upload-Content-Length": str(total),
                    },
                    data=json.dumps(
                        {
                            "name": file_path.name,
                            "parents": [folder_id],
                            "appProperties": {DRIVE_MACHINE_PROPERTY: _validate_machine_id(machine_id)},
                        }
                    ),
                    timeout=request_deadline(60),
                )
                init.raise_for_status()
                upload_url = init.headers["Location"]
            except BackupTimeoutError:
                raise
            except (*_DRIVE_TRANSIENT_NETWORK_ERRORS, requests.Timeout):
                stall_started = time.monotonic() if stall_started is None else stall_started
                stalled_attempts += 1
                backoff()
        stalled_attempts, slept, stall_started = 0, 0.0, None
        if progress_callback is not None:
            progress_callback(upload_url, sent)
        with file_path.open("rb") as handle:
            while sent < total or probe:
                is_probe = probe
                if is_probe:
                    chunk = b""
                    content_range = f"bytes */{total}"
                else:
                    handle.seek(sent)
                    expected = min(chunk_size, total - sent)
                    chunk = handle.read(expected)
                    if len(chunk) != expected:
                        raise RuntimeError("Backup file changed during upload")
                    content_range = f"bytes {sent}-{sent + len(chunk) - 1}/{total}"
                try:
                    response = _drive_put_with_deadline(
                        session,
                        upload_url,
                        headers={
                            "Authorization": f"Bearer {credentials.token}",
                            "Content-Length": str(len(chunk)),
                            "Content-Range": content_range,
                        },
                        data=chunk,
                        deadline=request_deadline(max(deadline_floor, len(chunk) / minimum_rate)),
                    )
                    if response.status_code in {200, 201}:
                        return response.json()
                    if is_probe and response.status_code in {404, 410}:
                        raise DriveSessionExpiredError("Drive upload session expired")
                    if response.status_code in {429, 500, 502, 503, 504}:
                        raise _DriveRetryableHTTPError(f"Drive HTTP {response.status_code}")
                    response.raise_for_status()  # Auth/consent and other 4xx fail immediately.
                    if response.status_code != 308:
                        raise RuntimeError(f"Unexpected Drive status {response.status_code}")
                    uploaded_range = response.headers.get("Range")
                    if not uploaded_range and sent > 0:
                        raise _DriveRequestDeadlineExceeded(
                            "Drive omitted Range after a confirmed prefix; query session"
                        )
                    confirmed = int(uploaded_range.rsplit("-", 1)[1]) + 1 if uploaded_range else 0
                    if confirmed < sent or confirmed > total:
                        raise RuntimeError(
                            f"Drive returned invalid confirmed offset {confirmed}; expected {sent}..{total}"
                        )
                    http_attempts = 0
                    if confirmed > sent:
                        sent = confirmed
                        if progress_callback is not None:
                            progress_callback(upload_url, sent)
                        stalled_attempts, slept, stall_started = 0, 0.0, None
                        print(f"drive upload progress: {sent}/{total} bytes", flush=True)
                    elif not is_probe or confirmed == total:
                        raise _DriveRequestDeadlineExceeded("Drive upload returned no newly confirmed bytes")
                    elif recovering:
                        backoff()
                    probe, recovering = confirmed == total, False
                except BackupTimeoutError:
                    raise
                except (_DriveRequestDeadlineExceeded, *_DRIVE_TRANSIENT_NETWORK_ERRORS):
                    stall_started = time.monotonic() if stall_started is None else stall_started
                    stalled_attempts += 1
                    recovering = True
                    session.close()
                    session = requests.Session()
                    if is_probe:
                        backoff()
                    probe = True
                except _DriveRetryableHTTPError:
                    http_attempts += 1
                    if http_attempts >= max_attempts:
                        raise
                    _sleep(min(60, 2 ** min(http_attempts, 6), request_deadline(60)))
    except BaseException as exc:
        _redact_upload_exception(exc)
        raise
    finally:
        session.close()
    raise RuntimeError("Drive upload ended without final response")


def download_drive_file_raw(service: Any, *, file_id: str, destination: Path) -> Path:
    """Download a Drive artifact to a local path for restore verification."""
    from googleapiclient.http import MediaIoBaseDownload

    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with destination.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _require_raw_floor(destination.parent, forthcoming=8 * 1024 * 1024)
            _, done = downloader.next_chunk()
            _require_raw_floor(destination.parent)
    return destination


DRIVE_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


def _download_drive_to_stream(service: Any, *, file_id: str, sink: Any) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    downloader = MediaIoBaseDownload(sink, request, chunksize=DRIVE_DOWNLOAD_CHUNK_BYTES)
    done = False
    while not done:
        _, done = _retry_transient_drive_call(downloader.next_chunk, what="Drive snapshot download")


def _drive_stream_hashes(service: Any, *, file_id: str, zstd: bool) -> tuple[str, str]:
    """A bounded pipe carries the download directly into decompression, never a disk file."""
    read_fd, write_fd = os.pipe()
    md5 = hashlib.md5()
    errors: list[BaseException] = []

    def download() -> None:
        try:
            with os.fdopen(write_fd, "wb") as writer:

                class Sink:
                    def write(self, data: bytes) -> int:
                        written = writer.write(data)
                        md5.update(data[:written])
                        return written

                _download_drive_to_stream(service, file_id=file_id, sink=Sink())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=download, name="backup-drive-stream", daemon=True)
    thread.start()
    try:
        with os.fdopen(read_fd, "rb") as reader:
            raw_hash = _archive_stream_sha256(reader, zstd=zstd)
    finally:
        # Closing the reader also unblocks a writer if the decoder rejects the stream early.
        thread.join()
    if errors:
        raise errors[0]
    return raw_hash, md5.hexdigest()


def _parse_snapshot_date(
    name: str, *, prefix: str = "", suffix: str | tuple[str, ...] = DB_SNAPSHOT_SUFFIXES
) -> dt.date | None:
    if prefix and not name.startswith(prefix):
        return None
    matched = next(
        (candidate for candidate in ((suffix,) if isinstance(suffix, str) else suffix) if name.endswith(candidate)),
        None,
    )
    if matched is None:
        return None
    stem = name[len(prefix) : len(name) - len(matched)]
    try:
        return dt.date.fromisoformat(stem[:10])
    except ValueError:
        return None


def _verified_snapshot_names_from_log(log_path: Path, *, db_path: Path | None = None) -> set[str]:
    verified: set[str] = set()
    try:
        lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return verified
    latest = {}
    for line in lines:
        try:
            payload = json.loads(line)
            if isinstance(payload, dict) and isinstance(payload.get("snapshot"), str):
                latest[payload["snapshot"]] = payload
        except json.JSONDecodeError:
            continue
    for payload in latest.values():
        if db_path is not None and payload.get("db") != str(db_path):
            continue
        if payload.get("uploaded") is not True or payload.get("verified") is not True:
            continue
        snapshot = payload.get("snapshot")
        if isinstance(snapshot, str):
            name = Path(snapshot).name
            if _parse_snapshot_date(name) is not None:
                verified.add(name)
    return verified


def prune_local_compressed_snapshots(
    output_dir: Path,
    *,
    keep_latest: int = DEFAULT_LOCAL_COMPRESSED_KEEP,
    verified_drive_names: set[str] | None = None,
    protected_names: set[str] | None = None,
) -> list[str]:
    """Cap local .db.zst/.db.gz snapshots without deleting an archive lacking remote coverage.

    The newest Drive-verified archive of EACH format is always kept, so switching
    compressors (or rolling back) never deletes the last verified copy of either kind.
    """
    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")
    output_dir = Path(output_dir).expanduser()
    if not output_dir.exists():
        return []

    verified_names = verified_drive_names or set()
    dated: list[tuple[dt.date, Path]] = []
    for path in output_dir.iterdir():
        parsed = _parse_snapshot_date(path.name)
        if parsed and path.is_file():
            dated.append((parsed, path))

    dated.sort(key=lambda pair: (pair[0], pair[1].name), reverse=True)
    newest_verified_per_kind: dict[str, str] = {}
    for _, path in dated:
        kind = _snapshot_suffix(path.name)
        if kind is not None and path.name in verified_names:
            newest_verified_per_kind.setdefault(kind, path.name)
    deleted: list[str] = []
    for index, (snapshot_date, path) in enumerate(dated):
        if index < keep_latest or path.name in newest_verified_per_kind.values():
            continue
        if path.name in (protected_names if protected_names is not None else _protected_snapshot_names(output_dir)):
            continue
        newer_verified = sum(name > path.name for name in verified_names)
        if path.name not in verified_names and newer_verified < keep_latest:
            continue
        path.unlink(missing_ok=True)
        deleted.append(path.name)
    return deleted


# Pre-zstd name, kept for callers and tests that still import it.
prune_local_gzip_snapshots = prune_local_compressed_snapshots


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - backup restore verification needs MD5 parity with Drive tooling.
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gunzip_test(path: Path) -> None:
    subprocess.run(["gunzip", "-t", str(path)], check=True, capture_output=True, text=True)


def _decompress_gzip_to(gzip_path: Path, destination: Path) -> None:
    with gzip.open(gzip_path, "rb") as src, destination.open("wb") as dst:
        _copy_with_raw_floor(src, dst, destination.parent)


def _env_flag_enabled(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _drive_retention_enabled() -> bool:
    return DRIVE_RETENTION_ENABLED or os.environ.get(BACKUP_DRIVE_RETENTION_ENV, "").strip().lower() in {
        "1",
        "true",
    }


def _should_run_full_verify(date_stamp: str | None) -> bool:
    if _env_flag_enabled(BACKUP_FULL_VERIFY_ENV):
        return True
    try:
        backup_date = dt.date.fromisoformat((date_stamp or _today())[:10])
    except ValueError:
        return False
    return backup_date.weekday() == 6


def verify_sqlite_backup_artifact(
    artifact: SQLiteBackupArtifact,
    *,
    full: bool = False,
    service: Any | None = None,
    file_id: str | None = None,
) -> dict[str, Any]:
    """Run restore verification against the local archive (.db.zst or .db.gz) or the downloaded Drive copy."""
    mode = "full" if full else "quick"
    result: dict[str, Any] = {
        "verified": False,
        "verification_mode": mode,
        "sentinel_snapshot_chunks": artifact.sentinel_chunks,
        "sentinel_verified_chunks": None,
    }
    verify_path = artifact.gzip_path

    try:
        local_md5 = _md5_file(artifact.gzip_path)
        result["local_md5"] = local_md5
        if artifact.raw_sha256 is not None:
            # Byte equality transfers both integrity_check and COUNT(*) from the raw DB.
            if artifact.local_integrity != "ok" or artifact.local_integrity_pragma != "integrity_check":
                raise ValueError("Streaming verification requires the raw integrity_check proof")
            zstd = artifact.gzip_path.name.endswith(ZSTD_SNAPSHOT_SUFFIX)
            result["compression"] = "zstd" if zstd else "gzip"
            with artifact.gzip_path.open("rb") as source:
                raw_hash = _archive_stream_sha256(source, zstd=zstd)
            if raw_hash != artifact.raw_sha256:
                raise ValueError("Local archive SHA-256 differs from integrity-checked raw snapshot")
            if full:
                if service is None or not file_id:
                    raise ValueError("full verification requires Drive service and file_id")
                drive_hash, drive_md5 = _drive_stream_hashes(service, file_id=file_id, zstd=zstd)
                result.update(drive_md5=drive_md5, drive_md5_match=drive_md5 == local_md5)
                if drive_md5 != local_md5 or drive_hash != artifact.raw_sha256:
                    raise ValueError("Drive download md5 or raw SHA-256 mismatch")
            result.update(
                verified=True,
                archive_test=True,
                raw_sha256_match=True,
                sentinel_verified_chunks=artifact.sentinel_chunks,
                pragma="ok",
                verification_proof="stream_sha256_of_integrity_checked_raw",
            )
            if not zstd:
                result["gzip_test"] = True
            return result
        with tempfile.TemporaryDirectory(prefix="brainlayer-restore-verify-", dir=artifact.gzip_path.parent) as tmp:
            tmp_dir = Path(tmp)
            (tmp_dir / ".backup-date").write_text(artifact.gzip_path.name[:10], encoding="ascii")
            if full:
                if service is None or not file_id:
                    result["verification_error"] = "full verification requires Drive service and file_id"
                    return result
                drive_copy = tmp_dir / artifact.gzip_path.name
                download_drive_file_raw(service, file_id=file_id, destination=drive_copy)
                drive_md5 = _md5_file(drive_copy)
                result["drive_md5"] = drive_md5
                result["drive_md5_match"] = drive_md5 == local_md5
                if drive_md5 != local_md5:
                    result["verification_error"] = "Drive download md5 mismatch"
                    return result
                verify_path = drive_copy

            is_gzip = verify_path.name.endswith(GZIP_SNAPSHOT_SUFFIX)
            result["compression"] = "gzip" if is_gzip else "zstd"
            _archive_test(verify_path)
            result["archive_test"] = True
            if is_gzip:
                result["gzip_test"] = True  # Pre-zstd receipt field, kept for gzip archives.
            restored = tmp_dir / "restored.db"
            _decompress_archive_to(verify_path, restored)
            verified_chunks = _validate_backup_target(restored)
            result["sentinel_verified_chunks"] = verified_chunks
            if verified_chunks != artifact.sentinel_chunks:
                result["verification_error"] = (
                    f"sentinel mismatch: snapshot={artifact.sentinel_chunks} verified={verified_chunks}"
                )
                return result
            pragma_name = "integrity_check" if full else "quick_check"
            result["pragma"] = _optional_sqlite_pragma_check(restored, pragma_name)
            result["verified"] = True
    except Exception as exc:
        result.setdefault("archive_test", False)
        if artifact.gzip_path.name.endswith(GZIP_SNAPSHOT_SUFFIX):
            result.setdefault("gzip_test", False)
        result["verification_error"] = _redact_upload_secrets(str(exc))
    return result


def verify_drive_upload(
    service: Any,
    *,
    file_id: str,
    expected_name: str,
    expected_size: int,
    expected_machine_id: str,
) -> None:
    """Verify that Drive can see the uploaded file with the expected name and byte size."""
    metadata = _execute_drive_read(
        service.files().get(fileId=file_id, fields="id,name,size,trashed,appProperties", supportsAllDrives=True),
        what="Drive upload verification",
    )
    if metadata.get("trashed"):
        raise RuntimeError(f"Uploaded Drive backup is trashed: {file_id}")
    if metadata.get("name") != expected_name:
        raise RuntimeError(f"Uploaded Drive backup name mismatch: {metadata.get('name')!r} != {expected_name!r}")
    try:
        actual_size = int(metadata.get("size", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Uploaded Drive backup size is not numeric: {metadata.get('size')!r}") from exc
    if actual_size != expected_size:
        raise RuntimeError(f"Uploaded Drive backup size mismatch: {actual_size} != {expected_size}")
    properties = metadata.get("appProperties")
    actual_machine_id = properties.get(DRIVE_MACHINE_PROPERTY) if isinstance(properties, dict) else None
    if actual_machine_id != expected_machine_id:
        raise RuntimeError(f"Uploaded Drive backup machine mismatch: {actual_machine_id!r} != {expected_machine_id!r}")


def prune_drive_backups(
    service: Any,
    *,
    folder_parts: list[str] | None = None,
    machine_id: str | None = None,
    selected_folder_id: str | None = None,
    retention_policy: DriveRetentionPolicy = DAILY_RETENTION,
    verified_file_ids: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Trash Drive snapshots older than the explicitly enabled retention window.

    `verified_file_ids` is receipt evidence (see `_verified_drive_file_ids`) naming the
    exact Drive objects whose upload verified. For each format (.db.zst / .db.gz) the
    newest object in that set is never trashed; a format with no verified object in the
    listing has nothing trashed at all. No evidence (None or empty) trashes nothing.
    """
    if machine_id is not None and folder_parts is None:
        folder_id = selected_folder_id or ensure_drive_folder_chain(service, default_drive_folder_parts(machine_id))
        files = _list_machine_retention_backups(service, folder_id=folder_id, machine_id=machine_id)
    else:
        folder_id = ensure_drive_folder_chain(
            service, folder_parts if folder_parts is not None else DEFAULT_FOLDER_PARTS
        )
        files = _list_drive_backup_files(service, folder_id)
    dated = []
    for item in files:
        parsed = _parse_snapshot_date(
            item.get("name", ""),
            prefix=retention_policy.filename_prefix,
            suffix=retention_policy.filename_suffix,
        )
        if parsed:
            dated.append((parsed, item))

    dated.sort(key=lambda pair: (pair[0], pair[1].get("name", "")), reverse=True)
    keep_ids = {item["id"] for _, item in dated[: retention_policy.keep_latest]}
    # Never trash the newest VERIFIED object of either format, bound to its Drive file ID.
    # The listing carries no verification evidence, so a newer upload that failed
    # verification must not stand in for the last good copy (r1 review, #1421).
    verified = verified_file_ids or frozenset()
    newest_verified_per_kind: dict[str | None, str] = {}
    for _, item in dated:
        if item["id"] in verified:
            newest_verified_per_kind.setdefault(_snapshot_suffix(item.get("name", "")), item["id"])
    keep_ids |= set(newest_verified_per_kind.values())

    trashed: list[str] = []
    for _, item in dated:
        if item["id"] in keep_ids or _snapshot_suffix(item.get("name", "")) not in newest_verified_per_kind:
            continue
        service.files().update(
            fileId=item["id"],
            body={"trashed": True},
            supportsAllDrives=True,
        ).execute()
        trashed.append(item["name"])
    return trashed


def _verified_drive_file_ids(log_path: Path) -> set[str] | None:
    """Drive file IDs whose latest receipt proves a verified upload; None if unreadable.

    The latest receipt per object wins, so a later failure (md5 mismatch, aged-out weekly
    snapshot) revokes an earlier pass. A missing log, or any uncertain (non-receipt,
    non-progress-noise) line, yields None, which Drive retention treats as no evidence
    and trashes nothing.
    """
    try:
        # strict: an uncertain line could be the receipt that revoked a verification, so it
        # authorizes no trashing. Known progress noise (#1418) is not uncertainty.
        receipts = _read_backup_receipts(Path(log_path), strict=True)
    except (OSError, ValueError):  # BackupReceiptUnreadableError is a ValueError
        return None
    latest: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        drive_file = receipt.get("drive_file")
        file_id = drive_file.get("id") if isinstance(drive_file, dict) else None
        if isinstance(file_id, str) and file_id:
            latest[file_id] = receipt
    return {
        file_id
        for file_id, receipt in latest.items()
        if receipt.get("uploaded") is True
        and receipt.get("verified") is True
        and not receipt.get("verification_error")
        and receipt.get("drive_md5_match") is not False
    }


def _snapshot_created_time(receipt: Mapping[str, Any]) -> dt.datetime | None:
    """Exact VACUUM time, or conservative UTC midnight for a legacy dated artifact."""
    try:
        if "snapshot_created_at" in receipt:
            value = dt.datetime.fromisoformat(receipt["snapshot_created_at"])
            return value.astimezone(dt.UTC) if value.tzinfo is not None else None
        drive_file = receipt.get("drive_file")
        drive_name = drive_file.get("name") if isinstance(drive_file, dict) else drive_file
        # Recovery receipts across a prior machine identity may have only a Drive
        # filename. Its artifact date is conservative; attempted_at would relabel old bytes.
        for name in (receipt.get("snapshot"), drive_name):
            date = _parse_snapshot_date(Path(name).name) if isinstance(name, str) else None
            if date is not None:
                return dt.datetime.combine(date, dt.time(), dt.UTC)
        return None
    except (ValueError, TypeError):
        return None


def _resume_fresh_enough(receipt: Mapping[str, Any], now: dt.datetime) -> bool:
    created = _snapshot_created_time(receipt)
    hours = _configured_positive_number(BACKUP_RESUME_MAX_AGE_HOURS_ENV, 36)
    weekly = os.environ.get(BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS_ENV)
    if weekly is not None:
        hours = min(hours, float(weekly))
    return created is not None and dt.timedelta(0) <= now - created < dt.timedelta(hours=hours)


def _recent_verified_backup_for_reuse(
    log_path: Path, db_path: Path, *, now: dt.datetime, max_age_hours: float
) -> dict[str, Any] | None:
    try:
        receipts = _read_backup_receipts(log_path)
    except (OSError, ValueError):
        return None
    now = now.astimezone(dt.UTC)
    for receipt in reversed(receipts):
        try:
            created = _snapshot_created_time(receipt)
        except (ValueError, KeyError, TypeError):
            continue
        if created is None:
            continue
        age = now - created
        drive_file = receipt.get("drive_file")
        if (
            receipt.get("db") == str(db_path)
            and receipt.get("uploaded") is True
            and receipt.get("verified") is True
            and "error_type" not in receipt
            and isinstance(drive_file, dict)
            and isinstance(drive_file.get("id"), str)
            and dt.timedelta(0) <= age <= dt.timedelta(hours=max_age_hours)
        ):
            return receipt
    return None


def _local_snapshot_intact(receipt: Mapping[str, Any]) -> bool:
    """The durable #1387 proof is bound to these exact bytes, not a dated filename."""
    try:
        path = Path(receipt["snapshot"])
        return (
            receipt.get("local_verified") is True
            and not path.is_symlink()
            and path.is_file()
            and path.stat().st_nlink == 1
            and receipt.get("local_artifact") == _upload_identity(path)
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _recent_local_verified_backup(
    log_path: Path, db_path: Path, *, now: dt.datetime, max_age_hours: float
) -> dict[str, Any] | None:
    try:
        receipts = _read_backup_receipts(log_path)
    except (OSError, ValueError):
        return None
    latest = {r["snapshot"]: r for r in receipts if isinstance(r.get("snapshot"), str)}
    for receipt in reversed(list(latest.values())):
        created = _snapshot_created_time(receipt)
        if (
            receipt.get("db") == str(db_path)
            and "snapshot_created_at" in receipt
            and created is not None
            and local_backup_date(created) == local_backup_date(now)
            and dt.timedelta(0) <= now - created <= dt.timedelta(hours=max_age_hours)
            and _local_snapshot_intact(receipt)
            and receipt.get("local_integrity") == "ok"
            and receipt.get("local_integrity_pragma") == "integrity_check"
            and not receipt.get("verification_error")
            and not (receipt.get("uploaded") is True and receipt.get("verified") is not True)
        ):
            return receipt
    return None


def _protected_snapshot_names(staging_dir: Path) -> set[str]:
    pin = staging_dir / ".weekly-backup-pin.json"
    try:
        payload = json.loads(pin.read_text())
        if payload["date"] == local_backup_date().isoformat():
            return {Path(payload["snapshot"]).name}
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError, TypeError):
        return {p.name for p in staging_dir.iterdir() if _snapshot_suffix(p.name)}  # unreadable custody fails closed
    return set()


def _reclaim_superseded_local(log_path: Path, result: dict[str, Any], staging_dir: Path) -> None:
    if not (result.get("uploaded") is True and result.get("verified") is True):
        return
    try:
        receipts = _read_backup_receipts(log_path, strict=True)
    except (OSError, ValueError):
        print("Cannot validate local receipts; retaining all superseded artifacts", flush=True)
        return
    latest = {r["snapshot"]: r for r in receipts if isinstance(r.get("snapshot"), str)}
    protected = _protected_snapshot_names(staging_dir) | {
        Path(r["snapshot"]).name for r in receipts if r.get("verified") is True and isinstance(r.get("snapshot"), str)
    }
    for receipt in latest.values():
        path = Path(receipt["snapshot"])
        created, newer = _snapshot_created_time(receipt), _snapshot_created_time(result)
        if (
            any(receipt.get(k) != result.get(k) for k in ("db", "machine_id", "drive_folder", "backup_log_provenance"))
            or path.parent.resolve() != staging_dir.resolve()
            or path.name in protected
            or created is None
            or newer is None
            or created >= newer
            or receipt.get("uploaded") is True
            and receipt.get("verified") is True
        ):
            continue
        abandoned = receipt.get("upload_state") == "abandoned"
        if not abandoned and not _local_snapshot_intact(receipt):
            continue
        if not abandoned:
            path.unlink()
            result.setdefault("superseded_stalled_deleted", []).append(path.name)
        terminal = {
            k: v for k, v in receipt.items() if k not in {"error", "error_type", "error_code", "verification_error"}
        }
        _append_json_log(
            log_path,
            {**terminal, "local_verified": False, "upload_state": "superseded", "superseded_by": result["snapshot"]},
        )


def _local_upload_receipt(
    log_path: Path,
    context: dict[str, Any],
    staging_dir: Path,
    *,
    snapshot_name: str | tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    """Resume only the selected destination's exact artifact, including completed sessions."""
    snapshot_names = (snapshot_name,) if isinstance(snapshot_name, str) else snapshot_name
    try:
        receipts = _read_backup_receipts(log_path) if log_path.exists() else []
    except ValueError:
        if snapshot_names is None or any((staging_dir / name).exists() for name in snapshot_names):
            raise
        print("Cannot validate upload journal; retaining all existing compressed archives", flush=True)
        return None
    latest = {r["snapshot"]: r for r in receipts if isinstance(r.get("snapshot"), str)}
    for receipt in sorted(
        latest.values(), key=lambda r: _snapshot_created_time(r) or dt.datetime.min.replace(tzinfo=dt.UTC), reverse=True
    ):
        if any(receipt.get(k) != context.get(k) for k in ("db", "machine_id", "drive_folder", "backup_log_provenance")):
            continue
        if receipt.get("local_verified") is not True:
            continue
        path = Path(receipt["snapshot"])
        if path.parent.resolve() != staging_dir.resolve() or (
            snapshot_names is not None and path.name not in snapshot_names
        ):
            continue
        if receipt.get("uploaded") is True and receipt.get("verified") is True:
            continue
        if not _local_snapshot_intact(receipt):
            invalid = f"Local snapshot missing or changed: {path.name}; upload abandoned"
            _append_json_log(
                log_path,
                {
                    **receipt,
                    "upload_state": "abandoned",
                    "local_verified": False,
                    "error": invalid,
                    "error_code": "local_snapshot_invalid",
                },
            )
            from . import job_alerts

            job_alerts.report(
                "backup-upload-abandoned", invalid, path=job_alerts.alert_path(db_path=Path(context["db"]))
            )
            continue
        return receipt
    return None


def _serialized_backup_run(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        staging_dir = Path(bound.arguments.get("staging_dir", DEFAULT_STAGING_DIR)).expanduser()
        staging_dir.mkdir(parents=True, exist_ok=True)
        with (staging_dir / ".backup.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                error = f"backup already running in {staging_dir}"
                resolved_db_path = bound.arguments.get("db_path") or get_db_path()
                _append_json_log(
                    _backup_log_path(bound.arguments.get("log_path"), db_path=resolved_db_path),
                    {
                        "attempted_at": dt.datetime.now(dt.UTC).isoformat(),
                        "db": str(resolved_db_path),
                        "verified": False,
                        "uploaded": False,
                        "backup_log_provenance": _backup_log_provenance(),
                        "error_type": "BackupAlreadyRunningError",
                        "error": error,
                    },
                )
                raise BackupAlreadyRunningError(error) from exc
            try:
                # Weekly maintenance's first receipt check precedes this lock. A daily
                # backup can finish in that gap, so recheck before copying the DB.
                reuse_age = os.environ.get(BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS_ENV)
                if reuse_age is not None and bound.arguments.get("phase", "all") == "all":
                    hours = float(reuse_age)
                    if not math.isfinite(hours) or hours < 0:
                        raise ValueError(f"{BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS_ENV} must be finite and nonnegative")
                    db_path = Path(bound.arguments.get("db_path") or get_db_path())
                    log_path = _backup_log_path(bound.arguments.get("log_path"), db_path=db_path)
                    receipt = _recent_verified_backup_for_reuse(
                        log_path, db_path, now=dt.datetime.now(dt.UTC), max_age_hours=hours
                    )
                    if receipt is not None:
                        return receipt
                return func(*args, **kwargs)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    return wrapped


def _upload_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest}


def _stalled_uploads(log_path: Path, context: dict[str, Any], staging_dir: Path) -> list[dict[str, Any]]:
    """Latest receipt per artifact; uncertainty disables reclamation, never broadens it."""
    if not log_path.exists():
        return []
    try:
        receipts = _read_backup_receipts(log_path, strict=True)
        latest = {r["snapshot"]: r for r in receipts if isinstance(r, dict) and isinstance(r.get("snapshot"), str)}
        protected = {
            str(Path(r["snapshot"]).resolve())
            for r in receipts
            if isinstance(r, dict)
            and (r.get("uploaded") is True or r.get("verified") is True)
            and isinstance(r.get("snapshot"), str)
        }
        candidates = []
        for name, receipt in reversed(list(latest.items())):
            path = Path(name)
            state = receipt.get("resumable_upload")
            if (
                str(path.resolve()) in protected
                or receipt.get("uploaded") is not False
                or receipt.get("verified") is not False
                or not isinstance(receipt.get("resumable_upload"), dict)
                or dt.datetime.fromisoformat(receipt["attempted_at"])
                >= dt.datetime.fromisoformat(context["attempted_at"])
                or any(
                    receipt.get(k) != context.get(k)
                    for k in ("db", "machine_id", "drive_folder", "backup_log_provenance")
                )
            ):
                continue
            if (
                not isinstance(state, dict)
                or path.is_symlink()
                or not path.is_file()
                or path.parent.resolve() != staging_dir.resolve()
                or _parse_snapshot_date(path.name) is None
                or path.name not in {f"{path.name[:10]}{suffix}" for suffix in DB_SNAPSHOT_SUFFIXES}
                or path.stat().st_nlink != 1
                or _upload_identity(path) != {k: state.get(k) for k in ("path", "size", "sha256")}
            ):
                print(f"Retaining stalled backup with unconfirmed identity: {path.name}", flush=True)
                continue
            candidates.append(receipt)
        return sorted(
            candidates,
            key=lambda r: (
                _snapshot_created_time(r) or dt.datetime.min.replace(tzinfo=dt.UTC),
                dt.datetime.fromisoformat(r["attempted_at"]),
            ),
            reverse=True,
        )
    except (OSError, ValueError, TypeError, KeyError):
        print("Cannot validate stalled-backup receipts; retaining local artifacts", flush=True)
        return []


def _safe_upload_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.netloc == "www.googleapis.com"
        and parsed.path == "/upload/drive/v3/files"
        and "upload_id=" in parsed.query
    )


def _retained_target_collision(log_path: Path, target: Path) -> bool:
    if not target.exists() or not log_path.exists():
        return False
    try:
        receipts = _read_backup_receipts(log_path, strict=True)
        latest = {r["snapshot"]: r for r in receipts if isinstance(r.get("snapshot"), str)}
        for receipt in latest.values():
            name = receipt.get("snapshot")
            if (
                isinstance(name, str)
                and Path(name).resolve() == target.resolve()
                and not (receipt.get("uploaded") is True and receipt.get("verified") is True)
                and (
                    receipt.get("local_verified") is True
                    or isinstance(receipt.get("resumable_upload"), dict)
                    or receipt.get("error_code") == "drive_upload_stalled"
                )
            ):
                return True
        return False
    except (OSError, ValueError):
        return True  # An unreadable journal cannot authorize replacing an existing local artifact.


def _remove_superseded_stalls(receipts: list[dict[str, Any]], result: dict[str, Any], date_stamp: str) -> None:
    for receipt in receipts:
        path = Path(receipt["snapshot"])
        if not path.exists():
            continue
        try:
            if (
                path.name[:10] >= date_stamp
                or _snapshot_created_time(receipt) is None
                or _snapshot_created_time(result) is None
                or _snapshot_created_time(receipt) >= _snapshot_created_time(result)
                or path.is_symlink()
                or path.stat().st_nlink != 1
                or _upload_identity(path) != {k: receipt["resumable_upload"].get(k) for k in ("path", "size", "sha256")}
            ):
                print(f"Retaining stalled backup without proof it is superseded: {path.name}", flush=True)
                continue
            if path.name in _protected_snapshot_names(path.parent):
                continue
            path.unlink()
            result.setdefault("superseded_stalled_deleted", []).append(path.name)
            print(f"Removed superseded stalled backup: {path.name}", flush=True)
        except OSError:
            print(f"Could not reclaim stalled backup: {path.name}; retaining it", flush=True)


@_serialized_backup_run
def run_backup(
    db_path: Path | None = None,
    staging_dir: Path = DEFAULT_STAGING_DIR,
    folder_parts: list[str] | None = None,
    machine_id: str | None = None,
    log_path: Path | None = None,
    date_stamp: str | None = None,
    upload: bool = True,
    retention_policy: DriveRetentionPolicy = DAILY_RETENTION,
    remove_local_after_upload: bool = True,
    deadline_monotonic: float | None = None,
    phase: str = "all",
) -> dict[str, Any]:
    if phase not in {"all", "snapshot", "upload"}:
        raise ValueError("backup phase must be all, snapshot, or upload")
    upload = upload and phase != "snapshot"
    run_deadline = (
        deadline_monotonic
        if deadline_monotonic is not None
        else time.monotonic() + _configured_backup_timeout_seconds()
    )
    resume_state: dict[str, Any] | None = None
    nothing_pending = False
    resolved_db_path = db_path or get_db_path()
    resolved_date_stamp = date_stamp or _today()
    resolved_log_path = _backup_log_path(log_path, db_path=resolved_db_path)
    retention_enabled = _drive_retention_enabled()
    result: dict[str, Any] = {
        "attempted_at": dt.datetime.now(dt.UTC).isoformat(),
        "db": str(resolved_db_path),
        "uploaded": False,
        "local_removed": False,
        "verified": False,
        "backup_log_provenance": _backup_log_provenance(),
        "attempt_reclamation": "unknown",
        "writer_probe_error": None,
        "drive_retention": "enabled" if retention_enabled else "disabled",
        "retention_mode": "trash",
        "retention_deleted": [],
        "local_gzip_retention_deleted": [],
    }
    invocation = os.environ.get(BACKUP_INVOCATION_ENV, "")
    if re.fullmatch(r"[0-9a-f]{32}", invocation):
        result["backup_invocation_id"] = invocation  # This body runs only after staging-lock admission.

    def record_reclamation_status(status: str, error: str | None) -> None:
        result["attempt_reclamation"] = status
        result["writer_probe_error"] = error

    def record_integrity(value: str, pragma: str, seconds: float) -> None:
        result.update(local_integrity=value, local_integrity_pragma=pragma, integrity_duration_seconds=seconds)

    def record_capacity(raw: int, important: int | None, selected: int, required: int) -> None:
        result.update(
            {
                "raw_free_bytes": raw,
                "important_usage_free_bytes": important,
                "selected_free_bytes": selected,
                "required_bytes": required,
            }
        )

    try:
        resolved_machine_id = _validate_machine_id(machine_id) if machine_id is not None else resolve_machine_id()
        resolved_folder_parts = (
            list(folder_parts) if folder_parts is not None else default_drive_folder_parts(resolved_machine_id)
        )
        result.update(
            {
                "drive_folder": "/".join(resolved_folder_parts),
                "machine_id": resolved_machine_id,
            }
        )
        result["consumed_raw_crash_leftovers_deleted"] = _cleanup_consumed_raw_remnants(Path(staging_dir))
        if resolved_log_path.exists():
            try:
                _, noise_lines, skipped_lines = _scan_backup_receipts(resolved_log_path)
            except (OSError, ValueError):
                pass  # Each reader below decides how an unreadable journal fails.
            else:
                result.update(receipt_journal_noise_lines=noise_lines, receipt_journal_skipped_lines=skipped_lines)
                if noise_lines or skipped_lines:
                    print(
                        f"Backup receipt journal {resolved_log_path}: skipped {noise_lines} progress-noise lines "
                        f"and {skipped_lines} other non-receipt lines",
                        flush=True,
                    )
        local = (
            _local_upload_receipt(resolved_log_path, result, Path(staging_dir))
            if phase == "upload"
            else _recent_local_verified_backup(
                resolved_log_path,
                resolved_db_path,
                now=dt.datetime.now().astimezone(),
                max_age_hours=local_backup_max_age_hours(),
            )
            if phase == "snapshot"
            else _local_upload_receipt(
                resolved_log_path,
                result,
                Path(staging_dir),
                snapshot_name=tuple(f"{resolved_date_stamp}{suffix}" for suffix in DB_SNAPSHOT_SUFFIXES),
            )
        )
        if phase == "snapshot" and local is not None:
            created = _snapshot_created_time(local)
            now = dt.datetime.fromisoformat(result["attempted_at"])
            if created and local_backup_date(created) == local_backup_date(now) and _local_snapshot_intact(local):
                result.update(local)
                return result
        if phase == "upload" and local is None:
            nothing_pending = True
            result["upload_state"] = "nothing_pending"
            return result
        stalled = _stalled_uploads(resolved_log_path, result, Path(staging_dir)) if upload else []
        candidate = next(
            (
                r
                for r in stalled
                if _safe_upload_url(r["resumable_upload"].get("url"))
                and _resume_fresh_enough(r, dt.datetime.fromisoformat(result["attempted_at"]))
            ),
            None,
        )

        local_resume = local is not None and not (local.get("uploaded") and local.get("verified"))
        candidate = local if local_resume else candidate

        def prepare_artifact(resume: dict[str, Any] | None) -> SQLiteBackupArtifact:
            if resume is not None:
                raw_deleted, _ = _cleanup_raw_snapshots(Path(staging_dir))
                return SQLiteBackupArtifact(
                    gzip_path=Path(resume["snapshot"]),
                    uncompressed_path=None,
                    sentinel_chunks=resume["sentinel_snapshot_chunks"],
                    local_retention_deleted=raw_deleted,
                    stale_attempts_deleted=[],
                    surviving_attempts=[],
                    surviving_attempt_bytes=0,
                    surviving_attempt_growth_reserve_bytes=0,
                    attempt_reclamation="resumed",
                    writer_probe_error=None,
                    snapshot_created_at=_snapshot_created_time(resume).isoformat(),
                    local_integrity=resume.get("local_integrity"),
                    local_integrity_pragma=resume.get("local_integrity_pragma"),
                    snapshot_duration_seconds=resume.get("snapshot_duration_seconds"),
                    integrity_duration_seconds=resume.get("integrity_duration_seconds"),
                    raw_sha256=resume.get("raw_sha256"),
                )
            for suffix in DB_SNAPSHOT_SUFFIXES:
                target = Path(staging_dir) / f"{resolved_date_stamp}{suffix}"
                if _retained_target_collision(resolved_log_path, target):
                    raise RetainedStalledBackupConflictError(target.name)
            create_started = dt.datetime.now(dt.UTC).isoformat()
            try:
                artifact = create_sqlite_backup_artifact(
                    resolved_db_path,
                    staging_dir,
                    date_stamp=resolved_date_stamp,
                    reclamation_status_callback=record_reclamation_status,
                    capacity_status_callback=record_capacity,
                    integrity_full=True if phase == "snapshot" else None,
                    integrity_status_callback=record_integrity,
                )
            except RuntimeError as exc:
                reason = _redact_upload_secrets(str(exc))
                floor_breach = "raw free space floor breached" in reason
                if "Insufficient free space" in reason or floor_breach:
                    reason += f"; retained compressed snapshots: {_compressed_snapshot_names(Path(staging_dir))}"
                if floor_breach:
                    reason += f"; retained raw snapshots: {_retained_raw_snapshot_names(Path(staging_dir))}"
                if stalled and ("Insufficient free space" in reason or "raw free space floor breached" in reason):
                    raise RetainedStalledBackupCapacityError(
                        [Path(r["snapshot"]).name for r in stalled], reason=reason
                    ) from exc
                if "Insufficient free space" in reason or floor_breach:
                    raise RuntimeError(reason) from exc
                raise
            result["snapshot_created_at"] = getattr(artifact, "snapshot_created_at", None) or create_started
            return artifact

        for resume in (
            [candidate] if phase == "upload" or local_resume else [candidate, None] if candidate is not None else [None]
        ):
            artifact = prepare_artifact(resume)
            snapshot = artifact.gzip_path
            snapshot_size = snapshot.stat().st_size
            result.update(
                {
                    "snapshot": str(snapshot),
                    "snapshot_created_at": getattr(artifact, "snapshot_created_at", None)
                    or result.get("snapshot_created_at"),
                    "local_uncompressed_snapshot": (
                        str(artifact.uncompressed_path) if artifact.uncompressed_path else None
                    ),
                    "local_retention_deleted": artifact.local_retention_deleted,
                    "stale_attempts_deleted": getattr(artifact, "stale_attempts_deleted", []),
                    "surviving_attempts": getattr(artifact, "surviving_attempts", []),
                    "surviving_attempt_bytes": getattr(artifact, "surviving_attempt_bytes", 0),
                    "surviving_attempt_growth_reserve_bytes": getattr(
                        artifact,
                        "surviving_attempt_growth_reserve_bytes",
                        0,
                    ),
                    "attempt_reclamation": getattr(artifact, "attempt_reclamation", "unknown"),
                    "writer_probe_error": getattr(artifact, "writer_probe_error", None),
                    "local_integrity": getattr(artifact, "local_integrity", None),
                    "local_integrity_pragma": getattr(artifact, "local_integrity_pragma", None),
                    "snapshot_duration_seconds": getattr(artifact, "snapshot_duration_seconds", None),
                    "integrity_duration_seconds": getattr(artifact, "integrity_duration_seconds", None),
                    "raw_sha256": getattr(artifact, "raw_sha256", None),
                    "compressor": getattr(artifact, "compressor", None)
                    or ("zstd" if snapshot.name.endswith(ZSTD_SNAPSHOT_SUFFIX) else "gzip"),
                    "compress_duration_seconds": getattr(artifact, "compress_duration_seconds", None),
                    **(
                        {"compressor_fallback": artifact.compressor_fallback}
                        if getattr(artifact, "compressor_fallback", None)
                        else {}
                    ),
                    "sentinel_snapshot_chunks": artifact.sentinel_chunks,
                    "bytes": snapshot_size,
                }
            )
            # create_sqlite_backup_artifact has already proven gzip CRC/ISIZE,
            # SQLite header, sentinel and raw/gzip equality before dropping raw (#1387).
            with snapshot.open("rb") as durable:
                os.fsync(durable.fileno())
            directory_fd = os.open(snapshot.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            if resume is not None:
                for key in ("uploaded", "drive_file", "resumable_upload"):
                    if key in resume:
                        result[key] = resume[key]
                resume_state = resume.get("resumable_upload")
            result.update(local_verified=True, local_artifact=_upload_identity(snapshot), upload_state="pending")
            _append_json_log(resolved_log_path, result)
            if not upload:
                break
            if resume is not None and resume.get("uploaded") is True:
                uploaded = resume["drive_file"]
                service = build_drive_service()
                folder_id = ensure_drive_folder_chain(service, resolved_folder_parts)
                break  # Completed session: retry verification, never POST another upload.
            result["upload_state"] = "in_progress"
            credentials = get_drive_credentials()
            service = build_drive_service()
            folder_id = ensure_drive_folder_chain(service, resolved_folder_parts)
            assert_drive_folder_owned_by_machine(service, folder_id=folder_id, machine_id=resolved_machine_id)
            saved_session = resume.get("resumable_upload") if resume else None
            if saved_session is not None and not _safe_upload_url(saved_session.get("url")):
                saved_session = None  # Invalid capability cannot authorize a network destination.
                resume_state = None
                result.pop("resumable_upload", None)
            if saved_session is not None and saved_session.get("folder_id") != folder_id:
                if phase == "upload" or local_resume:
                    raise RuntimeError("Saved Drive destination changed; local snapshot retained")
                print("Saved Drive destination changed; preparing a fresh snapshot", flush=True)
                continue
            upload_identity = _upload_identity(snapshot)
            if saved_session is not None and upload_identity != {
                k: saved_session.get(k) for k in ("path", "size", "sha256")
            }:
                if phase == "upload" or local_resume:
                    raise RuntimeError("Saved artifact changed; local snapshot retained")
                print("Saved artifact changed before resume; preparing a fresh snapshot", flush=True)
                continue

            def record_upload_progress(url: str, confirmed: int) -> None:
                nonlocal resume_state
                next_state = {"url": url, "folder_id": folder_id, **upload_identity}
                first = resume_state != next_state
                resume_state = next_state
                result["bytes_confirmed"] = confirmed
                if first:
                    _append_json_log(
                        resolved_log_path,
                        {
                            **result,
                            "resumable_upload": resume_state,
                            "error_code": "drive_upload_interrupted",
                            "upload_in_progress": True,
                            "upload_state": "in_progress",
                        },
                    )

            try:
                uploaded = upload_file_to_drive_raw(
                    snapshot,
                    folder_id,
                    credentials,
                    machine_id=resolved_machine_id,
                    progress_callback=record_upload_progress,
                    run_deadline=run_deadline,
                    **({"resume_session_url": saved_session["url"]} if saved_session else {}),
                )
            except DriveSessionExpiredError:
                if phase == "upload" or local_resume:
                    # Drive confirmed expiration; restart the session for the SAME bytes.
                    uploaded = upload_file_to_drive_raw(
                        snapshot,
                        folder_id,
                        credentials,
                        machine_id=resolved_machine_id,
                        progress_callback=record_upload_progress,
                        run_deadline=run_deadline,
                    )
                    break
                if resume is None:
                    raise
                print("Saved Drive session expired; preparing a fresh snapshot", flush=True)
                continue
            break
        if upload:
            file_id = uploaded.get("id")
            if not file_id:
                raise RuntimeError(f"Drive upload response missing file id: {uploaded!r}")
            # Drive has completed the session. Persist that fact before verification can
            # fail or its final receipt can be truncated; an older checkpoint must not revive it.
            result.update({"uploaded": True, "drive_file": uploaded})
            resume_state = None
            result.pop("resumable_upload", None)
            _append_json_log(resolved_log_path, {**result, "verification_pending": True})
            # The service above has sat idle through an upload that can run for hours; its
            # keep-alive socket is dead by now (2026-09-23: a finished upload was recorded failed
            # when verification reset on it). Every post-upload call gets a fresh connection. The
            # rebuild may refresh the token over the network, so it gets the same bounded retry.
            service = _retry_transient_drive_call(build_drive_service, what="Drive service rebuild after upload")
            verify_drive_upload(
                service,
                file_id=file_id,
                expected_name=snapshot.name,
                expected_size=snapshot_size,
                expected_machine_id=resolved_machine_id,
            )
            result.update(
                verify_sqlite_backup_artifact(
                    artifact,
                    full=_should_run_full_verify(resolved_date_stamp),
                    service=service,
                    file_id=file_id,
                )
            )
            if not result["verified"] and not result.get("verification_error"):
                result["verification_error"] = "local backup verification failed without a diagnostic"
            weekly_age = os.environ.get(BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS_ENV)
            if result["verified"] and weekly_age is not None:
                try:
                    max_hours = float(weekly_age)
                    if not math.isfinite(max_hours) or max_hours < 0:
                        raise ValueError("weekly age must be finite and nonnegative")
                except (ValueError, TypeError):
                    result.update(
                        {
                            "verified": False,
                            "verification_error": f"invalid {BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS_ENV}; VACUUM not authorized",
                        }
                    )
                else:
                    created = _snapshot_created_time(result)
                    if (
                        created is None
                        or not 0 <= (dt.datetime.now(dt.UTC) - created).total_seconds() / 3600 <= max_hours
                    ):
                        result.update(
                            {
                                "verified": False,
                                "verification_error": "snapshot aged beyond weekly backup safety window; VACUUM not authorized",
                            }
                        )
            result["upload_state"] = "verified" if result["verified"] else "failed"
            if result["verified"]:
                _append_json_log(resolved_log_path, result)  # remote proof precedes reclamation
                _reclaim_superseded_local(resolved_log_path, result, snapshot.parent)
                from . import job_alerts

                job_alerts.report(
                    "backup-upload-abandoned", None, path=job_alerts.alert_path(db_path=Path(resolved_db_path))
                )
                result["temp_dirs_removed_after_verification"] = _prune_owned_temps_after_verified(
                    snapshot.parent, snapshot
                )
                if remove_local_after_upload and snapshot.name not in _protected_snapshot_names(snapshot.parent):
                    snapshot.unlink()
                    result["local_removed"] = True
                if resume is None:
                    _remove_superseded_stalls(stalled, result, snapshot.name[:10])
                verified_drive_names = _verified_snapshot_names_from_log(
                    resolved_log_path, db_path=Path(resolved_db_path)
                )
                verified_drive_names.add(snapshot.name)
                result["local_gzip_retention_deleted"] = prune_local_compressed_snapshots(
                    snapshot.parent,
                    verified_drive_names=verified_drive_names,
                    protected_names=_protected_snapshot_names(snapshot.parent)
                    | {
                        name for name in _compressed_snapshot_names(snapshot.parent) if name not in verified_drive_names
                    },
                )
                if retention_enabled:
                    try:
                        result["retention_deleted"] = prune_drive_backups(
                            service,
                            folder_parts=resolved_folder_parts if folder_parts is not None else None,
                            machine_id=resolved_machine_id,
                            selected_folder_id=folder_id,
                            retention_policy=retention_policy,
                            verified_file_ids=_verified_drive_file_ids(resolved_log_path),
                        )
                    except Exception as exc:
                        result.update(
                            {
                                "retention_error_type": type(exc).__name__,
                                "retention_error": _redact_upload_secrets(str(exc)),
                            }
                        )
                        raise

    except BackupStoppedError as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": _redact_upload_secrets(str(exc)),
                "stop_signal": signal.Signals(exc.signum).name,
            }
        )
        exc.receipt_logged = True
        raise
    except Exception as exc:
        result.update(
            {"error_type": getattr(exc, "error_type", type(exc).__name__), "error": _redact_upload_secrets(str(exc))}
        )
        error_code = getattr(exc, "error_code", None)
        if isinstance(error_code, str):
            result["error_code"] = error_code
        if isinstance(exc, DriveUploadStalledError):
            result.update(
                {
                    "error_type": exc.error_code,
                    "error_code": exc.error_code,
                    "bytes_confirmed": exc.bytes_confirmed,
                }
            )
        raise
    finally:
        result["retained_raw_snapshots"] = _retained_raw_snapshot_names(Path(staging_dir))
        raw = Path(staging_dir) / f"{resolved_date_stamp}.db"
        if (result.get("error") or result.get("verification_error")) and not (
            result["uploaded"] and result["verified"]
        ):
            result["upload_state"] = "failed"
        if raw.is_file() and not raw.is_symlink():
            result["local_uncompressed_snapshot"] = str(raw)
            result["raw_snapshot_retained_reason"] = "No verified compressed archive replaced this raw snapshot"
        if resume_state is not None and not result["uploaded"] and not result.get("drive_file"):
            result["resumable_upload"] = resume_state
            result.setdefault("error_code", "drive_upload_interrupted")
        if not nothing_pending:
            _append_json_log(resolved_log_path, result)
    return result


def _run_backup_process(timeout_seconds: int) -> int:
    previous_alarm_handler = None
    previous_alarm_handler = signal.getsignal(signal.SIGALRM)
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    previous_int_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGALRM, _raise_backup_timeout)
    signal.signal(signal.SIGTERM, _raise_backup_stopped)
    signal.signal(signal.SIGINT, _raise_backup_stopped)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        resolved_db_path = get_db_path()

        run_deadline = time.monotonic() + timeout_seconds

        def run_phase():
            return run_backup(
                deadline_monotonic=run_deadline,
                phase=os.environ.get("BRAINLAYER_BACKUP_PHASE", "all"),
                staging_dir=Path(os.environ.get("BRAINLAYER_BACKUP_STAGING_DIR", str(DEFAULT_STAGING_DIR))),
                # Prefer BRAINLAYER_BACKUP_DRIVE_FOLDER; BRAINLAYER_BACKUP_DRIVE_PATH
                # is a legacy alias. With neither set, run_backup derives the host folder.
                folder_parts=(
                    configured_folder.split("/")
                    if (
                        configured_folder := os.environ.get(
                            "BRAINLAYER_BACKUP_DRIVE_FOLDER",
                            os.environ.get("BRAINLAYER_BACKUP_DRIVE_PATH"),
                        )
                    )
                    else None
                ),
                log_path=_backup_log_path(None, db_path=resolved_db_path, env=os.environ),
            )

        result = run_phase()
        if os.environ.get("BRAINLAYER_BACKUP_PHASE") == "upload":
            while result.get("verified"):
                next_result = run_phase()
                if next_result.get("snapshot") == result.get("snapshot"):
                    break
                result = next_result
    except BackupStoppedError as exc:
        if not getattr(exc, "receipt_logged", False):
            _append_json_log(
                _backup_log_path(None, db_path=get_db_path(), env=os.environ),
                {
                    "attempted_at": dt.datetime.now(dt.UTC).isoformat(),
                    "db": str(get_db_path()),
                    "verified": False,
                    "uploaded": False,
                    "backup_log_provenance": _backup_log_provenance(),
                    "error_type": "BackupStoppedError",
                    "error": _redact_upload_secrets(str(exc)),
                    "stop_signal": signal.Signals(exc.signum).name,
                },
            )
        return 128 + exc.signum
    except BackupTimeoutError:
        print(f"brainlayer backup timed out after {timeout_seconds}s", flush=True)
        return 124
    except Exception as exc:
        print(
            f"brainlayer backup failed: {_redact_upload_secrets(str(exc))}\n{_redact_upload_secrets(traceback.format_exc())}",
            flush=True,
        )
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)
        signal.signal(signal.SIGTERM, previous_term_handler)
        signal.signal(signal.SIGINT, previous_int_handler)
    print(_redact_upload_secrets(json.dumps(result, sort_keys=True)), flush=True)
    return (
        0
        if (
            result.get("local_verified")
            if os.environ.get("BRAINLAYER_BACKUP_PHASE") == "snapshot"
            else result.get("verified", True) or result.get("upload_state") == "nothing_pending"
        )
        else 1
    )


def _supervise_backup_process(timeout_seconds: int, *, command: list[str] | None = None) -> int:
    guard_fd = os.environ.get("BRAINLAYER_CALENDAR_GUARD_FD")
    managed = guard_fd is not None
    if managed:
        if not stat.S_ISFIFO(os.fstat(int(guard_fd)).st_mode):
            raise ValueError("invalid calendar guard descriptor")
    lease = os.environ.get("BRAINLAYER_MAINTENANCE_FD")
    child_env = os.environ.copy()
    child_env[BACKUP_SUPERVISED_CHILD_ENV] = "1"
    child = subprocess.Popen(
        command or [sys.executable, "-m", "brainlayer.backup_daily"],
        env=child_env,
        start_new_session=not managed,
        pass_fds=(int(lease),) if managed and lease is not None else (),
    )
    previous_signal_handlers: dict[int, Any] = {}

    def signal_child(signum):
        (os.kill if managed else os.killpg)(child.pid, signum)

    def forward_shutdown_signal(signum, frame) -> None:  # noqa: ARG001
        if child.poll() is None:
            signal_child(signum)
        raise SystemExit(128 + signum)

    for shutdown_signal in (signal.SIGTERM, signal.SIGINT):
        previous_signal_handlers[shutdown_signal] = signal.getsignal(shutdown_signal)
        signal.signal(shutdown_signal, forward_shutdown_signal)
    try:
        try:
            return child.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            if child.poll() is None:
                signal_child(signal.SIGALRM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    signal_child(signal.SIGKILL)
                    child.wait()
            message = f"backup exceeded configured wall-clock timeout ({timeout_seconds}s)"
            resolved_db_path = get_db_path()
            _append_json_log(
                _backup_log_path(None, db_path=resolved_db_path, env=os.environ),
                {
                    "attempted_at": dt.datetime.now(dt.UTC).isoformat(),
                    "db": str(get_db_path()),
                    "uploaded": False,
                    "local_removed": False,
                    "verified": False,
                    "backup_log_provenance": _backup_log_provenance(),
                    "attempt_reclamation": "unknown",
                    "writer_probe_error": None,
                    "error_type": "BackupTimeoutError",
                    "error": message,
                    "timeout_seconds": timeout_seconds,
                    "timeout_enforced_by": "parent_process_supervisor",
                },
            )
            print(f"brainlayer backup timed out after {timeout_seconds}s", flush=True)
            return 124
    finally:
        if managed and child.poll() is None:
            child.kill()
            child.wait()
        for shutdown_signal, previous_handler in previous_signal_handlers.items():
            signal.signal(shutdown_signal, previous_handler)


def main() -> int:
    timeout_seconds = _configured_backup_timeout_seconds()
    snapshot_phase = os.environ.get("BRAINLAYER_BACKUP_PHASE") == "snapshot"
    if snapshot_phase:
        timeout_seconds = min(
            timeout_seconds,
            int(
                _configured_positive_number(
                    "BRAINLAYER_BACKUP_SNAPSHOT_TIMEOUT_SECONDS", default_snapshot_timeout_seconds()
                )
            ),
        )
        if budget := os.environ.get("BRAINLAYER_BACKUP_CALENDAR_SNAPSHOT_BUDGET_SECONDS"):
            timeout_seconds = min(timeout_seconds, int(budget))
        if timeout_seconds < 1:
            raise ValueError("snapshot timeout must be positive")
    if _env_flag_enabled(BACKUP_SUPERVISED_CHILD_ENV):
        return _run_backup_process(timeout_seconds)
    from .job_alerts import report

    key = "backup-snapshot" if snapshot_phase else "backup-daily"
    try:
        code = _supervise_backup_process(timeout_seconds)
    except Exception:
        report(key, "Database backup failed; check the backup log")
        raise
    report(key, None if code == 0 else "Database backup failed; check the backup log")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
