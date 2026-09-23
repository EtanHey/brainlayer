"""Daily BrainLayer database backups.

The backup path intentionally uses SQLite's online backup API instead of copying
the database file directly, so live WAL writes are folded into a consistent
snapshot without stopping BrainBar or the enrichment jobs.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import gzip
import hashlib
import json
import math
import os
import queue
import shutil
import signal
import socket
import sqlite3
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

import requests

from .paths import get_db_path

_sleep = time.sleep

DEFAULT_TOKEN_PATH = Path.home() / ".config" / "google-drive-mcp" / "tokens.json"
DEFAULT_CLIENT_PATH = Path.home() / ".config" / "google-drive-mcp" / "gcp-oauth.keys.json"
CANONICAL_MACHINE_ID = "MacBook-Pro"
CANONICAL_FOLDER_PARTS = ["Brain Drive", "06_ARCHIVE", "backups", "brainlayer-db"]
# Compatibility alias for callers that explicitly target the legacy M4 folder.
DEFAULT_FOLDER_PARTS = CANONICAL_FOLDER_PARTS
BACKUP_MACHINE_ID_ENV = "BRAINLAYER_MACHINE_ID"
DRIVE_MACHINE_PROPERTY = "brainlayer_machine"
DEFAULT_STAGING_DIR = Path.home() / ".local" / "share" / "brainlayer" / "backups"
DEFAULT_LOG_PATH = Path.home() / ".local" / "share" / "brainlayer" / "logs" / "backup-daily.log"
DEFAULT_BRAINBAR_SOCKET_PATH = "/tmp/brainbar.sock"
BACKUP_TIMEOUT_ENV = "BRAINLAYER_BACKUP_TIMEOUT_SECONDS"
BACKUP_CLIENT_TIMEOUT_ENV = "BRAINLAYER_BACKUP_CLIENT_TIMEOUT_SECONDS"
BACKUP_ATTEMPT_MAX_AGE_ENV = "BRAINLAYER_BACKUP_ATTEMPT_MAX_AGE_SECONDS"
BACKUP_FULL_VERIFY_ENV = "BRAINLAYER_BACKUP_FULL_VERIFY"
BACKUP_LOG_PATH_ENV = "BRAINLAYER_BACKUP_LOG_PATH"
BACKUP_LOG_PROVENANCE_ENV = "BRAINLAYER_BACKUP_LOG_PROVENANCE"
BACKUP_SUPERVISED_CHILD_ENV = "BRAINLAYER_BACKUP_SUPERVISED_CHILD"
BACKUP_SQLITE_CHECK_TIMEOUT_ENV = "BRAINLAYER_BACKUP_SQLITE_CHECK_TIMEOUT_SECONDS"
BACKUP_DRIVE_RETENTION_ENV = "BRAINLAYER_BACKUP_DRIVE_RETENTION"
DRIVE_UPLOAD_DEADLINE_FLOOR_ENV = "BRAINLAYER_DRIVE_UPLOAD_DEADLINE_FLOOR_SECONDS"
DRIVE_UPLOAD_MIN_BYTES_PER_SECOND_ENV = "BRAINLAYER_DRIVE_UPLOAD_MIN_BYTES_PER_SECOND"
DRIVE_UPLOAD_STALL_MAX_ATTEMPTS_ENV = "BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
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


@dataclass(frozen=True)
class DriveRetentionPolicy:
    keep_latest: int
    filename_prefix: str = ""
    filename_suffix: str = ".db.gz"

    def __post_init__(self) -> None:
        if self.keep_latest < 1:
            raise ValueError("keep_latest must be at least 1")


DAILY_RETENTION = DriveRetentionPolicy(keep_latest=DEFAULT_DRIVE_KEEP)
# Daily and weekly jobs share one Drive folder, so both use the same opt-in Drive cap.
WEEKLY_RETENTION = DriveRetentionPolicy(keep_latest=DEFAULT_DRIVE_KEEP)


@dataclass(frozen=True)
class SQLiteBackupArtifact:
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


def _today() -> str:
    return dt.datetime.now(dt.UTC).date().isoformat()


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
            ["scutil", "--get", "LocalHostName"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None
    if completed is not None and completed.returncode == 0 and completed.stdout.strip():
        return _validate_machine_id(completed.stdout)

    # Non-macOS development and CI hosts do not ship scutil. gethostname is a
    # portability fallback, not the production identity source.
    return _validate_machine_id(socket.gethostname().removesuffix(".local"))


def default_drive_folder_parts(machine_id: str) -> list[str]:
    """Return the per-machine Drive folder, preserving the M4's legacy folder."""
    resolved = _validate_machine_id(machine_id)
    if resolved == CANONICAL_MACHINE_ID:
        return list(CANONICAL_FOLDER_PARTS)
    return [*CANONICAL_FOLDER_PARTS[:-1], f"brainlayer-db-{resolved}"]


def _append_json_log(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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


class DriveUploadStalledError(RuntimeError):
    error_code = "drive_upload_stalled"

    def __init__(self, *, bytes_confirmed: int, total_bytes: int, attempts: int):
        self.bytes_confirmed = bytes_confirmed
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


def _drive_upload_stall_max_attempts() -> int:
    value = _configured_positive_number(DRIVE_UPLOAD_STALL_MAX_ATTEMPTS_ENV, DEFAULT_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS)
    if value != int(value):
        raise ValueError(f"{DRIVE_UPLOAD_STALL_MAX_ATTEMPTS_ENV} must be an integer")
    return int(value)


class _DriveRequestDeadlineExceeded(TimeoutError):
    pass


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


def _sqlite_pragma_check(db_path: Path, pragma_name: str) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(f"PRAGMA {pragma_name}").fetchone()
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


def create_sqlite_backup_artifact(
    db_path: Path,
    output_dir: Path,
    date_stamp: str | None = None,
    socket_path: Path | str | None = None,
    keep_uncompressed: bool = True,
    local_uncompressed_keep: int = DEFAULT_LOCAL_UNCOMPRESSED_KEEP,
    reclamation_status_callback: Callable[[str, str | None], None] | None = None,
) -> SQLiteBackupArtifact:
    """Create a restorable `.db.gz` snapshot through BrainBar's single-writer socket."""
    db_path = Path(db_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    date_stamp = date_stamp or _today()

    if not db_path.exists():
        raise FileNotFoundError(f"BrainLayer database not found: {db_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    writer_probe_error: str | None = None
    try:
        writer_started_at = _brainbar_writer_started_at(socket_path)
    except BackupTimeoutError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        writer_started_at = None
        writer_probe_error = f"{type(exc).__name__}: {exc}"
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
    # The snapshot and its gzip coexist during compression. Each is at most
    # roughly the logical DB size; the fixed margin covers gzip overhead.
    required_bytes = (db_size * 2) + (512 * 1024 * 1024) + surviving_attempt_growth_reserve_bytes
    raw_free_bytes = shutil.disk_usage(output_dir).free
    important_free_bytes = _important_usage_capacity_bytes(output_dir)
    free_bytes = important_free_bytes if important_free_bytes is not None else raw_free_bytes
    print(
        f"Backup capacity: raw={raw_free_bytes} important_usage={important_free_bytes} "
        f"selected={free_bytes} required={required_bytes}",
        flush=True,
    )
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient free space for backup in {output_dir}: "
            f"{free_bytes} bytes free, {required_bytes} bytes required; "
            f"{len(surviving_attempt_paths)} recent attempts reserve "
            f"{surviving_attempt_growth_reserve_bytes} growth bytes"
        )
    final_gz = output_dir / f"{date_stamp}.db.gz"
    final_raw = output_dir / f"{date_stamp}.db"
    sentinel_chunks = 0
    uncompressed_path: Path | None = None

    with tempfile.TemporaryDirectory(prefix="brainlayer-backup-", dir=output_dir) as tmp:
        raw_snapshot = Path(tmp) / f"{date_stamp}.db"
        request_brainbar_vacuum_into(raw_snapshot, socket_path=socket_path, attempt_dir=output_dir)
        sentinel_chunks = _validate_backup_target(raw_snapshot)

        temp_gz = Path(tmp) / final_gz.name
        with raw_snapshot.open("rb") as src, gzip.open(temp_gz, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        shutil.move(str(temp_gz), final_gz)

        if keep_uncompressed:
            temp_raw = Path(tmp) / final_raw.name
            shutil.move(str(raw_snapshot), temp_raw)
            shutil.move(str(temp_raw), final_raw)
            uncompressed_path = final_raw

    deleted = (
        prune_local_uncompressed_snapshots(output_dir, keep_latest=local_uncompressed_keep) if keep_uncompressed else []
    )
    if uncompressed_path is not None and uncompressed_path.name in deleted:
        uncompressed_path = None
    return SQLiteBackupArtifact(
        gzip_path=final_gz,
        uncompressed_path=uncompressed_path,
        sentinel_chunks=sentinel_chunks,
        local_retention_deleted=deleted,
        stale_attempts_deleted=stale_attempts_deleted,
        surviving_attempts=[path.name for path in surviving_attempt_paths],
        surviving_attempt_bytes=surviving_attempt_bytes,
        surviving_attempt_growth_reserve_bytes=surviving_attempt_growth_reserve_bytes,
        attempt_reclamation=attempt_reclamation,
        writer_probe_error=writer_probe_error,
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


def request_brainbar_vacuum_into(
    target_path: Path,
    socket_path: Path | str | None = None,
    timeout_seconds: int | None = None,
    max_attempts: int = 3,
    retry_backoff_seconds: int = 60,
    attempt_dir: Path | None = None,
) -> None:
    target_path = Path(target_path).expanduser()
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
    for attempt in range(1, max_attempts + 1):
        attempt_path = resolved_attempt_dir / f".{target_path.name}.attempt-{attempt}-{uuid.uuid4().hex}"
        request = {
            "jsonrpc": "2.0",
            "id": attempt,
            "method": "tools/call",
            "params": {
                "name": "brain_backup_vacuum_into",
                "arguments": {"target_path": str(attempt_path)},
            },
        }
        terminal_response_received = False
        try:
            response = _send_brainbar_json_request(
                resolved_socket_path,
                request,
                timeout_seconds=resolved_timeout_seconds,
            )
            terminal_response_received = True
            if response.get("error"):
                raise RuntimeError(f"BrainBar backup request failed: {response['error']}")
            result = response.get("result") or {}
            if result.get("isError"):
                content = result.get("content") or []
                text = content[0].get("text") if content and isinstance(content[0], dict) else result
                raise RuntimeError(f"BrainBar backup request failed: {text}")
            if not attempt_path.exists():
                raise RuntimeError(f"BrainBar backup did not create snapshot: {attempt_path}")
            try:
                _validate_backup_target(attempt_path)
            except Exception:
                # A tool response is terminal, so this attempt is no longer being written.
                attempt_path.unlink(missing_ok=True)
                _backup_attempt_completion_marker(attempt_path).unlink(missing_ok=True)
                raise
            os.replace(attempt_path, target_path)
            _backup_attempt_completion_marker(attempt_path).unlink(missing_ok=True)
            for prior_path in unconfirmed_attempt_paths:
                prior_path.unlink(missing_ok=True)
                _backup_attempt_completion_marker(prior_path).unlink(missing_ok=True)
            return
        except BackupTimeoutError:
            raise
        except Exception as exc:
            last_error = exc
            existing_target_note = ""
            if terminal_response_received:
                # BrainBar's request queue is serial. A terminal response proves this
                # attempt and every earlier request are no longer writing.
                for completed_path in [*unconfirmed_attempt_paths, attempt_path]:
                    completed_path.unlink(missing_ok=True)
                    _backup_attempt_completion_marker(completed_path).unlink(missing_ok=True)
                unconfirmed_attempt_paths.clear()
            elif attempt_path.exists():
                # A lost response does not prove VACUUM INTO is finished. Never inspect,
                # unlink, or promote a path that BrainBar may still be writing.
                unconfirmed_attempt_paths.append(attempt_path)
                existing_target_note = f"; preserving isolated attempt target: {attempt_path}"
            if attempt >= max_attempts:
                print(
                    f"BrainBar vacuum snapshot attempt {attempt}/{max_attempts} failed: {exc}{existing_target_note}",
                    flush=True,
                )
                break
            print(
                f"BrainBar vacuum snapshot attempt {attempt}/{max_attempts} failed: {exc}{existing_target_note}; "
                f"retrying in {retry_backoff_seconds}s",
                flush=True,
            )
            _sleep(retry_backoff_seconds)
    if last_error is not None:
        raise last_error


def _send_brainbar_json_request(
    socket_path: Path,
    request: dict[str, Any],
    timeout_seconds: int | None,
) -> dict[str, Any]:
    payload = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout_seconds)
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


def _atomic_write_text(path: Path, content: str) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(content)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def get_drive_credentials(token_path: Path = DEFAULT_TOKEN_PATH, client_path: Path = DEFAULT_CLIENT_PATH):
    """Load and refresh Google Drive OAuth credentials from the existing MCP auth files."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_path = Path(token_path).expanduser()
    client_path = Path(client_path).expanduser()
    if not token_path.exists():
        raise FileNotFoundError(f"Google Drive token file not found: {token_path}")
    if not client_path.exists():
        raise FileNotFoundError(f"Google OAuth client file not found: {client_path}")

    lock_path = token_path.with_suffix(token_path.suffix + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        token_data = json.loads(token_path.read_text())
        client_data = json.loads(client_path.read_text())["installed"]

        expiry = token_data.get("expiry")
        if not expiry and token_data.get("expiry_date"):
            expiry = dt.datetime.fromtimestamp(int(token_data["expiry_date"]) / 1000, tz=dt.UTC).isoformat()

        parsed_expiry = dt.datetime.fromisoformat(expiry.replace("Z", "+00:00")) if expiry else None
        if parsed_expiry and parsed_expiry.tzinfo:
            parsed_expiry = parsed_expiry.astimezone(dt.UTC).replace(tzinfo=None)
        elif parsed_expiry:
            parsed_expiry = parsed_expiry.replace(tzinfo=None)

        creds = Credentials(
            token=token_data.get("access_token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=client_data["token_uri"],
            client_id=client_data["client_id"],
            client_secret=client_data["client_secret"],
            scopes=token_data.get("scope", " ".join(DRIVE_SCOPES)).split(),
            expiry=parsed_expiry,
        )

        # google-auth Credentials.expired compares against a naive UTC helper, so keep expiry comparisons naive UTC.
        refresh_before = dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(hours=2)
        if creds.expired or not creds.valid or (creds.expiry and creds.expiry < refresh_before):
            creds.refresh(Request())
            token_data["access_token"] = creds.token
            token_data["expiry"] = creds.expiry.isoformat() if creds.expiry else None
            _atomic_write_text(token_path, json.dumps(token_data, indent=2, sort_keys=True) + "\n")

    return creds


def build_drive_service(token_path: Path = DEFAULT_TOKEN_PATH, client_path: Path = DEFAULT_CLIENT_PATH):
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


def assert_drive_folder_owned_by_machine(service: Any, *, folder_id: str, machine_id: str) -> None:
    """Refuse a target folder containing snapshots owned by another host."""
    other_machine_ids: set[str] = set()
    page_token = None
    while True:
        result = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken,files(id,name,appProperties)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
            )
            .execute()
        )
        for item in result.get("files", []):
            if _parse_snapshot_date(item.get("name", "")) is None:
                continue
            properties = item.get("appProperties")
            owner = properties.get(DRIVE_MACHINE_PROPERTY) if isinstance(properties, dict) else None
            # Snapshots predating FU #99 belong to the canonical M4. A new host
            # must never silently adopt the legacy pool.
            resolved_owner = owner if isinstance(owner, str) and owner else CANONICAL_MACHINE_ID
            if resolved_owner != machine_id:
                other_machine_ids.add(resolved_owner)
        page_token = result.get("nextPageToken")
        if not page_token:
            break
    if other_machine_ids:
        raise DriveFolderOwnedByOtherMachineError(
            folder_id=folder_id,
            machine_id=machine_id,
            other_machine_ids=other_machine_ids,
        )


def upload_file_to_drive_raw(
    file_path: Path,
    folder_id: str,
    credentials: Any,
    *,
    machine_id: str,
    chunk_size: int = 8 * 1024 * 1024,
    max_attempts: int = 30,
) -> dict[str, Any]:
    """Upload large backups with Drive's raw resumable protocol."""
    file_path = Path(file_path)
    total = file_path.stat().st_size
    metadata = {
        "name": file_path.name,
        "parents": [folder_id],
        "appProperties": {DRIVE_MACHINE_PROPERTY: _validate_machine_id(machine_id)},
    }
    init = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&supportsAllDrives=true"
        # md5Checksum is REQUIRED: retention coverage compares it against the surviving
        # object. Without it the integrity branch silently becomes dead code (PR #815 review).
        "&fields=id,name,size,md5Checksum,appProperties",
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "application/gzip",
            "X-Upload-Content-Length": str(total),
        },
        data=json.dumps(metadata),
        timeout=60,
    )
    init.raise_for_status()
    upload_url = init.headers["Location"]

    sent = 0
    deadline_floor = _configured_positive_number(
        DRIVE_UPLOAD_DEADLINE_FLOOR_ENV,
        DEFAULT_DRIVE_UPLOAD_DEADLINE_FLOOR_SECONDS,
        maximum=threading.TIMEOUT_MAX,
    )
    minimum_rate = _configured_positive_number(
        DRIVE_UPLOAD_MIN_BYTES_PER_SECOND_ENV, DEFAULT_DRIVE_UPLOAD_MIN_BYTES_PER_SECOND
    )
    stall_limit = _drive_upload_stall_max_attempts()
    stalled_attempts = 0
    session = requests.Session()
    try:
        with file_path.open("rb") as handle:
            while sent < total:
                handle.seek(sent)
                expected = min(chunk_size, total - sent)
                chunk = handle.read(expected)
                if len(chunk) != expected:
                    raise RuntimeError(
                        f"Backup file changed during upload: expected {expected} bytes, got {len(chunk)}"
                    )
                start = sent
                end = sent + len(chunk) - 1
                headers = {
                    "Authorization": f"Bearer {credentials.token}",
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"bytes {start}-{end}/{total}",
                }
                for attempt in range(1, max_attempts + 1):
                    try:
                        deadline = max(deadline_floor, len(chunk) / minimum_rate)
                        response = _drive_put_with_deadline(
                            session, upload_url, headers=headers, data=chunk, deadline=deadline
                        )
                        if response.status_code in {200, 201}:
                            return response.json()
                        if response.status_code == 308:
                            uploaded_range = response.headers.get("Range")
                            confirmed = (
                                int(uploaded_range.rsplit("-", 1)[1]) + 1
                                if uploaded_range and "-" in uploaded_range
                                else sent
                            )
                            if confirmed < sent or confirmed > total:
                                raise RuntimeError(
                                    f"Drive upload returned invalid confirmed offset {confirmed}; "
                                    f"expected {sent}..{total}"
                                )
                            if confirmed == sent:
                                raise _DriveRequestDeadlineExceeded("Drive upload returned no newly confirmed bytes")
                            sent = confirmed
                            stalled_attempts = 0
                            print(f"drive upload progress: {sent}/{total} bytes", flush=True)
                            break
                        if response.status_code in {429, 500, 502, 503, 504}:
                            raise RuntimeError(f"retryable HTTP {response.status_code}: {response.text[:200]}")
                        response.raise_for_status()
                    except _DriveRequestDeadlineExceeded:
                        previously_confirmed = sent
                        stalled_attempts += 1
                        query_headers = {
                            "Authorization": f"Bearer {credentials.token}",
                            "Content-Length": "0",
                            "Content-Range": f"bytes */{total}",
                        }
                        while True:
                            session = requests.Session()
                            try:
                                status = _drive_put_with_deadline(
                                    session, upload_url, headers=query_headers, data=b"", deadline=deadline_floor
                                )
                                if status.status_code in {200, 201}:
                                    return status.json()
                                if status.status_code in {429, 500, 502, 503, 504}:
                                    raise requests.RequestException(f"offset query HTTP {status.status_code}")
                                if status.status_code != 308:
                                    status.raise_for_status()
                                uploaded_range = status.headers.get("Range")
                                confirmed = (
                                    int(uploaded_range.rsplit("-", 1)[1]) + 1
                                    if uploaded_range and "-" in uploaded_range
                                    else 0
                                )
                                if confirmed < sent or confirmed > total:
                                    raise RuntimeError(
                                        f"Drive offset query returned invalid confirmed offset {confirmed}; "
                                        f"expected {sent}..{total}"
                                    )
                                sent = confirmed
                                break
                            except (_DriveRequestDeadlineExceeded, requests.RequestException) as query_error:
                                session.close()
                                stalled_attempts += 1
                                if stalled_attempts >= stall_limit:
                                    raise DriveUploadStalledError(
                                        bytes_confirmed=sent, total_bytes=total, attempts=stalled_attempts
                                    ) from query_error
                                sleep_seconds = min(60, 2**stalled_attempts)
                                print(
                                    f"drive upload offset query retry attempt={stalled_attempts}/{stall_limit}: "
                                    f"{query_error}; sleeping {sleep_seconds}s",
                                    flush=True,
                                )
                                _sleep(sleep_seconds)
                        if sent > previously_confirmed:
                            stalled_attempts = 0
                            print(f"drive upload progress: {sent}/{total} bytes", flush=True)
                            break
                        if stalled_attempts >= stall_limit:
                            raise DriveUploadStalledError(
                                bytes_confirmed=sent, total_bytes=total, attempts=stalled_attempts
                            )
                        sleep_seconds = min(60, 2**stalled_attempts)
                        print(
                            f"drive upload stalled; confirmed={sent}/{total} "
                            f"attempt={stalled_attempts}/{stall_limit}; sleeping {sleep_seconds}s",
                            flush=True,
                        )
                        _sleep(sleep_seconds)
                        break
                    except Exception as exc:
                        if attempt >= max_attempts:
                            raise
                        sleep_seconds = min(60, 2 ** min(attempt, 6))
                        print(
                            f"drive upload retry chunk={start}-{end} attempt={attempt}/{max_attempts}: {exc}; "
                            f"sleeping {sleep_seconds}s",
                            flush=True,
                        )
                        _sleep(sleep_seconds)
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
            _, done = downloader.next_chunk()
    return destination


def _parse_snapshot_date(name: str, *, prefix: str = "", suffix: str = ".db.gz") -> dt.date | None:
    if prefix and not name.startswith(prefix):
        return None
    if not name.endswith(suffix):
        return None
    stem = name[len(prefix) : len(name) - len(suffix)]
    try:
        return dt.date.fromisoformat(stem[:10])
    except ValueError:
        return None


def _verified_snapshot_names_from_log(log_path: Path) -> set[str]:
    verified: set[str] = set()
    try:
        lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return verified
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("uploaded") is not True or payload.get("verified") is not True:
            continue
        snapshot = payload.get("snapshot")
        if isinstance(snapshot, str):
            name = Path(snapshot).name
            if _parse_snapshot_date(name) is not None:
                verified.add(name)
    return verified


def prune_local_gzip_snapshots(
    output_dir: Path,
    *,
    keep_latest: int = DEFAULT_LOCAL_COMPRESSED_KEEP,
    verified_drive_names: set[str] | None = None,
) -> list[str]:
    """Cap local gzip snapshots without deleting an archive lacking remote coverage."""
    if keep_latest < 1:
        raise ValueError("keep_latest must be at least 1")
    output_dir = Path(output_dir).expanduser()
    if not output_dir.exists():
        return []

    verified_names = verified_drive_names or set()
    verified_dates = {parsed for name in verified_names if (parsed := _parse_snapshot_date(name)) is not None}
    dated: list[tuple[dt.date, Path]] = []
    for path in output_dir.iterdir():
        parsed = _parse_snapshot_date(path.name)
        if parsed and path.is_file():
            dated.append((parsed, path))

    dated.sort(key=lambda pair: pair[0], reverse=True)
    deleted: list[str] = []
    for index, (snapshot_date, path) in enumerate(dated):
        if index < keep_latest:
            continue
        if path.name not in verified_names:
            newer_verified = sum(date > snapshot_date for date in verified_dates)
            if newer_verified < keep_latest:
                continue
        path.unlink(missing_ok=True)
        deleted.append(path.name)
    return deleted


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
        shutil.copyfileobj(src, dst, length=1024 * 1024)


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
    """Run restore verification against the local gzip or the downloaded Drive copy."""
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
        with tempfile.TemporaryDirectory(prefix="brainlayer-restore-verify-", dir=artifact.gzip_path.parent) as tmp:
            tmp_dir = Path(tmp)
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

            _gunzip_test(verify_path)
            result["gzip_test"] = True
            restored = tmp_dir / "restored.db"
            _decompress_gzip_to(verify_path, restored)
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
        result.setdefault("gzip_test", False)
        result["verification_error"] = str(exc)
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
    metadata = (
        service.files()
        .get(fileId=file_id, fields="id,name,size,trashed,appProperties", supportsAllDrives=True)
        .execute()
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
    folder_parts: list[str] = DEFAULT_FOLDER_PARTS,
    retention_policy: DriveRetentionPolicy = DAILY_RETENTION,
) -> list[str]:
    """Trash Drive snapshots older than the explicitly enabled retention window."""
    folder_id = ensure_drive_folder_chain(service, folder_parts)
    files: list[dict[str, str]] = []
    page_token = None
    while True:
        result = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken,files(id,name)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
            )
            .execute()
        )
        files.extend(result.get("files", []))
        page_token = result.get("nextPageToken")
        if not page_token:
            break
    dated = []
    for item in files:
        parsed = _parse_snapshot_date(
            item.get("name", ""),
            prefix=retention_policy.filename_prefix,
            suffix=retention_policy.filename_suffix,
        )
        if parsed:
            dated.append((parsed, item))

    dated.sort(key=lambda pair: pair[0], reverse=True)
    keep_ids = {item["id"] for _, item in dated[: retention_policy.keep_latest]}

    trashed: list[str] = []
    for _, item in dated:
        if item["id"] in keep_ids:
            continue
        service.files().update(
            fileId=item["id"],
            body={"trashed": True},
            supportsAllDrives=True,
        ).execute()
        trashed.append(item["name"])
    return trashed


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
) -> dict[str, Any]:
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

    def record_reclamation_status(status: str, error: str | None) -> None:
        result["attempt_reclamation"] = status
        result["writer_probe_error"] = error

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
        artifact = create_sqlite_backup_artifact(
            resolved_db_path,
            staging_dir,
            date_stamp=resolved_date_stamp,
            reclamation_status_callback=record_reclamation_status,
        )
        snapshot = artifact.gzip_path
        snapshot_size = snapshot.stat().st_size
        result.update(
            {
                "snapshot": str(snapshot),
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
                "sentinel_snapshot_chunks": artifact.sentinel_chunks,
                "bytes": snapshot_size,
            }
        )
        if upload:
            credentials = get_drive_credentials()
            service = build_drive_service()
            folder_id = ensure_drive_folder_chain(service, resolved_folder_parts)
            assert_drive_folder_owned_by_machine(
                service,
                folder_id=folder_id,
                machine_id=resolved_machine_id,
            )
            uploaded = upload_file_to_drive_raw(
                snapshot,
                folder_id,
                credentials,
                machine_id=resolved_machine_id,
            )
            file_id = uploaded.get("id")
            if not file_id:
                raise RuntimeError(f"Drive upload response missing file id: {uploaded!r}")
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
            if result["verified"]:
                if remove_local_after_upload:
                    snapshot.unlink()
                    result["local_removed"] = True
                verified_drive_names = _verified_snapshot_names_from_log(resolved_log_path)
                verified_drive_names.add(snapshot.name)
                local_gzip_deleted = prune_local_gzip_snapshots(
                    snapshot.parent,
                    verified_drive_names=verified_drive_names,
                )
                deleted = (
                    prune_drive_backups(
                        service,
                        folder_parts=resolved_folder_parts,
                        retention_policy=retention_policy,
                    )
                    if retention_enabled
                    else []
                )
            else:
                local_gzip_deleted = []
                deleted = []
            result.update(
                {
                    "uploaded": True,
                    "drive_file": uploaded,
                    "retention_deleted": deleted,
                    "local_gzip_retention_deleted": local_gzip_deleted,
                }
            )
    except Exception as exc:
        result.update({"error_type": type(exc).__name__, "error": str(exc)})
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
        _append_json_log(resolved_log_path, result)
    return result


def _run_backup_process(timeout_seconds: int) -> int:
    previous_alarm_handler = None
    previous_alarm_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_backup_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        resolved_db_path = get_db_path()
        result = run_backup(
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
    except BackupTimeoutError:
        print(f"brainlayer backup timed out after {timeout_seconds}s", flush=True)
        return 124
    except Exception as exc:
        print(f"brainlayer backup failed: {exc}\n{traceback.format_exc()}", flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("verified", True) else 1


def _supervise_backup_process(timeout_seconds: int, *, command: list[str] | None = None) -> int:
    child_env = os.environ.copy()
    child_env[BACKUP_SUPERVISED_CHILD_ENV] = "1"
    child = subprocess.Popen(
        command or [sys.executable, "-m", "brainlayer.backup_daily"],
        env=child_env,
        start_new_session=True,
    )
    previous_signal_handlers: dict[int, Any] = {}

    def forward_shutdown_signal(signum, frame) -> None:  # noqa: ARG001
        if child.poll() is None:
            os.killpg(child.pid, signum)
        raise SystemExit(128 + signum)

    for shutdown_signal in (signal.SIGTERM, signal.SIGINT):
        previous_signal_handlers[shutdown_signal] = signal.getsignal(shutdown_signal)
        signal.signal(shutdown_signal, forward_shutdown_signal)
    try:
        try:
            return child.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            message = f"backup exceeded configured wall-clock timeout ({timeout_seconds}s)"
            resolved_db_path = get_db_path()
            _append_json_log(
                _backup_log_path(None, db_path=resolved_db_path, env=os.environ),
                {
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
        for shutdown_signal, previous_handler in previous_signal_handlers.items():
            signal.signal(shutdown_signal, previous_handler)


def main() -> int:
    timeout_seconds = _configured_backup_timeout_seconds()
    if _env_flag_enabled(BACKUP_SUPERVISED_CHILD_ENV):
        return _run_backup_process(timeout_seconds)
    return _supervise_backup_process(timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
