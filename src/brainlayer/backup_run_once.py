"""One local backup generation with no retention, upload, or cleanup side effects."""

from __future__ import annotations

import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable

from . import backup_daily, jsonl_backup


class BackupRunOnceError(RuntimeError):
    """A backup generation could not be completed."""


class BackupRunOnceBusyError(BackupRunOnceError):
    """A destination-bound run is already active in another process."""


_DESTINATION_BIND_LOCK = threading.Lock()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, int]:
    result = path.stat()
    return {"st_dev": result.st_dev, "st_ino": result.st_ino}


def _acquire_destination_process_lock(destination_fd: int, identity: tuple[int, int]) -> None:
    """Lock the selected destination directory inode without a mutable lock path."""
    try:
        fcntl.flock(destination_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            raise BackupRunOnceBusyError(
                f"backup run already active for destination identity: {identity}"
            ) from exc
        raise BackupRunOnceError(f"unable to lock backup destination identity: {identity}") from exc


class _BoundSource(AbstractContextManager):
    """Read one no-follow descriptor and reject replacement during archiving."""

    def __init__(self, fd: int, path: Path) -> None:
        self._handle = os.fdopen(fd, "rb")
        self._before = os.fstat(fd)
        self.name = str(path)

    def __enter__(self) -> "_BoundSource":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            after = os.fstat(self._handle.fileno())
            if exc_type is None and (
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                != (self._before.st_dev, self._before.st_ino, self._before.st_size, self._before.st_mtime_ns)
            ):
                raise BackupRunOnceError(f"transcript source changed while being archived: {self.name}")
        finally:
            self._handle.close()

    def fileno(self) -> int:
        return self._handle.fileno()

    def read(self, size: int = -1) -> bytes:
        return self._handle.read(size)


def _open_candidate_bound(
    candidate: jsonl_backup.JsonlCandidate, roots: list[jsonl_backup.BackupSourceRoot]
) -> _BoundSource:
    candidate_abs = Path(os.path.abspath(candidate.path.expanduser()))
    for root in roots:
        root_abs = Path(os.path.abspath(root.path.expanduser()))
        try:
            relative = candidate_abs.relative_to(root_abs)
        except ValueError:
            continue
        root_fd = os.open(root_abs, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fd = root_fd
        try:
            parts = relative.parts
            if not parts:
                raise BackupRunOnceError(f"transcript candidate is a directory: {candidate.path}")
            for part in parts[:-1]:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                if fd != root_fd:
                    os.close(fd)
                fd = next_fd
            leaf_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
            if fd != root_fd:
                os.close(fd)
            os.close(root_fd)
            fd = root_fd = -1
            if not stat.S_ISREG(os.fstat(leaf_fd).st_mode):
                os.close(leaf_fd)
                raise BackupRunOnceError(f"transcript candidate is not a regular file: {candidate.path}")
            return _BoundSource(leaf_fd, candidate.path)
        except OSError as exc:
            if fd >= 0 and fd != root_fd:
                os.close(fd)
            if root_fd >= 0:
                os.close(root_fd)
            raise BackupRunOnceError(f"transcript candidate cannot be opened safely: {candidate.path}") from exc
        except Exception:
            if fd >= 0 and fd != root_fd:
                os.close(fd)
            if root_fd >= 0:
                os.close(root_fd)
            raise
    raise BackupRunOnceError(f"transcript candidate escapes configured roots: {candidate.path}")


def _validate_roots(
    transcript_roots: list[Path | jsonl_backup.BackupSourceRoot],
) -> list[jsonl_backup.BackupSourceRoot]:
    roots = [jsonl_backup._source_root_config(root) for root in transcript_roots]
    for root in roots:
        path = root.path.expanduser()
        if not path.exists() or not path.is_dir():
            raise BackupRunOnceError(f"configured transcript root is missing or not a directory: {path}")
    return roots


def _validate_candidates(
    candidates: list[jsonl_backup.JsonlCandidate], roots: list[jsonl_backup.BackupSourceRoot]
) -> None:
    for candidate in candidates:
        if candidate.path.is_symlink():
            raise BackupRunOnceError(f"transcript candidate is a symlink: {candidate.path}")
        resolved = candidate.path.resolve()
        allowed = any(resolved.is_relative_to(root.path.expanduser().resolve()) for root in roots)
        if not allowed:
            raise BackupRunOnceError(f"transcript candidate escapes configured roots: {candidate.path}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _fsync_file(temporary)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _offline_snapshot(
    source: Path,
    target: Path,
    *,
    expected_source_identity: tuple[int, int] | None = None,
    total_deadline_seconds: float = 300.0,
    stalled_progress_seconds: float = 30.0,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    source = source.expanduser()
    if not source.is_absolute():
        raise BackupRunOnceError("offline backup source must be an absolute path")
    if expected_source_identity is not None:
        observed = _identity(source)
        if tuple(observed.values()) != expected_source_identity:
            raise BackupRunOnceError(
                f"offline backup source changed before snapshot: expected={expected_source_identity} actual={tuple(observed.values())}"
            )
    source_uri = f"file:{source.resolve()}?mode=ro"
    if total_deadline_seconds <= 0 or stalled_progress_seconds <= 0:
        raise ValueError("offline backup deadlines must be positive")
    started = monotonic()
    last_progress = started
    previous_remaining: int | None = None
    calls = busy_count = locked_count = restart_count = 0

    def progress(status: int, remaining: int, total: int) -> None:
        nonlocal calls, busy_count, last_progress, previous_remaining, restart_count
        nonlocal locked_count
        now = monotonic()
        calls += 1
        if status == sqlite3.SQLITE_BUSY:
            busy_count += 1
        elif status == sqlite3.SQLITE_LOCKED:
            locked_count += 1
        if status in {sqlite3.SQLITE_OK, sqlite3.SQLITE_DONE}:
            if previous_remaining is None or remaining < previous_remaining:
                last_progress = now
            elif remaining > previous_remaining:
                restart_count += 1
            previous_remaining = remaining
        elapsed = now - started
        stalled = now - last_progress
        status_name = {
            sqlite3.SQLITE_OK: "SQLITE_OK",
            sqlite3.SQLITE_BUSY: "SQLITE_BUSY",
            sqlite3.SQLITE_LOCKED: "SQLITE_LOCKED",
            sqlite3.SQLITE_DONE: "SQLITE_DONE",
        }.get(status, f"SQLITE_{status}")
        diagnostics = (
            f"status={status_name} remaining={remaining} total={total} "
            f"elapsed={elapsed:.3f}s stalled={stalled:.3f}s calls={calls} "
            f"busy={busy_count} locked={locked_count} restarts={restart_count}"
        )
        if elapsed >= total_deadline_seconds:
            raise BackupRunOnceError(f"offline SQLite backup total deadline exceeded: {diagnostics}")
        if stalled >= stalled_progress_seconds:
            raise BackupRunOnceError(f"offline SQLite backup stalled: {diagnostics}")

    source_db: sqlite3.Connection | None = None
    target_db: sqlite3.Connection | None = None
    primary_error: BaseException | None = None
    close_error: BaseException | None = None
    try:
        busy_timeout = min(5.0, total_deadline_seconds, stalled_progress_seconds)
        source_db = sqlite3.connect(source_uri, uri=True, timeout=busy_timeout)
        try:
            source_db.execute("BEGIN")
            source_db.execute("SELECT count(*) FROM sqlite_schema").fetchone()
        except sqlite3.Error as exc:
            raise BackupRunOnceError(f"offline SQLite read snapshot acquisition failed: {exc}") from exc
        target_db = sqlite3.connect(str(target), timeout=busy_timeout)
        source_db.backup(target_db, pages=1000, sleep=0.01, progress=progress)
        if expected_source_identity is not None:
            observed = _identity(source)
            if tuple(observed.values()) != expected_source_identity:
                raise BackupRunOnceError(
                    f"offline backup source changed after snapshot: expected={expected_source_identity} actual={tuple(observed.values())}"
                )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        for connection in (target_db, source_db):
            if connection is None:
                continue
            try:
                connection.close()
            except BaseException as exc:
                if close_error is None:
                    close_error = exc
        if primary_error is None and close_error is not None:
            raise close_error
    return backup_daily._validate_backup_target(target)


def _socket_snapshot(
    socket_path: Path,
    target: Path,
    generation: Path,
    *,
    expected_source_path: Path | None = None,
    expected_source_open_generation: str | None = None,
) -> dict[str, Any]:
    # BrainBar resolves the request in a separate process; never send the
    # fchdir-bound relative paths used by the local writer.
    outgoing_target = target.expanduser().resolve()
    outgoing_generation = generation.expanduser().resolve()
    attempt_path = outgoing_generation / f".{outgoing_target.name}.attempt-{uuid.uuid4().hex}"
    request = {
        "jsonrpc": "2.0",
        "id": "backup-run-once",
        "method": "tools/call",
        "params": {
            "name": "brain_backup_vacuum_into",
            "arguments": {"target_path": str(attempt_path)},
        },
    }
    def discard_terminal_attempt() -> None:
        attempt_path.unlink(missing_ok=True)
        backup_daily._backup_attempt_completion_marker(attempt_path).unlink(missing_ok=True)

    # Do not enter the terminal-response finally until JSON returned. A timeout
    # or lost response leaves its isolated path intact because BrainBar may
    # still be writing it.
    response = backup_daily._send_brainbar_json_request(
        Path(socket_path).expanduser(),
        request,
        timeout_seconds=backup_daily._configured_backup_client_timeout_seconds(),
    )
    try:
        if not isinstance(response, dict):
            raise BackupRunOnceError("BrainBar backup response was not a JSON object")
        if response.get("error"):
            raise BackupRunOnceError(f"BrainBar backup request failed: {response['error']}")
        result = response.get("result") or {}
        if not isinstance(result, dict):
            raise BackupRunOnceError("BrainBar backup response result was not an object")
        if result.get("isError"):
            content = result.get("content") or []
            text = content[0].get("text") if content and isinstance(content[0], dict) else result
            raise BackupRunOnceError(f"BrainBar backup request failed: {text}")
        content = result.get("content") or []
        text = content[0].get("text") if content and isinstance(content[0], dict) else None
        try:
            authority = json.loads(text) if isinstance(text, str) else None
        except json.JSONDecodeError as exc:
            raise BackupRunOnceError("BrainBar backup response did not contain source authority") from exc
        if not isinstance(authority, dict) or authority.get("status") != "ok":
            raise BackupRunOnceError("BrainBar backup response did not contain a successful source authority")
        source_path = authority.get("source_db_filename")
        source_generation = authority.get("source_open_generation")
        before = authority.get("source_observed_path_identity_before")
        after = authority.get("source_observed_path_identity_after")
        if not all(
            [
                isinstance(source_path, str) and source_path,
                isinstance(source_generation, str) and source_generation,
                isinstance(before, dict),
                isinstance(after, dict),
            ]
        ):
            raise BackupRunOnceError("BrainBar backup response omitted connection-derived source authority")
        for observation in (before, after):
            if observation.get("kind") != "pathname_observation_not_fd_proof":
                raise BackupRunOnceError("BrainBar backup response mislabeled observed source pathname identity")
        if expected_source_path is not None and source_path != str(Path(expected_source_path).expanduser().resolve()):
            expected_path = Path(expected_source_path).expanduser().resolve()
            raise BackupRunOnceError(
                f"BrainBar source mismatch: expected={expected_path} actual={source_path}"
            )
        if expected_source_open_generation is not None and source_generation != expected_source_open_generation:
            raise BackupRunOnceError(
                f"BrainBar source generation mismatch: expected={expected_source_open_generation} actual={source_generation}"
            )
        if not attempt_path.exists():
            raise BackupRunOnceError(f"BrainBar backup did not create snapshot: {attempt_path}")
        chunks = backup_daily._validate_backup_target(attempt_path)
        os.replace(attempt_path, outgoing_target)
        return {
            "chunks": chunks,
            "source_db_filename": source_path,
            "source_open_generation": source_generation,
            "source_observed_path_identity_before": before,
            "source_observed_path_identity_after": after,
        }
    finally:
        # A terminal response proves BrainBar is no longer writing this path.
        discard_terminal_attempt()


def _run_backup_once_bound(
    *,
    destination: Path,
    transcript_roots: list[Path | jsonl_backup.BackupSourceRoot],
    socket_path: Path | None = None,
    offline_db_path: Path | None = None,
    expected_source_path: Path | None = None,
    expected_source_open_generation: str | None = None,
    offline_source_open_generation: str | None = None,
    expected_offline_source_identity: tuple[int, int] | None = None,
    generation_id: str | None = None,
    now: dt.datetime | None = None,
    expected_destination_identity: tuple[int, int],
) -> dict[str, Any]:
    """Create one locally verified DB+JSONL generation."""
    destination = Path(destination).expanduser()
    if not destination.exists() or not destination.is_dir():
        raise BackupRunOnceError(f"backup destination must already exist as a directory: {destination}")
    resolved_destination = destination.resolve()
    actual_identity = _identity(destination)
    if tuple(actual_identity.values()) != tuple(expected_destination_identity):
        raise BackupRunOnceError(
            f"backup destination identity changed: expected={expected_destination_identity} actual={tuple(actual_identity.values())}"
        )
    if (socket_path is None) == (offline_db_path is None):
        raise BackupRunOnceError("choose exactly one of socket_path or offline_db_path")
    if not transcript_roots:
        raise BackupRunOnceError("transcript_roots must be explicitly provided")
    validated_roots = _validate_roots(transcript_roots)
    # Root validation may cross a mount replacement window; rebind the selected
    # destination before creating any generation output.
    actual_identity = _identity(destination)
    if tuple(actual_identity.values()) != tuple(expected_destination_identity):
        raise BackupRunOnceError(
            f"backup destination identity changed: expected={expected_destination_identity} actual={tuple(actual_identity.values())}"
        )
    timestamp = now or dt.datetime.now(dt.UTC)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    generation = destination / f".brainlayer-backup-{stamp}-{generation_id or uuid.uuid4().hex}"
    generation.mkdir(mode=0o700, parents=False, exist_ok=False)
    incomplete = generation / "receipt.incomplete.json"
    receipt = {
        "status": "incomplete",
        "generation": generation.name,
        "started_at": timestamp.isoformat(),
        "remote_verified": False,
        "cleanup_permitted": False,
        "destination_path": str(resolved_destination),
        "destination_identity": _identity(destination),
        "transcript_roots": [str(root.path.expanduser().resolve()) for root in validated_roots],
    }
    _atomic_json(incomplete, receipt)
    try:
        raw_db = generation / "brainlayer.db"
        db_source = "brainbar_socket" if socket_path is not None else "explicit_offline_db"
        socket_snapshot = (
            _socket_snapshot(
                Path(socket_path),
                raw_db,
                generation,
                expected_source_path=expected_source_path,
                expected_source_open_generation=expected_source_open_generation,
            )
            if socket_path is not None
            else None
        )
        if socket_snapshot is not None:
            chunks = socket_snapshot["chunks"]
        elif expected_offline_source_identity is None:
            chunks = _offline_snapshot(Path(offline_db_path), raw_db)
        else:
            chunks = _offline_snapshot(
                Path(offline_db_path), raw_db,
                expected_source_identity=expected_offline_source_identity,
            )
        _fsync_file(raw_db)
        integrity = backup_daily._sqlite_pragma_check(raw_db, "quick_check")
        if integrity != "ok":
            raise BackupRunOnceError(f"SQLite quick_check failed: {integrity}")
        candidates = jsonl_backup._discover_jsonl_candidates(validated_roots)
        _validate_candidates(candidates, validated_roots)
        if not candidates:
            raise BackupRunOnceError("no explicitly wired transcript roots produced candidates")
        archive, digests, sizes = jsonl_backup.create_jsonl_bundle_with_digests(
            candidates,
            generation,
            date_stamp=stamp,
            source_opener=lambda candidate: _open_candidate_bound(candidate, validated_roots),
        )
        verified = jsonl_backup.verify_jsonl_bundle(
            archive, expected_candidates=candidates, expected_digests=digests, expected_sizes=sizes
        )
        if not verified.get("verified"):
            raise BackupRunOnceError(verified.get("verification_error", "JSONL verification failed"))
        complete = {
            "status": "local_verified",
            "generation": generation.name,
            "completed_at": dt.datetime.now(dt.UTC).isoformat(),
            "db_source_mode": db_source,
            "db_snapshot": raw_db.name,
            "db_snapshot_chunks": chunks,
            "db_quick_check": integrity,
            "db_snapshot_bytes": raw_db.stat().st_size,
            "db_snapshot_sha256": _sha256(raw_db),
            "jsonl_bundle": archive.name,
            "jsonl_bundle_bytes": archive.stat().st_size,
            "jsonl_bundle_sha256": _sha256(archive),
            "jsonl_file_count": len(candidates),
            "jsonl_sources": [
                {
                    "path": candidate.path.resolve().as_posix(),
                    "bytes": sizes[candidate.path.as_posix()],
                    "sha256": digests[candidate.path.as_posix()],
                }
                for candidate in candidates
            ],
            "db_source_path": str(Path(offline_db_path).expanduser().resolve())
            if offline_db_path is not None
            else socket_snapshot["source_db_filename"],
            "db_transport_socket": None if socket_path is None else str(Path(socket_path).expanduser().resolve()),
            "db_source_open_generation": (
                offline_source_open_generation
                if socket_snapshot is None
                else socket_snapshot["source_open_generation"]
            ),
            "db_source_observed_path_identity_before": (
                {"kind": "scheduled_saved_source_binding", "st_dev": expected_offline_source_identity[0], "st_ino": expected_offline_source_identity[1]}
                if socket_snapshot is None and expected_offline_source_identity is not None
                else (None if socket_snapshot is None else socket_snapshot["source_observed_path_identity_before"])
            ),
            "db_source_observed_path_identity_after": (
                {"kind": "scheduled_saved_source_binding", "st_dev": expected_offline_source_identity[0], "st_ino": expected_offline_source_identity[1]}
                if socket_snapshot is None and expected_offline_source_identity is not None
                else (None if socket_snapshot is None else socket_snapshot["source_observed_path_identity_after"])
            ),
            "scheduled_offline_source_binding": (
                {
                    "version": 1,
                    "path": str(Path(offline_db_path).expanduser().resolve()),
                    "st_dev": expected_offline_source_identity[0],
                    "st_ino": expected_offline_source_identity[1],
                    "source_open_generation": offline_source_open_generation,
                }
                if socket_snapshot is None and expected_offline_source_identity is not None
                else None
            ),
            "destination_path": str(resolved_destination),
            "destination_identity": _identity(destination),
            "local_verified": True,
            "remote_verified": False,
            "remote_status": "unverified",
            "cleanup_permitted": False,
        }
        actual_identity = _identity(destination)
        if tuple(actual_identity.values()) != tuple(expected_destination_identity):
            raise BackupRunOnceError(
                f"backup destination identity changed: expected={expected_destination_identity} actual={tuple(actual_identity.values())}"
            )
        _atomic_json(generation / "receipt.complete.json", complete)
        incomplete.unlink(missing_ok=True)
        return complete
    except Exception as exc:
        receipt.update(
            {
                "status": "failed",
                "failed_at": dt.datetime.now(dt.UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        try:
            _atomic_json(incomplete, receipt)
        except Exception as receipt_error:
            raise exc from receipt_error
        raise


def _managed_backup_config(config_path: Path) -> tuple[Path, tuple[int, int]]:
    from . import config as config_module

    enabled = config_module.configured_brainlayer_env_value("BRAINLAYER_BACKUP_ENABLED", config_path)
    if enabled not in {"1", "true", "yes", "on"}:
        raise BackupRunOnceError("backup is disabled in the managed BrainLayer config")
    destination_value = config_module.configured_brainlayer_literal_value(
        "BRAINLAYER_BACKUP_DESTINATION", config_path
    )
    if not destination_value or not destination_value.startswith("/"):
        raise BackupRunOnceError("managed backup destination is missing or not absolute")
    raw_dev = config_module.configured_brainlayer_env_value("BRAINLAYER_BACKUP_DESTINATION_ST_DEV", config_path)
    raw_ino = config_module.configured_brainlayer_env_value("BRAINLAYER_BACKUP_DESTINATION_ST_INO", config_path)
    try:
        identity = (int(raw_dev), int(raw_ino))
    except (TypeError, ValueError) as exc:
        raise BackupRunOnceError("managed backup destination identity is missing or invalid") from exc
    return Path(destination_value), identity


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate one local BrainLayer backup without retention or upload.")
    parser.add_argument("--config", type=Path, help="managed BrainLayer env file (default: ~/.config/brainlayer/brainlayer.env)")
    authority = parser.add_mutually_exclusive_group()
    authority.add_argument("--socket-path", type=Path)
    authority.add_argument("--offline-db-path", type=Path)
    parser.add_argument(
        "--transcript-root",
        type=Path,
        action="append",
        dest="transcript_roots",
        help="legacy JSONL-only path root; repeat for every configured root",
    )
    parser.add_argument(
        "--transcript-root-spec",
        action="append",
        dest="transcript_root_specs",
        help="JSON source-root spec with path, include_globs, exclude_globs, and source_class",
    )
    parser.add_argument("--generation-id")
    args = parser.parse_args(argv)
    if not args.transcript_roots and not args.transcript_root_specs:
        parser.error("at least one --transcript-root-spec or --transcript-root is required")
    from . import config as config_module
    config_path = (args.config or config_module.get_user_env_path()).expanduser()
    try:
        destination, identity = _managed_backup_config(config_path)
        roots = list(args.transcript_roots or [])
        for raw_spec in args.transcript_root_specs or []:
            try:
                spec = json.loads(raw_spec)
                roots.append(jsonl_backup.BackupSourceRoot(
                    Path(spec["path"]),
                    tuple(spec["include_globs"]),
                    tuple(spec.get("exclude_globs", ())),
                    str(spec.get("source_class", "transcript")),
                ))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise BackupRunOnceError("invalid --transcript-root-spec JSON") from exc
        if not roots:
            raise BackupRunOnceError("at least one transcript root or source-root spec is required")
        offline_db_path = args.offline_db_path.expanduser() if args.offline_db_path is not None else None
        if offline_db_path is not None and not offline_db_path.is_absolute():
            raise BackupRunOnceError("--offline-db-path must be an absolute explicit source path")
        result = run_backup_once(
            destination=destination,
            transcript_roots=roots,
            socket_path=None if offline_db_path is not None else backup_daily._brainbar_socket_path(args.socket_path),
            offline_db_path=offline_db_path,
            generation_id=args.generation_id,
            expected_destination_identity=identity,
        )
    except BackupRunOnceBusyError as exc:
        print(json.dumps({
            "status": "refused",
            "reason": "destination_busy",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "generation": None,
            "remote_verified": False,
            "cleanup_permitted": False,
        }, sort_keys=True))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}))
        return 1
    print(json.dumps({
        "status": result["status"],
        "receipt": str(destination / result["generation"] / "receipt.complete.json"),
        "db_source_mode": result["db_source_mode"],
        "db_source_path": result["db_source_path"],
        "db_transport_socket": result["db_transport_socket"],
        "remote_verified": False,
        "cleanup_permitted": False,
        "transcript_roots": [str(root.path.expanduser().resolve()) if isinstance(root, jsonl_backup.BackupSourceRoot) else str(Path(root).expanduser().resolve()) for root in roots],
    }, sort_keys=True))
    return 0


def run_backup_once(
    *,
    destination: Path,
    transcript_roots: list[Path | jsonl_backup.BackupSourceRoot],
    socket_path: Path | None = None,
    offline_db_path: Path | None = None,
    expected_source_path: Path | None = None,
    expected_source_open_generation: str | None = None,
    offline_source_open_generation: str | None = None,
    expected_offline_source_identity: tuple[int, int] | None = None,
    generation_id: str | None = None,
    now: dt.datetime | None = None,
    expected_destination_identity: tuple[int, int],
) -> dict[str, Any]:
    """Run against the selected directory inode, even if its pathname is replaced."""
    destination = Path(destination).expanduser()
    try:
        destination_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise BackupRunOnceError(f"backup destination must already exist as an accessible directory: {destination}") from exc
    try:
        selected = os.fstat(destination_fd)
        actual = (selected.st_dev, selected.st_ino)
        if actual != tuple(expected_destination_identity):
            raise BackupRunOnceError(
                f"backup destination identity changed: expected={expected_destination_identity} actual={actual}"
            )
        _acquire_destination_process_lock(destination_fd, actual)
        try:
            with _DESTINATION_BIND_LOCK:
                previous_cwd = os.open(".", os.O_RDONLY)
                try:
                    os.fchdir(destination_fd)
                    result = _run_backup_once_bound(
                        destination=Path("."),
                        transcript_roots=transcript_roots,
                        socket_path=socket_path,
                        offline_db_path=offline_db_path,
                        expected_source_path=expected_source_path,
                        expected_source_open_generation=expected_source_open_generation,
                        offline_source_open_generation=offline_source_open_generation,
                        expected_offline_source_identity=expected_offline_source_identity,
                        generation_id=generation_id,
                        now=now,
                        expected_destination_identity=expected_destination_identity,
                    )
                    try:
                        visible_identity = _identity(destination)
                    except OSError:
                        visible_identity = None
                    if visible_identity is None or tuple(visible_identity.values()) != tuple(expected_destination_identity):
                        generation_name = str(result.get("generation", ""))
                        actual = None if visible_identity is None else tuple(visible_identity.values())
                        replacement_error = (
                            f"backup destination identity changed after generation: "
                            f"expected={expected_destination_identity} actual={actual}"
                        )
                        if generation_name:
                            try:
                                os.unlink(f"{generation_name}/receipt.complete.json", dir_fd=destination_fd)
                            except OSError:
                                pass
                            try:
                                _atomic_json(
                                    Path(generation_name) / "receipt.incomplete.json",
                                    {
                                        "status": "failed",
                                        "generation": generation_name,
                                        "failed_at": dt.datetime.now(dt.UTC).isoformat(),
                                        "error_type": "BackupRunOnceError",
                                        "error": replacement_error,
                                        "destination_identity": expected_destination_identity,
                                        "remote_verified": False,
                                        "cleanup_permitted": False,
                                    },
                                )
                            except Exception:
                                pass
                        raise BackupRunOnceError(replacement_error)
                    return result
                finally:
                    os.fchdir(previous_cwd)
                    os.close(previous_cwd)
        finally:
            fcntl.flock(destination_fd, fcntl.LOCK_UN)
    finally:
        os.close(destination_fd)


if __name__ == "__main__":
    raise SystemExit(main())
