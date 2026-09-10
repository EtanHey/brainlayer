"""One local backup generation with no retention, upload, or cleanup side effects."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
import stat
import threading
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from . import backup_daily, jsonl_backup


class BackupRunOnceError(RuntimeError):
    """A backup generation could not be completed."""


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


def _offline_snapshot(source: Path, target: Path) -> int:
    source_uri = f"file:{source.resolve()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source_db, sqlite3.connect(str(target)) as target_db:
        source_db.backup(target_db, pages=1000, sleep=0.01)
    return backup_daily._validate_backup_target(target)


def _socket_snapshot(socket_path: Path, target: Path, generation: Path) -> int:
    backup_daily.request_brainbar_vacuum_into(target, socket_path=socket_path, max_attempts=1, attempt_dir=generation)
    return backup_daily._validate_backup_target(target)


def _run_backup_once_bound(
    *,
    destination: Path,
    transcript_roots: list[Path | jsonl_backup.BackupSourceRoot],
    socket_path: Path | None = None,
    offline_db_path: Path | None = None,
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
        chunks = (
            _socket_snapshot(Path(socket_path), raw_db, generation)
            if socket_path is not None
            else _offline_snapshot(Path(offline_db_path), raw_db)
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
            else str(Path(socket_path).expanduser().resolve()),
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
    parser.add_argument("--socket-path", type=Path)
    parser.add_argument(
        "--transcript-root",
        type=Path,
        action="append",
        dest="transcript_roots",
        help="legacy path-only transcript root; repeat for every configured root",
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
        result = run_backup_once(
            destination=destination,
            transcript_roots=roots,
            socket_path=backup_daily._brainbar_socket_path(args.socket_path),
            generation_id=args.generation_id,
            expected_destination_identity=identity,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}))
        return 1
    print(json.dumps({
        "status": result["status"],
        "receipt": str(destination / result["generation"] / "receipt.complete.json"),
        "db_source_mode": result["db_source_mode"],
        "db_source_path": result["db_source_path"],
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
        with _DESTINATION_BIND_LOCK:
            previous_cwd = os.open(".", os.O_RDONLY)
            try:
                os.fchdir(destination_fd)
                result = _run_backup_once_bound(
                    destination=Path("."),
                    transcript_roots=transcript_roots,
                    socket_path=socket_path,
                    offline_db_path=offline_db_path,
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
                    if generation_name:
                        try:
                            os.unlink(f"{generation_name}/receipt.complete.json", dir_fd=destination_fd)
                        except OSError:
                            pass
                    actual = None if visible_identity is None else tuple(visible_identity.values())
                    raise BackupRunOnceError(
                        f"backup destination identity changed after generation: expected={expected_destination_identity} actual={actual}"
                    )
                return result
            finally:
                os.fchdir(previous_cwd)
                os.close(previous_cwd)
    finally:
        os.close(destination_fd)


if __name__ == "__main__":
    raise SystemExit(main())
