"""One local backup generation with no retention, upload, or cleanup side effects."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from . import backup_daily, jsonl_backup


class BackupRunOnceError(RuntimeError):
    """A backup generation could not be completed."""


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
    stat = path.stat()
    return {"st_dev": stat.st_dev, "st_ino": stat.st_ino}


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


def run_backup_once(
    *,
    destination: Path,
    transcript_roots: list[Path | jsonl_backup.BackupSourceRoot],
    socket_path: Path | None = None,
    offline_db_path: Path | None = None,
    generation_id: str | None = None,
    now: dt.datetime | None = None,
    expected_destination_identity: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Create one locally verified DB+JSONL generation."""
    destination = Path(destination).expanduser()
    if not destination.exists() or not destination.is_dir():
        raise BackupRunOnceError(f"backup destination must already exist as a directory: {destination}")
    resolved_destination = destination.resolve()
    if expected_destination_identity is not None:
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
            candidates, generation, date_stamp=stamp
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
        if expected_destination_identity is not None:
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
