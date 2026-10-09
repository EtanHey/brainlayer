"""Explicit recovery of preserved snapshots; never creates snapshots or runs retention."""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from . import backup_daily as daily
from .job_alerts import report


def read_source_receipt(
    log_path: Path,
    *,
    line_number: int,
    snapshot_date: str,
    staging_dir: Path,
    machine_id: str,
    folder_parts: list[str],
    expected_raw_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind operator provenance to one unchanged failed receipt and dated pair."""
    if type(line_number) is not int or line_number < 1:
        raise ValueError("source receipt line must be positive")
    if dt.date.fromisoformat(snapshot_date).isoformat() != snapshot_date:
        raise ValueError("snapshot date must be YYYY-MM-DD")
    if type(expected_raw_bytes) is not int or expected_raw_bytes < 1:
        raise ValueError("expected raw bytes must be positive")
    line = None
    with log_path.open("rb") as handle:
        for number, candidate in enumerate(handle, 1):
            if number == line_number:
                line = candidate
                break
    if line is None or not line.endswith(b"\n"):
        raise ValueError("source receipt line missing or incomplete")
    receipt = json.loads(line)
    if not isinstance(receipt, dict):
        raise ValueError("source receipt must be an object")
    attempted = dt.datetime.fromisoformat(receipt["attempted_at"])
    if attempted.tzinfo is None or attempted > dt.datetime.now(dt.UTC):
        raise ValueError("source attempt time must be timezone-aware and not future")
    raw = staging_dir / f"{snapshot_date}.db"
    compressed = staging_dir / f"{snapshot_date}.db.gz"
    if receipt.get("snapshot") != str(compressed):
        raise ValueError("source receipt snapshot does not match dated staging path")
    if receipt.get("local_uncompressed_snapshot") not in (None, str(raw)):
        raise ValueError("source receipt raw path mismatch")
    if receipt.get("machine_id") != machine_id or receipt.get("drive_folder") != "/".join(folder_parts):
        raise ValueError("source receipt machine or Drive folder mismatch")
    if receipt.get("uploaded") is not False or receipt.get("verified") is not False:
        raise ValueError("source receipt must identify an unverified failed upload")
    if receipt.get("backup_log_provenance") != daily._backup_log_provenance():
        raise ValueError("source receipt provenance mismatch")
    for key in ("bytes", "sentinel_snapshot_chunks"):
        if type(receipt.get(key)) is not int or receipt[key] < 1:
            raise ValueError(f"source receipt {key} must be positive")
    facts = {
        "source_receipt_line": line_number,
        "source_receipt_sha256": hashlib.sha256(line).hexdigest(),
        "source_attempted_at": receipt["attempted_at"],
        "snapshot_date": snapshot_date,
        "snapshot": str(compressed),
        "local_uncompressed_snapshot": str(raw),
        "bytes": receipt["bytes"],
        "raw_bytes": expected_raw_bytes,
        "sentinel_snapshot_chunks": receipt["sentinel_snapshot_chunks"],
    }
    return receipt, facts


def _list_all(service: Any, query: str) -> list[dict[str, Any]]:
    items = []
    token = None
    seen = set()
    while True:
        response = daily._execute_drive_read(
            service.files().list(
                q=query,
                spaces="drive",
                pageSize=1000,
                pageToken=token,
                supportsAllDrives=True,
                fields="nextPageToken,files(id,name,size,md5Checksum,parents,trashed,mimeType,appProperties)",
            ),
            what="Drive recovery candidate listing",
        )
        if not isinstance(response.get("files"), list):
            raise RuntimeError("Drive recovery listing is unknown")
        items.extend(response["files"])
        token = response.get("nextPageToken")
        if not token:
            return items
        if token in seen:
            raise RuntimeError("Drive recovery listing repeated page token")
        seen.add(token)


def resolve_existing_folder(service: Any, folder_parts: list[str]) -> str:
    """Read the original folder chain; missing or ambiguous folders never create one."""
    parent = "root"
    if not folder_parts or any(not part for part in folder_parts):
        raise ValueError("Drive folder chain must be explicit")
    for name in folder_parts:
        items = _list_all(
            service,
            f"trashed = false and mimeType = '{daily.DRIVE_FOLDER_MIME}' "
            f"and name = '{daily._escape_drive_query_value(name)}' "
            f"and '{daily._escape_drive_query_value(parent)}' in parents",
        )
        if len(items) != 1 or not items[0].get("id") or items[0].get("name") != name:
            raise RuntimeError("Original Drive folder is missing, unknown or ambiguous")
        parent = items[0]["id"]
    return parent


def find_verified_drive_match(
    service: Any, *, folder_id: str, name: str, size: int, md5: str, machine_id: str
) -> dict[str, Any] | None:
    candidates = _list_all(
        service,
        f"trashed = false and '{daily._escape_drive_query_value(folder_id)}' in parents "
        f"and name = '{daily._escape_drive_query_value(name)}'",
    )
    matches = []
    for item in candidates:
        properties = item.get("appProperties")
        if (
            item.get("name") != name
            or item.get("parents") != [folder_id]
            or item.get("trashed") is not False
            or not item.get("id")
            or not isinstance(properties, dict)
            or properties.get(daily.DRIVE_MACHINE_PROPERTY) != machine_id
        ):
            raise RuntimeError("Drive recovery candidate identity or ownership is unknown")
        try:
            actual_size = int(item["size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Drive recovery candidate size is unknown") from exc
        if actual_size == size and item.get("md5Checksum") == md5:
            matches.append(item)
    if len(matches) > 1:
        raise RuntimeError("Multiple complete Drive matches; recovery is ambiguous")
    return matches[0] if matches else None


def _append_durable_receipt(log_path: Path, result: dict[str, Any]) -> None:
    # The source log already exists. Fsync before reporting healthy; append leaves
    # the original failed record byte-for-byte intact.
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def recover_existing_artifact(
    *,
    snapshot_date: str,
    source_receipt_line: int,
    expected_raw_bytes: int,
    staging_dir: Path = daily.DEFAULT_STAGING_DIR,
    log_path: Path | None = None,
    machine_id: str | None = None,
    folder_parts: list[str] | None = None,
    sqlite_timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Recover one preserved pair under the daily job's exclusive nonblocking lock."""
    staging_dir = Path(staging_dir).expanduser().absolute()
    log_path = daily._backup_log_path(log_path, db_path=daily.get_db_path()).expanduser()
    result: dict[str, Any] = {
        "operation": "recover_existing_artifact",
        "attempt_id": uuid.uuid4().hex,
        "attempted_at": dt.datetime.now(dt.UTC).isoformat(),
        "snapshot_date": snapshot_date,
        "source_receipt_line": source_receipt_line,
        "db": str(daily.get_db_path()),
        "uploaded": False,
        "verified": False,
        "local_removed": False,
        "retention_mode": "none",
        "retention_deleted": [],
        "backup_log_provenance": daily._backup_log_provenance(),
    }
    descriptors = []
    lock = None
    try:
        if staging_dir.is_symlink() or not staging_dir.is_dir():
            raise ValueError("recovery staging directory must already exist and not be a symlink")
        lock = (staging_dir / ".backup.lock").open("a+b")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise daily.BackupAlreadyRunningError("backup already running; recovery refused") from exc
        machine = daily._validate_machine_id(machine_id) if machine_id is not None else daily.resolve_machine_id()
        folders = list(folder_parts) if folder_parts is not None else daily.default_drive_folder_parts(machine)
        _, facts = read_source_receipt(
            log_path,
            line_number=source_receipt_line,
            snapshot_date=snapshot_date,
            staging_dir=staging_dir,
            machine_id=machine,
            folder_parts=folders,
            expected_raw_bytes=expected_raw_bytes,
        )
        result.update(facts, machine_id=machine, drive_folder="/".join(folders))
        raw_path, gzip_path = Path(facts["local_uncompressed_snapshot"]), Path(facts["snapshot"])
        identities = []
        for path in (raw_path, gzip_path):
            descriptor, identity = daily._open_stable_regular_file(path)
            descriptors.append(descriptor)
            identities.append(identity)
        if identities[0][2] != expected_raw_bytes or identities[1][2] != facts["bytes"]:
            raise ValueError("preserved raw/compressed sizes differ from operator/source receipt")
        result["artifact_mtime_at"] = dt.datetime.fromtimestamp(os.fstat(descriptors[1]).st_mtime, dt.UTC).isoformat()

        def assert_unchanged():
            for path, descriptor, identity in zip((raw_path, gzip_path), descriptors, identities, strict=True):
                daily._assert_open_file_unchanged(descriptor, identity, path.name)
                if daily._file_identity(path.lstat()) != identity:
                    raise RuntimeError("preserved artifact pathname changed during recovery")

        result.update(
            daily.verify_preserved_artifact_pair(
                raw_path,
                gzip_path,
                expected_sentinel_chunks=facts["sentinel_snapshot_chunks"],
                sqlite_timeout_seconds=sqlite_timeout_seconds,
            )
        )
        assert_unchanged()
        result["local_md5"] = result["gzip_md5"]
        service = daily.build_drive_service()
        folder_id = resolve_existing_folder(service, folders)
        daily.assert_drive_folder_owned_by_machine(service, folder_id=folder_id, machine_id=machine)
        result["drive_folder_id"] = folder_id
        match = find_verified_drive_match(
            service,
            folder_id=folder_id,
            name=gzip_path.name,
            size=facts["bytes"],
            md5=result["local_md5"],
            machine_id=machine,
        )
        assert_unchanged()
        if match is None:
            name = f"{snapshot_date}-recovery-{result['attempt_id']}.db.gz"
            result.update(drive_action="new_unique_upload", drive_name=name)
            match = daily.upload_file_to_drive_raw(
                gzip_path,
                folder_id,
                daily.get_drive_credentials(),
                machine_id=machine,
                drive_name=name,
                attempt_id=result["attempt_id"],
                source_descriptor=descriptors[1],
            )
            service = daily._retry_transient_drive_call(daily.build_drive_service, what="Drive recovery rebuild")
        else:
            name = gzip_path.name
            result.update(drive_action="reuse_existing", drive_name=name)
        if not isinstance(match.get("id"), str) or not match["id"]:
            raise RuntimeError("Drive recovery response missing file identity")
        # Preserve the candidate identity even when the subsequent read fails.
        result["drive_file"] = match
        metadata = daily.verify_drive_upload(
            service,
            file_id=match["id"],
            expected_name=name,
            expected_size=facts["bytes"],
            expected_machine_id=machine,
            expected_md5=result["local_md5"],
            expected_parent=folder_id,
        )
        result.update(drive_file=metadata, uploaded=True, drive_md5_match=True)
        if (
            result["drive_action"] == "new_unique_upload"
            and metadata.get("appProperties", {}).get("brainlayer_recovery_attempt") != result["attempt_id"]
        ):
            raise RuntimeError("Drive recovery attempt ownership mismatch")
        assert_unchanged()
        result.update(verified=True, completed_at=dt.datetime.now(dt.UTC).isoformat())
        _append_durable_receipt(log_path, result)
        report("backup-daily", None)
        return result
    except Exception as exc:
        result.update(verified=False, error_type=type(exc).__name__, error=str(exc))
        _append_durable_receipt(log_path, result)
        report("backup-daily", "Preserved backup recovery failed; check the backup log")
        raise
    finally:
        if lock is not None:
            lock.close()
        for descriptor in descriptors:
            os.close(descriptor)
