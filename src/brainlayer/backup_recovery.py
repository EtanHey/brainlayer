"""Explicit recovery of preserved snapshots; never creates snapshots or runs retention."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from . import backup_daily as daily


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
        if (
            item.get("name") != name
            or item.get("parents") != [folder_id]
            or item.get("trashed") is not False
            or not item.get("id")
            or item.get("appProperties", {}).get(daily.DRIVE_MACHINE_PROPERTY) != machine_id
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
