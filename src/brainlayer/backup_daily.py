"""Daily BrainLayer database backups.

The backup path intentionally uses SQLite's online backup API instead of copying
the database file directly, so live WAL writes are folded into a consistent
snapshot without stopping BrainBar or the enrichment jobs.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import gzip
import json
import os
import shutil
import signal
import socket
import sqlite3
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .paths import get_db_path

DEFAULT_TOKEN_PATH = Path.home() / ".config" / "google-drive-mcp" / "tokens.json"
DEFAULT_CLIENT_PATH = Path.home() / ".config" / "google-drive-mcp" / "gcp-oauth.keys.json"
DEFAULT_FOLDER_PARTS = ["Brain Drive", "06_ARCHIVE", "backups", "brainlayer-db"]
DEFAULT_STAGING_DIR = Path.home() / ".local" / "share" / "brainlayer" / "backups"
DEFAULT_BRAINBAR_SOCKET_PATH = "/tmp/brainbar.sock"
BACKUP_TIMEOUT_ENV = "BRAINLAYER_BACKUP_TIMEOUT_SECONDS"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
DEFAULT_DAILY_KEEP = 7
DEFAULT_WEEKLY_KEEP = 4


@dataclass(frozen=True)
class DriveRetentionPolicy:
    keep_latest: int

    def __post_init__(self) -> None:
        if self.keep_latest < 1:
            raise ValueError("keep_latest must be at least 1")


DAILY_RETENTION = DriveRetentionPolicy(keep_latest=DEFAULT_DAILY_KEEP)
WEEKLY_RETENTION = DriveRetentionPolicy(keep_latest=DEFAULT_WEEKLY_KEEP)


def _today() -> str:
    return dt.datetime.now(dt.UTC).date().isoformat()


def _escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


class BackupTimeoutError(TimeoutError):
    pass


def _configured_backup_timeout_seconds() -> int | None:
    raw = os.environ.get(BACKUP_TIMEOUT_ENV)
    if raw is None or raw.strip() == "":
        return None
    try:
        seconds = int(raw)
    except ValueError as exc:
        raise ValueError(f"{BACKUP_TIMEOUT_ENV} must be an integer number of seconds") from exc
    return seconds if seconds > 0 else None


def _raise_backup_timeout(signum, frame) -> None:  # noqa: ARG001
    raise BackupTimeoutError("backup exceeded configured wall-clock timeout")


def create_sqlite_backup_gzip(
    db_path: Path,
    output_dir: Path,
    date_stamp: str | None = None,
    socket_path: Path | str | None = None,
) -> Path:
    """Create a restorable `.db.gz` snapshot through BrainBar's single-writer socket."""
    db_path = Path(db_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    date_stamp = date_stamp or _today()

    if not db_path.exists():
        raise FileNotFoundError(f"BrainLayer database not found: {db_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    required_bytes = (db_path.stat().st_size * 2) + (512 * 1024 * 1024)
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient free space for backup in {output_dir}: "
            f"{free_bytes} bytes free, {required_bytes} bytes required"
        )
    final_gz = output_dir / f"{date_stamp}.db.gz"

    with tempfile.TemporaryDirectory(prefix="brainlayer-backup-", dir=output_dir) as tmp:
        raw_snapshot = Path(tmp) / f"{date_stamp}.db"
        request_brainbar_vacuum_into(raw_snapshot, socket_path=socket_path)
        target = sqlite3.connect(f"file:{raw_snapshot}?mode=ro", uri=True)
        try:
            integrity = target.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise RuntimeError(f"Backup integrity check failed: {integrity!r}")
        finally:
            target.close()

        temp_gz = Path(tmp) / final_gz.name
        with raw_snapshot.open("rb") as src, gzip.open(temp_gz, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        shutil.move(str(temp_gz), final_gz)

    return final_gz


def _brainbar_socket_path(socket_path: Path | str | None = None) -> Path:
    if socket_path is not None:
        return Path(socket_path).expanduser()
    return Path(os.environ.get("BRAINBAR_SOCKET_PATH", DEFAULT_BRAINBAR_SOCKET_PATH)).expanduser()


def request_brainbar_vacuum_into(
    target_path: Path, socket_path: Path | str | None = None, timeout_seconds: int = 300
) -> None:
    target_path = Path(target_path).expanduser()
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "brain_backup_vacuum_into",
            "arguments": {"target_path": str(target_path)},
        },
    }
    response = _send_brainbar_json_request(_brainbar_socket_path(socket_path), request, timeout_seconds=timeout_seconds)
    if response.get("error"):
        raise RuntimeError(f"BrainBar backup request failed: {response['error']}")
    result = response.get("result") or {}
    if result.get("isError"):
        content = result.get("content") or []
        text = content[0].get("text") if content and isinstance(content[0], dict) else result
        raise RuntimeError(f"BrainBar backup request failed: {text}")
    if not target_path.exists():
        raise RuntimeError(f"BrainBar backup did not create snapshot: {target_path}")


def _send_brainbar_json_request(socket_path: Path, request: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
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


def upload_file_to_drive_raw(
    file_path: Path,
    folder_id: str,
    credentials: Any,
    chunk_size: int = 8 * 1024 * 1024,
    max_attempts: int = 30,
) -> dict[str, Any]:
    """Upload large backups with Drive's raw resumable protocol."""
    file_path = Path(file_path)
    total = file_path.stat().st_size
    metadata = {"name": file_path.name, "parents": [folder_id]}
    init = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&supportsAllDrives=true&fields=id,name,size",
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
    with file_path.open("rb") as handle:
        while sent < total:
            handle.seek(sent)
            expected = min(chunk_size, total - sent)
            chunk = handle.read(expected)
            if len(chunk) != expected:
                raise RuntimeError(f"Backup file changed during upload: expected {expected} bytes, got {len(chunk)}")
            start = sent
            end = sent + len(chunk) - 1
            headers = {
                "Authorization": f"Bearer {credentials.token}",
                "Content-Length": str(len(chunk)),
                "Content-Range": f"bytes {start}-{end}/{total}",
            }
            for attempt in range(1, max_attempts + 1):
                try:
                    response = requests.put(upload_url, headers=headers, data=chunk, timeout=120)
                    if response.status_code in {200, 201}:
                        return response.json()
                    if response.status_code == 308:
                        uploaded_range = response.headers.get("Range")
                        if uploaded_range and "-" in uploaded_range:
                            sent = int(uploaded_range.rsplit("-", 1)[1]) + 1
                        else:
                            sent = end + 1
                        print(f"drive upload progress: {sent}/{total} bytes", flush=True)
                        break
                    if response.status_code in {429, 500, 502, 503, 504}:
                        raise RuntimeError(f"retryable HTTP {response.status_code}: {response.text[:200]}")
                    response.raise_for_status()
                except Exception as exc:
                    if attempt >= max_attempts:
                        raise
                    sleep_seconds = min(60, 2 ** min(attempt, 6))
                    print(
                        f"drive upload retry chunk={start}-{end} attempt={attempt}/{max_attempts}: {exc}; "
                        f"sleeping {sleep_seconds}s",
                        flush=True,
                    )
                    time.sleep(sleep_seconds)

    raise RuntimeError("Drive upload ended without final response")


def _parse_snapshot_date(name: str) -> dt.date | None:
    if not name.endswith(".db.gz"):
        return None
    try:
        return dt.date.fromisoformat(name[:10])
    except ValueError:
        return None


def verify_drive_upload(service: Any, *, file_id: str, expected_name: str, expected_size: int) -> None:
    """Verify that Drive can see the uploaded file with the expected name and byte size."""
    metadata = service.files().get(fileId=file_id, fields="id,name,size,trashed", supportsAllDrives=True).execute()
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


def prune_drive_backups(
    service: Any,
    *,
    folder_parts: list[str] = DEFAULT_FOLDER_PARTS,
    retention_policy: DriveRetentionPolicy = DAILY_RETENTION,
) -> list[str]:
    """Keep only the latest N verified snapshots in the Drive backup folder."""
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
        parsed = _parse_snapshot_date(item.get("name", ""))
        if parsed:
            dated.append((parsed, item))

    dated.sort(key=lambda pair: pair[0], reverse=True)
    keep_ids = {item["id"] for _, item in dated[: retention_policy.keep_latest]}

    deleted: list[str] = []
    for _, item in dated:
        if item["id"] in keep_ids:
            continue
        service.files().delete(fileId=item["id"], supportsAllDrives=True).execute()
        deleted.append(item["name"])
    return deleted


def run_backup(
    db_path: Path | None = None,
    staging_dir: Path = DEFAULT_STAGING_DIR,
    folder_parts: list[str] = DEFAULT_FOLDER_PARTS,
    date_stamp: str | None = None,
    upload: bool = True,
    retention_policy: DriveRetentionPolicy = DAILY_RETENTION,
    remove_local_after_upload: bool = True,
) -> dict[str, Any]:
    snapshot = create_sqlite_backup_gzip(db_path or get_db_path(), staging_dir, date_stamp=date_stamp)
    snapshot_size = snapshot.stat().st_size
    result: dict[str, Any] = {
        "db": str(db_path or get_db_path()),
        "snapshot": str(snapshot),
        "bytes": snapshot_size,
        "uploaded": False,
        "local_removed": False,
    }
    if upload:
        credentials = get_drive_credentials()
        service = build_drive_service()
        folder_id = ensure_drive_folder_chain(service, folder_parts)
        uploaded = upload_file_to_drive_raw(snapshot, folder_id, credentials)
        file_id = uploaded.get("id")
        if not file_id:
            raise RuntimeError(f"Drive upload response missing file id: {uploaded!r}")
        verify_drive_upload(
            service,
            file_id=file_id,
            expected_name=snapshot.name,
            expected_size=snapshot_size,
        )
        if remove_local_after_upload:
            snapshot.unlink()
            result["local_removed"] = True
        deleted = prune_drive_backups(
            service,
            folder_parts=folder_parts,
            retention_policy=retention_policy,
        )
        result.update({"uploaded": True, "drive_file": uploaded, "retention_deleted": deleted})
    return result


def main() -> int:
    timeout_seconds = _configured_backup_timeout_seconds()
    previous_alarm_handler = None
    if timeout_seconds is not None:
        previous_alarm_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_backup_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        result = run_backup(
            staging_dir=Path(os.environ.get("BRAINLAYER_BACKUP_STAGING_DIR", str(DEFAULT_STAGING_DIR))),
            # Prefer BRAINLAYER_BACKUP_DRIVE_FOLDER; BRAINLAYER_BACKUP_DRIVE_PATH is a legacy alias before DEFAULT_FOLDER_PARTS.
            folder_parts=os.environ.get(
                "BRAINLAYER_BACKUP_DRIVE_FOLDER",
                os.environ.get("BRAINLAYER_BACKUP_DRIVE_PATH", "/".join(DEFAULT_FOLDER_PARTS)),
            ).split("/"),
        )
    except BackupTimeoutError:
        print(f"brainlayer backup timed out after {timeout_seconds}s", flush=True)
        return 124
    except Exception as exc:
        print(f"brainlayer backup failed: {exc}\n{traceback.format_exc()}", flush=True)
        return 1
    finally:
        if timeout_seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm_handler)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
