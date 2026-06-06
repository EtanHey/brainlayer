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
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
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
DEFAULT_LOG_PATH = Path.home() / ".local" / "share" / "brainlayer" / "logs" / "backup-daily.log"
DEFAULT_BRAINBAR_SOCKET_PATH = "/tmp/brainbar.sock"
BACKUP_TIMEOUT_ENV = "BRAINLAYER_BACKUP_TIMEOUT_SECONDS"
BACKUP_FULL_VERIFY_ENV = "BRAINLAYER_BACKUP_FULL_VERIFY"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
DEFAULT_DAILY_KEEP = 7
DEFAULT_WEEKLY_KEEP = 4
DEFAULT_LOCAL_UNCOMPRESSED_KEEP = 2


@dataclass(frozen=True)
class DriveRetentionPolicy:
    keep_latest: int
    filename_prefix: str = ""
    filename_suffix: str = ".db.gz"

    def __post_init__(self) -> None:
        if self.keep_latest < 1:
            raise ValueError("keep_latest must be at least 1")


DAILY_RETENTION = DriveRetentionPolicy(keep_latest=DEFAULT_DAILY_KEEP)
WEEKLY_RETENTION = DriveRetentionPolicy(keep_latest=DEFAULT_WEEKLY_KEEP)


@dataclass(frozen=True)
class SQLiteBackupArtifact:
    gzip_path: Path
    uncompressed_path: Path | None
    sentinel_chunks: int
    local_retention_deleted: list[str]


def _today() -> str:
    return dt.datetime.now(dt.UTC).date().isoformat()


def _append_json_log(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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


def _sqlite_pragma_check(db_path: Path, pragma_name: str) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(f"PRAGMA {pragma_name}").fetchone()
    finally:
        conn.close()
    return str(row[0]) if row else ""


def _count_chunks(db_path: Path) -> int:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else 0


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


def create_sqlite_backup_artifact(
    db_path: Path,
    output_dir: Path,
    date_stamp: str | None = None,
    socket_path: Path | str | None = None,
    keep_uncompressed: bool = True,
    local_uncompressed_keep: int = DEFAULT_LOCAL_UNCOMPRESSED_KEEP,
) -> SQLiteBackupArtifact:
    """Create a restorable `.db.gz` snapshot through BrainBar's single-writer socket."""
    db_path = Path(db_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    date_stamp = date_stamp or _today()

    if not db_path.exists():
        raise FileNotFoundError(f"BrainLayer database not found: {db_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    required_bytes = (db_path.stat().st_size * 3) + (512 * 1024 * 1024)
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient free space for backup in {output_dir}: "
            f"{free_bytes} bytes free, {required_bytes} bytes required"
        )
    final_gz = output_dir / f"{date_stamp}.db.gz"
    final_raw = output_dir / f"{date_stamp}.db"
    sentinel_chunks = 0
    uncompressed_path: Path | None = None

    with tempfile.TemporaryDirectory(prefix="brainlayer-backup-", dir=output_dir) as tmp:
        raw_snapshot = Path(tmp) / f"{date_stamp}.db"
        request_brainbar_vacuum_into(raw_snapshot, socket_path=socket_path)
        target = sqlite3.connect(f"file:{raw_snapshot}?mode=ro", uri=True)
        try:
            integrity = target.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise RuntimeError(f"Backup integrity check failed: {integrity!r}")
            row = target.execute("SELECT COUNT(*) FROM chunks").fetchone()
            sentinel_chunks = int(row[0]) if row else 0
        finally:
            target.close()

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
    timeout_seconds: int = 300,
    max_attempts: int = 3,
    retry_backoff_seconds: int = 60,
) -> None:
    target_path = Path(target_path).expanduser()
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "brain_backup_vacuum_into",
            "arguments": {"target_path": str(target_path)},
        },
    }
    resolved_socket_path = _brainbar_socket_path(socket_path)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = _send_brainbar_json_request(resolved_socket_path, request, timeout_seconds=timeout_seconds)
            if response.get("error"):
                raise RuntimeError(f"BrainBar backup request failed: {response['error']}")
            result = response.get("result") or {}
            if result.get("isError"):
                content = result.get("content") or []
                text = content[0].get("text") if content and isinstance(content[0], dict) else result
                raise RuntimeError(f"BrainBar backup request failed: {text}")
            if not target_path.exists():
                raise RuntimeError(f"BrainBar backup did not create snapshot: {target_path}")
            return
        except BackupTimeoutError:
            raise
        except Exception as exc:
            last_error = exc
            existing_target_note = ""
            if target_path.exists():
                try:
                    pragma = _sqlite_pragma_check(target_path, "quick_check")
                except Exception as check_exc:
                    target_path.unlink(missing_ok=True)
                    existing_target_note = f"; removing invalid existing target after quick_check error: {check_exc}"
                else:
                    if pragma == "ok":
                        print(
                            f"BrainBar vacuum snapshot attempt {attempt}/{max_attempts} failed: {exc}; "
                            "target exists and passed quick_check",
                            flush=True,
                        )
                        return
                    target_path.unlink(missing_ok=True)
                    existing_target_note = f"; removing invalid existing target after quick_check={pragma!r}"
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
            time.sleep(retry_backoff_seconds)
    if last_error is not None:
        raise last_error


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
            pragma_name = "integrity_check" if full else "quick_check"
            pragma = _sqlite_pragma_check(restored, pragma_name)
            result["pragma"] = pragma
            if pragma != "ok":
                result["verification_error"] = f"PRAGMA {pragma_name} failed: {pragma!r}"
                return result
            verified_chunks = _count_chunks(restored)
            result["sentinel_verified_chunks"] = verified_chunks
            if verified_chunks != artifact.sentinel_chunks:
                result["verification_error"] = (
                    f"sentinel mismatch: snapshot={artifact.sentinel_chunks} verified={verified_chunks}"
                )
                return result
            result["verified"] = True
    except Exception as exc:
        result.setdefault("gzip_test", False)
        result["verification_error"] = str(exc)
    return result


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
        parsed = _parse_snapshot_date(
            item.get("name", ""),
            prefix=retention_policy.filename_prefix,
            suffix=retention_policy.filename_suffix,
        )
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
    log_path: Path = DEFAULT_LOG_PATH,
    date_stamp: str | None = None,
    upload: bool = True,
    retention_policy: DriveRetentionPolicy = DAILY_RETENTION,
    remove_local_after_upload: bool = True,
) -> dict[str, Any]:
    resolved_date_stamp = date_stamp or _today()
    artifact = create_sqlite_backup_artifact(db_path or get_db_path(), staging_dir, date_stamp=resolved_date_stamp)
    snapshot = artifact.gzip_path
    snapshot_size = snapshot.stat().st_size
    result: dict[str, Any] = {
        "db": str(db_path or get_db_path()),
        "snapshot": str(snapshot),
        "local_uncompressed_snapshot": str(artifact.uncompressed_path) if artifact.uncompressed_path else None,
        "local_retention_deleted": artifact.local_retention_deleted,
        "sentinel_snapshot_chunks": artifact.sentinel_chunks,
        "bytes": snapshot_size,
        "uploaded": False,
        "local_removed": False,
        "verified": False,
    }
    try:
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
                deleted = prune_drive_backups(
                    service,
                    folder_parts=folder_parts,
                    retention_policy=retention_policy,
                )
            else:
                deleted = []
            result.update({"uploaded": True, "drive_file": uploaded, "retention_deleted": deleted})
    except Exception as exc:
        result.update({"error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        _append_json_log(log_path, result)
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
            log_path=Path(os.environ.get("BRAINLAYER_BACKUP_LOG_PATH", str(DEFAULT_LOG_PATH))),
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
    return 0 if result.get("verified", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
