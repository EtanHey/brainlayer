"""Nightly JSONL transcript backups to Google Drive and iCloud Drive.

Install note: commit `launchd/com.brainlayer.jsonl-backup.plist`, then install it
after merge with the repo's launchd flow or a manual `launchctl bootstrap`; this
module intentionally does not install the agent itself.

Source format policy: Claude/Codex/Cursor/Gemini JSONL files are backed up as
plain transcript files. Antigravity has no stable text export contract, so this
job backs up its raw restore units: conversation SQLite DB/WAL/SHM files,
opaque `.pb` implicit records, and per-session `.system_generated` /
`.tempmediaStorage` artifacts.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import backup_daily
from .queue_io import enqueue_store

DEFAULT_FOLDER_PARTS = ["Brain Drive", "06_ARCHIVE", "backups", "claude-jsonl"]
DEFAULT_FOREVER_FOLDER_PARTS = ["Brain Drive", "06_ARCHIVE", "backups", "claude-jsonl-forever"]
DEFAULT_STATE_PATH = Path.home() / ".local" / "share" / "brainlayer" / "jsonl-backup-state.json"
DEFAULT_STAGING_DIR = Path.home() / ".local" / "share" / "brainlayer" / "jsonl-backups"
DEFAULT_LOG_PATH = Path.home() / ".local" / "share" / "brainlayer" / "logs" / "jsonl-backup.log"
DEFAULT_ICLOUD_DIR = (
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Archives" / "brainlayer-jsonl-backups"
)
# A distinct sibling of golems' reserved path avoids shared naming/pruning ownership.
DEFAULT_ACTIVE_SKIP_SECONDS = 10 * 60
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_ICLOUD_TIMEOUT_SECONDS = 300
ICLOUD_DIR_ENV = "BRAINLAYER_JSONL_BACKUP_ICLOUD_DIR"
JSONL_RETENTION = backup_daily.DriveRetentionPolicy(
    keep_latest=30,
    filename_prefix="claude-jsonl-",
    filename_suffix=".tar.gz",
)


@dataclass(frozen=True)
class BackupSourceRoot:
    path: Path
    include_globs: tuple[str, ...] = ("**/*.jsonl",)


def default_source_roots() -> list[BackupSourceRoot]:
    home = Path.home()
    return [
        BackupSourceRoot(home / ".claude" / "projects"),
        BackupSourceRoot(home / ".claude-archive"),
        BackupSourceRoot(home / ".codex" / "sessions"),
        BackupSourceRoot(home / ".cursor" / "sessions", ("**/*.jsonl", "**/*.json")),
        BackupSourceRoot(
            home / ".cursor" / "projects",
            ("**/agent-transcripts/**/*.jsonl", "**/agent-transcripts/**/*.json"),
        ),
        BackupSourceRoot(home / ".cursor" / "acp-sessions", ("**/*.json",)),
        BackupSourceRoot(home / ".cursor" / "plans", ("**/*.md",)),
        BackupSourceRoot(home / ".gemini" / "sessions"),
        # Antigravity has no stable text export contract; back up its raw restore units.
        BackupSourceRoot(
            home / ".gemini" / "antigravity-cli" / "conversations",
            ("**/*.db", "**/*.db-wal", "**/*.db-shm"),
        ),
        BackupSourceRoot(home / ".gemini" / "antigravity-cli" / "implicit", ("**/*.pb",)),
        BackupSourceRoot(
            home / ".gemini" / "antigravity-cli" / "brain",
            (
                "**/.system_generated/**/*.jsonl",
                "**/.system_generated/**/*.json",
                "**/.system_generated/**/*.log",
                "**/.system_generated/**/*.md",
                "**/.system_generated/**/*.png",
                "**/.system_generated/**/*.jpg",
                "**/.system_generated/**/*.mp4",
                "**/.tempmediaStorage/**/*.png",
                "**/.tempmediaStorage/**/*.jpg",
                "**/.tempmediaStorage/**/*.mp4",
            ),
        ),
        BackupSourceRoot(home / ".gemini" / "antigravity-cli" / "cache", ("last_conversations.json", "projects.json")),
    ]


DEFAULT_SOURCE_ROOTS = default_source_roots()


@dataclass(frozen=True)
class JsonlCandidate:
    path: Path
    root: Path
    root_index: int
    mtime: float
    size: int


def _source_root_config(root: Path | BackupSourceRoot) -> BackupSourceRoot:
    if isinstance(root, BackupSourceRoot):
        return root
    return BackupSourceRoot(Path(root))


def _today() -> str:
    return dt.datetime.now(dt.UTC).date().isoformat()


def _configured_backup_timeout_seconds() -> int | None:
    raw = os.environ.get(backup_daily.BACKUP_TIMEOUT_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_TIMEOUT_SECONDS
    try:
        seconds = int(raw)
    except ValueError as exc:
        raise ValueError(f"{backup_daily.BACKUP_TIMEOUT_ENV} must be an integer number of seconds") from exc
    return seconds if seconds > 0 else None


def _configured_icloud_dir() -> Path | None:
    """Return the opt-in iCloud destination; Drive-only is the default."""
    raw = os.environ.get(ICLOUD_DIR_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


def _load_state(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        return {"files": {}}
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(data, dict):
        return {"files": {}}
    files = data.get("files")
    if not isinstance(files, dict):
        data["files"] = {}
    return data


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _append_json_log(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _discover_jsonl_candidates(source_roots: list[Path | BackupSourceRoot]) -> list[JsonlCandidate]:
    candidates: list[JsonlCandidate] = []
    seen: set[Path] = set()
    for index, root in enumerate(source_roots):
        config = _source_root_config(root)
        expanded_root = config.path.expanduser()
        if not expanded_root.exists():
            continue
        for include_glob in config.include_globs:
            for path in sorted(expanded_root.glob(include_glob)):
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                stat = path.stat()
                candidates.append(
                    JsonlCandidate(
                        path=path,
                        root=expanded_root,
                        root_index=index,
                        mtime=stat.st_mtime,
                        size=stat.st_size,
                    )
                )
    candidates.sort(key=lambda item: item.path.as_posix())
    return candidates


def _state_matches(
    entry: Any,
    candidate: JsonlCandidate,
    surviving_archives: dict[str, str | None] | None = None,
) -> bool:
    """A file is covered only while a SURVIVING archive object holds its exact bytes.

    ``surviving_archives`` is the set of archive object names still present in the
    backup folder. When it is supplied (the upload path) an entry must name one of
    them and its recorded content hash must still match the file on disk. Entries
    written before archive provenance was recorded name no archive, so they cannot
    prove survival and are deliberately treated as uncovered -- re-bundling costs a
    night of upload, trusting them can cost the only remaining copy.
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("mtime") != candidate.mtime or entry.get("size") != candidate.size:
        return False
    if surviving_archives is None:
        return True
    archive_id = entry.get("archive_id")
    if not isinstance(archive_id, str) or archive_id not in surviving_archives:
        return False
    recorded_md5 = entry.get("archive_md5")
    if isinstance(recorded_md5, str) and recorded_md5:
        live_md5 = surviving_archives[archive_id]
        # An object we cannot re-verify cannot prove coverage. Fail closed: the cost is
        # re-bundling, the cost of the other direction is the only remaining copy.
        if not isinstance(live_md5, str) or live_md5 != recorded_md5:
            return False
    recorded_hash = entry.get("sha256")
    if not isinstance(recorded_hash, str) or not recorded_hash:
        return False
    try:
        return recorded_hash == _sha256_file(candidate.path)
    except OSError:
        # The source vanished between discovery and hashing. It cannot prove coverage, and it
        # must not abort the run: this job is what PREVENTS data loss, so a single unreadable
        # file taking the whole nightly backup down is the wrong failure. _drop_vanished keeps
        # it out of the bundle too, so returning False here cannot strand it in `changed`.
        return False


def _backup_unit_key(candidate: JsonlCandidate) -> tuple[int, str]:
    path = candidate.path.as_posix()
    for sidecar_suffix in ("-wal", "-shm"):
        if path.endswith(f".db{sidecar_suffix}"):
            return (candidate.root_index, path[: -len(sidecar_suffix)])
    return (candidate.root_index, path)


def _select_backup_candidates(
    candidates: list[JsonlCandidate],
    *,
    state: dict[str, Any],
    now: float,
    active_skip_seconds: int,
    surviving_archives: dict[str, str | None] | None = None,
) -> tuple[list[JsonlCandidate], list[JsonlCandidate], int, int]:
    grouped: dict[tuple[int, str], list[JsonlCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(_backup_unit_key(candidate), []).append(candidate)

    changed: list[JsonlCandidate] = []
    active: list[JsonlCandidate] = []
    covered = 0
    vanished = 0
    state_files = state.get("files", {})
    for backup_unit in grouped.values():
        if any(now - candidate.mtime < active_skip_seconds for candidate in backup_unit):
            active.extend(backup_unit)
            continue
        if all(
            _state_matches(state_files.get(candidate.path.as_posix()), candidate, surviving_archives)
            for candidate in backup_unit
        ):
            covered += len(backup_unit)
            continue
        readable = [c for c in backup_unit if c.path.exists()]
        vanished += len(backup_unit) - len(readable)
        changed.extend(readable)
    return changed, active, covered, vanished


def _list_surviving_archives(service: Any, folder_parts: list[str]) -> dict[str, str | None]:
    """Surviving archive objects in the backup folder, keyed by Drive object ID.

    Keyed by ID, never by name: Drive permits duplicate names in one folder, so a
    same-named replacement would otherwise masquerade as the original object and
    prove a survival that never happened. The value is the object's md5Checksum
    when Drive reports one, so out-of-band modification of a surviving object is
    detectable too.
    """
    folder_id = backup_daily.ensure_drive_folder_chain(service, folder_parts)
    surviving: dict[str, str | None] = {}
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken,files(id,name,md5Checksum)",
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
            )
            .execute()
        )
        for item in response.get("files", []):
            file_id = item.get("id")
            if isinstance(file_id, str) and file_id:
                surviving[file_id] = item.get("md5Checksum")
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return surviving


def _archive_name(candidate: JsonlCandidate) -> str:
    try:
        relative = candidate.path.relative_to(candidate.root)
    except ValueError:
        relative = Path(candidate.path.name)
    return f"source-{candidate.root_index}/{relative.as_posix()}"


def _forever_enabled() -> bool:
    return os.environ.get("BRAINLAYER_JSONL_FOREVER", "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_gzip_payload(path: Path) -> str:
    """Address valid gzip archives by their logical payload, not header metadata.

    Retrying a bundle can change the gzip header timestamp while preserving the
    tar payload. A payload-derived destination therefore reuses the same iCloud
    object after a later Drive failure instead of accumulating untracked copies.
    Non-gzip inputs retain byte-addressed behavior for compatibility.
    """
    digest = hashlib.sha256()
    try:
        with gzip.open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except (gzip.BadGzipFile, EOFError, OSError):
        return _sha256_file(path)
    return digest.hexdigest()


_ICLOUD_STATUS_SCRIPT = r"""
ObjC.import("Foundation");
const args = $.NSProcessInfo.processInfo.arguments;
const action = ObjC.unwrap(args.objectAtIndex(args.count - 2));
const path = ObjC.unwrap(args.lastObject);
const url = $.NSURL.fileURLWithPath(path);
function resourceValue(key) {
    const value = Ref();
    const error = Ref();
    if (!url.getResourceValueForKeyError(value, key, error)) {
        const detail = error[0] ? ObjC.unwrap(error[0].localizedDescription) : "unknown error";
        throw new Error(detail);
    }
    if (value[0] === undefined || value[0] === null) return null;
    const unwrapped = ObjC.unwrap(value[0]);
    return unwrapped === undefined ? null : unwrapped;
}
if (action === "download") {
    const error = Ref();
    if (!$.NSFileManager.defaultManager.startDownloadingUbiquitousItemAtURLError(url, error)) {
        const detail = error[0] ? ObjC.unwrap(error[0].localizedDescription) : "unknown error";
        throw new Error(detail);
    }
}
const uploadError = resourceValue($.NSURLUbiquitousItemUploadingErrorKey);
JSON.stringify({
    is_ubiquitous: resourceValue($.NSURLIsUbiquitousItemKey),
    is_uploaded: resourceValue($.NSURLUbiquitousItemIsUploadedKey),
    is_uploading: resourceValue($.NSURLUbiquitousItemIsUploadingKey),
    downloading_status: resourceValue($.NSURLUbiquitousItemDownloadingStatusKey),
    uploading_error: uploadError === null ? null : String(uploadError)
});
"""


def _normalized_icloud_download_status(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    prefix = "NSURLUbiquitousItemDownloadingStatus"
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return text[:1].lower() + text[1:] if text else text


def _quarantine_unverified_icloud_item(path: Path) -> None:
    """Hide an unverified iCloud item without deleting personal backup data."""
    if not path.exists():
        return
    hidden_name = path.name if path.name.startswith(".") else f".{path.name}"
    os.replace(path, path.with_name(f"{hidden_name}.{uuid.uuid4().hex}.unverified"))


def _icloud_item_state(
    path: Path,
    *,
    request_download: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Read authoritative iCloud state, optionally forcing cloud materialization."""
    try:
        completed = subprocess.run(
            [
                "/usr/bin/osascript",
                "-l",
                "JavaScript",
                "-e",
                _ICLOUD_STATUS_SCRIPT,
                "--",
                "download" if request_download else "status",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or str(exc)
        raise RuntimeError(f"iCloud status probe failed for {path}: {detail}") from exc
    try:
        state = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"iCloud status returned invalid JSON: {completed.stdout!r}") from exc
    if not isinstance(state, dict):
        raise RuntimeError(f"iCloud status returned a non-object: {state!r}")
    state["downloading_status"] = _normalized_icloud_download_status(state.get("downloading_status"))
    return state


def copy_archive_to_icloud(
    archive_path: Path,
    icloud_dir: Path,
    *,
    timeout_seconds: float = DEFAULT_ICLOUD_TIMEOUT_SECONDS,
    poll_interval_seconds: float = 2.0,
) -> dict[str, Any]:
    """Copy to iCloud, force materialization, then verify authoritative state and bytes."""
    archive_path = Path(archive_path).expanduser()
    icloud_dir = Path(icloud_dir).expanduser()
    icloud_dir.mkdir(parents=True, exist_ok=True)
    expected_size = archive_path.stat().st_size
    expected_sha256 = _sha256_file(archive_path)
    logical_sha256 = _sha256_gzip_payload(archive_path)
    suffix = "".join(archive_path.suffixes)
    stem = archive_path.name[: -len(suffix)] if suffix else archive_path.name
    destination = icloud_dir / f"{stem}-{logical_sha256}{suffix}"
    placeholder = destination.with_name(f".{destination.name}.icloud")

    def receipt(*, reused: bool) -> dict[str, Any]:
        actual_size = destination.stat().st_size
        actual_sha256 = _sha256_file(destination)
        return {
            "path": str(destination),
            "uploaded": True,
            "materialization": "MATERIALIZED",
            "bytes": actual_size,
            "sha256": actual_sha256,
            "source_sha256": expected_sha256,
            "logical_sha256": logical_sha256,
            "downloading_status": "current",
            "reused": reused,
        }

    # A logical address is immutable. If a prior verified copy already occupies
    # it, materialize and validate that object rather than replacing it before a
    # retry is proven. Gzip header metadata may differ while the tar payload is
    # identical, which is exactly what the logical address represents.
    if destination.exists() or placeholder.exists():
        deadline = time.monotonic() + timeout_seconds

        def existing_remaining_seconds() -> float:
            return max(deadline - time.monotonic(), 0.001)

        status_path = placeholder if not destination.exists() and placeholder.exists() else destination
        state = _icloud_item_state(
            status_path,
            request_download=True,
            timeout_seconds=existing_remaining_seconds(),
        )
        while True:
            uploading_error = state.get("uploading_error")
            if uploading_error:
                raise RuntimeError(f"existing iCloud upload failed for {destination}: {uploading_error}")
            uploaded = state.get("is_uploaded") is True and state.get("is_uploading") is False
            materialized = state.get("downloading_status") == "current" and destination.is_file()
            if state.get("is_ubiquitous") is True and uploaded and materialized:
                actual_logical_sha256 = _sha256_gzip_payload(destination)
                if actual_logical_sha256 != logical_sha256:
                    raise RuntimeError(
                        "iCloud logical-address collision: "
                        f"path={destination} expected={logical_sha256} actual={actual_logical_sha256}"
                    )
                return receipt(reused=True)
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"existing iCloud copy was not uploaded and materialized within {timeout_seconds}s: "
                    f"path={destination} state={state!r}"
                )
            time.sleep(min(poll_interval_seconds, existing_remaining_seconds()))
            status_path = placeholder if not destination.exists() and placeholder.exists() else destination
            state = _icloud_item_state(
                status_path,
                request_download=True,
                timeout_seconds=existing_remaining_seconds(),
            )

    temp_path = icloud_dir / f".{destination.name}.{os.getpid()}.partial"
    try:
        with archive_path.open("rb") as source, temp_path.open("xb") as destination_handle:
            shutil.copyfileobj(source, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)

    deadline = time.monotonic() + timeout_seconds

    def remaining_seconds() -> float:
        return max(deadline - time.monotonic(), 0.001)

    verified = False
    try:
        state = _icloud_item_state(destination, request_download=True, timeout_seconds=remaining_seconds())
        while True:
            uploading_error = state.get("uploading_error")
            if uploading_error:
                raise RuntimeError(f"iCloud upload failed for {destination}: {uploading_error}")
            uploaded = state.get("is_uploaded") is True and state.get("is_uploading") is False
            materialized = state.get("downloading_status") == "current" and destination.is_file()
            if state.get("is_ubiquitous") is True and uploaded and materialized:
                actual_size = destination.stat().st_size
                actual_sha256 = _sha256_file(destination)
                if actual_size != expected_size or actual_sha256 != expected_sha256:
                    raise RuntimeError(
                        "iCloud copy content mismatch: "
                        f"expected size={expected_size} sha256={expected_sha256}, "
                        f"actual size={actual_size} sha256={actual_sha256}"
                    )
                verified = True
                return receipt(reused=False)
            if time.monotonic() >= deadline:
                placeholder = destination.with_name(f".{destination.name}.icloud")
                materialization = "PLACEHOLDER" if placeholder.exists() or not destination.exists() else "PENDING"
                raise RuntimeError(
                    f"iCloud copy was not uploaded and materialized within {timeout_seconds}s: "
                    f"path={destination} materialization={materialization} state={state!r}"
                )
            time.sleep(min(poll_interval_seconds, remaining_seconds()))
            placeholder = destination.with_name(f".{destination.name}.icloud")
            status_path = placeholder if not destination.exists() and placeholder.exists() else destination
            state = _icloud_item_state(status_path, request_download=True, timeout_seconds=remaining_seconds())
    finally:
        if not verified:
            _quarantine_unverified_icloud_item(destination)
            _quarantine_unverified_icloud_item(destination.with_name(f".{destination.name}.icloud"))


def _upload_forever_files(
    candidates: list[JsonlCandidate],
    *,
    service: Any,
    credentials: Any,
    staging_dir: Path,
    forever_folder_parts: list[str],
) -> list[dict[str, Any]]:
    staging_dir = Path(staging_dir).expanduser()
    staging_dir.mkdir(parents=True, exist_ok=True)
    uploaded: list[dict[str, Any]] = []
    folder_ids_by_root_index: dict[int, str] = {}
    for candidate in candidates:
        with tempfile.TemporaryDirectory(prefix=".forever-", dir=staging_dir) as tmp_dir:
            suffix = candidate.path.suffix or ".bin"
            temp_path = Path(tmp_dir) / f"payload{suffix}"
            shutil.copyfile(candidate.path, temp_path)
            sha256 = _sha256_file(temp_path)
            forever_name = f"{sha256}{suffix}"
            forever_path = Path(tmp_dir) / forever_name
            os.replace(temp_path, forever_path)
            folder_parts = [*forever_folder_parts, f"source-{candidate.root_index}"]
            folder_id = folder_ids_by_root_index.get(candidate.root_index)
            if folder_id is None:
                folder_id = backup_daily.ensure_drive_folder_chain(service, folder_parts)
                folder_ids_by_root_index[candidate.root_index] = folder_id
            drive_file = backup_daily.upload_file_to_drive_raw(forever_path, folder_id, credentials)
            file_id = drive_file.get("id")
            if not file_id:
                raise RuntimeError(f"Drive forever upload response missing file id: {drive_file!r}")
            backup_daily.verify_drive_upload(
                service,
                file_id=file_id,
                expected_name=forever_name,
                expected_size=forever_path.stat().st_size,
            )
            uploaded.append(
                {
                    "source_root_index": candidate.root_index,
                    "source_suffix": suffix,
                    "drive_file": drive_file,
                    "sha256": sha256,
                    "folder_parts": folder_parts,
                }
            )
    return uploaded


class _HashingReader:
    """File wrapper that digests exactly the bytes handed to tarfile.

    tarfile pulls through ``read``, so the digest is of the archived content itself --
    not of a separate read that could see different bytes -- at O(buffer) memory rather
    than O(file). The largest real source JSONL is ~375MB, so that distinction matters.
    """

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self._handle.read(size)
        self._digest.update(chunk)
        return chunk

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def create_jsonl_bundle_with_digests(
    candidates: list[JsonlCandidate], staging_dir: Path, *, date_stamp: str
) -> tuple[Path, dict[str, str]]:
    if not candidates:
        raise ValueError("create_jsonl_bundle requires at least one candidate")
    staging_dir = Path(staging_dir).expanduser()
    staging_dir.mkdir(parents=True, exist_ok=True)
    archive_path = staging_dir / f"claude-jsonl-{date_stamp}.tar.gz"
    with tempfile.NamedTemporaryFile(
        prefix=f".{archive_path.name}.", suffix=".tmp", dir=staging_dir, delete=False
    ) as tmp:
        temp_path = Path(tmp.name)
    digests: dict[str, str] = {}
    try:
        with tarfile.open(temp_path, "w:gz") as tar:
            for candidate in candidates:
                # Hash and archive the SAME bytes, without holding the file in memory.
                # Re-reading the source afterwards could record a digest for content the
                # archive does not contain -- a file that changed while keeping its mtime
                # and size would then read as covered. Sources reach ~375MB, so the digest
                # is taken from the very stream tarfile consumes rather than from a copy.
                info = tar.gettarinfo(str(candidate.path), arcname=_archive_name(candidate))
                with candidate.path.open("rb") as handle:
                    reader = _HashingReader(handle)
                    tar.addfile(info, reader)
                digests[candidate.path.as_posix()] = reader.hexdigest()
        os.replace(temp_path, archive_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return archive_path, digests


def create_jsonl_bundle(candidates: list[JsonlCandidate], staging_dir: Path, *, date_stamp: str) -> Path:
    """Backwards-compatible wrapper returning only the archive path."""
    archive_path, _ = create_jsonl_bundle_with_digests(candidates, staging_dir, date_stamp=date_stamp)
    return archive_path


def verify_jsonl_bundle(archive_path: Path, *, expected_file_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "verified": False,
        "bundled_file_count": expected_file_count,
        "archive_listing_count": 0,
    }
    try:
        subprocess.run(["gunzip", "-t", str(archive_path)], check=True, capture_output=True, text=True)
        result["gzip_test"] = True
        listing = subprocess.run(["tar", "-tzf", str(archive_path)], check=True, capture_output=True, text=True)
        entries = [line for line in listing.stdout.splitlines() if line.strip() and not line.endswith("/")]
        result["archive_listing_count"] = len(entries)
        if len(entries) != expected_file_count:
            result["verification_error"] = (
                f"tar listing count mismatch: expected={expected_file_count} actual={len(entries)}"
            )
            return result
        result["verified"] = True
    except Exception as exc:
        result.setdefault("gzip_test", False)
        result["verification_error"] = str(exc)
    return result


def _update_state_for_uploaded(
    state: dict[str, Any],
    candidates: list[JsonlCandidate],
    archive_name: str | None = None,
    *,
    archive_id: str | None = None,
    archive_md5: str | None = None,
    digests: dict[str, str] | None = None,
    icloud_dir: Path | None = None,
) -> dict[str, Any]:
    """Record which archive object carries each file, and the bytes it carried.

    Without this provenance a later prune cannot know it is removing the last
    surviving copy of a file, which is exactly how a covered file becomes
    unrecoverable while state still reports it as backed up.
    """
    files = dict(state.get("files") or {})
    for candidate in candidates:
        entry: dict[str, Any] = {"mtime": candidate.mtime, "size": candidate.size}
        if archive_name and archive_id:
            entry["archive"] = archive_name
            entry["archive_id"] = archive_id
            if archive_md5:
                entry["archive_md5"] = archive_md5
            digest = (digests or {}).get(candidate.path.as_posix())
            entry["sha256"] = digest if digest else _sha256_file(candidate.path)
        files[candidate.path.as_posix()] = entry
    updated = {"files": files, "updated_at": dt.datetime.now(dt.UTC).isoformat()}
    if icloud_dir is not None:
        updated["icloud_directory"] = str(Path(icloud_dir).expanduser())
        updated["icloud_verified"] = True
    return updated


def _enqueue_run_summary(result: dict[str, Any], *, queue_dir: Path | None) -> None:
    if result["status"] == "uploaded":
        content = (
            f"JSONL backup uploaded {result['bundled_file_count']} files to Drive; "
            f"verified={result['verified']} archive={result.get('archive')}"
        )
        importance = 7 if result.get("verified") else 9
    elif result["status"] == "created":
        content = (
            f"JSONL backup created local bundle with {result['bundled_file_count']} files; "
            f"verified={result['verified']} archive={result.get('archive')}"
        )
        importance = 6 if result.get("verified") else 9
    else:
        content = f"JSONL backup {result['message']}"
        importance = 5
    enqueue_store(
        content=content,
        memory_type="milestone",
        project="brainlayer",
        tags=["backup", "jsonl"],
        importance=importance,
        source="jsonl_backup",
        queue_dir=queue_dir,
    )


def run_backup(
    *,
    source_roots: list[Path | BackupSourceRoot] | None = None,
    state_path: Path = DEFAULT_STATE_PATH,
    staging_dir: Path = DEFAULT_STAGING_DIR,
    log_path: Path = DEFAULT_LOG_PATH,
    queue_dir: Path | None = None,
    folder_parts: list[str] = DEFAULT_FOLDER_PARTS,
    date_stamp: str | None = None,
    now: float | None = None,
    upload: bool = True,
    active_skip_seconds: int = DEFAULT_ACTIVE_SKIP_SECONDS,
    forever_folder_parts: list[str] = DEFAULT_FOREVER_FOLDER_PARTS,
    icloud_dir: Path | None = None,
) -> dict[str, Any]:
    date_stamp = date_stamp or _today()
    now = time.time() if now is None else now
    attempted_at = dt.datetime.fromtimestamp(now, dt.UTC).isoformat()
    roots = source_roots or DEFAULT_SOURCE_ROOTS
    state_path = Path(state_path).expanduser()
    state = _load_state(state_path)
    selection_state = state
    icloud_bootstrap_pending = False
    if upload and icloud_dir is not None:
        configured_icloud_dir = str(Path(icloud_dir).expanduser())
        icloud_covered = state.get("icloud_verified") is True and state.get("icloud_directory") == configured_icloud_dir
        if not icloud_covered:
            # Legacy state proves only Drive coverage. The first iCloud-enabled run
            # must seed iCloud with every source rather than falsely returning no-op.
            selection_state = {"files": {}}
            icloud_bootstrap_pending = True
    candidates = _discover_jsonl_candidates(roots)
    credentials = None
    service = None
    surviving_archives: dict[str, str | None] | None = None
    if upload:
        # Authenticate only when a listing is actually needed. When no entry claims
        # archive-backed coverage there is nothing to verify, so a run that would be a
        # clean no-op does not touch Drive. Once entries DO claim coverage the listing is
        # unavoidable -- survival cannot be proven without asking.
        if any(isinstance(e, dict) and e.get("archive_id") for e in (state.get("files") or {}).values()):
            credentials = backup_daily.get_drive_credentials()
            service = backup_daily.build_drive_service()
            surviving_archives = _list_surviving_archives(service, folder_parts)
        else:
            surviving_archives = {}
    changed, active, covered, vanished = _select_backup_candidates(
        candidates,
        state=selection_state,
        now=now,
        active_skip_seconds=active_skip_seconds,
        surviving_archives=surviving_archives,
    )

    if not changed:
        result: dict[str, Any] = {
            "attempted_at": attempted_at,
            "status": "no-op",
            "uploaded": False,
            "verified": True,
            "already_covered_files": covered,
            "discovered_file_count": len(candidates),
            "skipped_active_count": len(active),
            "vanished_source_count": vanished,
            "message": f"no-op, {covered} files already covered",
        }
        _append_json_log(log_path, result)
        _enqueue_run_summary(result, queue_dir=queue_dir)
        return result

    archive_path, bundle_digests = create_jsonl_bundle_with_digests(changed, staging_dir, date_stamp=date_stamp)
    archive_size = archive_path.stat().st_size
    result = {
        "attempted_at": attempted_at,
        "status": "uploaded" if upload else "created",
        "archive": str(archive_path),
        "bytes": archive_size,
        "uploaded": False,
        "verified": False,
        "bundled_file_count": len(changed),
        "skipped_active_count": len(active),
        "already_covered_files": covered,
        "vanished_source_count": vanished,
        "source_file_count": len(candidates),
        "retention_deleted": [],
        "forever_uploaded_file_count": 0,
        "forever_files": [],
    }

    result.update(verify_jsonl_bundle(archive_path, expected_file_count=len(changed)))
    if result["verified"] and upload:
        # iCloud goes first: a failed iCloud verification must not create an
        # unrecorded duplicate Drive object that consumes the retention window.
        if icloud_dir is not None:
            result["icloud_copy"] = copy_archive_to_icloud(archive_path, icloud_dir)
        if service is None:
            credentials = backup_daily.get_drive_credentials()
            service = backup_daily.build_drive_service()
        folder_id = backup_daily.ensure_drive_folder_chain(service, folder_parts)
        uploaded = backup_daily.upload_file_to_drive_raw(archive_path, folder_id, credentials)
        file_id = uploaded.get("id")
        if not file_id:
            raise RuntimeError(f"Drive upload response missing file id: {uploaded!r}")
        backup_daily.verify_drive_upload(
            service,
            file_id=file_id,
            expected_name=archive_path.name,
            expected_size=archive_size,
        )
        result.update({"uploaded": True, "drive_file": uploaded})
        _atomic_write_json(
            state_path,
            _update_state_for_uploaded(
                state,
                changed,
                archive_path.name,
                archive_id=file_id,
                archive_md5=uploaded.get("md5Checksum"),
                digests=bundle_digests,
                # An active source was deliberately omitted from this bootstrap.
                # Leave the global marker unset so the next run seeds that source.
                icloud_dir=icloud_dir if not (icloud_bootstrap_pending and active) else None,
            ),
        )
        try:
            deleted = backup_daily.prune_drive_backups(
                service,
                folder_parts=folder_parts,
                retention_policy=JSONL_RETENTION,
            )
            result["retention_deleted"] = deleted
            if _forever_enabled():
                forever_files = _upload_forever_files(
                    changed,
                    service=service,
                    credentials=credentials,
                    staging_dir=staging_dir,
                    forever_folder_parts=forever_folder_parts,
                )
                result["forever_files"] = forever_files
                result["forever_uploaded_file_count"] = len(forever_files)
        finally:
            archive_path.unlink(missing_ok=True)
            result["local_archive_removed"] = True

    _append_json_log(log_path, result)
    _enqueue_run_summary(result, queue_dir=queue_dir)
    return result


def _raise_backup_timeout(signum, frame) -> None:  # noqa: ARG001
    raise backup_daily.BackupTimeoutError("jsonl backup exceeded configured wall-clock timeout")


def _append_terminal_failure(log_path: Path, result: dict[str, Any]) -> None:
    """Persist terminal failures without hiding the original failure if logging also breaks."""
    try:
        _append_json_log(log_path, result)
    except Exception as exc:
        result["attempt_log_error"] = str(exc)


def main() -> int:
    timeout_seconds = _configured_backup_timeout_seconds()
    log_path = Path(os.environ.get("BRAINLAYER_JSONL_BACKUP_LOG_PATH", str(DEFAULT_LOG_PATH)))
    previous_alarm_handler = None
    if timeout_seconds is not None:
        previous_alarm_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_backup_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        result = run_backup(
            staging_dir=Path(os.environ.get("BRAINLAYER_JSONL_BACKUP_STAGING_DIR", str(DEFAULT_STAGING_DIR))),
            state_path=Path(os.environ.get("BRAINLAYER_JSONL_BACKUP_STATE_PATH", str(DEFAULT_STATE_PATH))),
            log_path=log_path,
            folder_parts=os.environ.get("BRAINLAYER_JSONL_BACKUP_DRIVE_FOLDER", "/".join(DEFAULT_FOLDER_PARTS)).split(
                "/"
            ),
            icloud_dir=_configured_icloud_dir(),
        )
    except backup_daily.BackupTimeoutError:
        result = {
            "attempted_at": dt.datetime.now(dt.UTC).isoformat(),
            "status": "failed",
            "uploaded": False,
            "verified": False,
            "error": f"timed out after {timeout_seconds}s",
        }
        _append_terminal_failure(log_path, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 124
    except Exception as exc:
        result = {
            "attempted_at": dt.datetime.now(dt.UTC).isoformat(),
            "status": "failed",
            "uploaded": False,
            "verified": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _append_terminal_failure(log_path, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 1
    finally:
        if timeout_seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm_handler)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("verified", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
