"""Nightly JSONL transcript backups to Google Drive.

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
DEFAULT_ACTIVE_SKIP_SECONDS = 10 * 60
DEFAULT_TIMEOUT_SECONDS = 1800
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


def _state_matches(entry: Any, candidate: JsonlCandidate) -> bool:
    if not isinstance(entry, dict):
        return False
    return entry.get("mtime") == candidate.mtime and entry.get("size") == candidate.size


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
) -> tuple[list[JsonlCandidate], list[JsonlCandidate], int]:
    grouped: dict[tuple[int, str], list[JsonlCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(_backup_unit_key(candidate), []).append(candidate)

    changed: list[JsonlCandidate] = []
    active: list[JsonlCandidate] = []
    covered = 0
    state_files = state.get("files", {})
    for backup_unit in grouped.values():
        if any(now - candidate.mtime < active_skip_seconds for candidate in backup_unit):
            active.extend(backup_unit)
            continue
        if all(_state_matches(state_files.get(candidate.path.as_posix()), candidate) for candidate in backup_unit):
            covered += len(backup_unit)
            continue
        changed.extend(backup_unit)
    return changed, active, covered


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


def create_jsonl_bundle(candidates: list[JsonlCandidate], staging_dir: Path, *, date_stamp: str) -> Path:
    if not candidates:
        raise ValueError("create_jsonl_bundle requires at least one candidate")
    staging_dir = Path(staging_dir).expanduser()
    staging_dir.mkdir(parents=True, exist_ok=True)
    archive_path = staging_dir / f"claude-jsonl-{date_stamp}.tar.gz"
    with tempfile.NamedTemporaryFile(
        prefix=f".{archive_path.name}.", suffix=".tmp", dir=staging_dir, delete=False
    ) as tmp:
        temp_path = Path(tmp.name)
    try:
        with tarfile.open(temp_path, "w:gz") as tar:
            for candidate in candidates:
                tar.add(candidate.path, arcname=_archive_name(candidate), recursive=False)
        os.replace(temp_path, archive_path)
    finally:
        temp_path.unlink(missing_ok=True)
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


def _update_state_for_uploaded(state: dict[str, Any], candidates: list[JsonlCandidate]) -> dict[str, Any]:
    files = dict(state.get("files") or {})
    for candidate in candidates:
        files[candidate.path.as_posix()] = {"mtime": candidate.mtime, "size": candidate.size}
    return {"files": files, "updated_at": dt.datetime.now(dt.UTC).isoformat()}


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
) -> dict[str, Any]:
    date_stamp = date_stamp or _today()
    now = time.time() if now is None else now
    roots = source_roots or DEFAULT_SOURCE_ROOTS
    state_path = Path(state_path).expanduser()
    state = _load_state(state_path)
    candidates = _discover_jsonl_candidates(roots)
    changed, active, covered = _select_backup_candidates(
        candidates,
        state=state,
        now=now,
        active_skip_seconds=active_skip_seconds,
    )

    if not changed:
        result: dict[str, Any] = {
            "status": "no-op",
            "uploaded": False,
            "verified": True,
            "already_covered_files": covered,
            "discovered_file_count": len(candidates),
            "skipped_active_count": len(active),
            "message": f"no-op, {covered} files already covered",
        }
        _append_json_log(log_path, result)
        _enqueue_run_summary(result, queue_dir=queue_dir)
        return result

    archive_path = create_jsonl_bundle(changed, staging_dir, date_stamp=date_stamp)
    archive_size = archive_path.stat().st_size
    result = {
        "status": "uploaded" if upload else "created",
        "archive": str(archive_path),
        "bytes": archive_size,
        "uploaded": False,
        "verified": False,
        "bundled_file_count": len(changed),
        "skipped_active_count": len(active),
        "already_covered_files": covered,
        "source_file_count": len(candidates),
        "retention_deleted": [],
        "forever_uploaded_file_count": 0,
        "forever_files": [],
    }

    if upload:
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

    result.update(verify_jsonl_bundle(archive_path, expected_file_count=len(changed)))
    if result["verified"] and upload:
        _atomic_write_json(state_path, _update_state_for_uploaded(state, changed))
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


def main() -> int:
    timeout_seconds = _configured_backup_timeout_seconds()
    previous_alarm_handler = None
    if timeout_seconds is not None:
        previous_alarm_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_backup_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        result = run_backup(
            staging_dir=Path(os.environ.get("BRAINLAYER_JSONL_BACKUP_STAGING_DIR", str(DEFAULT_STAGING_DIR))),
            state_path=Path(os.environ.get("BRAINLAYER_JSONL_BACKUP_STATE_PATH", str(DEFAULT_STATE_PATH))),
            log_path=Path(os.environ.get("BRAINLAYER_JSONL_BACKUP_LOG_PATH", str(DEFAULT_LOG_PATH))),
            folder_parts=os.environ.get("BRAINLAYER_JSONL_BACKUP_DRIVE_FOLDER", "/".join(DEFAULT_FOLDER_PARTS)).split(
                "/"
            ),
        )
    except backup_daily.BackupTimeoutError:
        result = {
            "status": "failed",
            "uploaded": False,
            "verified": False,
            "error": f"timed out after {timeout_seconds}s",
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        return 124
    except Exception as exc:
        result = {
            "status": "failed",
            "uploaded": False,
            "verified": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
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
