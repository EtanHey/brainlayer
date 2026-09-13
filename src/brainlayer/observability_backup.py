from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from brainlayer import jsonl_backup
from brainlayer.backup_retention_invariant import inspect_jsonl_retention_invariant
from brainlayer.health_check import (
    DEFAULT_JSONL_BACKUP_MAX_AGE_SECONDS,
    _jsonl_backup_attempt_time,
    inspect_jsonl_backup_health,
)

LABEL = "com.brainlayer.jsonl-backup"
THRESHOLD_HOURS = DEFAULT_JSONL_BACKUP_MAX_AGE_SECONDS // 3600


def _path(env: Mapping[str, str], name: str) -> Path:
    value = env.get(name)
    return Path(value).expanduser() if value else Path(f"/__brainlayer_missing_input__/{name}")


def _read_json_lines(path: Path, *, mixed: bool) -> tuple[list[dict[str, Any]], str, int]:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return [], "missing", 0
    except OSError:
        return [], "malformed", 0
    if size == 0:
        return [], "empty", 0
    records: list[dict[str, Any]] = []
    skipped = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return [], "malformed", 0
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            if mixed:
                skipped += 1
                continue
            return [], "malformed", 0
        if not isinstance(item, dict):
            if mixed:
                skipped += 1
                continue
            return [], "malformed", 0
        records.append(item)
    if not records:
        return [], "malformed", skipped
    return records, "read", skipped


def _coverage(records: list[dict[str, Any]], now: datetime) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for record in reversed(records):
        archive_id = record.get("archive_id")
        attempted_at = _jsonl_backup_attempt_time(record)
        if record.get("verified") is True and isinstance(archive_id, str) and archive_id and attempted_at:
            age = max(0.0, (now.astimezone(UTC) - attempted_at).total_seconds() / 3600)
            return (
                {
                    "at": attempted_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                    "age_hours": round(age, 3),
                    "archive_id": archive_id,
                    "verified": True,
                },
                record,
            )
    return None, None


def _daily_backup_attempt_time(record: dict[str, Any]) -> datetime | None:
    attempted_at = record.get("attempted_at")
    if isinstance(attempted_at, str) and attempted_at:
        try:
            parsed = datetime.fromisoformat(attempted_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo else None
    snapshot = record.get("snapshot")
    if not isinstance(snapshot, str):
        return None
    match = re.search(r"(?:^|/)\d{4}-\d{2}-\d{2}\.db\.gz$", snapshot)
    if not match:
        return None
    date = snapshot.rsplit("/", 1)[-1][:-6]
    try:
        return datetime.fromisoformat(f"{date}T05:00:00").astimezone(UTC)
    except ValueError:
        return None


def _retention_status() -> str:
    source = Path(jsonl_backup.__file__).resolve()
    sibling = source.with_name("backup_daily.py")
    try:
        errors = inspect_jsonl_retention_invariant(
            source.read_text(encoding="utf-8"),
            backup_daily_source=sibling.read_text(encoding="utf-8"),
        )
    except OSError:
        return "unknown"
    return "FAIL" if errors else "PASS"


def _daily_snapshot(records: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None, bool]:
    real = [record for record in records if record.get("backup_log_provenance") == "real"]
    if not real:
        return None, None, False
    all_errors = all(record.get("error") or record.get("error_type") for record in real)
    latest = real[-1]
    error_type = str(latest["error_type"]) if latest.get("error_type") else None
    if "google-drive-mcp/tokens.json" in str(latest.get("error", "")):
        error_type = "drive_credentials_missing"
    elif all_errors and error_type is None:
        error_type = "backup_error"
    for record in reversed(real):
        attempted_at = _daily_backup_attempt_time(record)
        destination = record.get("destination")
        if attempted_at and isinstance(destination, str) and record.get("snapshot"):
            return (
                {
                    "last_at": attempted_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                    "destination": destination,
                    "verified": record.get("verified") is True,
                },
                error_type,
                all_errors,
            )
    return None, error_type, all_errors


def _surviving_archives_30d(records: list[dict[str, Any]], now: datetime) -> int:
    cutoff = now.astimezone(UTC).timestamp() - 30 * 24 * 3600
    archive_ids: set[str] = set()
    for record in records:
        archive_id = record.get("archive_id")
        attempted_at = _jsonl_backup_attempt_time(record)
        if (
            record.get("verified") is True
            and isinstance(archive_id, str)
            and archive_id
            and attempted_at is not None
            and cutoff <= attempted_at.timestamp() <= now.astimezone(UTC).timestamp()
        ):
            archive_ids.add(archive_id)
    return len(archive_ids)


def _unmeasurable(reason: str, inputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {"state": "unmeasurable", "reason": reason, "inputs": inputs}


def build_backups_section(
    *,
    env: Mapping[str, str],
    record_input: Callable[..., dict],
    now: datetime,
) -> dict[str, Any]:
    """Return the schema-v1 backup measurement, recording every touched input."""
    jsonl_path = _path(env, "BRAINLAYER_OBSERVABILITY_JSONL_BACKUP_LOG")
    daily_path = _path(env, "BRAINLAYER_OBSERVABILITY_BACKUP_DAILY_LOG")
    launchd_path = _path(env, "BRAINLAYER_OBSERVABILITY_LAUNCHD_OUTPUT")
    disabled_path = _path(env, "BRAINLAYER_OBSERVABILITY_DISABLED_DIR")

    jsonl_records, jsonl_status, _ = _read_json_lines(jsonl_path, mixed=False)
    jsonl_input = record_input(jsonl_path, status=jsonl_status, rows_or_bytes=None)
    daily_records, daily_status, skipped = _read_json_lines(daily_path, mixed=True)
    daily_input = record_input(daily_path, status=daily_status, rows_or_bytes=None, skipped_lines=skipped)

    try:
        launchd_text = launchd_path.read_text(encoding="utf-8")
        launchd_status = "empty" if not launchd_text else "read"
    except FileNotFoundError:
        launchd_text, launchd_status = "", "missing"
    except (OSError, UnicodeDecodeError):
        launchd_text, launchd_status = "", "malformed"
    launchd_input = record_input(launchd_path, status=launchd_status, rows_or_bytes=None)

    try:
        disabled_count = len(list(disabled_path.iterdir()))
        disabled_present, disabled_status = True, "read"
    except FileNotFoundError:
        disabled_count, disabled_present, disabled_status = None, False, "missing"
    except OSError:
        disabled_count, disabled_present, disabled_status = None, False, "malformed"
    disabled_input = record_input(
        disabled_path,
        status=disabled_status,
        rows_or_bytes=disabled_count,
        in_section_inputs=False,
    )
    inputs = [jsonl_input, daily_input, launchd_input]

    for item in [*inputs, disabled_input]:
        if item.get("status") == "future":
            return _unmeasurable(f"input mtime is later than generated_at: {item['path']}", inputs)
    jsonl_status = str(jsonl_input.get("status"))
    daily_status = str(daily_input.get("status"))
    launchd_status = str(launchd_input.get("status"))
    if jsonl_status == "missing":
        return _unmeasurable(f"required input missing: {jsonl_input['path']}", inputs)
    if jsonl_status == "malformed":
        return _unmeasurable(f"malformed JSON line in {jsonl_input['path']}", inputs)
    if jsonl_status == "empty":
        return _unmeasurable(f"required input empty: {jsonl_input['path']}", inputs)
    if daily_status != "read":
        return _unmeasurable(f"backup daily log is {daily_status}: {daily_input['path']}", inputs)
    if not any(record.get("backup_log_provenance") == "real" for record in daily_records):
        return _unmeasurable("backup daily log has no real-provenance receipts", inputs)
    if launchd_status != "read":
        return _unmeasurable(f"launchd output is {launchd_status}", inputs)
    if LABEL not in launchd_text and "Could not find service" not in launchd_text:
        return _unmeasurable("launchd output is unrecognized", inputs)
    if disabled_input.get("status") == "malformed":
        return _unmeasurable(f"disabled launchd directory is unreadable: {disabled_input['path']}", inputs)

    last_upload, _ = _coverage(jsonl_records, now)
    health, health_issue = inspect_jsonl_backup_health(
        jsonl_path,
        now=now,
        max_age_seconds=DEFAULT_JSONL_BACKUP_MAX_AGE_SECONDS,
    )
    snapshot, error_type, all_daily_errors = _daily_snapshot(daily_records)
    if error_type is None and health_issue is not None and health.state in {"invalid", "stale", "failed"}:
        error_type = health_issue.code
    if all_daily_errors or health.state in {"stale", "failed"}:
        freshness = "stale"
    elif health.state in {"no_op", "verified_bundle"}:
        freshness = "fresh"
    else:
        freshness = "unknown"
    bootstrapped = "Could not find service" not in launchd_text and LABEL in launchd_text

    return {
        "state": "measured",
        "reason": "",
        "inputs": inputs,
        "last_verified_upload": last_upload,
        "freshness": freshness,
        "threshold_hours": THRESHOLD_HOURS,
        "retention_invariant": _retention_status(),
        "surviving_archives_30d": _surviving_archives_30d(jsonl_records, now),
        "launchd": {
            "label": LABEL,
            "bootstrapped": bootstrapped,
            "disabled_dir_present": disabled_present,
        },
        "db_snapshot": snapshot,
        "error_type": error_type,
    }
