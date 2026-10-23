"""Versioned health-check state consumed by the BrainBar menu-bar badge."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .health_check import HealthCheckResult, HealthIssue

SCHEMA_VERSION = 1
BADGE_STATE_ENV = "BRAINLAYER_BADGE_STATE_PATH"
FIRST_RUN_DEADLINE_SECONDS = 600

DATA_LOSS_CODES = frozenset(
    {
        "backup_daily_verification_failed",
        "jsonl_backup_attempt_absent_multiple_nights",
        "jsonl_backup_attempt_failed",
        "jsonl_backup_attempt_invalid",
        "jsonl_backup_attempt_missing",
        "jsonl_backup_attempt_stale",
    }
)

# Eligibility alone never suppresses an issue. A trusted producer must attach a
# current owned episode with a fixed deadline. Backlog/age and data-loss faults
# cannot be excused by snapshot activity or by positive throughput alone.
SUPPRESSIBLE_CODES = frozenset({"brain_search_canary_failed"})
EPISODE_MAX_SECONDS = 3600
EPISODE_FRESH_SECONDS = 300

if DATA_LOSS_CODES & SUPPRESSIBLE_CODES:
    raise RuntimeError("data-loss badge codes must never be suppressible")


def badge_state_path(db_path: Path, *, env: Mapping[str, str] = os.environ) -> Path:
    """Resolve the dedicated state file beside the selected database."""

    if override := env.get(BADGE_STATE_ENV):
        return Path(override).expanduser()
    return db_path.expanduser().resolve().parent / "badge-state.json"


def _iso_utc(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _issue_payload(issue: HealthIssue) -> dict[str, str]:
    return {"code": issue.code, "severity": issue.severity, "message": issue.message}


def _episode_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("episode timestamps require a timezone")
    return parsed


def _expected_episode(issue: HealthIssue, now: datetime) -> dict[str, Any] | None:
    episode = issue.expected_episode
    if issue.severity != "critical" or issue.code not in SUPPRESSIBLE_CODES or not isinstance(episode, dict):
        return None
    try:
        started, observed, deadline = (_episode_time(episode[key]) for key in ("started_at", "observed_at", "deadline"))
        valid = (
            episode["kind"] in ("snapshot", "maintenance")
            and all(isinstance(episode[key], str) and episode[key].strip() for key in ("owner", "episode_id"))
            and isinstance(episode["expected_codes"], list)
            and issue.code in episode["expected_codes"]
            and started <= observed <= now < deadline
            and (deadline - started).total_seconds() <= EPISODE_MAX_SECONDS
            and (now - observed).total_seconds() <= EPISODE_FRESH_SECONDS
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if not valid:
        return None
    return {
        key: episode[key]
        for key in ("kind", "owner", "episode_id", "started_at", "observed_at", "deadline", "expected_codes")
    }


def build_badge_state_document(result: HealthCheckResult, *, now: datetime | None = None) -> dict[str, Any]:
    """Keep all alerts; only explicit bounded evidence can excuse a critical."""

    now = now or datetime.fromisoformat(result.checked_at.replace("Z", "+00:00"))
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    active, suppressed = [], []
    for issue in result.issues:
        payload = _issue_payload(issue)
        if episode := _expected_episode(issue, now):
            suppressed.append({**payload, "episode": episode})
        else:
            active.append(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso_utc(result.checked_at),
        "alerts": {
            "state": "measured",
            "reason": "",
            "inputs": [],
            "badge_on": any(issue["severity"] != "warning" for issue in active),
            "active": active,
            "suppressed": suppressed,
        },
    }


def build_pending_first_run_document(now: datetime) -> dict[str, Any]:
    """Build the install-time state that gives RunAtLoad two intervals to emit."""

    generated_at = now.astimezone(UTC)
    expected_by = generated_at.timestamp() + FIRST_RUN_DEADLINE_SECONDS
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso_utc(generated_at.isoformat()),
        "alerts": {
            "state": "pending_first_run",
            "reason": "awaiting first health-check run",
            "inputs": [],
            "expected_first_run_by": datetime.fromtimestamp(expected_by, UTC).isoformat().replace("+00:00", "Z"),
        },
    }


def _temporary_badge_state(destination: Path, document: Mapping[str, Any]) -> Path:
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _fsync_parent(path: Path) -> None:
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_badge_state(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically replace the consumed badge document."""

    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_badge_state(destination, document)
    try:
        os.replace(temporary, destination)
        _fsync_parent(destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_badge_state_if_missing(path: Path, document: Mapping[str, Any]) -> bool:
    """Create the initial badge document without replacing measured state."""

    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_badge_state(destination, document)
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return False
        _fsync_parent(destination)
        return True
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    from .paths import get_db_path

    write_badge_state_if_missing(
        badge_state_path(get_db_path()),
        build_pending_first_run_document(datetime.now(UTC)),
    )
