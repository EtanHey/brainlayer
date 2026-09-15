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

# This is the only badge-suppression gate. The interim marker is deliberately
# absent from this module and retires when the badge ships. Data-loss codes are
# excluded structurally, so no configuration file can silence them.
SUPPRESSIBLE_CODES = frozenset(
    {
        "brain_search_canary_failed",
        "drain_no_progress",
        "drain_unloaded",
        "enrichment_unloaded",
        "health_check_unloaded",
        "hotlane_backlog_disabled",
        "hotlane_dead",
        "lock_holder_wedge",
        "missing_embeddings_climbing",
        "missing_embeddings_not_draining",
        "observability_unloaded",
        "pause_sentinel_stale",
        "queue_backed_up",
        "watch_unloaded",
        "watcher_stalled",
    }
)

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


def build_badge_state_document(result: HealthCheckResult) -> dict[str, Any]:
    """Build schema v1 from one completed health-check result."""

    critical = [issue for issue in result.issues if issue.severity == "critical"]
    active = [_issue_payload(issue) for issue in critical if issue.code not in SUPPRESSIBLE_CODES]
    suppressed = [_issue_payload(issue) for issue in critical if issue.code in SUPPRESSIBLE_CODES]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso_utc(result.checked_at),
        "alerts": {
            "state": "measured",
            "reason": "",
            "inputs": [],
            "badge_on": bool(active),
            "active": active,
            "suppressed": suppressed,
        },
    }


def write_badge_state(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically replace the consumed badge document."""

    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
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
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
