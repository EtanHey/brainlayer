"""Loud liveness probe for the BrainLayer drain heartbeat."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DRAIN_LIVENESS_STALE_SECONDS = 300.0
DEFAULT_ENRICH_DAILY_USD_CAP = 5.0
ENRICH_DAILY_COST_COUNTER_FILENAME = "enrich-daily-cost.json"
STALLED_CODE = "drain_liveness_stalled"
QUOTA_BLOCKED_CODE = "drain_liveness_quota_blocked"
UNKNOWN_BLOCKER_CODE = "drain_liveness_blocker_unknown"


@dataclass(frozen=True)
class DrainLivenessIssue:
    code: str
    severity: str
    message: str
    details: dict[str, Any]


class _EnrichCostCounterUnreadable(RuntimeError):
    """Raised when a present quota counter cannot be trusted."""


def _parse_updated_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _bounded_nonnegative_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return max(0.0, default)
    return max(0.0, parsed)


def _enrich_daily_usd_cap() -> float:
    return _bounded_nonnegative_float(os.environ.get("BRAINLAYER_ENRICH_DAILY_USD_CAP"), DEFAULT_ENRICH_DAILY_USD_CAP)


def _enrich_cost_counter_path(path: Path | None = None) -> Path:
    override_dir = os.environ.get("BRAINLAYER_ENRICH_COST_DIR")
    if override_dir:
        return Path(override_dir).expanduser() / ENRICH_DAILY_COST_COUNTER_FILENAME

    if path is not None:
        return path.expanduser()

    from .paths import get_db_path

    return get_db_path().parent / ENRICH_DAILY_COST_COUNTER_FILENAME


def _read_enrich_cost_record(path: Path, today: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"date": today, "spent_usd": 0.0}
    except OSError as exc:
        raise _EnrichCostCounterUnreadable(f"could not read {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _EnrichCostCounterUnreadable(f"could not parse {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise _EnrichCostCounterUnreadable(f"{path} did not contain a JSON object")
    if data.get("date") != today:
        return {"date": today, "spent_usd": 0.0}
    return data


def _daily_cap_blocker(now: datetime, *, enrich_cost_counter_path: Path | None = None) -> tuple[str | None, str | None]:
    try:
        cap_usd = _enrich_daily_usd_cap()
        today = now.astimezone().date().isoformat()
        record = _read_enrich_cost_record(_enrich_cost_counter_path(enrich_cost_counter_path), today)
        try:
            spent_usd = float(record.get("spent_usd", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            return None, f"invalid spent_usd in quota counter: {exc}"
    except _EnrichCostCounterUnreadable as exc:
        return None, str(exc)
    except Exception:
        return None, None

    if spent_usd >= cap_usd:
        return f"enrichment daily cap reached: spent=${spent_usd:.6f} cap=${cap_usd:.2f}", None
    return None, None


def check_drain_liveness(
    *,
    drain_label: str,
    drain_loaded: bool | None,
    queue_count: int | None,
    enrichment_backlog: int | None,
    drain_health: dict[str, Any],
    now: datetime,
    stale_seconds: float = DEFAULT_DRAIN_LIVENESS_STALE_SECONDS,
    enrich_cost_counter_path: Path | None = None,
    quota_or_throttle_blocker: str | None = None,
) -> DrainLivenessIssue | None:
    """Return a loud issue when a loaded drain has backlog but no fresh heartbeat."""
    queue_backlog = _positive_int(queue_count)
    enrichment_backlog_count = _positive_int(enrichment_backlog)
    backlog = queue_backlog + enrichment_backlog_count
    if drain_loaded is not True or backlog <= 0:
        return None

    heartbeat_at = _parse_updated_at(drain_health.get("updated_at"))
    heartbeat_age = None if heartbeat_at is None else max(0.0, now.timestamp() - heartbeat_at.timestamp())
    if heartbeat_age is not None and heartbeat_age < max(0.0, stale_seconds):
        return None

    blocker = None
    blocker_error = None
    if queue_backlog == 0 and enrichment_backlog_count > 0:
        if quota_or_throttle_blocker:
            blocker = quota_or_throttle_blocker
        else:
            blocker, blocker_error = _daily_cap_blocker(
                now,
                enrich_cost_counter_path=enrich_cost_counter_path,
            )
    details = {
        "backlog_count": backlog,
        "drain_cycles": drain_health.get("drain_cycles"),
        "drain_label": drain_label,
        "drained_total": drain_health.get("drained_total"),
        "enrichment_backlog": enrichment_backlog_count,
        "heartbeat_age_seconds": round(heartbeat_age, 3) if heartbeat_age is not None else None,
        "queue_count": queue_backlog,
        "stale_seconds": stale_seconds,
        "updated_at": drain_health.get("updated_at"),
    }
    if blocker:
        details["blocker"] = blocker
        return DrainLivenessIssue(
            QUOTA_BLOCKED_CODE,
            "warning",
            f"DRAIN_LIVENESS_QUOTA_BLOCKED: {blocker}; loaded drain heartbeat is stale but backlog is blocked",
            details,
        )
    if blocker_error:
        details["blocker_error"] = blocker_error
        return DrainLivenessIssue(
            UNKNOWN_BLOCKER_CODE,
            "warning",
            (
                "DRAIN_LIVENESS_BLOCKER_UNKNOWN: loaded drain heartbeat is stale, "
                f"but quota/throttle blocker state could not be read: {blocker_error}"
            ),
            details,
        )

    stale_description = "missing" if heartbeat_at is None else f"stale for {heartbeat_age:.0f}s"
    return DrainLivenessIssue(
        STALLED_CODE,
        "fatal",
        (
            f"DRAIN_LIVENESS_STALLED: {drain_label} is loaded and backlog={backlog} "
            f"(queue={queue_backlog}, enrichment={enrichment_backlog_count}), "
            f"but drain-health updated_at is {stale_description}; no quota/throttle blocker detected"
        ),
        details,
    )
