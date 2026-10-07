"""Loud liveness probe for the BrainLayer drain heartbeat."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .alarm import BrainLayerAlarm, build_alarm

DEFAULT_DRAIN_LIVENESS_STALE_SECONDS = 300.0
STALLED_CODE = "drain_liveness_stalled"
PROGRESS_STALLED_CODE = "drain_progress_stalled"
PROGRESS_UNKNOWN_CODE = "drain_progress_unknown"


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


def check_drain_liveness(
    *,
    drain_label: str,
    drain_loaded: bool | None,
    queue_count: int | None,
    drain_health: dict[str, Any],
    now: datetime,
    stale_seconds: float = DEFAULT_DRAIN_LIVENESS_STALE_SECONDS,
) -> BrainLayerAlarm | None:
    """Return a loud issue for stale heartbeat or reported progress failure."""
    queue_backlog = _positive_int(queue_count)
    backlog = queue_backlog
    if drain_loaded is not True:
        return None

    heartbeat_at = _parse_updated_at(drain_health.get("updated_at"))
    heartbeat_age = None if heartbeat_at is None else max(0.0, now.timestamp() - heartbeat_at.timestamp())
    heartbeat_fresh = heartbeat_age is not None and heartbeat_age < max(0.0, stale_seconds)
    progress_state = drain_health.get("state")
    progress_alarm = progress_state == PROGRESS_UNKNOWN_CODE or (
        queue_backlog and progress_state == PROGRESS_STALLED_CODE
    )
    if progress_alarm and (heartbeat_fresh or backlog <= 0):
        reason = drain_health.get("reason") or "drain reported unhealthy progress without a reason"
        condition = "unmeasurable" if progress_state == PROGRESS_UNKNOWN_CODE else "stalled"
        return build_alarm(
            progress_state,
            f"DRAIN_PROGRESS_UNHEALTHY: {drain_label} queue progress {condition}; {reason}",
            {"queue_count": queue_backlog, "drained_total": drain_health.get("drained_total"), "reason": reason},
        )
    if backlog <= 0:
        return None

    if heartbeat_fresh:
        return None

    details = {
        "backlog_count": backlog,
        "drain_cycles": drain_health.get("drain_cycles"),
        "drain_label": drain_label,
        "drained_total": drain_health.get("drained_total"),
        "heartbeat_age_seconds": round(heartbeat_age, 3) if heartbeat_age is not None else None,
        "queue_count": queue_backlog,
        "stale_seconds": stale_seconds,
        "updated_at": drain_health.get("updated_at"),
    }
    stale_description = "missing" if heartbeat_at is None else f"stale for {heartbeat_age:.0f}s"
    return build_alarm(
        STALLED_CODE,
        (
            f"DRAIN_LIVENESS_STALLED: {drain_label} is loaded and backlog={backlog} "
            f"but drain-health updated_at is {stale_description}"
        ),
        details,
    )
