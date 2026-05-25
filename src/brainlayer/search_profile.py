"""Opt-in search latency profile logging."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)
_RESERVED_FIELDS = {"ts", "scope", "step", "query_id", "dur_ms"}


def enabled() -> bool:
    return os.environ.get("BRAINLAYER_SEARCH_PROFILE") == "1"


def new_query_id() -> str:
    return f"q-{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.perf_counter()


def dur_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def emit(scope: str, step: str, query_id: str | None = None, dur_ms: float | None = None, **fields: Any) -> None:
    if not enabled():
        return

    event: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "scope": scope,
        "step": step,
    }
    if query_id:
        event["query_id"] = query_id
    if dur_ms is not None:
        event["dur_ms"] = dur_ms
    for key, value in fields.items():
        if key not in _RESERVED_FIELDS:
            event[key] = value
    try:
        payload = json.dumps(event, sort_keys=True, separators=(",", ":"))
    except TypeError:
        safe_event = {key: value if _is_json_safe(value) else repr(value) for key, value in event.items()}
        payload = json.dumps(safe_event, sort_keys=True, separators=(",", ":"))
    logger.info(payload)


def _is_json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True
