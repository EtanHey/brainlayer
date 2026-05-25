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
    event.update(fields)
    logger.info(json.dumps(event, sort_keys=True, separators=(",", ":")))
