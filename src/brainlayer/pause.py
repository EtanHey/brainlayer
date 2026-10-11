"""Shared pause-sentinel parsing for launchd healing and queue drain."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_PAUSE_SENTINEL_PATH = Path("~/.local/share/brainlayer/pause.sentinel").expanduser()


def pause_sentinel_state(
    path: Path,
    now: datetime | None = None,
) -> tuple[dict[str, Any], bool, bool]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}, False, False
    if not isinstance(payload, dict) or not payload:
        return {}, False, False

    expires_at = _parse_iso_datetime(payload.get("expires_at"))
    current = now or datetime.now(UTC)
    stale = expires_at is not None and current > expires_at
    return payload, not stale, stale


def pause_applies_to_label(payload: dict[str, Any], label: str) -> bool:
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return False
    return label in {str(item) for item in labels}


def queue_contains_only_enrichment(queue_dir: Path, expected_count: int) -> bool:
    """Fail closed on races or malformed files; names alone cannot prove pause safety."""
    if expected_count <= 0:
        return False
    try:
        paths = [path for path in queue_dir.expanduser().glob("*.jsonl") if path.is_file()]
        if len(paths) != expected_count:
            return False
        for path in paths:
            saw_event = False
            with path.open(encoding="utf-8") as queue_file:
                for line in queue_file:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    saw_event = True
                    if not isinstance(event, dict) or event.get("kind") != "enrichment_update":
                        return False
            if not saw_event:
                return False
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return True


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
