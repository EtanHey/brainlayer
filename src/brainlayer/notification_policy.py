"""Condition-scoped policy for suppressing notifications that are noisy by design.

Python callers use :func:`by_design_reason`. Shell callers use
``python -m brainlayer.notification_policy <condition>``; exit 0 means suppress and
stdout is the reason, while exit 1 means alert normally.

Leads can mark an additional exact condition in
``~/.local/share/brainlayer/by-design-notifications.json`` (override with
``BRAINLAYER_BY_DESIGN_REASON_FILE``)::

    {"conditions": {"tier0:state_stale": "planned health-check maintenance"}}

The marker deliberately has no wildcard. Invalid or unreadable markers fail open to
notification rather than hiding a real incident.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from .pause import DEFAULT_PAUSE_SENTINEL_PATH, pause_applies_to_label, pause_sentinel_state

DEFAULT_REASON_FILE = Path("~/.local/share/brainlayer/by-design-notifications.json").expanduser()
DEFAULT_DISABLED_DIR = Path("~/Library/LaunchAgents/.disabled-retention-P0").expanduser()
ENRICHMENT_LABEL = "com.brainlayer.enrichment"
BACKUP_DAILY_PLIST = "com.brainlayer.backup-daily.plist"
MAX_REASON_FILE_BYTES = 64 * 1024
FALSE_VALUES = {"0", "false", "no", "off", "disabled"}


def _path_from_env(env: Mapping[str, str], name: str, default: Path) -> Path:
    return Path(env.get(name, str(default))).expanduser()


def _explicit_reason(condition: str, env: Mapping[str, str]) -> str | None:
    path = _path_from_env(env, "BRAINLAYER_BY_DESIGN_REASON_FILE", DEFAULT_REASON_FILE)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(descriptor, encoding="utf-8") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None
            contents = handle.read(MAX_REASON_FILE_BYTES + 1)
        if len(contents.encode("utf-8")) > MAX_REASON_FILE_BYTES:
            return None
        payload = json.loads(contents)
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("conditions"), dict):
        return None
    reason = payload["conditions"].get(condition)
    return reason.strip() if isinstance(reason, str) and reason.strip() else None


def _enrichment_pause_reason(env: Mapping[str, str], now: datetime) -> str | None:
    for variable in ("BRAINLAYER_LAUNCHD_ENRICHMENT_ENABLED", "BRAINLAYER_ENRICH_ENABLED"):
        value = env.get(variable)
        if value is not None and value.strip().lower() in FALSE_VALUES:
            return f"enrichment is disabled by configuration ({variable})"
    sentinel_path = _path_from_env(env, "BRAINLAYER_PAUSE_SENTINEL_PATH", DEFAULT_PAUSE_SENTINEL_PATH)
    payload, active, _stale = pause_sentinel_state(sentinel_path, now)
    if active and pause_applies_to_label(payload, ENRICHMENT_LABEL):
        paused_at = payload.get("paused_at")
        if isinstance(paused_at, str) and paused_at:
            return f"enrichment is paused since {paused_at}"
        return "enrichment is paused"
    return None


def by_design_reason(
    condition: str,
    *,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    pause_sentinel_path: Path | None = None,
) -> str | None:
    """Return why ``condition`` is intentional, or ``None`` when it must alert."""

    resolved_env = os.environ if env is None else env
    if reason := _explicit_reason(condition, resolved_env):
        return reason
    if condition == "enrichment_backlog":
        if pause_sentinel_path is not None:
            resolved_env = {**resolved_env, "BRAINLAYER_PAUSE_SENTINEL_PATH": str(pause_sentinel_path)}
        return _enrichment_pause_reason(resolved_env, now or datetime.now(UTC))
    if condition == "backup_freshness":
        disabled_dir = _path_from_env(
            resolved_env,
            "BRAINLAYER_BY_DESIGN_DISABLED_DIR",
            DEFAULT_DISABLED_DIR,
        )
        if (disabled_dir / BACKUP_DAILY_PLIST).is_file():
            return "backup-daily is parked on P0"
    return None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m brainlayer.notification_policy CONDITION", file=sys.stderr)
        return 2
    reason = by_design_reason(args[0])
    if reason is None:
        return 1
    print(reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
