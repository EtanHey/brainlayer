"""Condition-scoped explanations for incidents that are noisy by design.

Python callers use :func:`by_design_reason` to annotate logged incidents. The
module CLI remains as a compatibility surface for older installed callers: exit
0 means a reason exists and stdout carries it; exit 1 means no reason exists.

Leads can mark an additional exact condition in
``~/.local/share/brainlayer/by-design-notifications.json`` (override with
``BRAINLAYER_BY_DESIGN_REASON_FILE``)::

    {"conditions": {"tier0:state_stale": "planned health-check maintenance"}}

The marker deliberately has no wildcard. Invalid or unreadable markers yield no
explanation rather than hiding a real incident.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

DEFAULT_REASON_FILE = Path("~/.local/share/brainlayer/by-design-notifications.json").expanduser()
DEFAULT_DISABLED_DIR = Path("~/Library/LaunchAgents/.disabled-retention-P0").expanduser()
BACKUP_DAILY_PLIST = "com.brainlayer.backup-daily.plist"
MAX_REASON_FILE_BYTES = 64 * 1024


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


def by_design_reason(
    condition: str,
    *,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    pause_sentinel_path: Path | None = None,
) -> str | None:
    """Return why ``condition`` is intentional, or ``None`` when unexplained."""

    resolved_env = os.environ if env is None else env
    if reason := _explicit_reason(condition, resolved_env):
        return reason
    if condition == "enrichment_backlog":
        return "enrichment producers are retired; existing metadata is historical"
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
