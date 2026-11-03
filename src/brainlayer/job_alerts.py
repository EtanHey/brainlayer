"""Durable, deduplicated backup/maintenance failure notices."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping

from brainlayer.paths import get_db_path

LOG = logging.getLogger(__name__)
ALERT_PATH_ENV = "BRAINLAYER_JOB_ALERT_PATH"


def alert_path(*, env: Mapping[str, str] | None = None, db_path: Path | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get(ALERT_PATH_ENV) or (db_path or get_db_path()).with_name("job-alerts.json")).expanduser()


def _notify(message: str) -> None:
    if os.environ.get("BRAINLAYER_TEST_PATH_PROVENANCE") == "pytest":
        return
    subprocess.run(
        [
            "osascript",
            "-e",
            'on run argv\n display notification (item 1 of argv) with title "BrainLayer"\nend run',
            message,
        ],
        timeout=8,
        check=True,
        capture_output=True,
    )


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _alert_records(path: Path | None = None) -> dict[str, dict[str, str | None]]:
    try:
        value = json.loads((path or alert_path()).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}
        records = {}
        for key, item in value.items():
            if isinstance(item, str):  # Read pre-S32 files without losing their live alerts.
                records[str(key)] = {"reason": item, "reported_at": None}
            elif isinstance(item, dict) and isinstance(item.get("reason"), str):
                reported_at = item.get("reported_at")
                records[str(key)] = {
                    "reason": item["reason"],
                    "reported_at": reported_at if isinstance(reported_at, str) else None,
                }
        return records
    except (FileNotFoundError, OSError, ValueError):
        return {}


def active_alerts(path: Path | None = None) -> dict[str, str]:
    """Return current reasons, accepting both legacy strings and timestamped records."""
    return {key: record["reason"] for key, record in _alert_records(path).items()}


def alert_reported_at(path: Path | None = None) -> dict[str, str]:
    """Return alert occurrence times where available; legacy records have no timestamp."""
    return {key: at for key, record in _alert_records(path).items() if (at := record["reported_at"]) is not None}


def report(
    key: str,
    reason: str | None,
    *,
    path: Path | None = None,
    notify: Callable[[str], None] = _notify,
) -> bool:
    """Set or clear one episode. Return true only when the visible state changes.

    Only a new failure posts a notification, and only when no other active key already
    carries the identical message (one incident fanned out across keys posts once).
    Recovery updates the alert file and the health log but never notifies.
    """
    destination = path or alert_path()
    with _locked(destination):
        records = _alert_records(destination)
        before = records.get(key, {}).get("reason")
        if reason is None:
            if before is None:
                return False
            records.pop(key)
            message = f"BrainLayer {key} recovered"
            post = False
        else:
            if before is not None and before == reason:
                records[key]["reported_at"] = datetime.now(UTC).isoformat(timespec="microseconds")
                from brainlayer.drive_credentials import _atomic_write

                _atomic_write(destination, records)
                return False
            if before is not None and key != "drive-consent":
                return False
            post = reason not in (record["reason"] for other, record in records.items() if other != key)
            records[key] = {"reason": reason, "reported_at": datetime.now(UTC).isoformat(timespec="microseconds")}
            message = reason
        from brainlayer.drive_credentials import _atomic_write

        _atomic_write(destination, records)
    LOG.warning("health event job=%s state=%s reason=%s", key, "recovered" if reason is None else "failed", message)
    if not post:
        return True
    try:
        notify(message)
    except (OSError, subprocess.SubprocessError) as exc:
        LOG.warning("macOS notification failed for %s: %s", key, type(exc).__name__)
    return True
