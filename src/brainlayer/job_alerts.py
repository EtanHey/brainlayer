"""Durable, deduplicated backup/maintenance failure notices."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
from contextlib import contextmanager
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


def active_alerts(path: Path | None = None) -> dict[str, str]:
    try:
        value = json.loads((path or alert_path()).read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}
    except (FileNotFoundError, OSError, ValueError):
        return {}


def report(
    key: str,
    reason: str | None,
    *,
    path: Path | None = None,
    notify: Callable[[str], None] = _notify,
) -> bool:
    """Set or clear one episode. Return true only when the visible state changes."""
    destination = path or alert_path()
    with _locked(destination):
        current = active_alerts(destination)
        before = current.get(key)
        if reason is None:
            if before is None:
                return False
            current.pop(key)
            message = f"BrainLayer {key} recovered"
        else:
            if before is not None and (before == reason or key != "drive-consent"):
                return False
            current[key] = reason
            message = reason
        from brainlayer.drive_credentials import _atomic_write

        _atomic_write(destination, current)
    LOG.warning("health event job=%s state=%s reason=%s", key, "recovered" if reason is None else "failed", message)
    try:
        notify(message)
    except (OSError, subprocess.SubprocessError) as exc:
        LOG.warning("macOS notification failed for %s: %s", key, type(exc).__name__)
    return True
