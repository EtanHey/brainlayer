"""Suite hygiene: a test must never put a notification on a real person's screen.

Measured 2026-09-06: `tests/test_stability_health_check.py` calls `run_health_check()` 25
times and patches `_push_notification` in exactly one of them, so the other 24 fired real
macOS popups built from fixture data -- fake pids, pytest tmp db_paths -- into the
developer's Notification Center, indistinguishable from a production alert. One such card
read `BrainLayer lock-holder wedge holder pid=27542 command=/opt/homebrew/bin/brainlayer
index db_path=/private/var/folders/.../T/pytest-of-...`, which is a fixture string, not a
process that ever existed.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from brainlayer import health_check

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD_ENV = "BRAINLAYER_FORBID_DESKTOP_NOTIFICATION"


def test_the_guard_is_armed_for_this_very_test():
    """conftest arms it for every test, with no marker to lift it."""
    assert os.environ.get(GUARD_ENV) == "1"
    assert health_check.desktop_notifications_forbidden() is True


def test_push_notification_spawns_no_process_while_the_guard_is_armed(monkeypatch):
    spawned: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda args, **_kwargs: spawned.append(args))

    health_check._push_notification("BrainLayer heal action", "kickstart:com.brainlayer.drain")

    assert spawned == [], f"a test reached osascript: {spawned}"


def test_guard_off_would_notify_so_the_test_above_is_not_vacuous(monkeypatch):
    """Without the arming, the same call does reach the notifier.

    Pins that the assertion above measures the guard rather than a code path that never
    fires -- the failure mode this whole file exists to prevent.
    """
    monkeypatch.delenv(GUARD_ENV, raising=False)
    spawned: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda args, **_kwargs: spawned.append(args))

    health_check._push_notification("BrainLayer heal action", "kickstart:com.brainlayer.drain")

    assert len(spawned) == 1
    assert spawned[0][0] == "osascript"


def test_every_osascript_notification_site_in_the_package_checks_the_guard():
    """Discovery, not enumeration: a notification site added later inherits the assertion."""
    sources = [
        *(REPO_ROOT / "src").rglob("*.py"),
        *(REPO_ROOT / "scripts").rglob("*.py"),
    ]
    unguarded = []
    for path in sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "display notification" not in text:
            continue
        if GUARD_ENV not in text and "desktop_notifications_forbidden" not in text:
            unguarded.append(str(path.relative_to(REPO_ROOT)))
    assert not unguarded, f"osascript notification sites with no test guard: {unguarded}"
