"""Etan's zero-toast contract: BrainLayer has no OS or phone push delivery path."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_health_check_has_no_desktop_notification_delivery_path() -> None:
    source = (REPO_ROOT / "src/brainlayer/health_check.py").read_text(encoding="utf-8")

    assert "_push_notification" not in source
    assert "display notification" not in source
    assert "osascript" not in source


def test_watchdogs_have_no_toast_or_phone_push_delivery_path() -> None:
    for relative_path in (
        "scripts/tier0-watchdog.sh",
        "scripts/launchd/throughput-watchdog.py",
    ):
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "osascript" not in source, relative_path
        assert "localhost:3847/notify" not in source, relative_path
