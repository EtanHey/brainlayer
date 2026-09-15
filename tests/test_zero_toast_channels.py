"""Etan's zero-toast contract: BrainLayer has no OS or phone push delivery path."""

import subprocess
import sys
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


def test_tier0_install_no_longer_requires_notification_policy_environment() -> None:
    source = (REPO_ROOT / "scripts/launchd/install.sh").read_text(encoding="utf-8")
    tier0_installer = source.split("install_tier0_watchdog() {", 1)[1].split("install_throughput_watchdog() {", 1)[0]

    assert "install_env_runner" not in tier0_installer
    assert "verify_config_file" not in tier0_installer
    assert "__BRAINLAYER_ENV_RUN__" not in tier0_installer
    assert "__PYTHON_BIN__" not in tier0_installer


def test_health_check_command_actually_emits_timestamped_incident_log() -> None:
    probe = """
from typer.testing import CliRunner
import sys
import brainlayer.health_check as health_check
from brainlayer.cli import app

def fake_run_health_check(_config):
    health_check._log_health_event(
        "heal:watcher_stalled",
        "com.brainlayer.watch failed repeatedly",
        timestamp="2026-09-15T07:15:00+00:00",
    )
    return health_check.HealthCheckResult(
        checked_at="2026-09-15T07:15:00+00:00",
        ok=True,
    )

health_check.run_health_check = fake_run_health_check
result = CliRunner().invoke(app, ["health-check", "--json"])
sys.stderr.write(result.stderr)
raise SystemExit(result.exit_code)
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "2026-09-15" in completed.stderr
    assert "condition=heal:watcher_stalled" in completed.stderr
    assert "com.brainlayer.watch failed repeatedly" in completed.stderr
