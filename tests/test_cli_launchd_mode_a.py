from __future__ import annotations

import json
import os

from typer.testing import CliRunner

from brainlayer.cli import app
from brainlayer.health_check import HealthCheckResult


def test_health_check_cli_forwards_mode_a_config_fields(monkeypatch, tmp_path):
    captured = {}

    def fake_run_health_check(config):
        captured["config"] = config
        return HealthCheckResult(checked_at="2026-06-20T10:00:00+00:00", ok=True)

    monkeypatch.setattr("brainlayer.health_check.run_health_check", fake_run_health_check)
    result = CliRunner().invoke(
        app,
        [
            "health-check",
            "--db",
            str(tmp_path / "brainlayer.db"),
            "--state-path",
            str(tmp_path / "state.json"),
            "--watch-label",
            "com.test.watch",
            "--drain-label",
            "com.test.drain",
            "--health-check-label",
            "com.test.health",
            "--watch-plist-path",
            str(tmp_path / "watch.plist"),
            "--drain-plist-path",
            str(tmp_path / "drain.plist"),
            "--health-check-plist-path",
            str(tmp_path / "health.plist"),
            "--source-jsonl-glob",
            str(tmp_path / "projects/**/*.jsonl"),
            "--pause-sentinel-path",
            str(tmp_path / "pause.sentinel"),
            "--drain-health-path",
            str(tmp_path / "drain-health.json"),
            "--queue-dir",
            str(tmp_path / "queue"),
            "--offsets-path",
            str(tmp_path / "offsets.json"),
            "--watcher-health-path",
            str(tmp_path / "watcher-health.json"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.watch_label == "com.test.watch"
    assert config.drain_label == "com.test.drain"
    assert config.health_check_label == "com.test.health"
    assert config.watch_plist_path == tmp_path / "watch.plist"
    assert config.drain_plist_path == tmp_path / "drain.plist"
    assert config.health_check_plist_path == tmp_path / "health.plist"
    assert config.source_jsonl_globs == [str(tmp_path / "projects/**/*.jsonl")]
    assert config.pause_sentinel_path == tmp_path / "pause.sentinel"
    assert config.drain_health_path == tmp_path / "drain-health.json"
    assert config.queue_dir == tmp_path / "queue"
    assert config.offsets_path == tmp_path / "offsets.json"
    assert config.watcher_health_path == tmp_path / "watcher-health.json"


def test_pause_and_resume_record_labels_and_call_launchctl(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    monkeypatch.setattr("brainlayer.cli._run_launchctl", lambda args: commands.append(args) or 0)

    pause_path = tmp_path / "pause.sentinel"
    pause_result = CliRunner().invoke(
        app,
        [
            "pause",
            "--pause-sentinel-path",
            str(pause_path),
            "--label",
            "com.test.watch",
            "--label",
            "com.test.drain",
            "--ttl-seconds",
            "60",
        ],
    )

    assert pause_result.exit_code == 0, pause_result.output
    payload = json.loads(pause_path.read_text(encoding="utf-8"))
    assert payload["labels"] == ["com.test.watch", "com.test.drain"]
    assert ["launchctl", "bootout", f"gui/{os.getuid()}/com.test.watch"] in commands
    assert ["launchctl", "bootout", f"gui/{os.getuid()}/com.test.drain"] in commands

    resume_result = CliRunner().invoke(app, ["resume", "--pause-sentinel-path", str(pause_path)])

    assert resume_result.exit_code == 0, resume_result.output
    assert not pause_path.exists()
    assert any(command[:2] == ["launchctl", "bootstrap"] for command in commands)


def test_reconcile_launchd_bootstraps_all_mode_a_labels(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    monkeypatch.setattr("brainlayer.cli._run_launchctl", lambda args: commands.append(args) or 0)

    result = CliRunner().invoke(
        app,
        [
            "reconcile-launchd",
            "--watch-plist-path",
            str(tmp_path / "watch.plist"),
            "--drain-plist-path",
            str(tmp_path / "drain.plist"),
            "--health-check-plist-path",
            str(tmp_path / "health.plist"),
            "--hotlane-plist-path",
            str(tmp_path / "hotlane.plist"),
            "--enrichment-plist-path",
            str(tmp_path / "enrichment.plist"),
        ],
    )

    assert result.exit_code == 0, result.output
    command_text = "\n".join(" ".join(command) for command in commands)
    assert "com.brainlayer.watch" in command_text
    assert "com.brainlayer.drain" in command_text
    assert "com.brainlayer.health-check" in command_text
    assert "com.brainlayer.hotlane-brainbar" in command_text
    assert "com.brainlayer.enrichment" in command_text
