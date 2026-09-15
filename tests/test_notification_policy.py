from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brainlayer.notification_policy import by_design_reason, main, may_notify


def test_notification_policy_denies_unknown_conditions_by_default() -> None:
    assert may_notify("watcher_stopped", env={}) == (
        False,
        "condition is not allow-listed for desktop notification",
    )


def test_notification_policy_allows_only_explicit_data_loss_conditions() -> None:
    assert may_notify("jsonl_backup_attempt_failed", env={}) == (True, "backup verification failed")
    assert may_notify("backup_daily_verification_failed", env={}) == (True, "backup verification failed")


def test_explicit_marker_can_suppress_but_not_enable_notification(tmp_path: Path) -> None:
    marker = tmp_path / "by-design-notifications.json"
    marker.write_text(
        '{"conditions":{"jsonl_backup_attempt_failed":"maintenance","watcher_stopped":"wrongly marked enabled"}}',
        encoding="utf-8",
    )
    env = {"BRAINLAYER_BY_DESIGN_REASON_FILE": str(marker)}

    assert may_notify("jsonl_backup_attempt_failed", env=env) == (False, "maintenance")
    assert may_notify("watcher_stopped", env=env) == (False, "wrongly marked enabled")


def test_cli_contract_keeps_zero_for_suppress_and_one_for_alert(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("BRAINLAYER_BY_DESIGN_REASON_FILE", str(tmp_path / "missing.json"))

    assert main(["watcher_stopped"]) == 0
    assert "not allow-listed" in capsys.readouterr().out
    assert main(["jsonl_backup_attempt_failed"]) == 1
    assert capsys.readouterr().out == ""


def test_paused_enrichment_only_suppresses_enrichment_backlog(tmp_path: Path) -> None:
    sentinel = tmp_path / "pause.sentinel"
    sentinel.write_text(
        json.dumps(
            {
                "paused_at": datetime.now(UTC).isoformat(),
                "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                "labels": ["com.brainlayer.enrichment"],
            }
        ),
        encoding="utf-8",
    )
    env = {"BRAINLAYER_PAUSE_SENTINEL_PATH": str(sentinel)}

    assert "enrichment" in (by_design_reason("enrichment_backlog", env=env) or "")
    assert by_design_reason("watcher_stopped", env=env) == ("condition is not allow-listed for desktop notification")


def test_disabled_enrichment_suppresses_only_enrichment_backlog() -> None:
    for variable in ("BRAINLAYER_LAUNCHD_ENRICHMENT_ENABLED", "BRAINLAYER_ENRICH_ENABLED"):
        for value in ("0", "false", "NO", "Off", "disabled"):
            env = {variable: value}
            assert by_design_reason("enrichment_backlog", env=env) == (
                f"enrichment is disabled by configuration ({variable})"
            )
            assert by_design_reason("watcher_stopped", env=env) == (
                "condition is not allow-listed for desktop notification"
            )


def test_parked_backup_suppresses_backup_freshness_only(tmp_path: Path) -> None:
    disabled_dir = tmp_path / ".disabled-retention-P0"
    disabled_dir.mkdir()
    (disabled_dir / "com.brainlayer.backup-daily.plist").write_text("parked\n", encoding="utf-8")
    env = {"BRAINLAYER_BY_DESIGN_DISABLED_DIR": str(disabled_dir)}

    assert by_design_reason("backup_freshness", env=env) == "backup-daily is parked on P0"
    assert by_design_reason("watcher_stopped", env=env) == ("condition is not allow-listed for desktop notification")


def test_explicit_reason_file_is_condition_scoped(tmp_path: Path) -> None:
    marker = tmp_path / "by-design-notifications.json"
    marker.write_text(
        json.dumps({"conditions": {"tier0:state_stale": "planned health-check maintenance"}}),
        encoding="utf-8",
    )
    env = {"BRAINLAYER_BY_DESIGN_REASON_FILE": str(marker)}

    assert by_design_reason("tier0:state_stale", env=env) == "planned health-check maintenance"
    assert by_design_reason("tier0:label_unloaded", env=env) == (
        "condition is not allow-listed for desktop notification"
    )


def test_malformed_explicit_reason_file_still_fails_closed(tmp_path: Path) -> None:
    marker = tmp_path / "by-design-notifications.json"
    marker.write_text("not json", encoding="utf-8")

    assert (
        by_design_reason(
            "tier0:state_stale",
            env={"BRAINLAYER_BY_DESIGN_REASON_FILE": str(marker)},
        )
        == "condition is not allow-listed for desktop notification"
    )


def test_non_utf8_explicit_reason_file_still_fails_closed(tmp_path: Path) -> None:
    marker = tmp_path / "by-design-notifications.json"
    marker.write_bytes(b"\xff\xfe")

    assert (
        by_design_reason(
            "tier0:state_stale",
            env={"BRAINLAYER_BY_DESIGN_REASON_FILE": str(marker)},
        )
        == "condition is not allow-listed for desktop notification"
    )


def test_deeply_nested_reason_file_still_fails_closed(tmp_path: Path) -> None:
    marker = tmp_path / "by-design-notifications.json"
    marker.write_text("[" * 9_999, encoding="utf-8")

    assert (
        by_design_reason(
            "tier0:state_stale",
            env={"BRAINLAYER_BY_DESIGN_REASON_FILE": str(marker)},
        )
        == "condition is not allow-listed for desktop notification"
    )


def test_fifo_reason_marker_fails_closed_without_blocking(tmp_path: Path) -> None:
    marker = tmp_path / "by-design-notifications.json"
    os.mkfifo(marker)

    assert (
        by_design_reason(
            "tier0:state_stale",
            env={"BRAINLAYER_BY_DESIGN_REASON_FILE": str(marker)},
        )
        == "condition is not allow-listed for desktop notification"
    )
