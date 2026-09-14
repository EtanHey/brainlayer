from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brainlayer.notification_policy import by_design_reason


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
    assert by_design_reason("watcher_stopped", env=env) is None


def test_disabled_enrichment_suppresses_only_enrichment_backlog() -> None:
    env = {"BRAINLAYER_LAUNCHD_ENRICHMENT_ENABLED": "0"}

    assert by_design_reason("enrichment_backlog", env=env) == "enrichment is disabled by configuration"
    assert by_design_reason("watcher_stopped", env=env) is None


def test_parked_backup_suppresses_backup_freshness_only(tmp_path: Path) -> None:
    disabled_dir = tmp_path / ".disabled-retention-P0"
    disabled_dir.mkdir()
    (disabled_dir / "com.brainlayer.backup-daily.plist").write_text("parked\n", encoding="utf-8")
    env = {"BRAINLAYER_BY_DESIGN_DISABLED_DIR": str(disabled_dir)}

    assert by_design_reason("backup_freshness", env=env) == "backup-daily is parked on P0"
    assert by_design_reason("watcher_stopped", env=env) is None


def test_explicit_reason_file_is_condition_scoped(tmp_path: Path) -> None:
    marker = tmp_path / "by-design-notifications.json"
    marker.write_text(
        json.dumps({"conditions": {"tier0:state_stale": "planned health-check maintenance"}}),
        encoding="utf-8",
    )
    env = {"BRAINLAYER_BY_DESIGN_REASON_FILE": str(marker)}

    assert by_design_reason("tier0:state_stale", env=env) == "planned health-check maintenance"
    assert by_design_reason("tier0:label_unloaded", env=env) is None


def test_malformed_explicit_reason_file_fails_open_to_alert(tmp_path: Path) -> None:
    marker = tmp_path / "by-design-notifications.json"
    marker.write_text("not json", encoding="utf-8")

    assert (
        by_design_reason(
            "tier0:state_stale",
            env={"BRAINLAYER_BY_DESIGN_REASON_FILE": str(marker)},
        )
        is None
    )


def test_non_utf8_explicit_reason_file_fails_open_to_alert(tmp_path: Path) -> None:
    marker = tmp_path / "by-design-notifications.json"
    marker.write_bytes(b"\xff\xfe")

    assert (
        by_design_reason(
            "tier0:state_stale",
            env={"BRAINLAYER_BY_DESIGN_REASON_FILE": str(marker)},
        )
        is None
    )
