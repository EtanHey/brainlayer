"""Job alerts retain only safe reasons and notify once per episode."""

import datetime as dt
import json
import re
from pathlib import Path

import pytest

from brainlayer.job_alerts import active_alerts, report


@pytest.fixture(autouse=True)
def isolated_maintenance_log(tmp_path, monkeypatch):
    path = tmp_path / "maintenance.log"
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_LOG_PATH", str(path))
    return path


def test_failure_episode_and_recovery(tmp_path):
    path = tmp_path / "alerts.json"
    notifications = []
    notify = notifications.append
    assert report("backup-daily", "Database backup failed", path=path, notify=notify)
    assert not report("backup-daily", "Database backup failed again", path=path, notify=notify)
    assert active_alerts(path) == {"backup-daily": "Database backup failed"}
    assert report("backup-daily", None, path=path, notify=notify)
    assert not report("backup-daily", None, path=path, notify=notify)
    assert notifications == ["Database backup failed", "BrainLayer backup-daily recovered"]
    assert active_alerts(path) == {}
    assert path.stat().st_mode & 0o777 == 0o600


def test_consent_warns_once_then_escalates_and_reauthorization_clears(tmp_path):
    path = tmp_path / "alerts.json"
    notices = []
    kwargs = {"path": path, "notify": notices.append}
    warn = "Google Drive access for BrainLayer backups expires tomorrow: click Reconnect in BrainBar"
    expired = "Drive access expired: Reconnect in BrainBar"
    assert report("drive-consent", warn, **kwargs)
    assert not report("drive-consent", warn, **kwargs)
    assert report("drive-consent", expired, **kwargs)
    assert report("drive-consent", None, **kwargs)
    assert notices == [warn, expired, "BrainLayer drive-consent recovered"]


def test_database_backup_exit_reports_and_clears(monkeypatch):
    from brainlayer import backup_daily, job_alerts

    notices = []
    monkeypatch.setattr(job_alerts, "report", lambda key, reason: notices.append((key, reason)))
    monkeypatch.setattr(backup_daily, "_env_flag_enabled", lambda key: False)
    monkeypatch.setattr(backup_daily, "_configured_backup_timeout_seconds", lambda: 60)
    monkeypatch.setattr(backup_daily, "_supervise_backup_process", lambda seconds: 1)
    assert backup_daily.main() == 1
    monkeypatch.setattr(backup_daily, "_supervise_backup_process", lambda seconds: 0)
    assert backup_daily.main() == 0
    assert notices[0][0] == "backup-daily" and "failed" in notices[0][1]
    assert notices[1] == ("backup-daily", None)


@pytest.mark.parametrize(
    "code, reason",
    [
        (1, "fixture"),
        (76, "fixture"),
        (77, "maintenance lock timed out"),
        (75, "failed to resume 1 launchd service(s): watch"),
        (75, "failed to quiesce launchd service watch; it remains loaded"),
        (75, "unknown maintenance failure"),
        (75, "outside quiet window"),
        (75, "prefix queue depth growing: before=1 after=2"),
        (75, "queue depth growing: before=1 after=2; failure"),
        (75, "unexpected writer holds brainlayer db: pid=42 command=FAILED TO RESUME fd=9u"),
        (1, "queue depth growing: before=1 after=2"),
    ],
)
def test_weekly_maintenance_abort_reports_failure(monkeypatch, isolated_maintenance_log, code, reason):
    from brainlayer import job_alerts, maintenance

    notices = []
    monkeypatch.setattr(job_alerts, "report", lambda key, reason: notices.append((key, reason)))

    def abort(mode, *, dry_run):
        raise maintenance.MaintenanceAbort(reason, code=code)

    monkeypatch.setattr(maintenance, "run_maintenance", abort)
    assert maintenance.main(["--full"]) == code
    assert notices[0][0] == "maintenance-full"
    assert reason in notices[0][1]
    assert "Retry" in notices[0][1]
    assert str(maintenance.MaintenanceConfig().log_path) in notices[0][1]
    event = json.loads(isolated_maintenance_log.read_text())
    assert event["status"] == "aborted"
    assert event["mode"] == "full"
    assert event["reason"] == reason
    assert dt.datetime.fromisoformat(event["ts"]).tzinfo is not None


@pytest.mark.parametrize(
    "reason",
    [
        "outside quiet window: now=2026-10-05T12:00:00+00:00 start_hour=4 duration_minutes=30",
        "recent queue write activity: 2 file(s) modified recently",
        "queue depth growing: before=1 after=2",
        "unexpected writer holds brainlayer db: pid=42 command=python3 fd=9u",
    ],
)
@pytest.mark.parametrize("uppercase", [False, True])
def test_maintenance_deferral_logs_without_failure_alert(monkeypatch, isolated_maintenance_log, reason, uppercase):
    reason = reason.upper() if uppercase else reason
    from brainlayer import job_alerts, maintenance

    notices = []
    monkeypatch.setattr(job_alerts, "report", lambda *args: notices.append(args))

    def defer(*args, **kwargs):
        raise maintenance.MaintenanceAbort(reason)

    monkeypatch.setattr(maintenance, "run_maintenance", defer)
    assert maintenance.main(["--light"]) == 75
    assert notices == []
    event = json.loads(isolated_maintenance_log.read_text())
    assert event["status"] == "deferred"
    assert event["mode"] == "light"
    assert event["reason"] == reason


def test_dry_run_abort_does_not_write_log_or_alert(monkeypatch, isolated_maintenance_log):
    from brainlayer import job_alerts, maintenance

    notices = []
    monkeypatch.setattr(job_alerts, "report", lambda *args: notices.append(args))

    def abort(*args, **kwargs):
        raise maintenance.MaintenanceAbort("synthetic failure", code=1)

    monkeypatch.setattr(maintenance, "run_maintenance", abort)
    assert maintenance.main(["--light", "--dry-run"]) == 1
    assert notices == []
    assert not isolated_maintenance_log.exists()


def test_maintenance_alert_episode_clears_after_warning_success(tmp_path, monkeypatch):
    from brainlayer import job_alerts, maintenance

    path = tmp_path / "alerts.json"
    notices = []
    real_report = job_alerts.report
    monkeypatch.setattr(
        job_alerts, "report", lambda key, reason: real_report(key, reason, path=path, notify=notices.append)
    )

    def abort(*args, **kwargs):
        raise maintenance.MaintenanceAbort("search latency pathological", code=1)

    monkeypatch.setattr(maintenance, "run_maintenance", abort)
    assert maintenance.main(["--light"]) == 1
    assert "search latency pathological" in job_alerts.active_alerts(path)["maintenance-light"]
    result = maintenance.MaintenanceResult(mode="light", dry_run=False)
    result.warnings.append("search above target")
    monkeypatch.setattr(maintenance, "run_maintenance", lambda *args, **kwargs: result)
    assert maintenance.main(["--light"]) == 0
    assert job_alerts.active_alerts(path) == {}
    assert notices[-1] == "BrainLayer maintenance-light recovered"


def test_unexpected_maintenance_failure_alert_does_not_expose_exception_value(monkeypatch, isolated_maintenance_log):
    from brainlayer import job_alerts, maintenance

    notices = []
    monkeypatch.setattr(job_alerts, "report", lambda key, reason: notices.append(reason))

    def fail(*args, **kwargs):
        raise RuntimeError("private-value")

    monkeypatch.setattr(maintenance, "run_maintenance", fail)
    with pytest.raises(RuntimeError):
        maintenance.main(["--light"])
    assert "RuntimeError" in notices[0]
    assert "hit an unexpected error (RuntimeError)" in notices[0]
    assert "Retry after resolving the gate" not in notices[0]
    assert "private-value" not in notices[0]
    assert str(maintenance.MaintenanceConfig().log_path) in notices[0]
    event = json.loads(isolated_maintenance_log.read_text())
    assert event["status"] == "failed"
    assert event["mode"] == "light"
    assert event["reason"] == "RuntimeError"
    assert "private-value" not in isolated_maintenance_log.read_text()
    assert dt.datetime.fromisoformat(event["ts"]).tzinfo is not None


def test_maintenance_deliberate_deferrals_match_swift_allowlist():
    from brainlayer import maintenance

    swift = (
        Path(__file__).resolve().parents[1] / "brain-bar/Sources/BrainBar/BrainLayerLaunchdActivity.swift"
    ).read_text()
    block = swift.split("static let deliberateDeferrals = [", 1)[1].split("]", 1)[0]
    patterns = re.findall(r'#"(.*?)"#', block)
    assert len(patterns) == 4
    assert tuple(patterns) == maintenance.DELIBERATE_DEFERRALS
