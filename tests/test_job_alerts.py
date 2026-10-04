"""Job alerts retain only safe reasons and notify once per episode."""

from brainlayer.job_alerts import active_alerts, report


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


def test_weekly_maintenance_abort_reports_failure(monkeypatch):
    from brainlayer import job_alerts, maintenance

    notices = []
    monkeypatch.setattr(job_alerts, "report", lambda key, reason: notices.append((key, reason)))

    def abort(mode, *, dry_run):
        raise maintenance.MaintenanceAbort("fixture")

    monkeypatch.setattr(maintenance, "run_maintenance", abort)
    assert maintenance.main(["--full"]) == 75
    assert notices[0][0] == "maintenance-full"
    assert "fixture" in notices[0][1]
    assert "Retry" in notices[0][1]
    assert str(maintenance.MaintenanceConfig().log_path) in notices[0][1]


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


def test_unexpected_maintenance_failure_alert_does_not_expose_exception_value(monkeypatch):
    import pytest

    from brainlayer import job_alerts, maintenance

    notices = []
    monkeypatch.setattr(job_alerts, "report", lambda key, reason: notices.append(reason))

    def fail(*args, **kwargs):
        raise RuntimeError("private-value")

    monkeypatch.setattr(maintenance, "run_maintenance", fail)
    with pytest.raises(RuntimeError):
        maintenance.main(["--light"])
    assert "RuntimeError" in notices[0]
    assert "private-value" not in notices[0]
    assert str(maintenance.MaintenanceConfig().log_path) in notices[0]
