"""Retention events remain alert inputs independently of backup history rendering."""

import json

from brainlayer import job_alerts, jsonl_backup


def test_undated_retention_event_still_reaches_job_alert_input(tmp_path, monkeypatch):
    error = "Invented inventory failure"
    log = tmp_path / "jsonl-backup.log"
    alerts = tmp_path / "job-alerts.json"
    monkeypatch.setattr(jsonl_backup, "_configured_backup_timeout_seconds", lambda: None)
    monkeypatch.setattr(jsonl_backup, "_configured_icloud_dir", lambda: None)
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOG_PATH", str(log))
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_STAGING_DIR", str(tmp_path / "staging"))
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_STATE_PATH", str(tmp_path / "state.json"))

    def backup(**kwargs):
        jsonl_backup._retention_warning(kwargs["log_path"], error)
        return {"verified": True, "local_retention_error": error}

    monkeypatch.setattr(jsonl_backup, "run_backup", backup)
    report = job_alerts.report
    monkeypatch.setattr(job_alerts, "report", lambda key, reason: report(key, reason, path=alerts))
    assert jsonl_backup.main() == 0
    assert json.loads(log.read_text()) == {"event": "local_retention_error", "error": error}
    assert job_alerts.active_alerts(alerts) == {"jsonl-retention": error}
