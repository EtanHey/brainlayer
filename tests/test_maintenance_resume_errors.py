"""Real scratch installer stderr reaches maintenance reasons and alert reporting."""

import json

import pytest


@pytest.mark.parametrize("prefix", ["", "x" * 5000], ids=["short", "bounded-tail"])
def test_installer_stderr_tail_survives_resume_failure_and_job_alert(tmp_path, monkeypatch, capsys, prefix):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("BRAINLAYER_SUPERVISOR_SOCKET", raising=False)
    from brainlayer import job_alerts, maintenance

    script = tmp_path / "scripts/launchd/install.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\nprintf '%s\\n' '" + prefix + "ERROR: X' >&2\necho ignored-stdout\nexit 1\n")
    script.chmod(0o700)
    monkeypatch.setattr(maintenance, "_maintenance_keep_down_services", lambda: set())
    monkeypatch.setattr(maintenance, "_service_is_deliberately_paused", lambda service: False)
    failures = maintenance._resume_services(tmp_path, ("index",), {"index": True})
    assert len(failures) == 1
    reason = maintenance._format_resume_failures(failures)
    assert "ERROR: X" in reason
    assert "ignored-stdout" not in reason
    assert len(reason.split("stderr tail: ", 1)[1]) <= 2048
    alerts, logs = [], []

    def abort(*args, **kwargs):
        raise maintenance.MaintenanceAbort(reason)

    monkeypatch.setattr(maintenance, "run_maintenance", abort)
    monkeypatch.setattr(maintenance, "_write_log", lambda path, record: logs.append(record))
    monkeypatch.setattr(job_alerts, "report", lambda key, message: alerts.append((key, message)))
    assert maintenance.main(["--light"]) == 75
    assert json.loads(capsys.readouterr().out)["reason"] == reason
    assert logs[0]["reason"] == reason
    assert alerts[0][0] == "maintenance-light" and "ERROR: X" in alerts[0][1]
