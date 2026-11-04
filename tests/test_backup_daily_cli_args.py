"""#1438: ``python -m brainlayer.backup_daily --help`` must not start a backup.

Every production caller (``scripts/launchd/backup-daily.sh`` under env-run, the
in-module supervisor's child, and maintenance's weekly snapshot) invokes the
module with no arguments, so a bare invocation and the explicit ``run`` verb
still run; anything else prints usage and exits before DB, lock, snapshot or
upload work.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys

import pytest

from brainlayer import backup_daily, job_alerts


@pytest.fixture
def no_backup_work(monkeypatch):
    calls: list[str] = []

    def refuse(name):
        def _refuse(*args, **kwargs):  # noqa: ARG001
            calls.append(name)
            raise AssertionError(f"{name} reached on a usage-only invocation")

        return _refuse

    monkeypatch.setattr(backup_daily, "_supervise_backup_process", refuse("supervise"))
    monkeypatch.setattr(backup_daily, "_run_backup_process", refuse("run_backup"))
    monkeypatch.setattr(job_alerts, "report", refuse("report"))
    return calls


@pytest.mark.parametrize(("argv", "code"), [(["--help"], 0), (["-h"], 0), (["--bogus"], 2), (["backup"], 2)])
def test_usage_invocations_exit_before_any_backup_work(no_backup_work, capsys, argv, code):
    with pytest.raises(SystemExit) as excinfo:
        backup_daily.main(argv)

    assert excinfo.value.code == code
    out, err = capsys.readouterr()
    assert "usage:" in (out if code == 0 else err)
    assert no_backup_work == []


@pytest.mark.parametrize("argv", [None, [], ["run"]])
def test_bare_and_run_invocations_still_run(monkeypatch, argv):
    ran: list[int] = []
    monkeypatch.setattr(sys, "argv", ["pytest", "--help"])  # library callers never read sys.argv
    monkeypatch.setattr(job_alerts, "report", lambda key, reason: None)
    monkeypatch.setattr(backup_daily, "_env_flag_enabled", lambda key: False)
    monkeypatch.setattr(backup_daily, "_configured_backup_timeout_seconds", lambda: 60)
    monkeypatch.setattr(backup_daily, "_supervise_backup_process", lambda seconds: ran.append(seconds) or 0)

    assert (backup_daily.main() if argv is None else backup_daily.main(argv)) == 0
    assert ran == [60]


def test_module_help_process_touches_no_backup_state(tmp_path):
    """Real ``python -m`` process boundary against a disposable scratch DB."""
    db = tmp_path / "data" / "brainlayer.db"
    db.parent.mkdir()
    sqlite3.connect(db).close()
    env = os.environ.copy()
    env.pop(backup_daily.BACKUP_SUPERVISED_CHILD_ENV, None)
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "BRAINLAYER_DB": str(db),
            "BRAINLAYER_BACKUP_STAGING_DIR": str(tmp_path / "staging"),
            "BRAINLAYER_BACKUP_LOG_PATH": str(tmp_path / "backup.log"),
            "BRAINLAYER_BACKUP_LOG_PROVENANCE": "pytest",
            "BRAINLAYER_JOB_ALERT_PATH": str(tmp_path / "alerts.json"),
            "BRAINLAYER_DRIVE_TOKEN_PATH": str(tmp_path / "missing-token.json"),
            "BRAINLAYER_DRIVE_CLIENT_PATH": str(tmp_path / "missing-client.json"),
            "BRAINLAYER_FORBID_BRAINBAR_SOCKET": "1",
            "BRAINLAYER_BACKUP_TIMEOUT_SECONDS": "5",
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "brainlayer.backup_daily", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data"]
    assert sorted(p.name for p in db.parent.iterdir()) == ["brainlayer.db"]
