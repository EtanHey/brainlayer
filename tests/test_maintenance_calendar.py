import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from brainlayer import job_alerts
from brainlayer.maintenance_calendar import Step, calendar_steps, run_calendar
from tests.test_maintenance_control import control, native  # noqa: F401
from tests.test_service_supervisor import eventually, exists


@pytest.mark.parametrize("weekday", [0, 6])
def test_legacy_commands_order_and_sunday_tail(weekday, tmp_path):
    steps = calendar_steps("/keg/python", "/keg/brainlayer", tmp_path, weekday)
    names = [s.name for s in steps]
    assert names[:6] == ["index", "backup-daily", "maintenance-nightly", "jsonl-backup", "p0-counter", "wal-checkpoint"]
    assert names[6:] == (["maintenance-weekly", "decay", "repair-fts"] if weekday == 6 else [])
    assert all(s.argv[0] == str(tmp_path / "brainlayer-env-run.sh") for s in steps)
    assert steps[2].argv[-4:] == ("/keg/python", "-m", "brainlayer.maintenance", "--light")
    assert steps[5].argv[1:] == ("/keg/brainlayer", "wal-checkpoint", "--mode", "TRUNCATE", "--retry-busy")
    if weekday == 6:
        assert steps[6].argv[-1] == "--full"


@pytest.fixture
def runner(request, monkeypatch):
    path, _, query, _ = request.getfixturevalue("control")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(path / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_TEST_PATH_PROVENANCE", "pytest")
    return path, query


def test_failure_and_deferral_recorded_continue_without_clearing_deferred_alert(runner):
    path, query = runner
    output = path / "order"
    steps = [
        Step(
            name,
            (
                sys.executable,
                "-c",
                f"from pathlib import Path; p=Path({str(output)!r}); p.open('a').write({name!r}+'\\n'); raise SystemExit({code})",
            ),
            5,
        )
        for name, code in (("index", 2), ("backup-daily", 75), ("p0-counter", 0))
    ]
    job_alerts.report("backup-daily", "prior failure")
    assert asyncio.run(run_calendar(steps, path / "calendar", path / "logs")) == 1
    assert output.read_text().splitlines() == [s.name for s in steps]
    records = [json.loads(l) for l in (path / "calendar" / "runs.jsonl").read_text().splitlines()]
    results = [r for r in records if r["event"] == "finished"]
    assert [r["returncode"] for r in results] == [2, 75, 0]
    assert [r["status"] for r in results] == ["failed", "deferred", "ok"]
    assert "index" in job_alerts.active_alerts() and job_alerts.active_alerts()["backup-daily"] == "prior failure"
    eventually(lambda: bool(query()["components"]["watch"]["pid"]))


def test_nested_lease_verified_and_timeout_reaps_descendants_before_next_step(runner):
    path, query = runner
    marker = path / "descendant"
    leaf = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    source = f"import subprocess,sys,time; from pathlib import Path; from brainlayer.maintenance_control import MaintenanceHold\nwith MaintenanceHold():\n c=subprocess.Popen((sys.executable,'-c',{leaf!r})); Path({str(marker)!r}).write_text(str(c.pid)); time.sleep(60)"
    next_step = f"import os; from pathlib import Path; pid=int(Path({str(marker)!r}).read_text());\ntry: os.kill(pid,0)\nexcept ProcessLookupError: pass\nelse: raise SystemExit(99)"
    steps = [
        Step("index", (sys.executable, "-c", source), 0.5),
        Step("p0-counter", (sys.executable, "-c", next_step), 5),
    ]
    assert asyncio.run(run_calendar(steps, path / "calendar", path / "logs")) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)
    records = [json.loads(l) for l in (path / "calendar" / "runs.jsonl").read_text().splitlines()]
    assert [r["returncode"] for r in records if r["event"] == "finished"] == [124, 0]
    eventually(lambda: bool(query()["components"]["watch"]["pid"]))


@pytest.mark.parametrize("death", ["freeze", "term", "guard", "both"])
def test_calendar_reaps_writer_before_resident_revival(request, death):
    path, _, query, _ = request.getfixturevalue("control")
    source = """
import asyncio,subprocess,sys
from pathlib import Path
from brainlayer import maintenance_control
from brainlayer.supervisor_adapters import SupervisorCommands
from brainlayer.maintenance_calendar import Step,run_calendar
p=Path(sys.argv[1]); maintenance_control.control_client=lambda:SupervisorCommands(p/'control.sock',lambda a:subprocess.CompletedProcess(a,0,'',''),pause_path=p/'pause')
writer="import os,time; from pathlib import Path; Path(%r).write_text(str(os.getpid())); time.sleep(60)" % str(p/'step')
raise SystemExit(asyncio.run(run_calendar([Step('index',(sys.executable,'-c',writer),60)],p/'calendar',p/'logs',guard_timeout=.4,install_signals=True)))
"""
    parent = subprocess.Popen((sys.executable, "-c", source, str(path)))
    step = None
    try:
        eventually(lambda: (path / "step").exists())
        step = int((path / "step").read_text())
        assert all(v["pid"] is None for v in query()["components"].values())
        if death in {"guard", "both"}:
            events = [json.loads(l) for l in (path / "calendar" / "runs.jsonl").read_text().splitlines()]
            os.kill(next(r["guard"] for r in events if r["event"] == "runner"), signal.SIGKILL)
        if death != "guard":
            os.kill(parent.pid, {"freeze": signal.SIGSTOP, "term": signal.SIGTERM, "both": signal.SIGKILL}[death])
        assert parent.wait(timeout=3) == (-signal.SIGKILL if death in {"freeze", "both"} else 1)
        if death == "both":
            assert exists(step)
            assert asyncio.run(run_calendar([], path / "calendar", path / "logs")) == 0
        eventually(lambda: not exists(step))
        eventually(lambda: bool(query()["components"]["watch"]["pid"]))
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if step and exists(step):
            os.killpg(step, signal.SIGKILL)


def test_full_quiet_window_at_0315_remains_deferred(tmp_path):
    import datetime as dt

    from brainlayer.maintenance import MaintenanceAbort, MaintenanceConfig, _check_quiet_window

    config = MaintenanceConfig(db_path=tmp_path / "copy.db", now_fn=lambda: dt.datetime(2026, 10, 11, 3, 15))
    with pytest.raises(MaintenanceAbort, match="outside quiet window"):
        _check_quiet_window(config)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_steps_refused_before_quiescence(timeout, runner):
    with pytest.raises(ValueError):
        Step("index", (sys.executable,), timeout)
    assert runner[1]()["components"]["watch"]["pid"]


@pytest.mark.parametrize("managed", [True, False])
def test_weekly_backup_remains_in_calendar_group_with_inherited_lease(tmp_path, monkeypatch, managed):
    from brainlayer import maintenance

    fd = os.open(tmp_path / "lease", os.O_CREAT | os.O_RDWR, 0o600)
    real_spawn = subprocess.Popen
    source = "import os,json; fd=os.environ.get('BRAINLAYER_MAINTENANCE_FD'); print(json.dumps(dict(verified=True,group=os.getpgrp(),fd=bool(fd) and bool(os.fstat(int(fd))),direct=os.environ.get('BRAINLAYER_BACKUP_SUPERVISED_CHILD')=='1')))"

    def spawn(args, **kwargs):
        return real_spawn((sys.executable, "-c", source), **kwargs)

    monkeypatch.setattr(maintenance.subprocess, "Popen", spawn)
    if managed:
        monkeypatch.setenv("BRAINLAYER_MAINTENANCE_FD", str(fd))
    else:
        monkeypatch.delenv("BRAINLAYER_MAINTENANCE_FD", raising=False)
    try:
        config = maintenance.MaintenanceConfig(
            db_path=tmp_path / "copy.db", backup_log_path=tmp_path / "backup.log", backup_staging_dir=tmp_path
        )
        result = maintenance._run_bounded_weekly_backup(config, 5)
        assert (result["group"] == os.getpgrp()) == managed
        assert result["fd"] == result["direct"] == managed
    finally:
        os.close(fd)


@pytest.mark.parametrize("gate", ["disabled", "sentinel", "config-off"])
def test_standing_calendar_hold_does_not_run_or_clear_alerts(runner, monkeypatch, gate):
    from brainlayer import maintenance_control

    path, _ = runner
    label = "com.brainlayer.p0-counter"
    if gate == "disabled":
        maintenance_control.control_client().fallback = lambda a: subprocess.CompletedProcess(
            a, 0, f'"{label}" => disabled', ""
        )
    elif gate == "sentinel":
        (path / "pause").write_text(json.dumps({"labels": [label]}))
    env_file = path / "env"
    env_file.write_text("BRAINLAYER_LAUNCHD_P0_COUNTER_ENABLED=0\n" if gate == "config-off" else "")
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(env_file))
    monkeypatch.setenv("HOME", str(path))
    marker = path / "unexpected"
    command = (
        str(Path.cwd() / "scripts/launchd/brainlayer-env-run.sh"),
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).touch()",
    )
    job_alerts.report("p0-counter", "prior failure")
    assert asyncio.run(run_calendar([Step("p0-counter", command, 1)], path / "calendar", path / "logs")) == 1
    assert not marker.exists() and job_alerts.active_alerts()["p0-counter"] == "prior failure"
    events = [json.loads(l) for l in (path / "calendar" / "runs.jsonl").read_text().splitlines()]
    assert [r["returncode"] for r in events if r["event"] == "finished"] == [75]
