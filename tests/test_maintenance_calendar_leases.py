import asyncio
import datetime as dt
import json
import os
import signal
import subprocess
import sys

import pytest

from brainlayer.maintenance_calendar import Step, run_calendar
from tests.calendar_watch_readiness import wait_for_calendar_watch
from tests.test_maintenance_control import control, native  # noqa: F401
from tests.test_service_supervisor import eventually, exists


@pytest.fixture
def runner(request, monkeypatch):
    path, _, query, _ = request.getfixturevalue("control")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(path / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_TEST_PATH_PROVENANCE", "pytest")
    return path, query


@pytest.mark.parametrize("step_name", ["maintenance-nightly", "index"])
def test_nested_lease_verified_and_timeout_reaps_descendants_before_next_step(runner, step_name, monkeypatch):
    path, query = runner
    (path / "fixture.env").write_text("")
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(path / "fixture.env"))
    marker = path / "calendar-descendant"
    leaf = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    source = f"import subprocess,sys,time; from pathlib import Path; from brainlayer.maintenance_control import MaintenanceHold\nwith MaintenanceHold():\n c=subprocess.Popen((sys.executable,'-c',{leaf!r})); Path({str(marker)!r}).write_text(str(c.pid)); time.sleep(60)"
    if step_name == "index":
        # Online index has no maintenance pause; do not spend its synthetic cap
        # waiting for resident writers to acknowledge a nested maintenance hold.
        source = source.replace("with MaintenanceHold():", "if True:")
    next_step = f"import os; from pathlib import Path; pid=int(Path({str(marker)!r}).read_text());\ntry: os.kill(pid,0)\nexcept ProcessLookupError: pass\nelse: raise SystemExit(99)"
    steps = [
        Step(step_name, (sys.executable, "-c", source), 2 if step_name == "index" else 0.5),
        Step("p0-counter", (sys.executable, "-c", next_step), 5),
    ]
    assert (
        asyncio.run(
            run_calendar(
                steps,
                path / "calendar",
                path / "logs",
                now_fn=lambda: dt.datetime(2026, 10, 12, 3 if step_name == "index" else 4, 30),
            )
        )
        == 1
    )
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)
    records = [json.loads(l) for l in (path / "calendar" / "runs.jsonl").read_text().splitlines()]
    assert [r["returncode"] for r in records if r["event"] == "finished"] == [124, 0]
    eventually(lambda: bool(query()["components"]["watch"]["pid"]))


@pytest.mark.parametrize("death", ["freeze", "term", "guard", "both"])
@pytest.mark.parametrize("chain", [False, True])
def test_calendar_reaps_writer_before_resident_revival(request, death, chain, startup_delay=0):
    path, _, query, _ = request.getfixturevalue("control")
    source = """
import asyncio,subprocess,sys,datetime as dt
from pathlib import Path
from brainlayer import maintenance_control
from brainlayer.supervisor_adapters import SupervisorCommands
from brainlayer.maintenance_calendar import Step,run_calendar
p=Path(sys.argv[1]); maintenance_control.control_client=lambda:SupervisorCommands(p/'control.sock',lambda a:subprocess.CompletedProcess(a,0,'',''),pause_path=p/'pause')
writer="import os,time; from pathlib import Path; time.sleep(%s); Path(%r).write_text(str(os.getpid())); time.sleep(60)" % (float(sys.argv[3]),str(p/'step'))
if sys.argv[2]=='chain': writer="import sys; from brainlayer.backup_daily import _supervise_backup_process; raise SystemExit(_supervise_backup_process(60,command=[sys.executable,'-c',%r]))" % writer
raise SystemExit(asyncio.run(run_calendar([Step('maintenance-nightly',(sys.executable,'-c',writer),60)],p/'calendar',p/'logs',guard_timeout=.4,install_signals=True,now_fn=lambda:dt.datetime(2026,10,12,4,30))))
"""
    errors = path / "calendar.stderr"
    with errors.open("ab") as output:
        parent = subprocess.Popen(
            (sys.executable, "-c", source, str(path), "chain" if chain else "plain", str(startup_delay)), stderr=output
        )
    step, group = None, None
    try:

        def ready():
            assert parent.poll() is None
            return (path / "step").exists()

        try:
            # Includes pause acknowledgement and two interpreter startups in the chain.
            eventually(ready, timeout=10)
        except AssertionError:
            events = path / "calendar" / "runs.jsonl"
            pytest.fail(
                f"calendar writer startup failed: returncode={parent.poll()}, stderr={errors.read_text()}, "
                f"events={events.read_text() if events.exists() else 'missing'}, "
                f"step_errors={[p.read_text() for p in sorted((path / 'logs').glob('*.err.log'))]}"
            )
        step = int((path / "step").read_text())
        group = os.getpgid(step)
        assert all(v["pid"] is None for v in query()["components"].values())
        if death in {"guard", "both"}:
            events = [json.loads(l) for l in (path / "calendar" / "runs.jsonl").read_text().splitlines()]
            os.kill(next(r["guard"] for r in events if r["event"] == "runner"), signal.SIGKILL)
        if death != "guard":
            os.kill(parent.pid, {"freeze": signal.SIGSTOP, "term": signal.SIGTERM, "both": signal.SIGKILL}[death])
        assert parent.wait(timeout=3) == (-signal.SIGKILL if death in {"freeze", "both"} else 1)
        if death == "both":
            os.kill(step, 0)
            # Cleanup still reaps the writer, but a killed run must be reported as failed.
            assert asyncio.run(run_calendar([], path / "calendar", path / "logs")) == 1
            recovered = [json.loads(line) for line in (path / "calendar" / "runs.jsonl").read_text().splitlines()]
            assert len([row for row in recovered if row["event"] == "abandoned"]) == 1
            assert recovered[-1]["failed"] == 1

        def dead():
            try:
                os.kill(step, 0)
            except ProcessLookupError:
                return True
            return False

        eventually(dead)
        wait_for_calendar_watch(query, path)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if group and exists(group):
            os.killpg(group, signal.SIGKILL)


def test_calendar_waits_for_slow_chained_writer(request):
    test_calendar_reaps_writer_before_resident_revival(request, "freeze", True, startup_delay=3.2)


def online_writer_probe(path):
    return f"""
import sqlite3,time
from brainlayer.maintenance_control import control_client
def count():
    try:
        with sqlite3.connect({str(path / "writer.db")!r}) as db:
            return db.execute("select count(*) from samples where who='resident'").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
before,end=count(),time.monotonic()+2
while count() <= before and time.monotonic()<end: time.sleep(.02)
assert count()>before
assert all(v['pid'] and not v['paused'] for v in control_client().request().values() if v['configured'])
"""


def test_live_ingestion_keeps_running_during_online_steps(runner):
    path, _ = runner
    steps = [
        Step(n, (sys.executable, "-c", online_writer_probe(path)), 5)
        for n in ("index", "backup-daily", "p0-counter", "wal-checkpoint")
    ]
    assert asyncio.run(run_calendar(steps, path / "calendar", path / "logs")) == 0


def test_only_maintenance_and_fts_hold_writers_and_wait_stays_online(runner):
    path, query = runner
    clock = [dt.datetime(2026, 10, 12, 3, 59, 30)]

    async def advance(seconds):
        assert all(v["pid"] and not v["paused"] for v in query()["components"].values() if v["configured"])
        clock[0] += dt.timedelta(seconds=30)
        await asyncio.sleep(0)

    online = online_writer_probe(path)
    held = "from brainlayer.maintenance_control import MaintenanceHold,control_client\nwith MaintenanceHold(): assert all(v['pid'] is None for v in control_client().request().values())"
    names = (
        "index",
        "maintenance-nightly",
        "p0-counter",
        "maintenance-weekly",
        "wal-checkpoint",
        "repair-fts",
        "jsonl-backup",
    )
    steps = [
        Step(n, (sys.executable, "-c", held if n.startswith("maintenance-") or n == "repair-fts" else online), 5)
        for n in names
    ]
    assert (
        asyncio.run(run_calendar(steps, path / "calendar", path / "logs", now_fn=lambda: clock[0], sleep_fn=advance))
        == 0
    )


def test_slow_ack_keeps_guard_pulsing_without_starting_writer_early(runner):
    path, query = runner
    source = """
import asyncio,datetime as dt,subprocess,sys,time
from pathlib import Path
from brainlayer import maintenance_control
from brainlayer.supervisor_adapters import SupervisorCommands
from brainlayer.maintenance_calendar import Step,run_calendar
p=Path(sys.argv[1]); c=SupervisorCommands(p/'control.sock',lambda a:subprocess.CompletedProcess(a,0,'',''),pause_path=p/'pause')
request=c.request
def slow(action='status',*args,**kw):
    if action=='pause': time.sleep(.6)
    return request(action,*args,**kw)
c.request=slow; maintenance_control.control_client=lambda:c
raise SystemExit(asyncio.run(run_calendar([Step('maintenance-nightly',(sys.executable,'-c','pass'),5)],p/'calendar',p/'logs',guard_timeout=.4,now_fn=lambda:dt.datetime(2026,10,12,4,30))))
"""
    assert subprocess.run((sys.executable, "-c", source, str(path)), timeout=5).returncode == 0
    eventually(lambda: bool(query()["components"]["watch"]["pid"]))
