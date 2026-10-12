"""Private-process writer exclusion, acknowledgement, recovery and hold ownership."""

import json
import os
import signal
import subprocess
import sys

import pytest

import tests.test_service_supervisor as native_tests
from brainlayer import maintenance_control
from brainlayer.maintenance_control import MaintenanceHold
from brainlayer.supervisor_adapters import SupervisorCommands
from tests.test_service_supervisor import native  # noqa: F401


@pytest.fixture
def control(request, monkeypatch):
    path, start, query = request.getfixturevalue("native")
    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_SOCKET", str(path / "control.sock"))
    writer = (
        "import sqlite3,time\nc=sqlite3.connect(%r)\nc.execute('create table if not exists samples(who)')\nwhile True:\n c.execute(\"insert into samples values('resident')\"); c.commit(); time.sleep(.02)"
        % str(path / "writer.db")
    )
    code = native_tests.BOOT.replace(
        "s=Supervisor(", f"watch=watch.rsplit('time.sleep(60)',1)[0] + {'exec(' + repr(writer) + ')'!r}\ns=Supervisor("
    )
    if getattr(request, "param", "") == "old":
        code = code.replace(
            "s=Supervisor(",
            'status=Component.status; Component.status=lambda c:{k:v for k,v in status(c).items() if k!="pause_owner"}\ns=Supervisor(',
        )
    monkeypatch.setattr(native_tests, "BOOT", code)
    client = SupervisorCommands(
        path / "control.sock", lambda a: subprocess.CompletedProcess(a, 0, "", ""), pause_path=path / "pause"
    )
    monkeypatch.setattr(maintenance_control, "control_client", lambda: client)
    process, initial, _ = start()
    yield path, process, query, initial
    if process.poll() is None:
        process.terminate()
        assert process.wait(timeout=2) == 0


def test_acknowledged_pause_preserves_off_and_prior_runtime_pause(control):
    path, _, query, initial = control
    query("pause", "drain")
    with MaintenanceHold() as hold:
        states = query()["components"]
        assert all(value["pid"] is None for value in states.values())
        assert all(not native_tests.exists(value["pid"]) for value in initial.values() if value["pid"])
        assert hold.fd is not None
    native_tests.eventually(lambda: bool(query()["components"]["watch"]["pid"]))
    states = query()["components"]
    assert states["watch"]["pid"] and states["drain"]["paused"]
    assert not states["off"]["configured"] and states["off"]["pid"] is None
    assert not (path / "maintenance.json").exists()


def test_new_manual_pause_is_not_undone_by_maintenance(control):
    _, _, query, _ = control
    with MaintenanceHold():
        with pytest.raises(RuntimeError, match="another owner"):
            maintenance_control.control_client().request("resume", "watch")
        query("pause", "watch")  # A later operator pause takes ownership.
    assert query()["components"]["watch"]["paused"]


def test_supervisor_restart_cannot_reintroduce_writers_during_lease(control):
    path, parent, query, initial = control
    replacement = None
    try:
        with MaintenanceHold():
            parent.kill()
            parent.wait(timeout=2)
            native_tests.eventually(
                lambda: not any(native_tests.exists(v["pid"]) for v in initial.values() if v["pid"])
            )
            replacement = subprocess.Popen((sys.executable, "-c", native_tests.BOOT, str(path)))

            def ready():
                try:
                    return query()["components"]["watch"]["pause_owner"] == ""
                except (OSError, KeyError):
                    return False

            native_tests.eventually(ready)
            assert all(v["pid"] is None for v in query()["components"].values())
        native_tests.eventually(lambda: bool(query()["components"]["watch"]["pid"]))
    finally:
        if replacement and replacement.poll() is None:
            replacement.terminate()
            assert replacement.wait(timeout=2) == 0


def test_owner_death_keeps_child_writer_lease_and_recovers_after_exit(control):
    path, _, query, _ = control
    source = r"""
import os,subprocess,sys,time
from pathlib import Path
from brainlayer import maintenance_control
from brainlayer.maintenance_control import MaintenanceHold
from brainlayer.supervisor_adapters import SupervisorCommands
p=Path(sys.argv[1])
with MaintenanceHold() as h:
 writer="import sqlite3,time\nc=sqlite3.connect(%r)\nwhile True:\n c.execute(\"insert into samples values('step')\"); c.commit(); time.sleep(.02)" % str(p/'writer.db')
 child=subprocess.Popen((sys.executable,'-c',writer),pass_fds=(h.fd,),start_new_session=True)
 (p/'holder-child').write_text(str(child.pid))
 time.sleep(60)
"""
    holder = subprocess.Popen((sys.executable, "-c", source, str(path)))
    child = None
    try:
        native_tests.eventually(lambda: (path / "holder-child").exists())
        child = int((path / "holder-child").read_text())
        import sqlite3

        def counts():
            with sqlite3.connect(path / "writer.db") as db:
                return dict(db.execute("select who,count(*) from samples group by who"))

        native_tests.eventually(lambda: counts().get("step", 0) > 1)
        before = counts()
        holder.kill()
        holder.wait(timeout=2)
        native_tests.eventually(lambda: counts().get("step", 0) > before["step"] + 3)
        assert counts()["resident"] == before["resident"]
        with pytest.raises(BlockingIOError), MaintenanceHold():
            pass
        assert all(v["pid"] is None for v in query()["components"].values())
        os.killpg(child, signal.SIGKILL)
        native_tests.eventually(lambda: not native_tests.exists(child))
        with MaintenanceHold():
            assert all(v["pid"] is None for v in query()["components"].values())
        native_tests.eventually(lambda: bool(query()["components"]["watch"]["pid"]))
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=2)
        if child and native_tests.exists(child):
            os.killpg(child, signal.SIGKILL)


def test_nested_child_hold_requires_inherited_fd_and_matching_token(control, monkeypatch):
    with MaintenanceHold() as outer:
        blocked = MaintenanceHold()
        with pytest.raises(BlockingIOError):
            blocked.__enter__()
        blocked.close()
        assert control[2]()["components"]["watch"]["pause_owner"] == outer.token
        monkeypatch.setenv("BRAINLAYER_MAINTENANCE_FD", str(outer.fd))
        monkeypatch.setenv("BRAINLAYER_MAINTENANCE_TOKEN", outer.token)
        with MaintenanceHold() as inner:
            assert inner.fd == outer.fd
        assert all(v["pid"] is None for v in control[2]()["components"].values())
        monkeypatch.setenv("BRAINLAYER_MAINTENANCE_TOKEN", "foreign")
        with pytest.raises(RuntimeError), MaintenanceHold():
            pass


def test_corrupt_recovery_journal_refuses_without_resuming(control):
    path, _, query, _ = control
    query("pause", "watch")
    (path / "maintenance.json").write_text(json.dumps({"owner": "lost", "components": []}))
    with pytest.raises(ValueError), MaintenanceHold():
        pass
    assert query()["components"]["watch"]["paused"]


@pytest.mark.parametrize("gate", ["keep-down", "sentinel", "operator"])
def test_late_standing_hold_is_preserved_and_releases_maintenance_ownership(control, monkeypatch, gate):
    client = maintenance_control.control_client()
    with MaintenanceHold():
        if gate == "keep-down":
            monkeypatch.setenv("BRAINLAYER_MAINTENANCE_KEEP_DOWN", "com.brainlayer.watch")
        elif gate == "sentinel":
            client.pause_path.write_text(json.dumps({"labels": ["com.brainlayer.watch"]}))
        else:
            client.fallback = lambda a: subprocess.CompletedProcess(a, 0, '"com.brainlayer.watch" => disabled', "")
    state = control[2]()["components"]["watch"]
    assert state["paused"] and state["pause_owner"] == "" and state["pid"] is None


@pytest.mark.parametrize("control", ["old"], indirect=True)
def test_owner_capability_required_before_pausing(control):
    with pytest.raises(RuntimeError), MaintenanceHold():
        pass
    assert control[2]()["components"]["watch"]["pid"]
