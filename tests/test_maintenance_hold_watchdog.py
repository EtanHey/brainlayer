"""P0 2026-10-19: a maintenance hold always ends within a bounded time.

Every process here is a private fixture under a /tmp control directory; no launchd,
no production DB, and every signal targets a PID this test spawned.
"""

import asyncio
import datetime as dt
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

import tests.test_service_supervisor as native_tests
from brainlayer import maintenance, maintenance_control
from brainlayer.maintenance_calendar import Step, run_calendar
from brainlayer.maintenance_control import MaintenanceHold, enforce_hold_expiry, hold_budget
from brainlayer.supervisor_guard import start_time
from tests.test_maintenance_control import control, native  # noqa: F401

HOLDER = """
import signal,subprocess,sys,time
from pathlib import Path
from brainlayer import maintenance_control
from brainlayer.supervisor_adapters import SupervisorCommands
p=Path(sys.argv[1])
if sys.argv[2]=='stubborn': signal.signal(signal.SIGTERM,signal.SIG_IGN)
maintenance_control.control_client=lambda: SupervisorCommands(p/'control.sock',lambda a:subprocess.CompletedProcess(a,0,'',''),pause_path=p/'pause')
h=maintenance_control.MaintenanceHold().__enter__()
(p/'held').write_text(str(h.expires_at)); time.sleep(60)
"""


def _supervised(request, *, fast=False):
    """With ``fast``, the fixture supervisor inherits short bounds: it must start after these are set."""
    if fast:
        monkeypatch = request.getfixturevalue("monkeypatch")
        monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_MAX_S", "1")
        monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_GRACE_S", "0.2")
        monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_KILL_AFTER_S", "0.5")
    return request.getfixturevalue("control")


def _paused(query):
    return all(v["pid"] is None for v in query()["components"].values())


def _watch_running(query):
    return bool(query()["components"]["watch"]["pid"])


def _hold_in_subprocess(path, mode):
    holder = subprocess.Popen((sys.executable, "-c", HOLDER, str(path), mode))
    native_tests.eventually(lambda: (path / "held").exists(), timeout=10)
    return holder


def test_hold_budget_ends_before_window_close_minus_reserve(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_MAX_S", "7200")
    assert hold_budget(dt.datetime(2026, 10, 19, 4, 0)) == 6600  # 06:00 - 600 s reserve
    assert hold_budget(dt.datetime(2026, 10, 19, 5, 49)) == 60
    assert hold_budget(dt.datetime(2026, 10, 19, 5, 55)) <= 0
    assert hold_budget(dt.datetime(2026, 10, 19, 14, 0)) == 7200  # Off-window holds keep only the cap.
    assert hold_budget(None) == 7200  # No window clock (manual FTS repair): the cap alone.
    monkeypatch.delenv("BRAINLAYER_MAINTENANCE_HOLD_MAX_S")
    assert hold_budget(dt.datetime(2026, 10, 19, 4, 0)) == maintenance_control.DEFAULT_HOLD_MAX_S


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "soon"])
def test_invalid_hold_cap_falls_back_to_default(monkeypatch, value):
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_MAX_S", value)
    assert hold_budget(dt.datetime(2026, 10, 19, 14, 0)) == maintenance_control.DEFAULT_HOLD_MAX_S


def test_hold_journal_records_owner_identity_and_expiry(request):
    path = _supervised(request)[0]
    before = time.time()
    with MaintenanceHold() as hold:
        entry = json.loads((path / "maintenance.json").read_text())
        assert entry["pid"] == os.getpid()
        assert entry["pid_start"] == start_time(os.getpid())
        assert entry["expires_at"] == hold.expires_at
        assert before < hold.expires_at <= time.time() + maintenance_control.DEFAULT_HOLD_MAX_S


def test_hold_refuses_to_start_when_window_has_no_room(request):
    path, _, query, _ = _supervised(request)
    with pytest.raises(maintenance_control.HoldWindowClosed):
        with MaintenanceHold(now_fn=lambda: dt.datetime(2026, 10, 19, 5, 55)):
            pass
    assert _watch_running(query)
    assert not (path / "maintenance.json").exists()


def test_sigkill_of_maintenance_owner_resumes_watcher(request):
    path, _, query, _ = _supervised(request, fast=True)
    holder = _hold_in_subprocess(path, "plain")
    try:
        assert _paused(query)
        holder.kill()
        holder.wait(timeout=2)
        native_tests.eventually(lambda: _watch_running(query))
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=2)
    assert not (path / "maintenance.json").exists()


def test_exception_inside_hold_resumes_watcher(request):
    path, _, query, _ = _supervised(request)
    with pytest.raises(ZeroDivisionError), MaintenanceHold():
        assert _paused(query)
        1 / 0
    native_tests.eventually(lambda: _watch_running(query))
    assert not (path / "maintenance.json").exists()


@pytest.mark.parametrize("mode", ["plain", "stubborn"])
def test_hang_beyond_cap_is_ended_by_supervisor_tick(request, mode):
    path, _, query, _ = _supervised(request, fast=True)
    holder = _hold_in_subprocess(path, mode)
    try:
        assert _paused(query)
        # Cap 1 s + grace 0.2 s, then TERM; a TERM-immune owner is KILLed 0.5 s later.
        assert holder.wait(timeout=6) == (-signal.SIGKILL if mode == "stubborn" else -signal.SIGTERM)
        native_tests.eventually(lambda: _watch_running(query))
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=2)
    assert not (path / "maintenance.json").exists()


def test_hung_hold_still_expires_after_supervisor_restart(request):
    """A replacement supervisor has no in-memory pause owner; the journal alone must drive expiry."""
    path, parent, query, _ = _supervised(request, fast=True)
    holder = _hold_in_subprocess(path, "plain")
    replacement = None
    try:
        parent.kill()
        parent.wait(timeout=2)
        replacement = subprocess.Popen((sys.executable, "-c", native_tests.BOOT, str(path)))
        assert holder.wait(timeout=8) == -signal.SIGTERM
        native_tests.eventually(lambda: _watch_running(query), timeout=5)
    finally:
        for process in (holder, replacement):
            if process and process.poll() is None:
                process.kill()
                process.wait(timeout=3)


def test_expiry_never_signals_a_reused_pid(tmp_path):
    bystander = subprocess.Popen((sys.executable, "-c", "import time; time.sleep(30)"))
    try:
        journal = tmp_path / "maintenance.json"
        journal.write_text(
            json.dumps(dict(owner="lost", components={}, pid=bystander.pid, pid_start="0:0", expires_at=0))
        )
        assert enforce_hold_expiry(tmp_path / "control.sock") is None
        time.sleep(0.1)
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait(timeout=2)


def test_expiry_waits_for_grace_before_signalling(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_GRACE_S", "60")
    owner = subprocess.Popen((sys.executable, "-c", "import time; time.sleep(30)"))
    try:
        native_tests.eventually(lambda: start_time(owner.pid) is not None)
        entry = dict(owner="t", components={}, pid=owner.pid, pid_start=start_time(owner.pid), expires_at=time.time())
        (tmp_path / "maintenance.json").write_text(json.dumps(entry))
        assert enforce_hold_expiry(tmp_path / "control.sock") is None
        assert owner.poll() is None
        assert enforce_hold_expiry(tmp_path / "control.sock", now=time.time() + 61) == signal.SIGTERM
        assert owner.wait(timeout=2) == -signal.SIGTERM
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=2)


def test_calendar_held_step_is_cut_at_hold_expiry_not_step_timeout(request, monkeypatch):
    path, _, query, _ = _supervised(request)
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_MAX_S", "1")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(path / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_TEST_PATH_PROVENANCE", "pytest")
    hang = "from brainlayer.maintenance_control import MaintenanceHold\nimport time\nwith MaintenanceHold(): time.sleep(60)"
    started = time.monotonic()
    code = asyncio.run(
        run_calendar(
            [Step("maintenance-weekly", (sys.executable, "-c", hang), 7300)],
            path / "calendar",
            path / "logs",
            now_fn=lambda: dt.datetime(2026, 10, 19, 4, 0, 9),
        )
    )
    assert code == 1
    assert time.monotonic() - started < 15
    events = [json.loads(line) for line in (path / "calendar" / "runs.jsonl").read_text().splitlines()]
    assert any(e["event"] == "hold-expired" and e["step"] == "maintenance-weekly" for e in events)
    assert [e["returncode"] for e in events if e["event"] == "finished"] == [124]
    native_tests.eventually(lambda: _watch_running(query))


def test_calendar_defers_held_step_when_window_leaves_no_hold_budget(request, monkeypatch):
    path, _, query, _ = _supervised(request)
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(path / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_TEST_PATH_PROVENANCE", "pytest")
    probe = "raise SystemExit(99)"
    code = asyncio.run(
        run_calendar(
            [Step("maintenance-nightly", (sys.executable, "-c", probe), 60)],
            path / "calendar",
            path / "logs",
            now_fn=lambda: dt.datetime(2026, 10, 19, 5, 55),
        )
    )
    assert code == 0
    events = [json.loads(line) for line in (path / "calendar" / "runs.jsonl").read_text().splitlines()]
    finished = [e for e in events if e["event"] == "finished"]
    assert finished[0]["returncode"] == 75 and finished[0]["reason"] == "maintenance-window"
    assert not any(e["event"] == "spawn" for e in events)
    assert _watch_running(query)


def test_calendar_weekly_never_waits_for_a_running_backup_under_hold(tmp_path, monkeypatch):
    import fcntl

    config = maintenance.MaintenanceConfig(
        db_path=tmp_path / "brainlayer.db",
        log_path=tmp_path / "maintenance.log",
        backup_staging_dir=tmp_path / "staging",
        backup_log_path=tmp_path / "backup.log",
    )
    config.backup_staging_dir.mkdir()
    monkeypatch.setenv("BRAINLAYER_CALENDAR_STEP", "1")
    monkeypatch.setattr(maintenance.time, "sleep", lambda _s: pytest.fail("held calendar step waited for backup"))
    with (config.backup_staging_dir / ".backup.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(maintenance.MaintenanceAbort, match="backup running") as exc:
            maintenance._weekly_backup(config)
    assert exc.value.code == 76


def test_vacuum_refuses_when_hold_expires_before_it_could_unwind(tmp_path):
    db = tmp_path / "copy.db"
    with sqlite3.connect(db) as conn:
        conn.execute("create table t(x)")
    with pytest.raises(maintenance.MaintenanceAbort, match="hold expires") as exc:
        maintenance._vacuum(db, expires_at=time.time() + maintenance.HOLD_UNWIND_S - 1)
    assert exc.value.code == 76


def test_vacuum_is_interrupted_at_hold_expiry(tmp_path, monkeypatch):
    interrupted = threading.Event()

    class SlowVacuum:
        def execute(self, sql):
            assert sql == "VACUUM"
            if not interrupted.wait(5):
                pytest.fail("VACUUM was not interrupted at hold expiry")
            raise sqlite3.OperationalError("interrupted")

        def interrupt(self):
            interrupted.set()

        def close(self):
            pass

    db = tmp_path / "copy.db"
    db.write_bytes(b"")
    monkeypatch.setattr(maintenance.sqlite3, "connect", lambda _path: SlowVacuum())
    with pytest.raises(maintenance.MaintenanceAbort, match="expired during VACUUM") as exc:
        maintenance._vacuum(db, expires_at=time.time() + maintenance.HOLD_UNWIND_S + 0.2)
    assert exc.value.code == 76


def test_run_maintenance_bounds_vacuum_by_the_supervisor_hold(tmp_path, monkeypatch):
    class FakeHold:
        expires_at = 1234.5
        closed = False

        def close(self):
            FakeHold.closed = True

    def quiesce(_services, booted_out, now_fn):
        assert now_fn is config.now_fn
        booted_out["_supervisor_hold"] = FakeHold()

    seen = []
    receipt = {"snapshot": str(tmp_path / "snap.db.gz")}
    db = tmp_path / "brainlayer.db"
    with sqlite3.connect(db) as conn:
        conn.execute("create table t(x)")
    config = maintenance.MaintenanceConfig(
        db_path=db,
        queue_dir=tmp_path / "queue",
        quarantine_root=tmp_path / "quarantine",
        log_path=tmp_path / "maintenance.log",
        backup_staging_dir=tmp_path / "staging",
        backup_log_path=tmp_path / "backup.log",
        now_fn=lambda: dt.datetime(2026, 10, 19, 4, 30),
        idle_sample_seconds=0,
        recent_write_grace_seconds=0,
    )
    for name in ("_assert_running_merged_code", "_run_gates", "_verify_search_latency", "_emit_telemetry"):
        monkeypatch.setattr(maintenance, name, lambda *_a, **_k: None)
    monkeypatch.setattr(maintenance, "_service_is_loaded", lambda *_a, **_k: True)
    monkeypatch.setattr(maintenance, "_quiesce_services", quiesce)
    monkeypatch.setattr(maintenance, "_weekly_backup", lambda _c: receipt)
    monkeypatch.setattr(maintenance, "_recent_weekly_verified_backup", lambda _c: receipt)
    monkeypatch.setattr(maintenance, "_checkpoint_full", lambda _db: (0, 0, 0))
    monkeypatch.setattr(maintenance, "_vacuum", lambda _db, **kw: seen.append(kw) or (1, 1))
    maintenance.run_maintenance("full", config=config)
    assert seen == [{"expires_at": 1234.5}]
    assert FakeHold.closed
