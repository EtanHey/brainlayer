"""P0 2026-10-19: a maintenance hold always ends within a bounded time.

Every process here is a private fixture under a /tmp control directory; no launchd,
no production DB, and every signal targets a PID this test spawned.
"""

import asyncio
import datetime as dt
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
import time

import pytest

import tests.test_service_supervisor as native_tests
from brainlayer import maintenance, maintenance_calendar, maintenance_control
from brainlayer.maintenance_calendar import Step, run_calendar
from brainlayer.maintenance_control import MaintenanceHold, boot_id, enforce_hold_expiry, hold_budget, hold_clock
from brainlayer.supervisor_guard import component_gate, start_time
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
(p/'held').write_text(str(h.deadline)); time.sleep(60)
"""
STUBBORN = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(30)"
WALL_STEP = 10**6  # About 11.6 days: any wall-clock-derived bound would visibly break.


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


def _stubborn_owner():
    owner = subprocess.Popen((sys.executable, "-c", STUBBORN), stdout=subprocess.PIPE, text=True)
    assert owner.stdout.readline() == "ready\n"  # TERM is ignored from here on.
    return owner


def _journal(directory, owner, *, token="gen-a", deadline=None, **overrides):
    entry = dict(
        owner=token,
        components={},
        pid=owner.pid,
        pid_start=start_time(owner.pid),
        boot=boot_id(),
        deadline=hold_clock() if deadline is None else deadline,
        expires_at=time.time(),
    )
    (directory / "maintenance.json").write_text(json.dumps(entry | overrides))
    return entry | overrides


def _reap(*processes):
    for process in processes:
        if process and process.poll() is None:
            process.kill()
            process.wait(timeout=3)


def _events(path):
    return [json.loads(line) for line in (path / "calendar" / "runs.jsonl").read_text().splitlines()]


def _calendar_env(monkeypatch, path):
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(path / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_TEST_PATH_PROVENANCE", "pytest")


# Budget policy ---------------------------------------------------------------------------


def test_hold_budget_is_window_bound_with_a_clock_and_cap_only_without(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_MAX_S", "7200")
    assert hold_budget(dt.datetime(2026, 10, 19, 4, 0)) == 6600  # 06:00 - 600 s reserve
    assert hold_budget(dt.datetime(2026, 10, 19, 5, 49)) == 60
    assert hold_budget(dt.datetime(2026, 10, 19, 5, 55)) <= 0
    assert hold_budget(dt.datetime(2026, 10, 19, 6, 1)) <= 0  # A window-bound hold never starts after it.
    assert hold_budget(dt.datetime(2026, 10, 19, 3, 59)) <= 0
    assert hold_budget(None) == 7200  # No window clock (manual FTS repair): the cap alone.
    assert hold_budget(dt.datetime(2026, 10, 19, 2, 30), window=(2, 60)) == 1200  # Custom config window.
    monkeypatch.delenv("BRAINLAYER_MAINTENANCE_HOLD_MAX_S")
    assert hold_budget(dt.datetime(2026, 10, 19, 4, 0)) == maintenance_control.DEFAULT_HOLD_MAX_S


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "soon"])
def test_invalid_hold_cap_falls_back_to_default(monkeypatch, value):
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_MAX_S", value)
    assert hold_budget(None) == maintenance_control.DEFAULT_HOLD_MAX_S


def test_hold_clock_ignores_wall_clock_steps(monkeypatch):
    before = hold_clock()
    monkeypatch.setattr(maintenance_control.time, "time", lambda: 0.0)
    assert 0 <= hold_clock() - before < 5


# Grant -----------------------------------------------------------------------------------


def test_hold_journal_records_generation_and_boot_scoped_deadline(request):
    path = _supervised(request)[0]
    with MaintenanceHold() as hold:
        entry = json.loads((path / "maintenance.json").read_text())
        assert entry["pid"] == os.getpid()
        assert entry["pid_start"] == start_time(os.getpid())
        assert entry["boot"] == boot_id()
        assert entry["owner"] == hold.token
        assert entry["deadline"] == hold.deadline
        assert 0 < hold.deadline - hold_clock() <= maintenance_control.DEFAULT_HOLD_MAX_S
        assert isinstance(entry["expires_at"], float)  # Wall time is for reporting only.


def test_hold_refuses_to_start_when_window_has_no_room(request):
    path, _, query, _ = _supervised(request)
    with pytest.raises(maintenance_control.HoldWindowClosed):
        with MaintenanceHold(now_fn=lambda: dt.datetime(2026, 10, 19, 5, 55)):
            pass
    assert _watch_running(query)
    assert not (path / "maintenance.json").exists()


def test_hold_refuses_without_a_readable_owner_identity(request, monkeypatch):
    path, _, query, _ = _supervised(request)
    client = maintenance_control.control_client()
    actions = []
    request_ = client.request
    monkeypatch.setattr(
        client, "request", lambda action="status", *a, **k: actions.append(action) or request_(action, *a, **k)
    )
    monkeypatch.setattr(maintenance_control, "start_time", lambda _pid: None)
    with pytest.raises(RuntimeError, match="owner identity"), MaintenanceHold():
        pass
    assert "pause" not in actions
    assert not (path / "maintenance.json").exists()
    assert _watch_running(query)


# Resume after death / exception / hang ---------------------------------------------------


def test_sigkill_of_maintenance_owner_resumes_watcher(request):
    path, _, query, _ = _supervised(request, fast=True)
    holder = _hold_in_subprocess(path, "plain")
    try:
        assert _paused(query)
        holder.kill()
        holder.wait(timeout=2)
        native_tests.eventually(lambda: _watch_running(query))
    finally:
        _reap(holder)
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
        # Cap 1 s + grace 0.2 s, then TERM; a TERM-immune owner is KILLed 0.5 s after a delivered TERM.
        assert holder.wait(timeout=6) == (-signal.SIGKILL if mode == "stubborn" else -signal.SIGTERM)
        native_tests.eventually(lambda: _watch_running(query))
    finally:
        _reap(holder)
    assert not (path / "maintenance.json").exists()
    assert not (path / "maintenance-escalation.json").exists()


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
        _reap(holder, replacement)


def test_delivered_term_survives_supervisor_restart(request, monkeypatch):
    """A restart must not hand a TERM-immune owner a fresh TERM cycle."""
    path, parent, query, _ = _supervised(request, fast=True)
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_KILL_AFTER_S", "3")  # Inherited by the replacement only.
    holder = _hold_in_subprocess(path, "stubborn")
    replacement = None
    try:
        native_tests.eventually(lambda: (path / "maintenance-escalation.json").exists(), timeout=6)
        term_at = json.loads((path / "maintenance-escalation.json").read_text())["term_at"]
        parent.kill()
        parent.wait(timeout=2)
        replacement = subprocess.Popen((sys.executable, "-c", native_tests.BOOT, str(path)))
        assert holder.wait(timeout=10) == -signal.SIGKILL
        assert hold_clock() - term_at < 8  # KILL followed the persisted TERM, not a new 3 s cycle.
        native_tests.eventually(lambda: _watch_running(query), timeout=5)
    finally:
        _reap(holder, replacement)


# Escalation is tied to the live hold generation ------------------------------------------


@pytest.fixture
def leased(tmp_path):
    """A held exclusive lease standing in for a live hold on a private control directory."""
    with component_gate(tmp_path, exclusive=True):
        yield tmp_path


def test_expiry_never_signals_a_reused_pid(leased):
    bystander = _stubborn_owner()
    try:
        _journal(leased, bystander, deadline=0, pid_start="0:0")
        assert enforce_hold_expiry(leased / "control.sock", now=10**9) is None
    finally:
        _reap(bystander)


def test_expiry_waits_for_grace_then_terms(leased, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_GRACE_S", "60")
    owner = _stubborn_owner()
    try:
        entry = _journal(leased, owner)
        assert enforce_hold_expiry(leased / "control.sock", now=entry["deadline"] + 59) is None
        assert enforce_hold_expiry(leased / "control.sock", now=entry["deadline"] + 61) == signal.SIGTERM
        assert enforce_hold_expiry(leased / "control.sock", now=entry["deadline"] + 62) is None  # TERM in flight.
        assert enforce_hold_expiry(leased / "control.sock", now=entry["deadline"] + 92) == signal.SIGKILL
        assert owner.wait(timeout=2) == -signal.SIGKILL
    finally:
        _reap(owner)


def test_expiry_never_signals_a_released_lease(tmp_path):
    owner = _stubborn_owner()
    try:
        entry = _journal(tmp_path, owner)  # Journal present, lease free: the hold has ended.
        assert enforce_hold_expiry(tmp_path / "control.sock", now=entry["deadline"] + 10**6) is None
    finally:
        _reap(owner)


def test_release_during_identity_observation_is_not_signalled(tmp_path, monkeypatch):
    owner = _stubborn_owner()
    gate = component_gate(tmp_path, exclusive=True)
    gate.__enter__()
    try:
        entry = _journal(tmp_path, owner)
        real = maintenance_control.start_time

        def release_then_observe(pid):
            (tmp_path / "maintenance.json").unlink()  # The owner restores and drops its lease
            gate.__exit__(None, None, None)  # between the first look and the signal.
            return real(pid)

        sent = []
        with monkeypatch.context() as scoped:
            scoped.setattr(maintenance_control, "start_time", release_then_observe)
            scoped.setattr(maintenance_control.os, "kill", lambda *a: sent.append(a))
            assert enforce_hold_expiry(tmp_path / "control.sock", now=entry["deadline"] + 10**6) is None
        assert sent == []
    finally:
        _reap(owner)


def test_successive_hold_by_the_same_owner_starts_with_term(leased, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_GRACE_S", "1")
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_KILL_AFTER_S", "5")
    owner = _stubborn_owner()
    try:
        first = _journal(leased, owner, token="gen-a")
        assert enforce_hold_expiry(leased / "control.sock", now=first["deadline"] + 2) == signal.SIGTERM
        second = _journal(leased, owner, token="gen-b", deadline=first["deadline"] + 100)
        # Long after gen-a's TERM: gen-b's first action must still be TERM, never KILL.
        assert enforce_hold_expiry(leased / "control.sock", now=second["deadline"] + 2) == signal.SIGTERM
        assert owner.poll() is None
    finally:
        _reap(owner)


@pytest.mark.parametrize(
    "bad", [dict(deadline=math.nan), dict(deadline=math.inf), dict(boot="previous-boot"), dict(deadline="soon")]
)
def test_unusable_generation_is_never_signalled(leased, bad):
    owner = _stubborn_owner()
    try:
        _journal(leased, owner, **bad)
        assert enforce_hold_expiry(leased / "control.sock", now=10**9) is None
    finally:
        _reap(owner)


@pytest.mark.parametrize("step", [WALL_STEP, -WALL_STEP])
def test_supervisor_expiry_ignores_wall_clock_steps(leased, monkeypatch, step):
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_GRACE_S", "0.2")
    owner = _stubborn_owner()
    try:
        entry = _journal(leased, owner, deadline=hold_clock() + 1.0)
        real_time = time.time
        monkeypatch.setattr(maintenance_control.time, "time", lambda: real_time() + step)
        assert enforce_hold_expiry(leased / "control.sock") is None  # Not due on the hold clock.
        while hold_clock() < entry["deadline"] + 0.3:
            time.sleep(0.05)
        assert enforce_hold_expiry(leased / "control.sock") == signal.SIGTERM
    finally:
        _reap(owner)


# Calendar --------------------------------------------------------------------------------


@pytest.mark.parametrize("wall_step", [0, WALL_STEP, -WALL_STEP])
def test_calendar_held_step_is_cut_at_hold_expiry_not_step_timeout(request, monkeypatch, wall_step):
    path, _, query, _ = _supervised(request)
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_HOLD_MAX_S", "1")
    _calendar_env(monkeypatch, path)
    real_time, offset, record = time.time, [0], maintenance_calendar.record

    def stepping_record(directory, **event):
        if event.get("event") == "spawn":
            offset[0] = wall_step  # The wall clock steps after the hold is granted and the child spawned.
        record(directory, **event)

    monkeypatch.setattr(time, "time", lambda: real_time() + offset[0])
    monkeypatch.setattr(maintenance_calendar, "record", stepping_record)
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
    events = _events(path)
    assert any(e["event"] == "hold-expired" and e["step"] == "maintenance-weekly" for e in events)
    assert [e["returncode"] for e in events if e["event"] == "finished"] == [124]
    native_tests.eventually(lambda: _watch_running(query))


@pytest.mark.parametrize(
    "name,clock",
    [("maintenance-nightly", dt.datetime(2026, 10, 19, 5, 55)), ("repair-fts", dt.datetime(2026, 10, 19, 6, 1))],
)
def test_calendar_defers_held_step_outside_the_hold_window(request, monkeypatch, name, clock):
    path, _, query, _ = _supervised(request)
    _calendar_env(monkeypatch, path)
    code = asyncio.run(
        run_calendar(
            [Step(name, (sys.executable, "-c", "raise SystemExit(99)"), 60)],
            path / "calendar",
            path / "logs",
            now_fn=lambda: clock,
        )
    )
    assert code == 0
    finished = [e for e in _events(path) if e["event"] == "finished"]
    assert finished[0]["returncode"] == 75 and finished[0]["reason"] == "maintenance-window"
    assert not any(e["event"] == "spawn" for e in _events(path))
    assert _watch_running(query)


def test_late_hold_refusal_defers_and_the_upload_still_runs(request, monkeypatch):
    path, _, _, _ = _supervised(request)
    _calendar_env(monkeypatch, path)
    budgets = iter([60.0])  # The precheck sees room; the clock crosses 05:50 before the grant.
    real_budget = maintenance_control.hold_budget
    monkeypatch.setattr(
        maintenance_control, "hold_budget", lambda now=None, **kw: next(budgets, -1.0) if now else real_budget(now)
    )
    marker = path / "uploaded"
    upload = f"from pathlib import Path; Path({str(marker)!r}).touch()"
    code = asyncio.run(
        run_calendar(
            [
                Step("maintenance-nightly", (sys.executable, "-c", "raise SystemExit(99)"), 60),
                Step("backup-daily", (sys.executable, "-c", upload), 60),
            ],
            path / "calendar",
            path / "logs",
            now_fn=lambda: dt.datetime(2026, 10, 19, 5, 49),
        )
    )
    assert code == 0
    finished = {e["step"]: e for e in _events(path) if e["event"] == "finished"}
    assert finished["maintenance-nightly"]["returncode"] == 75
    assert finished["maintenance-nightly"]["reason"] == "maintenance-window"
    assert finished["backup-daily"]["status"] == "ok" and marker.exists()
    assert not any(e["event"] == "step-error" for e in _events(path))


def test_standalone_window_refusal_is_a_deliberate_deferral():
    assert maintenance._is_deliberate_deferral(str(maintenance_control.HoldWindowClosed.message))


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


# VACUUM (real SQLite) --------------------------------------------------------------------


@pytest.fixture
def vacuum_db(tmp_path):
    db = tmp_path / "copy.db"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("create table t(id integer primary key, body blob)")
        conn.executemany("insert into t(body) values(?)", [(os.urandom(512),) for _ in range(4000)])
        conn.execute("delete from t where id % 3 = 0")  # Free pages give VACUUM real work.
    return db


def _rows(db):
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        return conn.execute("select count(*), sum(length(body)) from t").fetchone()


def _scripted_clock(monkeypatch, values):
    """Hold-clock readings: the preflight read first, then each progress-handler poll."""
    readings = iter(values)
    calls = []

    def clock():
        calls.append(1)
        return next(readings)

    monkeypatch.setattr(maintenance_control, "hold_clock", clock)
    monkeypatch.setattr(maintenance, "VACUUM_PROGRESS_OPS", 50)
    return calls


def test_vacuum_refuses_when_hold_expires_before_it_could_unwind(vacuum_db):
    before = _rows(vacuum_db)
    with pytest.raises(maintenance.MaintenanceAbort, match="hold expires") as exc:
        maintenance._vacuum(vacuum_db, deadline=hold_clock() + maintenance.HOLD_UNWIND_S - 1)
    assert exc.value.code == 76 and _rows(vacuum_db) == before


def test_vacuum_is_interrupted_mid_statement_and_rolls_back(vacuum_db, monkeypatch):
    before, free = _rows(vacuum_db), _free_pages(vacuum_db)
    deadline = 1000.0 + maintenance.HOLD_UNWIND_S
    calls = _scripted_clock(monkeypatch, [0.0] * 20 + [1000.0] * 10**6)  # Expiry lands mid-VACUUM.
    with pytest.raises(maintenance.MaintenanceAbort, match="expired during VACUUM") as exc:
        maintenance._vacuum(vacuum_db, deadline=deadline)
    assert exc.value.code == 76
    assert isinstance(exc.value.__cause__, sqlite3.OperationalError)
    assert exc.value.__cause__.sqlite_errorcode == sqlite3.SQLITE_INTERRUPT
    assert len(calls) > 20  # The handler ran inside VACUUM before it fired.
    assert _rows(vacuum_db) == before and _free_pages(vacuum_db) == free  # Rolled back, not compacted.


def test_vacuum_expiry_between_preflight_and_statement_still_interrupts(vacuum_db, monkeypatch):
    before = _rows(vacuum_db)
    _scripted_clock(monkeypatch, [0.0] + [1000.0] * 10**6)  # Expires right after the preflight read.
    with pytest.raises(maintenance.MaintenanceAbort, match="expired during VACUUM"):
        maintenance._vacuum(vacuum_db, deadline=1000.0 + maintenance.HOLD_UNWIND_S)
    assert _rows(vacuum_db) == before


def test_unrelated_vacuum_error_is_not_reported_as_expiry(vacuum_db):
    blocker = sqlite3.connect(vacuum_db, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(sqlite3.OperationalError) as exc:
            maintenance._vacuum(vacuum_db, deadline=hold_clock() + 3600)
        assert exc.value.sqlite_errorcode != sqlite3.SQLITE_INTERRUPT
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()


def _free_pages(db):
    with sqlite3.connect(db) as conn:
        return conn.execute("PRAGMA freelist_count").fetchone()[0]


def test_vacuum_completes_well_inside_its_hold(vacuum_db):
    before = _rows(vacuum_db)
    assert _free_pages(vacuum_db) > 0
    maintenance._vacuum(vacuum_db, deadline=hold_clock() + 3600)
    assert _free_pages(vacuum_db) == 0 and _rows(vacuum_db) == before


def test_run_maintenance_bounds_vacuum_and_hold_by_config_window(tmp_path, monkeypatch):
    class FakeHold:
        deadline = 1234.5
        closed = False

        def close(self):
            FakeHold.closed = True

    def quiesce(_services, booted_out, now_fn, window):
        assert now_fn is config.now_fn and window == (3, 90)
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
        now_fn=lambda: dt.datetime(2026, 10, 19, 3, 30),
        quiet_window_start_hour=3,
        quiet_window_duration_minutes=90,
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
    assert seen == [{"deadline": 1234.5}]
    assert FakeHold.closed


# Supervisor reporting --------------------------------------------------------------------


def test_restore_failure_after_restart_is_reported_without_pause_owner(tmp_path):
    from brainlayer.service_supervisor import Component, Supervisor

    async def tick():
        supervisor = Supervisor({"watch": Component(("true",), enabled=False)}, tmp_path / "control.sock")
        failed = asyncio.get_running_loop().create_future()
        failed.set_exception(ValueError("unknown maintenance hold journal"))
        supervisor.recovery = failed
        supervisor._tick()
        return supervisor.components["watch"].error

    assert "maintenance recovery failed" in asyncio.run(tick())
