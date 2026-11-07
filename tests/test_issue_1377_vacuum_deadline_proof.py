"""Proof-only fixtures for issue #1377 (weekly VACUUM deadline vs. service restoration).

Virtual hold clock, private temp SQLite files and fake supervisor objects only: no production DB,
no launchctl, no BrainBar socket, no resident services. These pin CURRENT behavior at d6b82f04;
they change no production source.
"""

import asyncio
import sqlite3

import pytest

from brainlayer import maintenance, maintenance_calendar, maintenance_control
from brainlayer.maintenance import HOLD_UNWIND_S, MaintenanceAbort


def _db(tmp_path):
    path = tmp_path / "private.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t(x)")
    conn.executemany("INSERT INTO t VALUES (?)", [("x" * 200,)] * 2000)
    conn.execute("DELETE FROM t WHERE rowid % 2 = 0")
    conn.commit()
    conn.close()
    return path


class VirtualClock:
    def __init__(self, now=0.0, step=0.0):
        self.now, self.step = now, step

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


def test_slow_backup_leaves_too_little_hold_time_so_vacuum_is_never_admitted(tmp_path, monkeypatch):
    """Backup verification consumed the window: VACUUM admission refuses before touching the DB."""
    db = _db(tmp_path)
    before = db.read_bytes()
    deadline = 1000.0
    monkeypatch.setattr(maintenance_control, "hold_clock", VirtualClock(now=deadline - HOLD_UNWIND_S + 1))
    with pytest.raises(MaintenanceAbort, match="expires before VACUUM could unwind"):
        maintenance._vacuum(db, deadline=deadline)
    assert db.read_bytes() == before


def test_size_estimate_that_overruns_the_cutoff_refuses_vacuum(tmp_path, monkeypatch):
    db = _db(tmp_path)
    monkeypatch.setenv(maintenance.VACUUM_SECONDS_PER_GB_ENV, "1e9")  # any size -> huge estimate
    monkeypatch.setattr(maintenance_control, "hold_clock", VirtualClock(now=0.0))
    with pytest.raises(MaintenanceAbort, match="too little time for VACUUM"):
        maintenance._vacuum(db, deadline=HOLD_UNWIND_S + 60)


def test_slow_vacuum_crossing_cutoff_is_interrupted_and_rolled_back(tmp_path, monkeypatch):
    """Admitted VACUUM that runs slow is stopped by the progress handler, not by a SIGKILL."""
    db = _db(tmp_path)
    monkeypatch.setenv(maintenance.VACUUM_SECONDS_PER_GB_ENV, "0")
    monkeypatch.setattr(maintenance, "VACUUM_PROGRESS_OPS", 1)
    deadline = HOLD_UNWIND_S + 5.0
    # Each hold_clock read advances 1 virtual second, so the cutoff falls mid-rebuild.
    monkeypatch.setattr(maintenance_control, "hold_clock", VirtualClock(now=0.0, step=1.0))
    with pytest.raises(MaintenanceAbort, match="expired during VACUUM; VACUUM rolled back") as caught:
        maintenance._vacuum(db, deadline=deadline)
    assert caught.value.code == 76
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert conn.execute("SELECT count(*) FROM t").fetchone() == (1000,)
    conn.close()


class _FakeHold:
    instances = []

    def __init__(self, client, now_fn=None):
        self.fd, self.token, self.deadline = 99, "tok", 123.0
        self.entered = self.exited = False
        _FakeHold.instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True


def _run_hold(monkeypatch, body_raises):
    _FakeHold.instances.clear()
    leases = []
    monkeypatch.setattr(maintenance_calendar, "MaintenanceHold", _FakeHold)
    monkeypatch.setattr(maintenance_calendar, "guard_lease", lambda guard, fd: leases.append(fd))
    step = maintenance_calendar.Step("maintenance-weekly", ("true",), 7300)

    async def go():
        async with maintenance_calendar.step_hold(object(), object(), {}, step) as env:
            assert env["BRAINLAYER_MAINTENANCE_HOLD_DEADLINE"] == "123.0"
            if body_raises:
                raise RuntimeError("owned process group did not exit")  # stop_group failure path

    if body_raises:
        with pytest.raises(RuntimeError):
            asyncio.run(go())
    else:
        asyncio.run(go())
    return _FakeHold.instances[0], leases


def test_supervisor_hold_is_released_by_the_calendar_parent_even_after_child_hard_kill(monkeypatch):
    """Normal path: the outer calendar (not the child's finally) transfers then releases the lease."""
    hold, leases = _run_hold(monkeypatch, body_raises=False)
    assert leases == [99, None]
    assert hold.exited


def test_lease_is_not_handed_back_when_child_group_cleanup_fails(monkeypatch):
    """If the child group cannot be confirmed gone, the guard keeps the lease; the hold still exits."""
    hold, leases = _run_hold(monkeypatch, body_raises=True)
    assert leases == [99]  # no guard_lease(None): writers stay paused until the guard proves cleanup
    assert hold.exited


def test_held_weekly_step_budget_is_the_hold_deadline_not_7300s():
    """Static pin: weekly full keeps its 7300 s bound, but held steps run on min(budget, hold left)."""
    import inspect

    source = inspect.getsource(maintenance_calendar._execute_step)
    assert "clock = maintenance_control.hold_clock" in source
    assert "budget = min(budget, remaining)" in source
    assert 'event="hold-expired"' in source
