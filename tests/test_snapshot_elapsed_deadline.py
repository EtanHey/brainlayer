"""#1422: one elapsed deadline from the calendar's kill, through startup, to the timeout receipt.

The ordering that matters is ``startup + work + grace + receipt < calendar kill``. These tests drive
the real scheduling boundary (``maintenance_calendar._bounded_snapshot_step``) and the real backup
entrypoint (``backup_daily.main`` -> ``_supervise_backup_process``) on a virtual clock shared by both
sides. The backup child is a synthetic process object; the calendar's TERM is modelled the way the
supervisor experiences it, as ``SystemExit`` the moment the clock reaches the kill (its
``forward_shutdown_signal``), so a receipt written at or after the kill counts as never written.
"""

import datetime as dt
import math
import os
import signal
import subprocess
import sys
import time
import types

import pytest

from brainlayer import backup_daily as backup
from brainlayer import maintenance_calendar as calendar

GB = 1_000_000_000
SHUTDOWN_TIMEOUT = 15


class Clock:
    def __init__(self):
        self.now = 10_000.0
        self.kill_at = None

    def advance(self, seconds):
        self.now += seconds
        if self.kill_at is not None and self.now >= self.kill_at:
            raise SystemExit(128 + signal.SIGTERM)  # the calendar TERM reaches the supervisor first


class SyntheticChild:
    def __init__(self, clock, *, work, ignores_alarm, alarm_unwind=2.0, wake_lag=0.0):
        self.clock, self.pid, self.returncode = clock, 4242, None
        self.finish_at = clock.now + work
        self.ignores_alarm, self.alarm_unwind, self.wake_lag = ignores_alarm, alarm_unwind, wake_lag
        self.signals = []

    def poll(self):
        if self.returncode is None and self.clock.now >= self.finish_at:
            self.returncode = 0
        return self.returncode

    def wait(self, timeout=None):
        if self.poll() is not None:
            return self.returncode
        if timeout is None:
            raise AssertionError("unbounded wait on a running child")
        if self.finish_at - self.clock.now <= timeout:
            self.clock.advance(self.finish_at - self.clock.now)
            return self.poll()
        lag, self.wake_lag = self.wake_lag, 0.0  # e.g. a delayed wake: the parent oversleeps once
        self.clock.advance(timeout + lag)
        raise subprocess.TimeoutExpired("synthetic-backup-child", timeout)

    def signal(self, signum):
        self.signals.append(signal.Signals(signum).name)
        if signum == signal.SIGKILL:
            self.returncode = -signal.SIGKILL
        elif signum == signal.SIGALRM and not self.ignores_alarm:
            self.finish_at = min(self.finish_at, self.clock.now + self.alarm_unwind)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    db = tmp_path / "brainlayer.db"
    with db.open("wb") as handle:
        handle.truncate(21 * GB)  # sparse; sizes the snapshot budget at 4200 s
    monkeypatch.setenv("BRAINLAYER_DB", str(db))
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "missing.env"))
    for key in (
        "BRAINLAYER_BACKUP_SNAPSHOT_TIMEOUT_SECONDS",
        "BRAINLAYER_BACKUP_TIMEOUT_SECONDS",
        "BRAINLAYER_BACKUP_CALENDAR_SNAPSHOT_BUDGET_SECONDS",
        "BRAINLAYER_BACKUP_CALENDAR_DEADLINE_MONOTONIC",
        "BRAINLAYER_CALENDAR_GUARD_FD",
        backup.BACKUP_SUPERVISED_CHILD_ENV,
    ):
        monkeypatch.delenv(key, raising=False)

    clock = Clock()
    fake_time = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time) if not k.startswith("__")})
    fake_time.monotonic = lambda: clock.now
    monkeypatch.setattr(calendar, "time", fake_time)
    monkeypatch.setattr(backup, "time", fake_time)

    state = types.SimpleNamespace(clock=clock, children=[], receipts=[], child_kwargs={})

    def popen(argv, **kwargs):
        clock.advance(state.popen_cost)  # fork/exec of the child is charged on the same clock
        child = SyntheticChild(clock, **state.child_kwargs)
        state.children.append(child)
        return child

    monkeypatch.setattr(
        backup, "subprocess", types.SimpleNamespace(Popen=popen, TimeoutExpired=subprocess.TimeoutExpired)
    )
    monkeypatch.setattr(os, "killpg", lambda pid, signum: state.children[-1].signal(signum))
    monkeypatch.setattr(backup, "_append_json_log", lambda path, row: state.receipts.append((clock.now, row)))
    monkeypatch.setattr("brainlayer.job_alerts.report", lambda *a, **kw: None)

    def run(start, *, wrapper_startup=0.0, popen_cost=0.0, **child_kwargs):
        state.popen_cost, state.child_kwargs = popen_cost, child_kwargs
        plan = calendar.calendar_steps("python", "cli", tmp_path, start.date())
        step = next(s for s in plan.select() if s.name == "backup-snapshot")
        bounded = calendar._bounded_snapshot_step(step, start, SHUTDOWN_TIMEOUT)
        state.bounded = bounded
        if bounded is None:
            state.outcome = "calendar-deferred"  # the run loop records code 75 and spawns nothing
            return state
        for key, value in bounded.env.items():
            monkeypatch.setenv(key, value)
        clock.kill_at = clock.now + bounded.timeout  # earliest possible kill: spawned at once
        state.kill_at = clock.kill_at
        try:
            clock.advance(wrapper_startup)  # env-run + Python start before main() runs
            state.outcome = backup.main()
        except SystemExit as exc:
            state.outcome = f"killed:{exc.code}"
        return state

    return run


def _timeout_receipts(state):
    return [(at, row) for at, row in state.receipts if row.get("error_type") == "BackupTimeoutError"]


@pytest.mark.parametrize(
    ("label", "start", "kwargs"),
    [
        # 4200/4140 with a 51 s startup and a child that uses the whole SIGALRM grace.
        ("4200-4140-51s-startup", (3, 45, 10), dict(wrapper_startup=51, work=9_999, ignores_alarm=True)),
        # Delayed wrapper startup alone, with a child that honours SIGALRM promptly.
        ("delayed-startup", (3, 45, 10), dict(wrapper_startup=100, work=9_999, ignores_alarm=False)),
        # Slow child creation on top of startup, child consumes the full grace.
        ("slow-child-creation", (3, 45, 10), dict(wrapper_startup=25, popen_cost=30, work=9_999, ignores_alarm=True)),
    ],
)
def test_timeout_receipt_is_written_strictly_before_the_calendar_kill(harness, label, start, kwargs):
    state = harness(dt.datetime(2026, 10, 20, *start), **kwargs)
    assert state.bounded.timeout == 4200, label
    receipts = _timeout_receipts(state)
    assert state.outcome == 124, (label, state.outcome)
    assert len(receipts) == 1, label
    written_at, row = receipts[0]
    assert written_at < state.kill_at, (label, written_at - state.kill_at)
    assert row["timeout_enforced_by"] == "parent_process_supervisor"


@pytest.mark.parametrize(
    ("label", "start", "env"),
    [
        ("late-06:00-clamp", (5, 59, 37), {}),  # 06:00 - 05:59:37 - 15 - 1 = 7 s
        ("low-override-7s", (3, 45, 10), {"BRAINLAYER_BACKUP_SNAPSHOT_TIMEOUT_SECONDS": "7"}),
        ("44s-window-below-reserve", (5, 59, 0), {}),  # 44 s window cannot hold the 60 s reserve
    ],
)
def test_a_window_without_work_plus_the_full_reserve_defers_and_spawns_nothing(harness, monkeypatch, label, start, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    state = harness(dt.datetime(2026, 10, 20, *start), wrapper_startup=1, work=9_999, ignores_alarm=True)
    assert state.outcome == "calendar-deferred", (label, state.outcome, state.receipts)
    assert state.children == [], label


def test_backup_refuses_before_spawning_when_startup_ate_the_work_budget(harness):
    # 05:58:00 leaves the calendar a 104 s window (44 s of work); a 50 s startup leaves none.
    state = harness(dt.datetime(2026, 10, 20, 5, 58, 0), wrapper_startup=50, work=9_999, ignores_alarm=True)
    assert state.bounded is not None and state.bounded.timeout == 104
    assert state.outcome == backup.BACKUP_DEFERRED_EXIT_CODE
    assert state.children == []  # no child process was started
    assert state.receipts == []


def test_grace_is_bounded_by_the_remaining_receipt_window(harness):
    # The parent oversleeps its alarm (e.g. a delayed wake) and wakes 8 s before the kill; an
    # unbounded 10 s grace would carry the receipt past it.
    state = harness(
        dt.datetime(2026, 10, 20, 3, 45, 10), wrapper_startup=5, work=9_999, ignores_alarm=True, wake_lag=52
    )
    receipts = _timeout_receipts(state)
    assert state.outcome == 124, state.outcome
    assert len(receipts) == 1
    assert receipts[0][0] < state.kill_at
    assert state.kill_at - receipts[0][0] >= backup.SUPERVISOR_RECEIPT_WRITE_SECONDS
    assert state.children[0].signals == ["SIGALRM", "SIGKILL"]


@pytest.mark.parametrize(("startup", "expected"), [(20, None), (4141, "deferred")])
def test_supervised_child_takes_its_own_timer_from_the_same_deadline(harness, monkeypatch, startup, expected):
    monkeypatch.setenv(backup.BACKUP_SUPERVISED_CHILD_ENV, "1")
    seen = []
    monkeypatch.setattr(backup, "_run_backup_process", lambda seconds: seen.append(seconds) or 0)
    state = harness(dt.datetime(2026, 10, 20, 3, 45, 10), work=9_999, ignores_alarm=True)
    # `harness` ran main() as the supervisor-less child at once; rerun after a startup delay.
    seen.clear()
    state.clock.kill_at = None
    state.clock.advance(startup)
    code = backup.main()
    deadline = float(state.bounded.env[backup.BACKUP_CALENDAR_DEADLINE_ENV])
    if expected == "deferred":
        assert code == backup.BACKUP_DEFERRED_EXIT_CODE and seen == []
    else:
        assert seen == [math.floor(deadline - state.clock.now - backup.SNAPSHOT_RECEIPT_RESERVE_SECONDS)]
        assert seen[0] < int(state.bounded.env["BRAINLAYER_BACKUP_TIMEOUT_SECONDS"])


def test_real_child_ignoring_sigalrm_is_killed_inside_the_receipt_window(tmp_path, monkeypatch):
    """A real OS child that ignores SIGALRM; reserve scaled down so the run takes ~3 s of wall time."""
    monkeypatch.delenv("BRAINLAYER_CALENDAR_GUARD_FD", raising=False)
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "brainlayer.db"))
    monkeypatch.setattr(backup, "SNAPSHOT_RECEIPT_RESERVE_SECONDS", 3)
    monkeypatch.setattr(backup, "SUPERVISOR_RECEIPT_WRITE_SECONDS", 1)
    written = []
    real_append = backup._append_json_log
    monkeypatch.setattr(
        backup, "_append_json_log", lambda path, row: (written.append(time.monotonic()), real_append(path, row))
    )
    child = "import signal, time; signal.signal(signal.SIGALRM, signal.SIG_IGN); time.sleep(60)"
    deadline = time.monotonic() + 4  # alarm at +1 s; grace min(10, 3 - 1) = 2 s; receipt by ~+3 s
    code = backup._supervise_backup_process(60, command=[sys.executable, "-c", child], deadline=deadline)
    assert code == 124
    assert len(written) == 1 and written[0] < deadline
