"""A recovery holding custody must not consume the scheduled full-calendar opportunity."""

import asyncio
import datetime as dt
import json
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from brainlayer import maintenance_calendar as calendar
from tests import test_maintenance_calendar_completion as completion

runner = completion.runner
NOW = dt.datetime(2026, 10, 23, 3, 15, tzinfo=dt.UTC)


@pytest.fixture
def priority_runner(runner, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOG_PATH", str(runner.directory.parent / "missing.log"))
    monkeypatch.setattr(calendar, "report", lambda *_a: None)

    def new_guard(*_a, **_k):
        stream = (runner.directory.parent / "heartbeat").open("ab", buffering=0)
        return SimpleNamespace(
            pid=12345,
            stdin=stream,
            lease_channel=SimpleNamespace(close=lambda: None),
            poll=lambda: 0 if stream.closed else None,
        )

    @asynccontextmanager
    async def no_writer_hold(_client, _guard, env, _step, _now):
        yield env

    monkeypatch.setattr(calendar, "launch_guard", new_guard)
    monkeypatch.setattr(calendar, "step_hold", no_writer_hold)
    runner.directory.mkdir(mode=0o700)
    (runner.directory / "state.json").write_text(json.dumps(dict(deferrals={}, last_weekly=NOW.date().isoformat())))
    return runner


def test_scheduled_full_calendar_waits_for_recovery_without_overlap_or_replaying_jobs(priority_runner, monkeypatch):
    runner = priority_runner
    plan = calendar.calendar_steps(sys.executable, "/unused", runner.logs, NOW.date())
    expected = [step.name for step in plan.select(NOW.date())]

    async def exercise():
        recovery_entered, release_recovery, full_contended = asyncio.Event(), asyncio.Event(), asyncio.Event()
        executed, active = [], []
        clock = [NOW]
        enter = calendar.Custody.__enter__

        def observe_contention(lease):
            try:
                return enter(lease)
            except BlockingIOError:
                if lease.directory == runner.directory:
                    full_contended.set()
                raise

        async def execute(step, env, **kwargs):
            assert not active, "two live calendar workers overlapped"
            active.append(step.name)
            executed.append(step.name)
            try:
                if not recovery_entered.is_set():
                    assert step.name == "jsonl-backup"
                    recovery_entered.set()
                    await release_recovery.wait()
                elif step.name == "index":
                    clock[0] = NOW.replace(hour=4, minute=5)
                return 0
            finally:
                active.pop()

        async def wait_for_release(_seconds):
            await release_recovery.wait()
            await asyncio.sleep(0)

        monkeypatch.setattr(calendar.Custody, "__enter__", observe_contention)
        monkeypatch.setattr(calendar, "_execute_step", execute)
        recovery = asyncio.create_task(
            calendar.run_calendar(plan, runner.directory, runner.logs, now_fn=lambda: NOW, transcript_recovery=True)
        )
        full = None
        try:
            await asyncio.wait_for(recovery_entered.wait(), 3)
            full = asyncio.create_task(
                calendar.run_calendar(
                    plan, runner.directory, runner.logs, now_fn=lambda: clock[0], sleep_fn=wait_for_release
                )
            )
            await asyncio.wait_for(full_contended.wait(), 3)
            assert not full.done(), "scheduled full calendar exited successfully instead of waiting for recovery"
            # Another recovery and another full invocation must neither barge ahead nor replay the plan.
            assert (
                await calendar.run_calendar(
                    plan, runner.directory, runner.logs, now_fn=lambda: NOW, transcript_recovery=True
                )
                == 0
            )
            assert await calendar.run_calendar(plan, runner.directory, runner.logs, now_fn=lambda: NOW) == 0
            assert executed == ["jsonl-backup"]
        finally:
            release_recovery.set()
            assert await asyncio.wait_for(recovery, 3) == 0
            if full is not None:
                assert await asyncio.wait_for(full, 3) == 0
        assert executed == ["jsonl-backup", *expected]
        assert sum(row["event"] == "runner" for row in runner.rows()) == 1
        with calendar.Custody(runner.directory):
            pass

    asyncio.run(exercise())


def test_full_calendar_admission_is_bounded_and_does_not_report_success_on_timeout(priority_runner, monkeypatch):
    runner = priority_runner
    plan = calendar.calendar_steps(sys.executable, "/unused", runner.logs, NOW.date())
    elapsed, delays = [0.0], []
    monkeypatch.setattr(calendar.maintenance_control, "hold_clock", lambda: elapsed[0])

    async def advance(seconds):
        delays.append(seconds)
        elapsed[0] += 10000

    async def forbidden(*_a, **_k):
        pytest.fail("admission timeout started a worker while recovery retained custody")

    monkeypatch.setattr(calendar, "_execute_step", forbidden)
    with calendar.Custody(runner.directory):
        assert (
            asyncio.run(
                calendar.run_calendar(plan, runner.directory, runner.logs, now_fn=lambda: NOW, sleep_fn=advance)
            )
            == 1
        )
    assert delays and sum(delays) <= 10000
    assert any(row["event"] == "calendar-admission-timeout" for row in runner.rows())
    assert not any(row["event"] == "runner" for row in runner.rows())
    # Timeout must release priority as well as leaving the original custody intact.
    with calendar.Custody(runner.directory / "full-calendar"):
        pass


def test_recovery_admission_probe_cannot_drop_a_scheduled_full_invocation(priority_runner, monkeypatch):
    runner = priority_runner
    plan = calendar.calendar_steps(sys.executable, "/unused", runner.logs, NOW.date())

    async def exercise():
        waiting, release = asyncio.Event(), asyncio.Event()
        executed, clock = [], [NOW]

        async def wait(_seconds):
            waiting.set()
            await release.wait()

        async def execute(step, _env, **_kwargs):
            executed.append(step.name)
            if step.name == "index":
                clock[0] = NOW.replace(hour=4, minute=5)
            return 0

        monkeypatch.setattr(calendar, "_execute_step", execute)
        with calendar.Custody(runner.directory / "recovery-admission"):
            task = asyncio.create_task(
                calendar.run_calendar(plan, runner.directory, runner.logs, now_fn=lambda: clock[0], sleep_fn=wait)
            )
            try:
                await asyncio.wait_for(waiting.wait(), 3)
                assert not task.done()
                assert await calendar.run_calendar(plan, runner.directory, runner.logs, now_fn=lambda: NOW) == 0
                assert not executed
            except BaseException:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise
        release.set()
        assert await asyncio.wait_for(task, 3) == 0
        assert executed == [step.name for step in plan.select(NOW.date())]

    asyncio.run(exercise())


def test_expired_wait_cannot_start_an_old_plan_after_custody_becomes_free(priority_runner, monkeypatch):
    runner = priority_runner
    plan = calendar.calendar_steps(sys.executable, "/unused", runner.logs, NOW.date())
    elapsed = [0.0]
    monkeypatch.setattr(calendar.maintenance_control, "hold_clock", lambda: elapsed[0])

    async def forbidden(*_a, **_k):
        pytest.fail("an overdue invocation started heavy work after wake")

    monkeypatch.setattr(calendar, "_execute_step", forbidden)
    owner = calendar.Custody(runner.directory)

    async def wake_next_day(_seconds):
        elapsed[0] += 86400
        owner.__exit__(None, None, None)

    with owner:
        assert (
            asyncio.run(
                calendar.run_calendar(plan, runner.directory, runner.logs, now_fn=lambda: NOW, sleep_fn=wake_next_day)
            )
            == 1
        )
    assert not any(row["event"] == "runner" for row in runner.rows())
    assert runner.rows()[-1]["event"] == "calendar-admission-timeout"


def test_cancelled_admission_releases_priority_without_starting_workers(priority_runner, monkeypatch):
    runner = priority_runner
    plan = calendar.calendar_steps(sys.executable, "/unused", runner.logs, NOW.date())

    async def exercise():
        waiting = asyncio.Event()

        async def wait(_seconds):
            waiting.set()
            await asyncio.Event().wait()

        with calendar.Custody(runner.directory):
            task = asyncio.create_task(
                calendar.run_calendar(plan, runner.directory, runner.logs, now_fn=lambda: NOW, sleep_fn=wait)
            )
            try:
                await asyncio.wait_for(waiting.wait(), 3)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        with calendar.Custody(runner.directory / "full-calendar"):
            pass

    asyncio.run(exercise())
