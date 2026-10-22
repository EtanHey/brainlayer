"""Completion regressions with real heartbeat pipes and isolated runner custody."""

import asyncio
import datetime as dt
import json
import os
import signal
import subprocess
import threading
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

import pytest

from brainlayer import job_alerts
from brainlayer import maintenance_calendar as calendar


@pytest.fixture
def runner(tmp_path, monkeypatch):
    env_file = tmp_path / "fixture.env"
    env_file.write_text("BRAINLAYER_BACKUP_SNAPSHOT_TIMEOUT_SECONDS=60\n")
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(env_file))
    monkeypatch.setattr(
        calendar.maintenance_control,
        "control_client",
        lambda: SimpleNamespace(path=tmp_path / "unused.sock", request=lambda: {}, _disabled=lambda *_a, **_k: set()),
    )
    reader, writer = os.pipe()
    stdin = os.fdopen(writer, "wb", buffering=0)
    guard = SimpleNamespace(
        pid=12345,
        stdin=stdin,
        lease_channel=SimpleNamespace(close=lambda: None),
        poll=lambda: 0 if stdin.closed else None,
    )
    monkeypatch.setattr(calendar, "launch_guard", lambda *_a, **_k: guard)
    writes, states = [], []
    original_write, atomic_write = os.write, calendar._atomic_write

    def observe_write(fd, data):
        written = original_write(fd, data)
        if fd == writer:
            writes.append(os.read(reader, written))
        return written

    def observe_state(path, value):
        atomic_write(path, value)
        states.append(json.loads(path.read_text()))

    monkeypatch.setattr(os, "write", observe_write)
    monkeypatch.setattr(calendar, "_atomic_write", observe_state)
    directory = tmp_path / "calendar"
    yield SimpleNamespace(
        directory=directory,
        logs=tmp_path / "logs",
        alerts=tmp_path / "alerts.json",
        writes=writes,
        states=states,
        guard=guard,
        rows=lambda: [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()],
    )
    stdin.close()
    os.close(reader)


async def wait_for(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


def test_stopped_child_cleanup_does_not_begin_guard_shutdown(runner, monkeypatch):
    reader, gate = os.pipe()
    child = SimpleNamespace(pid=23456, returncode=None, poll=lambda: None)
    monkeypatch.setattr(calendar, "gated_spawn", lambda *_a, **_k: (child, gate))
    monkeypatch.setattr(calendar, "start_time", lambda _pid: "fixture-birth")
    reaped = []

    async def stop_group(process, *_args):
        reaped.append(process.pid)

    monkeypatch.setattr(calendar, "stop_group", stop_group)
    runner.logs.mkdir()
    stop = asyncio.Event()
    stop.set()
    try:
        with calendar.Custody(runner.directory) as custody:
            result = asyncio.run(
                calendar._execute_step(
                    calendar.Step("index", ("/unused",), 60),
                    {},
                    directory=runner.directory,
                    logs=runner.logs,
                    custody=custody,
                    guard=runner.guard,
                    stop=stop,
                    guard_timeout=30,
                    shutdown_timeout=15,
                )
            )
            assert not custody.groups
    finally:
        os.close(reader)
    assert result == 143
    assert reaped == [child.pid]
    assert b"H\n" in runner.writes
    assert b"Q\n" not in runner.writes


@pytest.mark.parametrize("stop_at", ["finish", "protocol"])
def test_stop_keeps_real_heartbeats_and_custody_until_finish_exits(runner, monkeypatch, stop_at):
    entered, release = threading.Event(), threading.Event()
    handlers, executed = {}, []

    def notify(message):
        entered.set()
        assert release.wait(5), "finish worker was not released"
        raise subprocess.TimeoutExpired("osascript", 8)

    async def execute(step, env, **kwargs):
        executed.append(step.name)
        if stop_at == "protocol":
            raise RuntimeError("fixture protocol error")
        return 124

    monkeypatch.setattr(calendar, "_execute_step", execute)
    monkeypatch.setattr(calendar, "report", lambda k, r: job_alerts.report(k, r, path=runner.alerts, notify=notify))
    monkeypatch.setattr(signal, "signal", lambda *_a: None)

    async def exercise():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "add_signal_handler", lambda sig, callback: handlers.__setitem__(sig, callback))
        monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: handlers.pop(sig, None))
        task = asyncio.create_task(
            calendar.run_calendar(
                [calendar.Step(name, ("/unused",), 60) for name in ("index", "backup-daily")],
                runner.directory,
                runner.logs,
                now_fn=lambda: dt.datetime(2026, 10, 22, 3, 46),
                install_signals=True,
            )
        )
        try:
            await wait_for(entered.is_set)
            if stop_at == "finish":
                handlers[signal.SIGTERM]()  # Invoke the registered stop action without sending a signal.
            before = runner.writes.count(b"H\n")
            await wait_for(lambda: runner.writes.count(b"H\n") >= before + 3)
            assert not task.done()
            assert b"Q\n" not in runner.writes
            with pytest.raises(BlockingIOError), calendar.Custody(runner.directory):
                pass
        finally:
            release.set()
            result = await task
        assert result == 1

    asyncio.run(exercise())
    assert executed == ["index"]
    rows = runner.rows()
    assert [(r["step"], r["status"]) for r in rows if r["event"] == "finished"] == [
        ("index", "failed"),
        ("backup-daily", "skipped"),
    ]
    assert rows[-1]["event"] == "summary" and rows[-1]["failed"] == 1
    assert len(runner.states) == 1
    assert runner.writes[-1] == b"Q\n"
    with calendar.Custody(runner.directory):
        pass


@pytest.mark.parametrize(
    "first,hour,code",
    [("index", 4, 75), ("backup-snapshot", 6, 75), ("index", 3, 124), ("index", 3, 0), ("index", 3, 75)],
)
def test_alert_file_error_preserves_one_finish_and_independent_backups(runner, monkeypatch, first, hour, code):
    reported, executed = [], []
    locked = job_alerts._locked

    @contextmanager
    def fail_first_lock(path):
        if len(reported) == 1:
            raise OSError("private-alert-path")
        with locked(path):
            yield

    def report(key, reason):
        reported.append(key)
        return job_alerts.report(key, reason, path=runner.alerts, notify=lambda _message: None)

    async def execute(step, env, **kwargs):
        executed.append(step.name)
        return code if step.name == first else 0

    monkeypatch.setattr(job_alerts, "_locked", fail_first_lock)
    monkeypatch.setattr(calendar, "report", report)
    monkeypatch.setattr(calendar, "_execute_step", execute)
    result = asyncio.run(
        calendar.run_calendar(
            [calendar.Step(name, ("/unused",), 60) for name in (first, "jsonl-backup", "backup-daily")],
            runner.directory,
            runner.logs,
            now_fn=lambda: dt.datetime(2026, 10, 22, hour, 46),
            defer_after=1,
        )
    )
    assert reported == [calendar.KEYS[first], "jsonl-backup", "backup-daily"]
    assert executed == ([first] if hour == 3 else []) + ["jsonl-backup", "backup-daily"]
    rows = runner.rows()
    assert [(r["step"], r["returncode"]) for r in rows if r["event"] == "finished"] == [
        (first, code),
        ("jsonl-backup", 0),
        ("backup-daily", 0),
    ]
    assert [r["error"] for r in rows if r["event"] == "alert-error"] == ["OSError"]
    assert not any(r["event"] in {"step-error", "runner-error"} for r in rows)
    assert "private-alert-path" not in (runner.directory / "runs.jsonl").read_text()
    assert rows[-1]["event"] == "summary"
    assert rows[-1]["failed"] == int(code == 124)
    assert rows[-1]["deferred"] == int(code == 75)
    assert result == int(code == 124)
    assert len(runner.states) == 3
    assert runner.states[-1]["deferrals"] == (
        {calendar.KEYS[first]: {"date": "2026-10-22", "count": 1}} if code == 75 else {}
    )


@pytest.mark.parametrize("first,hour", [("index", 4), ("backup-snapshot", 6)])
def test_early_finish_state_error_is_fatal_without_duplicate_receipt(runner, monkeypatch, first, hour):
    def fail_state(*_args):
        raise OSError("private-state-path")

    monkeypatch.setattr(calendar, "_atomic_write", fail_state)
    monkeypatch.setattr(calendar, "report", lambda *_a: None)
    with pytest.raises(OSError):
        asyncio.run(
            calendar.run_calendar(
                [calendar.Step(first, ("/unused",), 60)],
                runner.directory,
                runner.logs,
                now_fn=lambda: dt.datetime(2026, 10, 22, hour, 46),
                defer_after=1,
            )
        )
    rows = runner.rows()
    assert [r["returncode"] for r in rows if r["event"] == "finished"] == [75]
    assert rows[-1]["event"] == "summary" and rows[-1]["failed"] == 1


def test_index_timeout_slow_report_then_window_wait_and_backup(runner, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    clock = [dt.datetime(2026, 10, 22, 3, 46)]
    executed = []

    def notify(message):
        entered.set()
        assert release.wait(5), "finish worker was not released"
        raise subprocess.TimeoutExpired("osascript", 8)

    async def execute(step, env, **kwargs):
        executed.append(step.name)
        return 124 if step.name == "index" else 0

    @asynccontextmanager
    async def hold(_client, _guard, env, *_args):
        yield env

    async def advance_window(_seconds):
        before = runner.writes.count(b"H\n")
        await wait_for(lambda: runner.writes.count(b"H\n") >= before + 2)
        clock[0] = clock[0].replace(hour=4, minute=0)

    monkeypatch.setattr(calendar, "report", lambda k, r: job_alerts.report(k, r, path=runner.alerts, notify=notify))
    monkeypatch.setattr(calendar, "_execute_step", execute)
    monkeypatch.setattr(calendar, "step_hold", hold)

    async def exercise():
        task = asyncio.create_task(
            calendar.run_calendar(
                [calendar.Step(name, ("/unused",), 60) for name in ("index", "maintenance-nightly", "backup-daily")],
                runner.directory,
                runner.logs,
                now_fn=lambda: clock[0],
                sleep_fn=advance_window,
            )
        )
        try:
            await wait_for(entered.is_set)
            before = runner.writes.count(b"H\n")
            await wait_for(lambda: runner.writes.count(b"H\n") >= before + 3)
        finally:
            release.set()
            result = await task
        assert result == 1

    asyncio.run(exercise())
    assert executed == ["index", "maintenance-nightly", "backup-daily"]
    rows = runner.rows()
    assert [(r["event"], r.get("step")) for r in rows] == [
        ("runner", None),
        ("finished", "index"),
        ("wait-start", "maintenance-nightly"),
        ("wait-finished", "maintenance-nightly"),
        ("finished", "maintenance-nightly"),
        ("finished", "backup-daily"),
        ("summary", None),
    ]
    assert rows[-1]["failed"] == 1
    assert len(runner.states) == 3
