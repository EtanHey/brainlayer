import asyncio
import datetime as dt
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from brainlayer import job_alerts
from brainlayer.maintenance_calendar import calendar_steps, run_calendar
from tests.test_maintenance_control import control, native  # noqa: F401


@pytest.mark.parametrize(
    "fixture,abandoned",
    [("abandoned-2026-10-27", True), ("completed-2026-10-26", False), ("success-2026-10-26", False)],
)
def test_calendar_replays_dead_run_without_recovery_summaries_hiding_failure(tmp_path, monkeypatch, fixture, abandoned):
    from brainlayer import maintenance_calendar as calendar

    directory = tmp_path / "calendar"
    directory.mkdir(mode=0o700)
    history = Path(__file__).parent / "fixtures/maintenance_calendar" / f"{fixture}.jsonl"
    (directory / "runs.jsonl").write_text(history.read_text())
    (directory / "state.json").write_text(json.dumps(dict(deferrals={}, last_weekly="2026-10-18")))
    alerts = tmp_path / "alerts.json"
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alerts))
    monkeypatch.setattr(calendar, "start_time", lambda _pid: None)
    monkeypatch.setattr(calendar, "_transcripts_fresh", lambda _now: True)
    now = dt.datetime(2026, 10, 27, 7)
    plan = calendar_steps(sys.executable, "/unused", tmp_path, now.date())
    for _ in range(2):
        assert (
            asyncio.run(run_calendar(plan, directory, tmp_path / "logs", now_fn=lambda: now, transcript_recovery=True))
            == 0
        )
    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    events = [row for row in rows if row["event"] == "abandoned"]
    assert len(events) == int(abandoned)
    state = json.loads((directory / "state.json").read_text())
    assert state["last_weekly"] == "2026-10-18"
    if abandoned:
        event = events[0]
        assert (
            event.items()
            >= dict(
                scheduled_date="2026-10-27",
                step="index",
                pid=90000001,
                runner_started_at=1793063702.7738612,
                reason="runner-vanished",
            ).items()
        )
        assert event["detected_at"] == event["ts"]
        assert event["kind"] == "nightly"
        assert "maintenance-weekly" in event["incomplete_steps"]
        assert state["incomplete_run"]["failed"] == 1
        assert rows[-1]["failed"] == 0
        outcomes = [row for row in rows if row.get("kind") == "nightly" and row["event"] == "summary"]
        assert len(outcomes) == 1 and outcomes[0]["failed"] == 1
        assert "maintenance-full" in job_alerts.active_alerts(alerts)
        assert not any(row["event"] == "spawn" for row in rows[len(history.read_text().splitlines()) :])
    else:
        assert "incomplete_run" not in state
        assert not job_alerts.active_alerts(alerts)


def test_abandoned_run_sets_every_step_alert_but_posts_one_notification(tmp_path, monkeypatch):
    from brainlayer import maintenance_calendar as calendar

    directory = tmp_path / "calendar"
    directory.mkdir(mode=0o700)
    history = Path(__file__).parent / "fixtures/maintenance_calendar/abandoned-2026-10-27.jsonl"
    (directory / "runs.jsonl").write_text(history.read_text())
    (directory / "state.json").write_text(json.dumps(dict(deferrals={}, last_weekly="2026-10-18")))
    alerts = tmp_path / "alerts.json"
    notices = []
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alerts))
    monkeypatch.setattr(calendar, "report", lambda k, r: job_alerts.report(k, r, path=alerts, notify=notices.append))
    monkeypatch.setattr(calendar, "start_time", lambda _pid: None)
    monkeypatch.setattr(calendar, "_transcripts_fresh", lambda _now: True)
    now = dt.datetime(2026, 10, 27, 7)
    plan = calendar_steps(sys.executable, "/unused", tmp_path, now.date())
    run = run_calendar(plan, directory, tmp_path / "logs", now_fn=lambda: now, transcript_recovery=True)
    assert asyncio.run(run) == 0
    abandoned = {k: v for k, v in job_alerts.active_alerts(alerts).items() if "abandoned during" in v}
    assert len(abandoned) > 1 and len(set(abandoned.values())) == 1
    assert notices == list(set(abandoned.values()))


def test_calendar_next_scheduled_plan_acquires_dead_holders_lock(tmp_path, monkeypatch):
    from brainlayer import maintenance_calendar as calendar

    directory = tmp_path / "calendar"
    # A real process exits normally while holding the lock; files remain, kernel ownership does not.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from contextlib import ExitStack; "
            "from brainlayer.supervisor_guard import Custody; "
            "p=Path(__import__('sys').argv[1]); s=ExitStack(); "
            "[s.enter_context(Custody(d)) for d in (p, p/'full-calendar', p/'recovery-admission')]",
            str(directory),
        ],
        check=True,
        env=os.environ | {"PYTHONPATH": str(Path(calendar.__file__).parents[1])},
    )
    (directory / "runs.jsonl").write_text(
        (Path(__file__).parent / "fixtures/maintenance_calendar/abandoned-2026-10-27.jsonl").read_text()
    )
    (directory / "state.json").write_text(json.dumps(dict(deferrals={}, last_weekly="2026-10-18")))
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "fixture.env"))
    monkeypatch.setattr(
        calendar.maintenance_control,
        "control_client",
        lambda: SimpleNamespace(path=tmp_path / "unused.sock", request=lambda: {}, _disabled=lambda *_a, **_k: set()),
    )
    executed = []

    async def execute(step, env, **kwargs):
        executed.append(step.name)
        return 0

    monkeypatch.setattr(calendar, "_execute_step", execute)
    clock = [dt.datetime(2026, 10, 28, 3, 45)]

    async def advance(_seconds):
        clock[0] = clock[0].replace(hour=4)

    now = clock[0]
    plan = calendar_steps(sys.executable, "/unused", tmp_path, now.date())
    # No held maintenance is executed in this admission test; the existing window tests own that contract.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def hold(_client, _guard, env, _step, _now):
        yield env

    monkeypatch.setattr(calendar, "step_hold", hold)
    assert asyncio.run(run_calendar(plan, directory, tmp_path / "logs", now_fn=lambda: clock[0], sleep_fn=advance)) == 0
    assert executed == [step.name for step in plan.select(dt.date(2026, 10, 18))]
    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    assert len([row for row in rows if row["event"] == "abandoned"]) == 1
    assert "incomplete_run" not in json.loads((directory / "state.json").read_text())


@pytest.mark.parametrize("shape", ["startup", "finished", "live", "orphan", "reused", "append-interrupted"])
def test_abandonment_identity_and_interrupted_receipt(tmp_path, monkeypatch, shape):
    from brainlayer import maintenance_calendar as calendar

    directory = tmp_path / "calendar"
    directory.mkdir(mode=0o700)
    rows = [
        dict(
            event="runner",
            guard=90000000,
            scheduled_date="2026-10-27",
            ts=1,
            guard_start_time="original",
            steps=["index"],
        )
    ]
    if shape != "startup":
        rows.append(dict(event="spawn", pid=90000001, step="index", ts=2))
    if shape == "finished":
        rows.append(dict(event="finished", step="index", status="ok", returncode=0, ts=3))
    (directory / "runs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    monkeypatch.setattr(
        calendar,
        "start_time",
        lambda pid: (
            "reused"
            if shape == "reused"
            else "original"
            if shape == "live" or (shape == "orphan" and pid == 90000000)
            else None
        ),
    )
    state = dict(deferrals={})
    plan = calendar_steps(sys.executable, "/unused", tmp_path, dt.date(2026, 10, 27))
    with calendar.Custody(directory):
        if shape == "append-interrupted":
            with monkeypatch.context() as patch:
                patch.setattr(calendar, "report", lambda *_args: (_ for _ in ()).throw(OSError("fixture")))
                calendar._recover_abandoned(directory, state, plan, today=plan.scheduled_date)
        for _ in range(2):
            calendar._recover_abandoned(directory, state, plan, today=plan.scheduled_date)
    recovered = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    events = [row for row in recovered if row["event"] == "abandoned"]
    assert len(events) == (0 if shape == "live" else 1)
    if events:
        assert events[0]["reason"] == ("step-orphaned" if shape == "orphan" else "runner-vanished")
        assert events[0]["step"] == (None if shape == "startup" else "index")
        assert state["incomplete_run"]["failed"] == recovered[-1]["failed"] == 1


@pytest.fixture
def custody_fix1(tmp_path, monkeypatch):
    from brainlayer import maintenance_calendar as calendar

    directory = tmp_path / "calendar"
    directory.mkdir(mode=0o700)
    (directory / "state.json").write_text('{"deferrals":{}}')
    rows = [
        dict(
            event="runner",
            scheduled_date="2026-10-27",
            guard=90000000,
            ts=1,
            steps=["jsonl-backup", "maintenance-nightly"],
        ),
        dict(event="spawn", step="jsonl-backup", pid=90000001, ts=2),
    ]
    journal = directory / "runs.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    alerts = tmp_path / "alerts.json"
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alerts))
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "fixture.env"))
    monkeypatch.setattr(calendar, "_transcripts_fresh", lambda _now: True)
    monkeypatch.setattr(
        calendar.maintenance_control,
        "control_client",
        lambda: SimpleNamespace(path=tmp_path / "unused.sock", request=lambda: {}, _disabled=lambda *_a, **_k: set()),
    )
    selected, executed = ["p0-counter", "jsonl-backup"], []
    monkeypatch.setattr(
        calendar.CalendarPlan,
        "select",
        lambda self, *_: [step for name in selected for step in self.candidates if step.name == name],
    )
    code = [0]

    async def execute(step, env, **kwargs):
        executed.append(step.name)
        return code[0]

    monkeypatch.setattr(calendar, "_execute_step", execute)
    clock = [dt.datetime(2026, 10, 27, 7)]

    def tick(recovery=True):
        plan = calendar_steps(sys.executable, "/unused", tmp_path, clock[0].date())
        return asyncio.run(
            run_calendar(plan, directory, tmp_path / "logs", now_fn=lambda: clock[0], transcript_recovery=recovery)
        )

    def read_rows():
        result = []
        for line in journal.read_text().splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict) and isinstance(row.get("event"), str):
                    result.append(row)
            except ValueError:
                pass
        return result

    return SimpleNamespace(
        calendar=calendar,
        directory=directory,
        journal=journal,
        alerts=alerts,
        selected=selected,
        executed=executed,
        code=code,
        clock=clock,
        tick=tick,
        rows=read_rows,
    )


@pytest.mark.parametrize("bad", ['{"event":', "[]\n", "{}\n", '{"event":"runner"}\n'])
def test_fix1_scan_damage_does_not_block_reap_or_ticks(custody_fix1, monkeypatch, caplog, bad):
    f = custody_fix1
    original = f.journal.read_text() + bad
    f.journal.write_text(original)
    reaped = []
    recover = f.calendar.Custody.recover

    def reap(custody, grace):
        reaped.append(True)
        return recover(custody, grace)

    monkeypatch.setattr(f.calendar.Custody, "recover", reap)
    birth = f.calendar.start_time

    def identity(pid):
        assert reaped, "stale writers must be reaped before scanning identities"
        return birth(pid)

    monkeypatch.setattr(f.calendar, "start_time", identity)
    assert f.tick() == f.tick() == 0
    assert len(reaped) == 2
    assert f.journal.read_text().startswith(original)
    assert len([row for row in f.rows() if row["event"] == "abandoned"]) == 1
    assert f.rows()[-1]["event"] == "summary" and f.rows()[-1]["failed"] == 0
    assert "skipped=1" in caplog.text


def test_fix1_alert_failure_is_logged_and_scheduled_work_runs(custody_fix1, monkeypatch, caplog):
    f = custody_fix1

    def denied(*args):
        raise PermissionError("fixture")

    monkeypatch.setattr(f.calendar, "report", denied)
    assert f.tick(False) == 0
    assert f.executed == f.selected
    assert any(row["event"] == "alert-error" for row in f.rows())
    assert "PermissionError" in caplog.text


def test_fix1_hourly_result_and_alert_are_independent_of_abandonment(custody_fix1, monkeypatch):
    f = custody_fix1
    assert f.tick() == 0
    prior = job_alerts.active_alerts(f.alerts)
    assert "jsonl-backup" in prior
    monkeypatch.setattr(f.calendar, "_transcripts_fresh", lambda _now: False)
    f.code[0] = 1
    assert f.tick() == 1
    f.code[0] = 0
    assert f.tick() == 0
    assert job_alerts.active_alerts(f.alerts) == prior
    summaries = [row for row in f.rows() if row["event"] == "summary"]
    assert [row["failed"] for row in summaries if row.get("kind") == "recovery"] == [0, 1, 0]
    assert len([row for row in summaries if row.get("kind") == "nightly"]) == 1


@pytest.mark.parametrize("code", [0, 75, 1])
def test_fix1_complete_next_nightly_retires_marker_and_replaced_alert(custody_fix1, code):
    f = custody_fix1
    assert f.tick() == 0
    job_alerts.report("index", "unrelated failure", path=f.alerts)
    f.clock[0] += dt.timedelta(days=1)
    f.code[0] = code
    assert f.tick(False) == int(code not in (0, 75))
    assert f.executed == f.selected
    assert "incomplete_run" not in json.loads((f.directory / "state.json").read_text())
    alerts = job_alerts.active_alerts(f.alerts)
    assert "maintenance-light" not in alerts  # This plan replaced the abandoned nightly step.
    assert alerts["index"] == "unrelated failure"
    assert f.rows()[-1]["kind"] == "nightly" and f.rows()[-1]["complete"] is True
    assert f.rows()[-1]["failed"] == int(code not in (0, 75)) * len(f.selected)


def test_fix1_term_during_early_finish_keeps_incomplete_plan(custody_fix1, monkeypatch):
    f = custody_fix1
    assert f.tick() == 0
    f.clock[0] += dt.timedelta(days=1)
    stop = []
    execute = f.calendar._execute_step

    async def capture(*args, **kwargs):
        stop.append(kwargs["stop"])
        return await execute(*args, **kwargs)

    finish = f.calendar._finish_step

    async def terminate(*args, **kwargs):
        # Models SIGTERM arriving during the shielded finish worker, without signalling a process.
        result = await finish(*args, **kwargs)
        stop[0].set()
        return result

    monkeypatch.setattr(f.calendar, "_execute_step", capture)
    monkeypatch.setattr(f.calendar, "_finish_step", terminate)
    assert f.tick(False) == 1
    assert f.executed == [f.selected[0]]
    assert "incomplete_run" in json.loads((f.directory / "state.json").read_text())
    assert f.rows()[-1]["kind"] == "nightly" and f.rows()[-1]["complete"] is False
    assert f.rows()[-1]["failed"] >= 1


@pytest.mark.parametrize("today,expected", [(27, ["2026-10-27"]), (28, ["2026-10-27"]), (29, [])])
def test_fix1_multiday_journal_does_not_backfill_old_runs(custody_fix1, today, expected):
    f = custody_fix1
    f.journal.write_text(
        (Path(__file__).parent / "fixtures/maintenance_calendar/multiday-2026-10-27.jsonl").read_text()
    )
    f.clock[0] = f.clock[0].replace(day=today)
    assert f.tick() == f.tick() == 0
    assert [row["scheduled_date"] for row in f.rows() if row["event"] == "abandoned"] == expected


def test_calendar_invocation_is_fresh_and_reaches_actual_child_and_terminal(tmp_path, monkeypatch):
    from brainlayer import maintenance_calendar as calendar

    (tmp_path / "fixture.env").write_text("")
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "fixture.env"))
    monkeypatch.setenv(calendar.BACKUP_INVOCATION_ENV, "f" * 32)
    monkeypatch.setattr(calendar, "report", lambda *_args: None)
    monkeypatch.setattr(
        calendar.maintenance_control,
        "control_client",
        lambda: SimpleNamespace(path=tmp_path / "unused.sock", request=lambda: {}, _disabled=lambda *_a, **_k: set()),
    )
    step = calendar.Step(
        "backup-daily",
        (sys.executable, "-c", f"import os; print(os.environ[{calendar.BACKUP_INVOCATION_ENV!r}])"),
        5,
        {calendar.BACKUP_INVOCATION_ENV: "f" * 32},
    )
    directory, logs = tmp_path / "calendar", tmp_path / "logs"
    for _ in range(2):
        assert (
            asyncio.run(calendar.run_calendar([step], directory, logs, now_fn=lambda: dt.datetime(2026, 10, 24, 5)))
            == 0
        )
    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    spawned = [r["backup_invocation_id"] for r in rows if r["event"] == "spawn"]
    finished = [r["backup_invocation_id"] for r in rows if r["event"] == "finished"]
    assert spawned == finished == (logs / "backup-daily.out.log").read_text().splitlines()
    assert len(set(spawned)) == 2 and "f" * 32 not in spawned


@pytest.mark.parametrize("first,at,code", [("index", 3, 124), ("index", 4, 75), ("backup-snapshot", 6, 75)])
@pytest.mark.parametrize(
    "notification_error",
    [subprocess.TimeoutExpired("osascript", 8), OSError("fixture"), subprocess.CalledProcessError(1, "osascript")],
)
def test_slow_failure_notification_keeps_heartbeat_and_finishes_calendar(
    tmp_path, monkeypatch, caplog, first, at, code, notification_error
):
    from brainlayer import maintenance_calendar as calendar

    entered, release = threading.Event(), threading.Event()
    blocked, executed, notifications = [], [], []
    alerts = tmp_path / "alerts.json"
    env_file = tmp_path / "fixture.env"
    env_file.write_text("BRAINLAYER_BACKUP_SNAPSHOT_TIMEOUT_SECONDS=60\n")
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(env_file))

    def notify(message):
        notifications.append(message)
        raise notification_error

    def report(key, reason):
        entered.set()
        if not release.wait(5):
            blocked.append(key)
        return job_alerts.report(key, reason, path=alerts, notify=notify)

    async def execute(step, env, **kwargs):
        executed.append(step.name)
        return 124 if step.name == "index" else 0

    monkeypatch.setattr(calendar, "report", report)
    monkeypatch.setattr(calendar, "_execute_step", execute)
    monkeypatch.setattr(
        calendar.maintenance_control,
        "control_client",
        lambda: SimpleNamespace(path=tmp_path / "unused.sock", request=lambda: {}, _disabled=lambda *_a, **_k: set()),
    )
    original_write = os.write

    def observe_heartbeat(fd, data):
        written = original_write(fd, data)
        if data == b"H\n" and entered.is_set():
            release.set()
        return written

    monkeypatch.setattr(os, "write", observe_heartbeat)
    steps = [calendar.Step(name, ("/unused",), 60) for name in (first, "jsonl-backup", "backup-daily")]
    directory = tmp_path / "calendar"
    result = asyncio.run(
        run_calendar(
            steps,
            directory,
            tmp_path / "logs",
            now_fn=lambda: dt.datetime(2026, 10, 22, at, 46),
            defer_after=1,
        )
    )
    assert not blocked, "failure reporting blocked the event loop that sends guard heartbeats"
    assert executed == (["index"] if at == 3 else []) + ["jsonl-backup", "backup-daily"]
    assert notifications
    assert "macOS notification failed" in caplog.text
    assert result == int(code == 124)
    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    assert [(row["step"], row["returncode"]) for row in rows if row["event"] == "finished"] == [
        (first, code),
        ("jsonl-backup", 0),
        ("backup-daily", 0),
    ]
    assert rows[-1]["event"] == "summary"
    assert rows[-1]["failed"] == int(code == 124)
    assert rows[-1]["deferred"] == int(code == 75)
    assert json.loads((directory / "state.json").read_text())["deferrals"] == (
        {calendar.KEYS[first]: {"date": "2026-10-22", "count": 1}} if code == 75 else {}
    )


def test_cancelled_calendar_retains_custody_until_finish_worker_exits(tmp_path, monkeypatch):
    from brainlayer import maintenance_calendar as calendar

    entered, release = threading.Event(), threading.Event()
    directory = tmp_path / "calendar"
    env_file = tmp_path / "fixture.env"
    env_file.write_text("")
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(env_file))

    def report(*args):
        entered.set()
        assert release.wait(5), "test did not release the finish worker"

    async def execute(*args, **kwargs):
        return 124

    monkeypatch.setattr(calendar, "report", report)
    monkeypatch.setattr(calendar, "_execute_step", execute)
    monkeypatch.setattr(
        calendar.maintenance_control,
        "control_client",
        lambda: SimpleNamespace(path=tmp_path / "unused.sock", request=lambda: {}, _disabled=lambda *_a, **_k: set()),
    )

    async def exercise():
        task = asyncio.create_task(
            run_calendar(
                [calendar.Step("index", ("/unused",), 60)],
                directory,
                tmp_path / "logs",
                now_fn=lambda: dt.datetime(2026, 10, 22, 3, 46),
            )
        )
        try:
            async with asyncio.timeout(3):
                while not entered.is_set():
                    await asyncio.sleep(0)
            task.cancel()
            await asyncio.wait({task}, timeout=0.5)
            task.cancel()
            await asyncio.sleep(0)
            # A later calendar must not obtain custody while this worker can still write state.
            assert not task.done()
            with pytest.raises(BlockingIOError), calendar.Custody(directory):
                pass
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(exercise())
    assert json.loads((directory / "state.json").read_text()) == {"deferrals": {}}
    with calendar.Custody(directory):
        pass


@pytest.mark.parametrize("failure_at", ["startup", "execute", "finish", "protocol", "after-failure", "post-execute"])
def test_calendar_always_summarizes_unexpected_failures(tmp_path, monkeypatch, failure_at):
    from brainlayer import maintenance_calendar as calendar

    path = tmp_path
    monkeypatch.setattr(
        calendar.maintenance_control,
        "control_client",
        lambda: SimpleNamespace(path=path / "unused.sock", request=lambda: {}, _disabled=lambda *_a, **_k: set()),
    )
    directory = path / "calendar"
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(path / "fixture.env"))
    (path / "fixture.env").write_text("")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(path / "alerts.json"))
    executed = []

    async def execute(step, env, **kwargs):
        executed.append(step.name)
        if failure_at == "after-failure" and step.name == "jsonl-backup":
            raise KeyError("private-value")
        if step.name == "index":
            if failure_at == "execute":
                raise KeyError("private-value")
            if failure_at == "protocol":
                raise RuntimeError("private-value")
            return 0 if failure_at == "post-execute" else 124
        return 0

    def fail_report(*args):
        raise KeyError("private-value")

    monkeypatch.setattr(calendar, "_execute_step", execute)
    if failure_at == "finish":
        monkeypatch.setattr(calendar, "report", fail_report)
    if failure_at == "startup":
        directory.mkdir(mode=0o700)
        (directory / "state.json").write_text('{"deferrals": null}')

    def now():
        if failure_at == "post-execute" and executed:
            raise ValueError("private-value")
        return dt.datetime(2026, 10, 22, 3, 46)

    run = run_calendar(
        [calendar.Step(name, ("/unused",), 60) for name in ("index", "jsonl-backup")],
        directory,
        path / "logs",
        now_fn=now,
    )
    if failure_at in {"protocol", "post-execute"}:
        assert asyncio.run(run) == 1
    else:
        with pytest.raises(ValueError if failure_at == "startup" else KeyError):
            asyncio.run(run)
    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "summary"
    assert rows[-1]["failed"] == (2 if failure_at == "after-failure" else 1)
    # Every runner error names its attempt; a startup failure has no header to inherit it from.
    errors = [r for r in rows if r["event"] == "runner-error"]
    if failure_at == "startup":
        assert not any(r["event"] == "runner" for r in rows) and errors
    assert all((r["scheduled_date"], r["kind"]) == (rows[-1]["scheduled_date"], rows[-1]["kind"]) for r in errors)
    assert "private-value" not in (directory / "runs.jsonl").read_text()
    assert executed == (
        [] if failure_at == "startup" else ["index", "jsonl-backup"] if failure_at == "after-failure" else ["index"]
    )
    with calendar.Custody(directory):
        pass


def test_index_failure_preserves_all_independent_nightly_steps(request, monkeypatch):
    from brainlayer import maintenance_calendar as calendar

    path, _, _, _ = request.getfixturevalue("control")
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(path / "fixture.env"))
    (path / "fixture.env").write_text("")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(path / "alerts.json"))
    clock = [dt.datetime(2026, 10, 22, 3, 46)]
    executed = []

    async def execute(step, env, **kwargs):
        executed.append(step.name)
        if step.name == "index":
            clock[0] = clock[0].replace(hour=4, minute=0)
            return 124
        return 0

    monkeypatch.setattr(calendar, "_execute_step", execute)
    plan = calendar_steps(sys.executable, "/unused", path, clock[0].date())
    assert (
        asyncio.run(
            run_calendar(plan.select(clock[0].date()), path / "calendar", path / "logs", now_fn=lambda: clock[0])
        )
        == 1
    )
    assert executed == ["index", "maintenance-nightly", "jsonl-backup", "p0-counter", "wal-checkpoint", "backup-daily"]
    rows = [json.loads(line) for line in (path / "calendar" / "runs.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "summary" and rows[-1]["failed"] == 1


@pytest.mark.parametrize("last_weekly", [None, dt.date(2026, 10, 11)])
def test_plan_keeps_weekly_receipt_dependency_and_nightly_upload_last(tmp_path, last_weekly):
    plan = calendar_steps("/fixture/python", "/fixture/cli", tmp_path, dt.date(2026, 10, 12))
    names = [step.name for step in plan.select(last_weekly)]
    weekly = last_weekly is None
    assert names == [
        "index",
        *(["backup-snapshot"] if weekly else []),
        "maintenance-weekly" if weekly else "maintenance-nightly",
        "jsonl-backup",
        "p0-counter",
        *(["decay", "repair-fts"] if weekly else []),
        "wal-checkpoint",
        *(["backup-daily"] if weekly else ["backup-daily"]),
    ]
    by_name = {step.name: step for step in plan.candidates}
    assert by_name["backup-daily"].argv[-1] == str(tmp_path / "backup-daily.sh")
    assert by_name["backup-daily"].timeout == 28890
    assert by_name["backup-snapshot"].env["BRAINLAYER_BACKUP_PHASE"] == "snapshot"
    assert by_name["jsonl-backup"].argv[-1] == str(tmp_path / "jsonl-backup.sh")
    assert by_name["jsonl-backup"].env["BRAINLAYER_JSONL_BACKUP_DRIVE_FOLDER"].endswith("/claude-jsonl")
    for name, mode in [("maintenance-nightly", "--light"), ("maintenance-weekly", "--full")]:
        assert by_name[name].argv[1:] == ("/fixture/python", "-m", "brainlayer.maintenance", mode)


@pytest.mark.parametrize("receipt", ["recent", "fresh", "unverified"])
@pytest.mark.parametrize("finish", [(3, 50), (4, 40), (6, 10)])
def test_sunday_full_waits_runs_real_vacuum_or_alerts_overrun(request, monkeypatch, finish, receipt):
    import sqlite3

    from brainlayer import maintenance, maintenance_calendar

    path, _, _, _ = request.getfixturevalue("control")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(path / "alerts.json"))
    db = path / "copy.db"
    with sqlite3.connect(db) as c:
        c.execute("create table padding(value blob)")
        c.executemany("insert into padding values(?)", [(b"x" * 1000,)] * 3000)
        c.execute("delete from padding")
    before = db.stat().st_size
    clock = [dt.datetime(2026, 10, 11, *finish)]
    seen = []
    backups = []

    snapshot = path / "verified.db.gz"
    snapshot.write_bytes(b"synthetic verified snapshot")
    from brainlayer.backup_daily import _upload_identity

    local_receipt = dict(local_verified=True, snapshot=str(snapshot), local_artifact=_upload_identity(snapshot))

    async def sleep(seconds):
        clock[0] += dt.timedelta(seconds=seconds)
        await asyncio.sleep(0)

    async def execute(step, env, **kwargs):
        seen.append(step.name)
        if step.name == "backup-snapshot":
            return int(receipt == "unverified")
        if step.name == "maintenance-weekly":
            cfg = maintenance.MaintenanceConfig(
                db_path=db,
                queue_dir=path / "queue",
                quarantine_root=path / "quarantine",
                log_path=path / "maint.log",
                backup_staging_dir=path / "staging",
                backup_log_path=path / "backup.log",
                now_fn=lambda: clock[0],
                idle_sample_seconds=0,
                recent_write_grace_seconds=0,
            )
            old = os.environ.copy()
            os.environ.update(env)
            try:
                if receipt == "unverified":
                    with pytest.raises(
                        maintenance.MaintenanceAbort, match="local verified snapshot unavailable"
                    ) as exc:
                        maintenance.run_maintenance("full", config=cfg)
                    assert exc.value.code == 75
                    assert db.stat().st_size == before
                    return exc.value.code
                result = maintenance.run_maintenance("full", config=cfg)
                assert (
                    result.backup_status == "local_verified" and result.vacuum_after_bytes < result.vacuum_before_bytes
                )
            finally:
                os.environ.clear()
                os.environ.update(old)
        return 0

    monkeypatch.setattr(maintenance_calendar, "_execute_step", execute, raising=False)
    for name in (
        "_assert_running_merged_code",
        "_check_idle",
        "_check_lsof_clean",
        "_verify_search_latency",
        "_emit_telemetry",
    ):
        monkeypatch.setattr(maintenance, name, lambda *_: None)
    monkeypatch.setattr(
        maintenance, "_recent_weekly_verified_backup", lambda *_: local_receipt if receipt != "unverified" else None
    )
    monkeypatch.setattr(maintenance, "_run_bounded_weekly_backup", lambda *_: pytest.fail("second snapshot"))
    plan = calendar_steps(sys.executable, "/unused/cli", path, dt.date(2026, 10, 11))
    rejected = receipt == "unverified" and finish[0] < 6
    assert asyncio.run(
        run_calendar(plan, path / "calendar", path / "logs", now_fn=lambda: clock[0], sleep_fn=sleep)
    ) == int(rejected)
    assert ("maintenance-weekly" in seen) == (finish[0] < 6)
    assert (db.stat().st_size < before) == (finish[0] < 6 and not rejected)
    assert backups == []
    assert seen[-1] == "backup-daily"
    assert ("maintenance-full" in job_alerts.active_alerts()) == (finish[0] >= 6)
    events = [json.loads(l) for l in (path / "calendar" / "runs.jsonl").read_text().splitlines()]
    assert any(e["event"] == "wait-start" for e in events) == (finish[0] < 4)
    if finish[0] < 6 and not rejected:
        assert json.loads((path / "calendar" / "state.json").read_text())["last_weekly"] == "2026-10-11"


@pytest.mark.parametrize("weekly", [True, False])
@pytest.mark.parametrize("late_step", ["index", "backup-daily", None])
def test_calendar_records_actual_window_blocker_and_weekly_substitution(request, monkeypatch, late_step, weekly):
    from brainlayer import maintenance_calendar

    path, _, _, _ = request.getfixturevalue("control")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(path / "alerts.json"))
    clock = [dt.datetime(2026, 10, 11, 3, 15)]
    seen = []
    directory = path / "calendar"
    if not weekly:
        directory.mkdir(mode=0o700)
        (directory / "state.json").write_text(json.dumps(dict(deferrals={}, last_weekly="2026-10-11")))

    async def execute(step, env, **kwargs):
        seen.append((step.name, clock[0]))
        if step.name == ("backup-snapshot" if weekly and late_step == "backup-daily" else late_step):
            clock[0] += dt.timedelta(hours=4)
            if late_step == "backup-daily" and not weekly:
                assert all(name in dict(seen) for name in ("maintenance-nightly", "jsonl-backup", "wal-checkpoint"))
        return 0

    async def sleep(seconds):
        clock[0] = clock[0].replace(hour=4, minute=0)
        await asyncio.sleep(0)

    monkeypatch.setattr(maintenance_calendar, "_execute_step", execute)
    plan = calendar_steps(sys.executable, "/unused/cli", path, dt.date(2026, 10, 11))
    assert (
        asyncio.run(run_calendar(plan, path / "calendar", path / "logs", now_fn=lambda: clock[0], sleep_fn=sleep)) == 0
    )
    rows = [json.loads(line) for line in (path / "calendar" / "runs.jsonl").read_text().splitlines()]
    name = "maintenance-weekly" if weekly else "maintenance-nightly"
    result = next(row for row in rows if row["event"] == "finished" and row["step"] == name)
    assert result.get("replaces") == ("maintenance-nightly" if weekly else None)
    missed = late_step == "index" or (weekly and late_step == "backup-daily")
    reason = "index-overran-window" if late_step == "index" else "quiet-window-missed" if missed else None
    assert result.get("reason") == reason
    assert result["status"] == ("deferred" if missed else "ok")
    if late_step == "backup-daily" and not weekly:
        assert dict(seen)[name].hour == 4
        assert dict(seen)["jsonl-backup"].hour == 4
        assert clock[0].hour == 8


@pytest.mark.parametrize("existing_receipt", ["yesterday", "weekly-safety"])
def test_slow_daily_backup_and_weekly_share_one_verified_upload(request, monkeypatch, existing_receipt):
    import sqlite3

    from brainlayer import backup_daily, maintenance, maintenance_calendar

    path, _, _, _ = request.getfixturevalue("control")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(path / "alerts.json"))
    clock = [dt.datetime(2026, 10, 11, 3, 15, tzinfo=dt.UTC)]
    db, log, staging = path / "copy.db", path / "backup.log", path / "staging"
    with sqlite3.connect(db) as connection:
        connection.execute("create table padding(value blob)")
        connection.executemany("insert into padding values(?)", [(b"x" * 1000,)] * 3000)
        connection.execute("delete from padding")
    before = db.stat().st_size
    prior = dict(
        attempted_at=(clock[0] - dt.timedelta(hours=22 if existing_receipt == "yesterday" else 1)).isoformat(),
        snapshot_created_at=(clock[0] - dt.timedelta(hours=22 if existing_receipt == "yesterday" else 1)).isoformat(),
        db=str(db),
        uploaded=True,
        verified=True,
        drive_file={"id": existing_receipt},
    )
    log.write_text(json.dumps(prior) + "\n")

    class ClockDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz)

    uploads, seen = [], []
    snapshot = path / "2026-10-11.db.gz"
    snapshot.write_bytes(b"synthetic verified snapshot")
    local = dict(
        db=str(db),
        snapshot=str(snapshot),
        local_verified=True,
        local_integrity="ok",
        local_integrity_pragma="integrity_check",
        uploaded=False,
        verified=False,
        local_artifact=backup_daily._upload_identity(snapshot),
        snapshot_created_at=clock[0].isoformat(),
    )

    cfg = maintenance.MaintenanceConfig(
        db_path=db,
        queue_dir=path / "queue",
        quarantine_root=path / "quarantine",
        log_path=path / "maint.log",
        backup_staging_dir=staging,
        backup_log_path=log,
        now_fn=lambda: clock[0],
        idle_sample_seconds=0,
        recent_write_grace_seconds=0,
    )

    async def execute(step, env, **kwargs):
        seen.append((step.name, clock[0]))
        with monkeypatch.context() as child:
            for key, value in env.items():
                child.setenv(key, value)
            if step.name == "backup-snapshot":
                log.write_text(json.dumps(local) + "\n")
            elif step.name == "backup-daily":
                assert db.stat().st_size < before
                uploads.append(snapshot)
                clock[0] += dt.timedelta(hours=4)
            elif step.name == "maintenance-weekly":
                result = maintenance.run_maintenance("full", config=cfg)
                assert result.backup_status == "local_verified"
                assert f"verified_local_backup={snapshot}" in result.actions
        return 0

    async def sleep(seconds):
        clock[0] = clock[0].replace(hour=4, minute=0)
        await asyncio.sleep(0)

    for name in (
        "_assert_running_merged_code",
        "_check_idle",
        "_check_lsof_clean",
        "_verify_search_latency",
        "_emit_telemetry",
    ):
        monkeypatch.setattr(maintenance, name, lambda *_: None)
    monkeypatch.setattr(
        maintenance, "_run_bounded_weekly_backup", lambda *_: pytest.fail("second upload requested by weekly")
    )
    monkeypatch.setattr(maintenance_calendar, "_execute_step", execute)
    plan = calendar_steps(sys.executable, "/unused/cli", path, clock[0].date())
    assert (
        asyncio.run(run_calendar(plan, path / "calendar", path / "logs", now_fn=lambda: clock[0], sleep_fn=sleep)) == 0
    )
    assert [name for name, _ in seen] == [
        "index",
        "backup-snapshot",
        "maintenance-weekly",
        "jsonl-backup",
        "p0-counter",
        "decay",
        "repair-fts",
        "wal-checkpoint",
        "backup-daily",
    ]
    assert uploads == [snapshot]
    assert dict(seen)["maintenance-weekly"].hour == 4
    assert dict(seen)["backup-daily"].hour == 4
    assert db.stat().st_size < before  # Real fixture VACUUM after verified receipt reuse.
    rows = [json.loads(line) for line in cfg.log_path.read_text().splitlines()]
    assert any(row.get("backup_status") == "local_verified" for row in rows)
    assert json.loads((path / "calendar" / "state.json").read_text())["last_weekly"] == "2026-10-11"


@pytest.mark.parametrize(
    "message,code,expected",
    [
        ("INDEX_DEFERRED resident_writer_busy committed_chunks=0", 75, "resident-writer-busy"),
        ("INDEX_DEFERRED maintenance_window=04:00-06:00", 75, "maintenance-window"),
        ("INDEX_DEFERRED invented", 75, None),
        ("INDEX_DEFERRED resident_writer_busy", 1, None),
        ("", 75, None),
        ("INDEX_DEFERRED ", 75, None),
    ],
)
def test_index_deferral_reason_is_current_run_and_allowlisted(request, monkeypatch, message, code, expected):
    from brainlayer import maintenance_calendar
    from brainlayer.maintenance_calendar import Step

    path, _, _, _ = request.getfixturevalue("control")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(path / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(path / "fixture.env"))
    (path / "fixture.env").write_text("")
    logs = path / "logs"
    logs.mkdir()
    (logs / "index.out.log").write_text("INDEX_DEFERRED resident_writer_busy\n")

    async def execute(step, env, **kwargs):
        with (logs / "index.out.log").open("a") as handle:
            handle.write(message + "\n")
        return code

    monkeypatch.setattr(maintenance_calendar, "_execute_step", execute)
    result = asyncio.run(
        run_calendar(
            [Step("index", (sys.executable, "-c", "pass"), 60)],
            path / "calendar",
            logs,
            now_fn=lambda: dt.datetime(2026, 10, 12, 3, 15),
        )
    )
    assert result == int(code == 1)
    row = next(
        json.loads(line)
        for line in (path / "calendar" / "runs.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "finished"
    )
    assert row.get("reason") == expected


@pytest.mark.parametrize("cancel", [False, True])
def test_slow_spawn_receipt_keeps_guard_live_without_early_child_exec(tmp_path, monkeypatch, cancel):
    from brainlayer import maintenance_calendar as calendar

    entered, release = threading.Event(), threading.Event()
    events, pulses, observations = [], [], []
    directory, logs, marker = tmp_path / "calendar", tmp_path / "logs", tmp_path / "executed"
    logs.mkdir()
    read_fd, write_fd = os.pipe()
    guard = SimpleNamespace(stdin=os.fdopen(write_fd, "wb", buffering=0), poll=lambda: None)
    original_record, original_write = calendar.record, os.write

    def record(path, **event):
        if event.get("event") == "spawn":
            events.append("receipt-enter")
            entered.set()
            assert release.wait(3)
            original_record(path, **event)
            events.append("durable-receipt")
        else:
            original_record(path, **event)

    def write(fd, data):
        if fd == write_fd and data == b"H\n" and entered.is_set():
            pulses.append(data)
        elif fd == write_fd and data.startswith(b"+"):
            events.append("registered")
        elif data == b"R":
            events.append("gate-release")
        return original_write(fd, data)

    monkeypatch.setattr(calendar, "record", record)
    monkeypatch.setattr(os, "write", write)

    # A watchdog thread releases even the old synchronous implementation, so red cannot hang.
    def observe():
        assert entered.wait(3)
        release.wait(0.5)  # Longer than the private 0.2 s heartbeat budget.
        observations.append((len(pulses), marker.exists()))
        release.set()

    observer = threading.Thread(target=observe)
    observer.start()

    async def scenario():
        stop, heartbeat_stop = asyncio.Event(), asyncio.Event()
        heartbeat = asyncio.create_task(calendar.pulse(guard, heartbeat_stop, 0.02))

        async def run():
            with calendar.Custody(directory) as custody:
                return await calendar._execute_step(
                    calendar.Step(
                        "jsonl-backup", (sys.executable, "-c", f"open({str(marker)!r},'a').write('once\\n')"), 2
                    ),
                    os.environ.copy(),
                    directory=directory,
                    logs=logs,
                    custody=custody,
                    guard=guard,
                    stop=stop,
                    guard_timeout=0.2,
                    shutdown_timeout=2,
                )

        task = asyncio.create_task(run())
        try:
            while not entered.is_set():
                await asyncio.sleep(0.01)
            if cancel:
                task.cancel()
                await asyncio.sleep(0.05)
                assert not task.done()
                with pytest.raises(BlockingIOError), calendar.Custody(directory):
                    pass
                task.cancel()  # Repeated cancellation must still drain the worker.
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                assert await task == 0
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            heartbeat_stop.set()
            await heartbeat

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        observer.join(3)
        guard.stdin.close()
        os.close(read_fd)
    assert observations and observations[0][0] >= 3, "receipt I/O starved the heartbeat loop"
    assert not observations[0][1], "child executed before its durable receipt"
    if cancel:
        assert "registered" not in events and "gate-release" not in events and not marker.exists()
    else:
        assert events.index("durable-receipt") < events.index("registered") < events.index("gate-release")
        assert marker.read_text() == "once\n"
    with calendar.Custody(directory):
        pass


def test_real_guard_kills_permanently_stuck_calendar_worker_and_reaps(tmp_path):
    import select
    import signal
    import time

    from brainlayer.supervisor_guard import group_exists

    code = r"""
import asyncio,datetime as dt,inspect,os,sys,time
from pathlib import Path
from types import SimpleNamespace
from brainlayer import maintenance_calendar as calendar
p=Path(sys.argv[1]); marker=p/'child-ready'
calendar.maintenance_control.control_client=lambda: SimpleNamespace(path=p/'unused.sock',request=lambda:{},_disabled=lambda *a,**kw:set())
execute=calendar._execute_step
def stuck():
 while True: time.sleep(.05)
async def wedged(*args,**kwargs):
 child=asyncio.create_task(execute(*args,**kwargs))
 while not marker.exists() or not marker.read_text():
  if child.done(): await child; raise RuntimeError('child stopped before wedge')
  await asyncio.sleep(.01)
 print(marker.read_text(),flush=True)
 await calendar._blocking(stuck)
 await child
calendar._execute_step=wedged
step=calendar.Step('jsonl-backup',(sys.executable,'-c',f"import os,time; open({str(marker)!r},'w').write(str(os.getpid())); time.sleep(60)"),60)
options={'operation_timeout':.5} if 'operation_timeout' in inspect.signature(calendar.run_calendar).parameters else {}
asyncio.run(calendar.run_calendar([step],p/'calendar',p/'logs',guard_timeout=.4,shutdown_timeout=1,now_fn=lambda:dt.datetime(2026,10,27,5),**options))
"""
    errors = (tmp_path / "runner.err").open("wb")
    parent = subprocess.Popen((sys.executable, "-c", code, str(tmp_path)), stdout=subprocess.PIPE, stderr=errors)
    group = None
    try:
        assert select.select([parent.stdout], [], [], 5)[0]
        line = parent.stdout.readline()
        assert line, (tmp_path / "runner.err").read_text()
        group = int(line)
        assert parent.wait(timeout=3) == -signal.SIGKILL
        deadline = time.monotonic() + 3
        while group_exists(group) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not group_exists(group)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=2)
        if group:
            deadline = time.monotonic() + 3
            while group_exists(group) and time.monotonic() < deadline:
                time.sleep(0.02)
            if group_exists(group):
                os.killpg(group, signal.SIGKILL)
        errors.close()


def test_backup_budget_failure_never_spawns_a_child(tmp_path, monkeypatch):
    from brainlayer import maintenance_calendar as calendar

    spawned = []
    spawn = calendar.gated_spawn

    def tracked(*args, **kwargs):
        spawned.append(True)
        return spawn(*args, **kwargs)

    def invalid(step):
        raise ValueError("invalid backup timeout")

    monkeypatch.setattr(calendar, "gated_spawn", tracked)
    monkeypatch.setattr(calendar, "backup_deadline", invalid)
    logs = tmp_path / "logs"
    logs.mkdir()
    read, write = os.pipe()
    guard = SimpleNamespace(stdin=os.fdopen(write, "wb", buffering=0), poll=lambda: None)
    try:
        with calendar.Custody(tmp_path / "calendar") as custody:
            with pytest.raises(ValueError, match="invalid backup timeout"):
                asyncio.run(
                    calendar._execute_step(
                        calendar.Step("backup-daily", (sys.executable, "-c", "pass"), 2),
                        os.environ.copy(),
                        directory=custody.directory,
                        logs=logs,
                        custody=custody,
                        guard=guard,
                        stop=asyncio.Event(),
                        guard_timeout=0.4,
                        shutdown_timeout=1,
                    )
                )
        assert not spawned
    finally:
        guard.stdin.close()
        os.close(read)


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_calendar_rejects_invalid_operation_budget(tmp_path, budget):
    with pytest.raises(ValueError, match="invalid calendar timing"):
        asyncio.run(run_calendar([], tmp_path / "calendar", tmp_path / "logs", operation_timeout=budget))


def test_overdue_calendar_operation_emits_once_before_suppressing_heartbeat(monkeypatch):
    import time

    from brainlayer import maintenance_calendar as calendar

    emitted, writes = [], []
    operation = calendar.CalendarOperation("blocked-receipt", time.monotonic() - 1)

    async def scenario():
        stop = asyncio.Event()
        ticks = []

        async def sleep(interval):
            ticks.append(True)
            if len(ticks) == 2:
                stop.set()

        monkeypatch.setattr(calendar.asyncio, "sleep", sleep)
        monkeypatch.setattr(calendar.os, "write", lambda *args: writes.append(args))
        guard = SimpleNamespace(pid=1234, poll=lambda: None, stdin=SimpleNamespace(fileno=lambda: 999))
        await calendar.pulse(
            guard, stop, 0.01, {object(): operation}, 0.5, SimpleNamespace(emit=lambda **row: emitted.append(row))
        )

    asyncio.run(scenario())
    assert not writes and len(emitted) == 1
    row = emitted[0]
    assert row["event"] == "heartbeat-suppressed" and row["reason"] == "operation-overdue"
    assert row["operation"] == "blocked-receipt" and row["budget_s"] == 0.5
    assert row["deadline_monotonic"] == operation.started + 0.5


def test_calendar_guard_timeout_defaults_to_120_and_reads_override(tmp_path, monkeypatch):
    from brainlayer import maintenance_calendar as calendar

    seen = []

    async def admit(steps, directory, date, recovery, guard_timeout, shutdown_timeout, sleep_fn):
        seen.append(guard_timeout)
        return None, 0

    monkeypatch.setattr(calendar, "_admit_calendar", admit)
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "brainlayer.env"))
    monkeypatch.delenv("BRAINLAYER_SUPERVISOR_GUARD_TIMEOUT_SECONDS", raising=False)
    when = dt.date(2026, 10, 28)
    assert asyncio.run(run_calendar([], tmp_path / "calendar", tmp_path / "logs", scheduled_date=when)) == 0
    (tmp_path / "brainlayer.env").write_text("BRAINLAYER_SUPERVISOR_GUARD_TIMEOUT_SECONDS=600\n")
    assert asyncio.run(run_calendar([], tmp_path / "calendar", tmp_path / "logs", scheduled_date=when)) == 0
    assert seen == [120, 600]
    (tmp_path / "brainlayer.env").write_text("BRAINLAYER_SUPERVISOR_GUARD_TIMEOUT_SECONDS=5\n")
    with pytest.raises(ValueError, match="BRAINLAYER_SUPERVISOR_GUARD_TIMEOUT_SECONDS"):
        asyncio.run(run_calendar([], tmp_path / "calendar", tmp_path / "logs", scheduled_date=when))


def test_busy_maintenance_hold_defers_the_step_and_the_backup_still_runs(tmp_path, monkeypatch):
    """M4 2026-11-07: the hold flock was busy at 04:00, BlockingIOError aborted the plan and
    backup-daily never ran. A busy hold is a deferral, not a custody failure."""
    from brainlayer import maintenance_calendar as calendar

    tmp_path.chmod(0o700)
    directory = tmp_path / "calendar"
    directory.mkdir(mode=0o700)
    (directory / "state.json").write_text(json.dumps(dict(deferrals={}, last_weekly="2026-11-01")))
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "fixture.env"))
    monkeypatch.setattr(
        calendar.maintenance_control,
        "control_client",
        lambda: SimpleNamespace(path=tmp_path / "unused.sock", request=lambda: {}, _disabled=lambda *_a, **_k: set()),
    )

    class BusyHold:
        fd, token, deadline = None, "", None

        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            raise BlockingIOError(35, "Resource temporarily unavailable")

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(calendar, "MaintenanceHold", BusyHold)
    executed = []

    async def execute(step, env, **kwargs):
        executed.append(step.name)
        return 0

    monkeypatch.setattr(calendar, "_execute_step", execute)
    clock = [dt.datetime(2026, 11, 7, 3, 45)]

    async def advance(_seconds):
        clock[0] = clock[0].replace(hour=4)

    plan = calendar_steps(sys.executable, "/unused", tmp_path, clock[0].date())
    asyncio.run(run_calendar(plan, directory, tmp_path / "logs", now_fn=lambda: clock[0], sleep_fn=advance))
    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    finished = {row["step"]: row for row in rows if row["event"] == "finished"}
    assert finished["maintenance-nightly"]["status"] == "deferred", finished["maintenance-nightly"]
    assert finished["maintenance-nightly"].get("reason") == "maintenance-hold-busy"
    assert "backup-daily" in executed, executed
    assert not any(row["event"] == "step-error" for row in rows)


@pytest.mark.parametrize(
    "failure,timeouts,expected_calls,succeeds",
    [
        (TimeoutError, 1, 3, True),
        (TimeoutError, 2, 4, True),
        (TimeoutError, 3, 5, True),
        *[
            (error, 2, 4, True)
            for error in (ConnectionRefusedError, ConnectionResetError, BrokenPipeError, FileNotFoundError)
        ],
        (PermissionError, 1, 1, False),
        (ValueError, 2, 4, True),
        (RuntimeError, 1, 1, False),
    ],
)
def test_calendar_status_timeout_is_bounded_without_skipping_backup(
    tmp_path, monkeypatch, failure, timeouts, expected_calls, succeeds
):
    """M4 Nov 10: the status read before decay timed out; upload never spawned."""
    from brainlayer import maintenance_calendar as calendar

    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "fixture.env"))

    async def no_wait(_seconds):
        pass

    calls = []

    def status():
        calls.append("status")
        if len(calls) <= timeouts:
            raise failure("private error detail")
        return {}

    monkeypatch.setattr(
        calendar.maintenance_control,
        "control_client",
        lambda: SimpleNamespace(path=tmp_path / "unused.sock", request=status, _disabled=lambda *_a, **_k: set()),
    )
    marker = tmp_path / "executed"
    steps = [
        calendar.Step(
            name,
            (
                sys.executable,
                "-c",
                f"from pathlib import Path; p=Path({str(marker)!r}); p.open('a').write({name + chr(10)!r})",
            ),
            10,
        )
        for name in ("decay", "backup-daily")
    ]
    directory = tmp_path / "calendar"
    result = asyncio.run(
        run_calendar(
            steps, directory, tmp_path / "logs", now_fn=lambda: dt.datetime(2026, 11, 8, 5, 20), sleep_fn=no_wait
        )
    )
    assert result == (0 if succeeds else 1)
    assert len(calls) == expected_calls
    assert (marker.read_text().splitlines() if marker.exists() else []) == (
        ["decay", "backup-daily"] if succeeds else []
    )
    journal = (directory / "runs.jsonl").read_text()
    rows = [json.loads(line) for line in journal.splitlines()]
    finished = {row["step"]: row["status"] for row in rows if row["event"] == "finished"}
    assert finished == (
        {"decay": "ok", "backup-daily": "ok"} if succeeds else {"decay": "failed", "backup-daily": "skipped"}
    )
    retries = [row for row in rows if row["event"] in {"supervisor-status-timeout", "supervisor-status-unavailable"}]
    assert [row["attempt"] for row in retries] == (list(range(1, timeouts + 1)) if succeeds else [])
    assert all(row["step"] == "decay" for row in retries)
    assert all(
        row["event"] == ("supervisor-status-timeout" if failure is TimeoutError else "supervisor-status-unavailable")
        and row["error"] == failure.__name__
        for row in retries
    )
    assert "private error detail" not in journal
    with calendar.Custody(directory):
        pass


def test_busy_inherited_maintenance_lease_still_aborts(monkeypatch):
    """Review of #1668: only a FRESH busy hold defers; failing to re-lock an inherited lease is custody."""
    from brainlayer import maintenance_calendar as calendar

    class BusyHold:
        fd, token, deadline = None, "", None

        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            raise BlockingIOError(35, "Resource temporarily unavailable")

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(calendar, "MaintenanceHold", BusyHold)
    step = SimpleNamespace(name="maintenance-nightly")

    async def enter(inherited):
        if inherited:
            monkeypatch.setenv("BRAINLAYER_MAINTENANCE_FD", "9")
        else:
            monkeypatch.delenv("BRAINLAYER_MAINTENANCE_FD", raising=False)
        async with calendar.step_hold(None, None, {}, step):
            pass

    with pytest.raises(calendar.HoldBusy):
        asyncio.run(enter(False))
    with pytest.raises(BlockingIOError) as raised:
        asyncio.run(enter(True))
    assert not isinstance(raised.value, calendar.HoldBusy)


@pytest.mark.parametrize("outage,interrupt", [(8, False), (float("inf"), False), (float("inf"), True)])
def test_status_outage_defers_safely_and_stop_interrupts_backoff(tmp_path, monkeypatch, outage, interrupt):
    from brainlayer import maintenance_calendar as calendar

    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "fixture.env"))
    elapsed = [0.0]
    real_event = asyncio.Event
    events = []

    def event():
        value = real_event()
        events.append(value)
        return value

    monkeypatch.setattr(calendar.asyncio, "Event", event)

    async def advance(seconds):
        elapsed[0] += seconds
        if interrupt:
            events[0].set()  # The runner's stop event, before the heartbeat event.
        await asyncio.sleep(0)

    def status():
        if elapsed[0] < outage:
            raise TimeoutError("private detail")
        return {}

    monkeypatch.setattr(
        calendar.maintenance_control,
        "control_client",
        lambda: SimpleNamespace(path=tmp_path / "unused.sock", request=status, _disabled=lambda *_a, **_k: set()),
    )
    marker = tmp_path / "children"
    steps = [
        calendar.Step(name, (sys.executable, "-c", f"open({str(marker)!r}, 'a').write({name!r}+'\\n')"), 10)
        for name in ("decay", "backup-daily")
    ]
    directory = tmp_path / "calendar"
    result = asyncio.run(
        run_calendar(
            steps, directory, tmp_path / "logs", sleep_fn=advance, now_fn=lambda: dt.datetime(2026, 11, 8, 5, 20)
        )
    )
    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    finished = [row for row in rows if row["event"] == "finished"]
    if outage == 8:
        assert result == 0
        assert marker.read_text().splitlines() == ["decay", "backup-daily"]
    elif interrupt:
        assert result == 1
        assert not marker.exists()
        assert elapsed[0] < 1
    else:
        assert result == 0
        assert not marker.exists()
        assert len(finished) == 2
        assert all(row["status"] == "deferred" and row["reason"] == "supervisor-status-unavailable" for row in finished)
        assert rows[-1]["failed"] == 0
        assert elapsed[0] == pytest.approx(180)


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError,
        ConnectionRefusedError,
        ConnectionResetError,
        BrokenPipeError,
        FileNotFoundError,
        ValueError,
        lambda message: json.JSONDecodeError(message, "{", 1),
    ],
)
@pytest.mark.parametrize("outage,interrupt", [(8, False), (float("inf"), False), (float("inf"), True)])
def test_runner_start_status_backoff_at_real_adapter_seam(tmp_path, monkeypatch, outage, interrupt, failure):
    """M1 Nov 8 failed inside adapt_commands, before any calendar step could start."""
    import signal
    from contextlib import asynccontextmanager
    from dataclasses import replace

    from brainlayer import maintenance_calendar as calendar
    from brainlayer.supervisor_adapters import SupervisorCommands

    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "fixture.env"))
    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_SOCKET", str(tmp_path / "unused.sock"))
    elapsed, requests, callbacks = [0.0], [], {}

    def request(self, action="status", component="*", *, timeout=2, owner=""):
        assert action == "status" and component == "*" and timeout == 2 and owner == ""
        requests.append(elapsed[0])
        if elapsed[0] < outage:
            elapsed[0] += timeout  # The real socket's timeout, on a private deterministic clock.
            raise failure("private supervisor detail")
        return {}

    monkeypatch.setattr(SupervisorCommands, "request", request)
    monkeypatch.setattr(SupervisorCommands, "_disabled", lambda *_a, **_kw: set())

    @asynccontextmanager
    async def hold(_client, _guard, env, _step, _now):
        yield env  # Hold mutations are outside this status-read regression.

    monkeypatch.setattr(calendar, "step_hold", hold)

    async def advance(seconds):
        elapsed[0] += seconds
        if interrupt:
            callbacks[signal.SIGTERM]()
        await asyncio.sleep(0)

    marker = tmp_path / "children"
    now = dt.datetime(2026, 11, 10, 5, 20)
    plan = calendar.calendar_steps(sys.executable, "/unused", tmp_path, now.date())
    plan = replace(
        plan,
        candidates=tuple(
            replace(step, argv=(sys.executable, "-c", f"open({str(marker)!r}, 'a').write({step.name!r}+'\\n')"))
            for step in plan.candidates
        ),
    )
    directory = tmp_path / "calendar"
    directory.mkdir(mode=0o700)
    (directory / "state.json").write_text(json.dumps(dict(deferrals={}, last_weekly="2026-11-08")))
    names = [step.name for step in plan.select(dt.date(2026, 11, 8))]

    async def run():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "add_signal_handler", lambda sig, cb: callbacks.__setitem__(sig, cb))
        monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: callbacks.pop(sig, None))
        return await calendar.run_calendar(
            plan, directory, tmp_path / "logs", now_fn=lambda: now, sleep_fn=advance, install_signals=True
        )

    result = asyncio.run(run())
    journal = (directory / "runs.jsonl").read_text()
    rows = [json.loads(line) for line in journal.splitlines()]
    assert not any(row["event"] == "runner-error" for row in rows)
    assert "private supervisor detail" not in journal
    retries = [row for row in rows if row["event"] in {"supervisor-status-timeout", "supervisor-status-unavailable"}]
    assert all(row.get("phase") == "runner-start" for row in retries)
    assert all(
        row["event"] == ("supervisor-status-timeout" if failure is TimeoutError else "supervisor-status-unavailable")
        for row in retries
    )
    assert all(
        row["error"] == ("JSONDecodeError" if not isinstance(failure, type) else failure.__name__) for row in retries
    )
    if outage == 8:
        assert result == 0
        assert marker.read_text().splitlines() == [name for name in names if name != "index"]
        assert requests[:3] == pytest.approx([0, 4, 10])
        assert len(requests) == 3 + len(names) - 1  # Index is outside its pre-04:00 window.
        assert [row["attempt"] for row in retries] == [1, 2]
    elif interrupt:
        assert result == 1
        assert not marker.exists()
        assert len(requests) == 1
        assert elapsed[0] < 3
        assert rows[-1]["complete"] is False
    else:
        assert result == 0
        assert not marker.exists()
        assert len(requests) == 7
        assert elapsed[0] == pytest.approx(104)
        assert [row["attempt"] for row in retries] == list(range(1, 8))
        finished = [row for row in rows if row["event"] == "finished"]
        assert [row["step"] for row in finished] == names
        assert all(row["status"] == "deferred" and row["reason"] == "supervisor-status-unavailable" for row in finished)
        assert rows[-1]["failed"] == 0 and rows[-1]["deferred"] == len(names)
        selection = calendar.failed_recovery_selection(directory)
        assert [row["step"] for row in selection["failures"]] == names
        assert "jsonl-backup" in selection["steps"]
        assert "backup-daily" in [row["step"] for row in selection["deferred"]]
        outage = 0
        now += dt.timedelta(days=1)
        plan = replace(plan, scheduled_date=now.date())
        assert asyncio.run(run()) == 0
        assert marker.read_text().splitlines() == [name for name in names if name != "index"]
        assert set(json.loads((directory / "state.json").read_text())["deferrals"]) == {calendar.KEYS["index"]}
    with calendar.Custody(directory):
        pass


@pytest.mark.parametrize("failure", [PermissionError, RuntimeError])
def test_runner_start_status_custody_and_protocol_errors_still_abort(tmp_path, monkeypatch, failure):
    from brainlayer import maintenance_calendar as calendar
    from brainlayer.supervisor_adapters import SupervisorCommands

    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_SOCKET", str(tmp_path / "unused.sock"))
    calls = []

    def request(*_a, **_kw):
        calls.append("status")
        raise failure("private supervisor detail")

    monkeypatch.setattr(SupervisorCommands, "request", request)
    marker = tmp_path / "child"
    step = calendar.Step("backup-daily", (sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"), 10)
    directory = tmp_path / "calendar"
    with pytest.raises(failure):
        asyncio.run(calendar.run_calendar([step], directory, tmp_path / "logs"))
    assert calls == ["status"]
    assert not marker.exists()
    journal = (directory / "runs.jsonl").read_text()
    assert "runner-error" in journal and failure.__name__ in journal
    assert "supervisor-status-timeout" not in journal and "private supervisor detail" not in journal


@pytest.mark.parametrize("mode", ["child-failure", "protocol", "startup", "deferred"])
def test_backup_not_run_always_alerts_through_calendar(tmp_path, monkeypatch, mode):
    from brainlayer import maintenance_calendar as calendar

    alerts = tmp_path / "alerts.json"
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alerts))
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "fixture.env"))
    monkeypatch.setattr(
        calendar, "report", lambda key, reason: job_alerts.report(key, reason, path=alerts, notify=lambda _: None)
    )
    calls = []

    def status():
        calls.append(1)
        if mode == "protocol" and len(calls) == 1:
            raise RuntimeError("invalid protocol")
        if mode == "deferred":
            raise TimeoutError("unavailable")
        return {}

    def client():
        if mode == "startup":
            raise PermissionError("custody failure")
        return SimpleNamespace(path=tmp_path / "unused.sock", request=status, _disabled=lambda *_a, **_k: set())

    monkeypatch.setattr(calendar.maintenance_control, "control_client", client)
    marker = tmp_path / "backup-ran"
    steps = [
        calendar.Step("decay", (sys.executable, "-c", "raise SystemExit(1)"), 10),
        calendar.Step("backup-daily", (sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"), 10),
    ]

    async def advance(_seconds):
        await asyncio.sleep(0)

    def run():
        return asyncio.run(calendar.run_calendar(steps, tmp_path / "calendar", tmp_path / "logs", sleep_fn=advance))

    if mode == "startup":
        with pytest.raises(PermissionError):
            run()
    else:
        run()
    if mode == "child-failure":
        assert marker.exists(), "ordinary step failures must not prevent backup"
        assert "backup-daily" not in job_alerts.active_alerts(alerts)
    else:
        assert not marker.exists()
        assert ("backup-daily" in job_alerts.active_alerts(alerts)) == (mode != "deferred")


def test_good_backup_clears_prior_alert_through_calendar(tmp_path, monkeypatch):
    from brainlayer import maintenance_calendar as calendar

    alerts = tmp_path / "alerts.json"
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alerts))
    monkeypatch.setattr(
        calendar, "report", lambda key, reason: job_alerts.report(key, reason, path=alerts, notify=lambda _: None)
    )
    monkeypatch.setattr(
        calendar.maintenance_control,
        "control_client",
        lambda: SimpleNamespace(path=tmp_path / "unused.sock", request=lambda: {}, _disabled=lambda *_a, **_k: set()),
    )
    job_alerts.report(
        "backup-daily",
        "BrainLayer backup-daily did not complete; see calendar results and logs",
        path=alerts,
        notify=lambda _: None,
    )
    assert "backup-daily" in job_alerts.active_alerts(alerts)
    step = calendar.Step("backup-daily", (sys.executable, "-c", "pass"), 10)
    assert asyncio.run(calendar.run_calendar([step], tmp_path / "calendar", tmp_path / "logs")) == 0
    assert "backup-daily" not in job_alerts.active_alerts(alerts)


@pytest.mark.parametrize("code", [1, 23, 75])
def test_backup_exit_alert_policy_through_calendar(tmp_path, monkeypatch, code):
    from brainlayer import maintenance_calendar as calendar

    alerts = tmp_path / "alerts.json"
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alerts))
    monkeypatch.setattr(
        calendar, "report", lambda key, reason: job_alerts.report(key, reason, path=alerts, notify=lambda _: None)
    )
    monkeypatch.setattr(
        calendar.maintenance_control,
        "control_client",
        lambda: SimpleNamespace(path=tmp_path / "unused.sock", request=lambda: {}, _disabled=lambda *_a, **_k: set()),
    )
    step = calendar.Step("backup-daily", (sys.executable, "-c", f"raise SystemExit({code})"), 10)
    for night in range(1, 4 if code == 75 else 2):
        now = dt.datetime(2026, 11, night, 2, tzinfo=dt.UTC)
        assert asyncio.run(
            calendar.run_calendar([step], tmp_path / "calendar", tmp_path / "logs", now_fn=lambda: now)
        ) == int(code != 75)
        active = job_alerts.active_alerts(alerts)
        if code == 75 and night < 3:
            assert "backup-daily" not in active
        else:
            assert active["backup-daily"] == (
                "Deferred 3 nights" if code == 75 else f"BrainLayer backup-daily exited {code}; see its log"
            )
