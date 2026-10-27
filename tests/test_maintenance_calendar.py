import asyncio
import datetime as dt
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from brainlayer import job_alerts
from brainlayer.maintenance_calendar import calendar_steps, run_calendar
from tests.test_maintenance_control import control, native  # noqa: F401


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
        "wal-checkpoint",
        *(["decay", "repair-fts"] if weekly else []),
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
                    assert exc.value.code == 76
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
    assert ("maintenance-full" in job_alerts.active_alerts()) == (finish[0] >= 6 or rejected)
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
        "wal-checkpoint",
        "decay",
        "repair-fts",
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
