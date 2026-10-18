import asyncio
import datetime as dt
import json
import os
import sys

import pytest

from brainlayer import job_alerts
from brainlayer.maintenance_calendar import calendar_steps, run_calendar
from tests.test_maintenance_control import control, native  # noqa: F401


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
