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
        *(["backup-daily"] if weekly else []),
        "maintenance-weekly" if weekly else "maintenance-nightly",
        "jsonl-backup",
        "p0-counter",
        "wal-checkpoint",
        *(["decay", "repair-fts"] if weekly else []),
        *([] if weekly else ["backup-daily"]),
    ]
    by_name = {step.name: step for step in plan.candidates}
    assert by_name["backup-daily"].argv[-1] == str(tmp_path / "backup-daily.sh")
    assert by_name["backup-daily"].timeout == 28890
    assert by_name["backup-daily"].env["BRAINLAYER_BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS"] == "6"
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

    def backup(config, timeout):
        assert "backup-daily" in seen  # The failed daily attempt left no reusable receipt.
        assert not config.backup_log_path.exists()  # No recent verified receipt.
        assert 0 < timeout <= 7200
        backups.append(timeout)
        return dict(verified=receipt == "fresh", uploaded=True, drive_file={"id": "weekly"})

    async def sleep(seconds):
        clock[0] += dt.timedelta(seconds=seconds)
        await asyncio.sleep(0)

    async def execute(step, env, **kwargs):
        seen.append(step.name)
        if step.name == "backup-daily":
            return int(receipt != "recent")
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
                    with pytest.raises(maintenance.MaintenanceAbort, match="weekly backup was not verified") as exc:
                        maintenance.run_maintenance("full", config=cfg)
                    assert exc.value.code == 76
                    assert db.stat().st_size == before
                    return exc.value.code
                result = maintenance.run_maintenance("full", config=cfg)
                assert result.backup_status == "verified" and result.vacuum_after_bytes < result.vacuum_before_bytes
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
    if receipt == "recent":
        monkeypatch.setattr(
            maintenance, "_recent_verified_backup", lambda *_: dict(verified=True, drive_file={"id": "previous"})
        )
    monkeypatch.setattr(maintenance, "_run_bounded_weekly_backup", backup)
    plan = calendar_steps(sys.executable, "/unused/cli", path, dt.date(2026, 10, 11))
    rejected = receipt == "unverified" and finish[0] < 6
    assert asyncio.run(
        run_calendar(plan, path / "calendar", path / "logs", now_fn=lambda: clock[0], sleep_fn=sleep)
    ) == int(rejected or receipt != "recent")
    assert ("maintenance-weekly" in seen) == (finish[0] < 6)
    assert (db.stat().st_size < before) == (finish[0] < 6 and not rejected)
    assert len(backups) == int(receipt != "recent" and finish[0] < 6)
    assert seen[-1] == "repair-fts"
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
        if step.name == late_step:
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
    from types import SimpleNamespace

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

    # Patch only this module's clock; leave the supervisor's real time functions intact.
    monkeypatch.setattr(backup_daily, "dt", SimpleNamespace(datetime=ClockDateTime, UTC=dt.UTC, timedelta=dt.timedelta))
    uploads, seen = [], []

    @backup_daily._serialized_backup_run
    def upload(db_path, staging_dir, log_path):
        started = clock[0]
        clock[0] += dt.timedelta(hours=2, minutes=20)
        receipt = dict(
            attempted_at=started.isoformat(),
            db=str(db_path),
            uploaded=True,
            verified=True,
            drive_file={"id": "fresh-daily"},
        )
        with log_path.open("a") as handle:
            handle.write(json.dumps(receipt) + "\n")
        uploads.append(receipt)
        return receipt

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
    expected_id = "fresh-daily" if existing_receipt == "yesterday" else "weekly-safety"

    async def execute(step, env, **kwargs):
        seen.append((step.name, clock[0]))
        with monkeypatch.context() as child:
            for key, value in env.items():
                child.setenv(key, value)
            if step.name == "backup-daily":
                assert upload(db_path=db, staging_dir=staging, log_path=log)["drive_file"]["id"] == expected_id
            elif step.name == "maintenance-weekly":
                result = maintenance.run_maintenance("full", config=cfg)
                assert result.backup_status == "verified"
                assert f"verified_drive_backup={expected_id}" in result.actions
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
        "backup-daily",
        "maintenance-weekly",
        "jsonl-backup",
        "p0-counter",
        "wal-checkpoint",
        "decay",
        "repair-fts",
    ]
    assert len(uploads) == int(existing_receipt == "yesterday")
    if uploads:
        assert dict(seen)["maintenance-weekly"] - dict(seen)["backup-daily"] == dt.timedelta(hours=2, minutes=20)
        assert dict(seen)["maintenance-weekly"].hour == 5
    assert db.stat().st_size < before  # Real fixture VACUUM after verified receipt reuse.
    rows = [json.loads(line) for line in cfg.log_path.read_text().splitlines()]
    assert any(row.get("backup_status") == "reused_verified" for row in rows)
    assert json.loads((path / "calendar" / "state.json").read_text())["last_weekly"] == "2026-10-11"
