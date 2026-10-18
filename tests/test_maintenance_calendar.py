import asyncio
import datetime as dt
import json
import os
import sys

import pytest

from brainlayer import job_alerts
from brainlayer.maintenance_calendar import calendar_steps, run_calendar
from tests.test_maintenance_control import control, native  # noqa: F401


@pytest.mark.parametrize("finish", [(3, 50), (4, 40), (6, 10)])
def test_sunday_full_waits_runs_real_vacuum_or_alerts_overrun(request, monkeypatch, finish):
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

    async def sleep(seconds):
        clock[0] += dt.timedelta(seconds=seconds)
        await asyncio.sleep(0)

    async def execute(step, env, **kwargs):
        seen.append(step.name)
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
    monkeypatch.setattr(
        maintenance, "_recent_verified_backup", lambda *_: dict(verified=True, drive_file={"id": "daily"})
    )
    monkeypatch.setattr(maintenance, "_run_bounded_weekly_backup", lambda *_: pytest.fail("daily backup not reused"))
    plan = calendar_steps(sys.executable, "/unused/cli", path, dt.date(2026, 10, 11))
    assert (
        asyncio.run(run_calendar(plan, path / "calendar", path / "logs", now_fn=lambda: clock[0], sleep_fn=sleep)) == 0
    )
    assert ("maintenance-weekly" in seen) == (finish[0] < 6)
    assert (db.stat().st_size < before) == (finish[0] < 6)
    assert ("maintenance-full" in job_alerts.active_alerts()) == (finish[0] >= 6)
    events = [json.loads(l) for l in (path / "calendar" / "runs.jsonl").read_text().splitlines()]
    assert any(e["event"] == "wait-start" for e in events) == (finish[0] < 4)
    if finish[0] < 6:
        assert json.loads((path / "calendar" / "state.json").read_text())["last_weekly"] == "2026-10-11"


@pytest.mark.parametrize("late_step", ["index", "backup-daily", None])
def test_calendar_records_actual_window_blocker_and_weekly_substitution(request, monkeypatch, late_step):
    from brainlayer import maintenance_calendar

    path, _, _, _ = request.getfixturevalue("control")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(path / "alerts.json"))
    clock = [dt.datetime(2026, 10, 11, 3, 15)]

    async def execute(step, env, **kwargs):
        if step.name == late_step:
            clock[0] = clock[0].replace(hour=6, minute=10)
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
    weekly = next(row for row in rows if row["event"] == "finished" and row["step"] == "maintenance-weekly")
    assert weekly.get("replaces") == "maintenance-nightly"
    assert weekly.get("reason") == (
        "index-overran-window" if late_step == "index" else "quiet-window-missed" if late_step else None
    )
    assert weekly["status"] == ("deferred" if late_step else "ok")


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
