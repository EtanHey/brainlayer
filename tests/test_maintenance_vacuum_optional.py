"""Optional VACUUM outcomes on scratch databases; no resident services or live data."""

# Imported pytest fixtures are intentionally shadowed by test arguments.
# ruff: noqa: F811

import asyncio
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from brainlayer import backup_daily as backup
from brainlayer import job_alerts, maintenance
from brainlayer import maintenance_calendar as calendar
from tests.test_maintenance_calendar import control, native  # noqa: F401
from tests.test_maintenance_wal_reclaim import weekly  # noqa: F401
from tests.test_weekly_local_gate import local_lane  # noqa: F401
from tests.test_weekly_local_gate_r2 import local_clock  # noqa: F401

WEEKLY_BACKUP = maintenance._weekly_backup
RECENT_WEEKLY_BACKUP = maintenance._recent_weekly_verified_backup


@pytest.mark.parametrize("present,age_hours", [(False, 1), (True, 1), (True, 7)])
def test_r1_snapshot_reuse_preserves_weekly_vacuum_gate(
    weekly, local_lane, local_clock, monkeypatch, capsys, present, age_hours
):
    config, _, _, _ = weekly
    # A manual full run left today's verified Drive copy, possibly reclaimed locally.
    local_clock("UTC", dt.datetime(2026, 10, 29, 8, 0) - dt.timedelta(hours=age_hours))
    original = backup.run_backup(**local_lane, phase="snapshot")
    verified = dict(original, uploaded=True, verified=True, drive_file={"id": "verified-copy"})
    backup._append_json_log(local_lane["log_path"], verified)
    archive = Path(original["snapshot"])
    if not present:
        archive.unlink()
    now = local_clock("UTC", dt.datetime(2026, 10, 29, 8, 0))
    config.db_path = local_lane["db_path"]
    config.backup_staging_dir = local_lane["staging_dir"]
    config.backup_log_path = local_lane["log_path"]
    config.now_fn = lambda: now
    monkeypatch.setenv("HOME", str(config.db_path.parent))
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(config.db_path.with_name("alerts.json")))
    monkeypatch.setenv("BRAINLAYER_CALENDAR_STEP", "1")
    monkeypatch.setattr(maintenance, "MaintenanceConfig", lambda: config)
    monkeypatch.setattr(maintenance, "_weekly_backup", WEEKLY_BACKUP)
    monkeypatch.setattr(maintenance, "_recent_weekly_verified_backup", RECENT_WEEKLY_BACKUP)
    monkeypatch.setattr(maintenance, "_freelist_ratio", lambda _: 0.5)
    monkeypatch.setattr(maintenance, "_doctor_roundtrip", lambda _: (True, 0, ""), raising=False)
    vacuum = maintenance._vacuum
    vacuum_calls = []

    def real_vacuum(*args, **kwargs):
        vacuum_calls.append(True)
        return vacuum(*args, **kwargs)

    monkeypatch.setattr(maintenance, "_vacuum", real_vacuum)
    if age_hours > 6:
        # A new attempted_at cannot freshen old bytes or authorize an overwrite.
        before = archive.read_bytes()
        with pytest.raises(FileExistsError):
            backup.run_backup(**local_lane, phase="snapshot")
        assert archive.read_bytes() == before
        capsys.readouterr()
        assert maintenance.main(["--full"]) == 75
        assert vacuum_calls == []
        assert json.loads(capsys.readouterr().out)["status"] == "deferred"
    else:
        result = backup.run_backup(**local_lane, phase="snapshot")
        assert backup._local_snapshot_intact(result)
        assert result["snapshot_created_at"] == (original["snapshot_created_at"] if present else now.isoformat())
        capsys.readouterr()
        assert maintenance.main(["--full"]) == 0
        assert vacuum_calls == [True]
        assert json.loads(capsys.readouterr().out)["vacuum"] == "completed"


@pytest.mark.parametrize("cause", ["snapshot", "timeout", "short-hold", "vacuum-error", "restore-error"])
def test_optional_vacuum_outcome_and_alerts(weekly, monkeypatch, capsys, cause):
    config, resident, events, hold = weekly
    monkeypatch.setattr(maintenance, "MaintenanceConfig", lambda: config)
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(config.db_path.with_name("alerts.json")))
    job_alerts.report("maintenance-full", "old-release snapshot failure")
    job_alerts.report("unrelated", "retained")
    if cause == "snapshot":
        monkeypatch.setenv("BRAINLAYER_CALENDAR_STEP", "1")
        monkeypatch.setattr(maintenance, "_weekly_backup", WEEKLY_BACKUP)
        monkeypatch.setattr(maintenance, "_recent_weekly_verified_backup", lambda _: None)
        expected = "tonight's local verified snapshot unavailable; VACUUM skipped"
    elif cause == "timeout":
        expected = "fresh weekly backup timed out; VACUUM skipped"

        def timeout(_):
            raise maintenance.MaintenanceAbort(expected)

        monkeypatch.setattr(maintenance, "_weekly_backup", timeout)
    elif cause == "short-hold":
        hold.deadline = 0
        expected = "maintenance hold expires before VACUUM could unwind; VACUUM skipped"
    elif cause == "vacuum-error":
        expected = "VACUUM failed (OperationalError)"

        def fail(*_a, **_kw):
            raise sqlite3.OperationalError("synthetic VACUUM failure")

        monkeypatch.setattr(maintenance, "_vacuum", fail)
    else:
        expected = "failed to resume"
        monkeypatch.setattr(maintenance, "_resume_services", lambda *_a: [("watch", RuntimeError("restore failed"))])
    failed = cause in {"vacuum-error", "restore-error"}
    assert maintenance.main(["--full"]) == (76 if failed else 75)
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == ("aborted" if failed else "deferred")
    assert expected in receipt["reason"]
    rows = [json.loads(line) for line in config.log_path.read_text().splitlines()]
    assert rows[-1]["status"] == receipt["status"] and rows[-1]["reason"] == receipt["reason"]
    alerts = job_alerts.active_alerts()
    assert ("maintenance-full" in alerts) == failed
    assert alerts["unrelated"] == "retained"
    assert resident.execute("SELECT COUNT(*) FROM chunks").fetchone() == (50,)
    if not failed:
        assert any(isinstance(e, tuple) and e[0] == "resume" for e in events)


@pytest.mark.parametrize(
    "code,stdout,reason",
    [
        (1, "", "fresh weekly backup exited 1; VACUUM skipped"),
        (0, "not a receipt", "fresh weekly backup returned no receipt (0 stderr bytes); VACUUM skipped"),
    ],
)
def test_fresh_backup_failure_reason_survives_to_deferred_receipt(weekly, monkeypatch, capsys, code, stdout, reason):
    from types import SimpleNamespace

    config, _, _, _ = weekly
    monkeypatch.setattr(maintenance, "MaintenanceConfig", lambda: config)
    monkeypatch.delenv("BRAINLAYER_CALENDAR_STEP", raising=False)
    monkeypatch.setattr(maintenance, "_weekly_backup", WEEKLY_BACKUP)
    monkeypatch.setattr(maintenance, "_recent_weekly_verified_backup", lambda _: None)
    process = SimpleNamespace(returncode=code, communicate=lambda **_kw: (stdout, ""))
    monkeypatch.setattr(maintenance.subprocess, "Popen", lambda *_a, **_kw: process)
    assert maintenance.main(["--full"]) == 75
    assert json.loads(capsys.readouterr().out) == {"status": "deferred", "reason": reason}
    assert "maintenance-full" not in job_alerts.active_alerts()


def test_calendar_deferral_notice_waits_until_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    state = {"deferrals": {}}
    step = calendar.Step("maintenance-weekly", ("unused",), 60)
    for count in (1, 2, 3):
        assert calendar.finish(tmp_path, state, step, 75, dt.date(2026, 10, 10 + count), 3) == "deferred"
        assert job_alerts.active_alerts() == ({} if count < 3 else {"maintenance-full": "Deferred 3 nights"})


def test_low_freelist_completes_without_requesting_snapshot(weekly, monkeypatch, capsys):
    config, resident, events, _ = weekly
    resident.execute("VACUUM")
    monkeypatch.setattr(maintenance, "MaintenanceConfig", lambda: config)
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(config.db_path.with_name("alerts.json")))
    job_alerts.report("maintenance-full", "old failure")
    for name in ("_weekly_backup", "_recent_weekly_verified_backup", "_vacuum"):
        monkeypatch.setattr(maintenance, name, lambda *_a, **_kw: pytest.fail("snapshot/VACUUM requested"))
    assert maintenance.main(["--full"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["vacuum"] == "not_needed" and output["freelist_ratio"] == 0
    assert "vacuum: not_needed (freelist 0.0%)" in output["actions"]
    assert json.loads(config.log_path.read_text().splitlines()[-1])["status"] == "completed"
    assert not job_alerts.active_alerts()
    assert events[0] == "hold" and events[-3:] == list(maintenance.DEFAULT_SERVICES)


@pytest.mark.parametrize(
    "ratio,threshold,needed", [(0.099, "0.10", False), (0.10, "0.10", True), (0.15, "0.20", False)]
)
def test_freelist_threshold_is_inclusive_and_configurable(weekly, monkeypatch, ratio, threshold, needed):
    config, _, _, _ = weekly
    monkeypatch.setattr(maintenance, "_freelist_ratio", lambda _: ratio)
    monkeypatch.setenv(maintenance.VACUUM_FREELIST_RATIO_ENV, threshold)
    result = maintenance.run_maintenance("full", config=config)
    assert result.vacuum == ("completed" if needed else "not_needed")


def test_consecutive_nights_retry_weekly_then_select_light_and_keep_upload_input(
    request, local_lane, local_clock, monkeypatch, capsys, tmp_path
):
    request.getfixturevalue("control")
    clock = [dt.datetime(2026, 10, 11, 4, 30)]
    state_dir = tmp_path / "calendar"
    config = maintenance.MaintenanceConfig(
        db_path=local_lane["db_path"],
        queue_dir=tmp_path / "queue",
        quarantine_root=tmp_path / "quarantine",
        log_path=tmp_path / "maintenance.log",
        backup_staging_dir=local_lane["staging_dir"],
        backup_log_path=local_lane["log_path"],
        now_fn=lambda: clock[0],
        idle_sample_seconds=0,
        recent_write_grace_seconds=0,
    )
    monkeypatch.setattr(maintenance, "MaintenanceConfig", lambda: config)
    for name in ("_run_gates", "_verify_search_latency", "_emit_telemetry"):
        monkeypatch.setattr(maintenance, name, lambda *_a: None)
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(tmp_path / "alerts.json"))
    selected, uploads = [], []
    snapshot = None

    async def execute(step, env, **_kwargs):
        nonlocal snapshot
        selected.append(step.name)
        if step.name == "backup-snapshot":
            # An intact older archive remains uploadable after its VACUUM safety window expires.
            created = clock[0] - dt.timedelta(hours=7) if clock[0].day == 11 else clock[0]
            local_clock("UTC", created)
            snapshot = backup.run_backup(**local_lane, phase="snapshot")
            local_clock("UTC", clock[0])
            assert Path(snapshot["snapshot"]).is_file()
        elif step.name in {"maintenance-weekly", "maintenance-nightly"}:
            with monkeypatch.context() as child:
                for key, value in env.items():
                    child.setenv(key, value)
                child.setenv("BRAINLAYER_CALENDAR_STEP", "1")
                # Night one has no snapshot eligible for VACUUM (the six-hour gate).
                child.setenv(maintenance.VACUUM_FREELIST_RATIO_ENV, "0" if clock[0].day == 11 else "0.1")
                return maintenance.main(["--full" if step.name == "maintenance-weekly" else "--light"])
        elif step.name == "backup-daily" and step.env.get("BRAINLAYER_BACKUP_PHASE") == "upload":
            found = backup._local_upload_receipt(local_lane["log_path"], snapshot, local_lane["staging_dir"])
            assert found is not None and backup._local_snapshot_intact(found)
            uploads.append(Path(found["snapshot"]))
        return 0

    monkeypatch.setattr(calendar, "_execute_step", execute)
    for day in (11, 12, 13):
        clock[0] = clock[0].replace(day=day)
        local_clock("UTC", clock[0])
        selected.clear()
        plan = calendar.calendar_steps(sys.executable, "/unused", tmp_path, clock[0].date())
        assert asyncio.run(calendar.run_calendar(plan, state_dir, tmp_path / "logs", now_fn=lambda: clock[0])) == 0
        state = json.loads((state_dir / "state.json").read_text())
        assert ("maintenance-weekly" in selected) == (day < 13)
        assert ("maintenance-nightly" in selected) == (day == 13)
        assert state.get("last_weekly") == (None if day == 11 else "2026-10-11")
        assert "maintenance-full" not in job_alerts.active_alerts()
        if day == 11:
            assert state["deferrals"]["maintenance-full"]["count"] == 1
        elif day == 12:
            assert "maintenance-full" not in state["deferrals"]
    assert len(uploads) == 2 and all(p.is_file() for p in uploads)
    receipts = [json.loads(line) for line in config.log_path.read_text().splitlines()]
    assert any(r.get("status") == "deferred" for r in receipts)
    assert any(r.get("vacuum") == "not_needed" and r["status"] == "completed" for r in receipts)
    capsys.readouterr()


@pytest.mark.parametrize(
    "reason",
    [
        "recent queue write activity: 1 file(s) modified recently",
        "outside quiet window: now=12:00 start_hour=3 duration_minutes=60",
    ],
)
def test_gate_deferral_preserves_prior_vacuum_failure(weekly, monkeypatch, capsys, reason):
    config, _, events, _ = weekly
    monkeypatch.setattr(maintenance, "MaintenanceConfig", lambda: config)
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(config.db_path.with_name("alerts.json")))
    prior = "VACUUM failed (OperationalError)"
    job_alerts.report("maintenance-full", prior)

    def defer(_):
        raise maintenance.MaintenanceAbort(reason)

    monkeypatch.setattr(maintenance, "_run_gates", defer)
    assert maintenance.main(["--full"]) == 75
    assert json.loads(capsys.readouterr().out) == {"status": "deferred", "reason": reason}
    assert job_alerts.active_alerts()["maintenance-full"] == prior
    assert events == []


def test_freelist_probe_failure_defers_before_hold(weekly, monkeypatch, capsys):
    config, _, events, _ = weekly
    monkeypatch.setattr(maintenance, "MaintenanceConfig", lambda: config)
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(config.db_path.with_name("alerts.json")))
    job_alerts.report("maintenance-full", "stale snapshot failure")

    def fail(_):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(maintenance, "_freelist_ratio", fail)
    assert maintenance.main(["--full"]) == 75
    reason = "freelist probe failed: OperationalError; VACUUM skipped"
    assert json.loads(capsys.readouterr().out) == {"status": "deferred", "reason": reason}
    assert job_alerts.active_alerts() == {}
    assert events == []
