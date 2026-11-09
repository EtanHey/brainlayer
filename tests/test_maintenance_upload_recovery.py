"""Upload recovery drives real calendar custody and real upload children; Drive is isolated."""

import asyncio
import datetime as dt
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from brainlayer import backup_daily as backup
from brainlayer import job_alerts
from brainlayer import maintenance_calendar as calendar
from tests import test_maintenance_calendar_completion as completion

runner = completion.runner

NOW = dt.datetime(2026, 11, 10, 12)
CHILD = """
import os
from pathlib import Path
from types import SimpleNamespace
from brainlayer import backup_daily as b
assert os.environ['BRAINLAYER_BACKUP_PHASE'] == 'upload'
def forbidden(*a, **k):
    raise AssertionError('snapshot creation forbidden')
b.create_sqlite_backup_artifact = forbidden
b.get_drive_credentials = lambda: object()
b.ensure_drive_folder_chain = lambda *a: 'folder'
b.assert_drive_folder_owned_by_machine = lambda *a, **k: None
path = Path(os.environ['BRAINLAYER_BACKUP_STAGING_DIR']) / os.environ['BRAINLAYER_BACKUP_UPLOAD_SNAPSHOT']
metadata = dict(id='remote', name=path.name, size=str(path.stat().st_size),
                appProperties={b.DRIVE_MACHINE_PROPERTY: 'test-machine'}, md5Checksum=b._md5_file(path))
service = SimpleNamespace(files=lambda: SimpleNamespace(get=lambda **k: SimpleNamespace(execute=lambda **k: metadata)))
b.build_drive_service = lambda *a: service
b.verify_sqlite_backup_artifact = lambda *a, **k: dict(verified=True, verification_mode='quick', local_md5=b._md5_file(path))
def upload(*a, **k):
    if os.environ.get('FAIL_UPLOAD') == '1':
        raise RuntimeError('isolated upload failure')
    return metadata
b.upload_file_to_drive_raw = upload
result = b.run_backup(staging_dir=path.parent, phase='upload', remove_local_after_upload=False)
assert result['verified'] and result['uploaded']
"""


@pytest.fixture
def upload_lane(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "fixture.db"))
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "test-machine")
    monkeypatch.setenv("BRAINLAYER_BACKUP_STAGING_DIR", str(tmp_path))
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PATH", str(tmp_path / "receipts.jsonl"))
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PROVENANCE", "test")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(runner.alerts))
    monkeypatch.setenv(
        "PYTHONPATH", str(Path(calendar.__file__).parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    runner.directory.mkdir(mode=0o700)
    (runner.directory / "runs.jsonl").touch()
    script = tmp_path / "upload.py"
    script.write_text(CHILD)
    plan = calendar.calendar_steps(sys.executable, "/unused", tmp_path, NOW.date())
    plan = replace(
        plan,
        nightly_start=dt.time(3, 15),
        candidates=tuple(
            replace(step, argv=(sys.executable, str(script))) if step.name == "backup-daily" else step
            for step in plan.candidates
        ),
    )
    snapshot = tmp_path / "2026-11-10.db.zst"
    snapshot.write_bytes(b"isolated archive")
    receipt = dict(
        snapshot=str(snapshot),
        local_verified=True,
        uploaded=False,
        verified=False,
        local_artifact=backup._upload_identity(snapshot),
        db=str(tmp_path / "fixture.db"),
        machine_id="test-machine",
        drive_folder="/".join(backup.default_drive_folder_parts("test-machine")),
        backup_log_provenance="test",
        sentinel_snapshot_chunks=1,
        snapshot_created_at=NOW.replace(tzinfo=dt.UTC).isoformat(),
    )
    log = tmp_path / "receipts.jsonl"
    log.write_text(json.dumps(receipt) + "\n")
    return runner, plan, snapshot, log


def run(lane, now=NOW):
    runner, plan, _, _ = lane
    return asyncio.run(
        calendar.run_calendar(
            plan,
            runner.directory,
            runner.logs,
            now_fn=lambda: now,
            upload_recovery=True,
            guard_timeout=5,
            shutdown_timeout=2,
        )
    )


@pytest.mark.parametrize("previous", ["failed", "skipped"])
def test_pending_upload_is_retried_and_verified(upload_lane, previous):
    runner, _, snapshot, log = upload_lane
    calendar.record(
        runner.directory, event="runner", kind="nightly", scheduled_date="2026-11-09", steps=["backup-daily"]
    )
    calendar.record(runner.directory, event="finished", step="backup-daily", status=previous, returncode=1)
    calendar.record(runner.directory, event="summary", kind="nightly", scheduled_date="2026-11-09", failed=1)
    job_alerts.report("backup-daily", "last upload failed", path=runner.alerts, notify=lambda *_: None)
    assert run(upload_lane) == 0
    receipts = [json.loads(line) for line in log.read_text().splitlines()]
    assert receipts[-1]["uploaded"] is True and receipts[-1]["verified"] is True
    assert receipts[-1]["snapshot"] == str(snapshot)
    assert "backup-daily" not in job_alerts.active_alerts(runner.alerts)
    assert runner.rows()[-1]["failed"] == 0
    assert runner.rows()[-1]["scheduled_date"] == "2026-11-09"
    assert len([r for r in runner.rows() if r["event"] == "upload-recovery-attempt"]) == 1


@pytest.mark.parametrize("reason", ["no-snapshot", "exhausted", "nightly", "custody", "paused", "interrupted"])
def test_upload_admission_refuses_without_starting_child(upload_lane, monkeypatch, reason):
    runner, plan, snapshot, _ = upload_lane
    now = NOW
    if reason == "no-snapshot":
        snapshot.unlink()
    elif reason in {"exhausted", "interrupted"}:
        from brainlayer.maintenance_upload_recovery import pending_upload_step, reserve_upload_attempt

        runner.directory.mkdir(mode=0o700, exist_ok=True)
        # Reservations without child completion model interrupted attempts; they still consume the cap.
        for _ in range(2):
            _, reservation, _ = pending_upload_step(plan, runner.directory)
            reserve_upload_attempt(reservation)
    elif reason == "nightly":
        now = NOW.replace(hour=2, minute=30)
    elif reason == "paused":
        client = calendar.maintenance_control.control_client()
        client._disabled = lambda *_a, **_k: {"com.brainlayer.backup-daily"}
        monkeypatch.setattr(calendar.maintenance_control, "control_client", lambda: client)
    if reason == "custody":
        with calendar.Custody(runner.directory / "full-calendar"):
            with calendar.Custody(runner.directory / "recovery-admission"):
                assert run(upload_lane, now) == 0
    else:
        assert run(upload_lane, now) == 0
    assert not any(r["event"] == "spawn" for r in runner.rows())
    if reason == "no-snapshot":
        assert any(r.get("reason") == "no-pending-local-snapshot" for r in runner.rows())


def test_failed_child_consumes_attempt_and_retains_snapshot(upload_lane, monkeypatch):
    runner, _, snapshot, _ = upload_lane
    monkeypatch.setenv("FAIL_UPLOAD", "1")
    assert run(upload_lane) == 1
    assert snapshot.exists()
    assert runner.rows()[-1]["failed"] == 1
    assert list(json.loads((runner.directory / "upload-recovery.json").read_text()).values()) == [1]


def test_supervisor_registers_hourly_upload_recovery(tmp_path, monkeypatch):
    from brainlayer.service_supervisor import Component
    from brainlayer.supervisor_jobs import with_brainbar_monitor

    monkeypatch.setattr(sys, "platform", "darwin")
    source = Component(
        ("/unused",),
        env={
            "BRAINLAYER_ENV_FILE": str(tmp_path / "fixture.env"),
            "BRAINLAYER_BACKUP_STAGING_DIR": str(tmp_path / "backups"),
        },
        stderr=tmp_path / "logs/observability.err.log",
    )
    components = with_brainbar_monitor({"observability": source}, tmp_path / "control.sock")
    upload = components["database-upload-recovery"]
    assert upload.interval == 3600
    assert "--recover-upload" in upload.argv and "--recover-transcripts" not in upload.argv
    assert upload.env["BRAINLAYER_BACKUP_STAGING_DIR"] == str(tmp_path / "backups")
    assert upload.run_timeout < components["transcript-backup-recovery"].run_timeout
    assert upload.exit_timeout == 60


def test_exhausted_newest_snapshot_does_not_upload_it_via_another_attempt(upload_lane):
    from brainlayer.maintenance_upload_recovery import pending_upload_step, reserve_upload_attempt

    runner, plan, snapshot, log = upload_lane
    for _ in range(2):
        _, reservation, _ = pending_upload_step(plan, runner.directory)
        reserve_upload_attempt(reservation)
    older = snapshot.with_name("2026-11-09.db.zst")
    older.write_bytes(b"older pending snapshot")
    receipt = json.loads(log.read_text())
    receipt.update(snapshot=str(older), local_artifact=backup._upload_identity(older))
    with log.open("a") as stream:
        stream.write(json.dumps(receipt) + "\n")
    assert run(upload_lane) == 0
    assert [json.loads(line) for line in log.read_text().splitlines() if "snapshot" in json.loads(line)][-1][
        "snapshot"
    ] == str(older)
    assert snapshot.exists()


def test_real_upload_child_honors_selected_artifact(upload_lane):
    import subprocess

    _, _, snapshot, log = upload_lane
    selected = snapshot.with_name("2026-11-09.db.zst")
    selected.write_bytes(b"older selected archive")
    receipt = json.loads(log.read_text())
    receipt.update(
        snapshot=str(selected),
        local_artifact=backup._upload_identity(selected),
        snapshot_created_at="2026-11-09T12:00:00+00:00",
    )
    with log.open("a") as stream:
        stream.write(json.dumps(receipt) + "\n")
    result = subprocess.run(
        [sys.executable, str(snapshot.parent / "upload.py")],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "BRAINLAYER_BACKUP_PHASE": "upload",
            "BRAINLAYER_BACKUP_UPLOAD_SNAPSHOT": selected.name,
            "BRAINLAYER_BACKUP_UPLOAD_SHA256": receipt["local_artifact"]["sha256"],
        },
    )
    assert result.returncode == 0, result.stderr
    latest = {r["snapshot"]: r for r in map(json.loads, log.read_text().splitlines()) if "snapshot" in r}
    assert latest[str(selected)]["uploaded"] is True
    assert latest[str(snapshot)]["uploaded"] is False
