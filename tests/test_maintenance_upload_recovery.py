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
if 'UPLOAD_TEST_LOCAL_DATE' in os.environ:
    b.local_backup_date = lambda: b.dt.date.fromisoformat(os.environ['UPLOAD_TEST_LOCAL_DATE'])
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
prior = [r for r in b._read_backup_receipts(Path(os.environ['BRAINLAYER_BACKUP_LOG_PATH'])) if r.get('snapshot') == str(path)][-1]
def upload(*a, **k):
    if os.environ.get('INTERRUPT_UPLOAD') == '1':
        os._exit(23)
    if 'ADVANCE_UPLOAD' in os.environ:
        confirmed = min(path.stat().st_size, prior.get('bytes_confirmed', 0) + int(os.environ['ADVANCE_UPLOAD']))
        k['progress_callback']('https://www.googleapis.com/upload/drive/v3/files?upload_id=fixture', confirmed)
        if confirmed < path.stat().st_size:
            raise b.BackupTimeoutError('isolated per-attempt work budget exhausted')
    if os.environ.get('FAIL_UPLOAD') == '1':
        raise RuntimeError('isolated upload failure')
    return metadata
b.upload_file_to_drive_raw = upload
result = b.run_backup(staging_dir=path.parent, phase='upload', remove_local_after_upload=False)
assert result['verified'] and result['uploaded']
"""


@pytest.fixture
def upload_lane(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_TEST_LOCAL_DATE", NOW.date().isoformat())
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


@pytest.mark.parametrize("local_date", ["2026-11-10", "2026-11-11"])
@pytest.mark.parametrize("previous", ["failed", "skipped"])
def test_pending_upload_is_retried_and_verified(upload_lane, previous, local_date, monkeypatch):
    monkeypatch.setenv("UPLOAD_TEST_LOCAL_DATE", local_date)
    runner, _, snapshot, log = upload_lane
    calendar.record(
        runner.directory, event="runner", kind="nightly", scheduled_date="2026-11-09", steps=["backup-daily"]
    )
    calendar.record(runner.directory, event="finished", step="backup-daily", status=previous, returncode=1)
    calendar.record(runner.directory, event="summary", kind="nightly", scheduled_date="2026-11-09", failed=1)
    job_alerts.report("backup-daily", "last upload failed", path=runner.alerts, notify=lambda *_: None)
    assert run(upload_lane) == 0
    receipts = [json.loads(line) for line in log.read_text().splitlines()]
    latest = {r["snapshot"]: r for r in receipts if "snapshot" in r}
    assert latest[str(snapshot)]["uploaded"] is True and latest[str(snapshot)]["verified"] is True
    sweeps = [r for r in receipts if r.get("event") == "local_sweep"]
    assert bool(sweeps) == (local_date > "2026-11-10")
    assert snapshot.exists()
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
    assert "backup-daily" not in job_alerts.active_alerts(runner.alerts)
    if reason == "no-snapshot":
        assert any(r.get("reason") == "no-pending-local-snapshot" for r in runner.rows())


def test_failed_child_consumes_attempt_and_retains_snapshot(upload_lane, monkeypatch):
    runner, _, snapshot, _ = upload_lane
    monkeypatch.setenv("FAIL_UPLOAD", "1")
    assert run(upload_lane) == 1
    assert "backup-daily" in job_alerts.active_alerts(runner.alerts)
    assert snapshot.exists()
    assert runner.rows()[-1]["failed"] == 1
    assert list(json.loads((runner.directory / "upload-recovery.json").read_text()).values()) == [
        dict(attempts=1, bytes_confirmed=0)
    ]


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


@pytest.fixture
def repeating_upload_lane(upload_lane, monkeypatch):
    from types import SimpleNamespace

    lane, _, _, _ = upload_lane

    def guard(*_args, **_kwargs):
        stream = (lane.directory.parent / "heartbeat").open("ab", buffering=0)
        return SimpleNamespace(
            pid=12345,
            stdin=stream,
            lease_channel=SimpleNamespace(close=lambda: None),
            poll=lambda: 0 if stream.closed else None,
        )

    monkeypatch.setattr(calendar, "launch_guard", guard)
    return upload_lane


@pytest.mark.parametrize("initial_stall", [False, True])
def test_slow_upload_progress_can_complete_after_more_than_two_attempts(
    repeating_upload_lane, monkeypatch, initial_stall
):
    lane, _, snapshot, log = repeating_upload_lane
    if initial_stall:
        monkeypatch.setenv("ADVANCE_UPLOAD", "0")
        assert run(repeating_upload_lane) == 1
    monkeypatch.setenv("ADVANCE_UPLOAD", "5")
    for expected in (5, 10, 15):
        assert run(repeating_upload_lane) == 1
        latest = [r for r in backup._read_backup_receipts(log) if r.get("snapshot") == str(snapshot)][-1]
        assert latest["bytes_confirmed"] == expected
    assert run(repeating_upload_lane) == 0
    latest = [r for r in backup._read_backup_receipts(log) if r.get("snapshot") == str(snapshot)][-1]
    assert latest["verified"] is True and latest["uploaded"] is True
    reservations = [r for r in lane.rows() if r["event"] == "upload-recovery-attempt"]
    assert len(reservations) == (5 if initial_stall else 4)
    assert [r["attempt"] for r in reservations] == ([1, 2, 1, 1, 1] if initial_stall else [1, 1, 1, 1])
    state = json.loads((lane.directory / "upload-recovery.json").read_text())
    assert list(state.values()) == [dict(attempts=1, bytes_confirmed=15)]


@pytest.mark.parametrize("interrupted", [False, True])
def test_two_real_children_without_progress_exhaust_cap(repeating_upload_lane, monkeypatch, interrupted):
    lane, _, _, _ = repeating_upload_lane
    monkeypatch.setenv("INTERRUPT_UPLOAD" if interrupted else "ADVANCE_UPLOAD", "1" if interrupted else "0")
    assert run(repeating_upload_lane) == 1
    assert run(repeating_upload_lane) == 1
    assert run(repeating_upload_lane) == 0
    assert len([r for r in lane.rows() if r["event"] == "spawn"]) == 2
    assert lane.rows()[-1]["reason"] == "attempts-exhausted"
    state = json.loads((lane.directory / "upload-recovery.json").read_text())
    assert list(state.values()) == [dict(attempts=2, bytes_confirmed=0)]


@pytest.mark.parametrize(
    "entry",
    [
        1,
        True,
        None,
        {},
        dict(attempts=True, bytes_confirmed=0),
        dict(attempts=-1, bytes_confirmed=0),
        dict(attempts=3, bytes_confirmed=0),
        dict(attempts=1, bytes_confirmed=True),
        dict(attempts=1, bytes_confirmed=-1),
        dict(attempts=1, bytes_confirmed=1.5),
        dict(attempts=1, bytes_confirmed=0, extra=1),
    ],
)
def test_invalid_attempt_state_fails_closed(upload_lane, entry):
    from brainlayer.maintenance_upload_recovery import pending_upload_step

    lane, plan, snapshot, _ = upload_lane
    state = lane.directory / "upload-recovery.json"
    state.write_text(json.dumps({"snapshot-key": entry}))
    before = state.read_bytes()
    with calendar.Custody(lane.directory), pytest.raises(ValueError, match="Invalid upload recovery attempts"):
        pending_upload_step(plan, lane.directory)
    assert state.read_bytes() == before and snapshot.exists()


@pytest.mark.parametrize("committed", [True, -1, 1.5, "5", 999999])
def test_invalid_committed_byte_receipt_cannot_reset_attempts(upload_lane, committed):
    from brainlayer.maintenance_upload_recovery import pending_upload_step, reserve_upload_attempt

    lane, plan, _, log = upload_lane
    _, reservation, _ = pending_upload_step(plan, lane.directory)
    reserve_upload_attempt(reservation)
    receipt = json.loads(log.read_text())
    receipt["bytes_confirmed"] = committed
    log.write_text(json.dumps(receipt) + "\n")
    state = lane.directory / "upload-recovery.json"
    before = state.read_bytes()
    with calendar.Custody(lane.directory), pytest.raises(ValueError, match="Invalid committed upload bytes"):
        pending_upload_step(plan, lane.directory)
    assert state.read_bytes() == before
