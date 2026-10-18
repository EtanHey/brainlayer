"""Local snapshot safety and supervised upload handoff; no network or resident daemon."""

import datetime as dt
import json
import sqlite3
from pathlib import Path

import pytest

from brainlayer import backup_daily as backup
from brainlayer import maintenance, maintenance_calendar
from tests.test_maintenance_routine import _config


@pytest.fixture
def local_lane(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PROVENANCE", "real")
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "test-machine")
    monkeypatch.setattr(backup, "_should_run_full_verify", lambda _: False)
    monkeypatch.setattr(backup, "_important_usage_capacity_bytes", lambda _: None)
    monkeypatch.setattr(backup, "_brainbar_writer_started_at", lambda _: 0)
    db = tmp_path / "source.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE chunks(id TEXT)")
        conn.execute("INSERT INTO chunks VALUES ('one')")

    def snapshot(target, **kwargs):
        with sqlite3.connect(db) as conn:
            conn.execute("VACUUM INTO ?", (str(target),))

    monkeypatch.setattr(backup, "request_brainbar_vacuum_into", snapshot)
    monkeypatch.setattr(backup, "get_drive_credentials", lambda: pytest.fail("Drive before local handoff"))
    return dict(
        db_path=db,
        staging_dir=tmp_path / "backups",
        log_path=tmp_path / "receipt.jsonl",
        remove_local_after_upload=False,
    )


def config_for(args, tmp_path):
    config = _config(tmp_path, now=dt.datetime.now(dt.UTC))
    config.db_path = args["db_path"]
    config.backup_log_path = args["log_path"]
    config.backup_staging_dir = args["staging_dir"]
    return config


def test_snapshot_receipt_authorizes_weekly_before_any_drive_call(local_lane, monkeypatch, tmp_path):
    receipt = backup.run_backup(**local_lane, upload=False)
    assert receipt["local_verified"] is True
    assert receipt["uploaded"] is False and receipt["verified"] is False
    assert receipt["upload_state"] == "pending"
    assert receipt["local_artifact"] == backup._upload_identity(Path(receipt["snapshot"]))
    assert json.loads(local_lane["log_path"].read_text().splitlines()[-1])["local_verified"]
    monkeypatch.setattr(maintenance, "_run_bounded_weekly_backup", lambda *_: pytest.fail("second snapshot"))
    assert maintenance._weekly_backup(config_for(local_lane, tmp_path))["snapshot"] == receipt["snapshot"]


@pytest.mark.parametrize("damage", ["missing", "changed", "unverified", "other-db", "yesterday", "future"])
def test_local_gate_fails_closed(local_lane, monkeypatch, tmp_path, damage):
    receipt = backup.run_backup(**local_lane, upload=False)
    now = dt.datetime.now(dt.UTC)
    if damage == "missing":
        Path(receipt["snapshot"]).unlink()
    elif damage == "changed":
        Path(receipt["snapshot"]).write_bytes(b"changed")
    elif damage == "unverified":
        receipt["local_verified"] = False
    elif damage == "other-db":
        receipt["db"] = str(tmp_path / "different.db")
    else:
        receipt["snapshot_created_at"] = (now + dt.timedelta(days=-1 if damage == "yesterday" else 1)).isoformat()
    local_lane["log_path"].write_text(json.dumps(receipt) + "\n")
    monkeypatch.setenv("BRAINLAYER_CALENDAR_STEP", "1")
    monkeypatch.setattr(maintenance, "_run_bounded_weekly_backup", lambda *_: pytest.fail("calendar must fail closed"))
    with pytest.raises(maintenance.MaintenanceAbort) as exc:
        maintenance._weekly_backup(config_for(local_lane, tmp_path))
    assert exc.value.code == 76


def test_upload_phase_reuses_snapshot_and_completed_receipt(local_lane, monkeypatch):
    receipt = backup.run_backup(**local_lane, upload=False)
    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", lambda *a, **kw: pytest.fail("second snapshot"))
    monkeypatch.setattr(backup, "get_drive_credentials", lambda: object())
    monkeypatch.setattr(backup, "build_drive_service", lambda: object())
    monkeypatch.setattr(backup, "ensure_drive_folder_chain", lambda *a: "folder")
    monkeypatch.setattr(backup, "assert_drive_folder_owned_by_machine", lambda *a, **kw: None)
    monkeypatch.setattr(backup, "verify_drive_upload", lambda *a, **kw: None)
    uploads = []

    def upload(path, *args, **kwargs):
        uploads.append(path)
        assert path.read_bytes() == Path(receipt["snapshot"]).read_bytes()
        return {"id": "uploaded", "name": path.name}

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    result = backup.run_backup(**local_lane, phase="upload")
    assert result["verified"] is True
    assert backup.run_backup(**local_lane, phase="upload")["drive_file"]["id"] == "uploaded"
    assert uploads == [Path(receipt["snapshot"])]


def test_retention_never_prunes_pending_local_copy(tmp_path):
    for day in range(1, 6):
        (tmp_path / f"2026-10-{day:02}.db.gz").write_bytes(b"gzip")
    names = {f"2026-10-{day:02}.db.gz" for day in range(2, 6)}
    backup.prune_local_gzip_snapshots(tmp_path, keep_latest=3, verified_drive_names=names)
    assert (tmp_path / "2026-10-01.db.gz").exists()


def test_weekly_calendar_upload_runs_last(tmp_path):
    plan = maintenance_calendar.calendar_steps("python", "cli", tmp_path, dt.date(2026, 10, 18))
    names = [step.name for step in plan.select()]
    assert names[1:3] == ["backup-snapshot", "maintenance-weekly"]
    assert names[-1] == "backup-daily"
    steps = {step.name: step for step in plan.select()}
    assert steps["backup-snapshot"].env["BRAINLAYER_BACKUP_PHASE"] == "snapshot"
    assert steps["backup-daily"].env["BRAINLAYER_BACKUP_PHASE"] == "upload"


def test_observability_pending_is_distinct_from_uploaded(local_lane):
    from brainlayer.observability_backup import _daily_snapshot

    receipt = backup.run_backup(**local_lane, upload=False)
    snapshot, error, _ = _daily_snapshot([receipt])
    assert snapshot["local_verified"] is True
    assert snapshot["uploaded"] is False
    assert snapshot["upload_state"] == "pending"
    assert error is None


@pytest.mark.parametrize("failure", ["stalled", "verification"])
def test_upload_recovery_never_recreates_snapshot_or_reuploads_completed_session(local_lane, monkeypatch, failure):
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", lambda *a, **kw: pytest.fail("second snapshot"))
    monkeypatch.setattr(backup, "get_drive_credentials", lambda: object())
    monkeypatch.setattr(backup, "build_drive_service", lambda: object())
    monkeypatch.setattr(backup, "ensure_drive_folder_chain", lambda *a: "folder")
    monkeypatch.setattr(backup, "assert_drive_folder_owned_by_machine", lambda *a, **kw: None)
    attempts = []
    url = "https://www.googleapis.com/upload/drive/v3/files?upload_id=isolated"

    def upload(path, *args, **kwargs):
        attempts.append(kwargs.get("resume_session_url"))
        if failure == "stalled" and len(attempts) == 1:
            kwargs["progress_callback"](url, 0)
            raise backup.DriveUploadStalledError(bytes_confirmed=0, total_bytes=1, attempts=1)
        return {"id": "uploaded", "name": path.name}

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    monkeypatch.setattr(backup, "verify_drive_upload", lambda *a, **kw: None)
    if failure == "stalled":
        with pytest.raises(backup.DriveUploadStalledError):
            backup.run_backup(**local_lane, phase="upload")
    else:
        monkeypatch.setattr(
            backup, "verify_drive_upload", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("offline"))
        )
        with pytest.raises(RuntimeError, match="offline"):
            backup.run_backup(**local_lane, phase="upload")
        monkeypatch.setattr(backup, "verify_drive_upload", lambda *a, **kw: None)
    failed = json.loads(local_lane["log_path"].read_text().splitlines()[-1])
    assert failed["upload_state"] == "failed"
    from brainlayer.observability_backup import _daily_snapshot

    snapshot, error_type, _ = _daily_snapshot([failed])
    assert error_type == failed["error_type"]
    assert snapshot["upload_error"] == failed["error"]
    assert Path(receipt["snapshot"]).exists()
    assert maintenance._weekly_backup(config_for(local_lane, Path(local_lane["db_path"]).parent))["local_verified"]
    assert backup.run_backup(**local_lane, phase="upload")["verified"]
    assert attempts == ([None, url] if failure == "stalled" else [None])


def test_pending_snapshot_fields_validate_both_observability_schemas(local_lane):
    import jsonschema

    from brainlayer.observability_backup import _daily_snapshot

    receipt = backup.run_backup(**local_lane, phase="snapshot")
    snapshot, _, _ = _daily_snapshot([receipt])
    for version in (1, 2):
        schema = json.loads(Path(f"tests/fixtures/observability/observability-schema.v{version}.json").read_text())
        jsonschema.validate(snapshot, schema["$defs"]["backups_measured"]["properties"]["db_snapshot"])


def test_new_snapshot_does_not_hide_an_older_failed_upload(local_lane):
    from brainlayer.observability_backup import _daily_snapshot

    receipt = backup.run_backup(**local_lane, phase="snapshot")
    failed = dict(
        receipt, snapshot="older.db.gz", upload_state="failed", error="Drive offline", error_type="ConnectionError"
    )
    _, error, _ = _daily_snapshot([failed, receipt])
    assert error == "ConnectionError"
    assert _daily_snapshot([failed, receipt])[0]["upload_error"] == "Drive offline"
    completed = dict(failed, uploaded=True, verified=True, upload_state="verified")
    completed.pop("error")
    completed.pop("error_type")
    _, error, _ = _daily_snapshot([failed, receipt, completed])
    assert error is None


def test_combined_daily_command_reuses_todays_pending_snapshot(local_lane, monkeypatch):
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    monkeypatch.setattr(
        backup, "create_sqlite_backup_artifact", lambda *a, **kw: pytest.fail("pending copy overwritten")
    )
    # A credentials failure happens only after selecting the original bytes.
    monkeypatch.setattr(backup, "get_drive_credentials", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(RuntimeError, match="offline"):
        backup.run_backup(**local_lane)
    latest = json.loads(local_lane["log_path"].read_text().splitlines()[-1])
    assert latest["local_artifact"] == receipt["local_artifact"]


def test_invalid_older_copy_does_not_prevent_upload_of_intact_copy(local_lane):
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    invalid = dict(receipt, snapshot=str(local_lane["staging_dir"] / "2026-10-01.db.gz"))
    Path(invalid["snapshot"]).write_bytes(b"corrupt")
    # Put invalid first, then intact; the uploader must retain and report it while progressing.
    local_lane["log_path"].write_text(json.dumps(invalid) + "\n" + json.dumps(receipt) + "\n")
    assert (
        backup._local_upload_receipt(local_lane["log_path"], receipt, local_lane["staging_dir"])["snapshot"]
        == receipt["snapshot"]
    )
    records = [json.loads(line) for line in local_lane["log_path"].read_text().splitlines()]
    assert records[-1]["error_code"] == "local_snapshot_invalid"
    assert Path(invalid["snapshot"]).exists()


def test_local_gate_requires_explicit_creation_time_not_filename_midnight(local_lane):
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    receipt.pop("snapshot_created_at")
    local_lane["log_path"].write_text(json.dumps(receipt) + "\n")
    date = dt.date.fromisoformat(Path(receipt["snapshot"]).name[:10])
    now = dt.datetime.combine(date, dt.time(0, 30), tzinfo=dt.UTC)
    assert (
        backup._recent_local_verified_backup(local_lane["log_path"], local_lane["db_path"], now=now, max_age_hours=6)
        is None
    )


def test_snapshot_capacity_error_names_retained_gzip(local_lane, monkeypatch):
    local_lane["staging_dir"].mkdir()
    retained = local_lane["staging_dir"] / "2026-10-01.db.gz"
    retained.write_bytes(b"retained")

    def no_capacity(*args, **kwargs):
        raise RuntimeError("Insufficient free space for backup")

    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", no_capacity)
    with pytest.raises(RuntimeError, match=retained.name):
        backup.run_backup(**local_lane, phase="snapshot")
    assert retained.exists()
