"""Round-1 reviewer probes: safety, recovery, old-client compatibility and scheduling."""

# Pytest registers the imported fixture; test arguments intentionally shadow that import.
# ruff: noqa: F811

import datetime as dt
import json
from pathlib import Path

import pytest

from brainlayer import backup_daily as backup
from brainlayer import job_alerts, maintenance
from brainlayer import maintenance_calendar as calendar
from brainlayer.observability_backup import _daily_snapshot
from tests.test_weekly_local_gate import config_for, local_lane  # noqa: F401, F811


def append(log, receipt):
    with log.open("a") as out:
        out.write(json.dumps(receipt) + "\n")


def test_probe_d_failed_verification_cannot_fall_back_to_earlier_good_receipt(local_lane, tmp_path, monkeypatch):
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    append(
        local_lane["log_path"],
        dict(
            receipt, uploaded=True, verified=False, verification_error="integrity_check failed", upload_state="failed"
        ),
    )
    monkeypatch.setenv("BRAINLAYER_CALENDAR_STEP", "1")
    with pytest.raises(maintenance.MaintenanceAbort) as exc:
        maintenance._weekly_backup(config_for(local_lane, tmp_path))
    assert exc.value.code == 76
    calls = []
    monkeypatch.setattr(
        backup,
        "create_sqlite_backup_artifact",
        lambda *a, **kw: calls.append(1) or (_ for _ in ()).throw(RuntimeError("fresh needed")),
    )
    with pytest.raises(RuntimeError):
        backup.run_backup(**local_lane, phase="snapshot")
    assert (
        backup._recent_local_verified_backup(
            local_lane["log_path"], local_lane["db_path"], now=dt.datetime.now(dt.UTC), max_age_hours=6
        )
        is None
    )


@pytest.mark.parametrize("phase,pragma", [("all", "quick_check"), ("snapshot", "integrity_check")])
def test_raw_snapshot_gets_mandatory_sqlite_check_before_compression(local_lane, monkeypatch, phase, pragma):
    calls = []
    original = backup._sqlite_pragma_check

    def check(path, name):
        assert path.exists() and path.suffix == ".db"
        assert not list(local_lane["staging_dir"].glob("*.gz"))
        calls.append(name)
        return original(path, name)

    monkeypatch.setattr(backup, "_sqlite_pragma_check", check)
    receipt = backup.run_backup(**local_lane, phase=phase, upload=False)
    assert calls == [pragma]
    assert receipt["local_integrity"] == "ok"
    assert receipt["local_integrity_pragma"] == pragma


def test_raw_corruption_retains_raw_and_never_publishes_local_proof(local_lane, monkeypatch):
    monkeypatch.setattr(backup, "_sqlite_pragma_check", lambda *a: "malformed page")
    with pytest.raises(RuntimeError, match="malformed page"):
        backup.run_backup(**local_lane, phase="snapshot")
    receipt = backup._read_backup_receipts(local_lane["log_path"])[-1]
    assert not receipt.get("local_verified")
    assert list(local_lane["staging_dir"].glob("*.db"))
    assert not list(local_lane["staging_dir"].glob("*.gz"))


def test_probe_a_failed_snapshot_survives_empty_upload_and_its_alert_key(local_lane, monkeypatch, tmp_path):
    good = backup.run_backup(**local_lane, phase="snapshot")
    old = dict(good, uploaded=True, verified=True, drive_file={"id": "old", "name": "old.db.gz"})
    append(local_lane["log_path"], old)
    failed = dict(
        db=str(local_lane["db_path"]),
        attempted_at=dt.datetime.now(dt.UTC).isoformat(),
        backup_log_provenance="real",
        uploaded=False,
        verified=False,
        error_type="RuntimeError",
        error="Insufficient free space",
    )
    append(local_lane["log_path"], failed)
    before = local_lane["log_path"].read_bytes()
    result = backup.run_backup(**local_lane, phase="upload")
    assert result["upload_state"] == "nothing_pending"
    assert local_lane["log_path"].read_bytes() == before
    assert _daily_snapshot(backup._read_backup_receipts(local_lane["log_path"]))[1] == "RuntimeError"
    assert calendar.KEYS["backup-snapshot"] != calendar.KEYS["backup-daily"]
    alerts = tmp_path / "alerts.json"
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(alerts))
    steps = {s.name: s for s in calendar.calendar_steps("py", "cli", tmp_path, dt.date.today()).candidates}
    state = dict(deferrals={})
    directory = tmp_path / "calendar"
    directory.mkdir()
    calendar.finish(directory, state, steps["backup-snapshot"], 1, dt.date.today(), 3)
    calendar.finish(directory, state, steps["backup-daily"], 0, dt.date.today(), 3)
    assert calendar.KEYS["backup-snapshot"] in job_alerts.active_alerts(alerts)


def test_probe_b_abandoned_snapshot_is_terminal_alerts_once_and_does_not_pin_log(local_lane, monkeypatch):
    from brainlayer.log_cap import _cap_file

    receipt = backup.run_backup(**local_lane, phase="snapshot")
    Path(receipt["snapshot"]).unlink()
    alerts = []
    monkeypatch.setattr(job_alerts, "report", lambda key, reason, **kw: alerts.append((key, reason)))
    for _ in range(2):
        result = backup.run_backup(**local_lane, phase="upload")
        assert result["upload_state"] == "nothing_pending"
    latest = backup._read_backup_receipts(local_lane["log_path"])[-1]
    assert latest["upload_state"] == "abandoned" and latest["local_verified"] is False
    assert len([a for a in alerts if a[1]]) == 1
    with local_lane["log_path"].open("a") as out:
        out.write((json.dumps({"diagnostic": "x" * 100}) + "\n") * 30)
        out.write(json.dumps({"backup_log_provenance": "real", "uploaded": True, "verified": True}) + "\n")
    assert _cap_file(local_lane["log_path"], max_bytes=1500, keep_bytes=300, line_safe=True)


def test_probe_c_drain_is_newest_first_and_new_verified_reclaims_only_older_same_db(local_lane):
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    older_path = local_lane["staging_dir"] / "2026-10-01.db.gz"
    older_path.write_bytes(Path(receipt["snapshot"]).read_bytes())
    older = dict(
        receipt,
        snapshot=str(older_path),
        local_artifact=backup._upload_identity(older_path),
        snapshot_created_at=(dt.datetime.now(dt.UTC) - dt.timedelta(days=1)).isoformat(),
    )
    local_lane["log_path"].write_text(json.dumps(older) + "\n" + json.dumps(receipt) + "\n")
    assert (
        backup._local_upload_receipt(local_lane["log_path"], receipt, local_lane["staging_dir"])["snapshot"]
        == receipt["snapshot"]
    )


def test_pending_local_state_does_not_replace_last_offsite_verified_snapshot(local_lane):
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    previous = dict(
        receipt,
        snapshot="yesterday.db.gz",
        uploaded=True,
        verified=True,
        drive_file={"id": "previous", "name": "yesterday.db.gz"},
        upload_state="verified",
    )
    snapshot, error, _ = _daily_snapshot([previous, receipt])
    assert snapshot == dict(
        last_at=previous["snapshot_created_at"].replace("+00:00", "Z"), destination="yesterday.db.gz", verified=True
    )
    assert error is None


def test_snapshot_has_own_bounded_deadline_and_existing_service_identity(tmp_path):
    plan = calendar.calendar_steps("python", "cli", tmp_path, dt.date.today())
    step = next(s for s in plan.select() if s.name == "backup-snapshot")
    assert calendar.backup_deadline(step) < 28800
    bounded = calendar._bounded_snapshot_step(step, dt.datetime(2026, 10, 18, 5, 59), 2)
    assert bounded.timeout <= 58
    assert bounded.env["BRAINLAYER_BACKUP_TIMEOUT_SECONDS"] == str(int(bounded.timeout))
    assert calendar._service_name(step) == "backup-daily"


def test_snapshot_reuse_and_weekly_gate_share_the_local_age_knob(local_lane, monkeypatch, tmp_path):
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOCAL_MAX_AGE_HOURS", "0.00000001")
    assert maintenance.MaintenanceConfig().backup_reuse_max_age_hours == float("0.00000001")
    calls = []
    monkeypatch.setattr(
        backup,
        "create_sqlite_backup_artifact",
        lambda *a, **kw: calls.append(1) or (_ for _ in ()).throw(RuntimeError("new required")),
    )
    with pytest.raises(RuntimeError):
        backup.run_backup(**local_lane, phase="snapshot")
    assert backup.local_backup_max_age_hours() == maintenance.MaintenanceConfig().backup_reuse_max_age_hours


def test_superseded_pending_reclamation_preserves_weekly_pin_and_other_db(local_lane, tmp_path):
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    older = []
    for day in (1, 2, 3):
        path = local_lane["staging_dir"] / f"2026-10-{day:02}.db.gz"
        path.write_bytes(Path(receipt["snapshot"]).read_bytes())
        older.append(
            dict(
                receipt,
                snapshot=str(path),
                local_artifact=backup._upload_identity(path),
                snapshot_created_at=(dt.datetime.now(dt.UTC) - dt.timedelta(days=20 - day)).isoformat(),
            )
        )
    older[2]["db"] = "another.db"
    for item in older:
        append(local_lane["log_path"], item)
    pin = dict(snapshot=older[1]["snapshot"], date=dt.datetime.now().astimezone().date().isoformat())
    (local_lane["staging_dir"] / ".weekly-backup-pin.json").write_text(json.dumps(pin))
    verified = dict(receipt, uploaded=True, verified=True)
    backup._reclaim_superseded_local(local_lane["log_path"], verified, local_lane["staging_dir"])
    assert not Path(older[0]["snapshot"]).exists()
    assert Path(older[1]["snapshot"]).exists()
    assert Path(older[2]["snapshot"]).exists()
    latest = {r["snapshot"]: r for r in backup._read_backup_receipts(local_lane["log_path"])}
    assert latest[older[0]["snapshot"]]["upload_state"] == "superseded"


def test_weekly_pin_holds_backup_lock_through_quiesce(local_lane, tmp_path):
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    with maintenance._pin_weekly_backup(config_for(local_lane, tmp_path), receipt):
        assert maintenance._backup_lock_is_held(local_lane["staging_dir"])
        assert backup._protected_snapshot_names(local_lane["staging_dir"]) == {Path(receipt["snapshot"]).name}
    assert not maintenance._backup_lock_is_held(local_lane["staging_dir"])
    assert backup._protected_snapshot_names(local_lane["staging_dir"])  # retained for tonight after VACUUM


def test_v1_schema_remains_frozen_and_local_fields_only_live_in_v2_sibling(local_lane):
    import hashlib

    from brainlayer.observability_backup import _local_snapshot

    path = "tests/fixtures/observability/observability-schema.v1.json"
    assert (
        hashlib.sha256(Path(path).read_bytes()).hexdigest()
        == "d4f2fa2ad47074f640959c874e1d8dbcd61ac660ab660684d0ad3a21ce555814"
    )
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    assert _daily_snapshot([receipt])[0] is None
    assert _local_snapshot([receipt])["upload_state"] == "pending"
    assert (
        "local_snapshot"
        in json.loads(Path("tests/fixtures/observability/observability-schema.v2.json").read_text())["$defs"][
            "backups_measured"
        ]["properties"]
    )


def test_snapshot_entrypoint_uses_own_alert_and_timeout(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", "snapshot")
    monkeypatch.setenv("BRAINLAYER_BACKUP_TIMEOUT_SECONDS", "28800")
    monkeypatch.setenv("BRAINLAYER_BACKUP_CALENDAR_SNAPSHOT_BUDGET_SECONDS", "58")
    observed = []
    monkeypatch.setattr(backup, "_supervise_backup_process", lambda timeout: observed.append(timeout) or 1)
    monkeypatch.setattr(job_alerts, "report", lambda key, reason, **kw: observed.append(key))
    assert backup.main() == 1
    assert observed == [58, "backup-snapshot"]


def test_env_file_disable_stops_snapshot_without_new_launchd_label(tmp_path):
    import os
    import subprocess

    step = next(
        s
        for s in calendar.calendar_steps("python", "cli", tmp_path, dt.date.today()).select()
        if s.name == "backup-snapshot"
    )
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_LAUNCHD_BACKUP_DAILY_ENABLED=0\n")
    env = os.environ | dict(
        HOME=str(tmp_path),
        BRAINLAYER_ENV_FILE=str(env_file),
        BRAINLAYER_CALENDAR_STEP="1",
        BRAINLAYER_LAUNCHD_SERVICE=calendar._service_name(step),
        XPC_SERVICE_NAME=f"com.brainlayer.{calendar._service_name(step)}",
    )
    completed = subprocess.run(["bash", "scripts/launchd/brainlayer-env-run.sh", "false"], env=env, capture_output=True)
    assert completed.returncode == 75
    assert "BACKUP_DAILY_ENABLED" in completed.stderr.decode()
    assert (tmp_path / ".local/share/brainlayer/job-runs/com.brainlayer.backup-daily.started").exists()
    assert not (tmp_path / ".local/share/brainlayer/job-runs/com.brainlayer.backup-snapshot.started").exists()


def test_failed_raw_integrity_result_is_recorded_in_receipt(local_lane, monkeypatch):
    monkeypatch.setattr(backup, "_sqlite_pragma_check", lambda *a: "malformed page")
    with pytest.raises(RuntimeError):
        backup.run_backup(**local_lane, phase="snapshot")
    latest = backup._read_backup_receipts(local_lane["log_path"])[-1]
    assert latest["local_integrity"] == "malformed page"
    assert latest["local_integrity_pragma"] == "integrity_check"


def test_new_offsite_verified_copy_covers_abandoned_failure_without_deleting_changed_file(local_lane):
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    old_path = local_lane["staging_dir"] / "2026-10-01.db.gz"
    old_path.write_bytes(b"changed bytes retained")
    abandoned = dict(
        receipt,
        snapshot=str(old_path),
        local_verified=False,
        upload_state="abandoned",
        error="changed",
        snapshot_created_at=(dt.datetime.now(dt.UTC) - dt.timedelta(days=1)).isoformat(),
    )
    append(local_lane["log_path"], abandoned)
    verified = dict(receipt, uploaded=True, verified=True, drive_file={"name": "new.db.gz"}, upload_state="verified")
    append(local_lane["log_path"], verified)
    backup._reclaim_superseded_local(local_lane["log_path"], verified, local_lane["staging_dir"])
    records = backup._read_backup_receipts(local_lane["log_path"])
    assert _daily_snapshot(records)[1] is None
    assert old_path.read_bytes() == b"changed bytes retained"
    from brainlayer.observability_backup import _local_snapshot

    assert _local_snapshot(records)["destination"] == Path(receipt["snapshot"]).name
    assert _local_snapshot(records)["upload_state"] == "verified"


@pytest.mark.parametrize("damage", ["quick-only", "uploaded-unverified"])
def test_weak_or_unverified_latest_proof_cannot_authorize_weekly(local_lane, tmp_path, monkeypatch, damage):
    receipt = backup.run_backup(**local_lane, phase="snapshot")
    if damage == "quick-only":
        receipt["local_integrity_pragma"] = "quick_check"
    else:
        receipt.update(uploaded=True, verified=False)
    append(local_lane["log_path"], receipt)
    monkeypatch.setenv("BRAINLAYER_CALENDAR_STEP", "1")
    with pytest.raises(maintenance.MaintenanceAbort):
        maintenance._weekly_backup(config_for(local_lane, tmp_path))


def test_default_snapshot_cap_has_measured_headroom_and_shutdown_reserve(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_BACKUP_SNAPSHOT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("BRAINLAYER_ENV_FILE", str(tmp_path / "missing.env"))
    plan = calendar.calendar_steps("python", "cli", tmp_path, dt.date.today())
    step = next(s for s in plan.select() if s.name == "backup-snapshot")
    assert step.timeout == 2700  # measured 858.494s; 3x rounded up to five minutes
    bounded = calendar._bounded_snapshot_step(step, dt.datetime(2026, 10, 18, 5, 59), 2)
    assert bounded.timeout <= 57  # include scheduling headroom beyond group shutdown
