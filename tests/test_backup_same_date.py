"""Same-date nightly retries use proven bytes without replacing the archive."""

# Imported pytest fixtures are intentionally shadowed by test arguments.
# ruff: noqa: F811

import datetime as dt
import json
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as backup
from brainlayer import health_verdict as hv
from tests.test_weekly_local_gate_r2 import local_clock  # noqa: F401


@pytest.fixture
def same_date(tmp_path, monkeypatch, local_clock):
    now = local_clock("UTC", dt.datetime(2026, 11, 2, 12))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PROVENANCE", "real")
    db = tmp_path / "source.db"
    staging = tmp_path / "backups"
    staging.mkdir()
    created = now - dt.timedelta(minutes=10)
    archive = staging / f"{backup.local_backup_date(now)}.db.zst"
    archive.write_bytes(b"verified archive")
    log = tmp_path / "backup.jsonl"
    receipt = dict(
        db=str(db),
        snapshot=str(archive),
        snapshot_created_at=created.isoformat(),
        attempted_at=created.isoformat(),
        machine_id="test-machine",
        drive_folder="/".join(backup.default_drive_folder_parts("test-machine")),
        backup_log_provenance="real",
        uploaded=True,
        verified=True,
        local_verified=True,
        local_artifact=backup._upload_identity(archive),
        local_integrity="ok",
        local_integrity_pragma="integrity_check",
        drive_file={"id": "verified-copy", "name": archive.name},
    )
    for key, value in dict(
        BRAINLAYER_DB=str(db),
        BRAINLAYER_BACKUP_STAGING_DIR=str(staging),
        BRAINLAYER_MACHINE_ID="test-machine",
        BRAINLAYER_BACKUP_DRIVE_FOLDER=receipt["drive_folder"],
        BRAINLAYER_JOB_ALERT_PATH=str(tmp_path / "alerts.json"),
    ).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(backup, "_backup_log_path", lambda *a, **kw: log)
    probes = []
    monkeypatch.setattr(backup, "get_drive_credentials", lambda: probes.append("credentials") or object())
    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", lambda *a, **kw: pytest.fail("new snapshot"))
    return SimpleNamespace(db=db, staging=staging, archive=archive, log=log, receipt=receipt, now=now, probes=probes)


@pytest.mark.parametrize("phase", ["snapshot", "all", "upload"])
@pytest.mark.parametrize("revocation", ["verification_error", "drive_md5_match"])
@pytest.mark.parametrize("later", [False, True], ids=["on-proof", "later-receipt"])
@pytest.mark.parametrize("present", [False, True])
def test_r1_revoked_drive_proof_cannot_be_reused(same_date, monkeypatch, phase, revocation, later, present):
    lane = same_date
    if later:
        backup._append_json_log(lane.log, lane.receipt)
    revoked = dict(
        lane.receipt, **{revocation: "Drive download mismatch" if revocation == "verification_error" else False}
    )
    if later and not present:
        revoked["verified"] = False  # The original R1 repro: an older pass must not resurrect this copy.
    backup._append_json_log(lane.log, revoked)
    if not present:
        lane.archive.unlink()

    def fresh(*a, **kw):
        raise RuntimeError("fresh snapshot required")

    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", fresh)
    if phase == "upload":
        result = backup.run_backup(db_path=lane.db, staging_dir=lane.staging, log_path=lane.log, phase=phase)
        assert result.get("status") != "reused" and result["verified"] is False
    else:
        with pytest.raises(RuntimeError, match="fresh snapshot required"):
            backup.run_backup(db_path=lane.db, staging_dir=lane.staging, log_path=lane.log, phase=phase)
    facts = hv.backup_facts(backup._read_backup_receipts(lane.log))
    assert facts.local_verified_at is None and facts.offsite_verified_at is None


@pytest.mark.parametrize("phase", ["all", "upload"])
def test_r1_remote_reuse_probes_current_drive_consent(same_date, monkeypatch, phase):
    from brainlayer.drive_credentials import DriveCredentialError
    from brainlayer.observability_backup import _daily_snapshot

    lane = same_date
    backup._append_json_log(lane.log, lane.receipt)
    lane.archive.unlink()

    def expired():
        lane.probes.append("expired")
        raise DriveCredentialError("Drive access expired: Reconnect in BrainBar")

    monkeypatch.setattr(backup, "get_drive_credentials", expired)
    before = lane.log.read_bytes()
    if phase == "upload":
        result = backup.run_backup(db_path=lane.db, staging_dir=lane.staging, log_path=lane.log, phase=phase)
        assert result["upload_state"] == "nothing_pending"
        assert lane.log.read_bytes() == before
        assert lane.probes == []
        return
    with pytest.raises(DriveCredentialError, match="Drive access expired"):
        backup.run_backup(db_path=lane.db, staging_dir=lane.staging, log_path=lane.log, phase=phase)
    assert lane.probes == ["expired"]
    rows = backup._read_backup_receipts(lane.log)
    assert rows[-1]["error_type"] == "DriveCredentialError"
    assert rows[-1]["uploaded"] is False and rows[-1]["verified"] is False
    facts = hv.backup_facts(rows)
    assert facts.drive[1] == "consent_needed"
    assert facts.offsite_verified_at.isoformat() == lane.receipt["snapshot_created_at"]
    assert _daily_snapshot(rows)[1] == "DriveCredentialError"


@pytest.mark.parametrize("phase", ["all", "upload", "snapshot"])
def test_r1_reuse_drive_event_requires_todays_probe(same_date, phase):
    lane = same_date
    backup._append_json_log(lane.log, dict(lane.receipt, drive_auth_probe="ok"))
    expired = dict(
        attempted_at=(lane.now - dt.timedelta(minutes=5)).isoformat(),
        backup_log_provenance="real",
        error_type="DriveCredentialError",
        error="Drive access expired: Reconnect in BrainBar",
    )
    backup._append_json_log(lane.log, expired)
    before = lane.log.read_bytes()
    result = backup.run_backup(db_path=lane.db, staging_dir=lane.staging, log_path=lane.log, phase=phase)
    if phase == "upload":
        assert result["upload_state"] == "nothing_pending"
        assert lane.log.read_bytes() == before
        assert lane.probes == []
        assert hv.backup_facts(backup._read_backup_receipts(lane.log)).drive[1] == "consent_needed"
        return
    assert result["status"] == "reused"
    assert result["snapshot_created_at"] == lane.receipt["snapshot_created_at"]
    facts = hv.backup_facts(backup._read_backup_receipts(lane.log))
    if phase == "snapshot":
        assert lane.probes == []
        assert facts.drive[1] == "consent_needed"
    else:
        assert lane.probes == ["credentials"]
        assert result["drive_auth_probe"] == "ok"
        assert facts.drive == (dt.datetime.fromisoformat(result["attempted_at"]), "ok")


def test_r1_legacy_reused_receipt_does_not_clear_consent(same_date):
    lane = same_date
    failure = dict(
        attempted_at=(lane.now - dt.timedelta(minutes=5)).isoformat(),
        backup_log_provenance="real",
        error_type="DriveCredentialError",
        error="Drive access expired: Reconnect in BrainBar",
    )
    reused = dict(lane.receipt, status="reused", attempted_at=lane.now.isoformat())
    facts = hv.backup_facts([lane.receipt, failure, reused])
    assert facts.drive == (dt.datetime.fromisoformat(failure["attempted_at"]), "consent_needed")
    assert facts.offsite_verified_at.isoformat() == lane.receipt["snapshot_created_at"]


@pytest.mark.parametrize(
    "invalidation", [{"local_removed": True}, {"upload_state": "abandoned"}, {"upload_state": "superseded"}]
)
def test_r1_invalidated_local_proof_cannot_satisfy_snapshot(same_date, monkeypatch, invalidation):
    lane = same_date
    backup._append_json_log(lane.log, lane.receipt)
    backup._append_json_log(lane.log, dict(lane.receipt, **invalidation))

    def fresh(*a, **kw):
        raise RuntimeError("fresh snapshot required")

    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", fresh)
    with pytest.raises(RuntimeError, match="fresh snapshot required"):
        backup.run_backup(db_path=lane.db, staging_dir=lane.staging, log_path=lane.log, phase="snapshot")


@pytest.mark.parametrize(
    "proof,phase,present",
    [
        ("drive", "snapshot", True),
        ("drive", "upload", True),
        ("drive", "upload", False),
        ("drive", "all", True),
        ("drive", "all", False),
        ("local", "snapshot", True),
    ],
)
def test_same_date_verified_archive_reused(tmp_path, monkeypatch, capsys, phase, proof, present):
    monkeypatch.setenv("HOME", str(tmp_path))
    db = tmp_path / "source.db"
    staging = tmp_path / "backups"
    staging.mkdir()
    now = dt.datetime.now(dt.UTC)
    archive = staging / f"{backup.local_backup_date(now)}.db.zst"
    archive.write_bytes(b"verified archive")
    log = tmp_path / "backup.jsonl"
    receipt = dict(
        db=str(db),
        snapshot=str(archive),
        snapshot_created_at=now.isoformat(),
        attempted_at=now.isoformat(),
        machine_id="test-machine",
        drive_folder="test-folder",
        backup_log_provenance=backup._backup_log_provenance(),
        uploaded=proof == "drive",
        verified=proof == "drive",
        local_verified=True,
        local_artifact=backup._upload_identity(archive),
        local_integrity="ok",
        local_integrity_pragma="integrity_check",
        drive_file={"id": "verified-copy"} if proof == "drive" else None,
    )
    backup._append_json_log(log, receipt)
    monkeypatch.setenv("BRAINLAYER_DB", str(db))
    monkeypatch.setenv("BRAINLAYER_BACKUP_STAGING_DIR", str(staging))
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG", str(log))
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", phase)
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "test-machine")
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_FOLDER", "test-folder")
    monkeypatch.setattr(backup, "_backup_log_path", lambda *a, **kw: log)
    probes = []
    monkeypatch.setattr(backup, "get_drive_credentials", lambda: probes.append("credentials") or object())
    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", lambda *a, **kw: pytest.fail("new snapshot"))
    if not present:
        archive.unlink()  # Drive proof also works after normal local reclamation.
    from brainlayer import job_alerts

    alerts = []
    monkeypatch.delenv(backup.BACKUP_SUPERVISED_CHILD_ENV, raising=False)
    monkeypatch.setattr(backup, "_supervise_backup_process", backup._run_backup_process)
    monkeypatch.setattr(job_alerts, "report", lambda key, message: alerts.append(message))
    before = log.read_bytes()
    assert backup.main() == 0
    assert alerts == [None]
    if phase == "upload":
        result = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert result["upload_state"] == "nothing_pending"
        assert log.read_bytes() == before
        assert probes == []
        return
    result = json.loads(log.read_text().splitlines()[-1])
    assert result["status"] == "reused"
    assert result["snapshot"] == str(archive)
    assert result["snapshot_created_at"] == now.isoformat()


@pytest.mark.parametrize("phase", ["snapshot", "all"])
def test_unverified_same_date_file_blocks_without_overwrite(tmp_path, monkeypatch, phase):
    from tests.test_backup_daily import _create_source_db

    db = tmp_path / "source.db"
    _create_source_db(db, chunk_count=1)
    staging = tmp_path / "backups"
    staging.mkdir()
    archive = staging / f"{backup._today()}.db.zst"
    archive.write_bytes(b"partial")
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "test-machine")
    monkeypatch.setattr(backup, "_brainbar_writer_started_at", lambda *a: 0)
    import sqlite3

    def snapshot(target, **kwargs):
        with sqlite3.connect(db) as conn:
            conn.execute("VACUUM INTO ?", (str(target),))

    monkeypatch.setattr(backup, "request_brainbar_vacuum_into", snapshot)
    with pytest.raises(FileExistsError):
        backup.run_backup(db_path=db, staging_dir=staging, log_path=tmp_path / "receipt.jsonl", phase=phase)
    assert archive.read_bytes() == b"partial"
