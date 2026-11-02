"""Same-date nightly retries use proven bytes without replacing the archive."""

import datetime as dt
import json

import pytest

from brainlayer import backup_daily as backup


@pytest.mark.parametrize(
    "proof,phase", [("drive", "snapshot"), ("drive", "upload"), ("drive", "all"), ("local", "snapshot")]
)
@pytest.mark.parametrize("present", [True, False])
def test_same_date_verified_archive_reused(tmp_path, monkeypatch, phase, proof, present):
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
    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", lambda *a, **kw: pytest.fail("new snapshot"))
    if proof == "local" and not present:
        pytest.skip("local reuse requires intact bytes")
    if not present:
        archive.unlink()  # Drive proof also works after normal local reclamation.
    from brainlayer import job_alerts

    alerts = []
    monkeypatch.delenv(backup.BACKUP_SUPERVISED_CHILD_ENV, raising=False)
    monkeypatch.setattr(backup, "_supervise_backup_process", backup._run_backup_process)
    monkeypatch.setattr(job_alerts, "report", lambda key, message: alerts.append(message))
    assert backup.main() == 0
    assert alerts == [None]
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
