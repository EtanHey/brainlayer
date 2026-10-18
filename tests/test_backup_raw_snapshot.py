"""Raw snapshot ownership and only-copy protection; no production I/O."""

import datetime as dt
import gzip
import json
import os
import signal
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as backup
from tests.test_backup_daily import _create_source_db, _stub_backup_for_drive_upload


@pytest.fixture
def raw_lane(tmp_path, monkeypatch):
    source, output = tmp_path / "source.db", tmp_path / "backups"
    _create_source_db(source)
    output.mkdir()
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "test-machine")
    monkeypatch.setattr(backup, "_brainbar_writer_started_at", lambda *a: 1)
    monkeypatch.setattr(backup, "_important_usage_capacity_bytes", lambda *a: None)
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda *a: SimpleNamespace(free=10**12))
    monkeypatch.setattr(
        backup, "request_brainbar_vacuum_into", lambda target, **kw: target.write_bytes(source.read_bytes())
    )
    return source, output


@pytest.mark.parametrize("exit_path", ["success", "stall", "timeout", "stop", "error", "resume"])
def test_raw_removed_before_any_upload_exit(raw_lane, monkeypatch, exit_path):
    source, output = raw_lane
    raw, compressed = output / "2026-10-17.db", output / "2026-10-17.db.gz"
    creator = backup.create_sqlite_backup_artifact
    _stub_backup_for_drive_upload(backup, monkeypatch, compressed)
    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", creator)
    monkeypatch.setattr(backup, "verify_drive_upload", lambda *a, **kw: None)
    if exit_path == "resume":
        raw.write_bytes(source.read_bytes())
        compressed.write_bytes(gzip.compress(source.read_bytes()))
        saved = {
            "snapshot": str(compressed),
            "sentinel_snapshot_chunks": 1,
            "snapshot_created_at": dt.datetime.now(dt.UTC).isoformat(),
            "resumable_upload": {
                "url": "https://www.googleapis.com/upload/drive/v3/files?upload_id=fake",
                "folder_id": "folder-id",
                **backup._upload_identity(compressed),
            },
        }
        monkeypatch.setattr(backup, "_stalled_uploads", lambda *a: [saved])
        monkeypatch.setattr(
            backup, "create_sqlite_backup_artifact", lambda *a, **kw: pytest.fail("resume recreated snapshot")
        )
    success = exit_path in {"success", "resume"}
    errors = {
        "stall": backup.DriveUploadStalledError(bytes_confirmed=1, total_bytes=2, attempts=3),
        "timeout": backup.BackupTimeoutError("deadline"),
        "stop": backup.BackupStoppedError(signal.SIGTERM),
        "error": RuntimeError("upload failed"),
    }

    def upload(*a, **kw):
        assert not raw.exists()
        with gzip.open(compressed) as stream:
            assert stream.read() == source.read_bytes()
        if not success:
            raise errors[exit_path]
        return {"id": "fake"}

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    args = dict(db_path=source, staging_dir=output, date_stamp="2026-10-17", log_path=output / "receipt.jsonl")
    if success:
        backup.run_backup(**args)
    else:
        with pytest.raises(type(errors[exit_path])):
            backup.run_backup(**args)
    assert not raw.exists()
    assert compressed.exists() is (not success)


@pytest.mark.parametrize("failure", ["compress", "integrity"])
def test_failed_gzip_keeps_only_raw_and_receipt(raw_lane, monkeypatch, failure):
    source, output = raw_lane
    target = "_copy_with_raw_floor" if failure == "compress" else "_gunzip_test"
    monkeypatch.setattr(backup, target, lambda *a: (_ for _ in ()).throw(RuntimeError(failure)))
    log = output / "receipt.jsonl"
    with pytest.raises(RuntimeError, match=failure):
        backup.run_backup(source, staging_dir=output, date_stamp="2026-10-17", upload=False, log_path=log)
    assert (output / "2026-10-17.db").read_bytes() == source.read_bytes()
    assert json.loads(log.read_text().splitlines()[-1])["local_uncompressed_snapshot"].endswith("2026-10-17.db")


@pytest.mark.parametrize(
    "kind",
    ["valid", "missing", "corrupt", "symlink", "hardlink", "outside", "gzip-symlink", "gzip-hardlink", "directory"],
)
def test_leftover_cleanup_precedes_capacity_and_preserves_only_copy(raw_lane, monkeypatch, kind):
    source, output = raw_lane
    raw = output / ("not-a-snapshot.db" if kind == "outside" else "2026-10-16.db")
    raw.write_bytes(source.read_bytes())
    if kind == "symlink":
        raw.unlink()
        raw.symlink_to(source)
    if kind == "hardlink":
        raw.unlink()
        os.link(source, raw)
    if kind != "missing":
        raw.with_suffix(".db.gz").write_bytes(b"broken" if kind == "corrupt" else gzip.compress(source.read_bytes()))
    compressed = raw.with_suffix(".db.gz")
    if kind in {"gzip-symlink", "gzip-hardlink"}:
        other = output / "other.gz"
        compressed.rename(other)
        if kind == "gzip-symlink":
            compressed.symlink_to(other)
        else:
            os.link(other, compressed)
    if kind == "directory":
        raw.unlink()
        raw.mkdir()
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda *a: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match="Insufficient free space") as error:
        backup.create_sqlite_backup_artifact(source, output, date_stamp="2026-10-17")
    assert raw.exists() is (kind != "valid")
    if kind in {"missing", "corrupt"}:
        assert raw.name in str(error.value)


def test_changed_raw_is_retained_and_same_date_only_copy_is_not_overwritten(raw_lane, monkeypatch):
    source, output = raw_lane
    raw = output / "2026-10-17.db"
    raw.write_bytes(b"only copy")
    with pytest.raises(FileExistsError):
        backup.create_sqlite_backup_artifact(source, output, date_stamp="2026-10-17")
    raw.with_suffix(".db.gz").write_bytes(gzip.compress(b"only copy"))

    def replace_during_check(*a):
        replacement = output / "replacement"
        replacement.write_bytes(b"replacement")
        replacement.replace(raw)

    monkeypatch.setattr(backup, "_gunzip_test", replace_during_check)
    with pytest.raises(ValueError, match="changed"):
        backup._remove_raw_with_verified_gzip(output, raw)
    assert raw.read_bytes() == b"replacement"
