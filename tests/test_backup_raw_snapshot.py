"""Raw snapshot ownership and only-copy protection; no production I/O."""

import datetime as dt
import gzip
import json
import os
import signal
import subprocess
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as backup
from brainlayer import disk_space
from tests.test_backup_daily import _create_source_db, _stub_backup_for_drive_upload


def _last_run_receipt(log, *, snapshot=True):
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    receipt = next(row for row in reversed(rows) if row.get("event") != "local_sweep")
    assert "attempted_at" in receipt
    assert receipt.get("event") != "local_sweep"
    if snapshot:
        assert "snapshot" in receipt
    return receipt


@pytest.fixture
def raw_lane(tmp_path, monkeypatch):
    monkeypatch.setattr(backup, "_probe_raw_reclamation", lambda _path: (False, "legacy-path fixture"))
    source, output = tmp_path / "source.db", tmp_path / "backups"
    _create_source_db(source)
    output.mkdir()
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "test-machine")
    monkeypatch.setattr(backup, "_brainbar_writer_started_at", lambda *a: 1)
    monkeypatch.setattr(disk_space, "_important_usage_capacity_bytes", lambda *a: None)
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda *a: SimpleNamespace(free=10**12))
    monkeypatch.setattr(
        backup, "request_brainbar_vacuum_into", lambda target, **kw: target.write_bytes(source.read_bytes())
    )
    return source, output


def _archive_bytes(data: bytes, suffix: str) -> bytes:
    if suffix == ".db.gz":
        return gzip.compress(data)
    return subprocess.run(
        [backup._require_zstd(), "-q", "-c", "--check"], input=data, capture_output=True, check=True
    ).stdout


def _unarchive(path) -> bytes:
    if path.name.endswith(".gz"):
        with gzip.open(path) as stream:
            return stream.read()
    return subprocess.run([backup._require_zstd(), "-d", "-q", "-c", str(path)], capture_output=True, check=True).stdout


@pytest.mark.parametrize("compressor,suffix", [("zstd", ".db.zst"), ("gzip", ".db.gz")])
@pytest.mark.parametrize("drive_md5", [True, False])
@pytest.mark.parametrize("exit_path", ["success", "stall", "timeout", "stop", "error", "resume"])
def test_raw_removed_before_any_upload_exit(raw_lane, monkeypatch, exit_path, compressor, suffix, drive_md5):
    source, output = raw_lane
    monkeypatch.setenv("BRAINLAYER_BACKUP_COMPRESSOR", compressor)
    raw, compressed = output / "2026-10-17.db", output / f"2026-10-17{suffix}"
    creator = backup.create_sqlite_backup_artifact
    _stub_backup_for_drive_upload(backup, monkeypatch, compressed)
    monkeypatch.setattr(backup, "create_sqlite_backup_artifact", creator)
    monkeypatch.setattr(backup, "verify_drive_upload", lambda *a, **kw: None)
    monkeypatch.setattr(
        backup,
        "verify_sqlite_backup_artifact",
        lambda *a, **kw: {"verified": True, "verification_mode": "quick", "local_md5": backup._md5_file(compressed)},
    )
    if exit_path == "resume":
        raw.write_bytes(source.read_bytes())
        compressed.write_bytes(_archive_bytes(source.read_bytes(), suffix))
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
        assert _unarchive(compressed) == source.read_bytes()
        if not success:
            raise errors[exit_path]
        return {"id": "fake", **({"md5Checksum": backup._md5_file(compressed)} if drive_md5 else {})}

    monkeypatch.setattr(backup, "upload_file_to_drive_raw", upload)
    args = dict(db_path=source, staging_dir=output, date_stamp="2026-10-17", log_path=output / "receipt.jsonl")
    if success:
        result = backup.run_backup(**args)
        assert result["local_removed_skip_reason"] == (None if drive_md5 else "md5_unverified")
    else:
        with pytest.raises(type(errors[exit_path])):
            backup.run_backup(**args)
    assert not raw.exists()
    assert compressed.exists() is (not success or not drive_md5)


@pytest.mark.parametrize("compressor", ["zstd", "gzip"])
@pytest.mark.parametrize("failure", ["compress", "integrity"])
def test_failed_gzip_keeps_only_raw_and_receipt(raw_lane, monkeypatch, failure, compressor):
    source, output = raw_lane
    monkeypatch.setenv("BRAINLAYER_BACKUP_COMPRESSOR", compressor)
    target = {
        ("gzip", "compress"): "_copy_with_raw_floor",
        ("gzip", "integrity"): "_gunzip_test",
        ("zstd", "compress"): "_compress_zstd",
        ("zstd", "integrity"): "_zstd_stream_digest",
    }[compressor, failure]
    monkeypatch.setattr(backup, target, lambda *a, **kw: (_ for _ in ()).throw(RuntimeError(failure)))
    log = output / "receipt.jsonl"
    with pytest.raises(RuntimeError, match=failure):
        backup.run_backup(source, staging_dir=output, date_stamp="2026-10-17", upload=False, log_path=log)
    assert (output / "2026-10-17.db").read_bytes() == source.read_bytes()
    assert _last_run_receipt(log, snapshot=False)["local_uncompressed_snapshot"].endswith("2026-10-17.db")


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


@pytest.mark.parametrize("own_run", [False, True])
@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_gzip_and_directory_durable_before_raw_unlink(raw_lane, monkeypatch, own_run, platform):
    source, output = raw_lane
    raw = output / "2026-10-16.db"
    raw.write_bytes(source.read_bytes())
    raw.with_suffix(".db.gz").write_bytes(gzip.compress(source.read_bytes()))
    events = []
    unlink = backup.os.unlink
    monkeypatch.setattr(backup.sys, "platform", platform)
    monkeypatch.setattr(backup.fcntl, "F_FULLFSYNC", 51, raising=False)
    monkeypatch.setattr(backup.fcntl, "fcntl", lambda fd, op: events.append(("fullsync", os.fstat(fd).st_ino, op)))
    monkeypatch.setattr(backup.os, "fsync", lambda fd: events.append(("fsync", os.fstat(fd).st_ino)))

    def record_unlink(name, **kwargs):
        events.append(("unlink", name))
        unlink(name, **kwargs)

    monkeypatch.setattr(backup.os, "unlink", record_unlink)
    backup._remove_raw_with_verified_gzip(output, raw, **({"own_run": True} if own_run else {}))
    compressed_inode = raw.with_suffix(".db.gz").stat().st_ino
    assert events == [
        ("fullsync", compressed_inode, 51) if platform == "darwin" else ("fsync", compressed_inode),
        ("fsync", output.stat().st_ino),
        ("unlink", raw.name),
    ]


@pytest.mark.parametrize("capacity", [True, False])
@pytest.mark.parametrize("same_size", [True, False])
def test_mismatched_prior_pair_retained_and_named_in_receipt(raw_lane, monkeypatch, capacity, same_size):
    source, output = raw_lane
    raw = output / "2026-10-16.db"
    original = source.read_bytes()
    raw.write_bytes(original)
    # Same size and valid SQLite header still cannot establish an earlier run's binding.
    different = original[:-1] + bytes([original[-1] ^ 1]) if same_size else b"other snapshot"
    compressed = raw.with_suffix(".db.gz")
    compressed.write_bytes(gzip.compress(different))
    compressed_bytes = compressed.read_bytes()
    if not capacity:
        monkeypatch.setattr(backup.shutil, "disk_usage", lambda *a: SimpleNamespace(free=0))
    log = output / "receipt.jsonl"
    args = dict(db_path=source, staging_dir=output, date_stamp="2026-10-17", upload=False, log_path=log)
    if capacity:
        backup.run_backup(**args)
    else:
        with pytest.raises(RuntimeError, match=raw.name):
            backup.run_backup(**args)
    assert raw.read_bytes() == original
    assert compressed.read_bytes() == compressed_bytes
    receipt = _last_run_receipt(log, snapshot=False)
    assert raw.name in receipt["retained_raw_snapshots"]


@pytest.mark.parametrize("content", [b"not SQLite", b"SQLite format 3\x00short"])
def test_own_run_requires_sqlite_header_and_isize(raw_lane, content):
    source, output = raw_lane
    raw = output / "2026-10-17.db"
    raw.write_bytes(source.read_bytes())
    content = content.ljust(raw.stat().st_size, b"x") if content == b"not SQLite" else content
    raw.with_suffix(".db.gz").write_bytes(gzip.compress(content))
    with pytest.raises(ValueError, match="binding"):
        backup._remove_raw_with_verified_gzip(output, raw, own_run=True)
    assert raw.exists()


@pytest.mark.parametrize("phase", ["gzip", "directory"])
def test_durability_failure_keeps_both_files(raw_lane, monkeypatch, phase):
    source, output = raw_lane
    raw = output / "2026-10-16.db"
    raw.write_bytes(source.read_bytes())
    compressed = raw.with_suffix(".db.gz")
    compressed.write_bytes(gzip.compress(source.read_bytes()))
    monkeypatch.setattr(backup.sys, "platform", "linux")

    def fail_sync(fd):
        import stat

        is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if is_directory == (phase == "directory"):
            raise OSError("sync failed")

    monkeypatch.setattr(backup.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="sync failed"):
        backup._remove_raw_with_verified_gzip(output, raw)
    assert raw.exists() and compressed.exists()


@pytest.mark.parametrize("compressor", ["zstd", "gzip"])
def test_compression_floor_breach_names_retained_raw_with_stalled_gzip(raw_lane, monkeypatch, compressor):
    source, output = raw_lane
    monkeypatch.setenv("BRAINLAYER_BACKUP_COMPRESSOR", compressor)
    raw = output / "2026-10-17.db"
    stalled_gzip = output / "2026-10-16.db.gz"
    stalled_gzip.write_bytes(b"retained stalled gzip")
    monkeypatch.setattr(
        backup, "_stalled_uploads", lambda *a: [{"snapshot": str(stalled_gzip), "resumable_upload": {}}]
    )
    monkeypatch.setattr(
        backup,
        "_copy_with_raw_floor" if compressor == "gzip" else "_compress_zstd",
        lambda *a: (_ for _ in ()).throw(RuntimeError("raw free space floor breached")),
    )
    log = output / "receipt.jsonl"
    with pytest.raises(backup.RetainedStalledBackupCapacityError, match=raw.name):
        backup.run_backup(source, staging_dir=output, date_stamp="2026-10-17", log_path=log)
    assert raw.exists() and stalled_gzip.exists()
    receipt = _last_run_receipt(log, snapshot=False)
    assert raw.name in receipt["error"]
    assert raw.name in receipt["retained_raw_snapshots"]


def test_hashing_reads_in_bounded_chunks():
    import hashlib

    class GuardedStream:
        remaining = 3

        def read(self, size):
            assert 0 < size <= backup.COPY_CHUNK_BYTES
            self.remaining -= 1
            return b"block" if self.remaining >= 0 else b""

    assert backup._stream_sha256(GuardedStream()) == hashlib.sha256(b"block" * 3).digest()
