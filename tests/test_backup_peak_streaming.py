"""Small, real SQLite snapshots; no production database or BrainBar traffic."""

import dataclasses
import gzip
import json
import os
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as backup


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("phase", ["raw_copy", "gzip", "restore", "drive_download"])
def test_reclaiming_phase_budget(full, phase):
    db, compressed = 22_000_000_000, 9_600_000_000
    phases = backup._backup_phase_peaks(db, compressed, full=full, reclaim_raw=True)
    expansion = ((db + backup.RECLAIM_CHUNK_BYTES - 1) // backup.RECLAIM_CHUNK_BYTES) * backup.FRAME_OVERHEAD_BYTES
    expected = {
        "raw_copy": db,
        "gzip": db + expansion + backup.RECLAIM_CHUNK_BYTES,
        "restore": compressed,
    }
    if full:
        expected["drive_download"] = compressed
    assert phases.get(phase) == expected.get(phase)
    assert backup._peak_backup_required_bytes(db, compressed, full=full, reclaim_raw=True) == (
        expected["gzip"] + backup.MIN_RAW_FREE_BYTES + backup.COPY_CHUNK_BYTES
    )
    assert backup._peak_backup_required_bytes(db, compressed, full=full, reclaim_raw=True) < 24_300_000_000


@pytest.mark.parametrize("full", [False, True])
def test_unavailable_reclamation_keeps_old_budget(full):
    db, compressed = 22_000_000_000, 9_600_000_000
    assert backup._peak_backup_required_bytes(db, compressed, full=full, reclaim_raw=False) == (
        db + compressed * (2 if full else 1) + backup.MIN_RAW_FREE_BYTES + backup.COPY_CHUNK_BYTES
    )


@pytest.fixture
def snapshot_lane(tmp_path, monkeypatch):
    source, output = tmp_path / "source.db", tmp_path / "out"
    output.mkdir()
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, content BLOB)")
        conn.executemany("INSERT INTO chunks VALUES (?, ?)", [(str(n), os.urandom(4096)) for n in range(24)])
    monkeypatch.setattr(backup, "_brainbar_writer_started_at", lambda *a: 1)
    monkeypatch.setattr(backup, "_important_usage_capacity_bytes", lambda *a: None)
    monkeypatch.setattr(backup, "RECLAIM_CHUNK_BYTES", 16384)

    def snapshot(target, **kwargs):
        with sqlite3.connect(source) as conn:
            conn.execute("VACUUM INTO ?", (str(target),))
        return target.stat()

    monkeypatch.setattr(backup, "request_brainbar_vacuum_into", snapshot)
    return source, output


@pytest.fixture
def reclaim_volume(snapshot_lane, monkeypatch):
    _, output = snapshot_lane
    volume = SimpleNamespace(released=0, background=0, archive_bytes=0)

    def disk_usage(path):
        archive_bytes = sum(p.stat().st_size for p in output.rglob("*.db.*") if p.is_file())
        if archive_bytes > volume.archive_bytes:
            # Other apps allocate between frames, outside the bracket around each punch.
            volume.background += 96 * 1024 * 1024
        volume.archive_bytes = archive_bytes
        return SimpleNamespace(free=10**12 - archive_bytes - volume.background + volume.released)

    monkeypatch.setattr(backup.shutil, "disk_usage", disk_usage)
    return volume


@pytest.fixture(params=["portable", "native"])
def punch_lane(snapshot_lane, reclaim_volume, monkeypatch, request):
    _, output = snapshot_lane
    if request.param == "native":
        if not backup._probe_raw_reclamation(output)[0]:
            pytest.skip("staging filesystem cannot punch holes")
        monkeypatch.setattr(backup, "_probe_raw_reclamation", lambda path: (True, None))
        real_punch = backup._punch_raw_hole

        def native_punch(fd, offset, length):
            real_punch(fd, offset, length)
            reclaim_volume.released += length

        monkeypatch.setattr(backup, "_punch_raw_hole", native_punch)
        return snapshot_lane
    # Zero consumed bytes everywhere; emulate only allocation accounting. Production reads
    # the real fd's st_blocks. This fake never claims the filesystem freed physical storage.
    released = {}
    real_stat = os.fstat

    class AllocationStat:
        def __init__(self, info, blocks):
            self.info, self.st_blocks = info, blocks

        def __getattr__(self, name):
            return getattr(self.info, name)

    def fstat(fd):
        info = real_stat(fd)
        blocks = max(0, info.st_blocks - released.get((info.st_dev, info.st_ino), 0))
        return AllocationStat(info, blocks)

    def punch(fd, offset, length):
        info = real_stat(fd)
        assert os.pwrite(fd, b"\0" * length, offset) == length
        key = info.st_dev, info.st_ino
        released[key] = released.get(key, 0) + (length + 511) // 512
        reclaim_volume.released += length

    monkeypatch.setattr(backup, "_probe_raw_reclamation", lambda path: (True, None))
    monkeypatch.setattr(backup, "_punch_raw_hole", punch)
    monkeypatch.setattr(backup.os, "fstat", fstat)
    # Exercise the missing-F_FULLFSYNC platform on macOS too.
    monkeypatch.setattr(backup.sys, "platform", "linux")
    monkeypatch.delattr(backup.fcntl, "F_FULLFSYNC", raising=False)
    return snapshot_lane


@pytest.mark.parametrize("compressor", ["gzip", "zstd"])
def test_real_snapshot_reclaims_then_stream_verifies(punch_lane, monkeypatch, compressor):
    source, output = punch_lane
    monkeypatch.setenv(backup.BACKUP_COMPRESSOR_ENV, compressor)
    checked, punched = [], []
    real_check, real_punch = backup._sqlite_pragma_check, backup._punch_raw_hole

    def check(path, pragma):
        checked.append(pragma)
        return real_check(path, pragma)

    def punch(fd, offset, length):
        assert checked == ["integrity_check"]
        before = os.fstat(fd).st_blocks
        real_punch(fd, offset, length)
        punched.append((before, os.fstat(fd).st_blocks))

    monkeypatch.setattr(backup, "_sqlite_pragma_check", check)
    monkeypatch.setattr(backup, "_probe_raw_reclamation", lambda path: (True, None))
    monkeypatch.setattr(backup, "_punch_raw_hole", punch)
    artifact = backup.create_sqlite_backup_artifact(source, output, "2026-10-26")
    assert punched and any(after < before for before, after in punched)
    assert artifact.raw_sha256
    assert backup.verify_sqlite_backup_artifact(artifact)["verified"] is True
    assert list(output.iterdir()) == [artifact.archive_path]
    restored = source.parent / "restored.db"
    backup._decompress_archive_to(artifact.archive_path, restored)
    with sqlite3.connect(restored) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 24

    def no_restore(*a, **kw):
        pytest.fail("stream verification must not materialize a restored database")

    monkeypatch.setattr(backup, "_decompress_archive_to", no_restore)
    assert backup.verify_sqlite_backup_artifact(artifact)["verified"] is True
    wrong = dataclasses.replace(artifact, raw_sha256="0" * 64)
    assert backup.verify_sqlite_backup_artifact(wrong)["verified"] is False
    data = bytearray(artifact.archive_path.read_bytes())
    data[len(data) // 2] ^= 0x80
    artifact.archive_path.write_bytes(data)
    assert backup.verify_sqlite_backup_artifact(artifact)["verified"] is False


@pytest.mark.parametrize("compressor", ["gzip", "zstd"])
def test_streaming_drive_verification_never_writes_a_download(punch_lane, monkeypatch, compressor):
    from googleapiclient.http import HttpRequest
    from httplib2 import Response

    source, output = punch_lane
    monkeypatch.setenv(backup.BACKUP_COMPRESSOR_ENV, compressor)
    artifact = backup.create_sqlite_backup_artifact(source, output, "2026-10-26")
    payload = artifact.archive_path.read_bytes()
    calls = []

    class HTTP:
        def request(self, uri, method, *, headers):
            calls.append(headers["range"])
            start, end = map(int, headers["range"].removeprefix("bytes=").split("-"))
            end = min(end, len(payload) - 1)
            return Response({"status": "206", "content-range": f"bytes {start}-{end}/{len(payload)}"}), payload[
                start : end + 1
            ]

    service = SimpleNamespace(
        files=lambda: SimpleNamespace(get_media=lambda **kw: HttpRequest(HTTP(), lambda *a: None, "https://test/"))
    )
    monkeypatch.setattr(backup, "DRIVE_DOWNLOAD_CHUNK_BYTES", 1024, raising=False)
    result = backup.verify_sqlite_backup_artifact(artifact, full=True, service=service, file_id="test")
    assert result["verified"] is True, result
    assert result["drive_md5_match"] is True
    assert result["raw_sha256_match"] is True
    assert len(calls) > 1
    assert list(output.iterdir()) == [artifact.archive_path]
    payload = payload[:-5]
    assert (
        backup.verify_sqlite_backup_artifact(artifact, full=True, service=service, file_id="test")["verified"] is False
    )


def test_punch_failure_aborts_without_publishing_partial_snapshot(punch_lane, monkeypatch):
    source, output = punch_lane
    monkeypatch.setattr(backup, "_probe_raw_reclamation", lambda path: (True, None))

    def fail(*a):
        raise OSError("punch failed after selection")

    monkeypatch.setattr(backup, "_punch_raw_hole", fail)
    with pytest.raises(OSError, match="punch failed"):
        backup.create_sqlite_backup_artifact(source, output, "2026-10-26")
    assert list(output.iterdir()) == []
    with sqlite3.connect(source) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_raw_hash_survives_snapshot_receipt_resume(punch_lane, monkeypatch):
    source, output = punch_lane
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "test-machine")
    monkeypatch.setenv(backup.BACKUP_COMPRESSOR_ENV, "gzip")
    kwargs = dict(
        db_path=source, staging_dir=output, date_stamp="2026-10-26", upload=False, log_path=output / "log.jsonl"
    )
    created = backup.run_backup(**kwargs)
    assert created["raw_sha256"]
    monkeypatch.setattr(
        backup, "create_sqlite_backup_artifact", lambda *a, **kw: pytest.fail("unexpected new snapshot")
    )
    resumed = backup.run_backup(**kwargs)
    assert resumed["raw_sha256"] == created["raw_sha256"]


def test_probe_failure_logs_old_path(snapshot_lane, monkeypatch, capsys):
    source, output = snapshot_lane
    monkeypatch.setenv(backup.BACKUP_COMPRESSOR_ENV, "gzip")
    monkeypatch.setattr(backup, "_probe_raw_reclamation", lambda path: (False, "unsupported filesystem"))
    artifact = backup.create_sqlite_backup_artifact(source, output, "2026-10-26")
    assert artifact.raw_sha256 is None
    assert "unsupported filesystem" in capsys.readouterr().out
    assert backup.verify_sqlite_backup_artifact(artifact)["verified"] is True


def test_reclaimed_gzip_members_bind_to_a_leftover_raw(snapshot_lane):
    source, output = snapshot_lane
    raw = output / "2026-10-26.db"
    raw.write_bytes(source.read_bytes())
    data = raw.read_bytes()
    archive = raw.with_suffix(".db.gz")
    archive.write_bytes(gzip.compress(data[:16384]) + gzip.compress(data[16384:]))
    backup._remove_raw_with_verified_gzip(output, raw)
    assert not raw.exists()


@pytest.mark.parametrize("compressor", ["gzip", "zstd"])
@pytest.mark.parametrize("failure", ["compress", "integrity", "floor"])
def test_reclaiming_failure_cleans_attempt_and_records_error(punch_lane, monkeypatch, compressor, failure):
    source, output = punch_lane
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "test-machine")
    monkeypatch.setenv(backup.BACKUP_COMPRESSOR_ENV, compressor)
    punches = []
    real_punch = backup._punch_raw_hole

    def punch(*args):
        real_punch(*args)
        punches.append(args[1:])

    monkeypatch.setattr(backup, "_punch_raw_hole", punch)
    if failure == "compress":
        if compressor == "gzip":
            real_compress = gzip.compress

            def compress(*a, **kw):
                if punches:
                    raise RuntimeError("gzip compressor failed mid-stream")
                return real_compress(*a, **kw)

            monkeypatch.setattr(backup.gzip, "compress", compress)
            error, message = RuntimeError, "gzip compressor failed"
        else:
            real_run = subprocess.run

            def run(command, **kw):
                if punches and "-T1" in command:
                    raise subprocess.CalledProcessError(9, command, stderr=b"compressor failed")
                return real_run(command, **kw)

            monkeypatch.setattr(backup.subprocess, "run", run)
            error, message = subprocess.CalledProcessError, "exit status 9"
    elif failure == "integrity":
        monkeypatch.setattr(backup, "_archive_stream_sha256", lambda *a, **kw: "0" * 64)
        error, message = ValueError, "SHA-256"
    else:
        stalled = output / "2026-10-25.db.gz"
        stalled.write_bytes(b"stalled archive remains untouched")
        monkeypatch.setattr(backup, "_stalled_uploads", lambda *a: [{"snapshot": str(stalled), "resumable_upload": {}}])
        real_floor = backup._require_raw_floor

        def floor(path, *, forthcoming=0):
            if punches:
                # Drop the actual free-space reading only once compression has begun.
                monkeypatch.setattr(
                    backup.shutil, "disk_usage", lambda *a: SimpleNamespace(free=backup.MIN_RAW_FREE_BYTES - 1)
                )
            return real_floor(path, forthcoming=forthcoming)

        monkeypatch.setattr(backup, "_require_raw_floor", floor)
        error, message = backup.RetainedStalledBackupCapacityError, "raw free space floor breached"
    log = output / "receipt.jsonl"
    with pytest.raises(error, match=message):
        backup.run_backup(source, staging_dir=output, date_stamp="2026-10-26", upload=failure == "floor", log_path=log)
    assert punches  # A real prefix was consumed before the injected failure.
    assert not (output / "2026-10-26.db").exists()
    assert not list(output.glob("2026-10-26.db.*"))
    assert not list(output.glob("brainlayer-backup-*"))
    receipt = json.loads(log.read_text().splitlines()[-1])
    assert message in receipt["error"]
    assert receipt["verified"] is False
    if failure == "floor":
        assert stalled.read_bytes() == b"stalled archive remains untouched"
        assert stalled.name in receipt["error"]
    with sqlite3.connect(source) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_punch_that_frees_no_blocks_aborts(punch_lane, monkeypatch):
    source, output = punch_lane
    monkeypatch.setattr(backup, "_punch_raw_hole", lambda fd, offset, length: os.pwrite(fd, b"\0" * length, offset))
    with pytest.raises(RuntimeError, match="punched raw extents not reclaimed.*APFS snapshot"):
        backup.create_sqlite_backup_artifact(source, output, "2026-10-26")
    assert list(output.iterdir()) == []


def test_snapshot_pins_volume_space_despite_file_block_drop(punch_lane, monkeypatch):
    source, output = punch_lane
    punched = 0
    real_punch = backup._punch_raw_hole
    free = 10**12
    monkeypatch.setattr(backup, "RECLAIM_VOLUME_SLACK_BYTES", backup.RECLAIM_CHUNK_BYTES)

    def punch(fd, offset, length):
        nonlocal punched, free
        real_punch(fd, offset, length)
        punched += length
        free -= length  # No physical reclaim: volume free falls as the archive grows.

    monkeypatch.setattr(backup, "_punch_raw_hole", punch)
    monkeypatch.setattr(backup.shutil, "disk_usage", lambda *a: SimpleNamespace(free=free))
    with pytest.raises(RuntimeError, match="punched raw extents not reclaimed.*APFS snapshot"):
        backup.create_sqlite_backup_artifact(source, output, "2026-10-26")
    assert punched <= 2 * backup.RECLAIM_CHUNK_BYTES
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("compressor", ["gzip", "zstd"])
def test_background_volume_drift_does_not_abort_healthy_punches(punch_lane, reclaim_volume, monkeypatch, compressor):
    source, output = punch_lane
    monkeypatch.setenv(backup.BACKUP_COMPRESSOR_ENV, compressor)
    artifact = backup.create_sqlite_backup_artifact(source, output, "2026-10-26")
    assert reclaim_volume.background >= 300 * 1024 * 1024
    assert reclaim_volume.released == source.stat().st_size
    assert backup.verify_sqlite_backup_artifact(artifact)["verified"] is True
    with sqlite3.connect(source) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("kind", ["consumed", "unmarked", "mismatch", "date_mismatch", "outside", "symlink", "hardlink"])
def test_startup_consumed_cleanup_preserves_archives_and_unconsumed_raw(snapshot_lane, monkeypatch, kind):
    source, output = snapshot_lane
    directory = output / ("other-job" if kind == "outside" else "brainlayer-backup-crashed")
    directory.mkdir()
    raw = directory / "2026-10-25.db"
    raw.write_bytes(source.read_bytes())
    (directory / ".backup-date").write_text("2026-10-24" if kind == "date_mismatch" else "2026-10-25")
    if kind == "symlink":
        raw.unlink()
        raw.symlink_to(source)
    elif kind == "hardlink":
        raw.unlink()
        os.link(source, raw)
    info = raw.stat()
    marker = {"raw": raw.name, "dev": info.st_dev, "ino": info.st_ino + (kind == "mismatch"), "consumed": True}
    if kind != "unmarked":
        (directory / ".consumed").write_text(json.dumps(marker))
    archive = directory / "2026-10-25.db.gz"
    archive.write_bytes(b"archive is never removed by startup cleanup")

    def capacity(*a):
        assert raw.exists() is (kind != "consumed")  # Must run before admission.
        return SimpleNamespace(free=0)

    monkeypatch.setattr(backup.shutil, "disk_usage", capacity)
    monkeypatch.setattr(backup, "_probe_raw_reclamation", lambda *a: (False, "test"))
    with pytest.raises(RuntimeError, match="Insufficient free space"):
        backup.create_sqlite_backup_artifact(source, output, "2026-10-26")
    assert archive.read_bytes() == b"archive is never removed by startup cleanup"
    assert source.exists()


def test_consumed_marker_is_durable_before_first_punch(punch_lane, monkeypatch):
    source, output = punch_lane
    real_punch = backup._punch_raw_hole
    marker_seen = []

    def punch(fd, offset, length):
        raw = next(output.glob("brainlayer-backup-*/2026-10-26.db"))
        marker = json.loads((raw.parent / ".consumed").read_text())
        info = os.fstat(fd)
        assert marker == {"raw": raw.name, "dev": info.st_dev, "ino": info.st_ino, "consumed": True}
        marker_seen.append(True)
        real_punch(fd, offset, length)

    monkeypatch.setattr(backup, "_punch_raw_hole", punch)
    backup.create_sqlite_backup_artifact(source, output, "2026-10-26")
    assert marker_seen


def test_stream_download_chunk_size_and_bounded_retry(monkeypatch):
    import io

    from googleapiclient.http import HttpRequest
    from httplib2 import Response

    calls, sleeps = [], []

    class HTTP:
        def request(self, uri, method, *, headers):
            calls.append(headers["range"])
            if len(calls) < 3:
                raise ConnectionResetError("transient reset")
            return Response({"status": "200", "content-length": "3"}), b"abc"

    service = SimpleNamespace(
        files=lambda: SimpleNamespace(get_media=lambda **kw: HttpRequest(HTTP(), lambda *a: None, "https://test/"))
    )
    monkeypatch.setattr(backup, "_sleep", sleeps.append)
    monkeypatch.setenv(backup.DRIVE_UPLOAD_STALL_MAX_ATTEMPTS_ENV, "3")
    sink = io.BytesIO()
    backup._download_drive_to_stream(service, file_id="test", sink=sink)
    assert sink.getvalue() == b"abc"
    assert calls == ["bytes=0-8388607"] * 3
    assert sleeps == [2, 4]
    calls.clear()
    monkeypatch.setenv(backup.DRIVE_UPLOAD_STALL_MAX_ATTEMPTS_ENV, "2")
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        backup._download_drive_to_stream(service, file_id="test", sink=io.BytesIO())
    assert len(calls) == 2
