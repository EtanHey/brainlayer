"""Small, real SQLite snapshots; no production database or BrainBar traffic."""

import dataclasses
import gzip
import os
import sqlite3
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


@pytest.mark.parametrize("compressor", ["gzip", "zstd"])
def test_real_snapshot_reclaims_then_stream_verifies(snapshot_lane, monkeypatch, compressor):
    source, output = snapshot_lane
    if not backup._probe_raw_reclamation(output)[0]:
        pytest.skip("staging filesystem cannot punch holes")
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
def test_streaming_drive_verification_never_writes_a_download(snapshot_lane, monkeypatch, compressor):
    from googleapiclient.http import HttpRequest
    from httplib2 import Response

    source, output = snapshot_lane
    if not backup._probe_raw_reclamation(output)[0]:
        pytest.skip("staging filesystem cannot punch holes")
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
    monkeypatch.setattr(backup, "COPY_CHUNK_BYTES", 1024)
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


def test_punch_failure_aborts_without_publishing_partial_snapshot(snapshot_lane, monkeypatch):
    source, output = snapshot_lane
    monkeypatch.setattr(backup, "_probe_raw_reclamation", lambda path: (True, None))

    def fail(*a):
        raise OSError("punch failed after selection")

    monkeypatch.setattr(backup, "_punch_raw_hole", fail)
    with pytest.raises(OSError, match="punch failed"):
        backup.create_sqlite_backup_artifact(source, output, "2026-10-26")
    assert list(output.iterdir()) == []
    with sqlite3.connect(source) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_raw_hash_survives_snapshot_receipt_resume(snapshot_lane, monkeypatch):
    source, output = snapshot_lane
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
