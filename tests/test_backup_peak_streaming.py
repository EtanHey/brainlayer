"""Small, real SQLite snapshots; no production database or BrainBar traffic."""

import dataclasses
import os
import sqlite3

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
    source, output = snapshot_lane
    if not backup._probe_raw_reclamation(output)[0]:
        pytest.skip("staging filesystem cannot punch holes")
    monkeypatch.setenv(backup.BACKUP_COMPRESSOR_ENV, compressor)
    artifact = backup.create_sqlite_backup_artifact(source, output, "2026-10-26")
    payload = artifact.archive_path.read_bytes()
    calls = []

    def download(service, *, file_id, sink):
        calls.append(file_id)
        for start in range(0, len(payload), 1024):
            sink.write(payload[start : start + 1024])

    monkeypatch.setattr(backup, "_download_drive_to_stream", download)
    result = backup.verify_sqlite_backup_artifact(artifact, full=True, service=object(), file_id="test")
    assert result["verified"] is True, result
    assert result["drive_md5_match"] is True
    assert result["raw_sha256_match"] is True
    assert calls == ["test"]
    assert list(output.iterdir()) == [artifact.archive_path]


def test_probe_failure_logs_old_path(snapshot_lane, monkeypatch, capsys):
    source, output = snapshot_lane
    monkeypatch.setenv(backup.BACKUP_COMPRESSOR_ENV, "gzip")
    monkeypatch.setattr(backup, "_probe_raw_reclamation", lambda path: (False, "unsupported filesystem"))
    artifact = backup.create_sqlite_backup_artifact(source, output, "2026-10-26")
    assert artifact.raw_sha256 is None
    assert "unsupported filesystem" in capsys.readouterr().out
    assert backup.verify_sqlite_backup_artifact(artifact)["verified"] is True
