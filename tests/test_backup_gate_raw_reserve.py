"""Real admission and archive binding on synthetic staging files only."""

import fcntl
import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from brainlayer import backup_daily as backup
from brainlayer import backup_heavy_gate as gate


@pytest.fixture
def raw_admission(tmp_path, monkeypatch):
    import brainlayer

    assert Path(brainlayer.__file__).resolve().is_relative_to(Path.cwd())
    raw = tmp_path / "2026-11-23.db"
    payload = b"SQLite format 3\x00" + b"x" * 1024**2
    raw.write_bytes(payload)
    archive = tmp_path / (raw.name + ".gz")
    archive.write_bytes(gzip.compress(payload))
    journal = tmp_path / "receipts.jsonl"
    journal.write_text(json.dumps({"snapshot": str(archive), "local_verified": True}) + "\n")
    monkeypatch.setenv("BRAINLAYER_BACKUP_STAGING_DIR", str(tmp_path))
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PATH", str(journal))
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST_FREE_GIB", "40")
    monkeypatch.setenv("BRAINLAYER_BACKUP_PHASE", "snapshot")
    monkeypatch.setattr(backup, "get_db_path", lambda: tmp_path / "source.db")
    monkeypatch.setattr(backup, "_should_run_full_verify", lambda _: False)
    monkeypatch.setattr(backup, "_database_logical_size_bytes", lambda _: len(payload))
    monkeypatch.setattr(backup, "_estimated_gzip_bytes", lambda *_: 1024)
    monkeypatch.setattr(backup.backup_clone, "supported", lambda *_: False)
    monkeypatch.setattr(backup, "_peak_backup_required_bytes", lambda *a, **kw: 2 * 1024**3)

    def space(path, *, include_available):
        assert path == tmp_path
        assert include_available is False
        # Between old (peak + raw + floor) and new (peak + floor) requirements.
        return SimpleNamespace(free_now_bytes=42 * 1024**3 + len(payload) // 2)

    monkeypatch.setattr(backup.disk_space, "measure_disk_space", space)
    return raw, archive, payload


@pytest.mark.parametrize("codec", ["gzip", "zstd"])
def test_verified_raw_no_longer_defers_admission(raw_admission, codec, capsys):
    raw, archive, payload = raw_admission
    if codec == "zstd":
        binary = shutil.which("zstd")
        if binary is None:
            pytest.skip("zstd executable unavailable")
        archive = raw.with_suffix(".db.zst")
        subprocess.run([binary, "-q", "--check", str(raw), "-o", str(archive)], check=True)
    assert gate._disk_reason("m4") is None
    assert not raw.exists()
    assert archive.exists()
    # The launchd log traces every raw the gate reclaims.
    assert f"Backup removed verified raw snapshot before admission: {raw}" in capsys.readouterr().out


def _spy(monkeypatch, name):
    calls = []
    real = getattr(backup, name)
    monkeypatch.setattr(backup, name, lambda *a, **kw: calls.append(a) or real(*a, **kw))
    return calls


def test_failed_proof_runs_once_until_raw_or_archive_changes(raw_admission, monkeypatch):
    raw, archive, payload = raw_admission
    archive.write_bytes(gzip.compress(payload[:-1] + b"y"))  # Exists, same size: fails only after full passes.
    hashed, gunzipped = _spy(monkeypatch, "_stream_sha256"), _spy(monkeypatch, "_gunzip_test")
    failed_proofs = set()
    for _ in range(2):
        assert raw.name in gate._disk_reason("m4", failed_proofs)
    assert (len(hashed), len(gunzipped)) == (1, 1)
    for touched, expected in ((raw, 2), (archive, 3)):
        info = touched.stat()
        os.utime(touched, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
        assert raw.name in gate._disk_reason("m4", failed_proofs)
        assert raw.name in gate._disk_reason("m4", failed_proofs)
        assert (len(hashed), len(gunzipped)) == (expected, expected)
    assert raw.read_bytes() == payload


@pytest.mark.parametrize("free_gib", [100, 1])
def test_no_proof_when_reclaiming_cannot_change_admission(raw_admission, monkeypatch, free_gib):
    raw, _, _ = raw_admission
    monkeypatch.setattr(
        backup.disk_space, "measure_disk_space", lambda *a, **kw: SimpleNamespace(free_now_bytes=free_gib * 1024**3)
    )
    cleaned = _spy(monkeypatch, "_cleanup_raw_snapshots")
    reason = gate._disk_reason("m4")
    assert (reason is None) is (free_gib == 100)  # Room already, or short even without the raw.
    assert cleaned == []
    assert raw.exists()


def _run_gate(monkeypatch, staging, *, iterations):
    clock = [0.0]
    monkeypatch.setattr(
        gate, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0] + s))
    )
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_HOST", "m4")
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_LOCK_PATH", str(staging / "heavy.lock"))
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_LOG_PATH", str(staging / "heavy.log"))
    monkeypatch.setenv("BRAINLAYER_BACKUP_HEAVY_RETRY_SECONDS", "10")
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(staging / "alerts.json"))
    with pytest.raises(gate.BackupGateDeferred):
        with gate.backup_heavy_gate(deadline=10 * iterations - 5):
            pytest.fail("admitted")
    assert clock[0] == 10 * iterations - 5


def test_pressure_defers_before_any_proof(raw_admission, monkeypatch):
    raw, _, _ = raw_admission  # A verified pair the disk check would reclaim.
    monkeypatch.setattr(gate, "_pressure_reason", lambda *args: "swap too high")
    cleaned = _spy(monkeypatch, "_cleanup_raw_snapshots")
    _run_gate(monkeypatch, raw.parent, iterations=2)
    assert cleaned == []
    assert raw.exists()


def test_gate_retries_keep_the_failed_proof_memo(raw_admission, monkeypatch):
    raw, archive, payload = raw_admission
    archive.write_bytes(gzip.compress(payload[:-1] + b"y"))
    monkeypatch.setattr(gate, "_pressure_reason", lambda *args: None)
    hashed = _spy(monkeypatch, "_stream_sha256")
    _run_gate(monkeypatch, raw.parent, iterations=3)
    assert len(hashed) == 1
    assert raw.read_bytes() == payload


@pytest.mark.parametrize("unverified", ["mismatch", "missing"])
def test_unverified_raw_still_reserves_and_names_its_growth(raw_admission, unverified):
    raw, archive, payload = raw_admission
    if unverified == "missing":
        archive.unlink()
    else:
        archive.write_bytes(gzip.compress(payload[:-1] + b"y"))
    reason = gate._disk_reason("m4")
    assert reason is not None
    assert f"{raw.name}=0.001 GiB" in reason
    assert raw.read_bytes() == payload


def test_busy_staging_lock_keeps_verified_raw_reserved(raw_admission):
    raw, _, _ = raw_admission
    with (raw.parent / ".backup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reason = gate._disk_reason("m4")
        assert reason is not None
        assert raw.name in reason
        assert raw.exists()
