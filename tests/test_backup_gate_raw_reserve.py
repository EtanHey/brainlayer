"""Real admission and archive binding on synthetic staging files only."""

import fcntl
import gzip
import json
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
def test_verified_raw_no_longer_defers_admission(raw_admission, codec):
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
