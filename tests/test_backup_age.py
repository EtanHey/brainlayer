import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from brainlayer import backup_age
from brainlayer import jsonl_backup as backup
from brainlayer.job_alerts import active_alerts
from tests.test_jsonl_backup import _mock_drive_success, _write_jsonl


@pytest.fixture
def keys(tmp_path, monkeypatch):
    if os.sys.platform != "darwin":
        monkeypatch.setattr(backup_age, "executable", lambda name="age": Path("/usr/bin") / name)
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setattr(backup_age, "config_dir", lambda: config)
    monkeypatch.setattr(backup_age, "private_identity", lambda: None)
    generated = subprocess.run([str(backup_age.executable("age-keygen"))], capture_output=True, check=True)
    public = (
        subprocess.run(
            [str(backup_age.executable("age-keygen")), "-y"], input=generated.stdout, capture_output=True, check=True
        )
        .stdout.decode()
        .strip()
    )
    (config / "backup-age-recipient.txt").write_text(public)
    (config / "backup-age-recovery-confirmed").touch()
    return config, generated.stdout


def test_encrypted_upload_roundtrip_and_receipt(tmp_path, monkeypatch, keys):
    uploads = []
    _mock_drive_success(backup, monkeypatch, uploads)
    monkeypatch.setattr(backup.backup_daily, "resolve_machine_id", lambda: "fixture")

    def upload(path, *args, **kwargs):
        uploads.append(path)
        return {"id": "fixture", "md5Checksum": backup._md5_file(path)}

    monkeypatch.setattr(backup.backup_daily, "upload_file_to_drive_raw", upload)
    monkeypatch.setenv("BRAINLAYER_JSONL_FOREVER", "1")
    source = tmp_path / "sources"
    _write_jsonl(source / "session.jsonl", mtime=time.time() - 3600)
    result = backup.run_backup(
        source_roots=[source],
        state_path=tmp_path / "state.json",
        staging_dir=tmp_path / "staging",
        log_path=tmp_path / "log",
        queue_dir=tmp_path / "queue",
    )
    assert result["uploaded"]
    assert len(uploads) == 2 and all(p.name.endswith(".age") for p in uploads)
    archive = Path(result["archive"])
    assert not archive.with_suffix("").exists()
    assert backup_age.decrypted_sha256(archive, keys[1]) == result["plaintext_sha256"]
    receipt = json.loads((tmp_path / "state.json").read_text())["drive_archives"][archive.name]
    assert receipt["plaintext_sha256"] == result["plaintext_sha256"]
    assert receipt["size"] == archive.stat().st_size
    assert receipt["archive_md5"] == backup._md5_file(archive)


@pytest.mark.parametrize("failure", ["recipient", "tool", "encryption"])
def test_fail_closed_preserves_plaintext(tmp_path, monkeypatch, keys, failure):
    if failure == "recipient":
        (keys[0] / "backup-age-recipient.txt").unlink()
    elif failure == "tool":
        monkeypatch.setattr(backup_age, "executable", lambda *a: (_ for _ in ()).throw(RuntimeError("missing")))
    else:
        original_run = subprocess.run
        monkeypatch.setattr(
            backup_age.subprocess,
            "run",
            lambda command, **kw: (
                subprocess.CompletedProcess(command, 1) if "--encrypt" in command else original_run(command, **kw)
            ),
        )
    uploads = []
    _mock_drive_success(backup, monkeypatch, uploads)
    source = tmp_path / "sources"
    _write_jsonl(source / "session.jsonl", mtime=time.time() - 3600)
    result = backup.run_backup(
        source_roots=[source],
        state_path=tmp_path / "state.json",
        staging_dir=tmp_path / "staging",
        log_path=tmp_path / "log",
        queue_dir=tmp_path / "queue",
    )
    assert not uploads and result["status"] == "failed"
    assert Path(result["archive"]).exists()
    assert active_alerts(tmp_path / "job-alerts.json")["jsonl-backup-encryption"].startswith(
        "JSONL backup not encrypted: "
    )


@pytest.mark.parametrize("corrupt", [False, True])
def test_weekly_decrypt_pass_and_fail(tmp_path, monkeypatch, keys, corrupt):
    plain = tmp_path / "claude-jsonl-fixture.tar.gz"
    plain.write_bytes(b"fixture")
    cipher, metadata = backup._encrypt_archive(plain)
    state = {"drive_archives": {cipher.name: {"verified": True, "verified_at": 10, **metadata}}}
    if corrupt:
        cipher.write_bytes(b"broken ciphertext")
    monkeypatch.setattr(backup_age, "private_identity", lambda: keys[1])
    backup._weekly_decrypt_test(state, tmp_path, tmp_path / "state.json", 2_000_000_000)
    assert ("jsonl-backup-decrypt" in active_alerts(tmp_path / "job-alerts.json")) == corrupt
    assert ("age_decrypt_test_at" in state) != corrupt


def test_weekly_decrypt_skips_missing_key_and_respects_week(tmp_path, monkeypatch, keys):
    plain = tmp_path / "claude-jsonl-fixture.tar.gz"
    plain.write_bytes(b"fixture")
    cipher, metadata = backup._encrypt_archive(plain)
    state = {"drive_archives": {cipher.name: {"verified": True, **metadata}}}
    backup._weekly_decrypt_test(state, tmp_path, tmp_path / "state.json", 2_000_000_000)
    assert not active_alerts(tmp_path / "job-alerts.json")
    assert "age_decrypt_test_at" not in state
    state["age_decrypt_test_at"] = 2_000_000_000
    monkeypatch.setattr(backup_age, "private_identity", lambda: pytest.fail("not due"))
    backup._weekly_decrypt_test(state, tmp_path, tmp_path / "state.json", 2_000_000_001)


def test_gate_off_adds_one_health_note(tmp_path, monkeypatch, keys):
    (keys[0] / "backup-age-recovery-confirmed").unlink()
    source = tmp_path / "sources"
    source.mkdir()
    for _ in range(2):
        result = backup.run_backup(
            source_roots=[source],
            state_path=tmp_path / "state.json",
            staging_dir=tmp_path / "staging",
            log_path=tmp_path / "log",
            queue_dir=tmp_path / "queue",
        )
        assert result["status"] == "no-op"
    assert active_alerts(tmp_path / "job-alerts.json") == {
        "jsonl-backup-encryption-setup": "backup encryption not yet enabled"
    }
    assert not (tmp_path / "state.json").exists()


def test_ciphertext_reconcile_and_retention(tmp_path, monkeypatch, keys):
    from tests import test_jsonl_local_retention

    # Execute the existing randomized scenarios with .age fixture names.
    class CipherPath(type(tmp_path)):
        def __truediv__(self, name):
            if str(name).endswith(".tar.gz"):
                name += ".age"
            return super().__truediv__(name)

    test_jsonl_local_retention.test_random_retention_and_crashes(CipherPath(tmp_path), monkeypatch)
    path = tmp_path / "claude-jsonl-test.tar.gz.age"
    path.write_bytes(b"ciphertext fixture")
    state = {}
    remote = {
        path.name: [{"id": "id", "size": path.stat().st_size, "md5Checksum": backup._md5_file(path), "trashed": False}]
    }
    assert backup.reconcile_drive_receipts(tmp_path, state=state, now=time.time(), drive_objects=remote) == [path.name]
    assert state["drive_archives"][path.name]["verified"]


def test_streaming_large_file_bounded_rss(tmp_path, keys):
    # Independent processes measure the real age binary plus Python, with no allocator history.
    program = """
import json,resource,sys
from pathlib import Path
from brainlayer import backup_age
backup_age.config_dir = lambda: Path(sys.argv[2])
backup_age.executable = lambda name='age': Path(sys.argv[4])
path = Path(sys.argv[1])
with path.open('wb') as f:
    block = b'x' * 1048576
    for _ in range(int(sys.argv[3])): f.write(block)
backup_age.encrypt(path)
print(json.dumps([resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss]))
"""
    rss = []
    for size in (2, 200):
        result = subprocess.run(
            [
                os.sys.executable,
                "-c",
                program,
                str(tmp_path / f"{size}.tar.gz"),
                str(keys[0]),
                str(size),
                str(backup_age.executable()),
            ],
            capture_output=True,
            check=True,
        )
        rss.append(json.loads(result.stdout))
    scale = 1 if os.sys.platform == "darwin" else 1024
    assert all((large - small) * scale < 40 * 1024 * 1024 for small, large in zip(*rss))
