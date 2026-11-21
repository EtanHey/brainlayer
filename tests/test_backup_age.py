import io
import json
import os
import runpy
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from brainlayer import backup_age
from brainlayer import jsonl_backup as backup
from brainlayer.job_alerts import active_alerts, report
from tests.test_jsonl_backup import _mock_drive_success, _write_jsonl


@pytest.mark.parametrize("outcome", ["success", "failure", "bad-id", "bad-json", "missing"])
def test_recovery_command_suppresses_private_output(monkeypatch, capsys, outcome):
    """Execute the printed Python stages; all external commands are synthetic."""
    setup = Path(__file__).resolve().parents[1] / "scripts" / "backup_age_setup.py"
    monkeypatch.setattr(sys, "argv", [str(setup), "--recovery-command"])
    runpy.run_path(str(setup), run_name="__main__")
    stages = shlex.split(capsys.readouterr().out.strip())
    # Supply synthetic keychain output, never run security or op.
    assert stages[:7] == [
        "/usr/bin/security",
        "find-generic-password",
        "-s",
        "brainlayer-backup-age",
        "-w",
        "|",
        "python3",
    ]
    assert stages[7] == "-c"
    secret = "AGE-SECRET-KEY-SYNTHETIC-DO-NOT-USE"
    item_id = "a" * 26
    submitted = []

    def create(command, **kwargs):
        assert command == ["op", "item", "create", "-", "--format=json"]
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        submitted.append(json.loads(kwargs["input"]))
        if outcome == "missing":
            raise OSError(secret)
        value = {"id": secret if outcome == "bad-id" else item_id, "fields": submitted[-1]["fields"]}
        return subprocess.CompletedProcess(
            command,
            23 if outcome == "failure" else 0,
            stdout=secret if outcome == "bad-json" else json.dumps(value),
            stderr=secret,
        )

    monkeypatch.setattr(subprocess, "run", create)
    monkeypatch.setattr(sys, "stdin", io.StringIO(secret))
    exit_code = 0
    try:
        exec(stages[8], {"__name__": "__main__"})
    except SystemExit as exc:
        exit_code = exc.code
    # The old command ends in bare op: model its item echo without calling it.
    if stages[9:]:
        assert stages[9:] == ["|", "op", "item", "create", "-"]
        payload = capsys.readouterr().out
        print(payload)
    outward = capsys.readouterr()
    assert secret not in outward.out, "private input reached recovery stdout"
    assert secret not in outward.err, "private input reached recovery stderr"
    expected = 0 if outcome == "success" else 23 if outcome == "failure" else 1
    assert exit_code == expected
    assert outward.out == (f"STORED id={item_id}\n" if outcome == "success" else "ERROR\n")
    assert outward.err == ""
    assert submitted[0]["fields"] == [{"id": "identity", "type": "CONCEALED", "label": "age identity", "value": secret}]


@pytest.fixture
def keys(tmp_path, monkeypatch):
    if os.sys.platform != "darwin":
        monkeypatch.setattr(backup_age, "executable", lambda name="age": Path("/usr/bin") / name)
    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setattr(backup_age, "config_dir", lambda: config)
    keychain_lookup = backup_age.private_identity
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
    return config, generated.stdout, keychain_lookup


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
    assert result["encryption"] == "enabled"
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


@pytest.mark.parametrize("stale_alert", [False, True])
def test_gate_off_does_not_alert(tmp_path, monkeypatch, keys, stale_alert):
    (keys[0] / "backup-age-recovery-confirmed").unlink()
    alert_path = tmp_path / "job-alerts.json"
    if stale_alert:
        report("jsonl-backup-encryption-setup", "backup encryption not yet enabled", path=alert_path)
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
        assert not active_alerts(alert_path)
        assert result["encryption"] == "not_enabled"
    assert all(json.loads(line)["encryption"] == "not_enabled" for line in (tmp_path / "log").read_text().splitlines())
    assert not (tmp_path / "state.json").exists()


@pytest.mark.parametrize("failure", [36, 1, "timeout", "unavailable"])
def test_weekly_decrypt_skips_keychain_lookup_failure(tmp_path, monkeypatch, keys, failure):
    plain = tmp_path / "claude-jsonl-fixture.tar.gz"
    plain.write_bytes(b"fixture")
    cipher, metadata = backup._encrypt_archive(plain)
    state = {"drive_archives": {cipher.name: {"verified": True, **metadata}}, "age_decrypt_test_at": 1}
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    original_state = state_path.read_bytes()
    # Exercise the real lookup wrapper without contacting the operator's keychain.
    monkeypatch.setattr(backup_age, "private_identity", keys[2])

    def security_failure(command, **kwargs):
        assert command[0] == "/usr/bin/security"
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 15)
        if failure == "unavailable":
            raise OSError("security unavailable")
        return subprocess.CompletedProcess(command, failure, stdout=b"", stderr=b"locked")

    monkeypatch.setattr(backup_age.subprocess, "run", security_failure)
    monkeypatch.setattr(backup_age, "decrypted_sha256", lambda *args: pytest.fail("lookup failed; skip decrypt"))
    backup._weekly_decrypt_test(state, tmp_path, state_path, 2_000_000_000)
    assert not active_alerts(tmp_path / "job-alerts.json")
    assert state["age_decrypt_test_at"] == 1
    assert state_path.read_bytes() == original_state


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
