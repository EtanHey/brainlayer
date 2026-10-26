"""Consolidation keeps backup wire shape and log provenance intact."""

import json
import plistlib
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from brainlayer import log_cap, observability_backup
from tests.test_observability_backup import FIXTURES, NOW, _command_env


def test_archived_backup_observes_calendar_launcher_with_frozen_shape(monkeypatch, tmp_path):
    env, recorder = _command_env(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agents = tmp_path / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    # Match the real maintenance launcher's calendar; a Label alone cannot schedule a run.
    (agents / "com.brainlayer.maintenance.plist").write_bytes(
        plistlib.dumps(
            {
                "Label": "com.brainlayer.maintenance",
                "ProgramArguments": ["/bin/true"],
                "StartCalendarInterval": {"Hour": 3, "Minute": 15},
            }
        )
    )
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[-1].endswith("com.brainlayer.jsonl-backup"):
            return subprocess.CompletedProcess(argv, 113, b"", b"Could not find service")
        assert argv[-1].endswith("com.brainlayer.maintenance")
        return subprocess.CompletedProcess(argv, 0, b"com.brainlayer.maintenance = { state = waiting }", b"")

    monkeypatch.setattr(observability_backup.subprocess, "run", run)
    result = observability_backup.build_backups_section(env=env, record_input=recorder, now=NOW)
    assert result["state"] == "measured" and result["launchd"]["bootstrapped"] is True
    assert result["launchd"]["label"] == "com.brainlayer.jsonl-backup"
    schema = json.loads((FIXTURES / "observability-schema.v1.json").read_text())
    jsonschema.validate(result, {"$ref": "#/$defs/backups", "$defs": schema["$defs"]})
    assert len(calls) == 2
    assert any("com.brainlayer.maintenance" in item["path"] for item in result["inputs"])


def test_existing_legacy_plist_prevents_calendar_fallback(monkeypatch, tmp_path):
    env, recorder = _command_env(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agents = tmp_path / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    (agents / "com.brainlayer.jsonl-backup.plist").write_bytes(plistlib.dumps({"Label": "com.brainlayer.jsonl-backup"}))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 113, b"", b"Could not find service")

    monkeypatch.setattr(observability_backup.subprocess, "run", run)
    result = observability_backup.build_backups_section(env=env, record_input=recorder, now=NOW)
    assert result["launchd"]["bootstrapped"] is False and len(calls) == 1


@pytest.mark.parametrize("first_stderr", [b"permission denied", b"Could not find service"])
def test_unrecognized_or_failed_calendar_probe_keeps_evidence(monkeypatch, tmp_path, first_stderr):
    env, recorder = _command_env(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agents = tmp_path / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    (agents / "com.brainlayer.maintenance.plist").write_text("placeholder")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            return subprocess.CompletedProcess(argv, 113, b"", first_stderr)
        raise subprocess.TimeoutExpired(argv, 5)

    monkeypatch.setattr(observability_backup.subprocess, "run", run)
    result = observability_backup.build_backups_section(env=env, record_input=recorder, now=NOW)
    assert result["state"] == "unmeasurable"
    assert len(calls) == (2 if b"Could not find service" in first_stderr else 1)
    assert any("com.brainlayer.jsonl-backup" in item["path"] for item in result["inputs"])


def test_manifest_and_calendar_logs_survive_plist_archival(tmp_path):
    agents, logs = tmp_path / "agents", tmp_path / "logs"
    agents.mkdir()
    logs.mkdir()
    watch, calendar, backup = logs / "watch.err.log", logs / "index.out.log", logs / "backup-daily.log"
    watch.write_bytes(b"old\n" * 100 + b"watch tail\n")
    calendar.write_bytes(b"old\n" * 100 + b"calendar tail\n")
    backup.write_text("".join(json.dumps({"record": i, "backup_log_provenance": "real"}) + "\n" for i in range(20)))
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            {
                "com.brainlayer.watch": {
                    "enabled": True,
                    "plist": {
                        "Label": "com.brainlayer.watch",
                        "KeepAlive": True,
                        "ProgramArguments": ["/bin/true"],
                        "StandardErrorPath": str(watch),
                    },
                }
            }
        )
    )
    manifest.chmod(0o600)
    inode = backup.stat().st_ino
    assert set(
        log_cap.cap_job_logs(
            agents, manifest_path=manifest, calendar_logs=logs, backup_logs=(backup,), max_bytes=200, keep_bytes=100
        )
    ) == {watch, calendar, backup}
    assert backup.stat().st_ino == inode and backup.stat().st_size < 200
    assert json.loads(backup.read_text().splitlines()[-1])["record"] == 19
    assert all(json.loads(line)["backup_log_provenance"] == "real" for line in backup.read_text().splitlines())


def test_cap_refuses_to_erase_only_oversized_backup_receipt(tmp_path):
    backup = tmp_path / "backup.log"
    original = json.dumps({"backup_log_provenance": "real", "detail": "x" * 400}) + "\n"
    backup.write_text(original)
    with pytest.raises(ValueError, match="complete backup record"):
        log_cap.cap_job_logs(tmp_path, backup_logs=(backup,), max_bytes=200, keep_bytes=100)
    assert backup.read_text() == original


def test_cap_retains_final_receipt_larger_than_keep_limit(tmp_path):
    backup = tmp_path / "backup.log"
    record = json.dumps({"detail": "x" * 120}) + "\n"
    backup.write_text(record * 2)
    log_cap.cap_job_logs(tmp_path, backup_logs=(backup,), max_bytes=200, keep_bytes=100)
    assert backup.read_text() == record


def test_log_cap_module_accepts_explicit_backup_log(tmp_path):
    backup = tmp_path / "backup.log"
    backup.write_text("".join(json.dumps({"record": i}) + "\n" for i in range(40)))
    inode = backup.stat().st_ino
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "brainlayer.log_cap",
            "--agents-dir",
            str(tmp_path),
            "--backup-log",
            str(backup),
            "--max-bytes",
            "200",
            "--keep-bytes",
            "100",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert backup.stat().st_ino == inode
    assert json.loads(backup.read_text().splitlines()[-1]) == {"record": 39}


def test_cap_refuses_partial_final_receipt_without_discarding_complete_ones(tmp_path):
    backup = tmp_path / "backup.log"
    original = (json.dumps({"record": 1, "detail": "x" * 60}) + "\n") * 4 + '{"record": 2'
    backup.write_text(original)
    with pytest.raises(ValueError, match="incomplete backup record"):
        log_cap.cap_job_logs(tmp_path, backup_logs=(backup,), max_bytes=120, keep_bytes=60)
    assert backup.read_text() == original


def test_cap_refuses_while_receipt_writer_holds_log_lock(tmp_path):
    import fcntl

    backup = tmp_path / "backup.log"
    original = (json.dumps({"record": 1}) + "\n") * 40
    backup.write_text(original)
    with backup.open("a") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX)
        with pytest.raises(ValueError, match="busy backup log"):
            log_cap.cap_job_logs(tmp_path, backup_logs=(backup,), max_bytes=200, keep_bytes=100)
    assert backup.read_text() == original


@pytest.mark.parametrize("max_bytes", [200, 500])
def test_cap_preserves_last_real_receipt_or_refuses(tmp_path, max_bytes):
    backup = tmp_path / "backup.log"
    real = json.dumps({"backup_log_provenance": "real", "verified": True}) + "\n"
    original = real + (json.dumps({"backup_log_provenance": "pytest"}) + "\n") * 8
    backup.write_text(original)
    if max_bytes == 200:
        with pytest.raises(ValueError, match="real-provenance receipt"):
            log_cap.cap_job_logs(tmp_path, backup_logs=(backup,), max_bytes=max_bytes, keep_bytes=100)
        assert backup.read_text() == original
    else:
        # Force a cap while the full suffix containing the real receipt fits.
        backup.write_text((json.dumps({"old": "x" * 600}) + "\n") + original)
        log_cap.cap_job_logs(tmp_path, backup_logs=(backup,), max_bytes=max_bytes, keep_bytes=100)
        assert backup.read_text() == original


@pytest.mark.parametrize("module", ["backup_daily", "jsonl_backup"])
def test_receipt_append_waits_for_cap_lock(tmp_path, module):
    import fcntl
    import importlib
    import threading

    backup = tmp_path / "backup.log"
    backup.write_text("old\n")
    started, done = threading.Event(), threading.Event()

    def append():
        started.set()
        importlib.import_module(f"brainlayer.{module}")._append_json_log(backup, {"new": True})
        done.set()

    with backup.open("a") as cap:
        fcntl.flock(cap, fcntl.LOCK_EX)
        writer = threading.Thread(target=append)
        writer.start()
        assert started.wait(2)
        completed_before_unlock = done.wait(0.2)
    writer.join(5)
    assert not writer.is_alive() and done.is_set()
    assert not completed_before_unlock, "receipt append ignored the cap's shared lock"
    assert json.loads(backup.read_text().splitlines()[-1]) == {"new": True}
