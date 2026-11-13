import hashlib
import json
import os

import pytest

from brainlayer import jsonl_backup as b


@pytest.mark.parametrize("days", ["fourteen", "7", ""])
def test_invalid_setting_never_stops_backup(tmp_path, monkeypatch, days):
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION_DAYS", days)
    monkeypatch.setattr(b.backup_daily, "get_drive_credentials", lambda: None)
    monkeypatch.setattr(b.backup_daily, "build_drive_service", lambda: object())
    monkeypatch.setattr(b, "_list_surviving_archives", lambda *a: {})
    source = tmp_path / "source"
    source.mkdir()
    result = b.run_backup(
        source_roots=[source],
        staging_dir=tmp_path / "staging",
        state_path=tmp_path / "state",
        log_path=tmp_path / "log",
        queue_dir=tmp_path / "queue",
    )
    assert result["status"] == "no-op"
    assert result["local_retention_error"]


def test_sweep_exception_reported_and_backup_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "prune_local_archives", lambda *a, **k: (_ for _ in ()).throw(OSError("denied")))
    monkeypatch.setattr(b.backup_daily, "get_drive_credentials", lambda: None)
    monkeypatch.setattr(b.backup_daily, "build_drive_service", lambda: object())
    monkeypatch.setattr(b, "_list_surviving_archives", lambda *a: {})
    source = tmp_path / "source"
    source.mkdir()
    result = b.run_backup(
        source_roots=[source],
        staging_dir=tmp_path / "staging",
        state_path=tmp_path / "state",
        log_path=tmp_path / "log",
        queue_dir=tmp_path / "queue",
    )
    assert result["status"] == "no-op"
    assert "denied" in result["local_retention_error"]


def test_local_only_never_sweeps(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "prune_local_archives", lambda *a, **k: pytest.fail("swept without upload"))
    source = tmp_path / "source"
    source.mkdir()
    assert (
        b.run_backup(
            source_roots=[source],
            staging_dir=tmp_path / "staging",
            state_path=tmp_path / "state",
            log_path=tmp_path / "log",
            queue_dir=tmp_path / "queue",
            upload=False,
        )["status"]
        == "no-op"
    )


@pytest.mark.parametrize("md5", [None, "wrong"])
def test_drive_receipt_requires_content_match(tmp_path, md5):
    path = tmp_path / "a"
    path.write_bytes(b"abc")
    assert not b._drive_archive_receipt(path, {"id": "id", "md5Checksum": md5}, now=0)["verified"]


def test_switches_independent(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_RETENTION", "0")
    monkeypatch.delenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION", raising=False)
    assert b._local_retention_enabled()
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION", "false")
    assert not b._local_retention_enabled()


def test_live_presence_verified_age_and_siblings(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAINLAYER_JSONL_BACKUP_LOCAL_RETENTION", raising=False)
    base = tmp_path / "jsonl-backups"
    old = base / "release-old"
    current = base / "release-new"
    old.mkdir(parents=True)
    current.mkdir()
    now = 2000000000
    path = old / "claude-jsonl-2026-01-01.tar.gz"
    path.write_bytes(b"abc")
    os.utime(path, (0, 0))
    newest = current / "claude-jsonl-2026-02-01.tar.gz"
    newest.write_bytes(b"new")
    state = tmp_path / "state"
    receipt = dict(
        name=path.name,
        size=3,
        sha256=hashlib.sha256(b"abc").hexdigest(),
        archive_md5=hashlib.md5(b"abc").hexdigest(),
        verified=True,
        verified_at=now - 15 * 86400,
        archive_id="id",
    )
    state.write_text(json.dumps({"drive_archives": {path.name: receipt}}))
    args = dict(state_path=state, log_path=tmp_path / "log", now=now)
    assert b.prune_local_archives(current, surviving_archives={}, **args) == []
    receipt["verified_at"] = now
    state.write_text(json.dumps({"drive_archives": {path.name: receipt}}))
    assert b.prune_local_archives(current, surviving_archives={"id": receipt["archive_md5"]}, **args) == []
    receipt["verified_at"] = 0
    state.write_text(json.dumps({"drive_archives": {path.name: receipt}}))
    assert b.prune_local_archives(current, surviving_archives={"id": "wrong"}, **args) == []
    assert b.prune_local_archives(current, surviving_archives={"id": receipt["archive_md5"]}, **args) == [path.name]
    assert newest.exists()
    assert "local_retention_deleted" in (tmp_path / "log").read_text()


def test_static_guard_rejects_removed_local_content_check():
    from pathlib import Path

    from brainlayer.backup_retention_invariant import inspect_jsonl_retention_invariant

    source = Path("src/brainlayer/jsonl_backup.py").read_text()
    other = Path("src/brainlayer/backup_daily.py").read_text()
    assert inspect_jsonl_retention_invariant(source, backup_daily_source=other) == []
    weakened = source.replace('receipt.get("sha256") != _sha256_file(path)', "False")
    assert any(
        "local unlink must follow" in error
        for error in inspect_jsonl_retention_invariant(weakened, backup_daily_source=other)
    )


@pytest.mark.parametrize("crash", [False, True])
def test_real_run_revoke_restore_and_interrupted_receipt_note(tmp_path, monkeypatch, crash):
    monkeypatch.setenv("BRAINLAYER_JSONL_BACKUP_RETENTION", "1")
    source = tmp_path / "sessions"
    source.mkdir()
    transcript = source / "one.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "a real transcript message"}}) + "\n"
    )
    os.utime(transcript, (0, 0))
    state = tmp_path / "state"
    state.write_text(json.dumps({"drive_archives": {"old": {"archive_id": "old-id", "verified": True}}}))
    monkeypatch.setattr(b.backup_daily, "get_drive_credentials", lambda: None)
    monkeypatch.setattr(b.backup_daily, "build_drive_service", lambda: object())
    monkeypatch.setattr(b.backup_daily, "resolve_machine_id", lambda: "scratch")
    monkeypatch.setattr(b.backup_daily, "ensure_drive_folder_chain", lambda *a: "folder")
    monkeypatch.setattr(b.backup_daily, "verify_drive_upload", lambda *a, **k: None)
    monkeypatch.setattr(b, "_list_surviving_archives", lambda *a: {})
    monkeypatch.setattr(
        b.backup_daily,
        "upload_file_to_drive_raw",
        lambda path, *a, **k: {"id": "new-id", "md5Checksum": b._md5_file(path)},
    )

    def prune(*args, **kwargs):
        revoked = json.loads(state.read_text())
        assert revoked["drive_archives"] == {}
        assert revoked["local_retention_revoke_pending"]
        if crash:
            raise OSError("interrupted prune")
        return ["old"]

    monkeypatch.setattr(b, "_prune_drive_backups_to_trash", prune)
    args = dict(
        source_roots=[source],
        staging_dir=tmp_path / "staging",
        state_path=state,
        log_path=tmp_path / "log",
        queue_dir=tmp_path / "queue",
        now=2000000000,
    )
    if crash:
        with pytest.raises(OSError, match="interrupted prune"):
            b.run_backup(**args)
        assert json.loads(state.read_text())["local_retention_revoke_pending"]
        assert "receipt reconciliation required" in (tmp_path / "job-alerts.json").read_text()
    else:
        result = b.run_backup(**args)
        final = json.loads(state.read_text())
        assert not final["local_retention_revoke_pending"]
        assert "old" not in final["drive_archives"]
        assert final["drive_archives"][b.Path(result["archive"]).name]["verified"]


def test_listing_failure_disables_sweep_but_not_backup(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    state = tmp_path / "state"
    state.write_text(json.dumps({"drive_archives": {"old": {"archive_id": "id"}}}))
    monkeypatch.setattr(b.backup_daily, "get_drive_credentials", lambda: None)
    monkeypatch.setattr(b.backup_daily, "build_drive_service", lambda: object())
    monkeypatch.setattr(b, "_list_surviving_archives", lambda *a: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr(b, "prune_local_archives", lambda *a, **k: pytest.fail("swept without inventory"))
    result = b.run_backup(
        source_roots=[source],
        staging_dir=tmp_path / "staging",
        state_path=state,
        log_path=tmp_path / "log",
        queue_dir=tmp_path / "queue",
    )
    assert result["status"] == "no-op"
    assert "offline" in result["local_retention_error"]
