import json
import os
import time
from pathlib import Path

import pytest


def _write_jsonl(path: Path, line: str = '{"type":"message"}\n', *, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_run_jsonl_backup_uploads_incremental_bundle_verifies_and_enqueues_summary(tmp_path, monkeypatch):
    from brainlayer import jsonl_backup

    now = time.time()
    claude_root = tmp_path / "home" / ".claude" / "projects"
    archive_root = tmp_path / "home" / ".claude-archive"
    codex_root = tmp_path / "home" / ".codex" / "sessions"
    old_files = [
        _write_jsonl(claude_root / "project-a" / "session-a.jsonl", mtime=now - 3600),
        _write_jsonl(archive_root / "project-b" / "session-b.jsonl", mtime=now - 3600),
        _write_jsonl(codex_root / "2026" / "06" / "05" / "rollout.jsonl", mtime=now - 3600),
    ]
    active = _write_jsonl(claude_root / "project-a" / "active.jsonl", mtime=now - 60)
    uploads: list[Path] = []
    pruned = []

    monkeypatch.setattr(jsonl_backup.backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(jsonl_backup.backup_daily, "build_drive_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        jsonl_backup.backup_daily, "ensure_drive_folder_chain", lambda service, folder_parts: "folder-id"
    )
    monkeypatch.setattr(jsonl_backup.backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)

    def fake_upload(file_path, folder_id, credentials):  # noqa: ARG001
        uploads.append(Path(file_path))
        return {"id": "drive-jsonl-id", "name": Path(file_path).name, "size": str(Path(file_path).stat().st_size)}

    def fake_prune(service, *, folder_parts, retention_policy):  # noqa: ARG001
        pruned.append(retention_policy.keep_latest)
        return ["claude-jsonl-2026-05-01.tar.gz"]

    monkeypatch.setattr(jsonl_backup.backup_daily, "upload_file_to_drive_raw", fake_upload)
    monkeypatch.setattr(jsonl_backup.backup_daily, "prune_drive_backups", fake_prune)

    result = jsonl_backup.run_backup(
        source_roots=[claude_root, archive_root, codex_root],
        state_path=tmp_path / "state.json",
        staging_dir=tmp_path / "staging",
        log_path=tmp_path / "jsonl-backup.log",
        queue_dir=tmp_path / "queue",
        date_stamp="2026-06-05",
        now=now,
        upload=True,
    )

    assert uploads == [tmp_path / "staging" / "claude-jsonl-2026-06-05.tar.gz"]
    assert pruned == [30]
    assert result["status"] == "uploaded"
    assert result["uploaded"] is True
    assert result["verified"] is True
    assert result["bundled_file_count"] == 3
    assert result["archive_listing_count"] == 3
    assert result["skipped_active_count"] == 1
    assert active.as_posix() not in (tmp_path / "state.json").read_text()
    state = json.loads((tmp_path / "state.json").read_text())
    assert sorted(state["files"]) == sorted(path.as_posix() for path in old_files)
    assert len((tmp_path / "jsonl-backup.log").read_text().strip().splitlines()) == 1
    queued = list((tmp_path / "queue").glob("jsonl_backup-*.jsonl"))
    assert len(queued) == 1
    assert "JSONL backup uploaded 3 files" in queued[0].read_text()


def test_run_jsonl_backup_second_run_noops_when_state_covers_files(tmp_path, monkeypatch):
    from brainlayer import jsonl_backup

    now = time.time()
    source_root = tmp_path / "sessions"
    _write_jsonl(source_root / "covered.jsonl", mtime=now - 3600)
    uploads: list[Path] = []

    monkeypatch.setattr(jsonl_backup.backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(jsonl_backup.backup_daily, "build_drive_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        jsonl_backup.backup_daily, "ensure_drive_folder_chain", lambda service, folder_parts: "folder-id"
    )
    monkeypatch.setattr(jsonl_backup.backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(jsonl_backup.backup_daily, "prune_drive_backups", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        jsonl_backup.backup_daily,
        "upload_file_to_drive_raw",
        lambda file_path, folder_id, credentials: (
            uploads.append(Path(file_path))  # noqa: ARG005
            or {
                "id": f"drive-{len(uploads)}",
                "name": Path(file_path).name,
                "size": str(Path(file_path).stat().st_size),
            }
        ),
    )

    kwargs = {
        "source_roots": [source_root],
        "state_path": tmp_path / "state.json",
        "staging_dir": tmp_path / "staging",
        "log_path": tmp_path / "jsonl-backup.log",
        "queue_dir": tmp_path / "queue",
        "date_stamp": "2026-06-05",
        "now": now,
        "upload": True,
    }
    first = jsonl_backup.run_backup(**kwargs)
    second = jsonl_backup.run_backup(**kwargs)

    assert first["status"] == "uploaded"
    assert second["status"] == "no-op"
    assert second["uploaded"] is False
    assert second["already_covered_files"] == 1
    assert second["message"] == "no-op, 1 files already covered"
    assert len(uploads) == 1
    assert len((tmp_path / "jsonl-backup.log").read_text().strip().splitlines()) == 2


def test_run_jsonl_backup_upload_failure_is_loud(tmp_path, monkeypatch):
    from brainlayer import jsonl_backup

    now = time.time()
    source_root = tmp_path / "sessions"
    _write_jsonl(source_root / "changed.jsonl", mtime=now - 3600)

    monkeypatch.setattr(jsonl_backup.backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(jsonl_backup.backup_daily, "build_drive_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        jsonl_backup.backup_daily, "ensure_drive_folder_chain", lambda service, folder_parts: "folder-id"
    )
    monkeypatch.setattr(
        jsonl_backup.backup_daily,
        "upload_file_to_drive_raw",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("upload failed")),
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        jsonl_backup.run_backup(
            source_roots=[source_root],
            state_path=tmp_path / "state.json",
            staging_dir=tmp_path / "staging",
            log_path=tmp_path / "jsonl-backup.log",
            queue_dir=tmp_path / "queue",
            date_stamp="2026-06-05",
            now=now,
            upload=True,
        )

    assert not (tmp_path / "state.json").exists()


def test_corrupt_jsonl_bundle_verifies_false_and_main_returns_nonzero(tmp_path, monkeypatch, capsys):
    from brainlayer import jsonl_backup

    corrupt = tmp_path / "claude-jsonl-2026-06-05.tar.gz"
    corrupt.write_bytes(b"not a gzip")
    verification = jsonl_backup.verify_jsonl_bundle(corrupt, expected_file_count=1)
    assert verification["verified"] is False

    def fake_run_backup(**kwargs):  # noqa: ARG001
        return {"status": "uploaded", "archive": str(corrupt), "uploaded": True, **verification}

    monkeypatch.setattr(jsonl_backup, "run_backup", fake_run_backup)
    monkeypatch.setattr(jsonl_backup, "_configured_backup_timeout_seconds", lambda: None)

    assert jsonl_backup.main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is False


def test_jsonl_backup_launchd_plist_and_docstring_install_note_are_committed():
    module_path = Path("src/brainlayer/jsonl_backup.py")
    plist_path = Path("launchd/com.brainlayer.jsonl-backup.plist")

    assert module_path.is_file()
    assert plist_path.is_file()

    module = module_path.read_text()
    plist = plist_path.read_text()

    assert "Install note" in module
    assert "com.brainlayer.jsonl-backup" in plist
    assert "<integer>5</integer>" in plist
    assert "<integer>0</integer>" in plist
    assert "BRAINLAYER_BACKUP_TIMEOUT_SECONDS" in plist
    assert "1800" in plist
    assert ".local/share/brainlayer/logs/jsonl-backup.log" in plist
