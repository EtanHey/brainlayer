import hashlib
import json
import os
import tarfile
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


def test_default_source_roots_append_all_agent_cli_transcript_roots(monkeypatch):
    from brainlayer import jsonl_backup

    home = Path("/Users/tester")
    monkeypatch.setattr(jsonl_backup.Path, "home", lambda: home)

    roots = jsonl_backup.default_source_roots()

    assert [(root.path, root.include_globs) for root in roots] == [
        (home / ".claude" / "projects", ("**/*.jsonl",)),
        (home / ".claude-archive", ("**/*.jsonl",)),
        (home / ".codex" / "sessions", ("**/*.jsonl",)),
        (home / ".cursor" / "sessions", ("**/*.jsonl", "**/*.json")),
        (home / ".cursor" / "projects", ("**/agent-transcripts/**/*.jsonl", "**/agent-transcripts/**/*.json")),
        (home / ".cursor" / "acp-sessions", ("**/*.json",)),
        (home / ".cursor" / "plans", ("**/*.md",)),
        (home / ".gemini" / "sessions", ("**/*.jsonl",)),
        (home / ".gemini" / "antigravity-cli" / "conversations", ("**/*.db", "**/*.db-wal", "**/*.db-shm")),
        (home / ".gemini" / "antigravity-cli" / "implicit", ("**/*.pb",)),
        (
            home / ".gemini" / "antigravity-cli" / "brain",
            (
                "**/.system_generated/**/*.jsonl",
                "**/.system_generated/**/*.json",
                "**/.system_generated/**/*.log",
                "**/.system_generated/**/*.md",
                "**/.system_generated/**/*.png",
                "**/.system_generated/**/*.jpg",
                "**/.system_generated/**/*.mp4",
                "**/.tempmediaStorage/**/*.png",
                "**/.tempmediaStorage/**/*.jpg",
                "**/.tempmediaStorage/**/*.mp4",
            ),
        ),
        (home / ".gemini" / "antigravity-cli" / "cache", ("last_conversations.json", "projects.json")),
    ]


def test_appended_agent_cli_roots_cover_mixed_formats_and_keep_existing_archive_indices(tmp_path, monkeypatch):
    from brainlayer import jsonl_backup

    home = tmp_path / "home"
    monkeypatch.setattr(jsonl_backup.Path, "home", lambda: home)
    now = time.time()
    _write_jsonl(home / ".claude" / "projects" / "project-a" / "session.jsonl", mtime=now - 3600)
    _write_jsonl(home / ".claude-archive" / "project-b" / "archive.jsonl", mtime=now - 3600)
    _write_jsonl(home / ".codex" / "sessions" / "2026" / "06" / "session.jsonl", mtime=now - 3600)
    _write_jsonl(home / ".cursor" / "sessions" / "cursor-session.jsonl", mtime=now - 3600)
    _write_jsonl(home / ".cursor" / "sessions" / "cursor-session.json", mtime=now - 3600)
    _write_jsonl(
        home / ".cursor" / "projects" / "repo" / "agent-transcripts" / "agent" / "cursor.jsonl",
        mtime=now - 3600,
    )
    _write_jsonl(
        home / ".cursor" / "projects" / "repo" / "agent-transcripts" / "agent" / "cursor.json",
        mtime=now - 3600,
    )
    _write_jsonl(home / ".cursor" / "projects" / "repo" / "mcps" / "tool.json", mtime=now - 3600)
    _write_jsonl(home / ".cursor" / "acp-sessions" / "session" / "meta.json", mtime=now - 3600)
    _write_jsonl(home / ".cursor" / "plans" / "plan.md", line="# plan\n", mtime=now - 3600)
    _write_jsonl(home / ".gemini" / "sessions" / "gemini.jsonl", mtime=now - 3600)
    _write_jsonl(
        home / ".gemini" / "antigravity-cli" / "brain" / "session" / ".system_generated" / "logs" / "transcript.jsonl",
        mtime=now - 3600,
    )
    _write_jsonl(
        home / ".gemini" / "antigravity-cli" / "brain" / "session" / ".system_generated" / "messages" / "message.json",
        mtime=now - 3600,
    )
    _write_jsonl(
        home / ".gemini" / "antigravity-cli" / "brain" / "session" / ".system_generated" / "tasks" / "task.log",
        line="task log\n",
        mtime=now - 3600,
    )
    media = home / ".gemini" / "antigravity-cli" / "brain" / "session" / ".tempmediaStorage" / "screenshot.png"
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"png")
    os.utime(media, (now - 3600, now - 3600))
    _write_jsonl(home / ".gemini" / "antigravity-cli" / "mcp" / "tool.json", mtime=now - 3600)
    raw_files = [
        home / ".gemini" / "antigravity-cli" / "conversations" / "conv.db",
        home / ".gemini" / "antigravity-cli" / "conversations" / "conv.db-wal",
        home / ".gemini" / "antigravity-cli" / "conversations" / "conv.db-shm",
        home / ".gemini" / "antigravity-cli" / "implicit" / "implicit.pb",
    ]
    for path in raw_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"raw")
        os.utime(path, (now - 3600, now - 3600))
    _write_jsonl(home / ".gemini" / "antigravity-cli" / "cache" / "last_conversations.json", mtime=now - 3600)
    _write_jsonl(home / ".gemini" / "antigravity-cli" / "cache" / "projects.json", mtime=now - 3600)

    candidates = jsonl_backup._discover_jsonl_candidates(jsonl_backup.default_source_roots())
    archive = jsonl_backup.create_jsonl_bundle(candidates, tmp_path / "staging", date_stamp="2026-06-30")

    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()

    assert "source-0/project-a/session.jsonl" in names
    assert "source-1/project-b/archive.jsonl" in names
    assert "source-2/2026/06/session.jsonl" in names
    assert "source-3/cursor-session.jsonl" in names
    assert "source-3/cursor-session.json" in names
    assert "source-4/repo/agent-transcripts/agent/cursor.jsonl" in names
    assert "source-4/repo/agent-transcripts/agent/cursor.json" in names
    assert "source-4/repo/mcps/tool.json" not in names
    assert "source-5/session/meta.json" in names
    assert "source-6/plan.md" in names
    assert "source-7/gemini.jsonl" in names
    assert "source-8/conv.db" in names
    assert "source-8/conv.db-wal" in names
    assert "source-8/conv.db-shm" in names
    assert "source-9/implicit.pb" in names
    assert "source-10/session/.system_generated/logs/transcript.jsonl" in names
    assert "source-10/session/.system_generated/messages/message.json" in names
    assert "source-10/session/.system_generated/tasks/task.log" in names
    assert "source-10/session/.tempmediaStorage/screenshot.png" in names
    assert "source-11/last_conversations.json" in names
    assert "source-11/projects.json" in names
    assert not any(name.endswith("mcp/tool.json") for name in names)


def test_antigravity_sqlite_restore_unit_is_skipped_together_when_sidecar_is_active(tmp_path):
    from brainlayer import jsonl_backup

    now = time.time()
    conversations_root = tmp_path / "antigravity-cli" / "conversations"
    for name, mtime in {
        "conv.db": now - 3600,
        "conv.db-wal": now - 60,
        "conv.db-shm": now - 3600,
    }.items():
        path = conversations_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        os.utime(path, (mtime, mtime))

    result = jsonl_backup.run_backup(
        source_roots=[jsonl_backup.BackupSourceRoot(conversations_root, ("**/*.db", "**/*.db-wal", "**/*.db-shm"))],
        state_path=tmp_path / "state.json",
        staging_dir=tmp_path / "staging",
        log_path=tmp_path / "jsonl-backup.log",
        queue_dir=tmp_path / "queue",
        date_stamp="2026-06-30",
        now=now,
        upload=False,
    )

    assert result["status"] == "no-op"
    assert result["skipped_active_count"] == 3
    assert not (tmp_path / "staging" / "claude-jsonl-2026-06-30.tar.gz").exists()


def test_antigravity_sqlite_restore_unit_is_bundled_together_when_sidecar_changes(tmp_path):
    from brainlayer import jsonl_backup

    now = time.time()
    conversations_root = tmp_path / "antigravity-cli" / "conversations"
    db = conversations_root / "conv.db"
    wal = conversations_root / "conv.db-wal"
    shm = conversations_root / "conv.db-shm"
    for path, content in [(db, b"db"), (wal, b"wal"), (shm, b"shm")]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        os.utime(path, (now - 3600, now - 3600))
    state_path = tmp_path / "state.json"
    jsonl_backup._atomic_write_json(
        state_path,
        {
            "files": {
                db.as_posix(): {"mtime": db.stat().st_mtime, "size": db.stat().st_size},
                shm.as_posix(): {"mtime": shm.stat().st_mtime, "size": shm.stat().st_size},
            }
        },
    )

    result = jsonl_backup.run_backup(
        source_roots=[jsonl_backup.BackupSourceRoot(conversations_root, ("**/*.db", "**/*.db-wal", "**/*.db-shm"))],
        state_path=state_path,
        staging_dir=tmp_path / "staging",
        log_path=tmp_path / "jsonl-backup.log",
        queue_dir=tmp_path / "queue",
        date_stamp="2026-06-30",
        now=now,
        upload=False,
    )

    archive = Path(result["archive"])
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()

    assert result["status"] == "created"
    assert result["bundled_file_count"] == 3
    assert names == ["source-0/conv.db", "source-0/conv.db-shm", "source-0/conv.db-wal"]


def test_jsonl_forever_upload_uses_separate_folder_and_rolling_prune_only(tmp_path, monkeypatch):
    from brainlayer import jsonl_backup

    now = time.time()
    source_root = tmp_path / "sessions"
    source_file = source_root / "changed.db"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_bytes(b"sqlite bytes")
    os.utime(source_file, (now - 3600, now - 3600))
    folder_calls: list[list[str]] = []
    uploaded: list[tuple[Path, str, bytes]] = []
    pruned_folder_parts: list[list[str]] = []

    monkeypatch.setenv("BRAINLAYER_JSONL_FOREVER", "1")
    monkeypatch.setattr(jsonl_backup.backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(jsonl_backup.backup_daily, "build_drive_service", lambda *args, **kwargs: object())

    def fake_ensure(service, folder_parts):  # noqa: ARG001
        folder_calls.append(list(folder_parts))
        return "folder-" + "-".join(folder_parts[-2:])

    def fake_upload(file_path, folder_id, credentials):  # noqa: ARG001
        uploaded.append((Path(file_path), folder_id, Path(file_path).read_bytes()))
        return {
            "id": f"drive-{len(uploaded)}",
            "name": Path(file_path).name,
            "size": str(Path(file_path).stat().st_size),
        }

    def fake_prune(service, *, folder_parts, retention_policy):  # noqa: ARG001
        pruned_folder_parts.append(list(folder_parts))
        return []

    monkeypatch.setattr(jsonl_backup.backup_daily, "ensure_drive_folder_chain", fake_ensure)
    monkeypatch.setattr(jsonl_backup.backup_daily, "upload_file_to_drive_raw", fake_upload)
    monkeypatch.setattr(jsonl_backup.backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(jsonl_backup.backup_daily, "prune_drive_backups", fake_prune)

    result = jsonl_backup.run_backup(
        source_roots=[jsonl_backup.BackupSourceRoot(source_root, ("**/*.db",))],
        state_path=tmp_path / "state.json",
        staging_dir=tmp_path / "staging",
        log_path=tmp_path / "jsonl-backup.log",
        queue_dir=tmp_path / "queue",
        date_stamp="2026-06-30",
        now=now,
        upload=True,
    )

    assert result["verified"] is True
    assert result["forever_uploaded_file_count"] == 1
    assert uploaded[0][0].name == "claude-jsonl-2026-06-30.tar.gz"
    assert uploaded[1][0].name.endswith(".db")
    assert uploaded[1][2] == source_file.read_bytes()
    assert jsonl_backup.DEFAULT_FOLDER_PARTS in folder_calls
    assert jsonl_backup.DEFAULT_FOREVER_FOLDER_PARTS + ["source-0"] in folder_calls
    assert pruned_folder_parts == [jsonl_backup.DEFAULT_FOLDER_PARTS]
    assert jsonl_backup.DEFAULT_FOREVER_FOLDER_PARTS not in pruned_folder_parts
    assert not uploaded[0][0].exists()


def test_jsonl_forever_hashes_the_staged_copy_before_upload(tmp_path, monkeypatch):
    from brainlayer import jsonl_backup

    now = time.time()
    source_root = tmp_path / "sessions"
    source_file = source_root / "changed.db"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_bytes(b"source bytes")
    os.utime(source_file, (now - 3600, now - 3600))
    uploaded: list[tuple[Path, bytes]] = []

    monkeypatch.setenv("BRAINLAYER_JSONL_FOREVER", "1")
    monkeypatch.setattr(jsonl_backup.backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(jsonl_backup.backup_daily, "build_drive_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(jsonl_backup.backup_daily, "ensure_drive_folder_chain", lambda *args, **kwargs: "folder-id")
    monkeypatch.setattr(jsonl_backup.backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(jsonl_backup.backup_daily, "prune_drive_backups", lambda *args, **kwargs: [])

    def fake_copyfile(source, destination):  # noqa: ARG001
        Path(destination).write_bytes(b"staged bytes")
        return destination

    def fake_upload(file_path, folder_id, credentials):  # noqa: ARG001
        path = Path(file_path)
        uploaded.append((path, path.read_bytes()))
        return {"id": f"drive-{len(uploaded)}", "name": path.name, "size": str(path.stat().st_size)}

    monkeypatch.setattr(jsonl_backup.shutil, "copyfile", fake_copyfile)
    monkeypatch.setattr(jsonl_backup.backup_daily, "upload_file_to_drive_raw", fake_upload)

    result = jsonl_backup.run_backup(
        source_roots=[jsonl_backup.BackupSourceRoot(source_root, ("**/*.db",))],
        state_path=tmp_path / "state.json",
        staging_dir=tmp_path / "staging",
        log_path=tmp_path / "jsonl-backup.log",
        queue_dir=tmp_path / "queue",
        date_stamp="2026-06-30",
        now=now,
        upload=True,
    )

    staged_digest = hashlib.sha256(b"staged bytes").hexdigest()
    assert uploaded[1][0].name == f"{staged_digest}.db"
    assert uploaded[1][1] == b"staged bytes"
    assert result["forever_files"][0]["sha256"] == staged_digest
    assert "source" not in result["forever_files"][0]
    assert result["forever_files"][0]["source_root_index"] == 0
    assert result["forever_files"][0]["source_suffix"] == ".db"


def test_jsonl_forever_upload_caches_source_folder_lookup_per_root(tmp_path, monkeypatch):
    from brainlayer import jsonl_backup

    now = time.time()
    source_root = tmp_path / "sessions"
    for name in ("one.jsonl", "two.jsonl"):
        _write_jsonl(source_root / name, mtime=now - 3600)
    folder_calls: list[list[str]] = []
    uploaded: list[Path] = []

    monkeypatch.setenv("BRAINLAYER_JSONL_FOREVER", "1")
    monkeypatch.setattr(jsonl_backup.backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(jsonl_backup.backup_daily, "build_drive_service", lambda *args, **kwargs: object())

    def fake_ensure(service, folder_parts):  # noqa: ARG001
        folder_calls.append(list(folder_parts))
        return "folder-" + "-".join(folder_parts[-2:])

    def fake_upload(file_path, folder_id, credentials):  # noqa: ARG001
        path = Path(file_path)
        uploaded.append(path)
        return {"id": f"drive-{len(uploaded)}", "name": path.name, "size": str(path.stat().st_size)}

    monkeypatch.setattr(jsonl_backup.backup_daily, "ensure_drive_folder_chain", fake_ensure)
    monkeypatch.setattr(jsonl_backup.backup_daily, "upload_file_to_drive_raw", fake_upload)
    monkeypatch.setattr(jsonl_backup.backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(jsonl_backup.backup_daily, "prune_drive_backups", lambda *args, **kwargs: [])

    result = jsonl_backup.run_backup(
        source_roots=[source_root],
        state_path=tmp_path / "state.json",
        staging_dir=tmp_path / "staging",
        log_path=tmp_path / "jsonl-backup.log",
        queue_dir=tmp_path / "queue",
        date_stamp="2026-06-30",
        now=now,
        upload=True,
    )

    assert result["forever_uploaded_file_count"] == 2
    assert len(uploaded) == 3
    assert folder_calls.count(jsonl_backup.DEFAULT_FOREVER_FOLDER_PARTS + ["source-0"]) == 1


def test_jsonl_backup_persists_daily_state_when_forever_upload_fails(tmp_path, monkeypatch):
    from brainlayer import jsonl_backup

    now = time.time()
    source_root = tmp_path / "sessions"
    source_file = source_root / "changed.db"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_bytes(b"sqlite bytes")
    os.utime(source_file, (now - 3600, now - 3600))
    state_path = tmp_path / "state.json"
    uploads: list[Path] = []
    pruned_folder_parts: list[list[str]] = []

    monkeypatch.setenv("BRAINLAYER_JSONL_FOREVER", "1")
    monkeypatch.setattr(jsonl_backup.backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(jsonl_backup.backup_daily, "build_drive_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(jsonl_backup.backup_daily, "ensure_drive_folder_chain", lambda *args, **kwargs: "folder-id")
    monkeypatch.setattr(jsonl_backup.backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)

    def fake_upload(file_path, folder_id, credentials):  # noqa: ARG001
        path = Path(file_path)
        uploads.append(path)
        if len(uploads) == 2:
            raise RuntimeError("forever failed")
        return {"id": "daily-drive-id", "name": path.name, "size": str(path.stat().st_size)}

    def fake_prune(service, *, folder_parts, retention_policy):  # noqa: ARG001
        pruned_folder_parts.append(list(folder_parts))
        return []

    monkeypatch.setattr(jsonl_backup.backup_daily, "upload_file_to_drive_raw", fake_upload)
    monkeypatch.setattr(jsonl_backup.backup_daily, "prune_drive_backups", fake_prune)

    with pytest.raises(RuntimeError, match="forever failed"):
        jsonl_backup.run_backup(
            source_roots=[jsonl_backup.BackupSourceRoot(source_root, ("**/*.db",))],
            state_path=state_path,
            staging_dir=tmp_path / "staging",
            log_path=tmp_path / "jsonl-backup.log",
            queue_dir=tmp_path / "queue",
            date_stamp="2026-06-30",
            now=now,
            upload=True,
        )

    state = json.loads(state_path.read_text())
    assert source_file.as_posix() in state["files"]
    assert pruned_folder_parts == [jsonl_backup.DEFAULT_FOLDER_PARTS]
    assert not (tmp_path / "staging" / "claude-jsonl-2026-06-30.tar.gz").exists()


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


def test_local_only_jsonl_bundle_does_not_advance_upload_state(tmp_path, monkeypatch):
    from brainlayer import jsonl_backup

    now = time.time()
    source_root = tmp_path / "sessions"
    _write_jsonl(source_root / "changed.jsonl", mtime=now - 3600)
    state_path = tmp_path / "state.json"
    uploads: list[Path] = []

    local = jsonl_backup.run_backup(
        source_roots=[source_root],
        state_path=state_path,
        staging_dir=tmp_path / "staging",
        log_path=tmp_path / "jsonl-backup.log",
        queue_dir=tmp_path / "queue",
        date_stamp="2026-06-05",
        now=now,
        upload=False,
    )

    assert local["status"] == "created"
    assert local["verified"] is True
    assert not state_path.exists()

    monkeypatch.setattr(jsonl_backup.backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(jsonl_backup.backup_daily, "build_drive_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        jsonl_backup.backup_daily, "ensure_drive_folder_chain", lambda service, folder_parts: "folder-id"
    )
    monkeypatch.setattr(jsonl_backup.backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(jsonl_backup.backup_daily, "prune_drive_backups", lambda *args, **kwargs: [])

    def fake_upload(file_path, folder_id, credentials):  # noqa: ARG001
        uploads.append(Path(file_path))
        return {"id": "drive-jsonl-id", "name": Path(file_path).name, "size": str(Path(file_path).stat().st_size)}

    monkeypatch.setattr(jsonl_backup.backup_daily, "upload_file_to_drive_raw", fake_upload)

    uploaded = jsonl_backup.run_backup(
        source_roots=[source_root],
        state_path=state_path,
        staging_dir=tmp_path / "staging",
        log_path=tmp_path / "jsonl-backup.log",
        queue_dir=tmp_path / "queue",
        date_stamp="2026-06-05",
        now=now,
        upload=True,
    )

    assert uploaded["status"] == "uploaded"
    assert uploaded["bundled_file_count"] == 1
    assert len(uploads) == 1
    assert source_root.joinpath("changed.jsonl").as_posix() in state_path.read_text()


def test_load_jsonl_backup_state_ignores_non_dict_json(tmp_path):
    from brainlayer import jsonl_backup

    state_path = tmp_path / "state.json"
    state_path.write_text("[]", encoding="utf-8")

    assert jsonl_backup._load_state(state_path) == {"files": {}}


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
    script_plist_path = Path("scripts/launchd/com.brainlayer.jsonl-backup.plist")
    wrapper_path = Path("scripts/launchd/jsonl-backup.sh")
    install_path = Path("scripts/launchd/install.sh")

    assert module_path.is_file()
    assert plist_path.is_file()
    assert script_plist_path.is_file()
    assert wrapper_path.is_file()

    module = module_path.read_text()
    plist = plist_path.read_text()
    script_plist = script_plist_path.read_text()
    wrapper = wrapper_path.read_text()
    install = install_path.read_text()

    assert "Install note" in module
    assert "com.brainlayer.jsonl-backup" in plist
    assert "com.brainlayer.jsonl-backup" in script_plist
    assert "<integer>5</integer>" in plist
    assert "<integer>5</integer>" in script_plist
    assert "<integer>0</integer>" in plist
    assert "<integer>0</integer>" in script_plist
    assert "BRAINLAYER_BACKUP_TIMEOUT_SECONDS" in plist
    assert "BRAINLAYER_BACKUP_TIMEOUT_SECONDS" in wrapper
    assert "1800" in plist
    assert "1800" in wrapper
    assert ".local/share/brainlayer/logs/jsonl-backup.log" in plist
    assert "jsonl-backup" in install
    assert "install_jsonl_backup_script" in install
    assert "__BRAINLAYER_DIR_VALUE__" in wrapper
    assert "PYTHONPATH" in wrapper
    assert "__HOME__/.local/lib/brainlayer/jsonl-backup.sh" in script_plist
    assert "<key>SoftResourceLimits</key>" in plist
    assert "<key>SoftResourceLimits</key>" in script_plist
    assert "<key>NumberOfFiles</key>" in plist
    assert "<key>NumberOfFiles</key>" in script_plist
    assert "<integer>4096</integer>" in plist
    assert "<integer>4096</integer>" in script_plist
