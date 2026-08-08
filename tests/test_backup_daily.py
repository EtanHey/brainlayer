import gzip
import json
import os
import queue
import socket
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pytest


def _start_fake_brainbar_vacuum_server(socket_path: Path, source_db: Path):
    received: queue.Queue[dict] = queue.Queue()
    ready = threading.Event()

    def run() -> None:
        if socket_path.exists():
            socket_path.unlink()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            server.listen(2)
            ready.set()
            for _ in range(2):
                conn, _ = server.accept()
                with conn:
                    data = b""
                    while not data.endswith(b"\n"):
                        data += conn.recv(65_536)
                    request = json.loads(data.decode("utf-8"))
                    if request["method"] == "initialize":
                        response = {
                            "jsonrpc": "2.0",
                            "id": request["id"],
                            "result": {"serverInfo": {"backupWriterStartedAtUnix": time.time()}},
                        }
                    else:
                        received.put(request)
                        args = request["params"]["arguments"]
                        target_path = Path(args["target_path"])
                        with sqlite3.connect(source_db) as db:
                            db.execute("VACUUM INTO ?", (str(target_path),))
                        target_path.with_name(f"{target_path.name}.complete").write_text("complete\n")
                        response = {
                            "jsonrpc": "2.0",
                            "id": request["id"],
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps({"status": "ok", "target_path": str(target_path)}),
                                    }
                                ]
                            },
                        }
                    conn.sendall(json.dumps(response).encode("utf-8") + b"\n")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(timeout=2)
    return received, thread


def _create_source_db(path: Path, *, chunk_count: int = 1) -> None:
    conn = sqlite3.connect(path)
    journal_mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    assert journal_mode.upper() == "WAL"
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO chunks VALUES ('c1', 'hello')")
    for idx in range(2, chunk_count + 1):
        conn.execute("INSERT INTO chunks VALUES (?, ?)", (f"c{idx}", f"hello-{idx}"))
    conn.commit()
    conn.close()


def test_create_snapshot_gzip_is_restorable(tmp_path):
    from brainlayer.backup_daily import create_sqlite_backup_gzip

    source = tmp_path / "brainlayer.db"
    _create_source_db(source)
    socket_path = Path(f"/tmp/bb-{os.getpid()}-{uuid.uuid4().hex}.sock")
    _start_fake_brainbar_vacuum_server(socket_path, source)

    out_dir = tmp_path / "out"
    snapshot = create_sqlite_backup_gzip(source, out_dir, date_stamp="2026-05-13", socket_path=socket_path)

    assert snapshot == out_dir / "2026-05-13.db.gz"
    assert snapshot.exists()
    assert not list(out_dir.glob(".*.db.attempt-*.complete"))

    restored = tmp_path / "restored.db"
    with gzip.open(snapshot, "rb") as src, restored.open("wb") as dst:
        dst.write(src.read())

    restored_conn = sqlite3.connect(restored)
    try:
        assert restored_conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert restored_conn.execute("SELECT content FROM chunks WHERE id = 'c1'").fetchone()[0] == "hello"
    finally:
        restored_conn.close()


def test_run_backup_verifies_gzip_with_snapshot_sentinel_and_keeps_raw_snapshot(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=3)
    socket_path = Path(f"/tmp/bb-{os.getpid()}-{uuid.uuid4().hex}.sock")
    _start_fake_brainbar_vacuum_server(socket_path, source)
    staging_dir = tmp_path / "out"
    uploads: list[Path] = []

    monkeypatch.setenv("BRAINBAR_SOCKET_PATH", str(socket_path))
    monkeypatch.setattr(backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "build_drive_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "ensure_drive_folder_chain", lambda service, folder_parts: "folder-id")
    monkeypatch.setattr(backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(backup_daily, "prune_drive_backups", lambda *args, **kwargs: [])

    def fake_upload(file_path, folder_id, credentials):  # noqa: ARG001
        uploads.append(Path(file_path))
        return {"id": "drive-file-id", "name": Path(file_path).name, "size": str(Path(file_path).stat().st_size)}

    monkeypatch.setattr(backup_daily, "upload_file_to_drive_raw", fake_upload)

    result = backup_daily.run_backup(
        db_path=source,
        staging_dir=staging_dir,
        date_stamp="2026-06-05",
        upload=True,
        remove_local_after_upload=True,
    )

    assert uploads == [staging_dir / "2026-06-05.db.gz"]
    assert result["verified"] is True
    assert result["verification_mode"] == "quick"
    assert result["sentinel_snapshot_chunks"] == 3
    assert result["sentinel_verified_chunks"] == 3
    assert result["local_removed"] is True
    assert not (staging_dir / "2026-06-05.db.gz").exists()
    assert (staging_dir / "2026-06-05.db").exists()


def test_run_backup_full_verify_downloads_drive_copy_and_md5_compares(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    socket_path = Path(f"/tmp/bb-{os.getpid()}-{uuid.uuid4().hex}.sock")
    _start_fake_brainbar_vacuum_server(socket_path, source)
    uploaded_bytes: dict[str, bytes] = {}
    downloads: list[str] = []

    monkeypatch.setenv("BRAINBAR_SOCKET_PATH", str(socket_path))
    monkeypatch.setenv("BRAINLAYER_BACKUP_FULL_VERIFY", "1")
    monkeypatch.setattr(backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "build_drive_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "ensure_drive_folder_chain", lambda service, folder_parts: "folder-id")
    monkeypatch.setattr(backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(backup_daily, "prune_drive_backups", lambda *args, **kwargs: [])

    def fake_upload(file_path, folder_id, credentials):  # noqa: ARG001
        uploaded_bytes["drive-file-id"] = Path(file_path).read_bytes()
        return {"id": "drive-file-id", "name": Path(file_path).name, "size": str(Path(file_path).stat().st_size)}

    def fake_download(service, *, file_id: str, destination: Path) -> Path:  # noqa: ARG001
        downloads.append(file_id)
        destination.write_bytes(uploaded_bytes[file_id])
        return destination

    monkeypatch.setattr(backup_daily, "upload_file_to_drive_raw", fake_upload)
    monkeypatch.setattr(backup_daily, "download_drive_file_raw", fake_download)

    result = backup_daily.run_backup(
        db_path=source,
        staging_dir=tmp_path / "out",
        date_stamp="2026-06-05",
        upload=True,
        remove_local_after_upload=True,
    )

    assert downloads == ["drive-file-id"]
    assert result["verified"] is True
    assert result["verification_mode"] == "full"
    assert result["drive_md5_match"] is True
    assert result["local_md5"] == result["drive_md5"]
    assert result["sentinel_snapshot_chunks"] == 2
    assert result["sentinel_verified_chunks"] == 2


def test_prune_local_uncompressed_snapshots_keeps_two_newest(tmp_path):
    from brainlayer.backup_daily import prune_local_uncompressed_snapshots

    for day in range(1, 5):
        (tmp_path / f"2026-06-0{day}.db").write_bytes(f"db-{day}".encode())
    (tmp_path / "2026-06-04.db.gz").write_bytes(b"drive-only")
    (tmp_path / "not-a-snapshot.db").write_bytes(b"ignore")

    deleted = prune_local_uncompressed_snapshots(tmp_path, keep_latest=2)

    assert deleted == ["2026-06-02.db", "2026-06-01.db"]
    assert sorted(path.name for path in tmp_path.glob("2026-06-*.db")) == ["2026-06-03.db", "2026-06-04.db"]
    assert (tmp_path / "2026-06-04.db.gz").exists()
    assert (tmp_path / "not-a-snapshot.db").exists()


def test_create_snapshot_reports_no_uncompressed_path_when_current_raw_is_pruned(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "2026-06-05.db").write_bytes(b"newer")
    (out_dir / "2026-06-04.db").write_bytes(b"also-newer")

    def fake_vacuum_into(target_path, **kwargs):  # noqa: ARG001
        with sqlite3.connect(source) as db:
            db.execute("VACUUM INTO ?", (str(target_path),))

    monkeypatch.setattr(backup_daily, "request_brainbar_vacuum_into", fake_vacuum_into)

    artifact = backup_daily.create_sqlite_backup_artifact(
        source,
        out_dir,
        date_stamp="2026-06-03",
        keep_uncompressed=True,
        local_uncompressed_keep=2,
    )

    assert artifact.uncompressed_path is None
    assert artifact.local_retention_deleted == ["2026-06-03.db"]
    assert not (out_dir / "2026-06-03.db").exists()
    assert sorted(path.name for path in out_dir.glob("*.db")) == ["2026-06-04.db", "2026-06-05.db"]


def test_create_snapshot_routes_vacuum_into_over_brainbar_socket(tmp_path):
    from brainlayer.backup_daily import create_sqlite_backup_gzip

    source = tmp_path / "brainlayer.db"
    _create_source_db(source)
    socket_path = Path(f"/tmp/bb-{os.getpid()}-{uuid.uuid4().hex}.sock")
    received, thread = _start_fake_brainbar_vacuum_server(socket_path, source)

    snapshot = create_sqlite_backup_gzip(source, tmp_path / "out", date_stamp="2026-05-13", socket_path=socket_path)

    thread.join(timeout=2)
    request = received.get_nowait()
    assert request["method"] == "tools/call"
    assert request["params"]["name"] == "brain_backup_vacuum_into"
    requested_target = Path(request["params"]["arguments"]["target_path"])
    assert requested_target.name.startswith(".2026-05-13.db.attempt-1-")
    assert snapshot.name == "2026-05-13.db.gz"


def test_brainbar_vacuum_request_retries_closed_socket_with_backoff(tmp_path, monkeypatch, capsys):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    calls = []
    sleeps = []

    def flaky_send(socket_path, request, timeout_seconds):  # noqa: ARG001
        attempt_target = Path(request["params"]["arguments"]["target_path"])
        calls.append((socket_path, request["params"]["name"], timeout_seconds, attempt_target))
        if len(calls) < 3:
            raise RuntimeError("BrainBar socket closed without response: /tmp/brainbar.sock")
        _create_source_db(attempt_target, chunk_count=2)
        return {"result": {"content": [{"type": "text", "text": '{"status":"ok"}'}]}}

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", flaky_send)
    monkeypatch.setattr(backup_daily, "_sleep", lambda seconds: sleeps.append(seconds))

    backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert len(calls) == 3
    assert len({call[3] for call in calls}) == 3
    assert all(call[3] != target for call in calls)
    assert sleeps == [60, 60]
    output = capsys.readouterr().out
    assert "BrainBar vacuum snapshot attempt 1/3 failed" in output
    assert "BrainBar vacuum snapshot attempt 2/3 failed" in output
    assert "retrying in 60s" in output


def test_brainbar_vacuum_request_uses_configured_client_timeout(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    seen_timeouts = []

    def capture_timeout(socket_path, request, timeout_seconds):  # noqa: ARG001
        seen_timeouts.append(timeout_seconds)
        _create_source_db(Path(request["params"]["arguments"]["target_path"]), chunk_count=2)
        return {"result": {"content": [{"type": "text", "text": '{"status":"ok"}'}]}}

    monkeypatch.setenv("BRAINLAYER_BACKUP_CLIENT_TIMEOUT_SECONDS", "420")
    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", capture_timeout)

    backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert seen_timeouts == [420]


def test_brainbar_vacuum_request_defaults_to_outer_wall_clock_timeout(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    seen_timeouts = []

    def capture_timeout(socket_path, request, timeout_seconds):  # noqa: ARG001
        seen_timeouts.append(timeout_seconds)
        _create_source_db(Path(request["params"]["arguments"]["target_path"]), chunk_count=2)
        return {"result": {"content": [{"type": "text", "text": '{"status":"ok"}'}]}}

    monkeypatch.delenv("BRAINLAYER_BACKUP_CLIENT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", capture_timeout)

    backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert seen_timeouts == [None]


def test_backup_wall_clock_timeout_has_safe_default_and_rejects_disable(monkeypatch):
    from brainlayer import backup_daily

    monkeypatch.delenv("BRAINLAYER_BACKUP_TIMEOUT_SECONDS", raising=False)
    assert backup_daily._configured_backup_timeout_seconds() == 7200

    monkeypatch.setenv("BRAINLAYER_BACKUP_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="must be at least 1 second"):
        backup_daily._configured_backup_timeout_seconds()


def test_brainbar_vacuum_request_fails_loud_after_retry_budget(tmp_path, monkeypatch, capsys):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    calls = []
    sleeps = []

    def closed_socket(socket_path, request, timeout_seconds):  # noqa: ARG001
        calls.append(request["params"]["name"])
        raise RuntimeError("BrainBar socket closed without response: /tmp/brainbar.sock")

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", closed_socket)
    monkeypatch.setattr(backup_daily, "_sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(RuntimeError, match="BrainBar socket closed without response"):
        backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert calls == ["brain_backup_vacuum_into", "brain_backup_vacuum_into", "brain_backup_vacuum_into"]
    assert sleeps == [60, 60]
    output = capsys.readouterr().out
    assert "BrainBar vacuum snapshot attempt 1/3 failed" in output
    assert "BrainBar vacuum snapshot attempt 2/3 failed" in output
    assert "BrainBar vacuum snapshot attempt 3/3 failed" in output
    assert "retrying in 60s" in output
    assert not target.exists()


def test_brainbar_vacuum_request_does_not_retry_global_backup_timeout(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    calls = []
    sleeps = []

    def timed_out(socket_path, request, timeout_seconds):  # noqa: ARG001
        calls.append(request["params"]["name"])
        raise backup_daily.BackupTimeoutError("backup exceeded configured wall-clock timeout")

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", timed_out)
    monkeypatch.setattr(backup_daily, "_sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(backup_daily.BackupTimeoutError):
        backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert calls == ["brain_backup_vacuum_into"]
    assert sleeps == []


def test_brainbar_vacuum_request_does_not_promote_valid_target_after_lost_response(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    calls = []
    sleeps = []

    attempt_targets = []

    def closed_after_success(socket_path, request, timeout_seconds):  # noqa: ARG001
        attempt_target = Path(request["params"]["arguments"]["target_path"])
        calls.append(request["params"]["name"])
        attempt_targets.append(attempt_target)
        _create_source_db(attempt_target, chunk_count=2)
        if len(calls) == 1:
            raise RuntimeError("BrainBar socket closed without response: /tmp/brainbar.sock")
        return {"result": {"content": [{"type": "text", "text": '{"status":"ok"}'}]}}

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", closed_after_success)
    monkeypatch.setattr(backup_daily, "_sleep", lambda seconds: sleeps.append(seconds))

    backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert calls == ["brain_backup_vacuum_into", "brain_backup_vacuum_into"]
    assert sleeps == [60]
    assert not attempt_targets[0].exists()
    assert target.exists()


def test_brainbar_vacuum_request_rejects_zero_page_target_after_success_response(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    calls = []

    def decoy_then_valid(socket_path, request, timeout_seconds):  # noqa: ARG001
        attempt_target = Path(request["params"]["arguments"]["target_path"])
        calls.append(request["params"]["name"])
        if len(calls) == 1:
            attempt_target.touch()
        else:
            assert not target.exists()
            _create_source_db(attempt_target, chunk_count=2)
        return {"result": {"content": [{"type": "text", "text": '{"status":"ok"}'}]}}

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", decoy_then_valid)
    monkeypatch.setattr(backup_daily, "_sleep", lambda seconds: None)

    backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert calls == ["brain_backup_vacuum_into", "brain_backup_vacuum_into"]
    assert backup_daily._count_chunks(target) == 2


def test_brainbar_vacuum_request_rejects_empty_chunks_target_after_lost_response(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    calls = []

    def empty_then_valid(socket_path, request, timeout_seconds):  # noqa: ARG001
        attempt_target = Path(request["params"]["arguments"]["target_path"])
        calls.append(request["params"]["name"])
        if len(calls) == 1:
            with sqlite3.connect(attempt_target) as db:
                db.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT)")
            raise RuntimeError("BrainBar socket closed without response: /tmp/brainbar.sock")
        assert not target.exists()
        _create_source_db(attempt_target, chunk_count=2)
        return {"result": {"content": [{"type": "text", "text": '{"status":"ok"}'}]}}

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", empty_then_valid)
    monkeypatch.setattr(backup_daily, "_sleep", lambda seconds: None)

    backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert calls == ["brain_backup_vacuum_into", "brain_backup_vacuum_into"]
    assert backup_daily._count_chunks(target) == 2


def test_brainbar_vacuum_request_preserves_lost_response_attempt_before_retry(tmp_path, monkeypatch, capsys):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    calls = []
    sleeps = []

    def invalid_then_success(socket_path, request, timeout_seconds):  # noqa: ARG001
        attempt_target = Path(request["params"]["arguments"]["target_path"])
        calls.append(request["params"]["name"])
        if len(calls) == 1:
            attempt_target.write_bytes(b"not sqlite")
            raise RuntimeError("BrainBar socket closed without response: /tmp/brainbar.sock")
        assert not target.exists()
        _create_source_db(attempt_target, chunk_count=2)
        return {"result": {"content": [{"type": "text", "text": '{"status":"ok"}'}]}}

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", invalid_then_success)
    monkeypatch.setattr(backup_daily, "_sleep", lambda seconds: sleeps.append(seconds))

    backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert calls == ["brain_backup_vacuum_into", "brain_backup_vacuum_into"]
    assert sleeps == [60]
    output = capsys.readouterr().out
    assert "preserving isolated attempt target" in output
    assert "retrying in 60s" in output


def test_create_snapshot_preserves_failed_attempts_outside_temporary_directory(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    output_dir = tmp_path / "out"
    _create_source_db(source, chunk_count=2)
    attempt_targets = []

    def closed_after_write(socket_path, request, timeout_seconds):  # noqa: ARG001
        if request["method"] == "initialize":
            return {"result": {"serverInfo": {"backupWriterStartedAtUnix": time.time()}}}
        attempt_target = Path(request["params"]["arguments"]["target_path"])
        attempt_targets.append(attempt_target)
        _create_source_db(attempt_target, chunk_count=2)
        raise RuntimeError("BrainBar socket closed without response: /tmp/brainbar.sock")

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", closed_after_write)
    monkeypatch.setattr(backup_daily, "_sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="socket closed without response"):
        backup_daily.create_sqlite_backup_artifact(source, output_dir, date_stamp="2026-05-14")

    assert len(attempt_targets) == 3
    assert all(path.parent == output_dir for path in attempt_targets)
    assert all(path.exists() for path in attempt_targets)


def test_stale_attempt_sweep_deletes_only_completed_regular_files_older_than_one_run_interval(tmp_path):
    from brainlayer import backup_daily

    now = 200_000.0
    stale_completed = tmp_path / ".2026-05-12.db.attempt-1-stale-completed"
    stale_prior_writer = tmp_path / ".2026-05-12.db.attempt-1-stale-prior-writer"
    stale_current_writer = tmp_path / ".2026-05-12.db.attempt-1-stale-current-writer"
    current_writer = tmp_path / ".2026-05-13.db.attempt-1-current-writer"
    recent = tmp_path / ".2026-05-13.db.attempt-1-recent"
    target = tmp_path / "outside.db"
    symlink = tmp_path / ".2026-05-11.db.attempt-1-link"
    stale_completed.write_bytes(b"stale completed")
    stale_prior_writer.write_bytes(b"stale prior writer")
    stale_current_writer.write_bytes(b"stale current writer")
    current_writer.write_bytes(b"current writer")
    recent.write_bytes(b"recent")
    target.write_bytes(b"outside")
    symlink.symlink_to(target)
    stale_marker = backup_daily._backup_attempt_completion_marker(stale_completed)
    recent_marker = backup_daily._backup_attempt_completion_marker(recent)
    stale_marker.write_text("complete")
    recent_marker.write_text("complete")
    os.utime(stale_completed, (now - 90_000, now - 90_000))
    os.utime(stale_prior_writer, (now - 110_000, now - 110_000))
    os.utime(stale_current_writer, (now - 87_000, now - 87_000))
    os.utime(current_writer, (now - 60, now - 60))
    os.utime(stale_marker, (now - 90_000, now - 90_000))
    os.utime(recent, (now - 60, now - 60))
    os.utime(recent_marker, (now - 60, now - 60))

    deleted, surviving = backup_daily._sweep_stale_backup_attempts(
        tmp_path,
        max_age_seconds=86_400,
        now=now,
        writer_started_at=now - 100_000,
    )

    assert deleted == [stale_completed.name, stale_prior_writer.name]
    assert {path.name for path in surviving} == {
        stale_current_writer.name,
        current_writer.name,
        recent.name,
        symlink.name,
    }
    assert not stale_completed.exists()
    assert not stale_marker.exists()
    assert not stale_prior_writer.exists()
    assert stale_current_writer.exists()
    assert current_writer.exists()
    assert recent.exists()
    assert recent_marker.exists()
    assert symlink.is_symlink()
    assert target.read_bytes() == b"outside"


def test_stale_attempt_sweep_keeps_recent_attempt_from_prior_writer(tmp_path):
    from brainlayer import backup_daily

    now = 200_000.0
    recent_prior_writer = tmp_path / ".2026-05-13.db.attempt-1-recent-prior-writer"
    recent_prior_writer.write_bytes(b"recent prior writer")
    os.utime(recent_prior_writer, (now - 3_000, now - 3_000))

    deleted, surviving = backup_daily._sweep_stale_backup_attempts(
        tmp_path,
        max_age_seconds=86_400,
        now=now,
        writer_started_at=now - 1_000,
    )

    assert deleted == []
    assert surviving == [recent_prior_writer]
    assert recent_prior_writer.exists()


def test_create_snapshot_reports_degraded_reclamation_and_preserves_old_unmarked_attempt(tmp_path, monkeypatch, capsys):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    output_dir = tmp_path / "out"
    _create_source_db(source, chunk_count=2)
    output_dir.mkdir()
    old_unmarked = output_dir / ".2026-05-12.db.attempt-1-old-unmarked"
    old_unmarked.write_bytes(b"old unmarked")
    old_mtime = time.time() - backup_daily.DEFAULT_BACKUP_ATTEMPT_MAX_AGE_SECONDS - 60
    os.utime(old_unmarked, (old_mtime, old_mtime))

    def fake_vacuum_into(target_path, **kwargs):  # noqa: ARG001
        with sqlite3.connect(source) as db:
            db.execute("VACUUM INTO ?", (str(target_path),))

    monkeypatch.setattr(
        backup_daily,
        "_brainbar_writer_started_at",
        lambda socket_path=None: (_ for _ in ()).throw(RuntimeError("missing writer timestamp")),
    )
    monkeypatch.setattr(backup_daily, "request_brainbar_vacuum_into", fake_vacuum_into)

    artifact = backup_daily.create_sqlite_backup_artifact(source, output_dir, date_stamp="2026-05-14")

    assert artifact.attempt_reclamation == "degraded"
    assert artifact.writer_probe_error == "RuntimeError: missing writer timestamp"
    assert old_unmarked.name in artifact.surviving_attempts
    assert old_unmarked.exists()
    assert "attempt reclamation degraded: RuntimeError: missing writer timestamp" in capsys.readouterr().out


def test_create_snapshot_does_not_swallow_global_timeout_during_writer_probe(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    monkeypatch.setattr(
        backup_daily,
        "_brainbar_writer_started_at",
        lambda socket_path=None: (_ for _ in ()).throw(backup_daily.BackupTimeoutError("deadline")),
    )

    with pytest.raises(backup_daily.BackupTimeoutError, match="deadline"):
        backup_daily.create_sqlite_backup_artifact(source, tmp_path / "out")


def test_database_logical_size_includes_committed_wal_pages(tmp_path):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    conn = sqlite3.connect(source)
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].upper() == "WAL"
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT)")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        main_file_size = source.stat().st_size
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?)",
            [(f"c{idx}", "x" * 4096) for idx in range(256)],
        )
        conn.commit()

        assert backup_daily._database_logical_size_bytes(source) > main_file_size
    finally:
        conn.close()


def test_brainbar_writer_started_at_reads_initialize_server_info(monkeypatch):
    from brainlayer import backup_daily

    seen: dict[str, object] = {}

    def initialize_response(socket_path, request, timeout_seconds):
        seen.update(socket_path=socket_path, request=request, timeout_seconds=timeout_seconds)
        return {"result": {"serverInfo": {"backupWriterStartedAtUnix": 1234.5}}}

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", initialize_response)

    assert backup_daily._brainbar_writer_started_at("/tmp/brainbar.sock") == 1234.5
    assert seen["socket_path"] == Path("/tmp/brainbar.sock")
    assert seen["request"]["method"] == "initialize"


def test_recent_attempt_growth_is_reserved_in_disk_preflight(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    output_dir = tmp_path / "out"
    _create_source_db(source, chunk_count=2)
    output_dir.mkdir()
    recent = output_dir / ".2026-05-13.db.attempt-1-recent"
    recent.write_bytes(b"x")
    db_size = source.stat().st_size
    base_required = (db_size * 3) + (512 * 1024 * 1024)

    class Disk:
        free = base_required

    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: Disk())

    with pytest.raises(RuntimeError, match="1 recent attempts reserve"):
        backup_daily.create_sqlite_backup_artifact(source, output_dir, date_stamp="2026-05-14")


def test_terminal_response_does_not_clean_unowned_prior_run_attempts(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    prior_attempt = tmp_path / ".2026-05-13.db.attempt-3-prior"
    _create_source_db(prior_attempt, chunk_count=2)

    def successful_response(socket_path, request, timeout_seconds):  # noqa: ARG001
        attempt_target = Path(request["params"]["arguments"]["target_path"])
        _create_source_db(attempt_target, chunk_count=2)
        return {"result": {"content": [{"type": "text", "text": '{"status":"ok"}'}]}}

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", successful_response)

    backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert target.exists()
    assert prior_attempt.exists()


def test_create_snapshot_rejects_low_disk_space(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    conn = sqlite3.connect(source)
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT)")
    conn.commit()
    conn.close()

    class LowDisk:
        free = 1

    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: LowDisk())

    with pytest.raises(RuntimeError, match="Insufficient free space"):
        backup_daily.create_sqlite_backup_gzip(source, tmp_path / "out", date_stamp="2026-05-13")


def test_ensure_drive_folder_chain_creates_missing_folders():
    from brainlayer.backup_daily import ensure_drive_folder_chain

    class FakeExecute:
        def __init__(self, value):
            self.value = value

        def execute(self):
            return self.value

    class FakeFiles:
        def __init__(self):
            self.created = []

        def list(self, **kwargs):
            query = kwargs["q"]
            if "name = 'Brain Drive'" in query:
                return FakeExecute({"files": [{"id": "brain-drive"}]})
            return FakeExecute({"files": []})

        def create(self, body, fields=None, **kwargs):  # noqa: ARG002
            folder_id = f"folder-{body['name']}"
            self.created.append((body["name"], body["parents"][0]))
            return FakeExecute({"id": folder_id})

    class FakeService:
        def __init__(self):
            self._files = FakeFiles()

        def files(self):
            return self._files

    service = FakeService()

    result = ensure_drive_folder_chain(
        service,
        ["Brain Drive", "06_ARCHIVE", "backups", "brainlayer-db"],
    )

    assert result == "folder-brainlayer-db"
    assert ("06_ARCHIVE", "brain-drive") in service.files().created
    assert ("backups", "folder-06_ARCHIVE") in service.files().created
    assert ("brainlayer-db", "folder-backups") in service.files().created


def test_run_backup_verifies_upload_removes_local_and_rotates_last_n(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-05-30.db.gz"
    snapshot.write_bytes(b"backup-bytes")
    verified: list[tuple[str, str, int]] = []
    pruned: list[backup_daily.DriveRetentionPolicy] = []

    class FakeArtifact:
        gzip_path = snapshot
        uncompressed_path = None
        sentinel_chunks = 1
        local_retention_deleted: list[str] = []

    monkeypatch.setattr(backup_daily, "create_sqlite_backup_artifact", lambda *args, **kwargs: FakeArtifact())
    monkeypatch.setattr(backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "build_drive_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "ensure_drive_folder_chain", lambda service, folder_parts: "folder-id")
    monkeypatch.setattr(
        backup_daily,
        "verify_sqlite_backup_artifact",
        lambda *args, **kwargs: {
            "verified": True,
            "verification_mode": "quick",
            "sentinel_snapshot_chunks": 1,
            "sentinel_verified_chunks": 1,
        },
    )
    monkeypatch.setattr(
        backup_daily,
        "upload_file_to_drive_raw",
        lambda file_path, folder_id, credentials: {
            "id": "drive-file-id",
            "name": Path(file_path).name,
            "size": str(Path(file_path).stat().st_size),
        },
    )

    def fake_verify(service, *, file_id: str, expected_name: str, expected_size: int) -> None:  # noqa: ARG001
        verified.append((file_id, expected_name, expected_size))

    def fake_prune(service, *, folder_parts, retention_policy):  # noqa: ARG001
        pruned.append(retention_policy)
        return ["2026-05-01.db.gz"]

    monkeypatch.setattr(backup_daily, "verify_drive_upload", fake_verify)
    monkeypatch.setattr(backup_daily, "prune_drive_backups", fake_prune)

    result = backup_daily.run_backup(
        db_path=tmp_path / "brainlayer.db",
        staging_dir=tmp_path,
        date_stamp="2026-05-30",
        upload=True,
        retention_policy=backup_daily.DriveRetentionPolicy(keep_latest=7),
    )

    assert verified == [("drive-file-id", "2026-05-30.db.gz", len(b"backup-bytes"))]
    assert pruned == [backup_daily.DriveRetentionPolicy(keep_latest=7)]
    assert result["uploaded"] is True
    assert result["local_removed"] is True
    assert result["retention_deleted"] == ["2026-05-01.db.gz"]
    assert not snapshot.exists()


def test_run_backup_appends_result_to_file_log(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-05-30.db.gz"
    snapshot.write_bytes(b"backup-bytes")
    log_path = tmp_path / "backup-daily.log"

    class FakeArtifact:
        gzip_path = snapshot
        uncompressed_path = None
        sentinel_chunks = 1
        local_retention_deleted: list[str] = []
        attempt_reclamation = "degraded"
        writer_probe_error = "RuntimeError: missing writer timestamp"

    monkeypatch.setattr(backup_daily, "create_sqlite_backup_artifact", lambda *args, **kwargs: FakeArtifact())
    monkeypatch.setattr(backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "build_drive_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "ensure_drive_folder_chain", lambda service, folder_parts: "folder-id")
    monkeypatch.setattr(
        backup_daily,
        "verify_sqlite_backup_artifact",
        lambda *args, **kwargs: {
            "verified": True,
            "verification_mode": "quick",
            "sentinel_snapshot_chunks": 1,
            "sentinel_verified_chunks": 1,
        },
    )
    monkeypatch.setattr(
        backup_daily,
        "upload_file_to_drive_raw",
        lambda file_path, folder_id, credentials: {
            "id": "drive-file-id",
            "name": Path(file_path).name,
            "size": str(Path(file_path).stat().st_size),
        },
    )
    monkeypatch.setattr(backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(backup_daily, "prune_drive_backups", lambda *args, **kwargs: [])

    result = backup_daily.run_backup(
        db_path=tmp_path / "brainlayer.db",
        staging_dir=tmp_path,
        date_stamp="2026-05-30",
        upload=True,
        log_path=log_path,
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["snapshot"] == str(snapshot)
    assert logged["drive_file"]["id"] == "drive-file-id"
    assert logged["verified"] is True
    assert logged["attempt_reclamation"] == "degraded"
    assert logged["writer_probe_error"] == "RuntimeError: missing writer timestamp"
    assert logged == result


def test_run_backup_appends_file_log_when_upload_fails(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-05-30.db.gz"
    snapshot.write_bytes(b"backup-bytes")
    log_path = tmp_path / "backup-daily.log"

    class FakeArtifact:
        gzip_path = snapshot
        uncompressed_path = None
        sentinel_chunks = 1
        local_retention_deleted: list[str] = []

    monkeypatch.setattr(backup_daily, "create_sqlite_backup_artifact", lambda *args, **kwargs: FakeArtifact())
    monkeypatch.setattr(backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "build_drive_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "ensure_drive_folder_chain", lambda service, folder_parts: "folder-id")
    monkeypatch.setattr(
        backup_daily,
        "upload_file_to_drive_raw",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("drive unavailable")),
    )

    with pytest.raises(RuntimeError, match="drive unavailable"):
        backup_daily.run_backup(
            db_path=tmp_path / "brainlayer.db",
            staging_dir=tmp_path,
            date_stamp="2026-05-30",
            upload=True,
            log_path=log_path,
        )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["snapshot"] == str(snapshot)
    assert logged["uploaded"] is False
    assert logged["verified"] is False
    assert logged["error_type"] == "RuntimeError"
    assert logged["error"] == "drive unavailable"


def test_run_backup_logs_degraded_writer_probe_when_artifact_creation_fails(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    log_path = tmp_path / "backup-daily.log"
    _create_source_db(source, chunk_count=2)
    monkeypatch.setattr(
        backup_daily,
        "_brainbar_writer_started_at",
        lambda socket_path=None: (_ for _ in ()).throw(RuntimeError("daemon unavailable")),
    )
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("vacuum unavailable")),
    )

    with pytest.raises(RuntimeError, match="vacuum unavailable"):
        backup_daily.run_backup(
            db_path=source,
            staging_dir=tmp_path / "out",
            date_stamp="2026-05-30",
            upload=False,
            log_path=log_path,
        )

    logged = json.loads(log_path.read_text(encoding="utf-8"))
    assert logged["attempt_reclamation"] == "degraded"
    assert logged["writer_probe_error"] == "RuntimeError: daemon unavailable"
    assert logged["error_type"] == "RuntimeError"
    assert logged["error"] == "vacuum unavailable"


def test_run_backup_uses_env_log_path_without_explicit_log_path(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-05-30.db.gz"
    snapshot.write_bytes(b"backup-bytes")
    safe_log_path = tmp_path / "guarded" / "backup-daily.log"
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PATH", str(safe_log_path))
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PROVENANCE", "pytest")

    class FakeArtifact:
        gzip_path = snapshot
        uncompressed_path = None
        sentinel_chunks = 1
        local_retention_deleted: list[str] = []

    monkeypatch.setattr(backup_daily, "create_sqlite_backup_artifact", lambda *args, **kwargs: FakeArtifact())

    result = backup_daily.run_backup(
        db_path=tmp_path / "brainlayer.db",
        staging_dir=tmp_path,
        date_stamp="2026-05-30",
        upload=False,
    )

    lines = safe_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged == result
    assert logged["backup_log_provenance"] == "pytest"


def test_tests_autouse_backup_log_guard_points_away_from_prod_log() -> None:
    from brainlayer import backup_daily

    guarded = Path(os.environ["BRAINLAYER_BACKUP_LOG_PATH"]).expanduser()
    assert guarded != backup_daily.DEFAULT_LOG_PATH
    assert "pytest" in guarded.as_posix()


def test_prune_drive_backups_keeps_only_latest_n_snapshots():
    from brainlayer.backup_daily import DriveRetentionPolicy, prune_drive_backups

    class FakeExecute:
        def __init__(self, value):
            self.value = value

        def execute(self):
            return self.value

    class FakeFiles:
        def __init__(self):
            self.deleted: list[str] = []
            self.files = [{"id": f"id-{day}", "name": f"2026-05-{day:02d}.db.gz"} for day in range(1, 10)]

        def list(self, **kwargs):  # noqa: ARG002
            query = kwargs["q"]
            if "mimeType = 'application/vnd.google-apps.folder'" in query:
                return FakeExecute({"files": [{"id": "folder-id", "name": "brainlayer-db"}]})
            return FakeExecute({"files": self.files})

        def delete(self, fileId, **kwargs):  # noqa: N803, ARG002
            self.deleted.append(fileId)
            return FakeExecute({})

    class FakeService:
        def __init__(self):
            self._files = FakeFiles()

        def files(self):
            return self._files

    service = FakeService()

    deleted = prune_drive_backups(
        service,
        folder_parts=["brainlayer-db"],
        retention_policy=DriveRetentionPolicy(keep_latest=4),
    )

    assert deleted == [
        "2026-05-05.db.gz",
        "2026-05-04.db.gz",
        "2026-05-03.db.gz",
        "2026-05-02.db.gz",
        "2026-05-01.db.gz",
    ]
    assert service.files().deleted == ["id-5", "id-4", "id-3", "id-2", "id-1"]


def test_launchd_installer_knows_backup_target():
    install_path = Path("scripts/launchd/install.sh")
    wrapper_path = Path("scripts/launchd/backup-daily.sh")
    plist_path = Path("scripts/launchd/com.brainlayer.backup-daily.plist")

    assert install_path.is_file(), f"Installer not found at {install_path}; check test working directory"
    assert wrapper_path.is_file(), f"Backup wrapper not found at {wrapper_path}; check launchd wrapper is committed"
    assert plist_path.is_file(), f"Backup plist not found at {plist_path}; check launchd template is committed"

    install = install_path.read_text()
    wrapper = wrapper_path.read_text()
    plist = plist_path.read_text()

    assert "backup-daily" in install
    assert "install_backup_script" in install
    assert "escaped_brainlayer_dir" in install
    assert "__BRAINLAYER_DIR_VALUE__" in install
    assert "PYTHONPATH" in wrapper
    assert "__BRAINLAYER_DIR_VALUE__" in wrapper
    assert "<string>com.brainlayer.backup-daily</string>" in plist
    assert "<integer>3</integer>" in plist
    assert "<integer>17</integer>" in plist
    assert "<key>KeepAlive</key>" not in plist
    assert "<key>ExitTimeOut</key>" in plist
    assert "<integer>300</integer>" in plist
    assert "BRAINLAYER_BACKUP_CLIENT_TIMEOUT_SECONDS:=0" in wrapper
    assert "BRAINLAYER_BACKUP_TIMEOUT_SECONDS:=7200" in wrapper
    assert "BRAINLAYER_BACKUP_ATTEMPT_MAX_AGE_SECONDS:=86400" in wrapper
    assert "BRAINLAYER_BACKUP_LOG_PROVENANCE:=real" in wrapper


def test_main_enforces_configured_backup_timeout(monkeypatch, capsys):
    from brainlayer import backup_daily

    def slow_backup(**kwargs):  # noqa: ARG001
        time.sleep(5)

    monkeypatch.setenv("BRAINLAYER_BACKUP_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(backup_daily, "run_backup", slow_backup)

    assert backup_daily.main() == 124
    assert "brainlayer backup timed out after 1s" in capsys.readouterr().out
