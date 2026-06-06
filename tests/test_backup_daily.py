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
            server.listen(1)
            ready.set()
            conn, _ = server.accept()
            with conn:
                data = b""
                while not data.endswith(b"\n"):
                    data += conn.recv(65_536)
                request = json.loads(data.decode("utf-8"))
                received.put(request)
                args = request["params"]["arguments"]
                target_path = Path(args["target_path"])
                with sqlite3.connect(source_db) as db:
                    db.execute("VACUUM INTO ?", (str(target_path),))
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps({"status": "ok", "target_path": str(target_path)})}
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
    assert request["params"]["arguments"]["target_path"].endswith("/2026-05-13.db")
    assert snapshot.name == "2026-05-13.db.gz"


def test_brainbar_vacuum_request_retries_closed_socket_with_backoff(tmp_path, monkeypatch, capsys):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    calls = []
    sleeps = []

    def flaky_send(socket_path, request, timeout_seconds):  # noqa: ARG001
        calls.append((socket_path, request["params"]["name"], timeout_seconds))
        if len(calls) < 3:
            raise RuntimeError("BrainBar socket closed without response: /tmp/brainbar.sock")
        target.write_bytes(b"ok")
        return {"result": {"content": [{"type": "text", "text": '{"status":"ok"}'}]}}

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", flaky_send)
    monkeypatch.setattr(backup_daily.time, "sleep", lambda seconds: sleeps.append(seconds))

    backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert len(calls) == 3
    assert sleeps == [60, 60]
    output = capsys.readouterr().out
    assert "BrainBar vacuum snapshot attempt 1/3 failed" in output
    assert "BrainBar vacuum snapshot attempt 2/3 failed" in output
    assert "retrying in 60s" in output


def test_brainbar_vacuum_request_fails_loud_after_retry_budget(tmp_path, monkeypatch, capsys):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    calls = []
    sleeps = []

    def closed_socket(socket_path, request, timeout_seconds):  # noqa: ARG001
        calls.append(request["params"]["name"])
        raise RuntimeError("BrainBar socket closed without response: /tmp/brainbar.sock")

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", closed_socket)
    monkeypatch.setattr(backup_daily.time, "sleep", lambda seconds: sleeps.append(seconds))

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
    monkeypatch.setattr(backup_daily.time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(backup_daily.BackupTimeoutError):
        backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert calls == ["brain_backup_vacuum_into"]
    assert sleeps == []


def test_brainbar_vacuum_request_accepts_valid_target_after_lost_response(tmp_path, monkeypatch, capsys):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    calls = []
    sleeps = []

    def closed_after_success(socket_path, request, timeout_seconds):  # noqa: ARG001
        calls.append(request["params"]["name"])
        _create_source_db(target, chunk_count=2)
        raise RuntimeError("BrainBar socket closed without response: /tmp/brainbar.sock")

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", closed_after_success)
    monkeypatch.setattr(backup_daily.time, "sleep", lambda seconds: sleeps.append(seconds))

    backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert calls == ["brain_backup_vacuum_into"]
    assert sleeps == []
    assert "target exists and passed quick_check" in capsys.readouterr().out


def test_brainbar_vacuum_request_removes_invalid_target_before_retry(tmp_path, monkeypatch, capsys):
    from brainlayer import backup_daily

    target = tmp_path / "snapshot.db"
    calls = []
    sleeps = []

    def invalid_then_success(socket_path, request, timeout_seconds):  # noqa: ARG001
        calls.append(request["params"]["name"])
        if len(calls) == 1:
            target.write_bytes(b"not sqlite")
            raise RuntimeError("BrainBar socket closed without response: /tmp/brainbar.sock")
        assert not target.exists()
        _create_source_db(target, chunk_count=2)
        return {"result": {"content": [{"type": "text", "text": '{"status":"ok"}'}]}}

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", invalid_then_success)
    monkeypatch.setattr(backup_daily.time, "sleep", lambda seconds: sleeps.append(seconds))

    backup_daily.request_brainbar_vacuum_into(target, socket_path="/tmp/brainbar.sock")

    assert calls == ["brain_backup_vacuum_into", "brain_backup_vacuum_into"]
    assert sleeps == [60]
    output = capsys.readouterr().out
    assert "removing invalid existing target" in output
    assert "retrying in 60s" in output


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
    assert "BRAINLAYER_BACKUP_TIMEOUT_SECONDS:=300" in wrapper


def test_main_enforces_configured_backup_timeout(monkeypatch, capsys):
    from brainlayer import backup_daily

    def slow_backup(**kwargs):  # noqa: ARG001
        time.sleep(5)

    monkeypatch.setenv("BRAINLAYER_BACKUP_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(backup_daily, "run_backup", slow_backup)

    assert backup_daily.main() == 124
    assert "brainlayer backup timed out after 1s" in capsys.readouterr().out
