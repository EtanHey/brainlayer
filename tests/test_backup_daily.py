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


def _create_source_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    journal_mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    assert journal_mode.upper() == "WAL"
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO chunks VALUES ('c1', 'hello')")
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
