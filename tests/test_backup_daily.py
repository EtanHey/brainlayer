import gzip
import sqlite3
from pathlib import Path

import pytest


def test_create_snapshot_gzip_is_restorable(tmp_path):
    from brainlayer.backup_daily import create_sqlite_backup_gzip

    source = tmp_path / "brainlayer.db"
    conn = sqlite3.connect(source)
    journal_mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    assert journal_mode.upper() == "WAL"
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO chunks VALUES ('c1', 'hello')")
    conn.commit()
    conn.close()

    out_dir = tmp_path / "out"
    snapshot = create_sqlite_backup_gzip(source, out_dir, date_stamp="2026-05-13")

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


def test_upload_to_drive_creates_missing_folder_chain(tmp_path):
    from brainlayer.backup_daily import upload_file_to_drive

    class FakeExecute:
        def __init__(self, value):
            self.value = value

        def execute(self):
            return self.value

        def next_chunk(self):
            return None, self.value

    class FakeFiles:
        def __init__(self):
            self.created = []

        def list(self, **kwargs):
            query = kwargs["q"]
            if "name = 'Brain Drive'" in query:
                return FakeExecute({"files": [{"id": "brain-drive"}]})
            return FakeExecute({"files": []})

        def create(self, body, media_body=None, fields=None, **kwargs):  # noqa: ARG002
            if media_body is None:
                folder_id = f"folder-{body['name']}"
                self.created.append((body["name"], body["parents"][0]))
                return FakeExecute({"id": folder_id})
            self.created.append((body["name"], body["parents"][0], "file"))
            return FakeExecute({"id": "backup-file", "name": body["name"]})

    class FakeService:
        def __init__(self):
            self._files = FakeFiles()

        def files(self):
            return self._files

    gz = tmp_path / "2026-05-13.db.gz"
    gz.write_bytes(b"backup")
    service = FakeService()

    result = upload_file_to_drive(
        service,
        gz,
        folder_parts=["Brain Drive", "06_ARCHIVE", "backups", "brainlayer-db"],
    )

    assert result == {"id": "backup-file", "name": "2026-05-13.db.gz"}
    assert ("06_ARCHIVE", "brain-drive") in service.files().created
    assert ("backups", "folder-06_ARCHIVE") in service.files().created
    assert ("brainlayer-db", "folder-backups") in service.files().created
    assert ("2026-05-13.db.gz", "folder-brainlayer-db", "file") in service.files().created


def test_launchd_installer_knows_backup_target():
    install_path = Path("scripts/launchd/install.sh")
    plist_path = Path("scripts/launchd/com.brainlayer.backup-daily.plist")

    assert install_path.is_file(), f"Installer not found at {install_path}; check test working directory"
    assert plist_path.is_file(), f"Backup plist not found at {plist_path}; check launchd template is committed"

    install = install_path.read_text()
    plist = plist_path.read_text()

    assert "backup-daily" in install
    assert "install_backup_script" in install
    assert "<string>com.brainlayer.backup-daily</string>" in plist
    assert "<integer>3</integer>" in plist
    assert "<integer>17</integer>" in plist
