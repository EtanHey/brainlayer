import gzip
import io
import json
import os
import queue
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from brainlayer.backup_daily import _request_macos_purge as _real_macos_purge
from tests.drive_listing_assertions import (
    assert_non_trashed_drive_files_only_grow_or_are_trashed,
    snapshot_non_trashed_drive_ids,
)


@pytest.fixture(autouse=True)
def _stable_backup_machine_id(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "test-machine")
    monkeypatch.setenv("BRAINBAR_SOCKET_PATH", str(tmp_path / "no-brainbar.sock"))
    from brainlayer import backup_daily

    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda _path: None)
    monkeypatch.setattr(backup_daily, "_request_macos_purge", lambda _path, _bytes: None, raising=False)
    monkeypatch.setattr(backup_daily, "POST_PURGE_POLL_SECONDS", 0, raising=False)


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


def test_local_backup_cap_is_decoupled_from_drive_retention(monkeypatch):
    from brainlayer import backup_daily

    monkeypatch.delenv("BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS", raising=False)
    assert backup_daily.DEFAULT_LOCAL_COMPRESSED_KEEP == 3
    assert backup_daily.DEFAULT_LOCAL_UNCOMPRESSED_KEEP == 1
    assert backup_daily.DEFAULT_DRIVE_KEEP == 7
    assert backup_daily.DAILY_RETENTION.keep_latest == 7
    assert backup_daily.WEEKLY_RETENTION.keep_latest == 7
    assert backup_daily._drive_upload_stall_max_attempts() == 3


def test_noncanonical_machine_uses_its_own_default_drive_folder(monkeypatch):
    from brainlayer import backup_daily

    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "m1-brainlayer")

    machine_id = backup_daily.resolve_machine_id()

    assert machine_id == "m1-brainlayer"
    assert backup_daily.default_drive_folder_parts(machine_id) == [
        "Brain Drive",
        "06_ARCHIVE",
        "backups",
        "brainlayer-db-m1-brainlayer",
    ]
    assert backup_daily.default_drive_folder_parts(machine_id) != backup_daily.CANONICAL_FOLDER_PARTS


def test_canonical_m4_machine_keeps_existing_drive_folder():
    from brainlayer import backup_daily

    assert backup_daily.default_drive_folder_parts(backup_daily.CANONICAL_MACHINE_ID) == [
        "Brain Drive",
        "06_ARCHIVE",
        "backups",
        "brainlayer-db",
    ]


def test_backup_daily_plist_does_not_override_per_machine_drive_folder():
    import plistlib

    plist = plistlib.loads(Path("scripts/launchd/com.brainlayer.backup-daily.plist").read_bytes())

    assert "BRAINLAYER_BACKUP_DRIVE_FOLDER" not in plist["EnvironmentVariables"]


@pytest.mark.parametrize(
    ("value", "enabled"),
    (("0", False), ("", False), ("false", False), ("yes", False), ("on", False), ("1", True), ("true", True)),
)
def test_drive_retention_env_is_explicit_only(value, enabled, monkeypatch):
    from brainlayer import backup_daily

    monkeypatch.setattr(backup_daily, "DRIVE_RETENTION_ENABLED", False)
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_RETENTION", value)

    assert backup_daily._drive_retention_enabled() is enabled


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", str(threading.TIMEOUT_MAX * 2)])
def test_drive_upload_numeric_settings_reject_non_finite_or_unusable_timeouts(monkeypatch, value):
    from brainlayer import backup_daily

    monkeypatch.setenv("BRAINLAYER_DRIVE_UPLOAD_DEADLINE_FLOOR_SECONDS", value)

    with pytest.raises(ValueError, match="must be a positive number"):
        backup_daily._configured_positive_number(
            backup_daily.DRIVE_UPLOAD_DEADLINE_FLOOR_ENV,
            backup_daily.DEFAULT_DRIVE_UPLOAD_DEADLINE_FLOOR_SECONDS,
            maximum=threading.TIMEOUT_MAX,
        )


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
    monkeypatch.setattr(backup_daily, "assert_drive_folder_owned_by_machine", lambda *args, **kwargs: None)
    monkeypatch.setattr(backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(backup_daily, "prune_drive_backups", lambda *args, **kwargs: [])

    def fake_upload(file_path, folder_id, credentials, *, machine_id):  # noqa: ARG001
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
    assert result["pragma"] == "skipped"
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
    monkeypatch.setattr(backup_daily, "assert_drive_folder_owned_by_machine", lambda *args, **kwargs: None)
    monkeypatch.setattr(backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(backup_daily, "prune_drive_backups", lambda *args, **kwargs: [])

    def fake_upload(file_path, folder_id, credentials, *, machine_id):  # noqa: ARG001
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


def test_prune_local_gzip_snapshots_keeps_newest_three_and_requires_verified_coverage(tmp_path):
    from brainlayer.backup_daily import prune_local_gzip_snapshots

    for day in range(1, 7):
        (tmp_path / f"2026-06-{day:02d}.db.gz").write_bytes(f"gzip-{day}".encode())

    deleted = prune_local_gzip_snapshots(
        tmp_path,
        verified_drive_names={
            "2026-06-01.db.gz",
            "2026-06-02.db.gz",
            "2026-06-03.db.gz",
        },
    )

    assert deleted == ["2026-06-03.db.gz", "2026-06-02.db.gz", "2026-06-01.db.gz"]
    assert sorted(path.name for path in tmp_path.glob("*.db.gz")) == [
        "2026-06-04.db.gz",
        "2026-06-05.db.gz",
        "2026-06-06.db.gz",
    ]


def test_prune_local_gzip_snapshots_preserves_unverified_archive_without_three_newer_verified_copies(tmp_path):
    from brainlayer.backup_daily import prune_local_gzip_snapshots

    for day in range(1, 6):
        (tmp_path / f"2026-06-{day:02d}.db.gz").write_bytes(f"gzip-{day}".encode())

    deleted = prune_local_gzip_snapshots(
        tmp_path,
        verified_drive_names={"2026-06-02.db.gz", "2026-06-03.db.gz"},
    )

    assert deleted == ["2026-06-02.db.gz"]
    assert (tmp_path / "2026-06-01.db.gz").exists()
    assert (tmp_path / "2026-06-04.db.gz").exists()


def test_prune_local_gzip_snapshots_deletes_unverified_archive_with_three_newer_verified_copies(tmp_path):
    from brainlayer.backup_daily import prune_local_gzip_snapshots

    for day in range(1, 7):
        (tmp_path / f"2026-06-{day:02d}.db.gz").write_bytes(f"gzip-{day}".encode())

    deleted = prune_local_gzip_snapshots(
        tmp_path,
        verified_drive_names={
            "2026-06-04.db.gz",
            "2026-06-05.db.gz",
            "2026-06-06.db.gz",
        },
    )

    assert deleted == ["2026-06-03.db.gz", "2026-06-02.db.gz", "2026-06-01.db.gz"]
    assert sorted(path.name for path in tmp_path.glob("*.db.gz")) == [
        "2026-06-04.db.gz",
        "2026-06-05.db.gz",
        "2026-06-06.db.gz",
    ]


def test_run_backup_wires_verified_log_provenance_into_local_gzip_pruning(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-06-10.db.gz"
    snapshot.write_bytes(b"backup-bytes")
    for day in range(1, 10):
        (tmp_path / f"2026-06-{day:02d}.db.gz").write_bytes(f"gzip-{day}".encode())
    log_path = tmp_path / "backup-daily.log"
    log_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "uploaded": True,
                    "verified": True,
                    "snapshot": str(tmp_path / f"2026-06-{day:02d}.db.gz"),
                }
            )
            for day in (4, 6, 8)
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeArtifact:
        gzip_path = snapshot
        uncompressed_path = None
        sentinel_chunks = 1
        local_retention_deleted: list[str] = []

    monkeypatch.setattr(backup_daily, "create_sqlite_backup_artifact", lambda *args, **kwargs: FakeArtifact())
    monkeypatch.setattr(backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "build_drive_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "ensure_drive_folder_chain", lambda service, folder_parts: "folder-id")
    monkeypatch.setattr(backup_daily, "assert_drive_folder_owned_by_machine", lambda *args, **kwargs: None)
    monkeypatch.setattr(backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        backup_daily,
        "verify_sqlite_backup_artifact",
        lambda *args, **kwargs: {"verified": True, "verification_mode": "quick"},
    )
    monkeypatch.setattr(
        backup_daily,
        "upload_file_to_drive_raw",
        lambda file_path, folder_id, credentials, *, machine_id: {
            "id": "drive-file-id",
            "name": Path(file_path).name,
            "size": str(Path(file_path).stat().st_size),
        },
    )
    monkeypatch.setattr(backup_daily, "prune_drive_backups", lambda *args, **kwargs: [])

    result = backup_daily.run_backup(
        db_path=tmp_path / "brainlayer.db",
        staging_dir=tmp_path,
        date_stamp="2026-06-10",
        upload=True,
        remove_local_after_upload=False,
        log_path=log_path,
    )

    assert result["verified"] is True
    assert result["local_gzip_retention_deleted"] == [
        "2026-06-06.db.gz",
        "2026-06-05.db.gz",
        "2026-06-04.db.gz",
        "2026-06-03.db.gz",
        "2026-06-02.db.gz",
        "2026-06-01.db.gz",
    ]
    assert (tmp_path / "2026-06-07.db.gz").exists()
    assert (tmp_path / "2026-06-09.db.gz").exists()
    assert not (tmp_path / "2026-06-03.db.gz").exists()


def test_verified_snapshot_log_parser_accepts_only_uploaded_verified_snapshot_names(tmp_path):
    from brainlayer import backup_daily

    log_path = tmp_path / "backup-daily.log"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "uploaded": True,
                        "verified": True,
                        "snapshot": "/backups/2026-06-01.db.gz",
                    }
                ),
                json.dumps(
                    {
                        "uploaded": True,
                        "verified": False,
                        "snapshot": "/backups/2026-06-02.db.gz",
                    }
                ),
                "not-json",
                json.dumps({"uploaded": True, "verified": True, "snapshot": "not-a-snapshot.db"}),
                json.dumps(
                    {
                        "uploaded": False,
                        "verified": True,
                        "snapshot": "/backups/2026-06-03.db.gz",
                    }
                ),
                json.dumps(
                    {
                        "uploaded": True,
                        "verified": True,
                        "snapshot": "/backups/2026-06-04.db.gz",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert backup_daily._verified_snapshot_names_from_log(log_path) == {
        "2026-06-01.db.gz",
        "2026-06-04.db.gz",
    }


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
    assert backup_daily._configured_backup_timeout_seconds() == 28800

    monkeypatch.setenv("BRAINLAYER_BACKUP_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="must be at least 1 second"):
        backup_daily._configured_backup_timeout_seconds()


def test_backup_target_primary_gate_uses_pages_and_chunks_without_pragma_scan(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "snapshot.db"
    snapshot.touch()
    statements = []

    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def execute(self, statement):
            statements.append(statement)
            if statement == "PRAGMA page_count":
                return Cursor((128,))
            if statement == "SELECT COUNT(*) FROM chunks":
                return Cursor((3,))
            raise AssertionError(f"PRAGMA scan is beyond the primary gate: {statement}")

        def close(self):
            return None

    monkeypatch.setattr(backup_daily.sqlite3, "connect", lambda *args, **kwargs: Connection())

    assert backup_daily._validate_backup_target(snapshot) == 3
    assert statements == ["PRAGMA page_count", "SELECT COUNT(*) FROM chunks"]


def test_optional_sqlite_check_is_disabled_by_default_and_timeout_is_nonfatal(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "snapshot.db"
    _create_source_db(snapshot, chunk_count=2)
    monkeypatch.delenv("BRAINLAYER_BACKUP_SQLITE_CHECK_TIMEOUT_SECONDS", raising=False)

    assert backup_daily._configured_sqlite_check_timeout_seconds() == 0
    assert backup_daily._optional_sqlite_pragma_check(snapshot, "quick_check") == "skipped"

    monkeypatch.setenv("BRAINLAYER_BACKUP_SQLITE_CHECK_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(
        backup_daily.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=[sys.executable], timeout=1)),
    )

    assert backup_daily._optional_sqlite_pragma_check(snapshot, "quick_check") == "timeout"


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


def test_brainbar_writer_started_at_rejects_unexpected_initialize_shape(monkeypatch):
    from brainlayer import backup_daily

    monkeypatch.setattr(
        backup_daily,
        "_send_brainbar_json_request",
        lambda *args, **kwargs: {"result": "error"},
    )

    with pytest.raises(RuntimeError, match="missing backupWriterStartedAtUnix"):
        backup_daily._brainbar_writer_started_at("/tmp/brainbar.sock")


def test_recent_attempt_growth_is_reserved_in_disk_preflight(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    output_dir = tmp_path / "out"
    _create_source_db(source, chunk_count=2)
    output_dir.mkdir()
    recent = output_dir / ".2026-05-13.db.attempt-1-recent"
    recent.write_bytes(b"x")
    db_size = source.stat().st_size
    base_required = backup_daily._peak_backup_required_bytes(db_size, db_size, 0, full=False)

    class Disk:
        free = base_required

    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: Disk())
    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda _path: None)

    with pytest.raises(RuntimeError, match="1 recent attempts reserve"):
        backup_daily.create_sqlite_backup_artifact(source, output_dir, date_stamp="2026-05-14")


def test_important_capacity_cannot_bypass_surviving_attempt_raw_reserve(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    output_dir = tmp_path / "out"
    _create_source_db(source, chunk_count=2)
    output_dir.mkdir()
    (output_dir / ".2026-05-13.db.attempt-1-recent").write_bytes(b"x")
    raw_free = 20_000_000_000
    db_size = raw_free - backup_daily.MIN_RAW_FREE_BYTES - 1

    monkeypatch.setattr(backup_daily, "_database_logical_size_bytes", lambda _path: db_size)
    monkeypatch.setattr(backup_daily, "_brainbar_writer_started_at", lambda _socket_path: time.time())
    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: SimpleNamespace(free=raw_free))
    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda _path: 10**12)
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("writer reached")),
    )

    with pytest.raises(RuntimeError, match="Insufficient free space.*1 recent attempts reserve"):
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
    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda _path: None)

    with pytest.raises(RuntimeError, match="Insufficient free space"):
        backup_daily.create_sqlite_backup_gzip(source, tmp_path / "out", date_stamp="2026-05-13")


def test_create_snapshot_accepts_space_for_raw_and_gzip(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    db_size = backup_daily._database_logical_size_bytes(source)

    class Disk:
        free = backup_daily._peak_backup_required_bytes(db_size, db_size, 0, full=False)

    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: Disk())
    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda _path: None)

    def reached_writer(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("writer reached")

    monkeypatch.setattr(backup_daily, "request_brainbar_vacuum_into", reached_writer)
    with pytest.raises(RuntimeError, match="writer reached"):
        backup_daily.create_sqlite_backup_artifact(source, tmp_path / "out", date_stamp="2026-05-14")


def test_create_snapshot_requests_purge_before_writer(tmp_path, monkeypatch, capsys):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    db_size = 2 * 1024 * 1024 * 1024
    monkeypatch.setattr(backup_daily, "_database_logical_size_bytes", lambda _path: db_size)
    required = backup_daily._peak_backup_required_bytes(db_size, db_size, 0, full=False)
    raw_before = 5 * 1024 * 1024 * 1024
    raw_after = required + backup_daily.COPY_CHUNK_BYTES
    readings = iter([raw_before, raw_after])
    requested = []
    monkeypatch.setattr(
        backup_daily.shutil, "disk_usage", lambda _path: SimpleNamespace(free=next(readings, raw_after))
    )
    monkeypatch.setattr(backup_daily, "_request_macos_purge", lambda path, amount: requested.append((path, amount)))
    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda _path: required + 10**12)
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("writer reached")),
    )

    with pytest.raises(RuntimeError, match="writer reached"):
        backup_daily.create_sqlite_backup_artifact(source, tmp_path / "out", date_stamp="2026-05-14")
    assert requested == [(tmp_path / "out", required - raw_before + backup_daily.COPY_CHUNK_BYTES)]
    output = capsys.readouterr().out
    assert f"raw_before={raw_before}" in output
    assert f"raw_after={raw_after}" in output


def test_create_snapshot_skips_purge_when_raw_is_sufficient(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    db_size = backup_daily._database_logical_size_bytes(source)
    required = backup_daily._peak_backup_required_bytes(db_size, db_size, 0, full=False)
    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: SimpleNamespace(free=required))
    monkeypatch.setattr(backup_daily, "_request_macos_purge", lambda *_args: pytest.fail("purge not needed"))
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("writer reached")),
    )

    with pytest.raises(RuntimeError, match="writer reached"):
        backup_daily.create_sqlite_backup_artifact(source, tmp_path / "out", date_stamp="2026-05-14")


def test_create_snapshot_rejects_insufficient_purge_with_both_raw_readings(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    db_size = backup_daily._database_logical_size_bytes(source)
    required = backup_daily._peak_backup_required_bytes(db_size, db_size, 0, full=False)
    raw_before = required // 2
    raw_after = required - 1
    readings = iter([raw_before, raw_after])
    requested = []
    monkeypatch.setattr(
        backup_daily.shutil, "disk_usage", lambda _path: SimpleNamespace(free=next(readings, raw_after))
    )
    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda _path: required + 10**12)
    monkeypatch.setattr(backup_daily, "_request_macos_purge", lambda path, amount: requested.append(amount))
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *args, **kwargs: pytest.fail("writer must not run"),
    )

    with pytest.raises(RuntimeError, match=rf"raw_before={raw_before}.*raw_after={raw_after}.*required={required}"):
        backup_daily.create_sqlite_backup_artifact(source, tmp_path / "out", date_stamp="2026-05-14")
    assert requested == [required - raw_before + backup_daily.COPY_CHUNK_BYTES]


def test_create_snapshot_reports_unavailable_purge_api(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    db_size = backup_daily._database_logical_size_bytes(source)
    required = backup_daily._peak_backup_required_bytes(db_size, db_size, 0, full=False)
    raw = required - 1
    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: SimpleNamespace(free=raw))
    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda _path: required + 1)
    monkeypatch.setattr(
        backup_daily,
        "_request_macos_purge",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("CacheDelete unavailable")),
    )
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *args, **kwargs: pytest.fail("writer must not run"),
    )

    with pytest.raises(
        RuntimeError, match=rf"raw_before={raw} raw_after={raw}.*required={required}.*CacheDelete unavailable"
    ):
        backup_daily.create_sqlite_backup_artifact(source, tmp_path / "out", date_stamp="2026-05-14")


def test_create_snapshot_accepts_raw_gain_despite_purge_error(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    db_size = backup_daily._database_logical_size_bytes(source)
    required = backup_daily._peak_backup_required_bytes(db_size, db_size, 0, full=False)
    readings = iter([required - 1, required])
    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: SimpleNamespace(free=next(readings, required)))
    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda _path: required + 1)
    monkeypatch.setattr(
        backup_daily, "_request_macos_purge", lambda *_args: (_ for _ in ()).throw(RuntimeError("API failed"))
    )
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("writer reached")),
    )

    with pytest.raises(RuntimeError, match="writer reached"):
        backup_daily.create_sqlite_backup_artifact(source, tmp_path / "out", date_stamp="2026-05-14")


def test_create_snapshot_waits_for_delayed_raw_gain(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    db_size = backup_daily._database_logical_size_bytes(source)
    required = backup_daily._peak_backup_required_bytes(db_size, db_size, 0, full=False)
    readings = iter([required - 1, required - 1, required - 1, required])
    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: SimpleNamespace(free=next(readings, required)))
    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda _path: required + 1)
    monkeypatch.setattr(backup_daily, "_request_macos_purge", lambda *_args: None)
    monkeypatch.setattr(backup_daily.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("writer reached")),
    )

    with pytest.raises(RuntimeError, match="writer reached"):
        backup_daily.create_sqlite_backup_artifact(source, tmp_path / "out", date_stamp="2026-05-14")


def test_create_snapshot_waits_beyond_one_second_for_raw_gain(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    db_size = backup_daily._database_logical_size_bytes(source)
    required = backup_daily._peak_backup_required_bytes(db_size, db_size, 0, full=False)
    readings = iter([required - 1, *([required - 1] * 8), required])
    waited = []
    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: SimpleNamespace(free=next(readings, required)))
    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda _path: required + 1)
    monkeypatch.setattr(backup_daily, "_request_macos_purge", lambda *_args: None)
    monkeypatch.setattr(backup_daily, "POST_PURGE_POLL_SECONDS", 0.5, raising=False)
    monkeypatch.setattr(backup_daily.time, "sleep", lambda seconds: waited.append(seconds))
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("writer reached")),
    )

    with pytest.raises(RuntimeError, match="writer reached"):
        backup_daily.create_sqlite_backup_artifact(source, tmp_path / "out", date_stamp="2026-05-14")
    assert sum(waited) > 1


def test_cachedelete_does_not_relabel_backup_timeout(monkeypatch, tmp_path):
    from brainlayer import backup_daily

    monkeypatch.setattr(backup_daily.sys, "platform", "darwin")
    monkeypatch.setattr(
        backup_daily.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(backup_daily.BackupTimeoutError("deadline")),
    )
    with pytest.raises(backup_daily.BackupTimeoutError, match="deadline"):
        _real_macos_purge(tmp_path, 1)


def test_backup_run_lock_rejects_overlapping_run(tmp_path):
    from brainlayer import backup_daily

    entered = threading.Event()
    release = threading.Event()

    @backup_daily._serialized_backup_run
    def held_run(staging_dir, log_path):  # noqa: ARG001
        entered.set()
        assert release.wait(3)
        return {"verified": True}

    log_path = tmp_path / "backup.log"
    worker = threading.Thread(target=held_run, args=(tmp_path, log_path))
    worker.start()
    try:
        assert entered.wait(2)
        with pytest.raises(RuntimeError, match="backup already running"):
            held_run(tmp_path, log_path)
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    refusal = json.loads(log_path.read_text().splitlines()[-1])
    assert refusal["error_type"] == "BackupAlreadyRunningError"
    assert refusal["verified"] is False
    assert datetime.fromisoformat(refusal["attempted_at"]).tzinfo is not None


def test_create_snapshot_does_not_purge_without_enough_important_capacity(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    db_size = backup_daily._database_logical_size_bytes(source)
    required = backup_daily._peak_backup_required_bytes(db_size, db_size, 0, full=False)
    raw = required - 1
    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: SimpleNamespace(free=raw))
    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda _path: required - 1)
    monkeypatch.setattr(backup_daily, "_request_macos_purge", lambda *_args: pytest.fail("purge not viable"))
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *args, **kwargs: pytest.fail("writer must not run"),
    )

    with pytest.raises(RuntimeError, match=rf"raw_before={raw}.*raw_after={raw}.*required={required}"):
        backup_daily.create_sqlite_backup_artifact(source, tmp_path / "out", date_stamp="2026-05-14")


@pytest.mark.parametrize("failure", [RuntimeError("writer failed"), pytest.param("timeout", id="timeout")])
def test_snapshot_temp_dir_is_removed_on_exception_or_timeout(tmp_path, monkeypatch, failure):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    output_dir = tmp_path / "out"
    _create_source_db(source, chunk_count=2)
    error = backup_daily.BackupTimeoutError("deadline") if failure == "timeout" else failure
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(type(error), match=str(error)):
        backup_daily.create_sqlite_backup_artifact(source, output_dir, date_stamp="2026-05-14")
    assert list(output_dir.glob("brainlayer-backup-*")) == []


def test_abandoned_db_temp_removed_only_after_newer_verified_snapshot(tmp_path):
    from brainlayer import backup_daily

    abandoned = tmp_path / "brainlayer-backup-abandoned"
    abandoned.mkdir()
    (abandoned / ".backup-date").write_text("2026-05-13", encoding="ascii")
    (abandoned / "2026-05-13.db").write_bytes(b"old private snapshot")
    unmarked = tmp_path / "brainlayer-backup-unmarked"
    unmarked.mkdir()
    (unmarked / "2026-05-13.db").write_bytes(b"unproven")
    future = tmp_path / "brainlayer-backup-future"
    future.mkdir()
    (future / ".backup-date").write_text("2026-05-15", encoding="ascii")
    (future / "2026-05-15.db").write_bytes(b"future")
    verified = tmp_path / "2026-05-14.db.gz"
    verified.write_bytes(b"verified by caller")

    assert backup_daily._prune_owned_temps_after_verified(tmp_path, verified) == [abandoned.name]
    assert not abandoned.exists()
    assert unmarked.exists()
    assert future.exists()


def test_create_snapshot_rejects_one_raw_byte_despite_important_capacity(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: type("Disk", (), {"free": 1})())
    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda _path: 10**12)
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("writer reached")),
    )

    with pytest.raises(RuntimeError, match="raw_before=1 raw_after=1 important_usage=1000000000000"):
        backup_daily.create_sqlite_backup_artifact(source, tmp_path / "out")


def test_backup_peak_accounts_for_verify_and_prior_raw():
    from brainlayer import backup_daily

    db = 10_000
    gz = 4_000
    margin = backup_daily.MIN_RAW_FREE_BYTES + backup_daily.COPY_CHUNK_BYTES
    assert backup_daily._peak_backup_required_bytes(db, gz, 0, full=False) == 2 * db + gz + margin
    assert backup_daily._peak_backup_required_bytes(db, gz, 0, full=True) == 2 * db + 2 * gz + margin
    assert backup_daily._peak_backup_required_bytes(db, gz, db, full=False) == db + gz + margin
    assert backup_daily._peak_backup_required_bytes(db, gz, db, full=True) == db + 2 * gz + margin


def test_preflight_credits_only_raw_copy_pruned_before_verify(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    db_size = backup_daily._database_logical_size_bytes(source)
    gzip_size = max(1, db_size // 2)
    with_prior = tmp_path / "with-prior"
    without_prior = tmp_path / "without-prior"
    symlink_prior = tmp_path / "symlink-prior"
    with_prior.mkdir()
    without_prior.mkdir()
    symlink_prior.mkdir()
    (with_prior / "2026-05-13.db").write_bytes(b"x" * db_size)
    (symlink_prior / "2026-05-13.db").symlink_to(source)
    for folder in (with_prior, without_prior, symlink_prior):
        (folder / "2026-05-13.db.gz").write_bytes(b"z" * gzip_size)
    budget = backup_daily._peak_backup_required_bytes(db_size, gzip_size, db_size, full=False)
    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: type("Disk", (), {"free": budget})())
    monkeypatch.setattr(
        backup_daily,
        "request_brainbar_vacuum_into",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("writer reached")),
    )

    with pytest.raises(RuntimeError, match="writer reached"):
        backup_daily.create_sqlite_backup_artifact(source, with_prior, date_stamp="2026-05-14")
    with pytest.raises(RuntimeError, match="Insufficient free space"):
        backup_daily.create_sqlite_backup_artifact(source, without_prior, date_stamp="2026-05-14")
    with pytest.raises(RuntimeError, match="Insufficient free space"):
        backup_daily.create_sqlite_backup_artifact(source, symlink_prior, date_stamp="2026-05-14")


def test_decompress_aborts_when_raw_floor_breached(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "source.gz"
    with gzip.open(source, "wb") as handle:
        handle.write(b"x" * 2048)
    readings = iter([backup_daily.MIN_RAW_FREE_BYTES + 1024 * 1024, 1])
    monkeypatch.setattr(
        backup_daily.shutil,
        "disk_usage",
        lambda _path: type("Disk", (), {"free": next(readings)})(),
    )
    with pytest.raises(RuntimeError, match="raw free space floor"):
        backup_daily._decompress_gzip_to(source, tmp_path / "restored.db")


def test_copy_aborts_when_raw_floor_breached_during_write(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    readings = iter([backup_daily.MIN_RAW_FREE_BYTES + backup_daily.COPY_CHUNK_BYTES, 1])
    monkeypatch.setattr(
        backup_daily.shutil,
        "disk_usage",
        lambda _path: type("Disk", (), {"free": next(readings)})(),
    )
    with pytest.raises(RuntimeError, match="raw free space floor"):
        backup_daily._copy_with_raw_floor(io.BytesIO(b"x"), io.BytesIO(), tmp_path)


def test_important_usage_probe_falls_back_on_invalid_or_timed_out_reply(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    monkeypatch.setattr(backup_daily.sys, "platform", "darwin")
    for reply in ("unavailable", "not-a-number"):
        monkeypatch.setattr(
            backup_daily.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(stdout=reply),
        )
        assert backup_daily._important_usage_capacity_bytes(tmp_path) is None

    def timed_out(*args, **kwargs):  # noqa: ARG001
        raise subprocess.TimeoutExpired("osascript", 5)

    monkeypatch.setattr(backup_daily.subprocess, "run", timed_out)
    assert backup_daily._important_usage_capacity_bytes(tmp_path) is None
    monkeypatch.setattr(backup_daily.sys, "platform", "linux")
    assert backup_daily._important_usage_capacity_bytes(tmp_path) is None


def test_backup_capacity_probe_error_falls_back_and_logs_both_readings(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source, chunk_count=2)
    output_dir = tmp_path / "out"
    log_path = tmp_path / "backup.log"
    monkeypatch.setattr(
        backup_daily,
        "_important_usage_capacity_bytes",
        lambda _path: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda _path: type("Disk", (), {"free": 1})())
    with pytest.raises(RuntimeError, match="raw_before=1 raw_after=1 important_usage=None"):
        backup_daily.run_backup(db_path=source, staging_dir=output_dir, log_path=log_path, upload=False)
    result = json.loads(log_path.read_text().splitlines()[-1])
    assert result["raw_free_bytes"] == 1
    assert result["important_usage_free_bytes"] is None
    assert result["required_bytes"] > 1


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


def test_run_backup_refuses_folder_with_other_machine_snapshot_and_records_error(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-09-14.db.gz"
    snapshot.write_bytes(b"backup-bytes")
    log_path = tmp_path / "backup-daily.log"
    upload_calls: list[object] = []

    class FakeArtifact:
        gzip_path = snapshot
        uncompressed_path = None
        sentinel_chunks = 1
        local_retention_deleted: list[str] = []

    class FakeExecute:
        def execute(self):
            return {
                "files": [
                    {
                        "id": "other-snapshot",
                        "name": "2026-09-13.db.gz",
                        "appProperties": {"brainlayer_machine": "m4-other"},
                    }
                ]
            }

    class FakeFiles:
        def list(self, **kwargs):  # noqa: ARG002
            return FakeExecute()

    class FakeService:
        def files(self):
            return FakeFiles()

    monkeypatch.setattr(backup_daily, "create_sqlite_backup_artifact", lambda *args, **kwargs: FakeArtifact())
    monkeypatch.setattr(backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "build_drive_service", lambda *args, **kwargs: FakeService())
    monkeypatch.setattr(backup_daily, "ensure_drive_folder_chain", lambda *args, **kwargs: "folder-id")
    monkeypatch.setattr(
        backup_daily,
        "upload_file_to_drive_raw",
        lambda *args, **kwargs: upload_calls.append((args, kwargs)),
    )

    with pytest.raises(backup_daily.DriveFolderOwnedByOtherMachineError):
        backup_daily.run_backup(
            db_path=tmp_path / "brainlayer.db",
            staging_dir=tmp_path,
            folder_parts=["brainlayer-db-m1"],
            machine_id="m1",
            date_stamp="2026-09-14",
            upload=True,
            log_path=log_path,
        )

    assert upload_calls == []
    receipt = json.loads(log_path.read_text(encoding="utf-8"))
    assert receipt["uploaded"] is False
    assert receipt["error_code"] == "drive_folder_owned_by_other_machine"


def test_legacy_unmarked_snapshot_is_owned_by_canonical_m4():
    from brainlayer import backup_daily

    class FakeExecute:
        def execute(self):
            return {"files": [{"id": "legacy", "name": "2026-09-13.db.gz"}]}

    class FakeFiles:
        def list(self, **kwargs):  # noqa: ARG002
            return FakeExecute()

    class FakeService:
        def files(self):
            return FakeFiles()

    service = FakeService()
    backup_daily.assert_drive_folder_owned_by_machine(
        service,
        folder_id="legacy-folder",
        machine_id=backup_daily.CANONICAL_MACHINE_ID,
    )
    with pytest.raises(backup_daily.DriveFolderOwnedByOtherMachineError):
        backup_daily.assert_drive_folder_owned_by_machine(
            service,
            folder_id="legacy-folder",
            machine_id="m1",
        )


class _DriveResponse:
    def __init__(self, status_code, *, headers=None, payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload or {}
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_raw_drive_upload_sets_machine_app_property(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-09-14.db.gz"
    snapshot.write_bytes(b"backup")
    uploaded_metadata: dict[str, object] = {}

    def fake_post(*args, **kwargs):  # noqa: ARG001
        uploaded_metadata.update(json.loads(kwargs["data"]))
        return _DriveResponse(200, headers={"Location": "https://upload.test/session"})

    class SuccessfulSession:
        def put(self, *args, **kwargs):  # noqa: ARG002
            return _DriveResponse(200, payload={"id": "drive-file-id"})

        def close(self):
            pass

    monkeypatch.setattr(backup_daily.requests, "post", fake_post)
    monkeypatch.setattr(backup_daily.requests, "Session", SuccessfulSession)

    backup_daily.upload_file_to_drive_raw(
        snapshot,
        "folder-id",
        type("Credentials", (), {"token": "test-token"})(),
        machine_id="m1",
    )

    assert uploaded_metadata["appProperties"] == {"brainlayer_machine": "m1"}


def test_backup_receipt_records_drive_folder_and_machine_id(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-09-14.db.gz"
    snapshot.write_bytes(b"backup")

    class FakeArtifact:
        gzip_path = snapshot
        uncompressed_path = None
        sentinel_chunks = 1
        local_retention_deleted: list[str] = []

    monkeypatch.setattr(backup_daily, "create_sqlite_backup_artifact", lambda *args, **kwargs: FakeArtifact())

    result = backup_daily.run_backup(
        db_path=tmp_path / "brainlayer.db",
        staging_dir=tmp_path,
        folder_parts=["Brain Drive", "06_ARCHIVE", "backups", "brainlayer-db-m1"],
        machine_id="m1",
        date_stamp="2026-09-14",
        upload=False,
        log_path=tmp_path / "backup-daily.log",
    )

    assert result["drive_folder"] == "Brain Drive/06_ARCHIVE/backups/brainlayer-db-m1"
    assert result["machine_id"] == "m1"


def test_invalid_machine_id_is_recorded_in_backup_receipt(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    log_path = tmp_path / "backup-daily.log"
    monkeypatch.setenv("BRAINLAYER_MACHINE_ID", "bad machine id")

    with pytest.raises(backup_daily.InvalidMachineIdError):
        backup_daily.run_backup(
            db_path=tmp_path / "brainlayer.db",
            staging_dir=tmp_path,
            upload=False,
            log_path=log_path,
        )

    receipt = json.loads(log_path.read_text(encoding="utf-8"))
    assert receipt["error_code"] == "invalid_machine_id"
    assert receipt["uploaded"] is False


def _stub_backup_for_drive_upload(backup_daily, monkeypatch, snapshot):
    class FakeArtifact:
        gzip_path = snapshot
        uncompressed_path = None
        sentinel_chunks = 1
        local_retention_deleted: list[str] = []

    class Credentials:
        token = "test-token"

    service = object()
    monkeypatch.setattr(backup_daily, "create_sqlite_backup_artifact", lambda *args, **kwargs: FakeArtifact())
    monkeypatch.setattr(backup_daily, "get_drive_credentials", lambda *args, **kwargs: Credentials())
    monkeypatch.setattr(backup_daily, "build_drive_service", lambda *args, **kwargs: service)
    monkeypatch.setattr(backup_daily, "ensure_drive_folder_chain", lambda *args, **kwargs: "folder-id")
    monkeypatch.setattr(backup_daily, "assert_drive_folder_owned_by_machine", lambda *args, **kwargs: None)
    monkeypatch.setattr(backup_daily, "prune_drive_backups", lambda *args, **kwargs: [])
    monkeypatch.setattr(backup_daily, "prune_local_gzip_snapshots", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        backup_daily,
        "verify_sqlite_backup_artifact",
        lambda *args, **kwargs: {"verified": True, "verification_mode": "quick"},
    )
    monkeypatch.setattr(
        backup_daily.requests,
        "post",
        lambda *args, **kwargs: _DriveResponse(200, headers={"Location": "https://upload.test/session"}),
    )
    monkeypatch.setattr(
        backup_daily.requests,
        "put",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("deadline-managed session required")),
    )
    return service


def test_drive_upload_that_blocks_forever_fails_loudly_with_confirmed_bytes(tmp_path, monkeypatch):
    from brainlayer import backup_daily
    from brainlayer.observability_backup import _daily_snapshot

    snapshot = tmp_path / "2026-09-14.db.gz"
    snapshot.write_bytes(b"abcdef")
    log_path = tmp_path / "backup-daily.log"
    _stub_backup_for_drive_upload(backup_daily, monkeypatch, snapshot)
    ranges = []
    never = threading.Event()

    class BlockingSession:
        def put(self, url, *, headers, data, timeout):  # noqa: ARG002
            content_range = headers["Content-Range"]
            ranges.append(content_range)
            if content_range == "bytes */6":
                return _DriveResponse(308, headers={"Range": "bytes=0-2"})
            never.wait()

        def close(self):
            pass

    monkeypatch.setattr(backup_daily.requests, "Session", BlockingSession)
    monkeypatch.setenv("BRAINLAYER_DRIVE_UPLOAD_DEADLINE_FLOOR_SECONDS", "0.02")
    monkeypatch.setenv("BRAINLAYER_DRIVE_UPLOAD_MIN_BYTES_PER_SECOND", "1000000000")
    monkeypatch.setenv("BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(backup_daily, "_sleep", lambda _seconds: None)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="drive_upload_stalled"):
        backup_daily.run_backup(
            db_path=tmp_path / "brainlayer.db",
            staging_dir=tmp_path,
            date_stamp="2026-09-14",
            upload=True,
            log_path=log_path,
        )

    assert time.monotonic() - started < 1
    assert any(value == "bytes */6" for value in ranges)
    logged = json.loads(log_path.read_text(encoding="utf-8"))
    assert logged["error_type"] == "drive_upload_stalled"
    assert logged["error_code"] == "drive_upload_stalled"
    assert logged["bytes_confirmed"] == 3
    assert logged["uploaded"] is False
    assert logged["verified"] is False
    assert _daily_snapshot([{**logged, "backup_log_provenance": "real"}])[1] == "drive_upload_stalled"


def test_drive_upload_resumes_from_confirmed_offset_after_one_stall(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-09-14.db.gz"
    snapshot.write_bytes(b"abcdef")
    service = _stub_backup_for_drive_upload(backup_daily, monkeypatch, snapshot)
    ranges = []
    timeouts = []
    first_chunk = True
    query_attempts = 0
    never = threading.Event()
    verified = []

    class StallOnceSession:
        def put(self, url, *, headers, data, timeout):  # noqa: ARG002
            nonlocal first_chunk, query_attempts
            content_range = headers["Content-Range"]
            ranges.append(content_range)
            timeouts.append(timeout)
            if content_range == "bytes 0-5/6" and first_chunk:
                first_chunk = False
                never.wait()
            if content_range == "bytes */6":
                query_attempts += 1
                if query_attempts == 1:
                    return _DriveResponse(503)
                return _DriveResponse(308, headers={"Range": "bytes=0-2"})
            return _DriveResponse(
                200,
                payload={"id": "drive-file-id", "name": snapshot.name, "size": "6", "md5Checksum": "abc"},
            )

        def close(self):
            pass

    monkeypatch.setattr(backup_daily.requests, "Session", StallOnceSession)
    monkeypatch.setenv("BRAINLAYER_DRIVE_UPLOAD_DEADLINE_FLOOR_SECONDS", "0.02")
    monkeypatch.setenv("BRAINLAYER_DRIVE_UPLOAD_MIN_BYTES_PER_SECOND", "1000000000")
    monkeypatch.setattr(backup_daily, "_sleep", lambda _seconds: None)
    monkeypatch.setattr(
        backup_daily,
        "verify_drive_upload",
        lambda seen_service, **kwargs: verified.append((seen_service, kwargs)),
    )

    result = backup_daily.run_backup(
        db_path=tmp_path / "brainlayer.db",
        staging_dir=tmp_path,
        date_stamp="2026-09-14",
        upload=True,
        remove_local_after_upload=False,
        log_path=tmp_path / "backup-daily.log",
    )

    assert ranges == ["bytes 0-5/6", "bytes */6", "bytes */6", "bytes 3-5/6"]
    assert timeouts == pytest.approx([5.02] * 4)
    assert verified == [
        (
            service,
            {
                "file_id": "drive-file-id",
                "expected_name": snapshot.name,
                "expected_size": 6,
                "expected_machine_id": "test-machine",
            },
        )
    ]
    assert result["uploaded"] is True
    assert result["verified"] is True


def test_drive_upload_missing_range_never_advances_unconfirmed_bytes(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-09-14.db.gz"
    snapshot.write_bytes(b"abcdef")
    _stub_backup_for_drive_upload(backup_daily, monkeypatch, snapshot)
    ranges = []

    class NoConfirmedProgressSession:
        def put(self, url, *, headers, data, timeout):  # noqa: ARG002
            ranges.append(headers["Content-Range"])
            return _DriveResponse(308)

        def close(self):
            pass

    monkeypatch.setattr(backup_daily.requests, "Session", NoConfirmedProgressSession)
    monkeypatch.setenv("BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(backup_daily, "_sleep", lambda _seconds: None)

    with pytest.raises(backup_daily.DriveUploadStalledError) as exc_info:
        backup_daily.upload_file_to_drive_raw(
            snapshot,
            "folder-id",
            type("Credentials", (), {"token": "test-token"})(),
            machine_id="test-machine",
        )

    assert ranges == ["bytes 0-5/6", "bytes */6", "bytes 0-5/6", "bytes */6"]
    assert exc_info.value.bytes_confirmed == 0


class _RetentionExecute:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class _RetentionFiles:
    def __init__(self, count=8):
        self.items = [{"id": f"id-{day}", "name": f"2026-05-{day:02d}.db.gz"} for day in range(1, count + 1)]
        self.trashed: list[tuple[str, dict]] = []
        self.deleted: list[str] = []
        self.list_calls = 0

    def list(self, **kwargs):  # noqa: ARG002
        self.list_calls += 1
        return _RetentionExecute({"files": [item for item in self.items if not item.get("trashed")]})

    def update(self, *, fileId, body, **kwargs):  # noqa: N803, ARG002
        self.trashed.append((fileId, body))
        if body == {"trashed": True}:
            next(item for item in self.items if item["id"] == fileId)["trashed"] = True
        return _RetentionExecute({})

    def delete(self, *, fileId, **kwargs):  # noqa: N803, ARG002
        self.deleted.append(fileId)
        return _RetentionExecute({})


class _RetentionService:
    def __init__(self, count=8):
        self._files = _RetentionFiles(count)

    def files(self):
        return self._files


def _stub_verified_backup_run(backup_daily, monkeypatch, snapshot, service):
    class FakeArtifact:
        gzip_path = snapshot
        uncompressed_path = None
        sentinel_chunks = 1
        local_retention_deleted: list[str] = []

    monkeypatch.setattr(backup_daily, "create_sqlite_backup_artifact", lambda *args, **kwargs: FakeArtifact())
    monkeypatch.setattr(backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "build_drive_service", lambda *args, **kwargs: service)
    monkeypatch.setattr(backup_daily, "ensure_drive_folder_chain", lambda *args, **kwargs: "folder-id")
    monkeypatch.setattr(backup_daily, "assert_drive_folder_owned_by_machine", lambda *args, **kwargs: None)
    monkeypatch.setattr(backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        backup_daily,
        "verify_sqlite_backup_artifact",
        lambda *args, **kwargs: {"verified": True, "verification_mode": "quick"},
    )
    monkeypatch.setattr(
        backup_daily,
        "upload_file_to_drive_raw",
        lambda file_path, folder_id, credentials, *, machine_id: {
            "id": "drive-file-id",
            "name": Path(file_path).name,
            "size": str(Path(file_path).stat().st_size),
        },
    )


def test_run_backup_default_retention_leaves_eight_drive_snapshots_untouched(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-05-30.db.gz"
    snapshot.write_bytes(b"backup-bytes")
    service = _RetentionService(count=8)
    _stub_verified_backup_run(backup_daily, monkeypatch, snapshot, service)
    before = snapshot_non_trashed_drive_ids(service, folder_id="folder-id")

    result = backup_daily.run_backup(
        db_path=tmp_path / "brainlayer.db",
        staging_dir=tmp_path,
        date_stamp="2026-05-30",
        upload=True,
        log_path=tmp_path / "backup-daily.log",
    )
    after = snapshot_non_trashed_drive_ids(service, folder_id="folder-id")

    assert_non_trashed_drive_files_only_grow_or_are_trashed(before, after)
    assert result["drive_retention"] == "disabled"
    assert result["retention_mode"] == "trash"
    assert result["retention_deleted"] == []
    assert service.files().list_calls == 2
    assert service.files().trashed == []
    assert service.files().deleted == []


def test_run_backup_opted_in_retention_trashes_older_drive_snapshots(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-05-30.db.gz"
    snapshot.write_bytes(b"backup-bytes")
    service = _RetentionService(count=8)
    _stub_verified_backup_run(backup_daily, monkeypatch, snapshot, service)
    monkeypatch.setenv("BRAINLAYER_BACKUP_DRIVE_RETENTION", "1")
    before = snapshot_non_trashed_drive_ids(service, folder_id="folder-id")

    result = backup_daily.run_backup(
        db_path=tmp_path / "brainlayer.db",
        staging_dir=tmp_path,
        date_stamp="2026-05-30",
        upload=True,
        log_path=tmp_path / "backup-daily.log",
    )
    after = snapshot_non_trashed_drive_ids(service, folder_id="folder-id")

    assert_non_trashed_drive_files_only_grow_or_are_trashed(before, after, trashed_ids={"id-1"})
    assert result["drive_retention"] == "enabled"
    assert result["retention_mode"] == "trash"
    assert result["retention_deleted"] == ["2026-05-01.db.gz"]
    assert service.files().trashed == [("id-1", {"trashed": True})]
    assert service.files().deleted == []


def test_run_backup_unverified_upload_skips_drive_and_local_gzip_pruning(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-05-30.db.gz"
    snapshot.write_bytes(b"backup-bytes")
    drive_prune_calls: list[object] = []
    local_prune_calls: list[object] = []

    class FakeArtifact:
        gzip_path = snapshot
        uncompressed_path = None
        sentinel_chunks = 1
        local_retention_deleted: list[str] = []

    monkeypatch.setattr(backup_daily, "create_sqlite_backup_artifact", lambda *args, **kwargs: FakeArtifact())
    monkeypatch.setattr(backup_daily, "get_drive_credentials", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "build_drive_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(backup_daily, "ensure_drive_folder_chain", lambda service, folder_parts: "folder-id")
    monkeypatch.setattr(backup_daily, "assert_drive_folder_owned_by_machine", lambda *args, **kwargs: None)
    monkeypatch.setattr(backup_daily, "verify_drive_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        backup_daily,
        "verify_sqlite_backup_artifact",
        lambda *args, **kwargs: {"verified": False, "verification_mode": "quick"},
    )
    monkeypatch.setattr(
        backup_daily,
        "upload_file_to_drive_raw",
        lambda file_path, folder_id, credentials, *, machine_id: {
            "id": "drive-file-id",
            "name": Path(file_path).name,
            "size": str(Path(file_path).stat().st_size),
        },
    )
    monkeypatch.setattr(
        backup_daily,
        "prune_drive_backups",
        lambda *args, **kwargs: drive_prune_calls.append((args, kwargs)) or [],
    )
    monkeypatch.setattr(
        backup_daily,
        "prune_local_gzip_snapshots",
        lambda *args, **kwargs: local_prune_calls.append((args, kwargs)) or [],
    )

    result = backup_daily.run_backup(
        db_path=tmp_path / "brainlayer.db",
        staging_dir=tmp_path,
        date_stamp="2026-05-30",
        upload=True,
        remove_local_after_upload=True,
        log_path=tmp_path / "backup-daily.log",
    )

    assert result["uploaded"] is True
    assert result["verified"] is False
    assert result["retention_deleted"] == []
    assert result["local_gzip_retention_deleted"] == []
    assert drive_prune_calls == []
    assert local_prune_calls == []
    assert snapshot.exists()


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
    monkeypatch.setattr(backup_daily, "assert_drive_folder_owned_by_machine", lambda *args, **kwargs: None)
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
        lambda file_path, folder_id, credentials, *, machine_id: {
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
    assert logged["attempted_at"].endswith("+00:00")
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
    monkeypatch.setattr(backup_daily, "assert_drive_folder_owned_by_machine", lambda *args, **kwargs: None)
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


def test_prune_drive_backups_trashes_older_snapshots_without_hard_delete():
    from brainlayer.backup_daily import DriveRetentionPolicy, prune_drive_backups

    class FakeExecute:
        def __init__(self, value):
            self.value = value

        def execute(self):
            return self.value

    class FakeFiles:
        def __init__(self):
            self.deleted: list[str] = []
            self.trashed: list[tuple[str, dict]] = []
            self.files = [{"id": f"id-{day}", "name": f"2026-05-{day:02d}.db.gz"} for day in range(1, 10)]

        def list(self, **kwargs):  # noqa: ARG002
            query = kwargs["q"]
            if "mimeType = 'application/vnd.google-apps.folder'" in query:
                return FakeExecute({"files": [{"id": "folder-id", "name": "brainlayer-db"}]})
            return FakeExecute({"files": self.files})

        def delete(self, fileId, **kwargs):  # noqa: N803, ARG002
            self.deleted.append(fileId)
            return FakeExecute({})

        def update(self, *, fileId, body, **kwargs):  # noqa: N803, ARG002
            self.trashed.append((fileId, body))
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
    assert service.files().trashed == [
        ("id-5", {"trashed": True}),
        ("id-4", {"trashed": True}),
        ("id-3", {"trashed": True}),
        ("id-2", {"trashed": True}),
        ("id-1", {"trashed": True}),
    ]
    assert service.files().deleted == []


def test_weekly_shared_drive_pool_uses_the_nonshrinking_drive_cap():
    from brainlayer import backup_daily

    class FakeExecute:
        def __init__(self, value):
            self.value = value

        def execute(self):
            return self.value

    class FakeFiles:
        def __init__(self):
            self.deleted: list[str] = []
            self.trashed: list[tuple[str, dict]] = []
            self.files = [{"id": f"id-{day}", "name": f"2026-05-{day:02d}.db.gz"} for day in range(1, 9)]

        def list(self, **kwargs):  # noqa: ARG002
            query = kwargs["q"]
            if "mimeType = 'application/vnd.google-apps.folder'" in query:
                return FakeExecute({"files": [{"id": "folder-id", "name": "brainlayer-db"}]})
            return FakeExecute({"files": self.files})

        def delete(self, fileId, **kwargs):  # noqa: N803, ARG002
            self.deleted.append(fileId)
            return FakeExecute({})

        def update(self, *, fileId, body, **kwargs):  # noqa: N803, ARG002
            self.trashed.append((fileId, body))
            return FakeExecute({})

    class FakeService:
        def __init__(self):
            self._files = FakeFiles()

        def files(self):
            return self._files

    service = FakeService()

    deleted = backup_daily.prune_drive_backups(
        service,
        folder_parts=["brainlayer-db"],
        retention_policy=backup_daily.WEEKLY_RETENTION,
    )

    assert deleted == ["2026-05-01.db.gz"]
    assert service.files().trashed == [("id-1", {"trashed": True})]
    assert service.files().deleted == []


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
    assert "unset PYTHONPATH" in wrapper
    assert "export PYTHONPATH" not in wrapper
    assert "${BRAINLAYER_PYTHON:?" in wrapper
    assert "BRAINLAYER_PYTHON:-python3" not in wrapper
    assert "<string>com.brainlayer.backup-daily</string>" in plist
    assert "<integer>3</integer>" in plist
    assert "<integer>17</integer>" in plist
    assert "<key>KeepAlive</key>" not in plist
    assert "<key>ExitTimeOut</key>" in plist
    assert "<integer>300</integer>" in plist
    assert "BRAINLAYER_BACKUP_CLIENT_TIMEOUT_SECONDS:=0" in wrapper
    assert "BRAINLAYER_BACKUP_TIMEOUT_SECONDS:=28800" in wrapper
    assert "BRAINLAYER_BACKUP_SQLITE_CHECK_TIMEOUT_SECONDS:=0" in wrapper
    assert "BRAINLAYER_BACKUP_ATTEMPT_MAX_AGE_SECONDS:=86400" in wrapper
    assert "BRAINLAYER_BACKUP_LOG_PROVENANCE:=real" in wrapper


def test_main_enforces_configured_backup_timeout(monkeypatch, capsys):
    from brainlayer import backup_daily

    def slow_backup(**kwargs):  # noqa: ARG001
        time.sleep(5)

    monkeypatch.setenv("BRAINLAYER_BACKUP_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv(backup_daily.BACKUP_SUPERVISED_CHILD_ENV, "1")
    monkeypatch.setattr(backup_daily, "run_backup", slow_backup)

    assert backup_daily.main() == 124
    assert "brainlayer backup timed out after 1s" in capsys.readouterr().out


def test_backup_supervisor_enforces_timeout_outside_python_signal_delivery(tmp_path, monkeypatch, capsys):
    from brainlayer import backup_daily

    log_path = tmp_path / "backup.log"
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text(
        "import signal\n"
        "import time\n"
        "signal.signal(signal.SIGALRM, lambda signum, frame: None)\n"
        "while True:\n"
        "    time.sleep(60)\n"
    )
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PATH", str(log_path))
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PROVENANCE", "pytest")

    started = time.monotonic()
    exit_code = backup_daily._supervise_backup_process(1, command=[sys.executable, str(sleeper)])
    elapsed = time.monotonic() - started

    assert exit_code == 124
    assert elapsed < 13  # 1s deadline plus the parent's 10s graceful cleanup window.
    receipt = json.loads(log_path.read_text().strip())
    assert receipt["backup_log_provenance"] == "pytest"
    assert receipt["verified"] is False
    assert receipt["uploaded"] is False
    assert receipt["error_type"] == "BackupTimeoutError"
    assert datetime.fromisoformat(receipt["attempted_at"]).tzinfo is not None
    assert receipt["timeout_seconds"] == 1
    assert "timed out after 1s" in capsys.readouterr().out


def test_backup_supervisor_sigterm_unwinds_snapshot_temp_dir(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    child = tmp_path / "child.py"
    child.write_text(
        "import signal\n"
        "import tempfile\n"
        "from pathlib import Path\n"
        "from brainlayer import backup_daily\n"
        f"Path({str(output_dir / 'imported-from')!r}).write_text(str(Path(backup_daily.__file__).resolve()))\n"
        "def run_backup(**kwargs):\n"
        f"    with tempfile.TemporaryDirectory(prefix='brainlayer-backup-', dir={str(output_dir)!r}) as name:\n"
        "        (Path(name) / 'snapshot.db').write_bytes(b'private')\n"
        f"        Path({str(output_dir / 'ready')!r}).touch()\n"
        "        while True:\n"
        "            signal.pause()\n"
        "backup_daily.run_backup = run_backup\n"
        "raise SystemExit(backup_daily._run_backup_process(30))\n"
    )
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PATH", str(tmp_path / "backup.log"))
    worktree_src = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(worktree_src), os.environ.get("PYTHONPATH")))))

    assert backup_daily._supervise_backup_process(2, command=[sys.executable, str(child)]) == 124
    assert (output_dir / "ready").exists()
    assert (output_dir / "imported-from").read_text() == str(Path(backup_daily.__file__).resolve())
    assert list(output_dir.glob("brainlayer-backup-*")) == []


def test_backup_child_distinguishes_sigterm_stop_from_timeout(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    def stopped_backup(**kwargs):  # noqa: ARG001
        with tempfile.TemporaryDirectory(prefix="brainlayer-backup-", dir=tmp_path):
            os.kill(os.getpid(), signal.SIGTERM)

    log_path = tmp_path / "backup.log"
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PATH", str(log_path))
    monkeypatch.setattr(backup_daily, "run_backup", stopped_backup)

    assert backup_daily._run_backup_process(30) == 143
    assert list(tmp_path.glob("brainlayer-backup-*")) == []
    receipt = json.loads(log_path.read_text().splitlines()[-1])
    assert receipt["error_type"] == "BackupStoppedError"
    assert receipt["stop_signal"] == "SIGTERM"
    assert datetime.fromisoformat(receipt["attempted_at"]).tzinfo is not None


def test_supervisor_forwards_launchd_stop_without_timeout_receipt(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    child = tmp_path / "child.py"
    child.write_text(
        "import signal\nimport tempfile\nfrom pathlib import Path\nfrom brainlayer import backup_daily\n"
        "def run_backup(**kwargs):\n"
        f"    with tempfile.TemporaryDirectory(prefix='brainlayer-backup-', dir={str(output_dir)!r}):\n"
        f"        Path({str(output_dir / 'ready')!r}).touch()\n"
        "        while True: signal.pause()\n"
        "backup_daily.run_backup = run_backup\n"
        "raise SystemExit(backup_daily._run_backup_process(30))\n"
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import sys\nfrom brainlayer import backup_daily\n"
        "raise SystemExit(backup_daily._supervise_backup_process(30, command=[sys.executable, sys.argv[1]]))\n"
    )
    log_path = tmp_path / "backup.log"
    env = os.environ.copy()
    env["BRAINLAYER_BACKUP_LOG_PATH"] = str(log_path)
    env["PYTHONPATH"] = os.pathsep.join((str(Path(__file__).resolve().parents[1] / "src"), env.get("PYTHONPATH", "")))
    process = subprocess.Popen([sys.executable, str(parent), str(child)], env=env)
    try:
        deadline = time.monotonic() + 5
        while not (output_dir / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (output_dir / "ready").exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 143
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    deadline = time.monotonic() + 5
    while list(output_dir.glob("brainlayer-backup-*")) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert list(output_dir.glob("brainlayer-backup-*")) == []
    receipts = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert receipts[-1]["error_type"] == "BackupStoppedError"
    assert all(receipt.get("error_type") != "BackupTimeoutError" for receipt in receipts)


def test_stop_during_purge_never_reaches_writer(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    source = tmp_path / "brainlayer.db"
    _create_source_db(source)
    required = backup_daily._peak_backup_required_bytes(2 * 1024**3, 2 * 1024**3, 0, full=False)
    monkeypatch.setattr(backup_daily, "_brainbar_writer_started_at", lambda *_args: None)
    monkeypatch.setattr(backup_daily, "_database_logical_size_bytes", lambda *_args: 2 * 1024**3)
    monkeypatch.setattr(backup_daily.shutil, "disk_usage", lambda *_args: SimpleNamespace(free=required - 1))
    monkeypatch.setattr(backup_daily, "_important_usage_capacity_bytes", lambda *_args: required + 1)
    monkeypatch.setattr(backup_daily, "_request_macos_purge", lambda *_args: os.kill(os.getpid(), signal.SIGTERM))
    monkeypatch.setattr(
        backup_daily, "request_brainbar_vacuum_into", lambda *_args, **_kwargs: pytest.fail("writer reached")
    )
    previous = signal.signal(signal.SIGTERM, backup_daily._raise_backup_stopped)
    try:
        with pytest.raises(backup_daily.BackupStoppedError):
            backup_daily.create_sqlite_backup_artifact(source, tmp_path / "out", date_stamp="2026-05-14")
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_stop_during_vacuum_never_retries(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    requests = []

    def stop_on_request(*_args, **_kwargs):
        requests.append(1)
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(backup_daily, "_send_brainbar_json_request", stop_on_request)
    previous = signal.signal(signal.SIGTERM, backup_daily._raise_backup_stopped)
    try:
        with pytest.raises(backup_daily.BackupStoppedError):
            backup_daily.request_brainbar_vacuum_into(tmp_path / "snapshot.db", max_attempts=3, retry_backoff_seconds=0)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert len(requests) == 1


def test_lock_refusal_does_not_replace_running_backup_observability():
    from brainlayer.observability_backup import _daily_snapshot

    success = {
        "backup_log_provenance": "real",
        "attempted_at": "2026-09-22T02:00:00+00:00",
        "verified": True,
        "uploaded": True,
        "drive_file": "verified.db.gz",
    }
    refusal = {
        "backup_log_provenance": "real",
        "attempted_at": "2026-09-23T02:00:00+00:00",
        "error_type": "BackupAlreadyRunningError",
        "error": "backup already running",
    }
    snapshot, error_type, all_errors = _daily_snapshot([success, refusal])
    assert snapshot is not None and snapshot["destination"] == "verified.db.gz"
    assert error_type is None
    assert all_errors is False


def test_real_backup_stop_writes_one_complete_receipt(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    db_path = tmp_path / "brainlayer.db"
    _create_source_db(db_path)
    log_path = tmp_path / "backup.log"
    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_BACKUP_STAGING_DIR", str(tmp_path / "staging"))
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PATH", str(log_path))
    monkeypatch.setattr(
        backup_daily,
        "create_sqlite_backup_artifact",
        lambda *_args, **_kwargs: os.kill(os.getpid(), signal.SIGTERM),
    )

    assert backup_daily._run_backup_process(30) == 143
    receipts = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(receipts) == 1
    assert receipts[0]["error_type"] == "BackupStoppedError"
    assert receipts[0]["stop_signal"] == "SIGTERM"
    assert datetime.fromisoformat(receipts[0]["attempted_at"]).tzinfo is not None


# --- 2026-09-23 incident: a finished upload must never be marked failed ------------------------
# Evidence: backup-daily.out.log L38011-38094 (1.5.38). The last chunk returned 200; the run then
# died in verify_drive_upload on a ConnectionResetError from the httplib2 keep-alive socket that
# the Drive service had held idle through the ~3 h upload.


class _ScriptedUploadSession:
    """Plays one scripted outcome per PUT; an exception outcome is raised, not returned."""

    def __init__(self, script):
        self.script = script
        self.ranges: list[str] = []

    def put(self, url, *, headers, data, timeout):  # noqa: ARG002
        content_range = headers["Content-Range"]
        self.ranges.append(content_range)
        outcome = self.script(content_range, len(self.ranges))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self):
        pass


def _run_scripted_upload(backup_daily, monkeypatch, snapshot, script, *, chunk_size=2):
    session = _ScriptedUploadSession(script)
    posts = []

    def fake_post(*args, **kwargs):  # noqa: ARG001
        posts.append(kwargs["data"])
        return _DriveResponse(200, headers={"Location": "https://upload.test/session"})

    monkeypatch.setattr(backup_daily.requests, "post", fake_post)
    monkeypatch.setattr(backup_daily.requests, "Session", lambda: session)
    monkeypatch.setattr(backup_daily, "_sleep", lambda _seconds: None)
    result = backup_daily.upload_file_to_drive_raw(
        snapshot,
        "folder-id",
        type("Credentials", (), {"token": "test-token"})(),
        machine_id="m1",
        chunk_size=chunk_size,
    )
    return result, session.ranges, posts


def test_drive_upload_reset_on_middle_chunk_probes_and_resumes_from_confirmed_offset(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-09-14.db.gz"
    snapshot.write_bytes(b"abcdef")
    reset_once = [True]

    def script(content_range, _call):
        if content_range == "bytes 0-1/6":
            return _DriveResponse(308, headers={"Range": "bytes=0-1"})
        if content_range == "bytes 2-3/6" and reset_once[0]:
            reset_once[0] = False
            return backup_daily.requests.ConnectionError(ConnectionResetError(54, "Connection reset by peer"))
        if content_range == "bytes */6":
            # Drive kept the chunk whose response was lost.
            return _DriveResponse(308, headers={"Range": "bytes=0-3"})
        return _DriveResponse(200, payload={"id": "drive-file-id", "size": "6"})

    result, ranges, posts = _run_scripted_upload(backup_daily, monkeypatch, snapshot, script)

    assert ranges == ["bytes 0-1/6", "bytes 2-3/6", "bytes */6", "bytes 4-5/6"]
    assert result["id"] == "drive-file-id"
    assert len(posts) == 1


def test_drive_upload_reset_on_final_chunk_is_success_when_probe_reports_complete(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-09-14.db.gz"
    snapshot.write_bytes(b"abcdef")
    completed = {"id": "drive-file-id", "name": snapshot.name, "size": "6", "md5Checksum": "abc"}

    def script(content_range, _call):
        if content_range == "bytes 0-3/6":
            return _DriveResponse(308, headers={"Range": "bytes=0-3"})
        if content_range == "bytes 4-5/6":
            # The raw socket error the 09-23 run raised, not a requests wrapper.
            return ConnectionResetError(54, "Connection reset by peer")
        if content_range == "bytes */6":
            return _DriveResponse(200, payload=completed)
        raise AssertionError(f"unexpected PUT {content_range}")

    result, ranges, posts = _run_scripted_upload(backup_daily, monkeypatch, snapshot, script, chunk_size=4)

    assert result == completed
    assert ranges == ["bytes 0-3/6", "bytes 4-5/6", "bytes */6"]
    assert len(posts) == 1, "a completed session must never be re-created as a second Drive file"


def test_drive_upload_reset_on_final_chunk_finishes_last_byte_when_probe_returns_308(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-09-14.db.gz"
    snapshot.write_bytes(b"abcdef")

    def script(content_range, _call):
        if content_range == "bytes 0-3/6":
            return _DriveResponse(308, headers={"Range": "bytes=0-3"})
        if content_range == "bytes 4-5/6":
            return backup_daily.requests.exceptions.SSLError("EOF occurred in violation of protocol")
        if content_range == "bytes */6":
            return _DriveResponse(308, headers={"Range": "bytes=0-4"})
        if content_range == "bytes 5-5/6":
            return _DriveResponse(200, payload={"id": "drive-file-id", "size": "6"})
        raise AssertionError(f"unexpected PUT {content_range}")

    result, ranges, posts = _run_scripted_upload(backup_daily, monkeypatch, snapshot, script, chunk_size=4)

    assert ranges == ["bytes 0-3/6", "bytes 4-5/6", "bytes */6", "bytes 5-5/6"]
    assert result["id"] == "drive-file-id"
    assert len(posts) == 1


def test_drive_upload_persistent_resets_fail_bounded_with_confirmed_bytes(tmp_path, monkeypatch):
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-09-14.db.gz"
    snapshot.write_bytes(b"abcdef")
    monkeypatch.setenv("BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS", "2")

    def script(content_range, _call):
        if content_range == "bytes 0-1/6":
            return _DriveResponse(308, headers={"Range": "bytes=0-1"})
        if content_range == "bytes */6":
            return _DriveResponse(308, headers={"Range": "bytes=0-1"})
        return ConnectionResetError(54, "Connection reset by peer")

    with pytest.raises(backup_daily.DriveUploadStalledError) as exc_info:
        _run_scripted_upload(backup_daily, monkeypatch, snapshot, script)

    assert exc_info.value.bytes_confirmed == 2
    assert "confirmed 2/6 bytes after 2 stalled attempts" in str(exc_info.value)


class _MetadataRequest:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = 0

    def execute(self):
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _MetadataService:
    def __init__(self, request):
        self.request = request

    def files(self):
        return self

    def get(self, **kwargs):  # noqa: ARG002
        return self.request


_GOOD_METADATA = {
    "id": "drive-file-id",
    "name": "2026-09-14.db.gz",
    "size": "6",
    "trashed": False,
    "appProperties": {"brainlayer_machine": "m1"},
}


def test_verify_drive_upload_retries_a_transient_reset(monkeypatch):
    from brainlayer import backup_daily

    request = _MetadataRequest([ConnectionResetError(54, "Connection reset by peer"), _GOOD_METADATA])
    sleeps = []
    monkeypatch.setattr(backup_daily, "_sleep", sleeps.append)

    backup_daily.verify_drive_upload(
        _MetadataService(request),
        file_id="drive-file-id",
        expected_name="2026-09-14.db.gz",
        expected_size=6,
        expected_machine_id="m1",
    )

    assert request.calls == 2
    assert len(sleeps) == 1


def test_verify_drive_upload_persistent_resets_fail_bounded(monkeypatch):
    from brainlayer import backup_daily

    request = _MetadataRequest([ConnectionResetError(54, "Connection reset by peer")])
    monkeypatch.setenv("BRAINLAYER_DRIVE_UPLOAD_STALL_MAX_ATTEMPTS", "3")
    monkeypatch.setattr(backup_daily, "_sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match=r"Drive upload verification failed after 3 attempts"):
        backup_daily.verify_drive_upload(
            _MetadataService(request),
            file_id="drive-file-id",
            expected_name="2026-09-14.db.gz",
            expected_size=6,
            expected_machine_id="m1",
        )

    assert request.calls == 3


def test_verify_drive_upload_does_not_retry_a_mismatch(monkeypatch):
    from brainlayer import backup_daily

    request = _MetadataRequest([{**_GOOD_METADATA, "size": "5"}])
    monkeypatch.setattr(backup_daily, "_sleep", lambda _seconds: pytest.fail("a mismatch is not transient"))

    with pytest.raises(RuntimeError, match="size mismatch"):
        backup_daily.verify_drive_upload(
            _MetadataService(request),
            file_id="drive-file-id",
            expected_name="2026-09-14.db.gz",
            expected_size=6,
            expected_machine_id="m1",
        )

    assert request.calls == 1


def test_run_backup_verifies_on_a_fresh_drive_service_after_a_long_upload(tmp_path, monkeypatch):
    """Service built, ~3 h upload, first call after it resets: the finished upload stays a success."""
    from brainlayer import backup_daily

    snapshot = tmp_path / "2026-09-14.db.gz"
    snapshot.write_bytes(b"abcdef")
    _stub_backup_for_drive_upload(backup_daily, monkeypatch, snapshot)
    upload_finished = [False]

    class StaleAfterUploadRequest:
        def execute(self):
            if upload_finished[0]:
                raise ConnectionResetError(54, "Connection reset by peer")
            return {}

    class PreUploadService:
        """Its keep-alive socket dies during the upload; every later call resets."""

        def files(self):
            return self

        def get(self, **kwargs):  # noqa: ARG002
            return StaleAfterUploadRequest()

    fresh_service = _MetadataService(
        _MetadataRequest([{**_GOOD_METADATA, "appProperties": {"brainlayer_machine": "test-machine"}}])
    )
    services = [PreUploadService(), fresh_service]
    monkeypatch.setattr(backup_daily, "build_drive_service", lambda *args, **kwargs: services.pop(0))
    verify_services = []
    monkeypatch.setattr(
        backup_daily,
        "verify_sqlite_backup_artifact",
        lambda *args, **kwargs: verify_services.append(kwargs["service"]) or {"verified": True},
    )

    class FinishedSession:
        def put(self, url, *, headers, data, timeout):  # noqa: ARG002
            upload_finished[0] = True
            return _DriveResponse(200, payload={"id": "drive-file-id", "size": "6"})

        def close(self):
            pass

    monkeypatch.setattr(backup_daily.requests, "Session", FinishedSession)
    monkeypatch.setattr(backup_daily, "_sleep", lambda _seconds: None)

    result = backup_daily.run_backup(
        db_path=tmp_path / "brainlayer.db",
        staging_dir=tmp_path,
        date_stamp="2026-09-14",
        upload=True,
        remove_local_after_upload=False,
        log_path=tmp_path / "backup-daily.log",
    )

    assert result["uploaded"] is True
    assert result["verified"] is True
    assert "error_type" not in result
    assert services == [], "post-upload calls must run on a Drive service built after the upload"
    assert verify_services == [fresh_service]
    assert fresh_service.request.calls == 1


def test_transient_retry_never_swallows_the_whole_run_deadline(tmp_path, monkeypatch):
    """BackupTimeoutError subclasses TimeoutError; the SIGALRM run cap must still stop the run."""
    from brainlayer import backup_daily

    request = _MetadataRequest([backup_daily.BackupTimeoutError("backup timed out")])
    monkeypatch.setattr(backup_daily, "_sleep", lambda _seconds: pytest.fail("the run deadline was retried"))
    with pytest.raises(backup_daily.BackupTimeoutError):
        backup_daily.verify_drive_upload(
            _MetadataService(request),
            file_id="drive-file-id",
            expected_name="2026-09-14.db.gz",
            expected_size=6,
            expected_machine_id="m1",
        )
    assert request.calls == 1

    snapshot = tmp_path / "2026-09-14.db.gz"
    snapshot.write_bytes(b"abcdef")

    def script(content_range, _call):
        if content_range == "bytes */6":
            pytest.fail("the run deadline was treated as a lost response")
        return backup_daily.BackupTimeoutError("backup timed out")

    with pytest.raises(backup_daily.BackupTimeoutError):
        _run_scripted_upload(backup_daily, monkeypatch, snapshot, script)
