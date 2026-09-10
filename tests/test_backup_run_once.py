import json
import sqlite3
from pathlib import Path

import pytest

from brainlayer import backup_run_once


def _fixture_db(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, content TEXT)")
        db.execute("INSERT INTO chunks(content) VALUES ('synthetic fixture')")


def _root(path: Path) -> list[Path]:
    root = path / "projects"
    root.mkdir()
    (root / "synthetic.jsonl").write_text('{"synthetic":true}\n')
    return [root]


def test_run_once_creates_verified_db_and_jsonl_generation_without_remote_claim(tmp_path):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    result = backup_run_once.run_backup_once(
        destination=destination,
        offline_db_path=db,
        transcript_roots=_root(tmp_path),
        expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
        generation_id="one",
        now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc),
    )
    generation = destination / result["generation"]
    assert (
        result["status"] == "local_verified"
        and result["local_verified"] is True
        and result["remote_verified"] is False
        and result["cleanup_permitted"] is False
    )
    assert (
        (generation / "brainlayer.db").is_file()
        and (generation / "receipt.complete.json").is_file()
        and not (generation / "receipt.incomplete.json").exists()
    )


def test_run_once_requires_existing_destination_and_explicit_source_mode(tmp_path):
    with pytest.raises(backup_run_once.BackupRunOnceError, match="already exist"):
        backup_run_once.run_backup_once(
            destination=tmp_path / "missing", offline_db_path=tmp_path / "missing.db", transcript_roots=[],
            expected_destination_identity=(0, 0),
        )
    destination = tmp_path / "backups"
    destination.mkdir()
    with pytest.raises(backup_run_once.BackupRunOnceError, match="exactly one"):
        backup_run_once.run_backup_once(
            destination=destination,
            socket_path=tmp_path / "socket",
            offline_db_path=tmp_path / "fixture.db",
            transcript_roots=_root(tmp_path),
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
        )


def test_run_once_rejects_corrupt_db_before_complete_receipt_with_real_quick_check(tmp_path):
    db = tmp_path / "corrupt.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, content TEXT)")
        connection.execute("CREATE TABLE junk (payload BLOB)")
        connection.execute("INSERT INTO chunks(content) VALUES ('synthetic fixture')")
        connection.execute("INSERT INTO junk(payload) VALUES (zeroblob(200000))")
    with db.open("r+b") as handle:
        handle.seek(-4096, 2)
        handle.write(b"\xff" * 4096)
    destination = tmp_path / "backups"
    destination.mkdir()
    with pytest.raises(backup_run_once.BackupRunOnceError, match="quick_check failed"):
        backup_run_once.run_backup_once(
            destination=destination,
            offline_db_path=db,
            transcript_roots=_root(tmp_path),
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
            generation_id="real-corrupt",
            now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc),
        )
    generation = next(destination.glob(".brainlayer-backup-*-real-corrupt"))
    receipt = json.loads((generation / "receipt.incomplete.json").read_text())
    assert receipt["status"] == "failed"
    assert not (generation / "receipt.complete.json").exists()


def test_run_once_failure_keeps_incomplete_receipt_and_partial_artifacts(tmp_path, monkeypatch):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    monkeypatch.setattr(
        backup_run_once.backup_daily,
        "_validate_backup_target",
        lambda _: (_ for _ in ()).throw(RuntimeError("synthetic validation failure")),
    )
    with pytest.raises(RuntimeError, match="synthetic validation failure"):
        backup_run_once.run_backup_once(
            destination=destination,
            offline_db_path=db,
            transcript_roots=_root(tmp_path),
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
            generation_id="failed",
            now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc),
        )
    generation = next(destination.glob(".brainlayer-backup-*-failed"))
    receipt = json.loads((generation / "receipt.incomplete.json").read_text())
    assert (
        receipt["status"] == "failed"
        and not (generation / "receipt.complete.json").exists()
        and (generation / "brainlayer.db").is_file()
    )


def test_run_once_rejects_missing_root_and_outside_symlink(tmp_path):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    missing = tmp_path / "missing"
    with pytest.raises(backup_run_once.BackupRunOnceError, match="transcript root"):
        backup_run_once.run_backup_once(
            destination=destination, offline_db_path=db, transcript_roots=[tmp_path / "projects", missing],
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
        )
    root = tmp_path / "projects"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("outside")
    (root / "escape.jsonl").symlink_to(outside)
    with pytest.raises(backup_run_once.BackupRunOnceError, match="symlink"):
        backup_run_once.run_backup_once(
            destination=destination, offline_db_path=db, transcript_roots=[root], generation_id="escape",
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
        )


def test_run_once_binds_destination_identity_and_records_digests(tmp_path):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    roots = _root(tmp_path)
    identity = (destination.stat().st_dev, destination.stat().st_ino)
    result = backup_run_once.run_backup_once(
        destination=destination,
        offline_db_path=db,
        transcript_roots=roots,
        generation_id="identity",
        expected_destination_identity=identity,
    )
    assert result["destination_identity"] == {"st_dev": identity[0], "st_ino": identity[1]}
    assert len(result["db_snapshot_sha256"]) == 64 and len(result["jsonl_bundle_sha256"]) == 64
    with pytest.raises(backup_run_once.BackupRunOnceError, match="identity changed"):
        backup_run_once.run_backup_once(
            destination=destination,
            offline_db_path=db,
            transcript_roots=roots,
            generation_id="mismatch",
            expected_destination_identity=(identity[0], identity[1] + 1),
        )


def test_run_once_preserves_primary_error_when_failure_receipt_fails(tmp_path, monkeypatch):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    original = backup_run_once._atomic_json

    def fail_failed_receipt(path, payload):
        if payload.get("status") == "failed":
            raise OSError("receipt finalizer failed")
        return original(path, payload)

    monkeypatch.setattr(
        backup_run_once.backup_daily,
        "_validate_backup_target",
        lambda _: (_ for _ in ()).throw(RuntimeError("primary failure")),
    )
    monkeypatch.setattr(backup_run_once, "_atomic_json", fail_failed_receipt)
    with pytest.raises(RuntimeError, match="primary failure"):
        backup_run_once.run_backup_once(
            destination=destination, offline_db_path=db, transcript_roots=_root(tmp_path), generation_id="primary",
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
        )


def test_run_once_rejects_source_replacement_during_bundle(tmp_path, monkeypatch):
    db = tmp_path / "fixture.db"
    _fixture_db(db)
    destination = tmp_path / "backups"
    destination.mkdir()
    root = tmp_path / "projects"
    root.mkdir()
    source = root / "synthetic.jsonl"
    source.write_text('{"safe":true}\n')
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"outside":true}\n')
    original = backup_run_once.jsonl_backup.create_jsonl_bundle_with_digests

    def swap_then_bundle(candidates, staging_dir, **kwargs):
        source.unlink()
        source.symlink_to(outside)
        return original(candidates, staging_dir, **kwargs)

    monkeypatch.setattr(backup_run_once.jsonl_backup, "create_jsonl_bundle_with_digests", swap_then_bundle)
    with pytest.raises(backup_run_once.BackupRunOnceError, match="symlink|no-follow|transcript candidate"):
        backup_run_once.run_backup_once(
            destination=destination,
            offline_db_path=db,
            transcript_roots=[root],
            expected_destination_identity=(destination.stat().st_dev, destination.stat().st_ino),
            generation_id="source-swap",
        )
    generation = next(destination.glob(".brainlayer-backup-*-source-swap"))
    receipt = json.loads((generation / "receipt.incomplete.json").read_text())
    assert receipt["status"] == "failed"
