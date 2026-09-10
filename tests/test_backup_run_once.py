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
            destination=tmp_path / "missing", offline_db_path=tmp_path / "missing.db", transcript_roots=[]
        )
    destination = tmp_path / "backups"
    destination.mkdir()
    with pytest.raises(backup_run_once.BackupRunOnceError, match="exactly one"):
        backup_run_once.run_backup_once(
            destination=destination,
            socket_path=tmp_path / "socket",
            offline_db_path=tmp_path / "fixture.db",
            transcript_roots=_root(tmp_path),
        )


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
