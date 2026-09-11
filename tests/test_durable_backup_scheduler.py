from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from brainlayer import backup_run_once, durable_backup_scheduler as scheduler


def _config(path: Path, *, source: Path | None, destination: Path, enabled: bool = True) -> None:
    destination_stat = destination.stat()
    values = [
        f"BRAINLAYER_BACKUP_ENABLED={'1' if enabled else '0'}",
        f"BRAINLAYER_BACKUP_DESTINATION='{destination}'",
        f"BRAINLAYER_BACKUP_DESTINATION_ST_DEV={destination_stat.st_dev}",
        f"BRAINLAYER_BACKUP_DESTINATION_ST_INO={destination_stat.st_ino}",
        "BRAINLAYER_BACKUP_TRANSCRIPT_ROOT_SPECS='[]'",
    ]
    if source is not None:
        source_stat = source.stat()
        binding = {"version": 1, "path": str(source), "st_dev": source_stat.st_dev, "st_ino": source_stat.st_ino}
        values.append("BRAINLAYER_BACKUP_SCHEDULED_OFFLINE_SOURCE_BINDING='" + json.dumps(binding) + "'")
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def test_missing_saved_source_refuses_before_any_backup_request(tmp_path, monkeypatch):
    destination = tmp_path / "destination"
    destination.mkdir()
    env = tmp_path / "brainlayer.env"
    _config(env, source=None, destination=destination)
    monkeypatch.setattr(scheduler.backup_run_once, "run_backup_once", lambda **_: (_ for _ in ()).throw(AssertionError("called")))

    result = scheduler.run_scheduled_backup(config_path=env, state_path=tmp_path / "state.json")

    assert result == {"version": 1, "status": "refused", "reason": "scheduled offline source is not configured"}


def test_stale_saved_source_refuses_before_any_backup_request(tmp_path, monkeypatch):
    destination = tmp_path / "destination"
    destination.mkdir()
    source = tmp_path / "brainlayer.db"
    source.write_bytes(b"not a database")
    env = tmp_path / "brainlayer.env"
    _config(env, source=source, destination=destination)
    source.unlink()
    source.write_bytes(b"replacement")
    monkeypatch.setattr(scheduler.backup_run_once, "run_backup_once", lambda **_: (_ for _ in ()).throw(AssertionError("called")))

    result = scheduler.run_scheduled_backup(config_path=env, state_path=tmp_path / "state.json")

    assert result == {"version": 1, "status": "refused", "reason": "scheduled offline source changed"}


def test_saved_binding_is_passed_to_offline_helper_and_labeled_in_result(tmp_path, monkeypatch):
    destination = tmp_path / "destination"
    destination.mkdir()
    source = tmp_path / "brainlayer.db"
    connection = sqlite3.connect(source)
    connection.execute("create table example (value text)")
    connection.close()
    env = tmp_path / "brainlayer.env"
    _config(env, source=source, destination=destination)
    source_stat = source.stat()
    observed: dict[str, object] = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"generation": "generation-1"}

    monkeypatch.setattr(scheduler.backup_run_once, "run_backup_once", fake_run)
    monkeypatch.setattr(scheduler, "_roots", lambda _: [])

    result = scheduler.run_scheduled_backup(config_path=env, state_path=tmp_path / "state.json")

    assert "socket_path" not in observed
    assert observed["offline_db_path"] == source
    assert observed["expected_offline_source_identity"] == (source_stat.st_dev, source_stat.st_ino)
    assert result["status"] == "local_verified"
    assert result["saved_source"] == {
        "version": 1, "path": str(source), "st_dev": source_stat.st_dev, "st_ino": source_stat.st_ino,
    }


def test_disabled_schedule_never_requires_or_infers_source(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    env = tmp_path / "brainlayer.env"
    _config(env, source=None, destination=destination, enabled=False)

    result = scheduler.run_scheduled_backup(config_path=env, state_path=tmp_path / "state.json")

    assert result["status"] == "disabled"
    assert result["reason"] == "backup_disabled"


def test_offline_helper_refuses_post_snapshot_source_replacement(tmp_path, monkeypatch):
    source = tmp_path / "brainlayer.db"
    connection = sqlite3.connect(source)
    connection.execute("create table example (value text)")
    connection.close()
    expected = (source.stat().st_dev, source.stat().st_ino)
    calls = 0
    original_identity = backup_run_once._identity

    def identity(path: Path):
        nonlocal calls
        if path == source:
            calls += 1
            return {"st_dev": expected[0], "st_ino": expected[1] if calls == 1 else expected[1] + 1}
        return original_identity(path)

    monkeypatch.setattr(backup_run_once, "_identity", identity)

    with pytest.raises(backup_run_once.BackupRunOnceError, match="changed after snapshot"):
        backup_run_once._offline_snapshot(source, tmp_path / "snapshot.db", expected_source_identity=expected)
