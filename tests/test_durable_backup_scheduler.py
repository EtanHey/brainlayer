from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

import pytest

from brainlayer import backup_run_once
from brainlayer import durable_backup_scheduler as scheduler


def _config(
    path: Path,
    *,
    source: Path | None,
    destination: Path,
    enabled: bool = True,
    source_open_generation: str = "daemon-open-test",
) -> None:
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
        binding = {
            "version": 1,
            "path": str(source),
            "st_dev": source_stat.st_dev,
            "st_ino": source_stat.st_ino,
            "source_open_generation": source_open_generation,
        }
        values.append("BRAINLAYER_BACKUP_SCHEDULED_OFFLINE_SOURCE_BINDING='" + json.dumps(binding) + "'")
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def test_missing_saved_source_refuses_before_any_backup_request(tmp_path, monkeypatch):
    destination = tmp_path / "destination"
    destination.mkdir()
    env = tmp_path / "brainlayer.env"
    _config(env, source=None, destination=destination)
    monkeypatch.setattr(
        scheduler.backup_run_once, "run_backup_once", lambda **_: (_ for _ in ()).throw(AssertionError("called"))
    )

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
    monkeypatch.setattr(
        scheduler.backup_run_once, "run_backup_once", lambda **_: (_ for _ in ()).throw(AssertionError("called"))
    )

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
    assert observed["offline_source_open_generation"] == "daemon-open-test"
    assert observed["expected_offline_source_identity"] == (source_stat.st_dev, source_stat.st_ino)
    assert result["status"] == "local_verified"
    assert result["saved_source"] == {
        "version": 1,
        "path": str(source),
        "st_dev": source_stat.st_dev,
        "st_ino": source_stat.st_ino,
        "source_open_generation": "daemon-open-test",
    }


def test_saved_source_open_generation_is_required_and_preserved(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    source = tmp_path / "brainlayer.db"
    source.write_bytes(b"source")
    env = tmp_path / "brainlayer.env"
    _config(env, source=source, destination=destination, source_open_generation="  ")

    with pytest.raises(scheduler.ScheduledBackupRefusal, match="invalid source generation"):
        scheduler._load_binding(env)

    _config(env, source=source, destination=destination, source_open_generation="daemon-open-42")
    assert scheduler._load_binding(env)["source_open_generation"] == "daemon-open-42"


def test_failed_schedule_persists_bounded_backoff(tmp_path, monkeypatch):
    destination = tmp_path / "destination"
    destination.mkdir()
    source = tmp_path / "brainlayer.db"
    source.write_bytes(b"source")
    env = tmp_path / "brainlayer.env"
    _config(env, source=source, destination=destination)
    monkeypatch.setattr(scheduler, "_roots", lambda _: [])
    monkeypatch.setattr(
        scheduler.backup_run_once, "run_backup_once", lambda **_: (_ for _ in ()).throw(RuntimeError("offline failed"))
    )
    t0 = dt.datetime(2026, 9, 11, tzinfo=dt.UTC)

    first = scheduler.run_scheduled_backup(config_path=env, state_path=tmp_path / "state.json", now=t0)
    assert first["status"] == "failed"
    assert first["failure_count"] == 1
    assert first["retry_after"] == "2026-09-11T00:01:00+00:00"
    held = scheduler.run_scheduled_backup(
        config_path=env, state_path=tmp_path / "state.json", now=t0 + dt.timedelta(seconds=30)
    )
    assert held == {
        "version": 1,
        "status": "not_due",
        "next_due": "2026-09-11T00:01:00+00:00",
        "reason": "failure_backoff",
        "failure_count": 1,
        "retry_after": "2026-09-11T00:01:00+00:00",
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



def test_successful_schedule_scope_change_destination_is_immediately_due(tmp_path, monkeypatch):
    source = tmp_path / "brainlayer.db"
    source.write_bytes(b"source")
    destination_a = tmp_path / "destination-a"
    destination_b = tmp_path / "destination-b"
    destination_a.mkdir()
    destination_b.mkdir()
    env = tmp_path / "brainlayer.env"
    state = tmp_path / "state.json"
    _config(env, source=source, destination=destination_a)
    monkeypatch.setattr(scheduler, "_roots", lambda _: [])
    calls = 0

    def fake_run(**_kwargs):
        nonlocal calls
        calls += 1
        return {"generation": f"generation-{calls}"}

    monkeypatch.setattr(scheduler.backup_run_once, "run_backup_once", fake_run)
    t0 = dt.datetime(2026, 9, 11, tzinfo=dt.UTC)
    first = scheduler.run_scheduled_backup(config_path=env, state_path=state, now=t0)
    assert first["status"] == "local_verified"
    first_scope = json.loads(state.read_text())["schedule_scope"]

    _config(env, source=source, destination=destination_b)
    second = scheduler.run_scheduled_backup(
        config_path=env, state_path=state, now=t0 + dt.timedelta(seconds=30)
    )

    assert second["status"] == "local_verified"
    assert calls == 2
    assert json.loads(state.read_text())["schedule_scope"] != first_scope


def test_successful_schedule_scope_change_source_is_immediately_due(tmp_path, monkeypatch):
    source_a = tmp_path / "brainlayer-a.db"
    source_b = tmp_path / "brainlayer-b.db"
    source_a.write_bytes(b"source-a")
    source_b.write_bytes(b"source-b")
    destination = tmp_path / "destination"
    destination.mkdir()
    env = tmp_path / "brainlayer.env"
    state = tmp_path / "state.json"
    _config(env, source=source_a, destination=destination)
    monkeypatch.setattr(scheduler, "_roots", lambda _: [])
    calls = 0

    def fake_run(**_kwargs):
        nonlocal calls
        calls += 1
        return {"generation": f"generation-{calls}"}

    monkeypatch.setattr(scheduler.backup_run_once, "run_backup_once", fake_run)
    t0 = dt.datetime(2026, 9, 11, tzinfo=dt.UTC)
    assert scheduler.run_scheduled_backup(config_path=env, state_path=state, now=t0)["status"] == "local_verified"
    _config(env, source=source_b, destination=destination)

    second = scheduler.run_scheduled_backup(
        config_path=env, state_path=state, now=t0 + dt.timedelta(seconds=30)
    )

    assert second["status"] == "local_verified"
    assert calls == 2


def test_unchanged_schedule_scope_honors_next_due(tmp_path, monkeypatch):
    source = tmp_path / "brainlayer.db"
    source.write_bytes(b"source")
    destination = tmp_path / "destination"
    destination.mkdir()
    env = tmp_path / "brainlayer.env"
    state = tmp_path / "state.json"
    _config(env, source=source, destination=destination)
    monkeypatch.setattr(scheduler, "_roots", lambda _: [])
    calls = 0

    def fake_run(**_kwargs):
        nonlocal calls
        calls += 1
        return {"generation": f"generation-{calls}"}

    monkeypatch.setattr(scheduler.backup_run_once, "run_backup_once", fake_run)
    t0 = dt.datetime(2026, 9, 11, tzinfo=dt.UTC)
    assert scheduler.run_scheduled_backup(config_path=env, state_path=state, now=t0)["status"] == "local_verified"
    held = scheduler.run_scheduled_backup(
        config_path=env, state_path=state, now=t0 + dt.timedelta(seconds=30)
    )

    assert held == {
        "version": 1,
        "status": "not_due",
        "next_due": "2026-09-12T00:00:00+00:00",
    }
    assert calls == 1


@pytest.mark.parametrize(
    ("raw_state", "reason"),
    [
        ("[]", "scheduler state is malformed"),
        ("null", "scheduler state is malformed"),
        ('{"version": 2}', "scheduler state is unsupported"),
        ('{"version": 1, "status": []}', "scheduler state has invalid status"),
        ('{"version": 1, "next_due": "2026-09-11T00:01:00"}', "scheduler state next_due must be timezone-aware"),
    ],
)
def test_invalid_scheduler_state_refuses_without_overwrite(tmp_path, monkeypatch, raw_state, reason):
    source = tmp_path / "brainlayer.db"
    source.write_bytes(b"source")
    destination = tmp_path / "destination"
    destination.mkdir()
    env = tmp_path / "brainlayer.env"
    _config(env, source=source, destination=destination)
    state = tmp_path / "state.json"
    state.write_text(raw_state, encoding="utf-8")
    monkeypatch.setattr(
        scheduler.backup_run_once,
        "run_backup_once",
        lambda **_: (_ for _ in ()).throw(AssertionError("backup called")),
    )

    result = scheduler.run_scheduled_backup(
        config_path=env,
        state_path=state,
        now=dt.datetime(2026, 9, 11, tzinfo=dt.UTC),
    )

    assert result == {"version": 1, "status": "refused", "reason": reason}
    assert state.read_text(encoding="utf-8") == raw_state



def test_failure_backoff_is_ignored_after_scope_change(tmp_path, monkeypatch):
    source = tmp_path / "brainlayer.db"
    source.write_bytes(b"source")
    destination_a = tmp_path / "destination-a"
    destination_b = tmp_path / "destination-b"
    destination_a.mkdir()
    destination_b.mkdir()
    env = tmp_path / "brainlayer.env"
    state = tmp_path / "state.json"
    _config(env, source=source, destination=destination_a)
    monkeypatch.setattr(scheduler, "_roots", lambda _: [])
    calls = 0

    def failed_run(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("offline failed")

    monkeypatch.setattr(scheduler.backup_run_once, "run_backup_once", failed_run)
    t0 = dt.datetime(2026, 9, 11, tzinfo=dt.UTC)
    first = scheduler.run_scheduled_backup(config_path=env, state_path=state, now=t0)
    assert first["status"] == "failed"
    assert first["failure_count"] == 1
    _config(env, source=source, destination=destination_b)

    second = scheduler.run_scheduled_backup(
        config_path=env, state_path=state, now=t0 + dt.timedelta(seconds=30)
    )

    assert second["status"] == "failed"
    assert second["failure_count"] == 1
    assert calls == 2



def test_invalid_scheduler_state_refuses_when_disabled_without_overwrite(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    env = tmp_path / "brainlayer.env"
    _config(env, source=None, destination=destination, enabled=False)
    state = tmp_path / "state.json"
    state.write_text("[]", encoding="utf-8")

    result = scheduler.run_scheduled_backup(config_path=env, state_path=state)

    assert result == {"version": 1, "status": "refused", "reason": "scheduler state is malformed"}
    assert state.read_text(encoding="utf-8") == "[]"
