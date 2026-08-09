"""Tests for shared WAL checkpoint helpers."""

import plistlib
from pathlib import Path

import pytest


def test_resolve_db_path_uses_existing_cli_path(tmp_path, monkeypatch):
    import brainlayer.wal_checkpoint as wal_checkpoint

    db_path = tmp_path / "brainlayer.db"
    db_path.write_text("")
    monkeypatch.setattr(wal_checkpoint, "get_db_path", lambda: db_path)

    assert wal_checkpoint.resolve_db_path() == str(db_path)


def test_resolve_db_path_returns_none_when_missing(tmp_path, monkeypatch):
    import brainlayer.wal_checkpoint as wal_checkpoint

    monkeypatch.setattr(wal_checkpoint, "get_db_path", lambda: tmp_path / "missing.db")

    assert wal_checkpoint.resolve_db_path() is None


def test_checkpoint_rejects_invalid_mode(tmp_path):
    import brainlayer.wal_checkpoint as wal_checkpoint

    with pytest.raises(ValueError, match="Invalid checkpoint mode"):
        wal_checkpoint.checkpoint(str(tmp_path / "brainlayer.db"), mode="DROP TABLE chunks")


def test_checkpoint_guard_acquisition_times_out_while_another_holder_is_wedged(tmp_path):
    import brainlayer.wal_checkpoint as wal_checkpoint

    db_path = tmp_path / "brainlayer.db"
    with wal_checkpoint.checkpoint_guard(db_path, blocking=False) as acquired:
        assert acquired is True
        with pytest.raises(TimeoutError, match="checkpoint guard acquisition timed out"):
            wal_checkpoint.checkpoint(str(db_path), guard_timeout_seconds=0)


def test_checkpoint_guard_timeout_env_rejects_unbounded_values(monkeypatch):
    import brainlayer.wal_checkpoint as wal_checkpoint

    monkeypatch.setenv("BRAINLAYER_CHECKPOINT_GUARD_TIMEOUT_SECONDS", "inf")

    assert wal_checkpoint._guard_timeout_seconds() == 10.0


def test_retrying_truncate_backs_off_until_checkpoint_is_not_busy(tmp_path, monkeypatch):
    import brainlayer.wal_checkpoint as wal_checkpoint

    db_path = tmp_path / "brainlayer.db"
    db_path.write_bytes(b"")
    results = iter([(1, 20, 10), (1, 20, 20), (0, 0, 0)])
    delays: list[float] = []
    monkeypatch.setattr(wal_checkpoint, "resolve_db_path", lambda: str(db_path))
    monkeypatch.setattr(wal_checkpoint, "checkpoint", lambda _path, _mode: next(results))
    monkeypatch.setattr(wal_checkpoint, "get_wal_size", lambda _path: 0)

    result = wal_checkpoint.run_wal_checkpoint(
        "TRUNCATE",
        retry_busy=True,
        retry_base_seconds=1.0,
        retry_max_seconds=10.0,
        sleep_fn=delays.append,
    )

    assert result["busy"] == 0
    assert result["attempts"] == 3
    assert delays == [1.0, 2.0]


def test_non_truncate_checkpoint_never_retries_busy_result(tmp_path, monkeypatch):
    import brainlayer.wal_checkpoint as wal_checkpoint

    db_path = tmp_path / "brainlayer.db"
    db_path.write_bytes(b"")
    calls: list[str] = []
    monkeypatch.setattr(wal_checkpoint, "resolve_db_path", lambda: str(db_path))
    monkeypatch.setattr(
        wal_checkpoint,
        "checkpoint",
        lambda _path, mode: calls.append(mode) or (1, 20, 10),
    )
    monkeypatch.setattr(wal_checkpoint, "get_wal_size", lambda _path: 0)

    result = wal_checkpoint.run_wal_checkpoint("PASSIVE", retry_busy=True)

    assert result["busy"] == 1
    assert result["attempts"] == 1
    assert calls == ["PASSIVE"]


def test_retrying_truncate_returns_busy_after_bounded_attempts(tmp_path, monkeypatch):
    import brainlayer.wal_checkpoint as wal_checkpoint

    db_path = tmp_path / "brainlayer.db"
    db_path.write_bytes(b"")
    calls: list[str] = []
    delays: list[float] = []
    monkeypatch.setattr(wal_checkpoint, "resolve_db_path", lambda: str(db_path))
    monkeypatch.setattr(
        wal_checkpoint,
        "checkpoint",
        lambda _path, mode: calls.append(mode) or (1, 20, 10),
    )
    monkeypatch.setattr(wal_checkpoint, "get_wal_size", lambda _path: 0)

    result = wal_checkpoint.run_wal_checkpoint(
        "TRUNCATE",
        retry_busy=True,
        max_attempts=3,
        retry_base_seconds=1.0,
        retry_max_seconds=10.0,
        sleep_fn=delays.append,
    )

    assert result["busy"] == 1
    assert result["attempts"] == 3
    assert calls == ["TRUNCATE", "TRUNCATE", "TRUNCATE"]
    assert delays == [1.0, 2.0]


def test_launchagent_runs_retrying_truncate_every_night():
    plist_path = Path(__file__).resolve().parents[1] / "scripts/launchd/com.brainlayer.wal-checkpoint.plist"
    plist = plistlib.loads(plist_path.read_bytes())

    assert plist["StartCalendarInterval"] == {"Hour": 9, "Minute": 30}
    assert plist["ProgramArguments"][-1] == "--retry-busy"
    assert "KeepAlive" not in plist
