"""Tests for shared WAL checkpoint helpers."""

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
