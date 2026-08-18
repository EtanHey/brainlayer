"""Non-ISO timestamp TEXT columns must normalize to ISO-8601 UTC."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from brainlayer.chunk_origin_wipe import live_canonical_db_path
from brainlayer.timestamp_iso import (
    ISO_TIMESTAMP_COLUMNS,
    MIGRATION_NAME,
    PREIMAGE_TABLE,
    is_iso_utc,
    normalize_timestamp,
    normalize_timestamp_for_migration,
    normalize_timestamps,
)

GIT_SHA = "0123456789abcdef0123456789abcdef01234567"


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT,
            created_at TEXT,
            last_seen_at TEXT,
            archived_at TEXT,
            enriched_at TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO chunks(id, content, source, created_at, last_seen_at, archived_at, enriched_at)
        VALUES
          ('style_53_unix', 'unix archived', 'claude_code', '2026-08-18T00:00:00Z', '2026-08-18T00:00:00Z',
           '1780190105.85048', NULL),
          ('iso_already', 'already iso', 'claude_code', '2026-08-18T00:00:00Z', '2026-08-18T00:00:01+00:00',
           '2026-08-17T21:00:00Z', '2026-08-17T21:01:00.123Z'),
          ('created_unix', 'unix created', 'claude_code', '1780190105', NULL, NULL, NULL),
          ('naive_iso', 'naive created', 'claude_code', '2026-07-14T11:45:26.994441', NULL, NULL, NULL),
          ('whatsapp_local', 'whatsapp local', 'whatsapp', '2019-02-11T16:05:15', NULL, NULL, NULL),
          ('offset_jerusalem', 'local offset', 'claude_code', '2026-06-09T13:37:11+03:00', NULL, NULL, NULL),
          ('tilde_garbage', 'unparseable', 'claude_code', '2026-05-28T~12:35:00Z', NULL, NULL, NULL)
        """
    )
    conn.commit()
    return conn


def test_normalize_timestamp_converts_unix_float():
    converted = normalize_timestamp("1780190105.85048")
    assert converted == "2026-05-31T01:15:05.850480Z"
    assert is_iso_utc(converted)


def test_normalize_timestamp_leaves_iso_and_null():
    assert normalize_timestamp("2026-08-18T00:00:00Z") == "2026-08-18T00:00:00Z"
    assert normalize_timestamp(None) is None
    assert normalize_timestamp("") is None


def test_dry_run_does_not_write(tmp_path):
    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    conn.close()
    result = normalize_timestamps(db_path, git_sha=GIT_SHA, apply=False)
    assert result.would_update >= 2
    assert result.updated == 0
    conn = sqlite3.connect(db_path)
    archived = conn.execute("SELECT archived_at FROM chunks WHERE id='style_53_unix'").fetchone()[0]
    conn.close()
    assert archived == "1780190105.85048"


def test_normalize_timestamp_for_migration_whatsapp_jerusalem():
    converted = normalize_timestamp_for_migration("2019-02-11T16:05:15", source="whatsapp")
    assert converted == "2019-02-11T14:05:15Z"
    assert is_iso_utc(converted)


def test_normalize_timestamp_for_migration_unknown_naive_returns_none():
    assert normalize_timestamp_for_migration("2026-07-14T11:45:26.994441", source="claude_code") is None


def test_apply_rewrites_unix_floats_and_keeps_iso(tmp_path):
    db_path = tmp_path / "copy.db"
    _make_db(db_path).close()
    result = normalize_timestamps(db_path, git_sha=GIT_SHA, apply=True, spot_check=3)
    assert result.updated >= 2
    conn = sqlite3.connect(db_path)
    rows = {
        row[0]: row[1:]
        for row in conn.execute("SELECT id, created_at, last_seen_at, archived_at, enriched_at FROM chunks")
    }
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    applied = conn.execute("SELECT name FROM schema_migrations WHERE name=?", (MIGRATION_NAME,)).fetchone()
    conn.close()
    assert is_iso_utc(rows["style_53_unix"][2])
    assert is_iso_utc(rows["created_unix"][0])
    assert is_iso_utc(rows["iso_already"][0])
    assert rows["naive_iso"][0] == "2026-07-14T11:45:26.994441"
    assert rows["whatsapp_local"][0] == "2019-02-11T14:05:15Z"
    assert rows["offset_jerusalem"][0] == "2026-06-09T10:37:11Z"
    assert rows["tilde_garbage"][0] == "2026-05-28T~12:35:00Z"
    assert result.skipped_unparseable >= 1
    assert PREIMAGE_TABLE in names
    assert applied is not None
    assert all(item["ok"] for item in result.spot_checks)


def test_spot_check_fails_when_stored_instant_differs_from_preimage():
    import sqlite3

    from brainlayer.timestamp_iso import _spot_ok

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE chunks (id TEXT, source TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE timestamp_iso_preimage (id TEXT PRIMARY KEY, created_at TEXT)")
    conn.execute(
        "INSERT INTO chunks VALUES ('w1', 'whatsapp', '2019-02-11T16:05:15Z')"
    )
    conn.execute(
        "INSERT INTO timestamp_iso_preimage VALUES ('w1', '2019-02-11T16:05:15')"
    )
    stored = conn.execute("SELECT id, source, created_at FROM chunks WHERE id='w1'").fetchone()
    preimage = conn.execute(
        "SELECT id, created_at FROM timestamp_iso_preimage WHERE id='w1'"
    ).fetchone()
    assert not _spot_ok(stored, preimage, ["created_at"], source="whatsapp")
    conn.close()


def test_refuses_live_db_without_allow_live(tmp_path, monkeypatch):
    fake_live = tmp_path / "brainlayer.db"
    fake_live.write_bytes(b"")
    monkeypatch.setattr("brainlayer.timestamp_iso.live_canonical_db_path", lambda: fake_live)
    monkeypatch.setattr("brainlayer.chunk_origin_wipe.live_canonical_db_path", lambda: fake_live)
    with pytest.raises(RuntimeError, match="refusing to write the live BrainLayer DB"):
        normalize_timestamps(fake_live, git_sha=GIT_SHA, apply=False)


def test_iso_timestamp_columns_cover_archived_at():
    assert "archived_at" in ISO_TIMESTAMP_COLUMNS
    assert "created_at" in ISO_TIMESTAMP_COLUMNS
    assert live_canonical_db_path().name == "brainlayer.db"
