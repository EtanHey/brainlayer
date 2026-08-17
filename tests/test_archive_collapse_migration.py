"""Repair (c) data migration follows the proven repair-(b) safety pattern."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
PREIMAGE_TABLE = "archive_collapse_preimage"
MIGRATION_NAME = "2026_08_17_archive_collapse_archived_at"
CHUNKS_FTS_UPDATE_OF = "content, summary, tags, resolved_query, key_facts, resolved_queries, content_class"


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            summary TEXT,
            tags TEXT,
            resolved_query TEXT,
            key_facts TEXT,
            resolved_queries TEXT,
            content_class TEXT,
            created_at TEXT,
            archived_at TEXT,
            archived INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active',
            value_type TEXT,
            superseded_by TEXT,
            aggregated_into TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE chunks_fts (
            content TEXT,
            summary TEXT,
            tags TEXT,
            resolved_query TEXT,
            key_facts TEXT,
            resolved_queries TEXT,
            chunk_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE chunks_fts_operational (
            content TEXT, summary TEXT, tags TEXT, resolved_query TEXT,
            key_facts TEXT, resolved_queries TEXT, chunk_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE chunks_fts_trigram (
            content TEXT, summary TEXT, tags TEXT, resolved_query TEXT,
            key_facts TEXT, resolved_queries TEXT, chunk_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE chunk_fts_rowids (
            chunk_id TEXT PRIMARY KEY,
            fts_rowid INTEGER,
            operational_rowid INTEGER,
            trigram_rowid INTEGER
        )
        """
    )
    conn.execute("CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY)")
    return conn


def _install_chunks_fts_update_trigger(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TRIGGER IF EXISTS chunks_fts_update")
    conn.execute(
        f"""
        CREATE TRIGGER chunks_fts_update
        AFTER UPDATE OF {CHUNKS_FTS_UPDATE_OF} ON chunks BEGIN
            DELETE FROM chunks_fts WHERE chunk_id = old.id;
            INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
            SELECT new.content, new.summary, new.tags, new.resolved_query,
                   new.key_facts, new.resolved_queries, new.id;
            INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid)
            VALUES (new.id, last_insert_rowid())
            ON CONFLICT(chunk_id) DO UPDATE SET fts_rowid = excluded.fts_rowid;
        END
        """
    )


def _seed(conn: sqlite3.Connection) -> None:
    rows = [
        ("keep-active", "active memory", None, 0, "active", "high", None, None),
        ("flag-only", "flag without timestamp", None, 1, "active", "high", None, None),
        ("status-only", "status without timestamp", None, 0, "archived", "high", None, None),
        ("value-only", "value_type without timestamp", None, 0, "active", "ARCHIVED", None, None),
        ("all-four", "tetraplicated", "2026-08-01T00:00:00Z", 1, "archived", "ARCHIVED", None, None),
        ("lineage-superseded", "superseded not archive", None, 0, "superseded", "high", "newer", None),
        ("already-canonical", "timestamp only", "2026-08-02T00:00:00Z", 0, "active", "high", None, None),
    ]
    for row in rows:
        conn.execute(
            """
            INSERT INTO chunks (
                id, content, created_at, archived_at, archived, status, value_type,
                superseded_by, aggregated_into
            ) VALUES (?, ?, '2026-07-01T00:00:00Z', ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        conn.execute(
            "INSERT INTO chunks_fts(content, chunk_id) VALUES (?, ?)",
            (row[1], row[0]),
        )
        conn.execute(
            "INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid) VALUES (?, last_insert_rowid())",
            (row[0],),
        )
        conn.execute("INSERT INTO chunk_vectors_rowids(id) VALUES (?)", (row[0],))
    conn.commit()


def test_dry_run_does_not_write(tmp_path):
    from brainlayer.archive_collapse import collapse_archive_representations

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    conn.close()

    result = collapse_archive_representations(db_path, git_sha=GIT_SHA, apply=False)
    assert result.updated == 0
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT archived FROM chunks WHERE id = 'flag-only'").fetchone()[0] == 1
    assert (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (PREIMAGE_TABLE,),
        ).fetchone()
        is None
    )
    conn.close()
    assert result.would_update >= 4


def test_apply_backfills_timestamp_and_clears_twins(tmp_path):
    from brainlayer.archive_collapse import collapse_archive_representations

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    conn.close()

    result = collapse_archive_representations(db_path, git_sha=GIT_SHA, apply=True, spot_check=4)
    assert result.post_twin_count == 0
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    flag = conn.execute("SELECT * FROM chunks WHERE id = 'flag-only'").fetchone()
    assert flag["archived_at"]
    assert flag["archived"] == 0
    assert flag["status"] == "active"
    assert (flag["value_type"] or "").lower() != "archived"
    all_four = conn.execute("SELECT * FROM chunks WHERE id = 'all-four'").fetchone()
    assert all_four["archived_at"] == "2026-08-01T00:00:00Z"
    assert all_four["archived"] == 0
    assert all_four["status"] == "active"
    assert (all_four["value_type"] or "").lower() != "archived"
    lineage = conn.execute("SELECT * FROM chunks WHERE id = 'lineage-superseded'").fetchone()
    assert lineage["superseded_by"] == "newer"
    assert lineage["archived_at"] is None
    assert lineage["status"] == "superseded"
    canonical = conn.execute("SELECT * FROM chunks WHERE id = 'already-canonical'").fetchone()
    assert canonical["archived_at"] == "2026-08-02T00:00:00Z"
    details = json.loads(
        conn.execute("SELECT details FROM schema_migrations WHERE name = ?", (MIGRATION_NAME,)).fetchone()[0]
    )
    assert details["git_sha"] == GIT_SHA
    conn.close()
    assert result.aux_counts_before == result.aux_counts_after
    assert result.spot_checks
    assert all(item["ok"] for item in result.spot_checks)


def test_apply_does_not_fire_fts_update_trigger(tmp_path):
    from brainlayer.archive_collapse import collapse_archive_representations

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    before = conn.execute("SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = 'flag-only'").fetchone()[0]
    conn.close()

    collapse_archive_representations(db_path, git_sha=GIT_SHA, apply=True)

    conn = sqlite3.connect(db_path)
    after = conn.execute("SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = 'flag-only'").fetchone()[0]
    conn.close()
    assert after == before


def test_preimage_captures_twins_before_update(tmp_path):
    from brainlayer.archive_collapse import collapse_archive_representations

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    conn.close()

    collapse_archive_representations(db_path, git_sha=GIT_SHA, apply=True)

    conn = sqlite3.connect(db_path)
    pre = dict(conn.execute(f"SELECT id, archived FROM {PREIMAGE_TABLE}"))
    assert pre["flag-only"] == 1
    assert pre["all-four"] == 1
    assert "keep-active" not in pre
    assert "lineage-superseded" not in pre
    conn.close()


def test_refuses_live_canonical_db(tmp_path, monkeypatch):
    from brainlayer.archive_collapse import collapse_archive_representations
    from brainlayer.paths import resolve_db_path

    live = resolve_db_path()
    live.parent.mkdir(parents=True, exist_ok=True)
    conn = _make_db(live)
    _seed(conn)
    conn.close()

    with pytest.raises(RuntimeError, match="live BrainLayer DB"):
        collapse_archive_representations(live, git_sha=GIT_SHA, apply=True)


def test_spot_check_rereads_stored_values(tmp_path):
    from brainlayer.archive_collapse import collapse_archive_representations

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    conn.close()

    result = collapse_archive_representations(db_path, git_sha=GIT_SHA, apply=True, spot_check=3)
    conn = sqlite3.connect(db_path)
    stored = {
        row[0]: row[1:] for row in conn.execute("SELECT id, archived_at, archived, status, value_type FROM chunks")
    }
    conn.close()
    assert result.spot_checks
    for item in result.spot_checks:
        archived_at, archived, status, value_type = stored[item["id"]]
        assert item["stored_archived_at"] == archived_at
        assert item["stored_archived"] == archived
        assert item["stored_status"] == status
        assert item["stored_value_type"] == value_type
        assert item["ok"] is True


def test_dry_run_cli_spot_check_exits_zero(tmp_path):
    from brainlayer.archive_collapse import main

    db_path = tmp_path / "copy.db"
    conn = _make_db(db_path)
    _install_chunks_fts_update_trigger(conn)
    _seed(conn)
    conn.close()

    assert main(["--db", str(db_path), "--git-sha", GIT_SHA, "--spot-check", "3"]) == 0
