"""A failed SQLite statement may already have rolled back its savepoint."""

import apsw
import pytest

from brainlayer.dedupe import (
    merge_duplicate_chunk,
    merge_existing_chunk_content,
    merge_existing_chunk_seen,
)


def _connection_with_chunk():
    conn = apsw.Connection(":memory:")
    conn.execute(
        """CREATE TABLE chunks (
            id TEXT PRIMARY KEY, content TEXT, tags TEXT, importance REAL,
            half_life_days REAL, seen_count INTEGER, last_seen_at TEXT, created_at TEXT,
            char_count INTEGER, dedupe_hash TEXT, simhash INTEGER,
            simhash_band_0 INTEGER, simhash_band_1 INTEGER,
            simhash_band_2 INTEGER, simhash_band_3 INTEGER
        )"""
    )
    conn.execute(
        "INSERT INTO chunks(id, content, seen_count, created_at) VALUES (?, ?, ?, ?)",
        ("canonical", "Original content", 1, "2026-09-22T00:00:00Z"),
    )
    return conn


def _merge(conn, merge_kind):
    incoming = {"content": "New content", "created_at": "2026-09-22T01:00:00Z"}
    if merge_kind == "duplicate":
        merge_duplicate_chunk(
            conn,
            canonical_id="canonical",
            duplicate_id="duplicate",
            incoming=incoming,
            mechanism="sha256",
            hamming_distance_value=0,
            ensure_schema=False,
        )
    elif merge_kind == "seen":
        merge_existing_chunk_seen(
            conn,
            chunk_id="canonical",
            incoming={**incoming, "content": "Original content"},
            ensure_schema=False,
        )
    else:
        merge_existing_chunk_content(conn, chunk_id="canonical", incoming=incoming, ensure_schema=False)


@pytest.mark.parametrize("merge_kind", ["duplicate", "seen", "content"])
def test_merge_preserves_original_error_after_sqlite_rolls_back_savepoint(merge_kind):
    conn = _connection_with_chunk()
    conn.execute(
        """CREATE TRIGGER abort_merge BEFORE UPDATE ON chunks BEGIN
            SELECT RAISE(ROLLBACK, 'injected original failure');
        END"""
    )

    with pytest.raises(apsw.ConstraintError, match="injected original failure"):
        _merge(conn, merge_kind)

    assert conn.getautocommit()
    assert conn.execute("SELECT content, seen_count FROM chunks").fetchone() == ("Original content", 1)


@pytest.mark.parametrize("merge_kind", ["duplicate", "seen", "content"])
def test_merge_rolls_back_its_savepoint_when_it_survives_the_error(merge_kind):
    conn = _connection_with_chunk()

    # Missing alias/audit tables fail after UPDATE, while the savepoint still exists.
    with pytest.raises(apsw.SQLError, match="no such table"):
        _merge(conn, merge_kind)

    assert conn.getautocommit()
    assert conn.execute("SELECT content, seen_count FROM chunks").fetchone() == ("Original content", 1)
