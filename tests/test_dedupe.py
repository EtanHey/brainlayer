import json
import shutil
import sqlite3

from brainlayer.vector_store import VectorStore


def _chunk(chunk_id: str, content: str, *, created_at: str, tags=None, importance=None, half_life_days=None):
    chunk = {
        "id": chunk_id,
        "content": content,
        "metadata": {"session_id": "session-a"},
        "source_file": "test.jsonl",
        "project": "brainlayer",
        "content_type": "note",
        "value_type": "HIGH",
        "char_count": len(content),
        "source": "test",
        "created_at": created_at,
    }
    if tags is not None:
        chunk["tags"] = tags
    if importance is not None:
        chunk["importance"] = importance
    if half_life_days is not None:
        chunk["half_life_days"] = half_life_days
    return chunk


def test_normalized_exact_hash_ignores_timestamps_stopwords_and_whitespace():
    from brainlayer.dedupe import normalized_exact_hash

    left = "The API was ready at 2026-05-16T10:03:22Z\n\nfor the launch"
    right = "api ready launch"

    assert normalized_exact_hash(left) == normalized_exact_hash(right)


def test_simhash_week_bucket_prevents_weekly_milestone_collapse():
    from brainlayer.dedupe import hamming_distance, is_near_duplicate, simhash64

    week_1 = "Week 1 standup: shipped ingest dedupe tests and opened the review loop"
    week_2 = "Week 2 standup: shipped ingest dedupe tests and opened the review loop"

    left = simhash64(week_1, created_at="2026-05-04T09:00:00Z")
    right = simhash64(week_2, created_at="2026-05-11T09:00:00Z")

    assert hamming_distance(left, right) > 3
    assert not is_near_duplicate(
        week_1,
        week_2,
        created_at_a="2026-05-04T09:00:00Z",
        created_at_b="2026-05-11T09:00:00Z",
    )


def test_identical_reposts_collapse_to_single_chunk_with_seen_count_and_alias(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)

    content = "Identical repost about the FM1 ingest dedupe implementation milestone"
    store.upsert_chunks(
        [
            _chunk("chunk-a", content, created_at="2026-05-16T09:00:00Z", tags=["fm1"], importance=4),
            _chunk("chunk-b", content, created_at="2026-05-16T10:00:00Z", tags=["dedupe"], importance=9),
        ],
        [[0.1] * 1024, [0.2] * 1024],
    )

    row = (
        store.conn.cursor()
        .execute("SELECT id, seen_count, importance, tags, last_seen_at FROM chunks WHERE archived = 0")
        .fetchone()
    )
    alias = (
        store.conn.cursor()
        .execute("SELECT canonical_chunk_id FROM chunk_id_alias WHERE old_chunk_id = 'chunk-b'")
        .fetchone()
    )
    audit = (
        store.conn.cursor()
        .execute("SELECT chunk_id_dropped, chunk_id_kept, mechanism, hamming_distance FROM dedupe_audit")
        .fetchone()
    )

    assert row[0] == "chunk-a"
    assert row[1] == 2
    assert row[2] == 9
    assert set(json.loads(row[3])) == {"fm1", "dedupe"}
    assert row[4] == "2026-05-16T10:00:00Z"
    assert alias == ("chunk-a",)
    assert audit == ("chunk-b", "chunk-a", "sha256", 0)
    assert store.get_chunk("chunk-b")["id"] == "chunk-a"

    store.close()


def test_same_chunk_id_reposts_increment_seen_count(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    content = "Same deterministic chunk id should still count repeated sightings"

    store.upsert_chunks(
        [_chunk("same-id", content, created_at="2026-05-16T09:00:00Z", tags=["first"], importance=4)],
        [[0.1] * 1024],
    )
    store.upsert_chunks(
        [_chunk("same-id", content, created_at="2026-05-16T10:00:00Z", tags=["second"], importance=8)],
        [[0.2] * 1024],
    )

    row = (
        store.conn.cursor()
        .execute("SELECT seen_count, importance, tags, last_seen_at FROM chunks WHERE id = 'same-id'")
        .fetchone()
    )
    audit = (
        store.conn.cursor().execute("SELECT chunk_id_dropped, chunk_id_kept, mechanism FROM dedupe_audit").fetchone()
    )

    assert row == (2, 8.0, '["first", "second"]', "2026-05-16T10:00:00Z")
    assert audit == ("same-id", "same-id", "sha256_same_id")
    store.close()


def test_weekly_standups_remain_distinct_chunks(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)

    store.upsert_chunks(
        [
            _chunk(
                "week-1",
                "Week 1 standup: shipped ingest dedupe tests and opened the review loop",
                created_at="2026-05-04T09:00:00Z",
            ),
            _chunk(
                "week-2",
                "Week 2 standup: shipped ingest dedupe tests and opened the review loop",
                created_at="2026-05-11T09:00:00Z",
            ),
        ],
        [[0.1] * 1024, [0.2] * 1024],
    )

    count = store.conn.cursor().execute("SELECT COUNT(*) FROM chunks WHERE archived = 0").fetchone()[0]
    aliases = store.conn.cursor().execute("SELECT COUNT(*) FROM chunk_id_alias").fetchone()[0]

    assert count == 2
    assert aliases == 0

    store.close()


def test_backfill_merges_snapshot_duplicates_and_preserves_alias_refs(tmp_path):
    from brainlayer.dedupe import backfill_dedupe_database

    live_path = tmp_path / "live.db"
    snapshot_path = tmp_path / "snapshot.db"
    store = VectorStore(live_path)
    content = "Backfill duplicate milestone should collapse through the snapshot only"
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks(id, content, metadata, source_file, tags, importance, created_at, archived)
        VALUES (?, ?, '{}', 'seed', ?, ?, ?, 0)
        """,
        ("old-a", content, '["old"]', 3, "2026-05-16T09:00:00Z"),
    )
    cursor.execute(
        """
        INSERT INTO chunks(id, content, metadata, source_file, tags, importance, created_at, archived)
        VALUES (?, ?, '{}', 'seed', ?, ?, ?, 0)
        """,
        ("old-b", content, '["new"]', 8, "2026-05-16T11:00:00Z"),
    )
    store.close()
    shutil.copy2(live_path, snapshot_path)

    result = backfill_dedupe_database(snapshot_path, batch_size=1)

    assert result.merged == 1

    checked = VectorStore(snapshot_path)
    active_rows = (
        checked.conn.cursor()
        .execute("SELECT id, seen_count, importance, tags FROM chunks WHERE archived = 0")
        .fetchall()
    )
    alias = (
        checked.conn.cursor()
        .execute("SELECT canonical_chunk_id FROM chunk_id_alias WHERE old_chunk_id = 'old-b'")
        .fetchone()
    )

    assert active_rows == [("old-a", 2, 8.0, '["new", "old"]')]
    assert alias == ("old-a",)
    assert checked.get_chunk("old-b")["id"] == "old-a"
    checked.close()


def test_backfill_refuses_default_live_db_without_explicit_allow(monkeypatch, tmp_path):
    from brainlayer import dedupe
    from brainlayer.dedupe import backfill_dedupe_database

    live_path = tmp_path / "brainlayer.db"
    VectorStore(live_path).close()
    monkeypatch.setattr(dedupe, "DEFAULT_DB_PATH", live_path)

    try:
        backfill_dedupe_database(live_path)
    except ValueError as exc:
        assert "snapshot" in str(exc)
    else:
        raise AssertionError("live default DB backfill should require allow_live=True")


def test_alias_resolution_expires_after_grace_period(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    store.upsert_chunks(
        [_chunk("canonical", "Canonical alias target", created_at="2026-05-16T09:00:00Z")],
        [[0.1] * 1024],
    )
    store.conn.cursor().execute(
        """
        INSERT INTO chunk_id_alias(old_chunk_id, canonical_chunk_id, deprecated_at)
        VALUES (?, ?, datetime('now', '-91 days'))
        """,
        ("old-expired", "canonical"),
    )

    assert store.resolve_chunk_id("old-expired") == "old-expired"
    assert store.get_chunk("old-expired") is None

    store.close()


def test_dedupe_schema_exists_on_minimal_sqlite_connection(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    VectorStore(db_path).close()

    conn = sqlite3.connect(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()

    assert {"seen_count", "last_seen_at", "content_hash", "simhash"}.issubset(cols)
    assert {"dedupe_audit", "chunk_id_alias"}.issubset(tables)
