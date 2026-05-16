import json
import os
import shutil
import sqlite3

from brainlayer._helpers import serialize_f32
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


def test_normalized_exact_hash_preserves_timestamps_but_ignores_stopwords_and_whitespace():
    from brainlayer.dedupe import normalized_exact_hash

    left = "The API was ready\n\nfor the launch"
    right = "api ready launch"
    first_deadline = "Deploy at 2026-05-16T10:00:00Z"
    second_deadline = "Deploy at 2026-05-17T10:00:00Z"

    assert normalized_exact_hash(left) == normalized_exact_hash(right)
    assert normalized_exact_hash(first_deadline) != normalized_exact_hash(second_deadline)


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


def test_simhash_merge_refreshes_canonical_vector(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    first_words = [f"token{i}" for i in range(100)]
    second_words = first_words.copy()
    second_words[0] = "changed0"
    first = " ".join(first_words)
    second = " ".join(second_words)

    store.upsert_chunks(
        [
            _chunk("near-a", first, created_at="2026-05-16T09:00:00Z"),
            _chunk("near-b", second, created_at="2026-05-16T09:05:00Z"),
        ],
        [[0.1] * 1024, [0.7] * 1024],
    )

    rows = store.conn.cursor().execute("SELECT chunk_id, embedding FROM chunk_vectors").fetchall()

    assert [(row[0], row[1]) for row in rows] == [("near-a", serialize_f32([0.4] * 1024))]
    store.close()


def test_merged_content_list_does_not_double_number_existing_items():
    from brainlayer.dedupe import _merged_content

    first = _merged_content("original milestone content", "second milestone content")
    second = _merged_content(first, "third milestone content")

    assert "1. 1." not in second
    assert "\n\n1. original milestone content" in second
    assert "\n\n2. second milestone content" in second
    assert "\n\n3. third milestone content" in second


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


def test_timestamp_only_changes_remain_distinct_chunks(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)

    store.upsert_chunks(
        [
            _chunk("deadline-a", "Deploy at 2026-05-16T10:00:00Z", created_at="2026-05-16T09:00:00Z"),
            _chunk("deadline-b", "Deploy at 2026-05-17T10:00:00Z", created_at="2026-05-16T09:05:00Z"),
        ],
        [[0.1] * 1024, [0.2] * 1024],
    )

    count = store.conn.cursor().execute("SELECT COUNT(*) FROM chunks WHERE archived = 0").fetchone()[0]
    aliases = store.conn.cursor().execute("SELECT COUNT(*) FROM chunk_id_alias").fetchone()[0]

    assert count == 2
    assert aliases == 0
    store.close()


def test_enrichment_content_hash_does_not_disable_dedupe(tmp_path):
    from brainlayer.enrichment_controller import _content_hash

    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    content = "Enrichment can overwrite raw content hash without breaking dedupe"
    store.upsert_chunks([_chunk("hash-a", content, created_at="2026-05-16T09:00:00Z")], [[0.1] * 1024])
    store.conn.cursor().execute(
        "UPDATE chunks SET content_hash = ? WHERE id = ?",
        (_content_hash(content), "hash-a"),
    )

    store.upsert_chunks([_chunk("hash-b", content, created_at="2026-05-16T10:00:00Z")], [[0.2] * 1024])

    row = store.conn.cursor().execute("SELECT id, seen_count FROM chunks WHERE COALESCE(archived, 0) = 0").fetchone()
    alias = (
        store.conn.cursor()
        .execute("SELECT canonical_chunk_id FROM chunk_id_alias WHERE old_chunk_id = 'hash-b'")
        .fetchone()
    )

    assert row == ("hash-a", 2)
    assert alias == ("hash-a",)
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
    cursor.execute(
        "INSERT INTO kg_entities(id, name, entity_type) VALUES (?, ?, ?)", ("entity-1", "Entity One", "project")
    )
    cursor.execute(
        """
        INSERT INTO kg_entity_chunks(entity_id, chunk_id, relevance, context)
        VALUES (?, ?, ?, ?)
        """,
        ("entity-1", "old-b", 0.9, "duplicate link"),
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
    entity_link = (
        checked.conn.cursor().execute("SELECT chunk_id FROM kg_entity_chunks WHERE entity_id = 'entity-1'").fetchone()
    )

    assert active_rows == [("old-a", 2, 8.0, '["new", "old"]')]
    assert alias == ("old-a",)
    assert entity_link == ("old-a",)
    assert checked.get_chunk("old-b")["id"] == "old-a"
    checked.close()


def test_backfill_refuses_default_live_db_without_explicit_allow(monkeypatch, tmp_path):
    from brainlayer import dedupe
    from brainlayer.dedupe import backfill_dedupe_database

    live_path = tmp_path / "brainlayer.db"
    VectorStore(live_path).close()
    monkeypatch.setattr(dedupe, "get_db_path", lambda: live_path)

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


def test_alias_resolution_falls_back_when_alias_table_missing_on_readonly_legacy_db(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                source_file TEXT,
                project TEXT,
                content_type TEXT,
                value_type TEXT,
                tags TEXT,
                importance REAL,
                created_at TEXT,
                summary TEXT,
                superseded_by TEXT,
                aggregated_into TEXT,
                archived_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO chunks(id, content, metadata, source_file, project, content_type, value_type)
            VALUES (?, ?, '{}', 'legacy', 'brainlayer', 'note', 'HIGH')
            """,
            (
                "legacy-id",
                "legacy chunk",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    os.chmod(db_path, 0o444)
    store = None
    try:
        store = VectorStore(db_path)
        chunk = store.get_chunk("legacy-id")
    finally:
        if store is not None:
            store.close()
        os.chmod(db_path, 0o644)

    assert chunk["id"] == "legacy-id"


def test_dedupe_schema_exists_on_minimal_sqlite_connection(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    VectorStore(db_path).close()

    conn = sqlite3.connect(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()

    assert {"seen_count", "last_seen_at", "content_hash", "dedupe_hash", "simhash"}.issubset(cols)
    assert {"dedupe_audit", "chunk_id_alias"}.issubset(tables)
