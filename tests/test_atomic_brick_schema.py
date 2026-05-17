"""Tests for PR-A atomic-brick-compatible schema migration."""

from __future__ import annotations

import hashlib
from pathlib import Path

import apsw
import sqlite_vec

from brainlayer._helpers import serialize_f32
from brainlayer.vector_store import VectorStore


def _connect_with_vec(db_path: Path) -> apsw.Connection:
    conn = apsw.Connection(str(db_path))
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    return conn


def _create_legacy_db(db_path: Path, *, rows: int) -> None:
    conn = _connect_with_vec(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            source_file TEXT NOT NULL,
            project TEXT,
            content_type TEXT,
            value_type TEXT,
            char_count INTEGER,
            created_at TEXT,
            archived INTEGER DEFAULT 0,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived_at TEXT
        )
    """)
    cursor.execute("""
        CREATE VIRTUAL TABLE chunk_vectors USING vec0(
            chunk_id TEXT PRIMARY KEY,
            embedding FLOAT[1024]
        )
    """)
    cursor.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED
        )
    """)
    cursor.execute("""
        CREATE TABLE chunk_tags (
            chunk_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (chunk_id, tag)
        )
    """)
    for index in range(rows):
        chunk_id = f"legacy-{index:05d}"
        content = f"legacy content {index:05d}"
        cursor.execute(
            """
            INSERT INTO chunks (
                id, content, metadata, source_file, project, content_type,
                value_type, char_count, created_at, archived, superseded_by,
                aggregated_into, archived_at
            ) VALUES (?, ?, '{}', ?, 'atomic-test', 'assistant_text',
                      ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                content,
                f"source-{index % 7}.jsonl",
                "ARCHIVED" if index == 1 else "HIGH",
                len(content),
                f"2026-05-15T00:{index % 60:02d}:00Z",
                1 if index == 1 else 0,
                "legacy-00003" if index == 2 else None,
                "aggregate-00001" if index == 4 else None,
                "2026-05-15T01:00:00Z" if index == 1 else None,
            ),
        )
        vector = [float(index % 17) + (dim / 100_000.0) for dim in range(1024)]
        cursor.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, serialize_f32(vector)),
        )
        cursor.execute(
            """
            INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
            VALUES (?, NULL, NULL, NULL, NULL, NULL, ?)
            """,
            (content, chunk_id),
        )
        cursor.execute("INSERT INTO chunk_tags(chunk_id, tag) VALUES (?, ?)", (chunk_id, f"tag-{index % 3}"))
    conn.close()


def _checksum_rows(db_path: Path) -> tuple[str, int]:
    conn = _connect_with_vec(db_path)
    digest = hashlib.sha256()
    count = 0
    for chunk_id, content, embedding in conn.cursor().execute(
        """
        SELECT c.id, c.content, v.embedding
        FROM chunks c
        JOIN chunk_vectors v ON v.chunk_id = c.id
        ORDER BY c.id
        """
    ):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes(embedding))
        digest.update(b"\n")
        count += 1
    conn.close()
    return digest.hexdigest(), count


def test_atomic_brick_columns_are_added_to_existing_chunks_schema(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_db(db_path, rows=5)

    store = VectorStore(db_path)
    try:
        cols = {row[1] for row in store.conn.cursor().execute("PRAGMA table_info(chunks)")}
    finally:
        store.close()

    assert {"brick_id", "source_uri", "status", "ingested_at", "topic_cluster"}.issubset(cols)


def test_atomic_brick_status_backfills_from_existing_lifecycle_fields(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_db(db_path, rows=5)

    store = VectorStore(db_path)
    try:
        rows = dict(store.conn.cursor().execute("SELECT id, status FROM chunks ORDER BY id"))
    finally:
        store.close()

    assert rows["legacy-00000"] == "active"
    assert rows["legacy-00001"] == "archived"
    assert rows["legacy-00002"] == "superseded"
    assert rows["legacy-00004"] == "superseded"


def test_atomic_brick_migration_preserves_10k_content_and_vector_checksums(tmp_path):
    db_path = tmp_path / "legacy-10k.db"
    _create_legacy_db(db_path, rows=10_000)
    before_hash, before_count = _checksum_rows(db_path)

    store = VectorStore(db_path)
    store.close()

    after_hash, after_count = _checksum_rows(db_path)
    assert before_count == after_count == 10_000
    assert after_hash == before_hash


def test_atomic_brick_migration_preserves_vectorless_rows_fts_and_tags(tmp_path):
    db_path = tmp_path / "legacy-vectorless.db"
    _create_legacy_db(db_path, rows=8)
    conn = _connect_with_vec(db_path)
    conn.cursor().execute("DELETE FROM chunk_vectors WHERE chunk_id IN ('legacy-00006', 'legacy-00007')")
    before = (
        conn.cursor()
        .execute(
            """
        SELECT
            (SELECT COUNT(*) FROM chunks),
            (SELECT COUNT(*) FROM chunk_vectors),
            (SELECT COUNT(*) FROM chunks_fts),
            (SELECT COUNT(*) FROM chunk_tags)
        """
        )
        .fetchone()
    )
    conn.close()

    store = VectorStore(db_path)
    try:
        after = (
            store.conn.cursor()
            .execute(
                """
            SELECT
                (SELECT COUNT(*) FROM chunks),
                (SELECT COUNT(*) FROM chunk_vectors),
                (SELECT COUNT(*) FROM chunks_fts),
                (SELECT COUNT(*) FROM chunk_tags)
            """
            )
            .fetchone()
        )
        rows = dict(
            store.conn.cursor().execute(
                """
                SELECT c.id, COUNT(v.chunk_id)
                FROM chunks c
                LEFT JOIN chunk_vectors v ON v.chunk_id = c.id
                WHERE c.id IN ('legacy-00006', 'legacy-00007')
                GROUP BY c.id
                """
            )
        )
    finally:
        store.close()

    assert after == before
    assert rows == {"legacy-00006": 0, "legacy-00007": 0}


def test_atomic_brick_migration_recovers_from_partial_prior_run(tmp_path):
    db_path = tmp_path / "partial.db"
    _create_legacy_db(db_path, rows=5)
    conn = _connect_with_vec(db_path)
    conn.cursor().execute("ALTER TABLE chunks ADD COLUMN status TEXT DEFAULT 'active'")
    conn.cursor().execute("UPDATE chunks SET status = 'active' WHERE id = 'legacy-00001'")
    conn.close()

    store = VectorStore(db_path)
    try:
        cols = {row[1] for row in store.conn.cursor().execute("PRAGMA table_info(chunks)")}
        row = (
            store.conn.cursor()
            .execute("SELECT status, brick_id, source_uri, ingested_at FROM chunks WHERE id = 'legacy-00001'")
            .fetchone()
        )
    finally:
        store.close()

    assert {"brick_id", "source_uri", "status", "ingested_at", "topic_cluster"}.issubset(cols)
    assert row[0] == "archived"
    assert row[1] == "legacy-00001"
    assert row[2] == "source-1.jsonl"
    assert isinstance(row[3], int)


def test_atomic_brick_migration_recovers_when_columns_exist_without_marker(tmp_path):
    db_path = tmp_path / "partial-all-columns.db"
    _create_legacy_db(db_path, rows=5)
    conn = _connect_with_vec(db_path)
    cursor = conn.cursor()
    for statement in [
        "ALTER TABLE chunks ADD COLUMN brick_id TEXT",
        "ALTER TABLE chunks ADD COLUMN source_uri TEXT",
        "ALTER TABLE chunks ADD COLUMN status TEXT DEFAULT 'active'",
        "ALTER TABLE chunks ADD COLUMN ingested_at INTEGER",
        "ALTER TABLE chunks ADD COLUMN topic_cluster TEXT",
    ]:
        cursor.execute(statement)
    cursor.execute("UPDATE chunks SET status = 'active' WHERE id = 'legacy-00001'")
    conn.close()

    store = VectorStore(db_path)
    try:
        row = (
            store.conn.cursor()
            .execute("SELECT status, brick_id, source_uri, ingested_at FROM chunks WHERE id = 'legacy-00001'")
            .fetchone()
        )
        marker = (
            store.conn.cursor()
            .execute("SELECT applied_at FROM schema_migrations WHERE name = 'atomic_brick_chunks_v1'")
            .fetchone()
        )
    finally:
        store.close()

    assert row[0] == "archived"
    assert row[1] == "legacy-00001"
    assert row[2] == "source-1.jsonl"
    assert isinstance(row[3], int)
    assert marker is not None


def test_status_only_archived_rows_are_filtered_from_text_search(tmp_path):
    db_path = tmp_path / "status-search.db"
    store = VectorStore(db_path)
    try:
        store.upsert_chunks(
            [
                {
                    "id": "status-active",
                    "content": "UniqueStatusToken active memory",
                    "metadata": {},
                    "source_file": "test.jsonl",
                    "project": "atomic-test",
                    "content_type": "note",
                    "value_type": "HIGH",
                    "char_count": 31,
                    "created_at": "2026-05-15T00:00:00Z",
                },
                {
                    "id": "status-archived",
                    "content": "UniqueStatusToken archived memory",
                    "metadata": {},
                    "source_file": "test.jsonl",
                    "project": "atomic-test",
                    "content_type": "note",
                    "value_type": "HIGH",
                    "char_count": 33,
                    "created_at": "2026-05-15T00:01:00Z",
                },
            ],
            [[0.01] * 1024, [0.02] * 1024],
        )
        store.conn.cursor().execute(
            "UPDATE chunks SET status = 'archived', archived = 0, archived_at = NULL, superseded_by = NULL WHERE id = 'status-archived'"
        )

        active_results = store.search(query_text="UniqueStatusToken")
        active_ids = active_results["ids"][0] if active_results["ids"] else []
        historical_results = store.search(query_text="UniqueStatusToken", include_archived=True)
        historical_ids = historical_results["ids"][0] if historical_results["ids"] else []
    finally:
        store.close()

    assert "status-active" in active_ids
    assert "status-archived" not in active_ids
    assert {"status-active", "status-archived"}.issubset(set(historical_ids))


def test_chunk_lifecycle_updates_status_without_deleting_personal_rows(tmp_path):
    db_path = tmp_path / "lifecycle.db"
    store = VectorStore(db_path)
    try:
        store.upsert_chunks(
            [
                {
                    "id": "old",
                    "content": "personal row should be retained",
                    "metadata": {},
                    "source_file": "test.jsonl",
                    "project": "atomic-test",
                    "content_type": "note",
                    "value_type": "HIGH",
                    "char_count": 31,
                    "created_at": "2026-05-15T00:00:00Z",
                },
                {
                    "id": "new",
                    "content": "replacement row",
                    "metadata": {},
                    "source_file": "test.jsonl",
                    "project": "atomic-test",
                    "content_type": "note",
                    "value_type": "HIGH",
                    "char_count": 15,
                    "created_at": "2026-05-15T00:01:00Z",
                },
            ],
            [[0.01] * 1024, [0.02] * 1024],
        )

        assert store.supersede_chunk("old", "new") is True
        row = store.conn.cursor().execute("SELECT status, superseded_by FROM chunks WHERE id = 'old'").fetchone()
        assert row == ("superseded", "new")

        assert store.archive_chunk("new") is True
        row = (
            store.conn.cursor().execute("SELECT status, archived, archived_at FROM chunks WHERE id = 'new'").fetchone()
        )
        assert row[0] == "archived"
        assert row[1] == 1
        assert row[2] is not None

        retained = store.conn.cursor().execute("SELECT COUNT(*) FROM chunks WHERE id IN ('old', 'new')").fetchone()[0]
        assert retained == 2
    finally:
        store.close()


def test_reupsert_does_not_resurrect_archived_status(tmp_path):
    db_path = tmp_path / "status-resurrection.db"
    store = VectorStore(db_path)
    try:
        chunk = {
            "id": "personal-archive",
            "content": "Personal archived row should stay archived",
            "metadata": {},
            "source_file": "test.jsonl",
            "project": "atomic-test",
            "content_type": "note",
            "value_type": "HIGH",
            "char_count": 42,
            "created_at": "2026-05-15T00:00:00Z",
        }
        store.upsert_chunks([chunk], [[0.01] * 1024])
        assert store.archive_chunk("personal-archive") is True

        store.upsert_chunks([chunk], [[0.02] * 1024])

        row = (
            store.conn.cursor()
            .execute("SELECT status, archived, archived_at FROM chunks WHERE id = 'personal-archive'")
            .fetchone()
        )
    finally:
        store.close()

    assert row[0] == "archived"
    assert row[1] == 1
    assert row[2] is not None
