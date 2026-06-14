import hashlib

import apsw

from brainlayer.vector_store import VectorStore

SENTINEL_END = "9999-12-31T23:59:59.999999Z"


def _legacy_conn(tmp_path):
    conn = apsw.Connection(str(tmp_path / "legacy.db"))
    conn.cursor().execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            source_file TEXT NOT NULL,
            created_at TEXT
        )
        """
    )
    return conn


def _insert_chunk(conn, chunk_id: str, content: str, *, invalid_at: str | None = None) -> None:
    conn.cursor().execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, created_at, content_hash,
            valid_from, invalid_at, sys_period_start, sys_period_end
        )
        VALUES (?, ?, '{}', 'fixture', '2026-06-14T00:00:00+00:00', ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            content,
            hashlib.sha256(content.strip().encode("utf-8")).hexdigest(),
            "2026-06-14T00:00:00.000000Z",
            invalid_at,
            "2026-06-14T00:00:00.000000Z",
            SENTINEL_END,
        ),
    )


def test_migration_preserves_old_rows_on_update_and_delete(tmp_path):
    from brainlayer.bitemporal import apply_bitemporal_migration

    conn = _legacy_conn(tmp_path)
    apply_bitemporal_migration(conn)
    _insert_chunk(conn, "chunk-1", "original payload")

    conn.cursor().execute("UPDATE chunks SET content = ? WHERE id = ?", ("updated payload", "chunk-1"))
    conn.cursor().execute("DELETE FROM chunks WHERE id = ?", ("chunk-1",))

    history = list(
        conn.cursor().execute(
            "SELECT id, content, sys_period_start, sys_period_end FROM _chunks_history WHERE id = ? ORDER BY rowid",
            ("chunk-1",),
        )
    )

    assert [row[1] for row in history] == ["original payload", "updated payload"]
    assert history[0][2] == "2026-06-14T00:00:00.000000Z"
    assert history[0][3] != SENTINEL_END
    assert history[1][3] != SENTINEL_END
    assert conn.cursor().execute("SELECT COUNT(*) FROM chunks WHERE id = 'chunk-1'").fetchone()[0] == 0


def test_history_supports_as_of_queries_for_superseded_content(tmp_path):
    from brainlayer.bitemporal import apply_bitemporal_migration

    conn = _legacy_conn(tmp_path)
    apply_bitemporal_migration(conn)
    _insert_chunk(conn, "chunk-2", "before correction")

    conn.cursor().execute("UPDATE chunks SET content = ? WHERE id = ?", ("after correction", "chunk-2"))
    as_of_rows = list(
        conn.cursor().execute(
            """
            SELECT content FROM _chunks_history
            WHERE id = ?
              AND sys_period_start <= ?
              AND sys_period_end > ?
            """,
            ("chunk-2", "2026-06-14T00:00:01.000000Z", "2026-06-14T00:00:01.000000Z"),
        )
    )

    assert as_of_rows == [("before correction",)]


def test_current_view_uses_partial_invalid_at_index(tmp_path):
    from brainlayer.bitemporal import apply_bitemporal_migration

    conn = _legacy_conn(tmp_path)
    apply_bitemporal_migration(conn)
    _insert_chunk(conn, "active", "current")
    _insert_chunk(conn, "inactive", "old", invalid_at="2026-06-14T01:00:00.000000Z")

    current_ids = list(conn.cursor().execute("SELECT id FROM chunks WHERE invalid_at IS NULL ORDER BY id"))
    plan = " ".join(
        str(part)
        for row in conn.cursor().execute("EXPLAIN QUERY PLAN SELECT id FROM chunks WHERE invalid_at IS NULL")
        for part in row
    )

    assert current_ids == [("active",)]
    assert "idx_chunks_current_active" in plan


def test_supersede_updates_invalid_at_and_keeps_recoverable_history(tmp_path):
    from brainlayer.bitemporal import apply_bitemporal_migration, supersede_chunk

    conn = _legacy_conn(tmp_path)
    apply_bitemporal_migration(conn)
    _insert_chunk(conn, "chunk-3", "recover me")

    assert supersede_chunk(conn, "chunk-3", invalid_at="2026-06-14T02:00:00.000000Z") is True

    current_rows = list(conn.cursor().execute("SELECT id FROM chunks WHERE invalid_at IS NULL"))
    history_rows = list(
        conn.cursor().execute(
            """
            SELECT content FROM _chunks_history
            WHERE id = ?
              AND sys_period_start <= ?
              AND sys_period_end > ?
            """,
            ("chunk-3", "2026-06-14T00:00:01.000000Z", "2026-06-14T00:00:01.000000Z"),
        )
    )

    assert current_rows == []
    assert history_rows == [("recover me",)]


def test_bitemporal_migration_is_idempotent(tmp_path):
    from brainlayer.bitemporal import apply_bitemporal_migration

    conn = _legacy_conn(tmp_path)
    apply_bitemporal_migration(conn)
    first_columns = list(conn.cursor().execute("PRAGMA table_info(chunks)"))
    first_history_columns = list(conn.cursor().execute("PRAGMA table_info(_chunks_history)"))
    first_triggers = list(
        conn.cursor().execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'chunks_bitemporal_%' ORDER BY name"
        )
    )

    apply_bitemporal_migration(conn)

    assert list(conn.cursor().execute("PRAGMA table_info(chunks)")) == first_columns
    assert list(conn.cursor().execute("PRAGMA table_info(_chunks_history)")) == first_history_columns
    assert (
        list(
            conn.cursor().execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'chunks_bitemporal_%' ORDER BY name"
            )
        )
        == first_triggers
    )


def test_store_memory_populates_content_hash_and_temporal_columns(tmp_path):
    from brainlayer.store import store_memory

    store = VectorStore(tmp_path / "store.db")
    try:
        result = store_memory(
            store=store,
            embed_fn=None,
            content="Temporal store write",
            memory_type="note",
            project="brainlayer",
        )
        row = (
            store.conn.cursor()
            .execute(
                "SELECT content_hash, valid_from, invalid_at, sys_period_start, sys_period_end FROM chunks WHERE id = ?",
                (result["id"],),
            )
            .fetchone()
        )
    finally:
        store.close()

    assert row == (
        hashlib.sha256("Temporal store write".encode("utf-8")).hexdigest(),
        row[1],
        None,
        row[3],
        SENTINEL_END,
    )
    assert row[1] is not None
    assert row[3] is not None


def test_vector_store_startup_does_not_install_history_triggers_without_opt_in(tmp_path):
    store = VectorStore(tmp_path / "opt-in.db")
    try:
        cursor = store.conn.cursor()
        history_table = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = '_chunks_history'"
        ).fetchone()
        bitemporal_triggers = list(
            cursor.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'chunks_bitemporal_%'")
        )
    finally:
        store.close()

    assert history_table is None
    assert bitemporal_triggers == []
