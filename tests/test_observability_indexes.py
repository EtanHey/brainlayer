"""Exercise the producer's actual SQL against the migrated fixture schema."""

import sqlite3
from datetime import UTC, datetime

from brainlayer.observability_indexes import migrate_observability_indexes
from brainlayer.observability_surface import _author_unknown, _emitters, _stores


def test_producer_queries_avoid_chunk_table_scans():
    connection = sqlite3.connect(":memory:")
    connection.executescript("""
        CREATE TABLE schema_migrations(name TEXT PRIMARY KEY, applied_at TEXT);
        CREATE TABLE chunks(id TEXT PRIMARY KEY, content TEXT, created_at TEXT,
            source TEXT, sender TEXT, source_file TEXT, source_class TEXT,
            content_class TEXT, provenance_class TEXT, archived_at TEXT);
        CREATE INDEX idx_chunks_created ON chunks(created_at);
        CREATE INDEX idx_chunks_source_class ON chunks(source_class);
        CREATE INDEX idx_chunks_content_class ON chunks(content_class);
    """)
    connection.executemany(
        "INSERT INTO chunks VALUES (?, 'payload', ?, 'codex', NULL, 'fixture', 'cli', 'user', NULL, NULL)",
        [(str(i), f"2026-11-{10 if i < 950 else 11:02}T12:00:00Z") for i in range(1000)],
    )
    columns = {r[1] for r in connection.execute("PRAGMA table_info(chunks)")}
    now = datetime(2026, 11, 11, 12, tzinfo=UTC)
    builders = (_stores, _emitters, _author_unknown)
    expected = [builder(connection, columns, {}, now) for builder in builders]
    migrate_observability_indexes(connection.cursor())
    migrate_observability_indexes(connection.cursor())
    plans = []

    class Reader:
        def execute(self, sql, params=()):
            plans.extend(r[3] for r in connection.execute("EXPLAIN QUERY PLAN " + sql, params))
            return connection.execute(sql, params)

    actual = [builder(Reader(), columns, {}, now) for builder in builders]
    assert actual == expected
    assert any("idx_chunks_observability_time" in plan for plan in plans)
    assert any("idx_chunks_observability_source" in plan for plan in plans)
    assert any("idx_chunks_observability_author" in plan for plan in plans)
    assert not any(plan == "SCAN chunks" or "USING INDEX idx_chunks_source_class" in plan for plan in plans), plans


def test_legacy_schema_without_source_class_defers_migration():
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE schema_migrations(name TEXT PRIMARY KEY, applied_at TEXT);"
        "CREATE TABLE chunks(id TEXT, created_at TEXT);"
    )
    migrate_observability_indexes(connection.cursor())
    assert connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 0
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'idx_chunks_observability_%'"
        ).fetchone()[0]
        == 0
    )
