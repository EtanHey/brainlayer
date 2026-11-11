"""Exercise explicit migration and the producer's actual query plans."""

import json
import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from brainlayer.cli import app
from brainlayer.observability_indexes import REQUIRED_COLUMNS
from brainlayer.observability_surface import _author_unknown, _emitters, _stores


@pytest.fixture(autouse=True)
def ample_disk(monkeypatch):
    monkeypatch.setattr(
        "brainlayer.observability_indexes.measure_disk_space",
        lambda path: SimpleNamespace(free_now_bytes=1 << 60),
    )


def fixture_db(tmp_path):
    path = tmp_path / "fixture.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE chunks(id TEXT PRIMARY KEY, content TEXT, created_at TEXT,
                source TEXT, sender TEXT, source_file TEXT, source_class TEXT,
                content_class TEXT, provenance_class TEXT, archived_at TEXT);
            CREATE INDEX idx_chunks_created ON chunks(created_at);
            CREATE INDEX idx_chunks_source_class ON chunks(source_class);
            CREATE INDEX idx_chunks_content_class ON chunks(content_class);
        """)
        db.executemany(
            "INSERT INTO chunks VALUES (?, 'payload', ?, 'codex', NULL, 'fixture', 'cli', 'user', NULL, NULL)",
            [(str(i), f"2026-11-{10 if i < 950 else 11:02}T12:00:00Z") for i in range(1000)],
        )
    return path


def ensure(path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_DB", str(path))
    result = CliRunner().invoke(app, ["observability", "--ensure-indexes"])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_ensure_command_and_each_query_plan(tmp_path, monkeypatch):
    path = fixture_db(tmp_path)
    with sqlite3.connect(path) as db:
        columns = {r[1] for r in db.execute("PRAGMA table_info(chunks)")}
        now = datetime(2026, 11, 11, 12, tzinfo=UTC)
        builders = (_stores, _emitters, _author_unknown)
        expected = [builder(db, columns, {}, now) for builder in builders]
        assert ensure(path, monkeypatch)["status"] == "built"
        assert ensure(path, monkeypatch)["status"] == "present"
        plans = []

        class Reader:
            def execute(self, sql, params=()):
                plans.append([r[3] for r in db.execute("EXPLAIN QUERY PLAN " + sql, params)])
                return db.execute(sql, params)

        actual = [builder(Reader(), columns, {}, now) for builder in builders]
        assert actual == expected
        expected_indexes = (
            "idx_chunks_ingested_at",
            "idx_chunks_content_class",
            "idx_chunks_observability_time",
            "idx_chunks_observability_time",
            "idx_chunks_observability_source",
            "idx_chunks_observability_time",
            "idx_chunks_source_class",
            "idx_chunks_observability_author",
            "idx_chunks_observability_author",
            "idx_chunks_observability_author",
            "idx_chunks_observability_author",
            "idx_chunks_observability_author",
        )
        # COUNT(*) may choose any covering index; every other query has its own required plan.
        for i in range(1, len(expected_indexes)):
            assert expected_indexes[i] in " ".join(plans[i]), (i, plans[i])
            assert "SCAN chunks" not in plans[i], (i, plans[i])


@pytest.mark.parametrize("column", sorted(REQUIRED_COLUMNS))
def test_legacy_guard_checks_every_required_column(tmp_path, monkeypatch, column):
    path = fixture_db(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("DROP INDEX idx_chunks_created")
        db.execute("DROP INDEX idx_chunks_source_class")
        if column == "id":
            db.execute("ALTER TABLE chunks RENAME TO old_chunks")
            fields = [r[1] for r in db.execute("PRAGMA table_info(old_chunks)") if r[1] != column]
            db.execute("CREATE TABLE chunks AS SELECT " + ",".join(fields) + " FROM old_chunks")
        else:
            db.execute("ALTER TABLE chunks DROP COLUMN " + column)
    result = ensure(path, monkeypatch)
    assert result["status"] == "skipped"
    assert column in result["reason"]


def test_writable_vector_store_open_does_not_build_indexes(tmp_path, monkeypatch):
    from brainlayer.vector_store import VectorStore

    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(tmp_path / "pids"))
    store = VectorStore(tmp_path / "store.db")
    try:
        assert (
            list(
                store.conn.cursor().execute(
                    "SELECT name FROM sqlite_master WHERE name LIKE 'idx_chunks_observability_%'"
                )
            )
            == []
        )
    finally:
        store.close()


def test_short_disk_skips_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "brainlayer.observability_indexes.measure_disk_space", lambda path: SimpleNamespace(free_now_bytes=0)
    )
    result = ensure(fixture_db(tmp_path), monkeypatch)
    assert result["status"] == "skipped"
    assert "disk space" in result["reason"]


def test_build_checkpoints_outside_transaction_with_long_timeout(tmp_path, monkeypatch):
    from brainlayer import observability_indexes

    path = fixture_db(tmp_path)
    connect = sqlite3.connect
    statements = []
    options = []

    def traced(*args, **kwargs):
        options.append(kwargs)
        db = connect(*args, **kwargs)
        db.set_trace_callback(statements.append)
        return db

    monkeypatch.setattr(observability_indexes.sqlite3, "connect", traced)
    assert ensure(path, monkeypatch)["status"] == "built"
    assert options[0]["timeout"] == 300
    assert options[0]["isolation_level"] is None
    checkpoints = [i for i, sql in enumerate(statements) if sql == "PRAGMA wal_checkpoint(TRUNCATE)"]
    assert len(checkpoints) == 2
    assert checkpoints[0] < statements.index("BEGIN IMMEDIATE") < statements.index("COMMIT") < checkpoints[1]
