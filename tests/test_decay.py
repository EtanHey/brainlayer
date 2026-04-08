import math
import shutil
import sqlite3
from pathlib import Path

import pytest

from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    database = VectorStore(tmp_path / "decay.db")
    yield database
    database.close()


def test_decay_columns_exist_on_init(store):
    cursor = store.conn.cursor()
    columns = {row[1] for row in cursor.execute("PRAGMA table_info(chunks)")}

    assert "half_life_days" in columns
    assert "last_retrieved" in columns
    assert "retrieval_count" in columns
    assert "decay_score" in columns
    assert "pinned" in columns
    assert "archived" in columns
    assert "archived_at" in columns


def test_compute_decay_score_fresh_chunk(monkeypatch):
    import brainlayer.decay as decay

    monkeypatch.setattr(decay.time, "time", lambda: 1_000_000.0)

    score = decay.compute_decay_score(
        half_life_days=30.0,
        last_retrieved=None,
        created_at=1_000_000.0,
        retrieval_count=0,
        pinned=False,
    )

    assert score == pytest.approx(1.0)


def test_compute_decay_score_matches_fsrs_manual_math(monkeypatch):
    import brainlayer.decay as decay

    now = 1_000_000.0 + (30.0 * 86400.0)
    monkeypatch.setattr(decay.time, "time", lambda: now)

    score = decay.compute_decay_score(
        half_life_days=30.0,
        last_retrieved=None,
        created_at=1_000_000.0,
        retrieval_count=0,
        pinned=False,
    )
    expected = (1.0 + 30.0 / (9.0 * 30.0)) ** -1.0

    assert score == pytest.approx(expected, abs=0.01)


def test_compute_decay_score_retrieval_count_slows_decay(monkeypatch):
    import brainlayer.decay as decay

    now = 1_000_000.0 + (90.0 * 86400.0)
    monkeypatch.setattr(decay.time, "time", lambda: now)

    baseline = decay.compute_decay_score(
        half_life_days=30.0,
        last_retrieved=None,
        created_at=1_000_000.0,
        retrieval_count=0,
        pinned=False,
    )
    reinforced = decay.compute_decay_score(
        half_life_days=30.0,
        last_retrieved=None,
        created_at=1_000_000.0,
        retrieval_count=10,
        pinned=False,
    )

    assert reinforced > baseline


def test_compute_decay_score_pinned_chunks_are_immortal(monkeypatch):
    import brainlayer.decay as decay

    now = 1_000_000.0 + (3650.0 * 86400.0)
    monkeypatch.setattr(decay.time, "time", lambda: now)

    score = decay.compute_decay_score(
        half_life_days=7.0,
        last_retrieved=1_000_000.0,
        created_at=1_000_000.0,
        retrieval_count=100,
        pinned=True,
    )

    assert score == pytest.approx(1.0)


def test_compute_decay_score_has_floor(monkeypatch):
    import brainlayer.decay as decay

    now = 1_000_000.0 + (100_000.0 * 86400.0)
    monkeypatch.setattr(decay.time, "time", lambda: now)

    score = decay.compute_decay_score(
        half_life_days=7.0,
        last_retrieved=None,
        created_at=1_000_000.0,
        retrieval_count=0,
        pinned=False,
    )

    assert math.isfinite(score)
    assert score == pytest.approx(0.05)


def test_compute_decay_score_rejects_non_positive_half_life():
    import brainlayer.decay as decay

    with pytest.raises(ValueError, match="half_life_days"):
        decay.compute_decay_score(
            half_life_days=0.0,
            last_retrieved=None,
            created_at=1_000_000.0,
            retrieval_count=0,
            pinned=False,
        )


def test_compute_decay_score_rejects_negative_retrieval_count():
    import brainlayer.decay as decay

    with pytest.raises(ValueError, match="retrieval_count"):
        decay.compute_decay_score(
            half_life_days=30.0,
            last_retrieved=None,
            created_at=1_000_000.0,
            retrieval_count=-1,
            pinned=False,
        )


def test_decay_migration_sql_applies_cleanly(tmp_path):
    database_path = tmp_path / "migration.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            archived_at TEXT DEFAULT NULL
        )
        """
    )
    migration_sql = Path("migrations/002_decay_columns.sql").read_text()

    connection.executescript(migration_sql)

    columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
    assert "half_life_days" in columns
    assert "last_retrieved" in columns
    assert "retrieval_count" in columns
    assert "decay_score" in columns
    assert "pinned" in columns
    assert "archived" in columns


def test_decay_job_dry_run_does_not_modify_rows(store):
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type, char_count,
            source, created_at, half_life_days, retrieval_count, decay_score
        ) VALUES (
            'dry-run-target', 'dry run content', '{}', 'test.jsonl', 'decay-test',
            'assistant_text', 16, 'claude_code', '2025-01-01T00:00:00Z', 30.0, 0, 1.0
        )
        """
    )

    from brainlayer.decay_job import run_decay_job

    run_decay_job(store.db_path, now=1_800_000_000.0, dry_run=True, batch_size=100)

    row = cursor.execute("SELECT decay_score, archived, archived_at FROM chunks WHERE id = 'dry-run-target'").fetchone()
    assert row == (1.0, 0, None)


def test_decay_job_updates_scores_and_archives_stale_chunks(store):
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type, char_count,
            source, created_at, half_life_days, retrieval_count, decay_score
        ) VALUES (
            'archive-target', 'archive me', '{}', 'test.jsonl', 'decay-test',
            'assistant_text', 10, 'claude_code', '2020-01-01T00:00:00Z', 7.0, 0, 1.0
        )
        """
    )

    from brainlayer.decay_job import run_decay_job

    stats = run_decay_job(store.db_path, now=1_800_000_000.0, dry_run=False, batch_size=100)

    row = cursor.execute("SELECT decay_score, archived, archived_at FROM chunks WHERE id = 'archive-target'").fetchone()
    assert row[0] == pytest.approx(0.05)
    assert row[1] == 1
    assert float(row[2]) == 1_800_000_000.0
    assert stats["archived_rows"] >= 1


def test_backfill_seeds_half_life_and_pins_core_tags(tmp_path):
    source_db = tmp_path / "backfill-source.db"
    seeded = VectorStore(source_db)
    cursor = seeded.conn.cursor()
    cursor.executemany(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type, char_count, source,
            created_at, importance, tags
        ) VALUES (?, ?, '{}', 'test.jsonl', 'decay-test', 'assistant_text', ?, 'claude_code', ?, ?, ?)
        """,
        [
            ("important", "important chunk", 15, "2026-04-01T00:00:00Z", 10.0, '["decision"]'),
            ("medium", "medium chunk", 11, "2026-04-01T00:00:00Z", 5.0, '["note"]'),
            ("small", "small chunk", 10, "2026-04-01T00:00:00Z", 1.0, '["scratch"]'),
        ],
    )
    seeded.close()

    copy_db = tmp_path / "brain-copy.db"
    shutil.copy2(source_db, copy_db)

    from brainlayer.decay_backfill import backfill_decay_fields

    stats = backfill_decay_fields(copy_db)

    checked = VectorStore(copy_db)
    rows = {
        row[0]: row[1:]
        for row in checked.conn.cursor().execute(
            "SELECT id, half_life_days, pinned, last_retrieved FROM chunks ORDER BY id"
        )
    }
    checked.close()

    assert rows["important"][0] == 90.0
    assert rows["important"][1] == 1
    assert rows["important"][2] is not None
    assert rows["medium"][0] == 30.0
    assert rows["small"][0] == 7.0
    assert stats["updated_rows"] == 3


def test_backfill_adds_missing_decay_columns_on_pre_v3_db(tmp_path):
    database_path = tmp_path / "pre-v3.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            source_file TEXT NOT NULL,
            project TEXT,
            content_type TEXT,
            value_type TEXT,
            char_count INTEGER,
            source TEXT,
            sender TEXT,
            language TEXT,
            conversation_id TEXT,
            position INTEGER,
            context_summary TEXT,
            created_at TEXT,
            importance REAL,
            tags TEXT,
            archived_at TEXT DEFAULT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type, value_type, char_count, source,
            sender, language, conversation_id, position, context_summary, created_at, importance, tags
        )
        VALUES (
            'legacy', 'legacy chunk', '{}', 'test.jsonl', 'decay-test', 'assistant_text', 'note', 12, 'claude_code',
            'assistant', 'en', 'conv-1', 1, 'summary', '2026-04-01T00:00:00Z', 5.0, '["decision"]'
        )
        """
    )
    connection.commit()
    connection.close()

    from brainlayer.decay_backfill import backfill_decay_fields

    backfill_decay_fields(database_path)

    checked = sqlite3.connect(database_path)
    row = checked.execute("SELECT half_life_days, pinned, last_retrieved FROM chunks WHERE id = 'legacy'").fetchone()
    checked.close()

    assert row[0] == 30.0
    assert row[1] == 1
    assert row[2] is not None


def test_archive_chunk_sets_archived_flag(store):
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type, char_count, source
        ) VALUES (
            'archive-flag', 'archivable', '{}', 'test.jsonl', 'decay-test', 'assistant_text', 10, 'claude_code'
        )
        """
    )

    assert store.archive_chunk("archive-flag") is True

    row = cursor.execute("SELECT archived, archived_at, value_type FROM chunks WHERE id = 'archive-flag'").fetchone()
    assert row[0] == 1
    assert row[1] is not None
    assert row[2] == "ARCHIVED"


def test_init_backfills_archived_flag_for_preexisting_archived_rows(tmp_path):
    database_path = tmp_path / "archived-backfill.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            source_file TEXT NOT NULL,
            project TEXT,
            content_type TEXT,
            value_type TEXT,
            char_count INTEGER,
            source TEXT,
            sender TEXT,
            language TEXT,
            conversation_id TEXT,
            position INTEGER,
            context_summary TEXT,
            tags TEXT,
            tag_confidence REAL,
            summary TEXT,
            importance REAL,
            intent TEXT,
            enriched_at TEXT,
            primary_symbols TEXT,
            resolved_query TEXT,
            key_facts TEXT,
            resolved_queries TEXT,
            epistemic_level TEXT,
            version_scope TEXT,
            debt_impact TEXT,
            external_deps TEXT,
            created_at TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            sentiment_signals TEXT,
            half_life_days REAL DEFAULT 30.0,
            last_retrieved REAL DEFAULT NULL,
            retrieval_count INTEGER DEFAULT 0,
            decay_score REAL DEFAULT 1.0,
            pinned INTEGER DEFAULT 0,
            archived INTEGER DEFAULT 0,
            superseded_by TEXT,
            aggregated_into TEXT,
            archived_at TEXT
        )
        """
    )
    connection.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type, value_type, char_count, source, archived
        ) VALUES (
            'legacy-archived', 'legacy archived', '{}', 'test.jsonl', 'decay-test', 'assistant_text', 'ARCHIVED', 15,
            'claude_code', 0
        )
        """
    )
    connection.commit()
    connection.close()

    store = VectorStore(database_path)
    row = store.conn.cursor().execute("SELECT archived FROM chunks WHERE id = 'legacy-archived'").fetchone()
    store.close()

    assert row[0] == 1
