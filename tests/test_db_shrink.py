import sqlite3

import pytest


def _init_chunks(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            summary TEXT,
            tags TEXT,
            resolved_query TEXT,
            key_facts TEXT,
            resolved_queries TEXT,
            created_at TEXT,
            project TEXT,
            content_type TEXT
        );
        CREATE TABLE chunk_fts_rowids (
            chunk_id TEXT PRIMARY KEY,
            fts_rowid INTEGER,
            trigram_rowid INTEGER
        );
        """
    )


def test_guard_refuses_live_path_without_explicit_override(tmp_path, monkeypatch):
    from brainlayer import db_shrink

    live_path = tmp_path / "brainlayer.db"
    live_path.touch()
    monkeypatch.setattr(db_shrink, "get_db_path", lambda: live_path)

    with pytest.raises(ValueError, match="Refusing to write to the canonical live DB"):
        db_shrink.assert_not_live_db(live_path)

    db_shrink.assert_not_live_db(live_path, allow_live=True)


def test_migrate_fts_single_trigram_drops_redundant_table(tmp_path):
    from brainlayer.db_shrink import migrate_fts_single_trigram

    db_path = tmp_path / "fts.db"
    conn = sqlite3.connect(db_path)
    _init_chunks(conn)
    conn.executescript(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED,
            prefix='2 3 4', tokenize='unicode61 remove_diacritics 2'
        );
        CREATE VIRTUAL TABLE chunks_fts_trigram USING fts5(
            content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED,
            tokenize='trigram'
        );
        """
    )
    conn.execute(
        """
        INSERT INTO chunks(id, content, summary, tags, resolved_query, key_facts, resolved_queries, created_at)
        VALUES ('c1', 'searchable abcdef memory', '', '', '', '', '', '2026-01-01')
        """
    )
    conn.execute(
        """
        INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
        SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id FROM chunks
        """
    )
    conn.execute(
        """
        INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
        SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id FROM chunks
        """
    )
    conn.commit()
    conn.close()

    result = migrate_fts_single_trigram(db_path)

    checked = sqlite3.connect(db_path)
    schema = checked.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'").fetchone()[0]
    trigram_table = checked.execute("SELECT 1 FROM sqlite_master WHERE name = 'chunks_fts_trigram'").fetchone()
    fts_count = checked.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    meta_mode = checked.execute("SELECT value FROM brainlayer_meta WHERE key = 'fts_mode'").fetchone()[0]
    checked.close()

    assert result.chunk_count == 1
    assert result.fts_count == 1
    assert "tokenize='trigram'" in schema
    assert trigram_table is None
    assert fts_count == 1
    assert meta_mode == "single_trigram"


def test_migrate_fts_compact_dual_preserves_trigram_table(tmp_path):
    from brainlayer.db_shrink import migrate_fts_compact_dual

    db_path = tmp_path / "fts-dual.db"
    conn = sqlite3.connect(db_path)
    _init_chunks(conn)
    conn.executescript(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED,
            prefix='2 3 4', tokenize='unicode61 remove_diacritics 2'
        );
        CREATE VIRTUAL TABLE chunks_fts_trigram USING fts5(
            content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED,
            tokenize='trigram'
        );
        """
    )
    conn.execute(
        """
        INSERT INTO chunks(id, content, summary, tags, resolved_query, key_facts, resolved_queries, created_at)
        VALUES ('c1', 'searchable abcdef memory', '', '', '', '', '', '2026-01-01')
        """
    )
    conn.commit()
    conn.close()

    result = migrate_fts_compact_dual(db_path)

    checked = sqlite3.connect(db_path)
    schema = checked.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'").fetchone()[0]
    trigram_schema = checked.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts_trigram'").fetchone()[0]
    counts = checked.execute(
        "SELECT (SELECT COUNT(*) FROM chunks_fts), (SELECT COUNT(*) FROM chunks_fts_trigram)"
    ).fetchone()
    checked.close()

    assert result.mode == "compact_dual"
    assert "prefix=" not in schema
    assert "tokenize='trigram'" in trigram_schema
    assert counts == (1, 1)


def test_dedupe_deletes_duplicates_and_repoints_refs(tmp_path):
    from brainlayer.db_shrink import apply_content_dedup

    db_path = tmp_path / "dedup.db"
    conn = sqlite3.connect(db_path)
    _init_chunks(conn)
    conn.executescript(
        """
        CREATE TABLE chunk_id_alias (
            old_chunk_id TEXT PRIMARY KEY,
            canonical_chunk_id TEXT NOT NULL,
            deprecated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE dedupe_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id_dropped TEXT NOT NULL,
            chunk_id_kept TEXT NOT NULL,
            mechanism TEXT NOT NULL,
            hamming_distance INTEGER,
            ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE chunk_tags (
            chunk_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (chunk_id, tag)
        );
        CREATE TABLE kg_entity_chunks (
            entity_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            relevance REAL,
            context TEXT,
            PRIMARY KEY (entity_id, chunk_id)
        );
        CREATE TABLE chunk_vectors (
            chunk_id TEXT PRIMARY KEY,
            embedding BLOB
        );
        CREATE TABLE correction_pairs (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT
        );
        CREATE TABLE file_interactions (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT
        );
        CREATE TABLE kg_relations (
            id TEXT PRIMARY KEY,
            source_chunk_id TEXT
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED
        );
        """
    )
    rows = [
        ("qrel-id", "Duplicate content for the eval target", "2026-01-02"),
        ("dupe-id", "duplicate   content for the eval target", "2026-01-01"),
        ("old-id", "Another duplicate memory", "2026-01-01"),
        ("new-id", "another duplicate memory", "2026-01-03"),
        ("unique-id", "Unique memory", "2026-01-04"),
    ]
    conn.executemany(
        """
        INSERT INTO chunks(id, content, summary, tags, resolved_query, key_facts, resolved_queries, created_at)
        VALUES (?, ?, '', '', '', '', '', ?)
        """,
        rows,
    )
    conn.executemany(
        "INSERT INTO chunk_tags(chunk_id, tag) VALUES (?, ?)", [("dupe-id", "dupe-tag"), ("qrel-id", "qrel-tag")]
    )
    conn.execute(
        "INSERT INTO kg_entity_chunks(entity_id, chunk_id, relevance, context) VALUES ('e1', 'dupe-id', 0.7, 'ctx')"
    )
    conn.execute("INSERT INTO chunk_vectors(chunk_id, embedding) VALUES ('dupe-id', X'00')")
    conn.execute("INSERT INTO correction_pairs(id, chunk_id) VALUES (1, 'dupe-id')")
    conn.execute("INSERT INTO file_interactions(id, chunk_id) VALUES (1, 'dupe-id')")
    conn.execute("INSERT INTO kg_relations(id, source_chunk_id) VALUES ('r1', 'dupe-id')")
    conn.commit()
    conn.close()

    result = apply_content_dedup(db_path, protected_chunk_ids={"qrel-id"}, batch_size=2)

    checked = sqlite3.connect(db_path)
    chunk_ids = {row[0] for row in checked.execute("SELECT id FROM chunks")}
    alias = dict(checked.execute("SELECT old_chunk_id, canonical_chunk_id FROM chunk_id_alias"))
    tags = set(checked.execute("SELECT chunk_id, tag FROM chunk_tags"))
    entity_link = checked.execute("SELECT chunk_id FROM kg_entity_chunks WHERE entity_id = 'e1'").fetchone()[0]
    correction = checked.execute("SELECT chunk_id FROM correction_pairs WHERE id = 1").fetchone()[0]
    interaction = checked.execute("SELECT chunk_id FROM file_interactions WHERE id = 1").fetchone()[0]
    relation = checked.execute("SELECT source_chunk_id FROM kg_relations WHERE id = 'r1'").fetchone()[0]
    vector = checked.execute("SELECT 1 FROM chunk_vectors WHERE chunk_id = 'dupe-id'").fetchone()
    checked.close()

    assert result.duplicate_rows == 2
    assert result.deleted_rows == 2
    assert "dupe-id" not in chunk_ids
    assert "new-id" not in chunk_ids
    assert "qrel-id" in chunk_ids
    assert "old-id" in chunk_ids
    assert alias["dupe-id"] == "qrel-id"
    assert alias["new-id"] == "old-id"
    assert ("qrel-id", "dupe-tag") in tags
    assert entity_link == "qrel-id"
    assert correction == "qrel-id"
    assert interaction == "qrel-id"
    assert relation == "qrel-id"
    assert vector is None


def test_dedupe_without_fts_rebuild_preserves_fts_consistency(tmp_path):
    from brainlayer.db_shrink import apply_content_dedup

    db_path = tmp_path / "dedup-skip-fts.db"
    conn = sqlite3.connect(db_path)
    _init_chunks(conn)
    conn.executescript(
        """
        CREATE TABLE chunk_id_alias (
            old_chunk_id TEXT PRIMARY KEY,
            canonical_chunk_id TEXT NOT NULL,
            deprecated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE dedupe_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id_dropped TEXT NOT NULL,
            chunk_id_kept TEXT NOT NULL,
            mechanism TEXT NOT NULL,
            hamming_distance INTEGER,
            ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED
        );
        CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
            VALUES (
                new.content,
                new.summary,
                new.tags,
                new.resolved_query,
                new.key_facts,
                new.resolved_queries,
                new.id
            );
            INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid, trigram_rowid)
            VALUES (new.id, last_insert_rowid(), NULL)
            ON CONFLICT(chunk_id) DO UPDATE SET
                fts_rowid = excluded.fts_rowid,
                trigram_rowid = NULL;
        END;
        CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
            DELETE FROM chunks_fts
            WHERE rowid = (SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
            DELETE FROM chunk_fts_rowids WHERE chunk_id = old.id;
        END;
        """
    )
    rows = [
        ("canonical-id", "Duplicate content that should collapse", "2026-01-01"),
        ("dupe-id", "duplicate   content that should collapse", "2026-01-02"),
    ]
    conn.executemany(
        """
        INSERT INTO chunks(id, content, summary, tags, resolved_query, key_facts, resolved_queries, created_at)
        VALUES (?, ?, '', '', '', '', '', ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()

    result = apply_content_dedup(db_path, rebuild_fts=False)

    checked = sqlite3.connect(db_path)
    remaining_fts_ids = {
        row[0] for row in checked.execute("SELECT chunk_id FROM chunks_fts WHERE chunk_id IS NOT NULL")
    }
    delete_trigger = checked.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = 'chunks_fts_delete'"
    ).fetchone()
    checked.execute(
        """
        INSERT INTO chunks(id, content, summary, tags, resolved_query, key_facts, resolved_queries, created_at)
        VALUES ('fresh-id', 'Fresh searchable note', '', '', '', '', '', '2026-01-03')
        """
    )
    fresh_fts = checked.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'fresh-id'").fetchone()[0]
    checked.close()

    assert result.deleted_rows == 1
    assert remaining_fts_ids == {"canonical-id"}
    assert delete_trigger is not None
    assert fresh_fts == 1
