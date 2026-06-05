"""Tests for kg_cleanup — KG phase-1 apply/rollback executor.

Fixture DB only. Mirrors the live kg_* schema minimally (no FTS/triggers).
"""

import sqlite3

import pytest

from brainlayer.kg_cleanup import (
    DeviationError,
    apply_merges,
    apply_prune,
    ensure_log_table,
    rollback,
    select_prune,
)


@pytest.fixture
def con(tmp_path):
    con = sqlite3.connect(tmp_path / "kg_fixture.db")
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE kg_entities (
            id TEXT PRIMARY KEY, entity_type TEXT NOT NULL, name TEXT NOT NULL,
            user_verified INTEGER DEFAULT 0, importance REAL DEFAULT 0.5,
            status TEXT DEFAULT 'active', parent_id TEXT,
            UNIQUE(entity_type, name));
        CREATE TABLE kg_entity_aliases (
            alias TEXT NOT NULL, entity_id TEXT NOT NULL,
            alias_type TEXT DEFAULT 'name', created_at TEXT,
            PRIMARY KEY (alias, entity_id));
        CREATE TABLE kg_relations (
            id TEXT PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            UNIQUE(source_id, target_id, relation_type));
        CREATE TABLE kg_entity_chunks (
            entity_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
            relevance REAL DEFAULT 1.0,
            PRIMARY KEY (entity_id, chunk_id));
        """
    )
    rows = [
        # deterministic junk
        ("e1", "person", "PERSON_9c53c074", 0, 0.5, "active"),
        ("e2", "person", "[PERSON_c6278be6]", 0, 0.5, "active"),
        ("e3", "concept", "surface:42", 0, 0.5, "active"),
        ("e4", "concept", "R81", 0, 0.5, "active"),
        # junk-named but PROTECTED
        ("e5", "person", "PERSON_deadbeef", 1, 0.5, "active"),
        ("e6", "concept", "PR #99", 0, 0.9, "active"),
        # real entity, untouched
        ("e7", "person", "Etan Heyman", 0, 0.6, "active"),
        # merge cluster: canonical + two variants
        ("c1", "tool", "Codex", 0, 0.6, "active"),
        ("c2", "concept", "codex", 0, 0.5, "active"),
        ("c3", "person", "@codex", 0, 0.5, "active"),
    ]
    con.executemany(
        "INSERT INTO kg_entities (id, entity_type, name, user_verified, importance, status) VALUES (?,?,?,?,?,?)",
        rows,
    )
    con.executemany(
        "INSERT INTO kg_entity_chunks (entity_id, chunk_id) VALUES (?,?)",
        [("c1", "ch1"), ("c1", "ch2"), ("c2", "ch3"), ("c3", "ch1")],
    )
    con.execute(
        "INSERT INTO kg_relations (id, source_id, target_id, relation_type) VALUES ('r1', 'c2', 'e7', 'mentioned_by')"
    )
    con.commit()
    return con


CLUSTER = {
    "stem": "codex",
    "canonical": {"id": "c1", "name": "Codex", "type": "tool"},
    "members": [
        {"id": "c2", "name": "codex", "type": "concept"},
        {"id": "c3", "name": "@codex", "type": "person"},
    ],
}


def test_select_prune_matches_junk_families_only(con):
    rows = select_prune(con)
    names = {r["name"] for r in rows}
    assert names == {"PERSON_9c53c074", "[PERSON_c6278be6]", "surface:42", "R81"}


def test_select_prune_never_returns_protected(con):
    names = {r["name"] for r in select_prune(con)}
    assert "PERSON_deadbeef" not in names  # user_verified=1
    assert "PR #99" not in names  # importance >= 0.7


def test_apply_prune_archives_and_logs(con):
    ensure_log_table(con)
    rows = select_prune(con)
    n = apply_prune(con, rows, run_id="test-run", expected=4)
    assert n == 4
    statuses = dict(con.execute("SELECT id, status FROM kg_entities WHERE id IN ('e1','e2','e3','e4')").fetchall())
    assert set(statuses.values()) == {"archived"}
    # nothing hard-deleted
    assert con.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0] == 10
    log = con.execute("SELECT COUNT(*) FROM kg_cleanup_log WHERE run_id='test-run' AND action='prune'").fetchone()[0]
    assert log == 4


def test_apply_prune_deviation_guard(con):
    ensure_log_table(con)
    rows = select_prune(con)
    with pytest.raises(DeviationError):
        apply_prune(con, rows, run_id="test-run", expected=100)
    # guard aborts BEFORE any write
    active = con.execute("SELECT COUNT(*) FROM kg_entities WHERE status='active'").fetchone()[0]
    assert active == 10


def test_apply_merges_archives_losers_aliases_and_repoints(con):
    ensure_log_table(con)
    stats = apply_merges(con, [CLUSTER], run_id="test-run")
    assert stats["clusters"] == 1
    assert stats["rows_merged_away"] == 2
    # losers archived with parent_id -> canonical
    for loser in ("c2", "c3"):
        row = con.execute("SELECT status, parent_id FROM kg_entities WHERE id=?", (loser,)).fetchone()
        assert row["status"] == "archived"
        assert row["parent_id"] == "c1"
    # canonical untouched
    assert con.execute("SELECT status FROM kg_entities WHERE id='c1'").fetchone()["status"] == "active"
    # variant names resolvable via aliases
    aliases = {r["alias"] for r in con.execute("SELECT alias FROM kg_entity_aliases WHERE entity_id='c1'")}
    assert {"codex", "@codex"} <= aliases
    # chunks re-pointed to canonical (ch1 dedup-safe: c1 already has it)
    chunks = {r["chunk_id"] for r in con.execute("SELECT chunk_id FROM kg_entity_chunks WHERE entity_id='c1'")}
    assert chunks == {"ch1", "ch2", "ch3"}
    assert con.execute("SELECT COUNT(*) FROM kg_entity_chunks WHERE entity_id IN ('c2','c3')").fetchone()[0] == 0
    # relations re-pointed
    assert con.execute("SELECT source_id FROM kg_relations WHERE id='r1'").fetchone()["source_id"] == "c1"


def test_rollback_restores_exact_state(con):
    ensure_log_table(con)
    before_entities = con.execute("SELECT id, status, parent_id FROM kg_entities ORDER BY id").fetchall()
    before_chunks = con.execute(
        "SELECT entity_id, chunk_id FROM kg_entity_chunks ORDER BY entity_id, chunk_id"
    ).fetchall()
    before_rels = con.execute("SELECT id, source_id, target_id FROM kg_relations ORDER BY id").fetchall()

    apply_prune(con, select_prune(con), run_id="test-run", expected=4)
    apply_merges(con, [CLUSTER], run_id="test-run")
    rollback(con, run_id="test-run")

    assert [tuple(r) for r in con.execute("SELECT id, status, parent_id FROM kg_entities ORDER BY id")] == [
        tuple(r) for r in before_entities
    ]
    assert [
        tuple(r) for r in con.execute("SELECT entity_id, chunk_id FROM kg_entity_chunks ORDER BY entity_id, chunk_id")
    ] == [tuple(r) for r in before_chunks]
    assert [tuple(r) for r in con.execute("SELECT id, source_id, target_id FROM kg_relations ORDER BY id")] == [
        tuple(r) for r in before_rels
    ]
    # merge aliases removed on rollback
    assert con.execute("SELECT COUNT(*) FROM kg_entity_aliases").fetchone()[0] == 0
