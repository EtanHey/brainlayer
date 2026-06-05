"""Tests for kg_cleanup — KG phase-1 apply/rollback executor.

Fixture DB only. Mirrors the live kg_* schema minimally (no FTS/triggers).
"""

import json
import sqlite3

import pytest

import scripts.kg_cleanup_apply as kg_apply
import scripts.kg_flag_batch as kg_flag_batch
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
    "category": "prefix-variants",
    "source": "explicit",
    "canonical": {"id": "c1", "name": "Codex", "type": "tool"},
    "members": [
        {"id": "c2", "name": "codex", "type": "concept"},
        {"id": "c3", "name": "@codex", "type": "person"},
    ],
}


def _decisions_path(tmp_path, *, schema="kg-flag-decisions-v1", merge=None, keep=None, counts=None):
    merge = [] if merge is None else merge
    keep = [] if keep is None else keep
    if counts is None:
        counts = {
            "merge_clusters": len(merge),
            "rows_merged_away": sum(len(c["members"]) for c in merge),
            "keep": len(keep),
            "explicit": sum(1 for c in merge + keep if c.get("source") == "explicit"),
            "by_rule": sum(1 for c in merge + keep if c.get("source") == "by_rule"),
        }
    doc = {
        "schema": schema,
        "source": "kg-phase1-flag-batch-2026-06-05",
        "rules": {"identical-name": "merge"},
        "per_category": {},
        "counts": counts,
        "merge": merge,
        "keep": keep,
    }
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _cluster():
    return json.loads(json.dumps(CLUSTER))


KEEP_DECISION = {"stem": "agent c", "category": "diagnosis-flag", "source": "explicit"}


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


def test_decisions_schema_validation_fails_loud(tmp_path):
    path = _decisions_path(tmp_path, schema="wrong-schema")

    with pytest.raises(ValueError, match="kg-flag-decisions-v1"):
        kg_apply.load_decisions(path)


def test_decisions_dry_run_prints_counts_and_writes_nothing(con, tmp_path, capsys):
    path = _decisions_path(tmp_path, merge=[_cluster()], keep=[KEEP_DECISION])
    before_changes = con.total_changes

    report = kg_apply.run_decisions(con, path, run_id="test-run", execute=False)

    assert report["dry_run"] is True
    out = capsys.readouterr().out
    assert '"merge_clusters": 1' in out
    assert '"rows_merged_away": 2' in out
    assert '"keep": 1' in out
    assert con.total_changes == before_changes
    assert con.execute("SELECT name FROM sqlite_master WHERE name='kg_cleanup_log'").fetchone() is None
    assert con.execute("SELECT COUNT(*) FROM kg_entities WHERE status='active'").fetchone()[0] == 10


def test_decisions_execute_reuses_merge_path(con, tmp_path):
    path = _decisions_path(tmp_path, merge=[_cluster()])

    report = kg_apply.run_decisions(con, path, run_id="test-run", execute=True)

    assert report["merge_stats"] == {"clusters": 1, "rows_merged_away": 2}
    assert report["skipped_changed_members"] == []
    assert tuple(con.execute("SELECT status, parent_id FROM kg_entities WHERE id='c2'").fetchone()) == (
        "archived",
        "c1",
    )
    assert tuple(con.execute("SELECT status, parent_id FROM kg_entities WHERE id='c3'").fetchone()) == (
        "archived",
        "c1",
    )
    assert con.execute("SELECT COUNT(*) FROM kg_cleanup_log WHERE action='merge'").fetchone()[0] == 2


def test_decisions_execute_logs_keep_separate_only(con, tmp_path):
    path = _decisions_path(tmp_path, keep=[KEEP_DECISION])

    report = kg_apply.run_decisions(con, path, run_id="test-run", execute=True)

    assert report["keep_logged"] == 1
    assert con.execute("SELECT COUNT(*) FROM kg_entities WHERE status='active'").fetchone()[0] == 10
    row = con.execute(
        "SELECT action, entity_id, canonical_id, mechanism, evidence, payload_json FROM kg_cleanup_log"
    ).fetchone()
    assert row["action"] == "keep_separate"
    assert row["entity_id"] == "cat:diagnosis-flag:stem:agent c"
    assert row["canonical_id"] is None
    assert row["mechanism"] == "flag_decision_keep_separate"
    assert row["evidence"] == "diagnosis-flag"
    assert json.loads(row["payload_json"])["decision"] == KEEP_DECISION


def test_decisions_deviation_aborts_before_writes(con, tmp_path):
    path = _decisions_path(
        tmp_path,
        merge=[_cluster()],
        counts={"merge_clusters": 1, "rows_merged_away": 100, "keep": 0, "explicit": 1, "by_rule": 0},
    )

    with pytest.raises(DeviationError, match="rows_merged_away"):
        kg_apply.run_decisions(con, path, run_id="test-run", execute=True)

    assert con.execute("SELECT COUNT(*) FROM kg_entities WHERE status='active'").fetchone()[0] == 10
    assert con.execute("SELECT name FROM sqlite_master WHERE name='kg_cleanup_log'").fetchone() is None


def test_decisions_execute_skips_changed_members_and_reports(con, tmp_path):
    con.execute("UPDATE kg_entities SET status='archived' WHERE id='c3'")
    con.commit()
    path = _decisions_path(
        tmp_path,
        merge=[_cluster()],
        counts={"merge_clusters": 1, "rows_merged_away": 1, "keep": 0, "explicit": 1, "by_rule": 0},
    )

    report = kg_apply.run_decisions(con, path, run_id="test-run", execute=True)

    assert report["merge_stats"] == {"clusters": 1, "rows_merged_away": 1}
    assert report["skipped_changed_members"] == [{"stem": "codex", "name": "@codex", "id": "c3", "status": "archived"}]
    assert tuple(con.execute("SELECT status, parent_id FROM kg_entities WHERE id='c2'").fetchone()) == (
        "archived",
        "c1",
    )
    assert con.execute("SELECT parent_id FROM kg_entities WHERE id='c3'").fetchone()["parent_id"] is None
    assert con.execute("SELECT COUNT(*) FROM kg_cleanup_log WHERE action='merge'").fetchone()[0] == 1


def test_rollback_removes_keep_separate_log_rows(con, tmp_path):
    path = _decisions_path(tmp_path, merge=[_cluster()], keep=[KEEP_DECISION])

    kg_apply.run_decisions(con, path, run_id="test-run", execute=True)
    counts = rollback(con, run_id="test-run")

    assert counts["keep_separate"] == 1
    assert counts["merge"] == 2
    assert con.execute("SELECT COUNT(*) FROM kg_cleanup_log WHERE run_id='test-run'").fetchone()[0] == 0
    assert tuple(con.execute("SELECT status, parent_id FROM kg_entities WHERE id='c2'").fetchone()) == ("active", None)
    assert tuple(con.execute("SELECT status, parent_id FROM kg_entities WHERE id='c3'").fetchone()) == ("active", None)


def test_flag_batch_skips_logged_keep_separate_clusters(con, tmp_path):
    path = _decisions_path(tmp_path, keep=[{"stem": "codex", "category": "prefix-variants", "source": "explicit"}])
    kg_apply.run_decisions(con, path, run_id="test-run", execute=True)

    keep_decisions = kg_flag_batch.load_keep_separate_decisions(con)

    assert kg_flag_batch.should_skip_keep_separate("prefix-variants", "codex", keep_decisions)
    assert not kg_flag_batch.should_skip_keep_separate("case-only", "codex", keep_decisions)
