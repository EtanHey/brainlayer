"""Repair (f): FTS/vector completeness census, repair, rollback, write-path guarantee."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainlayer.chunk_write import insert_canonical_chunk  # noqa: E402
from brainlayer.index_completeness import (  # noqa: E402
    PREIMAGE_TABLE,
    census,
    repair_index_completeness,
    rollback_repair,
    route_of,
)
from brainlayer.vector_store import VectorStore  # noqa: E402

GIT_SHA = "0" * 40


@pytest.fixture
def store(tmp_path: Path):
    instance = VectorStore(tmp_path / "completeness.db")
    yield instance
    instance.close()


def _add(
    store: VectorStore, chunk_id: str, *, content: str = "alpha beta gamma delta", content_class: str = "knowledge"
) -> None:
    cursor = store.conn.cursor()
    insert_canonical_chunk(
        cursor,
        {"id": chunk_id, "content": content, "source": "test", "content_class": content_class},
    )


def _sqlite(store: VectorStore) -> sqlite3.Connection:
    """A stdlib connection on the same file; the migration uses sqlite3, not apsw."""
    return sqlite3.connect(store.db_path)


def _fts_snapshot(store: VectorStore) -> dict[str, list[tuple]]:
    """Rowid-level state of every lexical index plus the pointer table."""
    conn = _sqlite(store)
    try:
        snapshot = {
            table: sorted(conn.execute(f"SELECT id, c6, c0 FROM {table}_content"))
            for table in ("chunks_fts", "chunks_fts_trigram", "chunks_fts_operational")
        }
        snapshot["pointers"] = sorted(
            conn.execute("SELECT chunk_id, fts_rowid, trigram_rowid, operational_rowid FROM chunk_fts_rowids")
        )
        return snapshot
    finally:
        conn.close()


# ── the write path must produce lexical rows synchronously ───────────────────


@pytest.mark.parametrize(
    "content_class,expected_tables",
    [
        ("knowledge", ("chunks_fts", "chunks_fts_trigram")),
        ("decision", ("chunks_fts", "chunks_fts_trigram")),
        (None, ("chunks_fts", "chunks_fts_trigram")),
        ("operational", ("chunks_fts_operational",)),
        ("test", ()),
        ("cold", ()),
        ("benchmark", ()),
    ],
)
def test_canonical_insert_indexes_synchronously(store: VectorStore, content_class, expected_tables):
    """A canonical insert leaves the routed FTS rows in place before it returns.

    No queue, no catch-up job: if this ever regresses to asynchronous indexing,
    a chunk is unfindable between the insert and whenever the catch-up runs.
    """
    chunk_id = f"c-{content_class or 'null'}"
    _add(store, chunk_id, content="synchronous indexing guarantee", content_class=content_class)
    cursor = store.conn.cursor()
    for table in ("chunks_fts", "chunks_fts_trigram", "chunks_fts_operational"):
        rows = cursor.execute(f"SELECT COUNT(*) FROM {table}_content WHERE c6 = ?", (chunk_id,)).fetchone()[0]
        assert rows == (1 if table in expected_tables else 0), f"{table} rows for {content_class}"
    pointer = cursor.execute(
        "SELECT fts_rowid, trigram_rowid, operational_rowid FROM chunk_fts_rowids WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    if expected_tables:
        assert pointer is not None
        for table, value in zip(("chunks_fts", "chunks_fts_trigram", "chunks_fts_operational"), pointer):
            if table not in expected_tables:
                continue
            owner = cursor.execute(f"SELECT c6 FROM {table}_content WHERE id = ?", (value,)).fetchone()
            assert owner is not None and owner[0] == chunk_id, f"{table} pointer must resolve to this chunk"


def test_write_path_leaves_no_completeness_gap(store: VectorStore):
    """Bulk inserts through the canonical writer produce a clean census.

    This is the regression fence for the drift repair (f) cleans up: the census
    that finds 3,536 missing rows on the historical DB must find zero on a DB
    written only by today's write path.
    """
    for index in range(300):
        _add(store, f"bulk-{index}", content=f"payload number {index} " + "lorem ipsum " * 20)
    for index in range(20):
        _add(store, f"ops-{index}", content=f"operational note {index}", content_class="operational")
    store.conn.cursor().execute("UPDATE chunks SET content_class = 'operational' WHERE id = 'bulk-1'")
    store.conn.cursor().execute("UPDATE chunks SET content = content || ' edited' WHERE id = 'bulk-2'")
    store.conn.cursor().execute("UPDATE chunks SET summary = 'enriched' WHERE id = 'bulk-3'")

    conn = _sqlite(store)
    try:
        result = census(conn)
    finally:
        conn.close()
    assert result.missing_index_rows == {}
    assert result.duplicate_index_rows == {}
    assert result.misrouted_index_rows == {}
    assert result.mismatched_pointers == {}
    assert result.dangling_pointers == {}


def test_vector_debt_lands_in_the_embed_queue(store: VectorStore):
    """A chunk written without a vector is visible to the pipeline that owns embedding.

    The migration never embeds, so the only thing that makes this safe is that
    `reembed_backfill` finds the row by the same LEFT JOIN the census uses.
    """
    from brainlayer.reembed_backfill import count_unvectored_chunks, fetch_unvectored_batch

    _add(store, "unvectored-1", content="needs an embedding")
    conn = _sqlite(store)
    try:
        gap = census(conn)
    finally:
        conn.close()
    assert "unvectored-1" in gap.missing_vector_rowid
    assert count_unvectored_chunks(store) >= 1
    assert any(pending.chunk_id == "unvectored-1" for pending in fetch_unvectored_batch(store, batch_size=50))


def test_route_of_matches_the_trigger_predicates():
    """The migration's routing law is the write path's routing law."""
    from brainlayer import vector_store as vector_store_module

    source = Path(vector_store_module.__file__).read_text()
    assert "NOT IN ('operational', 'test', 'benchmark', 'cold')" in source
    assert route_of("operational") == "operational"
    for cls in ("test", "benchmark", "cold"):
        assert route_of(cls) == "none"
    for cls in (None, "knowledge", "decision", "anything-else"):
        assert route_of(cls) == "knowledge"


# ── census sees both directions ──────────────────────────────────────────────


def test_census_finds_a_missing_row_that_counts_hide(store: VectorStore):
    """A surplus row must not mask a missing one -- the #722 lesson, in a test."""
    _add(store, "present", content="findable text")
    _add(store, "vanished", content="unfindable text")
    cursor = store.conn.cursor()
    rowid = cursor.execute("SELECT id FROM chunks_fts_content WHERE c6 = 'vanished'").fetchone()[0]
    cursor.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rowid,))
    # counterweight: a duplicate row for the other chunk, so COUNT(*) balances
    cursor.execute(
        "INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id) "
        "VALUES ('findable text', NULL, NULL, NULL, NULL, NULL, 'present')"
    )
    knowledge_chunks = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    fts_rows = cursor.execute("SELECT COUNT(*) FROM chunks_fts_content").fetchone()[0]
    assert knowledge_chunks == fts_rows, "counts agree, which is exactly why counts are not enough"

    conn = _sqlite(store)
    try:
        result = census(conn)
    finally:
        conn.close()
    assert result.missing_index_rows["chunks_fts"] == ["vanished"]
    assert result.duplicate_index_rows["chunks_fts"] == ["present"]


def test_census_finds_orphans_and_bad_pointers(store: VectorStore):
    _add(store, "alive", content="alive text")
    _add(store, "doomed", content="doomed text")
    cursor = store.conn.cursor()
    doomed_rowid = cursor.execute("SELECT id FROM chunks_fts_content WHERE c6 = 'doomed'").fetchone()[0]
    # orphan the index row without firing the delete trigger
    cursor.execute("DROP TRIGGER chunks_fts_delete")
    cursor.execute("DROP TRIGGER chunks_fts_trigram_delete")
    cursor.execute("DELETE FROM chunks WHERE id = 'doomed'")
    # aim alive's pointer at the orphan row, and add a dangling pointer
    cursor.execute("UPDATE chunk_fts_rowids SET fts_rowid = ? WHERE chunk_id = 'alive'", (doomed_rowid,))
    cursor.execute(
        "INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid) VALUES ('ghost', 999999) "
        "ON CONFLICT(chunk_id) DO UPDATE SET fts_rowid = excluded.fts_rowid"
    )

    conn = _sqlite(store)
    try:
        result = census(conn)
    finally:
        conn.close()
    assert result.orphan_index_rows["chunks_fts"] == [doomed_rowid]
    assert result.mismatched_pointers["chunks_fts"] == ["alive"]
    assert result.dangling_pointers["chunks_fts"] == ["ghost"]
    assert result.orphan_pointer_rows == ["doomed", "ghost"]  # doomed's own pointer is orphaned too
    assert result.aux_to_chunk_total >= 4


def test_census_flags_a_misrouted_row(store: VectorStore):
    _add(store, "leaky", content="operational content", content_class="knowledge")
    store.conn.cursor().execute("UPDATE chunks SET content_class = 'test' WHERE id = 'leaky'")
    store.conn.cursor().execute(
        "INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id) "
        "VALUES ('operational content', NULL, NULL, NULL, NULL, NULL, 'leaky')"
    )
    conn = _sqlite(store)
    try:
        result = census(conn)
    finally:
        conn.close()
    assert "leaky" in result.misrouted_index_rows["chunks_fts"]


# ── repair ───────────────────────────────────────────────────────────────────


def _damage(store: VectorStore) -> None:
    cursor = store.conn.cursor()
    _add(store, "gap", content="gap chunk content")
    _add(store, "dupe", content="dupe chunk content")
    _add(store, "leak", content="leak chunk content")
    _add(store, "clean", content="clean chunk content")
    rowid = cursor.execute("SELECT id FROM chunks_fts_content WHERE c6 = 'gap'").fetchone()[0]
    cursor.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rowid,))
    cursor.execute(
        "INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id) "
        "VALUES ('stale dupe text', NULL, NULL, NULL, NULL, NULL, 'dupe')"
    )
    cursor.execute("UPDATE chunks SET content_class = 'cold' WHERE id = 'leak'")


def test_dry_run_writes_nothing(store: VectorStore, tmp_path: Path):
    _damage(store)
    conn = _sqlite(store)
    try:
        before = census(conn).summary()
    finally:
        conn.close()
    result = repair_index_completeness(store.db_path, git_sha=GIT_SHA, apply=False)
    assert result.apply is False
    assert result.inserted_rows == 0 and result.deleted_rows == 0
    conn = _sqlite(store)
    try:
        assert census(conn).summary() == before
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert PREIMAGE_TABLE not in tables
    finally:
        conn.close()


def test_repair_closes_every_gap_and_spot_checks_values(store: VectorStore):
    _damage(store)
    result = repair_index_completeness(store.db_path, git_sha=GIT_SHA, apply=True, spot_check=4)
    assert result.census_after["lexical_total"] == 0, result.census_after
    # the residue is vector debt, which a migration must never close by embedding
    assert result.census_after["vector_total"] == result.census_before["vector_total"]
    assert result.spot_checks and all(check["ok"] for check in result.spot_checks)

    conn = _sqlite(store)
    try:
        # VALUE check, not a count: the repaired row carries the chunk's own text
        indexed = conn.execute("SELECT c0 FROM chunks_fts_content WHERE c6 = 'gap'").fetchone()[0]
        content = conn.execute("SELECT content FROM chunks WHERE id = 'gap'").fetchone()[0]
        assert indexed == content
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts_content WHERE c6 = 'dupe'").fetchone()[0] == 1
        stale = conn.execute(
            "SELECT COUNT(*) FROM chunks_fts_content WHERE c6 = 'dupe' AND c0 = 'stale dupe text'"
        ).fetchone()[0]
        assert stale == 0, "the stale duplicate text must be gone, not merely deduplicated by count"
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts_content WHERE c6 = 'leak'").fetchone()[0] == 0
        # the migration never touches chunk data
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 4
    finally:
        conn.close()


def test_repair_makes_the_chunk_findable_by_a_real_query(store: VectorStore):
    """Serving proof: the repaired chunk comes back from an actual MATCH."""
    _add(store, "hidden", content="quokka telemetry manifest")
    cursor = store.conn.cursor()
    rowid = cursor.execute("SELECT id FROM chunks_fts_content WHERE c6 = 'hidden'").fetchone()[0]
    cursor.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rowid,))
    assert cursor.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'quokka'").fetchone()[0] == 0

    repair_index_completeness(store.db_path, git_sha=GIT_SHA, apply=True)

    conn = _sqlite(store)
    try:
        hits = conn.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'quokka'").fetchall()
    finally:
        conn.close()
    assert [row[0] for row in hits] == ["hidden"]


def test_repair_never_embeds_and_reports_vector_debt(store: VectorStore):
    _add(store, "novec", content="no vector for this one")
    result = repair_index_completeness(store.db_path, git_sha=GIT_SHA, apply=True)
    assert result.vector_debt["missing_vector_rowid"] >= 1
    assert result.vector_debt["owner"] == "brainlayer reembed-backfill"
    conn = _sqlite(store)
    try:
        # the migration wrote no vector rows of its own
        assert conn.execute("SELECT COUNT(*) FROM chunk_vectors_rowids").fetchone()[0] == 0
        actions = {row[0] for row in conn.execute(f"SELECT DISTINCT table_name FROM {PREIMAGE_TABLE}")}
    finally:
        conn.close()
    assert not any(name.startswith("chunk_vectors") for name in actions)


def test_repair_skips_a_row_that_changed_owner_under_it(store: VectorStore):
    """A rowid recorded by the census is re-verified before any delete."""
    from brainlayer import index_completeness

    _add(store, "victim", content="victim content")
    conn = _sqlite(store)
    try:
        rowid = conn.execute("SELECT id FROM chunks_fts_content WHERE c6 = 'victim'").fetchone()[0]
        index_completeness._ensure_preimage(conn)
        assert (
            index_completeness._delete_index_row(conn, "chunks_fts", rowid, "someone-else", "rebuild", "run-test")
            is False
        )
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts_content WHERE c6 = 'victim'").fetchone()[0] == 1
    finally:
        conn.close()


def test_rollback_restores_the_previous_index_state(store: VectorStore):
    _damage(store)
    conn = _sqlite(store)
    try:
        before = sorted((str(row[0]), str(row[1])) for row in conn.execute("SELECT c6, c0 FROM chunks_fts_content"))
        pointers_before = sorted(
            conn.execute("SELECT chunk_id, fts_rowid, trigram_rowid, operational_rowid FROM chunk_fts_rowids")
        )
    finally:
        conn.close()

    repair_index_completeness(store.db_path, git_sha=GIT_SHA, apply=True)
    stats = rollback_repair(store.db_path)
    assert stats["restored_rows"] > 0 and stats["removed_rows"] > 0

    conn = _sqlite(store)
    try:
        after = sorted((str(row[0]), str(row[1])) for row in conn.execute("SELECT c6, c0 FROM chunks_fts_content"))
        pointers_after = sorted(
            conn.execute("SELECT chunk_id, fts_rowid, trigram_rowid, operational_rowid FROM chunk_fts_rowids")
        )
    finally:
        conn.close()
    assert after == before, "rollback restores indexed VALUES, not just row counts"
    assert pointers_after == pointers_before


def test_rollback_restores_rows_at_their_original_rowids(store: VectorStore):
    """A restored pointer must still resolve; that requires the original rowid back.

    Restoring the text at a fresh rowid looks identical by content and by count,
    and leaves every restored pointer dangling.
    """

    def snapshot() -> tuple[dict[str, list[tuple]], int]:
        conn = _sqlite(store)
        try:
            rows = {
                table: sorted(conn.execute(f"SELECT id, c6 FROM {table}_content"))
                for table in ("chunks_fts", "chunks_fts_trigram", "chunks_fts_operational")
            }
            dangling = conn.execute(
                """
                SELECT COUNT(*) FROM chunk_fts_rowids p
                WHERE p.fts_rowid IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM chunks_fts_content c WHERE c.id = p.fts_rowid)
                """
            ).fetchone()[0]
        finally:
            conn.close()
        return rows, dangling

    _damage(store)
    before_rows, before_dangling = snapshot()

    repair_index_completeness(store.db_path, git_sha=GIT_SHA, apply=True)
    _, repaired_dangling = snapshot()
    assert repaired_dangling == 0, "the repair itself must leave no dangling pointer"

    rollback_repair(store.db_path)
    after_rows, after_dangling = snapshot()

    assert after_rows == before_rows, "every row must come back at the rowid it had"
    # The pre-state's own dangling pointer is part of the damage and must come
    # back too: a rollback restores the previous state, not a better one.
    assert after_dangling == before_dangling


def test_migration_receipt_carries_the_commit(store: VectorStore):
    """The canonical schema_migrations has no git_sha column, so details must carry it."""
    from brainlayer.index_completeness import MIGRATION_NAME

    _damage(store)
    sha = "a" * 40
    repair_index_completeness(store.db_path, git_sha=sha, apply=True, actor="repair-f-test")

    conn = _sqlite(store)
    try:
        row = conn.execute(
            "SELECT applied_at, details FROM schema_migrations WHERE name = ?", (MIGRATION_NAME,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "an applied migration must leave a receipt"
    details = json.loads(row[1])
    assert details["git_sha"] == sha
    assert details["actor"] == "repair-f-test"
    assert details["inserted_rows"] > 0


def test_repair_receipt_survives_a_rerun(store: VectorStore):
    """`name` is the primary key; a second run updates the receipt, never aborts."""
    _damage(store)
    repair_index_completeness(store.db_path, git_sha=GIT_SHA, apply=True)
    again = repair_index_completeness(store.db_path, git_sha=GIT_SHA, apply=True)
    assert again.census_after["lexical_total"] == 0


def test_repair_leaves_lifecycle_rows_the_census_did_not_report(store: VectorStore):
    """A migration must not delete more than its census reported.

    The misroute check is scoped to active chunks. An archived chunk whose class
    routes elsewhere is therefore never reported -- and must not be silently
    unindexed just because a pointer defect pulled it into the work set.
    """
    _add(store, "retired", content="retired but still indexed")
    cursor = store.conn.cursor()
    cursor.execute("UPDATE chunks SET content_class = 'cold' WHERE id = 'retired'")
    cursor.execute("UPDATE chunks SET archived_at = '2026-01-01T00:00:00Z' WHERE id = 'retired'")
    # put a row back in the knowledge index and aim a broken pointer at nothing
    cursor.execute(
        "INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id) "
        "VALUES ('retired but still indexed', NULL, NULL, NULL, NULL, NULL, 'retired')"
    )
    cursor.execute(
        "INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid) VALUES ('retired', 987654) "
        "ON CONFLICT(chunk_id) DO UPDATE SET fts_rowid = excluded.fts_rowid"
    )

    conn = _sqlite(store)
    try:
        reported = census(conn)
    finally:
        conn.close()
    assert "retired" not in reported.misrouted_index_rows.get("chunks_fts", [])
    assert "retired" in reported.dangling_pointers["chunks_fts"]

    repair_index_completeness(store.db_path, git_sha=GIT_SHA, apply=True)

    conn = _sqlite(store)
    try:
        rows = conn.execute("SELECT id FROM chunks_fts_content WHERE c6 = 'retired'").fetchall()
        pointer = conn.execute("SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = 'retired'").fetchone()[0]
    finally:
        conn.close()
    assert len(rows) == 1, "the unreported row must survive"
    assert pointer == rows[0][0], "and the pointer must resolve to it"


def test_apply_rollback_apply_rollback_cycle(store: VectorStore):
    """Preimages accumulate, so rollback must be scoped to ONE run.

    Replaying the whole table on the second rollback tries to undo an apply that
    was already undone and dies on a rowid collision -- which is how this was
    found, on the canonical-size copy.
    """
    _damage(store)
    snapshot = _fts_snapshot(store)

    first = repair_index_completeness(store.db_path, git_sha=GIT_SHA, apply=True)
    rollback_repair(store.db_path)
    assert _fts_snapshot(store) == snapshot

    second = repair_index_completeness(store.db_path, git_sha=GIT_SHA, apply=True)
    assert second.run_id != first.run_id
    stats = rollback_repair(store.db_path)
    assert stats["run_id"] == second.run_id
    assert _fts_snapshot(store) == snapshot, "the second rollback must land on the same state"

    # Both runs are now reversed, so a further rollback is a no-op rather than a
    # replay of an apply that was already undone.
    third = rollback_repair(store.db_path)
    assert third["run_id"] is None
    assert third["restored_rows"] == 0 and third["removed_rows"] == 0
    assert _fts_snapshot(store) == snapshot


def test_repair_refuses_the_live_db(tmp_path: Path, monkeypatch):
    from brainlayer import chunk_origin_wipe

    live = tmp_path / "brainlayer.db"
    sqlite3.connect(live).close()
    monkeypatch.setattr(chunk_origin_wipe, "_live_db_candidates", lambda: (live,))
    with pytest.raises(RuntimeError, match="refusing to write the live"):
        repair_index_completeness(live, git_sha=GIT_SHA, apply=True)


def test_orphan_deletes_only_index_rows(store: VectorStore):
    _add(store, "ghosted", content="ghosted content")
    cursor = store.conn.cursor()
    cursor.execute("DROP TRIGGER chunks_fts_delete")
    cursor.execute("DROP TRIGGER chunks_fts_trigram_delete")
    cursor.execute("DELETE FROM chunks WHERE id = 'ghosted'")
    _add(store, "kept", content="kept content")

    result = repair_index_completeness(store.db_path, git_sha=GIT_SHA, apply=True)
    assert result.orphans_deleted >= 1

    conn = _sqlite(store)
    try:
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts_content WHERE c6 = 'ghosted'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunks WHERE id = 'kept'").fetchone()[0] == 1
        payloads = conn.execute(f"SELECT payload FROM {PREIMAGE_TABLE} WHERE action = 'orphan'").fetchall()
    finally:
        conn.close()
    assert payloads and json.loads(payloads[0][0])[6] == "ghosted", "orphan deletes are preimaged"
