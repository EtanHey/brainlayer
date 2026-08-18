"""Repair (f): FTS/vector completeness — census, invariant, and repair.

Every active chunk must be findable BOTH ways: the lexical row its content_class
routes to, and a semantic vector. Historical drift left gaps that count-based
health checks cannot see, because a surplus row hides a missing one.

Two rules shape this module.

**Counts are blind (#722, PR723).** `check_fts5_health` compares
`COUNT(chunks WHERE knowledge-class)` against `COUNT(chunks_fts)` and calls the
index synced when the two agree. On the canonical DB they nearly agree while
3,536 active chunks have no `chunks_fts` row at all -- 4,551 chunks carry a
duplicate row and 2,508 rows leak in from classes that route elsewhere, and the
surplus masks the deficit. So every check here is set-difference on chunk ids,
in BOTH directions, never a count comparison.

**Vectors are the embed pipeline's work, not a migration's.** Missing FTS rows
are deterministic from `chunks.content` and rebuild in place. A missing vector
is not: producing one means running the embedding model. This migration never
embeds. It reports vector debt and hands it to `brainlayer reembed-backfill`,
which already owns that queue (`reembed_backfill.fetch_unvectored_batch`).

**What may be deleted.** Only aux INDEX rows -- `chunks_fts`,
`chunks_fts_trigram`, `chunks_fts_operational` entries, and `chunk_fts_rowids`
pointers. Every one of them is reproducible byte-for-byte from `chunks`, which
is why a delete here is not data loss. No row in `chunks` is ever written,
archived, or deleted by this migration. An orphan aux row (its chunk_id absent
from `chunks`) is deleted only after that absence is re-verified inside the
write transaction.

The vec0 tables are never written: `chunk_vectors` and its shadow tables are
owned by sqlite-vec, and a phantom vector (rowid present, validity bit clear)
is reported for the embed pipeline rather than surgically patched here.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .chunk_origin_wipe import assert_not_live_db

MIGRATION_NAME = "repair_f_index_completeness"
PREIMAGE_TABLE = "index_completeness_preimage"

#: Mirrors the FTS trigger predicates in vector_store.py. Kept as literals here
#: so the migration reads the same routing law the write path enforces; the
#: contract test asserts the two never drift apart.
NO_FTS_CLASSES = ("test", "benchmark", "cold")
OPERATIONAL_CLASS = "operational"

KNOWLEDGE_ROUTE = "knowledge"
OPERATIONAL_ROUTE = "operational"
NO_FTS_ROUTE = "none"

#: index table -> the route whose chunks belong in it.
ROUTED_TABLES = {
    "chunks_fts": KNOWLEDGE_ROUTE,
    "chunks_fts_trigram": KNOWLEDGE_ROUTE,
    "chunks_fts_operational": OPERATIONAL_ROUTE,
}

#: chunk_fts_rowids pointer column per index table.
POINTER_COLUMN = {
    "chunks_fts": "fts_rowid",
    "chunks_fts_trigram": "trigram_rowid",
    "chunks_fts_operational": "operational_rowid",
}

FTS_INSERT_COLUMNS = (
    "content",
    "summary",
    "tags",
    "resolved_query",
    "key_facts",
    "resolved_queries",
    "chunk_id",
)

ACTIVE_CHUNK_SQL = """
    content IS NOT NULL
    AND content != ''
    AND archived_at IS NULL
    AND superseded_by IS NULL
    AND aggregated_into IS NULL
"""

VECTOR_DIM = 1024
VECTOR_BYTES = VECTOR_DIM * 4


def route_of(content_class: str | None) -> str:
    """Which FTS index a chunk's class routes it into."""
    cls = content_class or "knowledge"
    if cls == OPERATIONAL_CLASS:
        return OPERATIONAL_ROUTE
    if cls in NO_FTS_CLASSES:
        return NO_FTS_ROUTE
    return KNOWLEDGE_ROUTE


def tables_for_route(route: str) -> tuple[str, ...]:
    return tuple(table for table, wanted in ROUTED_TABLES.items() if wanted == route)


@dataclass
class CompletenessCensus:
    """Both-direction completeness, as chunk-id sets rather than counts."""

    # chunk -> aux
    missing_index_rows: dict[str, list[str]] = field(default_factory=dict)
    duplicate_index_rows: dict[str, list[str]] = field(default_factory=dict)
    misrouted_index_rows: dict[str, list[str]] = field(default_factory=dict)
    missing_vector_rowid: list[str] = field(default_factory=list)
    missing_binary_vector: list[str] = field(default_factory=list)
    # aux -> chunk
    orphan_index_rows: dict[str, list[int]] = field(default_factory=dict)
    orphan_vector_rowids: list[str] = field(default_factory=list)
    orphan_pointer_rows: list[str] = field(default_factory=list)
    dangling_pointers: dict[str, list[str]] = field(default_factory=dict)
    mismatched_pointers: dict[str, list[str]] = field(default_factory=dict)
    # vector payload (VALUE, not pointer)
    phantom_vectors: list[str] = field(default_factory=list)
    vector_payload_checked: int = 0

    @property
    def chunk_to_aux_total(self) -> int:
        return (
            sum(len(ids) for ids in self.missing_index_rows.values())
            + sum(len(ids) for ids in self.duplicate_index_rows.values())
            + sum(len(ids) for ids in self.misrouted_index_rows.values())
            + len(self.missing_vector_rowid)
        )

    @property
    def aux_to_chunk_total(self) -> int:
        return (
            sum(len(ids) for ids in self.orphan_index_rows.values())
            + len(self.orphan_vector_rowids)
            + len(self.orphan_pointer_rows)
            + sum(len(ids) for ids in self.dangling_pointers.values())
            + sum(len(ids) for ids in self.mismatched_pointers.values())
            + len(self.phantom_vectors)
        )

    @property
    def lexical_total(self) -> int:
        """Gaps this migration can close: everything deterministic from chunks.content."""
        return (
            sum(len(ids) for ids in self.missing_index_rows.values())
            + sum(len(ids) for ids in self.duplicate_index_rows.values())
            + sum(len(ids) for ids in self.misrouted_index_rows.values())
            + sum(len(ids) for ids in self.orphan_index_rows.values())
            + len(self.orphan_pointer_rows)
            + sum(len(ids) for ids in self.dangling_pointers.values())
            + sum(len(ids) for ids in self.mismatched_pointers.values())
        )

    @property
    def vector_total(self) -> int:
        """Gaps only the embed pipeline can close. Never repaired in a migration."""
        return len(self.missing_vector_rowid) + len(self.orphan_vector_rowids) + len(self.phantom_vectors)

    @property
    def total(self) -> int:
        return self.chunk_to_aux_total + self.aux_to_chunk_total

    def summary(self) -> dict[str, Any]:
        return {
            "chunk_to_aux": {
                "missing_index_rows": {k: len(v) for k, v in self.missing_index_rows.items()},
                "duplicate_index_rows": {k: len(v) for k, v in self.duplicate_index_rows.items()},
                "misrouted_index_rows": {k: len(v) for k, v in self.misrouted_index_rows.items()},
                "missing_vector_rowid": len(self.missing_vector_rowid),
                "missing_binary_vector": len(self.missing_binary_vector),
                "total": self.chunk_to_aux_total,
            },
            "aux_to_chunk": {
                "orphan_index_rows": {k: len(v) for k, v in self.orphan_index_rows.items()},
                "orphan_vector_rowids": len(self.orphan_vector_rowids),
                "orphan_pointer_rows": len(self.orphan_pointer_rows),
                "dangling_pointers": {k: len(v) for k, v in self.dangling_pointers.items()},
                "mismatched_pointers": {k: len(v) for k, v in self.mismatched_pointers.items()},
                "phantom_vectors": len(self.phantom_vectors),
                "vector_payload_checked": self.vector_payload_checked,
                "total": self.aux_to_chunk_total,
            },
            "lexical_total": self.lexical_total,
            "vector_total": self.vector_total,
            "total": self.total,
        }


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _bit_set(validity: bytes, offset: int) -> bool:
    index = offset // 8
    if index >= len(validity):
        return False
    return bool(validity[index] & (1 << (offset % 8)))


ROUTE_CASE_SQL = """
    CASE
        WHEN COALESCE(content_class, 'knowledge') = 'operational' THEN 'operational'
        WHEN COALESCE(content_class, 'knowledge') IN ('test', 'benchmark', 'cold') THEN 'none'
        ELSE 'knowledge'
    END
"""


def _build_scratch(conn: sqlite3.Connection, present: set[str]) -> list[str]:
    """Stage routing and index membership in TEMP tables, not in Python.

    The dict version of this census peaked at 1.1GB of RSS on the canonical
    corpus, which is not a price doctor can pay on every run. TEMP tables live
    in SQLite's own temp database -- writable even when the main DB is opened
    `mode=ro` -- so the working set spills to disk and memory stays flat.
    """
    conn.execute("PRAGMA temp_store = FILE")
    conn.execute("CREATE TEMP TABLE _ic_route (chunk_id TEXT PRIMARY KEY, route TEXT, active INTEGER)")
    conn.execute(
        f"""
        INSERT INTO temp._ic_route(chunk_id, route, active)
        SELECT id, {ROUTE_CASE_SQL},
               CASE WHEN {ACTIVE_CHUNK_SQL.strip()} THEN 1 ELSE 0 END
        FROM chunks
        """
    )
    staged: list[str] = []
    for table in ROUTED_TABLES:
        if table not in present:
            continue
        conn.execute(f"CREATE TEMP TABLE _ic_own_{table} (row_id INTEGER PRIMARY KEY, chunk_id TEXT)")
        conn.execute(
            f"INSERT INTO temp._ic_own_{table}(row_id, chunk_id) "
            f"SELECT id, c6 FROM {table}_content WHERE c6 IS NOT NULL"
        )
        conn.execute(f"CREATE INDEX _ic_own_{table}_chunk ON _ic_own_{table}(chunk_id)")
        staged.append(table)
    return staged


def _drop_scratch(conn: sqlite3.Connection, staged: list[str]) -> None:
    """Remove this census's TEMP staging tables. Touches temp only, never the DB."""
    for table in staged:
        conn.execute(f"DROP TABLE IF EXISTS temp._ic_own_{table}")
    conn.execute("DROP TABLE IF EXISTS temp._ic_route")


def _ids(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = (), *, collect: bool) -> list[str]:
    """Ids for a defect, or a count-shaped stand-in when ids are not wanted.

    Doctor only needs to know a set is non-empty and how big it is; the
    migration needs the ids themselves. Collecting 16k ids is cheap, but
    collecting them on every doctor run over a growing corpus is not.
    """
    if collect:
        return [str(row[0]) for row in conn.execute(sql, params)]
    count = conn.execute(f"SELECT COUNT(*) FROM ({sql})", params).fetchone()[0]
    return [""] * int(count)


def census(
    conn: sqlite3.Connection,
    *,
    verify_vector_payload: bool = False,
    collect_ids: bool = True,
) -> CompletenessCensus:
    """Measure completeness in both directions. Reads only the main database."""
    present = _table_names(conn)
    result = CompletenessCensus()
    staged = _build_scratch(conn, present)
    try:
        # ── chunk -> aux ──────────────────────────────────────────────────
        for table in staged:
            wanted_route = ROUTED_TABLES[table]
            missing = _ids(
                conn,
                f"""
                SELECT r.chunk_id FROM temp._ic_route r
                WHERE r.active = 1 AND r.route = '{wanted_route}'
                  AND NOT EXISTS (SELECT 1 FROM temp._ic_own_{table} o WHERE o.chunk_id = r.chunk_id)
                ORDER BY r.chunk_id
                """,
                collect=collect_ids,
            )
            duplicate = _ids(
                conn,
                f"""
                SELECT chunk_id FROM temp._ic_own_{table}
                GROUP BY chunk_id HAVING COUNT(*) > 1 ORDER BY chunk_id
                """,
                collect=collect_ids,
            )
            misrouted = _ids(
                conn,
                f"""
                SELECT DISTINCT o.chunk_id FROM temp._ic_own_{table} o
                JOIN temp._ic_route r ON r.chunk_id = o.chunk_id
                WHERE r.active = 1 AND r.route <> '{wanted_route}'
                ORDER BY o.chunk_id
                """,
                collect=collect_ids,
            )
            if missing:
                result.missing_index_rows[table] = missing
            if duplicate:
                result.duplicate_index_rows[table] = duplicate
            if misrouted:
                result.misrouted_index_rows[table] = misrouted

        # ── aux -> chunk: index rows whose chunk is gone ──────────────────
        for table in staged:
            orphans = [
                int(row[0])
                for row in conn.execute(
                    f"""
                    SELECT o.row_id FROM temp._ic_own_{table} o
                    WHERE NOT EXISTS (SELECT 1 FROM temp._ic_route r WHERE r.chunk_id = o.chunk_id)
                    ORDER BY o.row_id
                    """
                )
            ]
            if orphans:
                result.orphan_index_rows[table] = orphans

        # ── vectors: pointer presence ─────────────────────────────────────
        if "chunk_vectors_rowids" in present:
            result.missing_vector_rowid = _ids(
                conn,
                """
                SELECT r.chunk_id FROM temp._ic_route r
                WHERE r.active = 1
                  AND NOT EXISTS (SELECT 1 FROM chunk_vectors_rowids v WHERE v.id = r.chunk_id)
                ORDER BY r.chunk_id
                """,
                collect=collect_ids,
            )
            result.orphan_vector_rowids = _ids(
                conn,
                """
                SELECT v.id FROM chunk_vectors_rowids v
                WHERE NOT EXISTS (SELECT 1 FROM temp._ic_route r WHERE r.chunk_id = v.id)
                ORDER BY v.id
                """,
                collect=collect_ids,
            )
        if "chunk_vectors_binary_rowids" in present:
            result.missing_binary_vector = _ids(
                conn,
                """
                SELECT r.chunk_id FROM temp._ic_route r
                WHERE r.active = 1
                  AND NOT EXISTS (SELECT 1 FROM chunk_vectors_binary_rowids v WHERE v.id = r.chunk_id)
                ORDER BY r.chunk_id
                """,
                collect=collect_ids,
            )

        # ── pointers: orphan, dangling, and aimed at someone else's row ───
        if "chunk_fts_rowids" in present and staged:
            result.orphan_pointer_rows = _ids(
                conn,
                """
                SELECT p.chunk_id FROM chunk_fts_rowids p
                WHERE NOT EXISTS (SELECT 1 FROM temp._ic_route r WHERE r.chunk_id = p.chunk_id)
                ORDER BY p.chunk_id
                """,
                collect=collect_ids,
            )
            for table in staged:
                column = POINTER_COLUMN[table]
                dangling = _ids(
                    conn,
                    f"""
                    SELECT p.chunk_id FROM chunk_fts_rowids p
                    WHERE p.{column} IS NOT NULL
                      AND NOT EXISTS (SELECT 1 FROM temp._ic_own_{table} o WHERE o.row_id = p.{column})
                    ORDER BY p.chunk_id
                    """,
                    collect=collect_ids,
                )
                mismatched = _ids(
                    conn,
                    f"""
                    SELECT p.chunk_id FROM chunk_fts_rowids p
                    JOIN temp._ic_own_{table} o ON o.row_id = p.{column}
                    WHERE p.{column} IS NOT NULL AND o.chunk_id <> p.chunk_id
                    ORDER BY p.chunk_id
                    """,
                    collect=collect_ids,
                )
                if dangling:
                    result.dangling_pointers[table] = dangling
                if mismatched:
                    result.mismatched_pointers[table] = mismatched

        # ── vector payload: a pointer is not a vector ─────────────────────
        if verify_vector_payload and "chunk_vectors_chunks" in present and "chunk_vectors_rowids" in present:
            result.phantom_vectors, result.vector_payload_checked = _verify_vector_payload(conn)
    finally:
        _drop_scratch(conn, staged)
    return result


def _verify_vector_payload(conn: sqlite3.Connection) -> tuple[list[str], int]:
    """Decode the vec0 validity bitmap and float blob for every active vector.

    A `chunk_vectors_rowids` row proves a pointer, not a vector. Doctor's
    `vector_parity_gap` counts pointers, so a row whose validity bit is clear,
    whose blob is missing, or whose 1024 floats are all zero reads as embedded
    while semantic search can never return it.
    """
    validity = {
        int(vec_chunk): bytes(bitmap)
        for vec_chunk, bitmap in conn.execute("SELECT chunk_id, validity FROM chunk_vectors_chunks")
    }
    phantom: list[str] = []
    checked = 0
    blob_cache: dict[int, bytes | None] = {}
    # Walk vec0 storage order, not chunk-id order: each vector blob holds 1024
    # vectors and is ~4MB, so visiting ids at random re-reads the same blob
    # thousands of times.
    rows = conn.execute(
        """
        SELECT v.chunk_id, v.chunk_offset, v.id
        FROM chunk_vectors_rowids v
        JOIN temp._ic_route r ON r.chunk_id = v.id
        WHERE r.active = 1
        ORDER BY v.chunk_id, v.chunk_offset
        """
    )
    for vec_chunk, offset, chunk_id in rows:
        checked += 1
        bitmap = validity.get(int(vec_chunk))
        if bitmap is None or not _bit_set(bitmap, int(offset)):
            phantom.append(str(chunk_id))
            continue
        if vec_chunk not in blob_cache:
            blob_cache.clear()
            found = conn.execute(
                "SELECT vectors FROM chunk_vectors_vector_chunks00 WHERE rowid = ?", (vec_chunk,)
            ).fetchone()
            blob_cache[vec_chunk] = bytes(found[0]) if found else None
        blob = blob_cache[vec_chunk]
        start = int(offset) * VECTOR_BYTES
        if blob is None or start + VECTOR_BYTES > len(blob):
            phantom.append(str(chunk_id))
            continue
        if blob[start : start + VECTOR_BYTES] == b"\x00" * VECTOR_BYTES:
            phantom.append(str(chunk_id))
    return phantom, checked


# ── repair ───────────────────────────────────────────────────────────────────


@dataclass
class RepairResult:
    apply: bool
    census_before: dict[str, Any]
    census_after: dict[str, Any] | None
    inserted_rows: int = 0
    deleted_rows: int = 0
    pointers_rewritten: int = 0
    orphans_deleted: int = 0
    chunks_touched: int = 0
    run_id: str | None = None
    vector_debt: dict[str, Any] = field(default_factory=dict)
    spot_checks: list[dict[str, Any]] = field(default_factory=list)
    batches: int = 0


def _ensure_preimage(conn: sqlite3.Connection) -> None:
    """Create (or migrate) the preimage table.

    `run_id` is what makes a second rollback safe. Preimages accumulate -- they
    are the retained audit trail, per the repair-b/d/e precedent -- so a rollback
    that replayed the whole table would try to undo runs already undone, and on
    an apply/rollback/apply cycle it aborts on a rowid collision. Rollback is
    therefore scoped to one run: the newest that has not already been reversed.
    """
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {PREIMAGE_TABLE} (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL DEFAULT 'legacy',
            chunk_id TEXT NOT NULL,
            table_name TEXT NOT NULL,
            action TEXT NOT NULL,
            rowid_value INTEGER,
            payload TEXT,
            recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )
        """
    )
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({PREIMAGE_TABLE})")}
    if "run_id" not in columns:
        conn.execute(f"ALTER TABLE {PREIMAGE_TABLE} ADD COLUMN run_id TEXT NOT NULL DEFAULT 'legacy'")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{PREIMAGE_TABLE}_chunk ON {PREIMAGE_TABLE}(chunk_id)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{PREIMAGE_TABLE}_run ON {PREIMAGE_TABLE}(run_id)")


def _new_run_id(conn: sqlite3.Connection) -> str:
    """A monotonic id for this apply, derived from the DB's own clock."""
    stamp = conn.execute("SELECT strftime('%Y%m%dT%H%M%f','now')").fetchone()[0]
    return f"run-{stamp}"


def _rollback_marker(run_id: str) -> str:
    return f"rolled_back:{run_id}"


def _chunk_payload(conn: sqlite3.Connection, chunk_id: str) -> tuple[Any, ...] | None:
    row = conn.execute(
        """
        SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id
        FROM chunks WHERE id = ?
        """,
        (chunk_id,),
    ).fetchone()
    return tuple(row) if row else None


def _delete_index_row(
    conn: sqlite3.Connection, table: str, rowid: int, chunk_id: str, action: str, run_id: str
) -> bool:
    """Delete one aux index row after re-verifying it belongs to chunk_id.

    The verification is the whole point: a rowid recorded during the census is
    not trusted at write time. If the row now carries a different chunk_id the
    delete is skipped, because deleting it would silently unindex someone else
    -- exactly the failure the mismatched pointers would have caused.
    """
    row = conn.execute(f"SELECT c0, c1, c2, c3, c4, c5, c6 FROM {table}_content WHERE id = ?", (rowid,)).fetchone()
    if row is None or str(row[6]) != chunk_id:
        return False
    conn.execute(
        f"INSERT INTO {PREIMAGE_TABLE}(run_id, chunk_id, table_name, action, rowid_value, payload) VALUES (?,?,?,?,?,?)",
        (run_id, chunk_id, table, action, rowid, json.dumps(list(row), default=str)),
    )
    conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
    return True


def _next_rowid(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COALESCE(MAX(id), 0) + 1 FROM {table}_content").fetchone()
    return int(row[0])


def _existing_index_rows(
    conn: sqlite3.Connection, chunk_ids: Sequence[str]
) -> tuple[dict[str, dict[str, list[int]]], dict[str, int]]:
    """Every existing FTS rowid for the chunks about to be repaired, in one scan each.

    Also returns the first free rowid per index. New rows are appended above
    the current maximum and rowids freed by a delete are never reused, so a
    pointer written in this run can never collide with a row this run removed.
    """
    present = _table_names(conn)
    conn.execute("PRAGMA temp_store = FILE")
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _ic_touched (chunk_id TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM temp._ic_touched")
    conn.executemany("INSERT OR IGNORE INTO temp._ic_touched(chunk_id) VALUES (?)", [(cid,) for cid in chunk_ids])
    rows: dict[str, dict[str, list[int]]] = {}
    next_rowid: dict[str, int] = {}
    for table in ROUTED_TABLES:
        if table not in present:
            continue
        found: dict[str, list[int]] = {}
        for rowid, chunk_id in conn.execute(
            f"""
            SELECT c.id, c.c6 FROM {table}_content c
            JOIN temp._ic_touched t ON t.chunk_id = c.c6
            """
        ):
            found.setdefault(str(chunk_id), []).append(int(rowid))
        rows[table] = found
        next_rowid[table] = _next_rowid(conn, table)
    return rows, next_rowid


def _insert_index_row(
    conn: sqlite3.Connection, table: str, payload: Sequence[Any], chunk_id: str, rowid: int, run_id: str
) -> int:
    """Insert one FTS row at an explicitly chosen rowid.

    The rowid is assigned by the caller rather than read back afterwards. Two
    reasons. Recovering it with `SELECT ... WHERE c6 = ?` is a full scan of the
    FTS content table -- `c6` carries no index -- which measured at ~2s per
    chunk on the canonical corpus, i.e. hours for the real repair. And
    `last_insert_rowid()` after an FTS5 insert can report a shadow-table rowid
    rather than the row's own, which is one candidate mechanism for the 2,661
    mismatched pointers this migration exists to fix. Choosing the value
    removes both problems: the pointer written afterwards cannot be a guess.
    """
    columns = ", ".join(("rowid", *FTS_INSERT_COLUMNS))
    placeholders = ", ".join("?" for _ in range(len(FTS_INSERT_COLUMNS) + 1))
    conn.execute(f"INSERT INTO {table}({columns}) VALUES ({placeholders})", [rowid, *payload])
    conn.execute(
        f"INSERT INTO {PREIMAGE_TABLE}(run_id, chunk_id, table_name, action, rowid_value, payload) VALUES (?,?,?,?,?,?)",
        (run_id, chunk_id, table, "insert", rowid, None),
    )
    return rowid


def _rewrite_pointer(conn: sqlite3.Connection, chunk_id: str, updates: dict[str, int | None], run_id: str) -> bool:
    if not updates:
        return False
    before = conn.execute(
        "SELECT fts_rowid, trigram_rowid, operational_rowid FROM chunk_fts_rowids WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    desired = {
        "fts_rowid": before[0] if before else None,
        "trigram_rowid": before[1] if before else None,
        "operational_rowid": before[2] if before else None,
    }
    desired.update(updates)
    if before is not None and tuple(desired.values()) == tuple(before):
        return False
    conn.execute(
        f"INSERT INTO {PREIMAGE_TABLE}(run_id, chunk_id, table_name, action, rowid_value, payload) VALUES (?,?,?,?,?,?)",
        (
            run_id,
            chunk_id,
            "chunk_fts_rowids",
            "pointer",
            None,
            json.dumps({"before": list(before) if before else None}),
        ),
    )
    conn.execute(
        """
        INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid, trigram_rowid, operational_rowid)
        VALUES (?,?,?,?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            fts_rowid = excluded.fts_rowid,
            trigram_rowid = excluded.trigram_rowid,
            operational_rowid = excluded.operational_rowid
        """,
        (chunk_id, desired["fts_rowid"], desired["trigram_rowid"], desired["operational_rowid"]),
    )
    return True


def _checkpoint(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    except sqlite3.Error:
        pass


def _detect_git_sha() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None
    return sha if len(sha) == 40 else None


def _record_migration(conn: sqlite3.Connection, *, git_sha: str, actor: str, payload: dict[str, Any]) -> None:
    if "schema_migrations" not in _table_names(conn):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(schema_migrations)")}
    if not {"name"} <= columns:
        return
    values: dict[str, Any] = {"name": MIGRATION_NAME}
    if "applied_at" in columns:
        values["applied_at"] = conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')").fetchone()[0]
    if "git_sha" in columns:
        values["git_sha"] = git_sha
    if "actor" in columns:
        values["actor"] = actor
    if "details" in columns:
        # The canonical table is (name PK, applied_at, details) with no git_sha
        # or actor column, so the receipt has to carry them itself -- dropping
        # them would leave a migration record that cannot be traced to a commit.
        details = dict(payload)
        details.setdefault("git_sha", git_sha)
        details.setdefault("actor", actor)
        values["details"] = json.dumps(details, sort_keys=True)
    names = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    # OR REPLACE: `name` is the primary key, so a re-run must update its receipt
    # rather than abort the whole migration on the very last statement.
    conn.execute(f"INSERT OR REPLACE INTO schema_migrations({names}) VALUES ({marks})", list(values.values()))


def _spot_check(conn: sqlite3.Connection, chunk_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Re-read repaired chunks and verify VALUES, not counts.

    For each sampled chunk: exactly one row in every index its class routes to,
    the indexed text byte-equal to `chunks.content`, no row in any index it
    does not route to, and a pointer that resolves back to this same chunk.
    """
    checks: list[dict[str, Any]] = []
    for chunk_id in chunk_ids:
        row = conn.execute("SELECT content, content_class FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            checks.append({"chunk_id": chunk_id, "ok": False, "why": ["chunk_missing"]})
            continue
        content, content_class = row
        route = route_of(content_class)
        why: list[str] = []
        pointers = conn.execute(
            "SELECT fts_rowid, trigram_rowid, operational_rowid FROM chunk_fts_rowids WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        pointer_by_table = dict(
            zip(("chunks_fts", "chunks_fts_trigram", "chunks_fts_operational"), pointers or (None, None, None))
        )
        for table, wanted_route in ROUTED_TABLES.items():
            rows = conn.execute(f"SELECT id, c0 FROM {table}_content WHERE c6 = ?", (chunk_id,)).fetchall()
            if wanted_route == route:
                if len(rows) != 1:
                    why.append(f"{table}:expected_1_row_got_{len(rows)}")
                    continue
                rowid, indexed = rows[0]
                if indexed != content:
                    why.append(f"{table}:indexed_text_differs_from_chunk_content")
                if pointer_by_table.get(table) != rowid:
                    why.append(f"{table}:pointer_does_not_resolve_to_this_row")
            elif rows:
                why.append(f"{table}:present_but_class_routes_to_{route}")
        checks.append({"chunk_id": chunk_id, "route": route, "ok": not why, "why": why})
    return checks


def repair_index_completeness(
    db_path: Path,
    *,
    git_sha: str,
    apply: bool = False,
    allow_live: bool = False,
    batch_size: int = 500,
    checkpoint_every: int = 3,
    spot_check: int = 0,
    delete_orphans: bool = True,
    actor: str = "repair-f",
) -> RepairResult:
    """Rebuild missing/duplicated/misrouted FTS rows and realign pointers.

    Never embeds, never writes `chunks`, never writes a vec0 table.
    """
    resolved = assert_not_live_db(Path(db_path), allow_live=allow_live)
    conn = sqlite3.connect(resolved, timeout=60)
    # Autocommit: this migration opens every write transaction by hand
    # (BEGIN IMMEDIATE / COMMIT per batch), and the driver's implicit
    # transaction would otherwise still be open from the census's TEMP writes.
    conn.isolation_level = None
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        before = census(conn, verify_vector_payload=True)
        result = RepairResult(apply=apply, census_before=before.summary(), census_after=None)
        result.vector_debt = {
            "missing_vector_rowid": len(before.missing_vector_rowid),
            "missing_binary_vector": len(before.missing_binary_vector),
            "phantom_vectors": len(before.phantom_vectors),
            "owner": "brainlayer reembed-backfill",
            "note": (
                "this migration never embeds; a missing vector is work for the embed "
                "pipeline, which finds these rows by the same LEFT JOIN"
            ),
            "sample": before.missing_vector_rowid[:10],
            "phantom_sample": before.phantom_vectors[:10],
        }

        # Chunks needing lexical work, and what to do with each.
        work: dict[str, dict[str, list[int] | None]] = {}
        for table, ids in before.missing_index_rows.items():
            for chunk_id in ids:
                work.setdefault(chunk_id, {}).setdefault(table, None)
        for table, ids in before.duplicate_index_rows.items():
            for chunk_id in ids:
                work.setdefault(chunk_id, {}).setdefault(table, None)
        for table, ids in before.misrouted_index_rows.items():
            for chunk_id in ids:
                work.setdefault(chunk_id, {}).setdefault(table, None)
        for table, ids in before.dangling_pointers.items():
            for chunk_id in ids:
                work.setdefault(chunk_id, {}).setdefault(table, None)
        for table, ids in before.mismatched_pointers.items():
            for chunk_id in ids:
                work.setdefault(chunk_id, {}).setdefault(table, None)

        result.chunks_touched = len(work)
        if not apply:
            result.census_after = before.summary()
            return result

        _ensure_preimage(conn)
        run_id = _new_run_id(conn)
        result.run_id = run_id
        touched: list[str] = sorted(work)
        # One scan per FTS table to find the touched chunks' existing rows.
        # `c6` (chunk_id) carries no index, so asking per chunk would be one
        # full scan of a multi-GB content table per chunk.
        existing_rows, next_rowid = _existing_index_rows(conn, touched)
        for index, start in enumerate(range(0, len(touched), batch_size), start=1):
            batch = touched[start : start + batch_size]
            conn.execute("BEGIN IMMEDIATE")
            try:
                for chunk_id in batch:
                    row = conn.execute(
                        f"SELECT content_class, CASE WHEN {ACTIVE_CHUNK_SQL.strip()} THEN 1 ELSE 0 END "
                        "FROM chunks WHERE id = ?",
                        (chunk_id,),
                    ).fetchone()
                    if row is None:
                        continue
                    route = route_of(row[0])
                    is_active = bool(row[1])
                    payload = _chunk_payload(conn, chunk_id)
                    pointer_updates: dict[str, int | None] = {}
                    for table, wanted_route in ROUTED_TABLES.items():
                        if table not in next_rowid:
                            continue
                        existing = sorted(existing_rows.get(table, {}).get(chunk_id, ()))
                        if wanted_route == route and payload is not None and payload[0]:
                            # Rebuild to exactly one row carrying current content.
                            for rowid in existing:
                                if _delete_index_row(conn, table, rowid, chunk_id, "rebuild", run_id):
                                    result.deleted_rows += 1
                            new_rowid = next_rowid[table]
                            next_rowid[table] += 1
                            _insert_index_row(conn, table, payload, chunk_id, new_rowid, run_id)
                            result.inserted_rows += 1
                            pointer_updates[POINTER_COLUMN[table]] = new_rowid
                        elif is_active:
                            # Wrong index for this class: drop the leaked rows.
                            for rowid in existing:
                                if _delete_index_row(conn, table, rowid, chunk_id, "misroute", run_id):
                                    result.deleted_rows += 1
                            pointer_updates[POINTER_COLUMN[table]] = None
                        else:
                            # Lifecycle-managed chunk sitting in an index its class
                            # does not route to. The census does not report this --
                            # its misroute check is scoped to active chunks -- and a
                            # migration must not delete more than it reported. Leave
                            # the rows and just re-aim the pointer at one of them, so
                            # the delete trigger can still find it.
                            pointer_updates[POINTER_COLUMN[table]] = existing[0] if existing else None
                    if _rewrite_pointer(conn, chunk_id, pointer_updates, run_id):
                        result.pointers_rewritten += 1
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            result.batches = index
            if checkpoint_every and index % checkpoint_every == 0:
                _checkpoint(conn)

        # Orphan aux rows: index rows and pointers whose chunk is gone. These
        # are index entries, never chunk data, and the absence is re-verified
        # inside the transaction before anything is removed.
        if delete_orphans:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for table, rowids in before.orphan_index_rows.items():
                    for rowid in rowids:
                        found = conn.execute(f"SELECT c6 FROM {table}_content WHERE id = ?", (rowid,)).fetchone()
                        if found is None:
                            continue
                        chunk_id = str(found[0])
                        still_gone = conn.execute("SELECT 1 FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
                        if still_gone is not None:
                            continue
                        if _delete_index_row(conn, table, rowid, chunk_id, "orphan", run_id):
                            result.orphans_deleted += 1
                for chunk_id in before.orphan_pointer_rows:
                    if conn.execute("SELECT 1 FROM chunks WHERE id = ?", (chunk_id,)).fetchone() is not None:
                        continue
                    row = conn.execute(
                        "SELECT fts_rowid, trigram_rowid, operational_rowid FROM chunk_fts_rowids WHERE chunk_id = ?",
                        (chunk_id,),
                    ).fetchone()
                    conn.execute(
                        f"INSERT INTO {PREIMAGE_TABLE}(run_id, chunk_id, table_name, action, rowid_value, payload) "
                        "VALUES (?,?,?,?,?,?)",
                        (
                            run_id,
                            chunk_id,
                            "chunk_fts_rowids",
                            "orphan_pointer",
                            None,
                            json.dumps(list(row) if row else None),
                        ),
                    )
                    conn.execute("DELETE FROM chunk_fts_rowids WHERE chunk_id = ?", (chunk_id,))
                    result.orphans_deleted += 1
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        _record_migration(
            conn,
            git_sha=git_sha,
            actor=actor,
            payload={
                "inserted_rows": result.inserted_rows,
                "deleted_rows": result.deleted_rows,
                "pointers_rewritten": result.pointers_rewritten,
                "orphans_deleted": result.orphans_deleted,
            },
        )
        conn.commit()
        _checkpoint(conn)

        after = census(conn, verify_vector_payload=True)
        result.census_after = after.summary()
        if spot_check and touched:
            step = max(1, len(touched) // spot_check)
            sample = touched[::step][:spot_check]
            result.spot_checks = _spot_check(conn, sample)
        return result
    finally:
        conn.close()


def _rollbackable_run(conn: sqlite3.Connection) -> str | None:
    """The newest apply that has not already been reversed."""
    reversed_runs = {
        str(row[0]).split(":", 1)[1]
        for row in conn.execute(f"SELECT action FROM {PREIMAGE_TABLE} WHERE action LIKE 'rolled_back:%'")
    }
    for (run_id,) in conn.execute(
        f"SELECT run_id FROM {PREIMAGE_TABLE} WHERE action NOT LIKE 'rolled_back:%' "
        "GROUP BY run_id ORDER BY MAX(seq) DESC"
    ):
        if str(run_id) not in reversed_runs:
            return str(run_id)
    return None


def rollback_repair(db_path: Path, *, allow_live: bool = False, run_id: str | None = None) -> dict[str, Any]:
    """Undo ONE apply from its preimages, newest write first.

    Inserted rows are deleted; deleted rows are re-inserted at their original
    rowid with their recorded text; pointer rows are restored to their recorded
    triple. The preimage table is retained afterwards as the audit trail
    (repair-b/d/e precedent), so the scope has to be a single run: replaying the
    whole table would try to undo applies already undone, and on an
    apply/rollback/apply cycle it aborts on a rowid collision instead.
    """
    resolved = assert_not_live_db(Path(db_path), allow_live=allow_live)
    conn = sqlite3.connect(resolved, timeout=60)
    conn.isolation_level = None
    stats: dict[str, Any] = {
        "restored_rows": 0,
        "removed_rows": 0,
        "restored_pointers": 0,
        "preimage_tables": "retained",
    }
    try:
        if PREIMAGE_TABLE not in _table_names(conn):
            return {**stats, "preimage": "absent"}
        _ensure_preimage(conn)
        target = run_id or _rollbackable_run(conn)
        stats["run_id"] = target
        if target is None:
            return {**stats, "preimage": "nothing left to roll back"}
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = conn.execute(
                f"SELECT seq, chunk_id, table_name, action, rowid_value, payload FROM {PREIMAGE_TABLE} "
                "WHERE run_id = ? AND action NOT LIKE 'rolled_back:%' ORDER BY seq DESC",
                (target,),
            ).fetchall()
            for _seq, chunk_id, table, action, rowid_value, payload in rows:
                if action == "insert":
                    found = conn.execute(f"SELECT c6 FROM {table}_content WHERE id = ?", (rowid_value,)).fetchone()
                    if found is not None and str(found[0]) == chunk_id:
                        conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid_value,))
                        stats["removed_rows"] += 1
                elif action in {"rebuild", "misroute", "orphan"}:
                    # Restore at the ORIGINAL rowid, not wherever FTS5 would put
                    # a fresh row. The pointer preimages restore the old rowid
                    # numbers, so a row that comes back at a different rowid
                    # leaves every restored pointer dangling -- a rollback that
                    # reports success while breaking what it claimed to fix.
                    values = json.loads(payload)
                    # Clear the slot first: this run may have inserted its own
                    # row at that rowid, and FTS5 has no upsert.
                    conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid_value,))
                    columns = ", ".join(("rowid", *FTS_INSERT_COLUMNS))
                    marks = ", ".join("?" for _ in range(len(FTS_INSERT_COLUMNS) + 1))
                    conn.execute(f"INSERT INTO {table}({columns}) VALUES ({marks})", [rowid_value, *values])
                    stats["restored_rows"] += 1
                elif action == "pointer":
                    before = json.loads(payload).get("before")
                    if before is None:
                        conn.execute("DELETE FROM chunk_fts_rowids WHERE chunk_id = ?", (chunk_id,))
                    else:
                        conn.execute(
                            """
                            INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid, trigram_rowid, operational_rowid)
                            VALUES (?,?,?,?)
                            ON CONFLICT(chunk_id) DO UPDATE SET
                                fts_rowid = excluded.fts_rowid,
                                trigram_rowid = excluded.trigram_rowid,
                                operational_rowid = excluded.operational_rowid
                            """,
                            (chunk_id, *before),
                        )
                    stats["restored_pointers"] += 1
                elif action == "orphan_pointer":
                    before = json.loads(payload)
                    if before:
                        conn.execute(
                            "INSERT OR REPLACE INTO chunk_fts_rowids"
                            "(chunk_id, fts_rowid, trigram_rowid, operational_rowid) VALUES (?,?,?,?)",
                            (chunk_id, *before),
                        )
                        stats["restored_pointers"] += 1
            # Mark the run reversed so a second --rollback moves on to the run
            # before it instead of replaying this one.
            conn.execute(
                f"INSERT INTO {PREIMAGE_TABLE}(run_id, chunk_id, table_name, action, rowid_value, payload) "
                "VALUES (?,?,?,?,?,?)",
                (target, "", "", _rollback_marker(target), None, json.dumps(stats, sort_keys=True)),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return stats
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repair (f) FTS/vector completeness. Default is dry-run. Requires --db; "
            "refuses the live canonical DB unless --allow-live. Never embeds, never "
            "writes the chunks table."
        )
    )
    parser.add_argument("--db", type=Path, required=True, help="Rehearsal copy DB path (required)")
    parser.add_argument("--apply", action="store_true", help="Write the lexical repairs")
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--git-sha", dest="git_sha")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=3)
    parser.add_argument("--spot-check", type=int, default=0)
    parser.add_argument("--keep-orphans", action="store_true", help="Do not delete orphan aux index rows")
    parser.add_argument("--actor", default="repair-f")
    parser.add_argument("--rollback", action="store_true", help="Undo the newest apply not yet reversed")
    parser.add_argument("--rollback-run", dest="rollback_run", help="Undo this run_id instead of the newest")
    parser.add_argument("--census-only", action="store_true", help="Print the census and exit")
    args = parser.parse_args(argv)

    if args.rollback or args.rollback_run:
        print(
            json.dumps(
                rollback_repair(args.db.expanduser(), allow_live=args.allow_live, run_id=args.rollback_run),
                sort_keys=True,
            )
        )
        return 0

    if args.census_only:
        conn = sqlite3.connect(f"file:{args.db.expanduser()}?mode=ro", uri=True)
        try:
            print(json.dumps(census(conn, verify_vector_payload=True).summary(), indent=2, sort_keys=True))
        finally:
            conn.close()
        return 0

    git_sha = args.git_sha or _detect_git_sha()
    if not git_sha:
        parser.error("--git-sha is required (40-char hex) when HEAD is not a full SHA")
    result = repair_index_completeness(
        args.db.expanduser(),
        git_sha=git_sha,
        apply=args.apply,
        allow_live=args.allow_live,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        spot_check=args.spot_check,
        delete_orphans=not args.keep_orphans,
        actor=args.actor,
    )
    print(
        json.dumps(
            {
                "apply": result.apply,
                "chunks_touched": result.chunks_touched,
                "inserted_rows": result.inserted_rows,
                "deleted_rows": result.deleted_rows,
                "pointers_rewritten": result.pointers_rewritten,
                "orphans_deleted": result.orphans_deleted,
                "batches": result.batches,
                "vector_debt": result.vector_debt,
                "census_before": result.census_before,
                "census_after": result.census_after,
                "spot_checks": result.spot_checks,
            },
            sort_keys=True,
        )
    )
    if result.apply and result.spot_checks and not all(check["ok"] for check in result.spot_checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
