#!/usr/bin/env python3
"""Move existing mixed FTS rows into P1.4 routed FTS tables.

Dry run is the default. Apply mode rebuilds FTS tables from `chunks` instead of
editing individual FTS rows, which keeps BM25 statistics deterministic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import apsw

from brainlayer.paths import DEFAULT_DB_PATH
from brainlayer.vector_store import VectorStore

KNOWLEDGE_CLASS_SQL = "COALESCE(content_class, 'knowledge') NOT IN ('operational', 'test', 'benchmark')"
OPERATIONAL_CLASS_SQL = "COALESCE(content_class, 'knowledge') = 'operational'"
COLD_CLASS_SQL = "COALESCE(content_class, 'knowledge') IN ('test', 'benchmark')"
FTS_COLUMNS = "content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id"
FTS_SELECT_COLUMNS = "content, summary, tags, resolved_query, key_facts, resolved_queries, id"


def _table_exists(cursor: apsw.Cursor, table_name: str) -> bool:
    return (
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
            (table_name,),
        ).fetchone()
        is not None
    )


def _ids(cursor: apsw.Cursor, sql: str, params: tuple[Any, ...] = ()) -> list[str]:
    return [row[0] for row in cursor.execute(sql, params)]


def _count(cursor: apsw.Cursor, sql: str) -> int:
    return int(cursor.execute(sql).fetchone()[0])


def _report(cursor: apsw.Cursor) -> dict[str, Any]:
    has_operational_fts = _table_exists(cursor, "chunks_fts_operational")
    has_trigram_fts = _table_exists(cursor, "chunks_fts_trigram")

    report: dict[str, Any] = {
        "chunk_counts": {
            "knowledge": _count(cursor, f"SELECT COUNT(*) FROM chunks WHERE {KNOWLEDGE_CLASS_SQL}"),
            "operational": _count(cursor, f"SELECT COUNT(*) FROM chunks WHERE {OPERATIONAL_CLASS_SQL}"),
            "cold": _count(cursor, f"SELECT COUNT(*) FROM chunks WHERE {COLD_CLASS_SQL}"),
        },
        "knowledge_fts_ids": _ids(cursor, "SELECT chunk_id FROM chunks_fts ORDER BY chunk_id"),
        "operational_fts_ids": [],
        "cold_fts_ids": _ids(
            cursor,
            f"""
            SELECT f.chunk_id
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.chunk_id
            WHERE {COLD_CLASS_SQL}
            ORDER BY f.chunk_id
            """,
        ),
        "knowledge_fts_operational_rows": _count(
            cursor,
            f"""
            SELECT COUNT(*)
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.chunk_id
            WHERE {OPERATIONAL_CLASS_SQL}
            """,
        ),
        "has_chunks_fts_operational": has_operational_fts,
        "has_chunks_fts_trigram": has_trigram_fts,
    }
    if has_operational_fts:
        report["operational_fts_ids"] = _ids(
            cursor,
            "SELECT chunk_id FROM chunks_fts_operational ORDER BY chunk_id",
        )
    return report


def _drop_fts_triggers(cursor: apsw.Cursor) -> None:
    for trigger_name in (
        "chunks_fts_insert",
        "chunks_fts_operational_insert",
        "chunks_fts_trigram_insert",
        "chunks_fts_delete",
        "chunks_fts_operational_delete",
        "chunks_fts_trigram_delete",
        "chunks_fts_update",
        "chunks_fts_operational_update",
        "chunks_fts_trigram_update",
    ):
        cursor.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")


def _insert_batches(cursor: apsw.Cursor, table_name: str, chunk_ids: list[str], batch_size: int) -> None:
    for start in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[start : start + batch_size]
        placeholders = ", ".join("?" for _ in batch)
        cursor.execute(
            f"""
            INSERT INTO {table_name}({FTS_COLUMNS})
            SELECT {FTS_SELECT_COLUMNS}
            FROM chunks
            WHERE id IN ({placeholders})
            ORDER BY id
            """,
            batch,
        )


def _sync_rowids(cursor: apsw.Cursor, *, include_trigram: bool) -> None:
    cursor.execute("DELETE FROM chunk_fts_rowids")
    cursor.execute("""
        INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid)
        SELECT chunk_id, rowid FROM chunks_fts WHERE chunk_id IS NOT NULL
    """)
    cursor.execute("""
        INSERT INTO chunk_fts_rowids(chunk_id, operational_rowid)
        SELECT chunk_id, rowid FROM chunks_fts_operational WHERE chunk_id IS NOT NULL
        ON CONFLICT(chunk_id) DO UPDATE SET operational_rowid = excluded.operational_rowid
    """)
    if include_trigram:
        cursor.execute("""
            INSERT INTO chunk_fts_rowids(chunk_id, trigram_rowid)
            SELECT chunk_id, rowid FROM chunks_fts_trigram WHERE chunk_id IS NOT NULL
            ON CONFLICT(chunk_id) DO UPDATE SET trigram_rowid = excluded.trigram_rowid
        """)


def migrate_fts_isolation(db_path: str | Path, *, dry_run: bool = True, batch_size: int = 5_000) -> dict[str, Any]:
    """Report or apply P1.4 FTS isolation for an existing BrainLayer DB."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    db_path = Path(db_path).expanduser()
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    if dry_run:
        conn = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READONLY)
        try:
            before = _report(conn.cursor())
        finally:
            conn.close()
        return {"dry_run": True, "db_path": str(db_path), "before": before, "after": None}

    # Ensure the new tables/rowid columns exist before bulk rebuilding them.
    schema_store = VectorStore(db_path)
    schema_store.close()

    conn = apsw.Connection(str(db_path))
    cursor = conn.cursor()
    before = _report(cursor)
    transaction_started = False
    try:
        cursor.execute("PRAGMA wal_checkpoint(FULL)")
        cursor.execute("BEGIN IMMEDIATE")
        transaction_started = True
        _drop_fts_triggers(cursor)

        knowledge_ids = _ids(cursor, f"SELECT id FROM chunks WHERE {KNOWLEDGE_CLASS_SQL} ORDER BY id")
        operational_ids = _ids(cursor, f"SELECT id FROM chunks WHERE {OPERATIONAL_CLASS_SQL} ORDER BY id")
        include_trigram = _table_exists(cursor, "chunks_fts_trigram")

        cursor.execute("DELETE FROM chunks_fts")
        cursor.execute("DELETE FROM chunks_fts_operational")
        if include_trigram:
            cursor.execute("DELETE FROM chunks_fts_trigram")
        cursor.execute("DELETE FROM chunk_fts_rowids")

        _insert_batches(cursor, "chunks_fts", knowledge_ids, batch_size)
        _insert_batches(cursor, "chunks_fts_operational", operational_ids, batch_size)
        if include_trigram:
            _insert_batches(cursor, "chunks_fts_trigram", knowledge_ids, batch_size)
        _sync_rowids(cursor, include_trigram=include_trigram)

        cursor.execute("COMMIT")
        transaction_started = False
        cursor.execute("PRAGMA wal_checkpoint(FULL)")
    except Exception:
        if transaction_started:
            cursor.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    # Recreate insert/delete/update triggers through the canonical initializer.
    trigger_store = VectorStore(db_path)
    try:
        after = _report(trigger_store.conn.cursor())
    finally:
        trigger_store.close()
    return {"dry_run": False, "db_path": str(db_path), "before": before, "after": after}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="BrainLayer SQLite DB path")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Default is dry run.")
    parser.add_argument("--batch-size", type=int, default=5_000, help="Chunk-id batch size for FTS rebuilds")
    args = parser.parse_args()

    report = migrate_fts_isolation(args.db, dry_run=not args.apply, batch_size=args.batch_size)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
