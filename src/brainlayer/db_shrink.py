"""Offline BrainLayer DB shrink migrations.

These helpers are intentionally path-parameterized. They are for snapshot
maintenance first; the canonical live DB requires an explicit guard override.

Repair (e), 2026-08-18: the physical-delete dedupe path was REMOVED from this
module (`apply_content_dedup` and its `_merge_duplicate_references` machinery,
which ended in `DELETE FROM chunks` stamped
`mechanism='normalized_content_physical_delete'`). It contradicted the lifecycle
law -- duplicates are archived with `aggregated_into` lineage, never deleted --
and it keyed on the lossy `normalized_exact_hash`, which lowercases and strips
stopwords. It had never run on the canonical DB (0 such rows in `dedupe_audit`).
Duplicate merging now lives in `brainlayer.dedupe_merge`, which only ever writes
lifecycle columns. `tests/test_db_shrink.py` guards against reintroduction.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import apsw
import sqlite_vec

from brainlayer.paths import get_db_path

FTS_COLUMNS = "content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED"
FTS_VALUE_COLUMNS = "content, summary, tags, resolved_query, key_facts, resolved_queries, id"
FTS_TRIGGER_COLUMNS = "content, summary, tags, resolved_query, key_facts, resolved_queries"
FTS_MODE_SINGLE_TRIGRAM = "single_trigram"
FTS_MODE_COMPACT_DUAL = "compact_dual"
LIVE_REFUSAL = "Refusing to write to the canonical live DB"


@dataclass(frozen=True)
class FtsMigrationResult:
    db_path: str
    mode: str
    chunk_count: int
    fts_count: int
    before_bytes: int
    after_bytes: int
    reclaimed_bytes: int


@dataclass(frozen=True)
class VacuumResult:
    db_path: str
    before_bytes: int
    after_bytes: int
    reclaimed_bytes: int


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def assert_not_live_db(db_path: str | Path, *, allow_live: bool = False) -> None:
    """Refuse writes to the canonical live DB unless explicitly overridden."""
    path = _resolve(db_path)
    live_path = _resolve(get_db_path())
    if path == live_path and not allow_live:
        raise ValueError(f"{LIVE_REFUSAL}; run against a snapshot or pass --i-know-this-is-live")


def _connect(db_path: str | Path) -> apsw.Connection:
    conn = apsw.Connection(str(_resolve(db_path)))
    conn.setbusytimeout(30_000)
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    return conn


def _table_exists(cursor: Any, table_name: str) -> bool:
    return (
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
            (table_name,),
        ).fetchone()
        is not None
    )


def _columns(cursor: Any, table_name: str) -> list[str]:
    return [str(row[1]) for row in cursor.execute(f"PRAGMA table_info({table_name})")]


def _schema_columns(cursor: Any) -> dict[str, list[str]]:
    tables = [
        str(row[0])
        for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
        )
    ]
    return {table_name: _columns(cursor, table_name) for table_name in tables}


def _db_file_bytes(db_path: str | Path) -> int:
    return _resolve(db_path).stat().st_size


def dbstat_sizes(db_path: str | Path, *, allow_live: bool = False) -> dict[str, int]:
    """Return per-object byte sizes from SQLite dbstat."""
    assert_not_live_db(db_path, allow_live=allow_live)
    conn = _connect(db_path)
    try:
        return {
            str(name): int(size)
            for name, size in conn.cursor().execute(
                "SELECT name, SUM(pgsize) FROM dbstat GROUP BY name ORDER BY SUM(pgsize) DESC"
            )
        }
    finally:
        conn.close()


def _ensure_meta(cursor: Any) -> None:
    cursor.execute("CREATE TABLE IF NOT EXISTS brainlayer_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")


def set_fts_mode(cursor: Any, mode: str) -> None:
    _ensure_meta(cursor)
    cursor.execute(
        "INSERT INTO brainlayer_meta(key, value) VALUES ('fts_mode', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (mode,),
    )


def get_fts_mode(cursor: Any) -> str | None:
    if not _table_exists(cursor, "brainlayer_meta"):
        return None
    row = cursor.execute("SELECT value FROM brainlayer_meta WHERE key = 'fts_mode'").fetchone()
    return str(row[0]) if row else None


def drop_fts_triggers(cursor: Any) -> None:
    for name in (
        "chunks_fts_insert",
        "chunks_fts_delete",
        "chunks_fts_update",
        "chunks_fts_trigram_insert",
        "chunks_fts_trigram_delete",
        "chunks_fts_trigram_update",
    ):
        cursor.execute(f"DROP TRIGGER IF EXISTS {name}")


def create_single_trigram_fts_schema(cursor: Any) -> None:
    """Create the compact FTS schema and triggers used by the migration."""
    cursor.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            {FTS_COLUMNS},
            tokenize='trigram'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunk_fts_rowids (
            chunk_id TEXT PRIMARY KEY,
            fts_rowid INTEGER,
            trigram_rowid INTEGER
        )
    """)
    cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_insert")
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
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
        END
    """)
    cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_delete")
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
            DELETE FROM chunks_fts
            WHERE rowid = (SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
            DELETE FROM chunk_fts_rowids WHERE chunk_id = old.id;
        END
    """)
    cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_update")
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_update
        AFTER UPDATE OF content, summary, tags, resolved_query, key_facts, resolved_queries ON chunks BEGIN
            DELETE FROM chunks_fts
            WHERE rowid = (SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
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
        END
    """)
    cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_insert")
    cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_delete")
    cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_update")


def create_compact_dual_fts_schema(cursor: Any) -> None:
    """Create current compact dual FTS schema: default FTS plus trigram FTS."""
    cursor.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            {FTS_COLUMNS}
        )
    """)
    cursor.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_trigram USING fts5(
            {FTS_COLUMNS},
            tokenize='trigram'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunk_fts_rowids (
            chunk_id TEXT PRIMARY KEY,
            fts_rowid INTEGER,
            trigram_rowid INTEGER
        )
    """)
    cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_insert")
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
            VALUES (new.content, new.summary, new.tags, new.resolved_query, new.key_facts, new.resolved_queries, new.id);
            INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid)
            VALUES (new.id, last_insert_rowid())
            ON CONFLICT(chunk_id) DO UPDATE SET fts_rowid = excluded.fts_rowid;
        END
    """)
    cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_insert")
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_trigram_insert AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
            VALUES (new.content, new.summary, new.tags, new.resolved_query, new.key_facts, new.resolved_queries, new.id);
            INSERT INTO chunk_fts_rowids(chunk_id, trigram_rowid)
            VALUES (new.id, last_insert_rowid())
            ON CONFLICT(chunk_id) DO UPDATE SET trigram_rowid = excluded.trigram_rowid;
        END
    """)
    cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_delete")
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
            DELETE FROM chunks_fts
            WHERE rowid = (SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
            DELETE FROM chunks_fts_trigram
            WHERE rowid = (SELECT trigram_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
            DELETE FROM chunk_fts_rowids WHERE chunk_id = old.id;
        END
    """)
    cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_delete")
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_trigram_delete AFTER DELETE ON chunks BEGIN
            DELETE FROM chunks_fts
            WHERE rowid = (SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
            DELETE FROM chunks_fts_trigram
            WHERE rowid = (SELECT trigram_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
            DELETE FROM chunk_fts_rowids WHERE chunk_id = old.id;
        END
    """)
    cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_update")
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_update
        AFTER UPDATE OF content, summary, tags, resolved_query, key_facts, resolved_queries ON chunks BEGIN
            DELETE FROM chunks_fts
            WHERE rowid = (SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
            INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
            VALUES (new.content, new.summary, new.tags, new.resolved_query, new.key_facts, new.resolved_queries, new.id);
            INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid)
            VALUES (new.id, last_insert_rowid())
            ON CONFLICT(chunk_id) DO UPDATE SET fts_rowid = excluded.fts_rowid;
        END
    """)
    cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_update")
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_trigram_update
        AFTER UPDATE OF content, summary, tags, resolved_query, key_facts, resolved_queries ON chunks BEGIN
            DELETE FROM chunks_fts_trigram
            WHERE rowid = (SELECT trigram_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
            INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
            VALUES (new.content, new.summary, new.tags, new.resolved_query, new.key_facts, new.resolved_queries, new.id);
            INSERT INTO chunk_fts_rowids(chunk_id, trigram_rowid)
            VALUES (new.id, last_insert_rowid())
            ON CONFLICT(chunk_id) DO UPDATE SET trigram_rowid = excluded.trigram_rowid;
        END
    """)


def _rebuild_fts(
    db_path: str | Path,
    *,
    mode: str,
    allow_live: bool = False,
) -> FtsMigrationResult:
    assert_not_live_db(db_path, allow_live=allow_live)
    path = _resolve(db_path)
    before_bytes = _db_file_bytes(path)
    conn = _connect(path)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA wal_checkpoint(FULL)")
        cursor.execute("BEGIN IMMEDIATE")
        try:
            drop_fts_triggers(cursor)
            cursor.execute("DROP TABLE IF EXISTS chunks_fts")
            cursor.execute("DROP TABLE IF EXISTS chunks_fts_trigram")
            cursor.execute("DELETE FROM chunk_fts_rowids")
            if mode == FTS_MODE_SINGLE_TRIGRAM:
                create_single_trigram_fts_schema(cursor)
            elif mode == FTS_MODE_COMPACT_DUAL:
                create_compact_dual_fts_schema(cursor)
            else:
                raise ValueError(f"Unsupported FTS mode: {mode}")
            cursor.execute(f"""
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                SELECT {FTS_VALUE_COLUMNS} FROM chunks
            """)
            if mode == FTS_MODE_COMPACT_DUAL:
                cursor.execute(f"""
                    INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                    SELECT {FTS_VALUE_COLUMNS} FROM chunks
                """)
                cursor.execute("""
                    INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid)
                    SELECT chunk_id, rowid FROM chunks_fts WHERE chunk_id IS NOT NULL
                    ON CONFLICT(chunk_id) DO UPDATE SET fts_rowid = excluded.fts_rowid
                """)
                cursor.execute("""
                    INSERT INTO chunk_fts_rowids(chunk_id, trigram_rowid)
                    SELECT chunk_id, rowid FROM chunks_fts_trigram WHERE chunk_id IS NOT NULL
                    ON CONFLICT(chunk_id) DO UPDATE SET trigram_rowid = excluded.trigram_rowid
                """)
            else:
                cursor.execute("""
                    INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid, trigram_rowid)
                    SELECT chunk_id, rowid, NULL FROM chunks_fts WHERE chunk_id IS NOT NULL
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        fts_rowid = excluded.fts_rowid,
                        trigram_rowid = NULL
                """)
            set_fts_mode(cursor, mode)
            chunk_count = int(cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            fts_count = int(cursor.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0])
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        cursor.execute("PRAGMA wal_checkpoint(FULL)")
    finally:
        conn.close()
    after_bytes = _db_file_bytes(path)
    return FtsMigrationResult(
        db_path=str(path),
        mode=mode,
        chunk_count=chunk_count,
        fts_count=fts_count,
        before_bytes=before_bytes,
        after_bytes=after_bytes,
        reclaimed_bytes=max(0, before_bytes - after_bytes),
    )


def migrate_fts_single_trigram(db_path: str | Path, *, allow_live: bool = False) -> FtsMigrationResult:
    """Replace dual/prefix FTS tables with one compact trigram FTS table."""
    return _rebuild_fts(db_path, mode=FTS_MODE_SINGLE_TRIGRAM, allow_live=allow_live)


def migrate_fts_compact_dual(db_path: str | Path, *, allow_live: bool = False) -> FtsMigrationResult:
    """Rebuild FTS without legacy prefix bloat while preserving trigram search."""
    return _rebuild_fts(db_path, mode=FTS_MODE_COMPACT_DUAL, allow_live=allow_live)


def vacuum_database(db_path: str | Path, *, allow_live: bool = False) -> VacuumResult:
    """Physically compact a snapshot DB after logical shrink operations."""
    assert_not_live_db(db_path, allow_live=allow_live)
    path = _resolve(db_path)
    before_bytes = _db_file_bytes(path)
    conn = _connect(path)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA wal_checkpoint(FULL)")
        cursor.execute("VACUUM")
        cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    after_bytes = _db_file_bytes(path)
    return VacuumResult(
        db_path=str(path),
        before_bytes=before_bytes,
        after_bytes=after_bytes,
        reclaimed_bytes=max(0, before_bytes - after_bytes),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--skip-fts", action="store_true")
    parser.add_argument("--fts-mode", choices=["compact-dual", "single-trigram"], default="compact-dual")
    parser.add_argument("--vacuum", action="store_true", help="Physically compact the snapshot after migrations")
    parser.add_argument("--i-know-this-is-live", action="store_true")
    args = parser.parse_args(argv)

    allow_live = bool(args.i_know_this_is_live)
    assert_not_live_db(args.db_path, allow_live=allow_live)
    started = time.time()
    results: dict[str, Any] = {"db_path": str(_resolve(args.db_path))}
    fts_mode = FTS_MODE_SINGLE_TRIGRAM if args.fts_mode == "single-trigram" else FTS_MODE_COMPACT_DUAL
    if not args.skip_fts:
        results["fts"] = asdict(_rebuild_fts(args.db_path, mode=fts_mode, allow_live=allow_live))
    if args.vacuum:
        results["vacuum"] = asdict(vacuum_database(args.db_path, allow_live=allow_live))
    results["elapsed_seconds"] = round(time.time() - started, 3)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
