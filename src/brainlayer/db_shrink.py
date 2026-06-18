"""Offline BrainLayer DB shrink migrations.

These helpers are intentionally path-parameterized. They are for snapshot
maintenance first; the canonical live DB requires an explicit guard override.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import apsw
import sqlite_vec

from brainlayer.dedupe import ensure_dedupe_schema, normalized_exact_hash
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
class DedupResult:
    db_path: str
    scanned_rows: int
    duplicate_groups: int
    duplicate_rows: int
    deleted_rows: int
    protected_chunk_ids: int


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


def _canonical_for_group(rows: list[tuple[str, str | None]], protected_chunk_ids: set[str]) -> str:
    ordered = sorted(rows, key=lambda row: (row[0] not in protected_chunk_ids, row[1] or "", row[0]))
    return ordered[0][0]


def analyze_content_duplicates(
    db_path: str | Path, *, allow_live: bool = False
) -> tuple[int, dict[str, list[tuple[str, str | None]]]]:
    """Return scanned row count and exact normalized-content duplicate groups."""
    assert_not_live_db(db_path, allow_live=allow_live)
    conn = _connect(db_path)
    groups: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    scanned = 0
    try:
        for chunk_id, content, created_at in conn.cursor().execute("SELECT id, content, created_at FROM chunks"):
            scanned += 1
            if content is None:
                continue
            content_hash = normalized_exact_hash(str(content))
            groups[content_hash].append((str(chunk_id), str(created_at) if created_at is not None else None))
    finally:
        conn.close()
    return scanned, {content_hash: rows for content_hash, rows in groups.items() if len(rows) > 1}


def _insert_or_ignore_repoint(
    cursor: Any,
    schema: dict[str, list[str]],
    table_name: str,
    duplicate_id: str,
    canonical_id: str,
) -> None:
    cols = schema.get(table_name, [])
    if "chunk_id" not in cols:
        return
    column_sql = ", ".join(cols)
    select_sql = ", ".join("? AS chunk_id" if col == "chunk_id" else col for col in cols)
    cursor.execute(
        f"INSERT OR IGNORE INTO {table_name} ({column_sql}) SELECT {select_sql} FROM {table_name} WHERE chunk_id = ?",
        (canonical_id, duplicate_id),
    )
    cursor.execute(f"DELETE FROM {table_name} WHERE chunk_id = ?", (duplicate_id,))


def _update_or_ignore(
    cursor: Any,
    schema: dict[str, list[str]],
    table_name: str,
    column_name: str,
    duplicate_id: str,
    canonical_id: str,
) -> None:
    if column_name not in schema.get(table_name, []):
        return
    cursor.execute(
        f"UPDATE OR IGNORE {table_name} SET {column_name} = ? WHERE {column_name} = ?",
        (canonical_id, duplicate_id),
    )


def _delete_by_chunk_id(cursor: Any, schema: dict[str, list[str]], table_name: str, duplicate_id: str) -> None:
    if "chunk_id" in schema.get(table_name, []):
        cursor.execute(f"DELETE FROM {table_name} WHERE chunk_id = ?", (duplicate_id,))


def _delete_fts_rows_for_chunk(cursor: Any, schema: dict[str, list[str]], duplicate_id: str) -> None:
    if "chunk_fts_rowids" not in schema:
        return
    row = cursor.execute(
        "SELECT fts_rowid, trigram_rowid FROM chunk_fts_rowids WHERE chunk_id = ?",
        (duplicate_id,),
    ).fetchone()
    if row is None:
        return
    fts_rowid, trigram_rowid = row
    if fts_rowid is not None and "chunks_fts" in schema:
        cursor.execute("DELETE FROM chunks_fts WHERE rowid = ?", (fts_rowid,))
    if trigram_rowid is not None and "chunks_fts_trigram" in schema:
        cursor.execute("DELETE FROM chunks_fts_trigram WHERE rowid = ?", (trigram_rowid,))


def _record_alias(cursor: Any, duplicate_id: str, canonical_id: str) -> None:
    cursor.execute(
        """
        INSERT INTO chunk_id_alias(old_chunk_id, canonical_chunk_id, deprecated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(old_chunk_id) DO UPDATE SET
            canonical_chunk_id = excluded.canonical_chunk_id,
            deprecated_at = excluded.deprecated_at
        """,
        (duplicate_id, canonical_id),
    )
    cursor.execute(
        """
        INSERT INTO dedupe_audit(chunk_id_dropped, chunk_id_kept, mechanism, hamming_distance, ts)
        VALUES (?, ?, 'normalized_content_physical_delete', 0, datetime('now'))
        """,
        (duplicate_id, canonical_id),
    )


def _merge_duplicate_references(
    cursor: Any,
    schema: dict[str, list[str]],
    duplicate_id: str,
    canonical_id: str,
    *,
    delete_fts_rows: bool = False,
) -> None:
    for table_name in ("chunk_tags", "kg_entity_chunks", "chunk_clusters", "agent_reads"):
        if table_name in schema:
            _insert_or_ignore_repoint(cursor, schema, table_name, duplicate_id, canonical_id)

    # sqlite-vec virtual tables can raise a primary-key error even with
    # INSERT OR IGNORE. Duplicate chunks do not need duplicate vectors.
    for table_name in ("chunk_vectors", "chunk_vectors_binary"):
        _delete_by_chunk_id(cursor, schema, table_name, duplicate_id)

    if delete_fts_rows:
        _delete_fts_rows_for_chunk(cursor, schema, duplicate_id)
    _delete_by_chunk_id(cursor, schema, "chunk_fts_rowids", duplicate_id)
    _record_alias(cursor, duplicate_id, canonical_id)
    cursor.execute("DELETE FROM chunks WHERE id = ?", (duplicate_id,))


def _bulk_update_alias_refs(
    cursor: Any,
    schema: dict[str, list[str]],
    table_name: str,
    column_name: str,
) -> None:
    if column_name not in schema.get(table_name, []):
        return
    cursor.execute(f"""
        UPDATE OR IGNORE {table_name}
        SET {column_name} = (
            SELECT canonical_chunk_id
            FROM chunk_id_alias
            WHERE old_chunk_id = {table_name}.{column_name}
        )
        WHERE {column_name} IN (SELECT old_chunk_id FROM chunk_id_alias)
    """)


def _bulk_repoint_direct_refs(cursor: Any, schema: dict[str, list[str]]) -> None:
    for table_name in ("chunk_events", "correction_pairs", "file_interactions"):
        _bulk_update_alias_refs(cursor, schema, table_name, "chunk_id")
    _bulk_update_alias_refs(cursor, schema, "kg_relations", "source_chunk_id")


def _bulk_repoint_chunk_self_refs(cursor: Any, schema: dict[str, list[str]]) -> None:
    for column_name in ("superseded_by", "aggregated_into", "consolidated_into"):
        if column_name not in schema.get("chunks", []):
            continue
        cursor.execute(f"""
            UPDATE OR IGNORE chunks
            SET {column_name} = (
                SELECT canonical_chunk_id
                FROM chunk_id_alias
                WHERE old_chunk_id = chunks.{column_name}
            )
            WHERE {column_name} IN (SELECT old_chunk_id FROM chunk_id_alias)
        """)


def apply_content_dedup(
    db_path: str | Path,
    *,
    protected_chunk_ids: Iterable[str] = (),
    batch_size: int = 1000,
    allow_live: bool = False,
    rebuild_fts: bool = True,
    fts_mode: str = FTS_MODE_COMPACT_DUAL,
) -> DedupResult:
    """Physically delete exact normalized-content duplicates from a snapshot DB."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    assert_not_live_db(db_path, allow_live=allow_live)
    protected = {str(chunk_id) for chunk_id in protected_chunk_ids}
    scanned, groups = analyze_content_duplicates(db_path, allow_live=allow_live)
    duplicates: list[tuple[str, str]] = []
    for rows in groups.values():
        canonical_id = _canonical_for_group(rows, protected)
        duplicates.extend((chunk_id, canonical_id) for chunk_id, _created_at in rows if chunk_id != canonical_id)

    path = _resolve(db_path)
    conn = _connect(path)
    deleted = 0
    try:
        cursor = conn.cursor()
        ensure_dedupe_schema(conn)
        if rebuild_fts:
            drop_fts_triggers(cursor)
        schema = _schema_columns(cursor)
        cursor.execute("PRAGMA wal_checkpoint(FULL)")
        for offset in range(0, len(duplicates), batch_size):
            cursor.execute("BEGIN IMMEDIATE")
            try:
                for duplicate_id, canonical_id in duplicates[offset : offset + batch_size]:
                    if cursor.execute("SELECT 1 FROM chunks WHERE id = ?", (duplicate_id,)).fetchone() is None:
                        continue
                    if cursor.execute("SELECT 1 FROM chunks WHERE id = ?", (canonical_id,)).fetchone() is None:
                        continue
                    _merge_duplicate_references(
                        cursor,
                        schema,
                        duplicate_id,
                        canonical_id,
                        delete_fts_rows=not rebuild_fts,
                    )
                    deleted += 1
                cursor.execute("COMMIT")
            except Exception:
                cursor.execute("ROLLBACK")
                raise
            cursor.execute("PRAGMA wal_checkpoint(FULL)")
        cursor.execute("BEGIN IMMEDIATE")
        try:
            _bulk_repoint_direct_refs(cursor, schema)
            _bulk_repoint_chunk_self_refs(cursor, schema)
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    if rebuild_fts:
        _rebuild_fts(path, mode=fts_mode, allow_live=allow_live)
    return DedupResult(
        db_path=str(path),
        scanned_rows=scanned,
        duplicate_groups=len(groups),
        duplicate_rows=len(duplicates),
        deleted_rows=deleted,
        protected_chunk_ids=len(protected),
    )


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


def load_protected_qrel_ids(qrels_path: str | Path | None) -> set[str]:
    if qrels_path is None:
        return set()
    payload = json.loads(Path(qrels_path).read_text())
    protected: set[str] = set()
    for judgments in payload.values():
        if isinstance(judgments, dict):
            protected.update(str(doc_id) for doc_id in judgments)
    return protected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--qrels-path")
    parser.add_argument("--skip-fts", action="store_true")
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument("--fts-mode", choices=["compact-dual", "single-trigram"], default="compact-dual")
    parser.add_argument("--vacuum", action="store_true", help="Physically compact the snapshot after migrations")
    parser.add_argument("--i-know-this-is-live", action="store_true")
    args = parser.parse_args(argv)

    allow_live = bool(args.i_know_this_is_live)
    assert_not_live_db(args.db_path, allow_live=allow_live)
    started = time.time()
    results: dict[str, Any] = {"db_path": str(_resolve(args.db_path))}
    fts_mode = FTS_MODE_SINGLE_TRIGRAM if args.fts_mode == "single-trigram" else FTS_MODE_COMPACT_DUAL
    if not args.skip_dedup:
        protected = load_protected_qrel_ids(args.qrels_path)
        results["dedup"] = asdict(
            apply_content_dedup(
                args.db_path,
                protected_chunk_ids=protected,
                batch_size=args.batch_size,
                allow_live=allow_live,
                rebuild_fts=not args.skip_fts,
                fts_mode=fts_mode,
            )
        )
    elif not args.skip_fts:
        results["fts"] = asdict(_rebuild_fts(args.db_path, mode=fts_mode, allow_live=allow_live))
    if args.vacuum:
        results["vacuum"] = asdict(vacuum_database(args.db_path, allow_live=allow_live))
    results["elapsed_seconds"] = round(time.time() - started, 3)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
