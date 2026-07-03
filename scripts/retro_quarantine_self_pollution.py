#!/usr/bin/env python3
"""Retroactively quarantine denylisted agent transcript chunks.

Dry run is the default. Production apply exists for the lead-gated Stage C, but
requires explicit confirmation flags and is not part of worker Stage A/B.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import apsw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brainlayer.ingest_denylist import is_denylisted
from brainlayer.paths import DEFAULT_DB_PATH
from brainlayer.provenance import AGENT_INFERENCE
from brainlayer.vector_store import VectorStore

FTS_COLUMNS = "content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id"
FTS_SELECT_COLUMNS = "content, summary, tags, resolved_query, key_facts, resolved_queries, id"
FTS_STATE_COLUMNS = "rowid, content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id"
QUARANTINE_MANIFEST_TABLE = "retro_self_pollution_quarantine_manifest"
DEFAULT_ESTIMATE = 244_152


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _quote_sqlite_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _provider(source_file: str) -> str:
    parts = Path(source_file).parts
    for marker, provider in (
        (".claude", "claude"),
        (".codex", "codex"),
        (".cursor", "cursor"),
        (".gemini", "gemini"),
    ):
        if marker in parts:
            return provider
    return "other"


def _is_direct_claude_session(source_file: str) -> bool:
    path = Path(os.path.abspath(os.path.expanduser(source_file)))
    parts = path.parts
    if ".claude" not in parts or "projects" not in parts:
        return False
    if "subagents" in parts:
        return False
    if any(part.startswith("wf_") for part in parts):
        return False
    return path.suffix == ".jsonl"


def _is_denylisted_source(source_file: str | None) -> bool:
    if not source_file or not str(source_file).strip():
        return False
    return is_denylisted(source_file)


def _batch(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _placeholders(values: list[Any]) -> str:
    if not values:
        raise ValueError("values must not be empty")
    return ", ".join("?" for _ in values)


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


def _ensure_manifest(cursor: apsw.Cursor) -> None:
    existing = cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (QUARANTINE_MANIFEST_TABLE,),
    ).fetchone()
    if existing is not None:
        pk_cols = [
            row[1]
            for row in sorted(
                cursor.execute(f"PRAGMA table_info({QUARANTINE_MANIFEST_TABLE})"),
                key=lambda item: item[5],
            )
            if row[5]
        ]
        if pk_cols != ["run_id", "chunk_id"]:
            old_table = f"{QUARANTINE_MANIFEST_TABLE}_old"
            cursor.execute(f"ALTER TABLE {QUARANTINE_MANIFEST_TABLE} RENAME TO {old_table}")
            cursor.execute(f"""
                CREATE TABLE {QUARANTINE_MANIFEST_TABLE} (
                    run_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    original_content_class TEXT,
                    original_provenance_class TEXT,
                    original_fts_rowid INTEGER,
                    original_trigram_rowid INTEGER,
                    original_operational_rowid INTEGER,
                    quarantined_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, chunk_id)
                )
            """)
            cursor.execute(f"""
                INSERT OR IGNORE INTO {QUARANTINE_MANIFEST_TABLE} (
                    run_id, chunk_id, original_content_class, original_provenance_class,
                    original_fts_rowid, original_trigram_rowid, original_operational_rowid, quarantined_at
                )
                SELECT run_id, chunk_id, original_content_class, original_provenance_class,
                       original_fts_rowid, original_trigram_rowid, original_operational_rowid, quarantined_at
                FROM {old_table}
            """)
            cursor.execute(f"DROP TABLE {old_table}")
            return
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {QUARANTINE_MANIFEST_TABLE} (
            run_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            original_content_class TEXT,
            original_provenance_class TEXT,
            original_fts_rowid INTEGER,
            original_trigram_rowid INTEGER,
            original_operational_rowid INTEGER,
            quarantined_at TEXT NOT NULL,
            PRIMARY KEY (run_id, chunk_id)
        )
    """)


def _recreate_fts_triggers(db_path: Path) -> None:
    store = VectorStore(db_path)
    store.close()


def _checkpoint(cursor: apsw.Cursor) -> None:
    cursor.execute("PRAGMA wal_checkpoint(FULL)")


def _optimize_fts(cursor: apsw.Cursor) -> None:
    for table_name in ("chunks_fts", "chunks_fts_operational", "chunks_fts_trigram"):
        try:
            cursor.execute(f"INSERT INTO {table_name}({table_name}) VALUES('optimize')")
        except apsw.SQLError:
            continue


def _vacuum_into(source_db_path: Path, snapshot_path: Path, *, replace_snapshot: bool = False) -> None:
    source_db_path = source_db_path.expanduser()
    snapshot_path = snapshot_path.expanduser()
    if source_db_path.resolve() == snapshot_path.resolve():
        raise ValueError("snapshot_path must differ from source db path")
    if snapshot_path.exists():
        if not replace_snapshot:
            raise FileExistsError(snapshot_path)
        snapshot_path.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(snapshot_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    conn = apsw.Connection(str(source_db_path), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        conn.cursor().execute(f"VACUUM INTO {_quote_sqlite_string(str(snapshot_path))}")
    finally:
        conn.close()


def _select_rows(db_path: str | Path) -> list[dict[str, Any]]:
    conn = apsw.Connection(str(_path(db_path)), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        cursor = conn.cursor()
        return [
            {
                "id": row[0],
                "source_file": row[1] or "",
                "source": row[2] or "",
                "content_class": row[3],
                "provenance_class": row[4],
            }
            for row in cursor.execute(
                "SELECT id, source_file, source, content_class, provenance_class FROM chunks ORDER BY id"
            )
        ]
    finally:
        conn.close()


def select_quarantine_ids(db_path: str | Path) -> list[str]:
    return [row["id"] for row in _select_rows(db_path) if _is_denylisted_source(row["source_file"])]


def run_dry_run(
    db_path: str | Path,
    *,
    sample_size: int = 100,
    random_seed: int = 0,
    estimate: int = DEFAULT_ESTIMATE,
) -> dict[str, Any]:
    rows = _select_rows(db_path)
    rng = random.Random(random_seed)
    candidates: list[dict[str, Any]] = []
    preserved: list[dict[str, Any]] = []
    provider_breakdown = {
        "quarantine_set": {"claude": 0, "codex": 0, "cursor": 0, "gemini": 0, "other": 0},
        "preserved": {"claude": 0, "codex": 0, "cursor": 0, "gemini": 0, "other": 0},
    }

    for row in rows:
        bucket = "quarantine_set" if _is_denylisted_source(row["source_file"]) else "preserved"
        provider_breakdown[bucket][_provider(row["source_file"])] += 1
        if bucket == "quarantine_set":
            candidates.append(row)
        else:
            preserved.append(row)
    provider_breakdown = {
        bucket: {provider: count for provider, count in counts.items() if count}
        for bucket, counts in provider_breakdown.items()
    }

    direct_source_files = sorted({row["source_file"] for row in rows if _is_direct_claude_session(row["source_file"])})
    direct_false_positives = [path for path in direct_source_files if _is_denylisted_source(path)]
    sample_count = min(sample_size, len(candidates))
    quarantine_sample = rng.sample(candidates, sample_count) if sample_count else []
    quarantine_sample_sources = [
        {"id": row["id"], "source_file": row["source_file"], "provider": _provider(row["source_file"])}
        for row in quarantine_sample
    ]
    direct_sample_count = min(sample_size, len(direct_source_files))
    direct_sample = rng.sample(direct_source_files, direct_sample_count) if direct_sample_count else []
    total = len(rows)
    quarantine_count = len(candidates)

    return {
        "dry_run": True,
        "db_path": str(_path(db_path)),
        "counts": {
            "quarantine_set": quarantine_count,
            "preserved": len(preserved),
            "total": total,
        },
        "percent_quarantine": round((quarantine_count / total * 100), 2) if total else 0.0,
        "estimate_reconciliation": {
            "estimate": estimate,
            "actual": quarantine_count,
            "delta": quarantine_count - estimate,
        },
        "provider_breakdown": provider_breakdown,
        "audit": {
            "quarantine_sample_size": len(quarantine_sample_sources),
            "quarantine_sample_denylist_confirmed": sum(
                1 for row in quarantine_sample if _is_denylisted_source(row["source_file"])
            ),
            "quarantine_sample_sources": quarantine_sample_sources,
            "direct_session_source_files_checked": len(direct_source_files),
            "direct_session_sample_size": len(direct_sample),
            "direct_session_sample_source_files": direct_sample,
            "direct_session_false_positive_count": len(direct_false_positives),
            "direct_session_false_positive_source_files": direct_false_positives,
            "stop_gate_passed": len(direct_false_positives) == 0,
        },
    }


def capture_restore_state(cursor: apsw.Cursor, chunk_ids: list[str]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for chunk_id in sorted(chunk_ids):
        chunk = cursor.execute(
            "SELECT id, content_class, provenance_class FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if chunk is None:
            raise ValueError(f"missing chunk: {chunk_id}")
        fts_rows = {}
        for table_name in ("chunks_fts", "chunks_fts_operational", "chunks_fts_trigram"):
            fts_rows[table_name] = [
                tuple(row)
                for row in cursor.execute(
                    f"SELECT {FTS_STATE_COLUMNS} FROM {table_name} WHERE chunk_id = ? ORDER BY rowid",
                    (chunk_id,),
                )
            ]
        state[chunk_id] = {
            "chunk": tuple(chunk),
            "fts": fts_rows,
        }
    return state


def _record_manifest(cursor: apsw.Cursor, chunk_ids: list[str], *, run_id: str, timestamp: str) -> None:
    for chunk_id in chunk_ids:
        row = cursor.execute(
            """
            SELECT c.content_class,
                   c.provenance_class,
                   (SELECT rowid FROM chunks_fts WHERE chunk_id = c.id ORDER BY rowid LIMIT 1),
                   (SELECT rowid FROM chunks_fts_trigram WHERE chunk_id = c.id ORDER BY rowid LIMIT 1),
                   (SELECT rowid FROM chunks_fts_operational WHERE chunk_id = c.id ORDER BY rowid LIMIT 1)
            FROM chunks c
            WHERE c.id = ?
            """,
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"missing chunk: {chunk_id}")
        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {QUARANTINE_MANIFEST_TABLE} (
                run_id, chunk_id, original_content_class, original_provenance_class,
                original_fts_rowid, original_trigram_rowid, original_operational_rowid, quarantined_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, chunk_id, row[0], row[1], row[2], row[3], row[4], timestamp),
        )


def _delete_fts_rows(cursor: apsw.Cursor, chunk_ids: list[str]) -> None:
    placeholders = _placeholders(chunk_ids)
    for table_name in ("chunks_fts", "chunks_fts_operational", "chunks_fts_trigram"):
        cursor.execute(f"DELETE FROM {table_name} WHERE chunk_id IN ({placeholders})", chunk_ids)
    cursor.execute(
        f"""
        UPDATE chunk_fts_rowids
        SET fts_rowid = NULL, trigram_rowid = NULL, operational_rowid = NULL
        WHERE chunk_id IN ({placeholders})
        """,
        chunk_ids,
    )


def _insert_operational_fts(cursor: apsw.Cursor, chunk_ids: list[str]) -> None:
    placeholders = _placeholders(chunk_ids)
    cursor.execute(
        f"""
        INSERT INTO chunks_fts_operational({FTS_COLUMNS})
        SELECT {FTS_SELECT_COLUMNS}
        FROM chunks
        WHERE id IN ({placeholders})
        ORDER BY id
        """,
        chunk_ids,
    )
    cursor.execute(
        f"""
        INSERT INTO chunk_fts_rowids(chunk_id, operational_rowid)
        SELECT chunk_id, rowid
        FROM chunks_fts_operational
        WHERE chunk_id IN ({placeholders})
        ON CONFLICT(chunk_id) DO UPDATE SET operational_rowid = excluded.operational_rowid
        """,
        chunk_ids,
    )


def _insert_fts_row(
    cursor: apsw.Cursor,
    *,
    table_name: str,
    chunk_id: str,
    rowid: int | None,
) -> int | None:
    if rowid is None:
        cursor.execute(
            f"INSERT INTO {table_name}({FTS_COLUMNS}) SELECT {FTS_SELECT_COLUMNS} FROM chunks WHERE id = ?",
            (chunk_id,),
        )
    else:
        cursor.execute(
            f"INSERT INTO {table_name}(rowid, {FTS_COLUMNS}) SELECT ?, {FTS_SELECT_COLUMNS} FROM chunks WHERE id = ?",
            (rowid, chunk_id),
        )
    stored = cursor.execute(f"SELECT rowid FROM {table_name} WHERE chunk_id = ?", (chunk_id,)).fetchone()
    return int(stored[0]) if stored else None


def apply_quarantine_ids(
    db_path: str | Path,
    chunk_ids: list[str],
    *,
    batch_size: int = 5_000,
    checkpoint_every: int = 3,
    run_id: str | None = None,
    bootstrap_schema: bool = True,
    finalize: bool = True,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not chunk_ids:
        return {"quarantined": 0, "run_id": run_id or ""}

    db_path = _path(db_path)
    if bootstrap_schema:
        _recreate_fts_triggers(db_path)
    run_id = run_id or f"retro-self-pollution-{_utc_now()}"
    conn = apsw.Connection(str(db_path))
    cursor = conn.cursor()
    _checkpoint(cursor)
    _drop_fts_triggers(cursor)
    _ensure_manifest(cursor)
    timestamp = _utc_now()
    batches_since_checkpoint = 0
    try:
        for ids in _batch(chunk_ids, batch_size):
            placeholders = _placeholders(ids)
            cursor.execute("BEGIN IMMEDIATE")
            _record_manifest(cursor, ids, run_id=run_id, timestamp=timestamp)
            _delete_fts_rows(cursor, ids)
            cursor.execute(
                f"""
                UPDATE chunks
                SET content_class = 'operational',
                    provenance_class = ?
                WHERE id IN ({placeholders})
                """,
                (AGENT_INFERENCE, *ids),
            )
            _insert_operational_fts(cursor, ids)
            cursor.execute("COMMIT")
            batches_since_checkpoint += 1
            if batches_since_checkpoint >= checkpoint_every:
                _checkpoint(cursor)
                batches_since_checkpoint = 0
    except Exception:
        if not conn.getautocommit():
            cursor.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    if finalize:
        _recreate_fts_triggers(db_path)
        conn = apsw.Connection(str(db_path))
        try:
            cursor = conn.cursor()
            _checkpoint(cursor)
            _optimize_fts(cursor)
            _checkpoint(cursor)
        finally:
            conn.close()
    return {"quarantined": len(chunk_ids), "run_id": run_id}


def unquarantine_ids(
    db_path: str | Path,
    chunk_ids: list[str],
    *,
    run_id: str,
    batch_size: int = 5_000,
    checkpoint_every: int = 3,
    bootstrap_schema: bool = True,
    finalize: bool = True,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not chunk_ids:
        return {"restored": 0, "run_id": run_id}

    db_path = _path(db_path)
    if bootstrap_schema:
        _recreate_fts_triggers(db_path)
    conn = apsw.Connection(str(db_path))
    cursor = conn.cursor()
    _checkpoint(cursor)
    _drop_fts_triggers(cursor)
    _ensure_manifest(cursor)
    restored = 0
    batches_since_checkpoint = 0
    try:
        for ids in _batch(chunk_ids, batch_size):
            cursor.execute("BEGIN IMMEDIATE")
            _delete_fts_rows(cursor, ids)
            for chunk_id in ids:
                manifest = cursor.execute(
                    f"""
                    SELECT original_content_class, original_provenance_class,
                           original_fts_rowid, original_trigram_rowid, original_operational_rowid
                    FROM {QUARANTINE_MANIFEST_TABLE}
                    WHERE chunk_id = ? AND run_id = ?
                    """,
                    (chunk_id, run_id),
                ).fetchone()
                if manifest is None:
                    raise ValueError(f"missing quarantine manifest row for {chunk_id!r} in run {run_id!r}")
                original_content_class, original_provenance_class = manifest[0], manifest[1]
                cursor.execute(
                    "UPDATE chunks SET content_class = ?, provenance_class = ? WHERE id = ?",
                    (original_content_class, original_provenance_class, chunk_id),
                )
                fts_rowid = None
                trigram_rowid = None
                operational_rowid = None
                if manifest[2] is not None:
                    fts_rowid = _insert_fts_row(
                        cursor,
                        table_name="chunks_fts",
                        chunk_id=chunk_id,
                        rowid=manifest[2],
                    )
                if manifest[3] is not None:
                    trigram_rowid = _insert_fts_row(
                        cursor,
                        table_name="chunks_fts_trigram",
                        chunk_id=chunk_id,
                        rowid=manifest[3],
                    )
                if manifest[4] is not None:
                    operational_rowid = _insert_fts_row(
                        cursor,
                        table_name="chunks_fts_operational",
                        chunk_id=chunk_id,
                        rowid=manifest[4],
                    )
                cursor.execute(
                    """
                    INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid, trigram_rowid, operational_rowid)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        fts_rowid = excluded.fts_rowid,
                        trigram_rowid = excluded.trigram_rowid,
                        operational_rowid = excluded.operational_rowid
                    """,
                    (chunk_id, fts_rowid, trigram_rowid, operational_rowid),
                )
                restored += 1
            cursor.execute("COMMIT")
            batches_since_checkpoint += 1
            if batches_since_checkpoint >= checkpoint_every:
                _checkpoint(cursor)
                batches_since_checkpoint = 0
    except Exception:
        if not conn.getautocommit():
            cursor.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    if finalize:
        _recreate_fts_triggers(db_path)
        conn = apsw.Connection(str(db_path))
        try:
            cursor = conn.cursor()
            _checkpoint(cursor)
            _optimize_fts(cursor)
            _checkpoint(cursor)
        finally:
            conn.close()
    return {"restored": restored, "run_id": run_id}


def run_revert_proof(
    db_path: str | Path,
    *,
    snapshot_path: str | Path,
    sample_size: int = 100,
    random_seed: int = 0,
    replace_snapshot: bool = False,
) -> dict[str, Any]:
    source_db_path = _path(db_path)
    snapshot = _path(snapshot_path)
    _vacuum_into(source_db_path, snapshot, replace_snapshot=replace_snapshot)
    _recreate_fts_triggers(snapshot)
    candidate_ids = select_quarantine_ids(snapshot)
    if not candidate_ids:
        raise ValueError("no denylisted chunks available for revert proof")
    rng = random.Random(random_seed)
    sample_ids = rng.sample(candidate_ids, min(sample_size, len(candidate_ids)))
    conn = apsw.Connection(str(snapshot))
    try:
        before = capture_restore_state(conn.cursor(), sample_ids)
    finally:
        conn.close()

    run_id = f"revert-proof-{_utc_now()}"
    proof_batch_size = max(1, min(500, len(sample_ids)))
    quarantine_report = apply_quarantine_ids(
        snapshot,
        sample_ids,
        batch_size=proof_batch_size,
        run_id=run_id,
        bootstrap_schema=False,
        finalize=False,
    )
    revert_report = unquarantine_ids(
        snapshot,
        sample_ids,
        run_id=run_id,
        batch_size=proof_batch_size,
        bootstrap_schema=False,
        finalize=False,
    )

    conn = apsw.Connection(str(snapshot))
    try:
        after = capture_restore_state(conn.cursor(), sample_ids)
    finally:
        conn.close()

    return {
        "source_db_path": str(source_db_path),
        "snapshot_db_path": str(snapshot),
        "sample_size": len(sample_ids),
        "sample_ids": sorted(sample_ids),
        "quarantine_report": quarantine_report,
        "revert_report": revert_report,
        "byte_identical_restoration": before == after,
    }


def run_apply(
    db_path: str | Path,
    *,
    backup_path: str | Path,
    batch_size: int = 5_000,
    checkpoint_every: int = 3,
    confirm_workers_stopped: bool = False,
    confirm_watcher_paused: bool = False,
) -> dict[str, Any]:
    if not confirm_workers_stopped or not confirm_watcher_paused:
        raise PermissionError("--apply requires --confirm-workers-stopped and --confirm-watcher-paused")
    db_path = _path(db_path)
    backup_path = _path(backup_path)
    _vacuum_into(db_path, backup_path, replace_snapshot=False)
    dry_run = run_dry_run(db_path)
    if not dry_run["audit"]["stop_gate_passed"]:
        raise RuntimeError("STOP gate failed: direct/control-session chunks are classified into quarantine set")
    ids = select_quarantine_ids(db_path)
    apply_report = apply_quarantine_ids(
        db_path,
        ids,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
    )
    after = run_dry_run(db_path)
    return {
        "dry_run_before": dry_run,
        "apply": apply_report,
        "dry_run_after": after,
        "backup_path": str(backup_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="BrainLayer SQLite DB path")
    parser.add_argument("--sample-size", type=int, default=100, help="Audit/revert sample size")
    parser.add_argument("--random-seed", type=int, default=0, help="Deterministic audit sample seed")
    parser.add_argument("--apply", action="store_true", help="Apply production quarantine. Default is dry run.")
    parser.add_argument("--revert-proof", action="store_true", help="Run Stage B against --snapshot-path")
    parser.add_argument("--snapshot-path", type=Path, help="VACUUM INTO snapshot path for --revert-proof")
    parser.add_argument("--replace-snapshot", action="store_true", help="Replace existing --snapshot-path")
    parser.add_argument("--backup-path", type=Path, help="Required production backup path for --apply")
    parser.add_argument("--batch-size", type=int, default=5_000, help="Apply batch size")
    parser.add_argument("--checkpoint-every", type=int, default=3, help="Checkpoint every N batches")
    parser.add_argument("--confirm-workers-stopped", action="store_true", help="Required for --apply")
    parser.add_argument("--confirm-watcher-paused", action="store_true", help="Required for --apply")
    args = parser.parse_args()

    if args.apply and args.revert_proof:
        parser.error("--apply and --revert-proof are mutually exclusive")
    if args.revert_proof:
        if args.snapshot_path is None:
            parser.error("--revert-proof requires --snapshot-path")
        report = run_revert_proof(
            args.db,
            snapshot_path=args.snapshot_path,
            sample_size=args.sample_size,
            random_seed=args.random_seed,
            replace_snapshot=args.replace_snapshot,
        )
    elif args.apply:
        if args.backup_path is None:
            parser.error("--apply requires --backup-path")
        report = run_apply(
            args.db,
            backup_path=args.backup_path,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            confirm_workers_stopped=args.confirm_workers_stopped,
            confirm_watcher_paused=args.confirm_watcher_paused,
        )
    else:
        report = run_dry_run(args.db, sample_size=args.sample_size, random_seed=args.random_seed)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
