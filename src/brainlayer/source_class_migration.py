"""Supervised offline migration for the chunks.source_class contract."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_provenance import derive_source_class
from .paths import get_db_path

MIGRATION_NAME = "2026_08_10_source_class_v1"
MIGRATION_EVENT_ID = "schema:2026_08_10_source_class_v1"
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SOURCE_CLASSES = (
    "cli-agent",
    "desktop",
    "subagent",
    "brain-worker",
    "fleet-coordination",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _validate_target(db_path: str | Path, git_sha: str) -> Path:
    if not _SHA_RE.fullmatch(git_sha):
        raise ValueError("git_sha must be an exact 40-character hexadecimal commit SHA")
    path = _resolved(db_path)
    if path == _resolved(get_db_path()):
        raise ValueError("source_class migration refuses the canonical BrainLayer database")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _schema_fingerprint(conn: sqlite3.Connection) -> str:
    rows = conn.execute("SELECT type, name, COALESCE(sql, '') FROM sqlite_master ORDER BY type, name").fetchall()
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _ensure_ledgers(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL,
            details TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS migration_events (
            id TEXT PRIMARY KEY,
            from_pattern TEXT NOT NULL,
            to_pattern TEXT NOT NULL,
            commit_hash TEXT NOT NULL,
            repo TEXT NOT NULL,
            detected_at REAL NOT NULL,
            confidence REAL,
            memories_weakened INTEGER DEFAULT 0,
            actor TEXT,
            path TEXT,
            prior_fingerprint TEXT,
            result_fingerprint TEXT,
            affected_objects TEXT,
            status TEXT,
            error TEXT,
            duration_seconds REAL,
            details TEXT
        )
        """
    )
    existing = _columns(conn, "migration_events")
    for name, sql_type in (
        ("actor", "TEXT"),
        ("path", "TEXT"),
        ("prior_fingerprint", "TEXT"),
        ("result_fingerprint", "TEXT"),
        ("affected_objects", "TEXT"),
        ("status", "TEXT"),
        ("error", "TEXT"),
        ("duration_seconds", "REAL"),
        ("details", "TEXT"),
    ):
        if name not in existing:
            conn.execute(f'ALTER TABLE migration_events ADD COLUMN "{name}" {sql_type}')


def _distribution(conn: sqlite3.Connection) -> dict[str, int]:
    counts = Counter()
    for source_class, count in conn.execute("SELECT source_class, COUNT(*) FROM chunks GROUP BY source_class"):
        counts[source_class if source_class is not None else "NULL"] += int(count)
    return dict(sorted(counts.items()))


def _validate_distribution(distribution: dict[str, int]) -> None:
    invalid = set(distribution) - {"NULL", *_SOURCE_CLASSES}
    if invalid:
        raise ValueError(f"unexpected source_class values: {sorted(invalid)}")


def _remove_brain_worker_fts_rows(conn: sqlite3.Connection) -> dict[str, int]:
    removed: dict[str, int] = {}
    tables = _tables(conn)
    for table in ("chunks_fts", "chunks_fts_operational", "chunks_fts_trigram"):
        if table not in tables or "chunk_id" not in _columns(conn, table):
            continue
        before = conn.total_changes
        conn.execute(
            f"DELETE FROM \"{table}\" WHERE chunk_id IN (SELECT id FROM chunks WHERE source_class = 'brain-worker')"
        )
        removed[table] = conn.total_changes - before
    if "chunk_fts_rowids" in tables:
        conn.execute(
            "DELETE FROM chunk_fts_rowids WHERE chunk_id IN (SELECT id FROM chunks WHERE source_class = 'brain-worker')"
        )
    return removed


def migrate_source_class(
    db_path: str | Path,
    *,
    git_sha: str,
    actor: str,
    batch_size: int = 5_000,
) -> dict[str, Any]:
    """Migrate one non-canonical DB and return an auditable receipt."""
    path = _validate_target(db_path, git_sha)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not actor.strip():
        raise ValueError("actor must be non-empty")

    started = time.monotonic()
    conn = sqlite3.connect(path, timeout=60.0)
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA foreign_keys = ON")
    prior_fingerprint = _schema_fingerprint(conn)
    row_count_before = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    _ensure_ledgers(conn)
    prior = conn.execute("SELECT details FROM schema_migrations WHERE name = ?", (MIGRATION_NAME,)).fetchone()
    if prior is not None and "source_class" in _columns(conn, "chunks"):
        receipt = json.loads(prior[0]) if prior[0] else {}
        recorded_sha = str(receipt.get("git_sha") or "").casefold()
        if recorded_sha != git_sha.casefold():
            conn.close()
            raise RuntimeError(
                f"migration ledger SHA mismatch: recorded {recorded_sha or 'missing'}, requested {git_sha.casefold()}"
            )
        distribution = _distribution(conn)
        _validate_distribution(distribution)
        receipt.update(
            {
                "already_applied": True,
                "row_count_before": row_count_before,
                "row_count_after": row_count_before,
                "distribution": distribution,
            }
        )
        conn.close()
        return receipt

    error: str | None = None
    try:
        if "source_class" not in _columns(conn, "chunks"):
            conn.execute("ALTER TABLE chunks ADD COLUMN source_class TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_class ON chunks(source_class)")
        conn.commit()

        select_columns = _columns(conn, "chunks")
        provenance_expr = "provenance_class" if "provenance_class" in select_columns else "NULL"
        source_expr = "source" if "source" in select_columns else "NULL"
        last_rowid = 0
        classified = 0
        ambiguous = 0
        batches = 0
        while True:
            rows = conn.execute(
                f"""
                SELECT rowid, id, source_file, {provenance_expr}, {source_expr}
                FROM chunks
                WHERE rowid > ? AND source_class IS NULL
                ORDER BY rowid
                LIMIT ?
                """,
                (last_rowid, batch_size),
            ).fetchall()
            if not rows:
                break
            updates: list[tuple[str, str]] = []
            for rowid, chunk_id, source_file, provenance_class, source in rows:
                last_rowid = int(rowid)
                source_class = derive_source_class(
                    str(source_file or ""), provenance_class=provenance_class, source=source
                )
                if source_class is None:
                    ambiguous += 1
                    continue
                updates.append((source_class, str(chunk_id)))
            if updates:
                conn.executemany("UPDATE chunks SET source_class = ? WHERE id = ?", updates)
                classified += len(updates)
            conn.commit()
            batches += 1
            if batches % 3 == 0:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()

        fts_removed = _remove_brain_worker_fts_rows(conn)
        distribution = _distribution(conn)
        _validate_distribution(distribution)
        row_count_after = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        result_fingerprint = _schema_fingerprint(conn)
        duration_seconds = time.monotonic() - started
        receipt: dict[str, Any] = {
            "migration": MIGRATION_NAME,
            "git_sha": git_sha.lower(),
            "database": str(path),
            "row_count_before": row_count_before,
            "row_count_after": row_count_after,
            "classified_rows": classified,
            "ambiguous_rows": ambiguous,
            "distribution": distribution,
            "batches": batches,
            "fts_rows_removed": fts_removed,
            "duration_seconds": duration_seconds,
            "already_applied": False,
        }
        details = json.dumps(receipt, sort_keys=True)
        conn.execute(
            "INSERT OR REPLACE INTO schema_migrations(name, applied_at, details) VALUES (?, ?, ?)",
            (MIGRATION_NAME, _utc_now(), details),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO migration_events(
                id, from_pattern, to_pattern, commit_hash, repo, detected_at,
                confidence, memories_weakened, actor, path, prior_fingerprint,
                result_fingerprint, affected_objects, status, error,
                duration_seconds, details
            ) VALUES (?, ?, ?, ?, 'brainlayer', ?, 1.0, 0, ?, ?, ?, ?, ?, 'success', NULL, ?, ?)
            """,
            (
                MIGRATION_EVENT_ID,
                "chunks without source_class",
                "chunks.source_class five-value taxonomy",
                git_sha.lower(),
                time.time(),
                actor,
                str(path),
                prior_fingerprint,
                result_fingerprint,
                json.dumps(["chunks.source_class", "idx_chunks_source_class", "chunks_fts"]),
                duration_seconds,
                details,
            ),
        )
        conn.commit()
        return receipt
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        conn.rollback()
        raise
    finally:
        if error is not None:
            try:
                _ensure_ledgers(conn)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO migration_events(
                        id, from_pattern, to_pattern, commit_hash, repo, detected_at,
                        confidence, memories_weakened, actor, path, prior_fingerprint,
                        affected_objects, status, error, duration_seconds
                    ) VALUES (?, ?, ?, ?, 'brainlayer', ?, 1.0, 0, ?, ?, ?, ?, 'failure', ?, ?)
                    """,
                    (
                        MIGRATION_EVENT_ID,
                        "chunks without source_class",
                        "chunks.source_class five-value taxonomy",
                        git_sha.lower(),
                        time.time(),
                        actor,
                        str(path),
                        prior_fingerprint,
                        json.dumps(["chunks.source_class"]),
                        error,
                        time.monotonic() - started,
                    ),
                )
                conn.commit()
            except sqlite3.Error:
                pass
        conn.close()
