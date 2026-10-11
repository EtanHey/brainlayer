"""Supervised offline migration for the chunks.source_class contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sqlite_vec

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


def _validate_target(db_path: str | Path, git_sha: str, *, read_only: bool = False) -> Path:
    if not _SHA_RE.fullmatch(git_sha):
        raise ValueError("git_sha must be an exact 40-character hexadecimal commit SHA")
    path = _resolved(db_path)
    for canonical in (get_db_path(), Path.home() / ".local/share/brainlayer/brainlayer.db"):
        canonical = _resolved(canonical)
        if path == canonical or (path.exists() and canonical.exists() and path.samefile(canonical)):
            if not read_only and os.environ.get("BRAINLAYER_OFFLINE_MIGRATOR_GATED_SWAP") != "1":
                raise ValueError(
                    "source_class migration refuses the canonical BrainLayer database unless "
                    "BRAINLAYER_OFFLINE_MIGRATOR_GATED_SWAP=1 is set for the supervised live run"
                )
            # A hardlink must use the canonical WAL/SHM paths as well as the gate.
            path = canonical
            break
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


def _load_vec0_if_needed(conn: sqlite3.Connection) -> None:
    uses_vec0 = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND sql LIKE '%USING vec0%' LIMIT 1"
    ).fetchone()
    if uses_vec0 is None:
        return
    conn.enable_load_extension(True)
    try:
        conn.load_extension(sqlite_vec.loadable_path())
    finally:
        conn.enable_load_extension(False)


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
    target = "SELECT id FROM chunks WHERE source_class = 'brain-worker'"
    for table in ("chunks_fts", "chunks_fts_operational", "chunks_fts_trigram", "chunk_fts_rowids"):
        if table not in tables or "chunk_id" not in _columns(conn, table):
            continue
        count = int(conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE chunk_id IN ({target})').fetchone()[0])
        conn.execute(f'DELETE FROM "{table}" WHERE chunk_id IN ({target})')
        removed[table] = count
    return removed


def _remove_brain_worker_vector_rows(conn: sqlite3.Connection) -> dict[str, int]:
    removed: dict[str, int] = {}
    tables = _tables(conn)
    target = "SELECT id FROM chunks WHERE source_class = 'brain-worker'"
    for table in ("chunk_vectors", "chunk_vectors_binary"):
        if table not in tables or "chunk_id" not in _columns(conn, table):
            continue
        count = int(conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE chunk_id IN ({target})').fetchone()[0])
        conn.execute(f'DELETE FROM "{table}" WHERE chunk_id IN ({target})')
        removed[table] = count
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
    _load_vec0_if_needed(conn)
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA foreign_keys = ON")
    prior_fingerprint = _schema_fingerprint(conn)
    row_count_before = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    _ensure_ledgers(conn)
    prior = conn.execute("SELECT details FROM schema_migrations WHERE name = ?", (MIGRATION_NAME,)).fetchone()
    if prior is not None and "source_class" in _columns(conn, "chunks"):
        try:
            receipt = json.loads(prior[0]) if prior[0] else {}
            recorded_sha = str(receipt.get("git_sha") or "").casefold()
            if recorded_sha != git_sha.casefold():
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
            return receipt
        finally:
            conn.close()

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
        vector_removed = _remove_brain_worker_vector_rows(conn)
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
            "vector_rows_removed": vector_removed,
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
                json.dumps(
                    [
                        "chunks.source_class",
                        "idx_chunks_source_class",
                        "chunks_fts",
                        "chunk_vectors",
                        "chunk_vectors_binary",
                    ]
                ),
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


def _checkpoint_full(conn: sqlite3.Connection) -> None:
    if conn.execute("PRAGMA wal_checkpoint(FULL)").fetchone()[0] != 0:
        raise RuntimeError("WAL checkpoint blocked; quiesce all writers before retrying")


_T3_ORIGIN = """(source = 't3' OR source_file LIKE '%/.codex/sessions/%'
                   OR provenance_class IN ('t3-thread', 't3-app-session'))
                AND source NOT IN ('chatgpt', 'claude_desktop', 'gemini_desktop')"""
_T3_CANDIDATE = f"source_class = 'desktop' AND {_T3_ORIGIN}"


def _changed_ids_hash(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def _t3_distribution(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        f"{provenance}/{source_class or 'NULL'}": int(count)
        for provenance, source_class, count in conn.execute(
            f"""SELECT provenance_class, source_class, COUNT(*) FROM chunks
               WHERE {_T3_ORIGIN}
               GROUP BY provenance_class, source_class ORDER BY provenance_class, source_class"""
        )
    }


def reclassify_t3_source_class(
    db_path: str | Path,
    *,
    git_sha: str,
    actor: str,
    batch_size: int = 5_000,
    apply: bool = False,
) -> dict[str, Any]:
    """Correct desktop T3 rows on a quiesced copy or gated live DB; default is read-only.

    This is separate from the initial NULL backfill and its one-time ledger:
    existing desktop labels must be corrected even if that ledger already exists.
    """
    path = _validate_target(db_path, git_sha, read_only=not apply)
    if not 5_000 <= batch_size <= 10_000:
        raise ValueError("batch_size must be between 5,000 and 10,000")
    if not actor.strip():
        raise ValueError("actor must be non-empty")

    conn = sqlite3.connect(f"{path.as_uri()}?mode={'rw' if apply else 'ro'}", uri=True, timeout=60.0)
    try:
        before = _t3_distribution(conn)
        candidate_rows = int(conn.execute(f"SELECT COUNT(*) FROM chunks WHERE {_T3_CANDIDATE}").fetchone()[0])
        updated_rows = batches = 0
        changed_ids: list[str] = []
        ledger_event_ids: list[str] = []
        run_id = str(uuid.uuid4())
        if apply and candidate_rows:
            _load_vec0_if_needed(conn)
            _checkpoint_full(conn)
            _ensure_ledgers(conn)
            conn.commit()
            last_rowid = -(2**63)
            while True:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(
                    f"""SELECT rowid, id FROM chunks
                        WHERE rowid > ? AND {_T3_CANDIDATE} ORDER BY rowid LIMIT ?""",
                    (last_rowid, batch_size),
                ).fetchall()
                if not rows:
                    conn.commit()
                    break
                cursor = conn.executemany(
                    f"UPDATE chunks SET source_class = 'cli-agent' WHERE rowid = ? AND {_T3_CANDIDATE}",
                    [(rowid,) for rowid, _ in rows],
                )
                if cursor.rowcount != len(rows):
                    raise RuntimeError("T3 candidate rows changed inside the migration batch")
                batch_ids = sorted(str(chunk_id) for _, chunk_id in rows)
                event_id = f"source-class:t3:{run_id}:{batches + 1}"
                details = {
                    "run_id": run_id,
                    "batch": batches + 1,
                    "changed_ids": batch_ids,
                    "changed_ids_sha256": _changed_ids_hash(batch_ids),
                    "changed_count": cursor.rowcount,
                    "prior_class": "desktop",
                    "result_class": "cli-agent",
                }
                conn.execute(
                    """INSERT INTO migration_events (
                        id, from_pattern, to_pattern, commit_hash, repo, detected_at,
                        confidence, memories_weakened, actor, path, affected_objects, status, details
                    ) VALUES (?, 'desktop T3 rows', 'cli-agent', ?, 'brainlayer', ?,
                              1.0, 0, ?, ?, '["chunks.source_class"]', 'success', ?)""",
                    (event_id, git_sha.lower(), time.time(), actor, str(path), json.dumps(details, sort_keys=True)),
                )
                updated_rows += cursor.rowcount
                last_rowid = int(rows[-1][0])
                conn.commit()
                changed_ids.extend(batch_ids)
                ledger_event_ids.append(event_id)
                batches += 1
                if batches % 3 == 0:
                    _checkpoint_full(conn)
            _checkpoint_full(conn)
        return {
            "migration": "2026_10_11_t3_source_class",
            "git_sha": git_sha.lower(),
            "actor": actor,
            "database": str(path),
            "dry_run": not apply,
            "batch_size": batch_size,
            "candidate_rows": candidate_rows,
            "updated_rows": updated_rows,
            "batches": batches,
            "ledger_event_ids": ledger_event_ids,
            "changed_ids_sha256": _changed_ids_hash(changed_ids),
            "before": before,
            "after": _t3_distribution(conn),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
