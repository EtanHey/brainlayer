"""Collapse archive tetraplication onto chunks.archived_at (time or NULL).

Rehearsal/live-window tool. Default is dry-run. Refuses the live canonical DB unless
allow_live=True after a lead-scheduled writer window.

Twin representations that this repair migrates-then-clears:
  archived INTEGER, status='archived', value_type='ARCHIVED'
Canonical remainder: archived_at. Lineage columns superseded_by / aggregated_into are kept.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunk_origin_wipe import AUX_COUNT_TABLES, assert_not_live_db

PREIMAGE_TABLE = "archive_collapse_preimage"
MIGRATION_NAME = "2026_08_17_archive_collapse_archived_at"
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

TWIN_PREDICATE = """(
  COALESCE(archived, 0) = 1
  OR lower(COALESCE(value_type, '')) = 'archived'
  OR lower(COALESCE(status, '')) = 'archived'
)"""


@dataclass
class ArchiveCollapseResult:
    scanned: int = 0
    updated: int = 0
    would_update: int = 0
    batches: int = 0
    checkpoints: int = 0
    post_twin_count: int = 0
    aux_counts_before: dict[str, int] = field(default_factory=dict)
    aux_counts_after: dict[str, int] = field(default_factory=dict)
    spot_checks: list[dict[str, Any]] = field(default_factory=list)
    backfill_timestamp: str = ""


def _validate_git_sha(git_sha: str) -> str:
    if not _SHA_RE.fullmatch(git_sha):
        raise ValueError("git_sha must be an exact 40-character hexadecimal commit SHA")
    return git_sha.lower()


def _aux_counts(conn: sqlite3.Connection) -> dict[str, int]:
    existing = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    counts: dict[str, int] = {}
    for table in AUX_COUNT_TABLES:
        if table not in existing:
            continue
        try:
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        except sqlite3.OperationalError:
            continue
    return counts


def _checkpoint(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA wal_checkpoint(FULL)")


def _twin_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM chunks WHERE {TWIN_PREDICATE}").fetchone()[0])


def _ensure_preimage(conn: sqlite3.Connection) -> None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (PREIMAGE_TABLE,),
    ).fetchone()
    if exists:
        conn.execute(
            f"""
            INSERT INTO {PREIMAGE_TABLE}(id, archived_at, archived, status, value_type)
            SELECT id, archived_at, archived, status, value_type FROM chunks
            WHERE {TWIN_PREDICATE}
              AND id NOT IN (SELECT id FROM {PREIMAGE_TABLE})
            """
        )
        conn.commit()
        return
    conn.execute(
        f"""
        CREATE TABLE {PREIMAGE_TABLE} AS
        SELECT id, archived_at, archived, status, value_type
        FROM chunks
        WHERE {TWIN_PREDICATE}
        """
    )
    conn.commit()


def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL,
            details TEXT
        )
        """
    )


def _spot_ok(archived_at: Any, archived: Any, status: Any, value_type: Any) -> bool:
    if archived_at is None or str(archived_at).strip() == "":
        return False
    if int(archived or 0) != 0:
        return False
    if str(status or "").lower() == "archived":
        return False
    if str(value_type or "").lower() == "archived":
        return False
    return True


def _spot_checks_from_store(
    conn: sqlite3.Connection,
    sample_ids: Iterable[str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for chunk_id in sample_ids:
        row = conn.execute(
            "SELECT archived_at, archived, status, value_type FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            checks.append({"id": chunk_id, "ok": False})
            continue
        archived_at, archived, status, value_type = row
        checks.append(
            {
                "id": chunk_id,
                "stored_archived_at": archived_at,
                "stored_archived": archived,
                "stored_status": status,
                "stored_value_type": value_type,
                "ok": _spot_ok(archived_at, archived, status, value_type),
            }
        )
    return checks


def collapse_archive_representations(
    db_path: Path,
    *,
    git_sha: str,
    apply: bool = False,
    batch_size: int = 5000,
    checkpoint_every: int = 3,
    allow_live: bool = False,
    spot_check: int = 0,
    actor: str = "repair-c",
) -> ArchiveCollapseResult:
    """Backfill archived_at from twin flags, then clear the twins."""
    sha = _validate_git_sha(git_sha)
    resolved = assert_not_live_db(db_path, allow_live=allow_live)
    batch_size = max(1, int(batch_size))
    checkpoint_every = max(1, int(checkpoint_every))
    spot_check = max(0, int(spot_check))
    backfill_ts = datetime.now(timezone.utc).isoformat()

    result = ArchiveCollapseResult(backfill_timestamp=backfill_ts)
    samples: list[str] = []
    sampled = 0
    rng = random.Random(0)

    conn = sqlite3.connect(str(resolved))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        required = {"archived_at", "archived", "status", "value_type"}
        missing = sorted(required - columns)
        if missing:
            raise RuntimeError(f"chunks missing archive columns: {missing}")

        result.would_update = _twin_count(conn)
        result.aux_counts_before = _aux_counts(conn)

        if apply:
            _ensure_preimage(conn)
            _checkpoint(conn)
            result.checkpoints += 1

        last_rowid = 0
        while True:
            rows = list(
                conn.execute(
                    f"""
                    SELECT rowid, id, archived_at, archived, status, value_type,
                           superseded_by, aggregated_into
                    FROM chunks
                    WHERE rowid > ?
                      AND {TWIN_PREDICATE}
                    ORDER BY rowid
                    LIMIT ?
                    """,
                    (last_rowid, batch_size),
                )
            )
            if not rows:
                break

            result.batches += 1
            updates: list[tuple[Any, ...]] = []
            for row in rows:
                last_rowid = int(row["rowid"])
                result.scanned += 1
                archived_at = row["archived_at"]
                if archived_at is None or str(archived_at).strip() == "":
                    archived_at = backfill_ts
                status = row["status"]
                if row["superseded_by"] or row["aggregated_into"]:
                    status = "superseded"
                elif str(status or "").lower() == "archived":
                    status = "active"
                value_type = row["value_type"]
                if str(value_type or "").lower() == "archived":
                    value_type = None
                updates.append((archived_at, 0, status, value_type, int(row["rowid"])))
                if spot_check:
                    sampled += 1
                    chunk_id = str(row["id"])
                    if len(samples) < spot_check:
                        samples.append(chunk_id)
                    else:
                        j = rng.randrange(sampled)
                        if j < spot_check:
                            samples[j] = chunk_id

            if apply and updates:
                before_changes = conn.total_changes
                conn.executemany(
                    """
                    UPDATE chunks
                    SET archived_at = ?,
                        archived = ?,
                        status = ?,
                        value_type = ?
                    WHERE rowid = ?
                    """,
                    updates,
                )
                result.updated += conn.total_changes - before_changes
                conn.commit()
                if result.batches % checkpoint_every == 0:
                    _checkpoint(conn)
                    result.checkpoints += 1
            elif apply:
                conn.commit()

        if apply and result.batches and result.batches % checkpoint_every != 0:
            _checkpoint(conn)
            result.checkpoints += 1

        result.post_twin_count = _twin_count(conn)
        result.aux_counts_after = _aux_counts(conn)
        if apply:
            _ensure_schema_migrations(conn)
            receipt = {
                "migration": MIGRATION_NAME,
                "git_sha": sha,
                "actor": actor,
                "updated": result.updated,
                "post_twin_count": result.post_twin_count,
                "backfill_timestamp": backfill_ts,
            }
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations(name, applied_at, details) VALUES (?, ?, ?)",
                (MIGRATION_NAME, datetime.now(timezone.utc).isoformat(), json.dumps(receipt, sort_keys=True)),
            )
            conn.commit()
        if samples:
            result.spot_checks = _spot_checks_from_store(conn, samples)
        return result
    finally:
        conn.close()


def _result_payload(db_path: Path, *, apply: bool, result: ArchiveCollapseResult) -> dict[str, Any]:
    return {
        "db": str(db_path),
        "mode": "apply" if apply else "dry-run",
        "scanned": result.scanned,
        "updated": result.updated,
        "would_update": result.would_update,
        "batches": result.batches,
        "checkpoints": result.checkpoints,
        "post_twin_count": result.post_twin_count,
        "aux_counts_before": result.aux_counts_before,
        "aux_counts_after": result.aux_counts_after,
        "backfill_timestamp": result.backfill_timestamp,
        "spot_checks_ok": (
            "n/a-dry-run"
            if not apply
            else (all(item["ok"] for item in result.spot_checks) if result.spot_checks else None)
        ),
        "spot_checks": result.spot_checks,
        "next": (
            "verify post_twin_count is 0 and review live-window plan"
            if apply
            else "rerun with --apply against a rehearsal copy (never the live DB)"
        ),
    }


def _detect_git_sha() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha if _SHA_RE.fullmatch(sha) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse archive twins onto chunks.archived_at. Default is dry-run. "
            "Requires --db; refuses the live canonical DB unless --allow-live is set."
        )
    )
    parser.add_argument("--db", type=Path, required=True, help="Rehearsal copy DB path (required)")
    parser.add_argument("--apply", action="store_true", help="Write collapsed archive columns")
    parser.add_argument("--git-sha", dest="git_sha", help="40-char commit SHA recorded in schema_migrations")
    parser.add_argument("--batch-size", type=int, default=5000, help="Rows to scan per transaction batch")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=3,
        help="Run WAL checkpoint(FULL) every N applied batches (AGENTS.md bulk-ops law)",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Permit the canonical live DB. Lead-scheduled live window only.",
    )
    parser.add_argument(
        "--spot-check",
        type=int,
        default=0,
        help="Reservoir-sample N collapsed rows and re-read stored values",
    )
    parser.add_argument("--actor", default="repair-c", help="Actor recorded in schema_migrations details")
    args = parser.parse_args(argv)

    git_sha = args.git_sha or _detect_git_sha()
    if not git_sha:
        parser.error("--git-sha is required (40-char hex) when HEAD is not a full SHA")

    result = collapse_archive_representations(
        args.db.expanduser(),
        git_sha=git_sha,
        apply=args.apply,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        allow_live=args.allow_live,
        spot_check=args.spot_check,
        actor=args.actor,
    )
    print(json.dumps(_result_payload(args.db.expanduser(), apply=args.apply, result=result), sort_keys=True))
    if args.apply and result.post_twin_count:
        return 1
    if args.apply and result.spot_checks and not all(item["ok"] for item in result.spot_checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
