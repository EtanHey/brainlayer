"""Normalize TEXT timestamp columns to ISO-8601 UTC (rehearsal/live-window).

Default is dry-run. Refuses the live canonical DB unless allow_live=True after a
lead-scheduled writer window. Follows the repair-b/c preimage + checkpoint pattern.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunk_origin_wipe import AUX_COUNT_TABLES, assert_not_live_db
from .chunk_origin_wipe import live_canonical_db_path as live_canonical_db_path

PREIMAGE_TABLE = "timestamp_iso_preimage"
MIGRATION_NAME = "2026_08_18_timestamp_iso_utc"
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|\+00:00)$")

ISO_TIMESTAMP_COLUMNS = (
    "created_at",
    "last_seen_at",
    "archived_at",
    "enriched_at",
    "valid_from",
    "invalid_at",
    "sys_period_start",
    "sys_period_end",
    "consolidated_at",
)


@dataclass
class TimestampIsoResult:
    scanned: int = 0
    updated: int = 0
    would_update: int = 0
    skipped_unparseable: int = 0
    batches: int = 0
    checkpoints: int = 0
    aux_counts_before: dict[str, int] = field(default_factory=dict)
    aux_counts_after: dict[str, int] = field(default_factory=dict)
    spot_checks: list[dict[str, Any]] = field(default_factory=list)
    column_counts: dict[str, int] = field(default_factory=dict)


def is_iso_utc(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and _ISO_UTC_RE.match(text) is not None


def _from_unix(number: float) -> str:
    seconds = number
    if number >= 1e18:
        seconds = number / 1e9
    elif number >= 1e12:
        seconds = number / 1e3
    stamp = datetime.fromtimestamp(seconds, timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def normalize_timestamp(value: Any) -> str | None:
    """Return ISO-8601 UTC, or None for empty inputs."""
    if value is None:
        return None
    if isinstance(value, datetime):
        stamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _from_unix(float(value))
    text = str(value).strip()
    if not text:
        return None
    if _ISO_UTC_RE.match(text):
        return text.replace("+00:00", "Z")
    iso_candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso_candidate)
    except ValueError:
        parsed = None
    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        number = float(text)
    except ValueError:
        return None
    if number <= 0:
        return None
    return _from_unix(number)


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


def _needs_normalize(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return not is_iso_utc(text)


def _ensure_preimage(conn: sqlite3.Connection, columns: list[str]) -> None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (PREIMAGE_TABLE,),
    ).fetchone()
    select_cols = ", ".join(["id", *columns])
    if exists:
        return
    conn.execute(f"CREATE TABLE {PREIMAGE_TABLE} AS SELECT {select_cols} FROM chunks WHERE 0")
    conn.commit()


def _spot_ok(row: sqlite3.Row, columns: list[str]) -> bool:
    for column in columns:
        value = row[column]
        if value in (None, ""):
            continue
        if not is_iso_utc(value):
            return False
    return True


def normalize_timestamps(
    db_path: Path,
    *,
    git_sha: str,
    apply: bool = False,
    batch_size: int = 5000,
    checkpoint_every: int = 3,
    allow_live: bool = False,
    spot_check: int = 0,
    actor: str = "repair-d",
) -> TimestampIsoResult:
    """Rewrite non-ISO TEXT timestamps to ISO-8601 UTC."""
    sha = _validate_git_sha(git_sha)
    resolved = assert_not_live_db(db_path, allow_live=allow_live)
    batch_size = max(1, int(batch_size))
    checkpoint_every = max(1, int(checkpoint_every))
    spot_check = max(0, int(spot_check))
    result = TimestampIsoResult()
    samples: list[str] = []
    sampled = 0
    rng = random.Random(0)

    conn = sqlite3.connect(str(resolved))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        present = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        columns = [column for column in ISO_TIMESTAMP_COLUMNS if column in present]
        if not columns:
            raise RuntimeError("chunks has no ISO timestamp columns to normalize")
        result.aux_counts_before = _aux_counts(conn)
        if apply:
            _ensure_preimage(conn, columns)
            _checkpoint(conn)
            result.checkpoints += 1

        last_rowid = 0
        while True:
            rows = list(
                conn.execute(
                    f"""
                    SELECT rowid, id, {", ".join(columns)}
                    FROM chunks
                    WHERE rowid > ?
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
            preimage_rows: list[tuple[Any, ...]] = []
            for row in rows:
                last_rowid = int(row["rowid"])
                result.scanned += 1
                changed = False
                new_values: list[Any] = []
                old_values: list[Any] = [row["id"]]
                for column in columns:
                    current = row[column]
                    old_values.append(current)
                    if _needs_normalize(current):
                        converted = normalize_timestamp(current)
                        if converted is not None and is_iso_utc(converted):
                            new_values.append(converted)
                            changed = True
                            result.column_counts[column] = result.column_counts.get(column, 0) + 1
                        else:
                            new_values.append(current)
                            result.skipped_unparseable += 1
                    else:
                        new_values.append(current)
                if not changed:
                    continue
                result.would_update += 1
                if spot_check:
                    sampled += 1
                    chunk_id = str(row["id"])
                    if len(samples) < spot_check:
                        samples.append(chunk_id)
                    else:
                        j = rng.randrange(sampled)
                        if j < spot_check:
                            samples[j] = chunk_id
                updates.append((*new_values, int(row["rowid"])))
                preimage_rows.append(tuple(old_values))

            if apply and updates:
                placeholders = ", ".join("?" for _ in columns)
                conn.executemany(
                    f"INSERT OR IGNORE INTO {PREIMAGE_TABLE}(id, {', '.join(columns)}) VALUES (?, {placeholders})",
                    preimage_rows,
                )
                assignments = ", ".join(f"{column} = ?" for column in columns)
                before = conn.total_changes
                conn.executemany(
                    f"UPDATE chunks SET {assignments} WHERE rowid = ?",
                    updates,
                )
                result.updated += conn.total_changes - before
                conn.commit()
                if result.batches % checkpoint_every == 0:
                    _checkpoint(conn)
                    result.checkpoints += 1
            elif apply:
                conn.commit()

        if apply and result.batches and result.batches % checkpoint_every != 0:
            _checkpoint(conn)
            result.checkpoints += 1

        result.aux_counts_after = _aux_counts(conn)
        if apply:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    name TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL,
                    details TEXT
                )
                """
            )
            receipt = {
                "migration": MIGRATION_NAME,
                "git_sha": sha,
                "actor": actor,
                "updated": result.updated,
                "skipped_unparseable": result.skipped_unparseable,
                "column_counts": result.column_counts,
            }
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations(name, applied_at, details) VALUES (?, ?, ?)",
                (MIGRATION_NAME, datetime.now(timezone.utc).isoformat(), json.dumps(receipt, sort_keys=True)),
            )
            conn.commit()
        if samples:
            checks: list[dict[str, Any]] = []
            for chunk_id in samples:
                stored = conn.execute(
                    f"SELECT id, {', '.join(columns)} FROM chunks WHERE id = ?",
                    (chunk_id,),
                ).fetchone()
                if stored is None:
                    checks.append({"id": chunk_id, "ok": False})
                    continue
                checks.append(
                    {
                        "id": chunk_id,
                        "ok": _spot_ok(stored, columns),
                        **{column: stored[column] for column in columns},
                    }
                )
            result.spot_checks = checks
        return result
    finally:
        conn.close()


def _result_payload(db_path: Path, *, apply: bool, result: TimestampIsoResult) -> dict[str, Any]:
    return {
        "db": str(db_path),
        "mode": "apply" if apply else "dry-run",
        "scanned": result.scanned,
        "updated": result.updated,
        "would_update": result.would_update,
        "skipped_unparseable": result.skipped_unparseable,
        "batches": result.batches,
        "checkpoints": result.checkpoints,
        "column_counts": result.column_counts,
        "aux_counts_before": result.aux_counts_before,
        "aux_counts_after": result.aux_counts_after,
        "spot_checks_ok": (
            "n/a-dry-run"
            if not apply
            else (all(item["ok"] for item in result.spot_checks) if result.spot_checks else None)
        ),
        "spot_checks": result.spot_checks,
        "next": (
            "verify remaining non-ISO counts are 0 and review live-window plan"
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
            "Normalize TEXT timestamp columns to ISO-8601 UTC. Default is dry-run. "
            "Requires --db; refuses the live canonical DB unless --allow-live is set."
        )
    )
    parser.add_argument("--db", type=Path, required=True, help="Rehearsal copy DB path (required)")
    parser.add_argument("--apply", action="store_true", help="Write normalized timestamps")
    parser.add_argument("--git-sha", dest="git_sha", help="40-char commit SHA recorded in schema_migrations")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=3)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--spot-check", type=int, default=0)
    parser.add_argument("--actor", default="repair-d")
    args = parser.parse_args(argv)
    git_sha = args.git_sha or _detect_git_sha()
    if not git_sha:
        parser.error("--git-sha is required (40-char hex) when HEAD is not a full SHA")
    result = normalize_timestamps(
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
    if args.apply and result.spot_checks and not all(item["ok"] for item in result.spot_checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
