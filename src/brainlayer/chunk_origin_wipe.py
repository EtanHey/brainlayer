"""Wipe enrichment-model labels out of chunks.chunk_origin and re-derive ingest origin.

Rehearsal/live-window tool. Default is dry-run. Refuses the live canonical DB unless
allow_live=True after a lead-scheduled writer window.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chunk_origin import LEGACY_MODEL_CHUNK_ORIGINS, detect_chunk_origin

AUX_COUNT_TABLES = (
    "chunks_fts",
    "chunks_fts_trigram",
    "chunks_fts_operational",
    "chunk_fts_rowids",
    "chunk_vectors_rowids",
)

_LIVE_DB_RELATIVE = Path(".local") / "share" / "brainlayer" / "brainlayer.db"


@dataclass
class ChunkOriginWipeResult:
    scanned: int = 0
    updated: int = 0
    batches: int = 0
    checkpoints: int = 0
    pre_wipe_legacy: int = 0
    post_wipe_legacy: int = 0
    derived: dict[str, int] = field(default_factory=dict)
    pre_wipe_distribution: dict[str, int] = field(default_factory=dict)
    post_wipe_distribution: dict[str, int] = field(default_factory=dict)
    aux_counts_before: dict[str, int] = field(default_factory=dict)
    aux_counts_after: dict[str, int] = field(default_factory=dict)
    spot_checks: list[dict[str, Any]] = field(default_factory=list)


def live_canonical_db_path() -> Path:
    """Return the canonical live DB path without going through pytest path guards."""
    return Path.home() / _LIVE_DB_RELATIVE


def assert_not_live_db(db_path: Path, *, allow_live: bool = False) -> Path:
    """Refuse the live canonical DB unless a lead-scheduled window set allow_live."""
    resolved = db_path.expanduser().resolve()
    canonical = live_canonical_db_path().expanduser().resolve()
    if resolved == canonical and not allow_live:
        raise RuntimeError(
            f"refusing to write the live BrainLayer DB at {canonical}; "
            "pass a rehearsal copy via --db, or --allow-live after a lead-scheduled window"
        )
    return resolved


def _origin_distribution(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT COALESCE(chunk_origin, '<null>'), COUNT(*) FROM chunks GROUP BY 1").fetchall()
    return {str(origin): int(count) for origin, count in rows}


def _legacy_count(conn: sqlite3.Connection, legacy: tuple[str, ...]) -> int:
    placeholders = ",".join("?" * len(legacy))
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM chunks WHERE chunk_origin IN ({placeholders})",
            legacy,
        ).fetchone()[0]
    )


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


def wipe_legacy_model_chunk_origins(
    db_path: Path,
    *,
    apply: bool = False,
    batch_size: int = 5000,
    checkpoint_every: int = 3,
    allow_live: bool = False,
    spot_check: int = 0,
) -> ChunkOriginWipeResult:
    """Re-derive chunk_origin for rows whose origin is a legacy enrichment model name."""
    resolved = assert_not_live_db(db_path, allow_live=allow_live)
    batch_size = max(1, int(batch_size))
    checkpoint_every = max(1, int(checkpoint_every))
    spot_check = max(0, int(spot_check))
    legacy = tuple(sorted(LEGACY_MODEL_CHUNK_ORIGINS))
    placeholders = ",".join("?" * len(legacy))

    result = ChunkOriginWipeResult()
    derived: Counter[str] = Counter()
    samples: list[tuple[str, str, str]] = []
    sampled = 0
    rng = random.Random(0)

    conn = sqlite3.connect(str(resolved))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        if "chunk_origin" not in columns:
            raise RuntimeError("chunks.chunk_origin column is missing")
        if "content" not in columns:
            raise RuntimeError("chunks.content column is missing")

        result.pre_wipe_distribution = _origin_distribution(conn)
        result.pre_wipe_legacy = _legacy_count(conn, legacy)
        result.aux_counts_before = _aux_counts(conn)

        if apply:
            _checkpoint(conn)
            result.checkpoints += 1

        last_rowid = 0
        while True:
            rows = list(
                conn.execute(
                    f"""
                    SELECT rowid, id, content, chunk_origin
                    FROM chunks
                    WHERE rowid > ?
                      AND chunk_origin IN ({placeholders})
                    ORDER BY rowid
                    LIMIT ?
                    """,
                    (last_rowid, *legacy, batch_size),
                )
            )
            if not rows:
                break

            result.batches += 1
            updates: list[tuple[str, int]] = []
            for row in rows:
                last_rowid = int(row["rowid"])
                result.scanned += 1
                content = row["content"]
                new_origin = detect_chunk_origin(content)
                derived[new_origin] += 1
                updates.append((new_origin, int(row["rowid"])))
                if spot_check:
                    sampled += 1
                    sample = (str(row["id"]), str(content), new_origin)
                    if len(samples) < spot_check:
                        samples.append(sample)
                    else:
                        j = rng.randrange(sampled)
                        if j < spot_check:
                            samples[j] = sample

            if apply and updates:
                before_changes = conn.total_changes
                conn.executemany(
                    f"""
                    UPDATE chunks
                    SET chunk_origin = ?
                    WHERE rowid = ?
                      AND chunk_origin IN ({placeholders})
                    """,
                    [(origin, rowid, *legacy) for origin, rowid in updates],
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

        result.derived = dict(derived)
        result.post_wipe_distribution = _origin_distribution(conn)
        result.post_wipe_legacy = _legacy_count(conn, legacy)
        result.aux_counts_after = _aux_counts(conn)
        result.spot_checks = [
            {
                "id": chunk_id,
                "derived": derived_origin,
                "expected": detect_chunk_origin(content),
                "ok": derived_origin == detect_chunk_origin(content)
                and derived_origin not in LEGACY_MODEL_CHUNK_ORIGINS,
            }
            for chunk_id, content, derived_origin in samples
        ]
        return result
    finally:
        conn.close()


def _result_payload(db_path: Path, *, apply: bool, result: ChunkOriginWipeResult) -> dict[str, Any]:
    return {
        "db": str(db_path),
        "mode": "apply" if apply else "dry-run",
        "scanned": result.scanned,
        "updated": result.updated,
        "batches": result.batches,
        "checkpoints": result.checkpoints,
        "pre_wipe_legacy": result.pre_wipe_legacy,
        "post_wipe_legacy": result.post_wipe_legacy,
        "derived": result.derived,
        "pre_wipe_distribution": result.pre_wipe_distribution,
        "post_wipe_distribution": result.post_wipe_distribution,
        "aux_counts_before": result.aux_counts_before,
        "aux_counts_after": result.aux_counts_after,
        "spot_checks_ok": all(item["ok"] for item in result.spot_checks) if result.spot_checks else None,
        "spot_checks": result.spot_checks,
        "next": (
            "verify remaining legacy count is 0 and review live-window plan"
            if apply
            else "rerun with --apply against a rehearsal copy (never the live DB)"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wipe legacy enrichment-model labels out of chunks.chunk_origin and re-derive "
            "via detect_chunk_origin. Default is dry-run. Requires --db; refuses the live "
            "canonical DB unless --allow-live is set for a lead-scheduled window."
        )
    )
    parser.add_argument("--db", type=Path, required=True, help="Rehearsal copy DB path (required)")
    parser.add_argument("--apply", action="store_true", help="Write re-derived origins to the DB")
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
        help="Reservoir-sample N re-derived rows and verify detect_chunk_origin matches",
    )
    args = parser.parse_args(argv)

    result = wipe_legacy_model_chunk_origins(
        args.db.expanduser(),
        apply=args.apply,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        allow_live=args.allow_live,
        spot_check=args.spot_check,
    )
    print(json.dumps(_result_payload(args.db.expanduser(), apply=args.apply, result=result), sort_keys=True))
    if args.apply and result.post_wipe_legacy:
        return 1
    if result.spot_checks and not all(item["ok"] for item in result.spot_checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
