#!/usr/bin/env python3
"""KG Cleanup Phase-1 — APPLY (Etan GO 2026-06-05, via orcClaude).

Applies the APPROVED dry-run scope:
  - prune: re-derives selection with the shared brainlayer.kg_cleanup logic,
    aborts if it deviates >1% from the approved count
  - merge: applies the exact clusters from the approved scope JSON
    (tier-1 + tier-2), skipping members whose state changed since dry-run

Usage:
    python3 scripts/kg_cleanup_apply.py --scope eval_results/kg-phase1-dryrun-2026-06-05.json \
        --run-id kg-phase1-2026-06-05 [--rollback]
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brainlayer.kg_cleanup import (  # noqa: E402
    DeviationError,
    apply_merges,
    apply_prune,
    ensure_log_table,
    rollback,
    select_prune,
)

DB = Path.home() / ".local/share/brainlayer/brainlayer.db"


def checkpoint(con):
    mode, logf, ckpt = con.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
    print(f"WAL checkpoint: busy={mode} log_frames={logf} checkpointed={ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rollback", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(DB, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=60000")
    ensure_log_table(con)

    if args.rollback:
        counts = rollback(con, args.run_id)
        print(f"ROLLED BACK run {args.run_id}: {counts}")
        return

    scope = json.loads(Path(args.scope).read_text())
    approved_prune = scope["auto_prune"]["total"]
    clusters = (
        scope["auto_merge"]["tier1_diagnosis_named"]["all_clusters"]
        + scope["auto_merge"]["tier2_evidence_not_named"]["all_clusters"]
    )

    checkpoint(con)

    # ---- prune (1% deviation guard inside apply_prune) ----
    selection = select_prune(con)
    try:
        n = apply_prune(con, selection, run_id=args.run_id, expected=approved_prune)
    except DeviationError as e:
        print(f"🛑 STOP: {e}")
        sys.exit(2)
    print(f"PRUNED (archived): {n} (approved: {approved_prune})")

    # ---- merges: verify members still active, then apply ----
    live_clusters, skipped_members = [], []
    for c in clusters:
        ok_members = []
        for m in c["members"]:
            row = con.execute("SELECT status FROM kg_entities WHERE id=?", (m["id"],)).fetchone()
            if row and row["status"] == "active":
                ok_members.append(m)
            else:
                skipped_members.append((c["stem"], m["name"], m["id"]))
        if ok_members:
            live_clusters.append({**c, "members": ok_members})
    stats = apply_merges(con, live_clusters, run_id=args.run_id)
    print(f"MERGED: {stats}  skipped_changed_members={len(skipped_members)}")
    for s in skipped_members:
        print(f"  skipped: {s}")

    checkpoint(con)

    # ---- verify ----
    active = con.execute("SELECT COUNT(*) FROM kg_entities WHERE status='active'").fetchone()[0]
    archived = con.execute("SELECT COUNT(*) FROM kg_entities WHERE status='archived'").fetchone()[0]
    log_counts = dict(
        con.execute(
            "SELECT action, COUNT(*) FROM kg_cleanup_log WHERE run_id=? GROUP BY action", (args.run_id,)
        ).fetchall()
    )
    protected_hit = con.execute(
        "SELECT COUNT(*) FROM kg_cleanup_log l JOIN kg_entities e"
        " ON e.id=l.entity_id WHERE l.run_id=?"
        " AND (e.user_verified=1 OR e.importance>=0.7)",
        (args.run_id,),
    ).fetchone()[0]
    print(
        json.dumps(
            {
                "active_after": active,
                "archived_after": archived,
                "log_counts": log_counts,
                "protected_rows_touched": protected_hit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
