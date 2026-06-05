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
    apply_keep_separate,
    apply_merges,
    apply_prune,
    ensure_log_table,
    rollback,
    select_prune,
)

DB = Path.home() / ".local/share/brainlayer/brainlayer.db"
DECISIONS_SCHEMA = "kg-flag-decisions-v1"
DECISION_APPLY_COUNT_KEYS = ("merge_clusters", "rows_merged_away", "keep")


def checkpoint(con):
    mode, logf, ckpt = con.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
    print(f"WAL checkpoint: busy={mode} log_frames={logf} checkpointed={ckpt}")


def load_decisions(path: str | Path) -> dict:
    decisions = json.loads(Path(path).read_text(encoding="utf-8"))
    if decisions.get("schema") != DECISIONS_SCHEMA:
        raise ValueError(f"decisions schema must be {DECISIONS_SCHEMA!r}, got {decisions.get('schema')!r}")
    if not isinstance(decisions.get("counts"), dict):
        raise ValueError("decisions JSON must include object field 'counts'")
    if not isinstance(decisions.get("merge", []), list):
        raise ValueError("decisions JSON field 'merge' must be a list")
    if not isinstance(decisions.get("keep", []), list):
        raise ValueError("decisions JSON field 'keep' must be a list")
    return decisions


def _decision_counts(decisions: dict) -> dict:
    counts = decisions["counts"]
    return {
        "merge_clusters": int(counts.get("merge_clusters", 0)),
        "rows_merged_away": int(counts.get("rows_merged_away", 0)),
        "keep": int(counts.get("keep", 0)),
        "explicit": int(counts.get("explicit", 0)),
        "by_rule": int(counts.get("by_rule", 0)),
    }


def _within_one_percent(actual: int, expected: int) -> bool:
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / expected <= 0.01


def _guard_decision_deviation(actual: dict, expected: dict) -> None:
    for key in DECISION_APPLY_COUNT_KEYS:
        actual_value = int(actual.get(key, 0))
        expected_value = int(expected.get(key, 0))
        if not _within_one_percent(actual_value, expected_value):
            raise DeviationError(f"{key} actual {actual_value} deviates >1% from decisions JSON {expected_value}")


def _prepare_decision_merges(
    con: sqlite3.Connection, decisions: dict
) -> tuple[list[dict], list[dict], list[dict], dict]:
    live_clusters = []
    skipped_members = []
    protected_losers = []
    for cluster in decisions.get("merge", []):
        ok_members = []
        for member in cluster.get("members", []):
            row = con.execute(
                "SELECT status, user_verified, importance FROM kg_entities WHERE id=?",
                (member["id"],),
            ).fetchone()
            if row and row["status"] == "active":
                ok_members.append(member)
                if row["user_verified"] == 1 or (row["importance"] or 0) >= 0.7:
                    protected_losers.append(
                        {
                            "stem": cluster.get("stem"),
                            "name": member.get("name"),
                            "id": member["id"],
                            "user_verified": row["user_verified"],
                            "importance": row["importance"],
                        }
                    )
            else:
                skipped_members.append(
                    {
                        "stem": cluster.get("stem"),
                        "name": member.get("name"),
                        "id": member["id"],
                        "status": row["status"] if row else "missing",
                    }
                )
        if ok_members:
            live_clusters.append({**cluster, "members": ok_members})
    actual = {
        "merge_clusters": len(live_clusters),
        "rows_merged_away": sum(len(cluster["members"]) for cluster in live_clusters),
        "keep": len(decisions.get("keep", [])),
    }
    return live_clusters, skipped_members, protected_losers, actual


def run_decisions(
    con: sqlite3.Connection | None,
    decisions_path: str | Path,
    run_id: str,
    execute: bool = False,
) -> dict:
    decisions = load_decisions(decisions_path)
    counts = _decision_counts(decisions)
    summary = {
        "mode": "decisions",
        "schema": decisions["schema"],
        "source": decisions.get("source"),
        "dry_run": not execute,
        "counts": counts,
    }
    if not execute:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    if con is None:
        raise ValueError("run_decisions execute=True requires a sqlite connection")

    live_clusters, skipped_members, protected_losers, actual = _prepare_decision_merges(con, decisions)
    _guard_decision_deviation(actual, counts)

    ensure_log_table(con)
    merge_stats = apply_merges(con, live_clusters, run_id=run_id)
    keep_logged = apply_keep_separate(con, decisions.get("keep", []), run_id=run_id)
    report = {
        **summary,
        "dry_run": False,
        "actual": actual,
        "merge_stats": merge_stats,
        "keep_logged": keep_logged,
        "skipped_changed_members": skipped_members,
        "protected_merge_losers": protected_losers,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def run_scope(con: sqlite3.Connection, scope_path: str | Path, run_id: str) -> None:
    ensure_log_table(con)
    scope = json.loads(Path(scope_path).read_text(encoding="utf-8"))
    approved_prune = scope["auto_prune"]["total"]
    clusters = (
        scope["auto_merge"]["tier1_diagnosis_named"]["all_clusters"]
        + scope["auto_merge"]["tier2_evidence_not_named"]["all_clusters"]
    )

    checkpoint(con)

    # ---- prune (1% deviation guard inside apply_prune) ----
    selection = select_prune(con)
    try:
        n = apply_prune(con, selection, run_id=run_id, expected=approved_prune)
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
    stats = apply_merges(con, live_clusters, run_id=run_id)
    print(f"MERGED: {stats}  skipped_changed_members={len(skipped_members)}")
    for s in skipped_members:
        print(f"  skipped: {s}")

    checkpoint(con)

    # ---- verify ----
    active = con.execute("SELECT COUNT(*) FROM kg_entities WHERE status='active'").fetchone()[0]
    archived = con.execute("SELECT COUNT(*) FROM kg_entities WHERE status='archived'").fetchone()[0]
    log_counts = dict(
        con.execute("SELECT action, COUNT(*) FROM kg_cleanup_log WHERE run_id=? GROUP BY action", (run_id,)).fetchall()
    )
    protected_hit = con.execute(
        "SELECT COUNT(*) FROM kg_cleanup_log l JOIN kg_entities e"
        " ON e.id=l.entity_id WHERE l.run_id=?"
        " AND (e.user_verified=1 OR e.importance>=0.7)",
        (run_id,),
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


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--scope")
    group.add_argument("--decisions")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    args = ap.parse_args()

    if not args.rollback and not args.scope and not args.decisions:
        ap.error("one of --scope or --decisions is required unless --rollback is set")

    if args.decisions and not args.execute and not args.rollback:
        try:
            run_decisions(None, args.decisions, run_id=args.run_id, execute=False)
        except ValueError as e:
            print(f"🛑 STOP: {e}", file=sys.stderr)
            sys.exit(2)
        return

    con = sqlite3.connect(DB, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=60000")

    if args.rollback:
        ensure_log_table(con)
        counts = rollback(con, args.run_id)
        print(f"ROLLED BACK run {args.run_id}: {counts}")
        return

    if args.decisions:
        try:
            run_decisions(con, args.decisions, run_id=args.run_id, execute=args.execute)
        except (DeviationError, ValueError) as e:
            print(f"🛑 STOP: {e}", file=sys.stderr)
            sys.exit(2)
        return

    run_scope(con, args.scope, run_id=args.run_id)


if __name__ == "__main__":
    main()
