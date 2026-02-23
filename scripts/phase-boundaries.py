#!/usr/bin/env python3
"""Query phase_commits table and print summary report.

Usage:
    python3 scripts/phase-boundaries.py
    python3 scripts/phase-boundaries.py --project brainlayer
    python3 scripts/phase-boundaries.py --phase "Phase 5"
"""

import argparse
import json
from pathlib import Path

from brainlayer.paths import DEFAULT_DB_PATH
from brainlayer.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Phase boundary report")
    parser.add_argument("--project", help="Filter by project")
    parser.add_argument("--phase", help="Filter by phase name")
    args = parser.parse_args()

    store = VectorStore(DEFAULT_DB_PATH)
    cursor = store.conn.cursor()

    # Build query
    where = []
    params = []
    if args.project:
        where.append("project = ?")
        params.append(args.project)
    if args.phase:
        where.append("phase_name LIKE ?")
        params.append(f"%{args.phase}%")

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    # Summary stats
    total = list(cursor.execute(f"SELECT COUNT(*) FROM phase_commits {where_sql}", params))[0][0]
    print(f"Phase Commits: {total}")

    if total == 0:
        print("No phase commits recorded yet.")
        store.close()
        return

    # By phase
    rows = list(
        cursor.execute(
            f"""SELECT phase_name, COUNT(*),
                       GROUP_CONCAT(DISTINCT outcome),
                       AVG(confidence_score)
                FROM phase_commits {where_sql}
                GROUP BY phase_name ORDER BY MIN(created_at)""",
            params,
        )
    )
    print(f"\n{'Phase':<25} {'Commits':>8} {'Outcomes':<30} {'Avg Conf':>8}")
    print("-" * 75)
    for phase, count, outcomes, avg_conf in rows:
        phase_display = phase or "(unlinked)"
        conf_display = f"{avg_conf:.2f}" if avg_conf else "N/A"
        print(f"{phase_display:<25} {count:>8} {outcomes or 'N/A':<30} {conf_display:>8}")

    # Recent commits
    recent = list(
        cursor.execute(
            f"""SELECT commit_hash, commit_message, phase_name, project,
                       outcome, created_at
                FROM phase_commits {where_sql}
                ORDER BY created_at DESC LIMIT 10""",
            params,
        )
    )
    if recent:
        print(f"\nRecent commits:")
        for hash_, msg, phase, proj, outcome, ts in recent:
            ts_short = (ts or "")[:19]
            print(f"  {hash_[:8]} {msg[:50]:<50} {phase or '':<15} {outcome or '':<10} {ts_short}")

    store.close()


if __name__ == "__main__":
    main()
