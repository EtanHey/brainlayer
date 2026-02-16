#!/usr/bin/env python3
"""Consolidate fragmented project names in Zikaron DB.

Merges variants (sub-packages, worktrees, old paths, pre-monorepo names)
into canonical project names. Generates rollback SQL before applying changes.

Usage:
    python3 scripts/consolidate_projects.py                    # dry-run (default)
    python3 scripts/consolidate_projects.py --generate-rollback # save rollback.sql
    python3 scripts/consolidate_projects.py --execute           # apply changes
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime

import apsw

DB_PATH = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"

# Tables with a 'project' column
PROJECT_TABLES = ["chunks", "session_context", "file_interactions", "topic_chains"]

# Static merge mappings: old_name → canonical_name
# Derived from DB audit of 34 project names (2026-02-16)
MERGE_MAP = {
    # golems ← 14 variants (sub-packages, pre-monorepo standalone repos)
    "-Users-etanheyman-Gits-golems": "golems",
    "-Users-etanheyman-Gits-golems-packages-autonomous": "golems",
    "-Users-etanheyman-Gits-golems-packages-content": "golems",
    "-Users-etanheyman-Gits-golems-packages-zikaron": "golems",
    "-Users-etanheyman-Gits-golems-packages-coach": "golems",
    "-Users-etanheyman-Gits-golems-packages-ralph": "golems",
    "claude-golem": "golems",
    "ralph": "golems",
    "zikaron": "golems",
    "-Users-etanheyman-Gits-recruiterGolem": "golems",
    "-Users-etanheyman-Gits-contentGolem": "golems",
    "-Users-etanheyman-Gits-tellerGolem": "golems",
    "-Users-etanheyman-Gits-monitorGolem": "golems",

    # domica ← 4 variants
    "-Users-etanheyman-Gits-domica": "domica",
    "domica-apps-public": "domica",
    "domica-worktrees-fix-blog": "domica",

    # songscript ← worktrees + nightshift variants
    "-Users-etanheyman-Gits-songscript": "songscript",
    "songscript-haiku": "songscript",
    "songscript-nightshift-1769910280178": "songscript",
    "-Users-etanheyman-Gits-songscript-nightshift-1770775282043": "songscript",
    "-Users-etanheyman-Gits-songscript-nightshift-1771120902191": "songscript",

    # etanheyman-com
    "-Users-etanheyman-Gits-etanheyman-com": "etanheyman-com",

    # EtanHey (github profile repo)
    "-Users-etanheyman-Gits-EtanHey": "EtanHey",

    # rudy-monorepo ← sub-package
    "rudy-monorepo-apps-jem": "rudy-monorepo",

    # taba (standalone)
    "-Users-etanheyman-Gits-taba": "taba",

    # Junk/ambiguous paths → subagents or unknown
    "-": "unknown",
    "-Users-etanheyman": "unknown",
    "-Users-etanheyman-Desktop": "unknown",
    "-Users-etanheyman-Desktop-Gits": "unknown",
    "-Users-etanheyman-Gits": "unknown",
}

# Regex patterns for dynamic matching (worktrees, nightshift)
WORKTREE_PATTERNS = [
    (re.compile(r"^(.+)-nightshift-\d+$"), None),  # strip nightshift suffix → base
    (re.compile(r"^(.+)-worktrees-.+$"), None),     # strip worktree suffix → base
]


def get_canonical(project: str) -> str:
    """Resolve a project name to its canonical form."""
    if project is None:
        return "unknown"

    # Direct mapping
    if project in MERGE_MAP:
        return MERGE_MAP[project]

    # Regex patterns
    for pattern, _ in WORKTREE_PATTERNS:
        m = pattern.match(project)
        if m:
            base = m.group(1)
            # Recurse in case the base also needs mapping
            return get_canonical(base)

    # Already canonical
    return project


def main():
    parser = argparse.ArgumentParser(description="Consolidate fragmented project names")
    parser.add_argument("--execute", action="store_true", help="Apply changes (default: dry-run)")
    parser.add_argument("--generate-rollback", action="store_true", help="Generate rollback.sql")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Database path")
    args = parser.parse_args()

    db = apsw.Connection(args.db)
    db.cursor().execute("PRAGMA busy_timeout = 5000")

    print("=== Zikaron Project Consolidation ===\n")

    # Gather all unique project names and counts per table
    total_updates = 0
    rollback_lines = []

    for table in PROJECT_TABLES:
        try:
            rows = list(db.cursor().execute(
                f"SELECT project, COUNT(*) FROM {table} GROUP BY project ORDER BY COUNT(*) DESC"
            ))
        except apsw.SQLError:
            print(f"  [{table}] — table not found or no project column, skipping")
            continue

        updates_for_table = 0
        print(f"[{table}] ({sum(r[1] for r in rows)} total rows)")

        for old_name, count in rows:
            canonical = get_canonical(old_name)
            if canonical != old_name:
                print(f"  {old_name} ({count}) → {canonical}")
                updates_for_table += count

                if args.generate_rollback:
                    # Export chunk IDs per old name for exact rollback
                    if old_name is None:
                        id_query = f"SELECT id FROM {table} WHERE project IS NULL"
                        set_clause = "SET project = NULL"
                    else:
                        safe_old = old_name.replace("'", "''")
                        id_query = f"SELECT id FROM {table} WHERE project = '{safe_old}'"
                        set_clause = f"SET project = '{safe_old}'"
                    try:
                        ids = [r[0] for r in db.cursor().execute(id_query)]
                        if ids:
                            # Batch into groups of 500 for SQL IN clause limits
                            for batch_start in range(0, len(ids), 500):
                                batch = ids[batch_start:batch_start + 500]
                                id_list = ", ".join(f"'{i}'" for i in batch)
                                rollback_lines.append(
                                    f"UPDATE {table} {set_clause} "
                                    f"WHERE id IN ({id_list});"
                                )
                    except apsw.SQLError:
                        # Table might not have 'id' column
                        rollback_lines.append(
                            f"-- WARN: Cannot generate exact rollback for {table} "
                            f"(no id column). Use DB backup instead."
                        )

        if updates_for_table == 0:
            print("  (no changes needed)")
        else:
            print(f"  → {updates_for_table} rows to update")
        total_updates += updates_for_table
        print()

    print(f"Total: {total_updates} rows across {len(PROJECT_TABLES)} tables\n")

    # Generate rollback SQL
    if args.generate_rollback:
        rollback_path = Path(__file__).parent / "rollback_consolidation.sql"
        rollback_path.write_text(
            f"-- Rollback for project consolidation ({datetime.now().isoformat()})\n"
            f"-- Uses per-chunk-ID mappings for exact reversal.\n"
            f"-- Alternative: restore from backup: cp zikaron.db.backup-YYYYMMDD zikaron.db\n"
            f"-- Run: sqlite3 ~/.local/share/zikaron/zikaron.db < rollback_consolidation.sql\n\n"
            + "\n".join(rollback_lines) + "\n"
        )
        print(f"Rollback SQL saved to: {rollback_path}")
        print()

    if not args.execute:
        print("DRY RUN — no changes made. Use --execute to apply.")
        return

    # Execute updates in a single transaction
    print("Applying updates...")
    cursor = db.cursor()
    cursor.execute("BEGIN IMMEDIATE")

    try:
        for table in PROJECT_TABLES:
            try:
                rows = list(cursor.execute(
                    f"SELECT DISTINCT project FROM {table}"
                ))
            except apsw.SQLError:
                continue

            table_updates = 0
            for (old_name,) in rows:
                canonical = get_canonical(old_name)
                if canonical != old_name:
                    if old_name is None:
                        cursor.execute(
                            f"UPDATE {table} SET project = ? WHERE project IS NULL",
                            (canonical,)
                        )
                    else:
                        cursor.execute(
                            f"UPDATE {table} SET project = ? WHERE project = ?",
                            (canonical, old_name)
                        )
                    affected = db.changes()
                    table_updates += affected

            print(f"  [{table}] {table_updates} rows updated")

        cursor.execute("COMMIT")
    except Exception:
        cursor.execute("ROLLBACK")
        print("\n  ✗ ERROR — all changes rolled back")
        raise

    # Integrity check
    print("\nRunning integrity check...")
    result = list(cursor.execute("PRAGMA integrity_check"))
    if result[0][0] == "ok":
        print("  ✓ Database integrity OK")
    else:
        print(f"  ✗ INTEGRITY ISSUE: {result}")
        sys.exit(1)

    # Show final project counts
    print("\nFinal project distribution:")
    for (proj, count) in cursor.execute(
        "SELECT project, COUNT(*) FROM chunks GROUP BY project ORDER BY COUNT(*) DESC"
    ):
        print(f"  {count:>8}  {proj}")

    print("\nDone!")


if __name__ == "__main__":
    main()
