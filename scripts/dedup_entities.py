#!/usr/bin/env python3
"""One-time dedup of kg_entities in BrainLayer's SQLite database.

Finds duplicate entities (same lower(name) + entity_type), picks a survivor
(most relations + evidence chunks), merges references, and deletes orphans.

Uses the existing merge_entities() function from entity_resolution.py for
safe, tested merging logic.

Usage:
    python3 scripts/dedup_entities.py                    # uses DEFAULT_DB_PATH
    python3 scripts/dedup_entities.py --db /path/to.db   # explicit DB
    python3 scripts/dedup_entities.py --dry-run           # preview only
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path so we can import brainlayer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.paths import DEFAULT_DB_PATH
from brainlayer.pipeline.entity_resolution import merge_entities
from brainlayer.vector_store import VectorStore


def find_duplicates(store: VectorStore) -> list[dict]:
    """Find duplicate entity groups: same lower(name) + entity_type, count > 1."""
    cursor = store._read_cursor()
    groups = list(
        cursor.execute(
            """
            SELECT LOWER(name) AS lname, entity_type, COUNT(*) AS cnt
            FROM kg_entities
            GROUP BY LOWER(name), entity_type
            HAVING cnt > 1
            ORDER BY cnt DESC
            """
        )
    )

    results = []
    for lname, etype, cnt in groups:
        # Fetch all entities in this group
        entities = list(
            cursor.execute(
                """
                SELECT e.id, e.name, e.created_at,
                    (SELECT COUNT(*) FROM kg_relations
                     WHERE source_id = e.id OR target_id = e.id) AS rel_count,
                    (SELECT COUNT(*) FROM kg_entity_chunks
                     WHERE entity_id = e.id) AS chunk_count
                FROM kg_entities e
                WHERE LOWER(e.name) = ? AND e.entity_type = ?
                ORDER BY rel_count DESC, chunk_count DESC
                """,
                (lname, etype),
            )
        )
        results.append(
            {
                "name": lname,
                "entity_type": etype,
                "count": cnt,
                "entities": [
                    {
                        "id": row[0],
                        "name": row[1],
                        "created_at": row[2],
                        "relations": row[3],
                        "chunks": row[4],
                    }
                    for row in entities
                ],
            }
        )
    return results


def pick_survivor(group: dict) -> tuple[dict, list[dict]]:
    """Pick the entity with the most relations + chunks as survivor."""
    entities = group["entities"]
    # Already sorted by (rel_count DESC, chunk_count DESC) from SQL
    survivor = entities[0]
    orphans = entities[1:]
    return survivor, orphans


def get_stats(store: VectorStore) -> dict:
    """Get entity/relation/link counts."""
    cursor = store._read_cursor()
    return {
        "entities": list(cursor.execute("SELECT COUNT(*) FROM kg_entities"))[0][0],
        "relations": list(cursor.execute("SELECT COUNT(*) FROM kg_relations"))[0][0],
        "entity_chunks": list(cursor.execute("SELECT COUNT(*) FROM kg_entity_chunks"))[0][0],
        "aliases": list(cursor.execute("SELECT COUNT(*) FROM kg_entity_aliases"))[0][0],
    }


def run_dedup(db_path: Path, dry_run: bool = False) -> dict:
    """Run the dedup process. Returns a report dict."""
    print(f"Database: {db_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}\n")

    store = VectorStore(db_path)
    before = get_stats(store)
    print(f"Before: {json.dumps(before, indent=2)}")

    duplicates = find_duplicates(store)
    if not duplicates:
        print("\nNo duplicates found. Database is clean.")
        store.close()
        return {
            "db_path": str(db_path),
            "before": before,
            "after": before,
            "duplicates_found": 0,
            "entities_removed": 0,
            "merges": [],
        }

    print(f"\nFound {len(duplicates)} duplicate group(s):\n")

    merges = []
    for group in duplicates:
        survivor, orphans = pick_survivor(group)
        print(f"  {group['entity_type']}/{group['name']} ({group['count']} entities)")
        print(
            f"    KEEP: {survivor['id']} "
            f"(name={survivor['name']!r}, rels={survivor['relations']}, chunks={survivor['chunks']})"
        )
        for orphan in orphans:
            print(
                f"    DROP: {orphan['id']} "
                f"(name={orphan['name']!r}, rels={orphan['relations']}, chunks={orphan['chunks']})"
            )
            merges.append(
                {
                    "survivor_id": survivor["id"],
                    "survivor_name": survivor["name"],
                    "orphan_id": orphan["id"],
                    "orphan_name": orphan["name"],
                    "entity_type": group["entity_type"],
                    "orphan_relations": orphan["relations"],
                    "orphan_chunks": orphan["chunks"],
                }
            )

    if dry_run:
        print("\nDry run — no changes made.")
        store.close()
        return {
            "db_path": str(db_path),
            "before": before,
            "after": before,
            "duplicates_found": len(duplicates),
            "entities_removed": 0,
            "merges": merges,
            "dry_run": True,
        }

    # Execute merges inside the VectorStore's existing connection (auto-commit)
    print("\nMerging...")
    for merge in merges:
        print(f"  Merging {merge['orphan_id']} -> {merge['survivor_id']}...")
        merge_entities(store, keep_id=merge["survivor_id"], merge_id=merge["orphan_id"])
    print("Done.")

    after = get_stats(store)
    print(f"\nAfter: {json.dumps(after, indent=2)}")

    entities_removed = before["entities"] - after["entities"]
    print(f"\nRemoved {entities_removed} duplicate entities.")

    store.close()
    return {
        "db_path": str(db_path),
        "before": before,
        "after": after,
        "duplicates_found": len(duplicates),
        "entities_removed": entities_removed,
        "merges": merges,
    }


def main():
    parser = argparse.ArgumentParser(description="Deduplicate kg_entities in BrainLayer DB")
    parser.add_argument("--db", type=Path, default=None, help="Path to .db file (default: DEFAULT_DB_PATH)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup step")
    args = parser.parse_args()

    db_path = args.db or DEFAULT_DB_PATH
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Backup
    if not args.dry_run and not args.no_backup:
        backup_path = db_path.with_suffix(".db.pre-dedup")
        print(f"Backing up: {db_path} -> {backup_path}")
        shutil.copy2(db_path, backup_path)
        # Also copy WAL/SHM if they exist
        for suffix in ["-wal", "-shm"]:
            wal = Path(str(db_path) + suffix)
            if wal.exists():
                shutil.copy2(wal, Path(str(backup_path) + suffix))
        print("Backup complete.\n")

    report = run_dedup(db_path, dry_run=args.dry_run)

    # Write report to stdout as JSON for scripting
    print(f"\n{'='*60}")
    print("REPORT:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
