#!/usr/bin/env python3
"""KG data cleanup — fix known misclassifications and bad relations.

Fixes:
1. Entity type corrections (e.g., "Yichus" person → project)
2. Self-referential relation removal
3. Ad-hoc relation type normalization
4. Entity importance recomputation

Usage:
    # Dry-run (default — shows what would change):
    python3 scripts/kg_cleanup.py

    # Apply changes:
    python3 scripts/kg_cleanup.py --apply

    # Stats only:
    python3 scripts/kg_cleanup.py --stats
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.paths import get_db_path
from brainlayer.pipeline.kg_extraction import CANONICAL_RELATION_TYPES, compute_entity_importance
from brainlayer.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Entity type corrections ──
# (entity_type, name) → new_entity_type
ENTITY_TYPE_FIXES: dict[tuple[str, str], str] = {
    ("person", "Yichus"): "project",
    ("person", "yichus"): "project",
    ("company", "Figma"): "technology",
    ("company", "Linear"): "technology",
    ("company", "Supabase"): "technology",
    ("company", "GitHub"): "technology",
    ("company", "GitLab"): "technology",
    # Agents misclassified as person/golem
    ("person", "Ralph"): "agent",
    ("golem", "Ralph"): "agent",
    ("person", "ClaudeGolem"): "agent",
    # golem → agent retype
    ("golem", "brainClaude"): "agent",
    ("golem", "coachClaude"): "agent",
    ("golem", "voiceClaude"): "agent",
    ("golem", "contentClaude"): "agent",
    ("golem", "golemsClaude"): "agent",
    ("golem", "recruiterGolem"): "agent",
    ("golem", "contentGolem"): "agent",
    ("golem", "tellerGolem"): "agent",
    ("golem", "ClaudeGolem"): "agent",
    ("golem", "ContentClaude"): "agent",
    ("golem", "orcClaude"): "agent",
    ("golem", "dashboardClaude"): "agent",
}

# ── Relations to delete ──
# (source_name, relation_type, target_name)
RELATIONS_TO_DELETE: list[tuple[str, str, str]] = [
    # Self-referential
    ("brainClaude", "maintains", "brainClaude"),
    # Likely misextractions
    ("Grammy", "framework_for", "golems"),
]


def print_stats(store: VectorStore):
    """Print current KG statistics."""
    cursor = store._read_cursor()

    # Entity counts by type
    rows = list(
        cursor.execute("SELECT entity_type, COUNT(*) FROM kg_entities GROUP BY entity_type ORDER BY COUNT(*) DESC")
    )
    total = sum(r[1] for r in rows)
    print(f"\n{'=' * 50}")
    print(f"KG Entities: {total}")
    print(f"{'=' * 50}")
    for etype, count in rows:
        print(f"  {etype:15s} {count:4d}")

    # Relation counts by type
    rows = list(
        cursor.execute(
            "SELECT relation_type, COUNT(*) FROM kg_relations WHERE expired_at IS NULL GROUP BY relation_type ORDER BY COUNT(*) DESC"
        )
    )
    total_rels = sum(r[1] for r in rows)
    print(f"\n{'=' * 50}")
    print(f"KG Relations: {total_rels} ({len(rows)} types)")
    print(f"{'=' * 50}")
    for rtype, count in rows:
        canonical = "✓" if rtype in CANONICAL_RELATION_TYPES else "✗"
        print(f"  {canonical} {rtype:25s} {count:4d}")

    # Self-referential check
    self_refs = list(
        cursor.execute(
            """SELECT e1.name, r.relation_type, e2.name
               FROM kg_relations r
               JOIN kg_entities e1 ON r.source_id = e1.id
               JOIN kg_entities e2 ON r.target_id = e2.id
               WHERE r.source_id = r.target_id AND r.expired_at IS NULL"""
        )
    )
    if self_refs:
        print(f"\n⚠ Self-referential relations: {len(self_refs)}")
        for src, rtype, tgt in self_refs:
            print(f"  {src} --{rtype}--> {tgt}")

    # Importance distribution
    rows = list(
        cursor.execute(
            "SELECT MIN(importance), MAX(importance), AVG(importance), COUNT(*) FROM kg_entities WHERE importance IS NOT NULL"
        )
    )
    if rows and rows[0][0] is not None:
        mn, mx, avg, cnt = rows[0]
        print(f"\nImportance: min={mn:.2f} max={mx:.2f} avg={avg:.2f} ({cnt} entities)")

    # Fact coverage
    with_fact = list(
        cursor.execute("SELECT COUNT(*) FROM kg_relations WHERE fact IS NOT NULL AND fact != '' AND expired_at IS NULL")
    )[0][0]
    print(f"Relations with facts: {with_fact}/{total_rels}")


def fix_entity_types(store: VectorStore, dry_run: bool = True) -> int:
    """Fix known entity type misclassifications."""
    cursor = store._read_cursor()
    fixed = 0

    for (old_type, name), new_type in ENTITY_TYPE_FIXES.items():
        rows = list(cursor.execute("SELECT id FROM kg_entities WHERE entity_type = ? AND name = ?", (old_type, name)))
        if not rows:
            continue

        entity_id = rows[0][0]
        if dry_run:
            logger.info("[DRY-RUN] Would retype %s (%s → %s)", name, old_type, new_type)
        else:
            write_cursor = store.conn.cursor()
            write_cursor.execute(
                "UPDATE kg_entities SET entity_type = ?, updated_at = datetime('now') WHERE id = ?",
                (new_type, entity_id),
            )
            logger.info("Retyped %s (%s → %s)", name, old_type, new_type)
        fixed += 1

    return fixed


def fix_relations(store: VectorStore, dry_run: bool = True) -> dict[str, int]:
    """Fix bad relations: delete known bad, normalize types, remove self-refs."""
    cursor = store._read_cursor()
    stats = {"deleted_known": 0, "deleted_self_ref": 0, "normalized": 0}

    # 1. Delete known bad relations
    for src_name, rtype, tgt_name in RELATIONS_TO_DELETE:
        rows = list(
            cursor.execute(
                """SELECT r.id FROM kg_relations r
                   JOIN kg_entities e1 ON r.source_id = e1.id
                   JOIN kg_entities e2 ON r.target_id = e2.id
                   WHERE e1.name = ? AND r.relation_type = ? AND e2.name = ?
                   AND r.expired_at IS NULL""",
                (src_name, rtype, tgt_name),
            )
        )
        for (rel_id,) in rows:
            if dry_run:
                logger.info("[DRY-RUN] Would delete relation: %s --%s--> %s", src_name, rtype, tgt_name)
            else:
                write_cursor = store.conn.cursor()
                write_cursor.execute(
                    "UPDATE kg_relations SET expired_at = datetime('now') WHERE id = ?",
                    (rel_id,),
                )
                logger.info("Expired relation: %s --%s--> %s", src_name, rtype, tgt_name)
            stats["deleted_known"] += 1

    # 2. Delete self-referential relations
    self_refs = list(
        cursor.execute("SELECT id, source_id FROM kg_relations WHERE source_id = target_id AND expired_at IS NULL")
    )
    for rel_id, _ in self_refs:
        if dry_run:
            logger.info("[DRY-RUN] Would expire self-referential relation %s", rel_id)
        else:
            write_cursor = store.conn.cursor()
            write_cursor.execute(
                "UPDATE kg_relations SET expired_at = datetime('now') WHERE id = ?",
                (rel_id,),
            )
            logger.info("Expired self-referential relation %s", rel_id)
        stats["deleted_self_ref"] += 1

    # 3. Normalize non-canonical relation types to related_to
    non_canonical = list(
        cursor.execute(
            "SELECT id, relation_type FROM kg_relations WHERE expired_at IS NULL AND relation_type NOT IN ({})".format(
                ",".join(f"'{t}'" for t in CANONICAL_RELATION_TYPES)
            )
        )
    )
    for rel_id, old_type in non_canonical:
        if dry_run:
            logger.info("[DRY-RUN] Would normalize relation type %s → related_to", old_type)
        else:
            write_cursor = store.conn.cursor()
            write_cursor.execute(
                "UPDATE kg_relations SET relation_type = 'related_to' WHERE id = ?",
                (rel_id,),
            )
            logger.info("Normalized relation type %s → related_to", old_type)
        stats["normalized"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="KG data cleanup")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run)")
    parser.add_argument("--stats", action="store_true", help="Print stats only")
    parser.add_argument("--importance", action="store_true", help="Recompute entity importance")
    args = parser.parse_args()

    db_path = get_db_path()
    logger.info("Using DB: %s", db_path)
    store = VectorStore(db_path)

    if args.stats:
        print_stats(store)
        store.close()
        return

    dry_run = not args.apply

    if dry_run:
        logger.info("=== DRY-RUN MODE (use --apply to commit changes) ===")

    # Fix entity types
    logger.info("\n--- Entity Type Fixes ---")
    entity_fixes = fix_entity_types(store, dry_run=dry_run)
    logger.info("Entity type fixes: %d", entity_fixes)

    # Fix relations
    logger.info("\n--- Relation Fixes ---")
    rel_stats = fix_relations(store, dry_run=dry_run)
    logger.info(
        "Relation fixes: %d deleted (known), %d deleted (self-ref), %d normalized",
        rel_stats["deleted_known"],
        rel_stats["deleted_self_ref"],
        rel_stats["normalized"],
    )

    # Recompute importance
    if args.importance or args.apply:
        logger.info("\n--- Recomputing Entity Importance ---")
        if dry_run:
            logger.info("[DRY-RUN] Would recompute entity importance")
        else:
            updated = compute_entity_importance(store)
            logger.info("Updated importance for %d entities", updated)

    if not dry_run:
        store.conn.cursor().execute("PRAGMA wal_checkpoint(FULL)")
        logger.info("WAL checkpoint done")

    print_stats(store)
    store.close()

    if dry_run:
        logger.info("\nTo apply changes: python3 scripts/kg_cleanup.py --apply")


if __name__ == "__main__":
    main()
