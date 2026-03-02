#!/usr/bin/env python3
"""KG entity deduplication and cleanup.

Merges duplicates, removes false positives, fixes misclassified entities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.paths import get_db_path
from brainlayer.pipeline.entity_resolution import merge_entities
from brainlayer.vector_store import VectorStore


def cleanup_kg(store: VectorStore, dry_run: bool = True):
    """Run all KG cleanup operations."""
    cursor = store._read_cursor()

    # ── 1. Merge person duplicates ──
    merges = [
        # (keep_type, keep_name, merge_type, merge_name)
        ("person", "Etan Heyman", "person", "etanheyman"),
    ]

    for keep_type, keep_name, merge_type, merge_name in merges:
        keep = store.get_entity_by_name(keep_type, keep_name)
        merge = store.get_entity_by_name(merge_type, merge_name)
        if keep and merge:
            print(f"MERGE: {merge_name} ({merge_type}) → {keep_name} ({keep_type})")
            if not dry_run:
                merge_entities(store, keep["id"], merge["id"])
        else:
            if not keep:
                print(f"  SKIP: keep entity not found: {keep_name} ({keep_type})")
            if not merge:
                print(f"  SKIP: merge entity not found: {merge_name} ({merge_type})")

    # ── 2. Reclassify mistyped entities ──
    reclassify = [
        # (old_type, old_name, new_type, new_name)
        ("person", "ClaudeGolem", "golem", "ClaudeGolem"),
    ]

    for old_type, old_name, new_type, new_name in reclassify:
        entity = store.get_entity_by_name(old_type, old_name)
        if entity:
            # Check if target already exists
            existing = store.get_entity_by_name(new_type, new_name)
            if existing:
                print(f"MERGE (reclassify): {old_name} ({old_type}) → {new_name} ({new_type})")
                if not dry_run:
                    merge_entities(store, existing["id"], entity["id"])
            else:
                print(f"RECLASSIFY: {old_name}: {old_type} → {new_type}")
                if not dry_run:
                    cursor.execute(
                        "UPDATE kg_entities SET entity_type = ? WHERE id = ?",
                        (new_type, entity["id"]),
                    )

    # ── 3. Cross-type merges (keep the more appropriate type) ──
    cross_merges = [
        # Keep technology, merge tool
        ("technology", "bun", "tool", "bun"),
        # Keep golem, merge tool
        ("golem", "Ralph", "tool", "Ralph"),
    ]

    for keep_type, keep_name, merge_type, merge_name in cross_merges:
        keep = store.get_entity_by_name(keep_type, keep_name)
        merge = store.get_entity_by_name(merge_type, merge_name)
        if keep and merge:
            print(f"CROSS-MERGE: {merge_name} ({merge_type}) → {keep_name} ({keep_type})")
            if not dry_run:
                # Move chunk links and relations, delete the merge entity
                merge_entities(store, keep["id"], merge["id"])

    # ── 4. Fix typos ──
    typo_merges = [
        ("tool", "code rabbiut", "technology", "CodeRabbit"),
    ]

    for src_type, src_name, dst_type, dst_name in typo_merges:
        src = store.get_entity_by_name(src_type, src_name)
        dst = store.get_entity_by_name(dst_type, dst_name)
        if src and dst:
            print(f"TYPO-MERGE: {src_name} ({src_type}) → {dst_name} ({dst_type})")
            if not dry_run:
                merge_entities(store, dst["id"], src["id"])
        elif src and not dst:
            print(f"RENAME: {src_name} → {dst_name} ({dst_type})")
            if not dry_run:
                cursor.execute(
                    "UPDATE kg_entities SET name = ?, entity_type = ? WHERE id = ?",
                    (dst_name, dst_type, src["id"]),
                )

    # ── 5. Remove false positives ──
    false_positives = [
        ("project", "AllFiltersModal"),  # React component
        ("project", "PropertyDrawer"),  # React component
        ("project", "CLAUDE.md"),  # Config file
        ("project", "feat/search-map"),  # Git branch
        ("technology", "promise"),  # JS concept, not a technology
        ("technology", "MCPs"),  # Plural of MCP, not a proper entity
        ("technology", "iOS"),  # Too generic from tag extraction
        ("technology", "Android"),  # Too generic from tag extraction
        ("topic", "ALL_BLOCKED"),  # Status code, not a topic
        ("tool", "/prd skill"),  # CLI command
        ("tool", "mcp__claude-in-chrome__computer"),  # MCP tool name
        ("tool", "tabs_context_mcp"),  # MCP tool name
        ("company", "Claude"),  # Not a company
        ("golem", "Claude"),  # Not a proper golem entry
    ]

    for etype, ename in false_positives:
        entity = store.get_entity_by_name(etype, ename)
        if entity:
            print(f"DELETE (false positive): {ename} ({etype})")
            if not dry_run:
                eid = entity["id"]
                cursor.execute("DELETE FROM kg_entity_chunks WHERE entity_id = ?", (eid,))
                cursor.execute("DELETE FROM kg_relations WHERE source_id = ? OR target_id = ?", (eid, eid))
                cursor.execute("DELETE FROM kg_entity_aliases WHERE entity_id = ?", (eid,))
                cursor.execute("DELETE FROM kg_vec_entities WHERE entity_id = ?", (eid,))
                cursor.execute("DELETE FROM kg_entities WHERE id = ?", (eid,))

    # ── 6. Add useful aliases ──
    aliases = [
        ("person", "Etan Heyman", ["Etan", "etanheyman", "EtanHey", "@EtanHey"]),
        ("person", "Dor Zohar", ["Dor"]),
        ("golem", "brainClaude", ["brainlayer-claude", "brainclaude"]),
        ("golem", "golemsClaude", ["golemsclaude", "orcClaude"]),
        ("project", "brainlayer", ["zikaron", "BrainLayer"]),
        ("project", "golems", ["Golems", "ClaudeGolem"]),
    ]

    for etype, ename, alias_list in aliases:
        entity = store.get_entity_by_name(etype, ename)
        if entity:
            for alias in alias_list:
                existing_alias = store.get_entity_by_alias(alias)
                if not existing_alias:
                    print(f"ALIAS: {alias} → {ename} ({etype})")
                    if not dry_run:
                        store.add_entity_alias(alias, entity["id"], alias_type="manual")

    # Print final stats
    ents = list(cursor.execute("SELECT COUNT(*) FROM kg_entities"))[0][0]
    rels = list(cursor.execute("SELECT COUNT(*) FROM kg_relations"))[0][0]
    links = list(cursor.execute("SELECT COUNT(*) FROM kg_entity_chunks"))[0][0]
    types = list(
        cursor.execute("SELECT entity_type, COUNT(*) FROM kg_entities GROUP BY entity_type ORDER BY COUNT(*) DESC")
    )
    print(f"\nFinal KG: {ents} entities, {rels} relations, {links} links")
    print("Types:", {t: c for t, c in types})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KG entity dedup/cleanup")
    parser.add_argument("--execute", action="store_true", help="Actually apply changes (default: dry run)")
    args = parser.parse_args()

    store = VectorStore(get_db_path())
    cleanup_kg(store, dry_run=not args.execute)
    store.close()
