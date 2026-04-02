#!/usr/bin/env python3
"""KG Entity Backfill — run entity extraction on chunks missing KG links.

Usage:
    python3 scripts/kg_backfill.py --batch 100      # Process 100 chunks
    python3 scripts/kg_backfill.py --batch 1000     # Process 1000 chunks
    python3 scripts/kg_backfill.py --dry-run         # Count only, no writes
"""

import argparse
import sys
import time

sys.path.insert(0, "src")

from brainlayer.paths import get_db_path
from brainlayer.pipeline.batch_extraction import process_batch
from brainlayer.vector_store import VectorStore


def get_unlinked_chunks(store: VectorStore, limit: int) -> list[dict]:
    """Get chunks that have no entity links, ordered by importance/recency."""
    cursor = store.conn.cursor()
    rows = cursor.execute(
        """
        SELECT c.id, c.content
        FROM chunks c
        WHERE c.id NOT IN (SELECT DISTINCT chunk_id FROM kg_entity_chunks)
        AND c.content IS NOT NULL AND length(c.content) > 100
        AND c.content_type IN ('user_message', 'assistant_text', 'ai_code')
        ORDER BY c.created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [{"id": row[0], "content": row[1]} for row in rows]


def count_kg_stats(store: VectorStore) -> dict:
    """Get current KG statistics."""
    cursor = store.conn.cursor()
    return {
        "entities": cursor.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0],
        "relations": cursor.execute("SELECT COUNT(*) FROM kg_relations").fetchone()[0],
        "links": cursor.execute("SELECT COUNT(*) FROM kg_entity_chunks").fetchone()[0],
    }


def main():
    parser = argparse.ArgumentParser(description="KG Entity Backfill")
    parser.add_argument("--batch", type=int, default=100, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Count only")
    args = parser.parse_args()

    store = VectorStore(get_db_path())
    before = count_kg_stats(store)
    print(f"Before: {before['entities']} entities, {before['relations']} relations, {before['links']} links")

    chunks = get_unlinked_chunks(store, args.batch)
    print(f"Found {len(chunks)} unlinked chunks to process (batch={args.batch})")

    if args.dry_run:
        print("Dry run — no changes made.")
        return

    if not chunks:
        print("No unlinked chunks found.")
        return

    start = time.time()
    stats = process_batch(chunks, store)
    elapsed = time.time() - start

    after = count_kg_stats(store)
    new_entities = after["entities"] - before["entities"]
    new_relations = after["relations"] - before["relations"]
    new_links = after["links"] - before["links"]

    print(f"\nProcessed {stats['chunks_processed']} chunks in {elapsed:.1f}s ({elapsed/len(chunks)*1000:.0f}ms/chunk)")
    print(f"  Entities found: {stats['entities_found']} (new: {new_entities})")
    print(f"  Relations found: {stats['relations_found']} (new: {new_relations})")
    print(f"  Entity-chunk links: +{new_links}")
    print(f"  Errors: {stats['errors']}")
    print(f"\nAfter: {after['entities']} entities, {after['relations']} relations, {after['links']} links")


if __name__ == "__main__":
    main()
