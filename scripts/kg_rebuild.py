#!/usr/bin/env python3
"""Local KG rebuild from seed matches and stored tags.

Usage: python3 scripts/kg_rebuild.py --tier1
       python3 scripts/kg_rebuild.py --stats

Direct Groq NER (Tier 2) is retired. Existing checkpoints are left untouched.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.paths import get_db_path
from brainlayer.pipeline.batch_extraction import DEFAULT_SEED_ENTITIES
from brainlayer.pipeline.entity_extraction import (
    ExtractedEntity,
    ExtractionResult,
    extract_entities_from_tags,
    extract_seed_entities,
)
from brainlayer.pipeline.kg_extraction import process_extraction_result
from brainlayer.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extracted_entity_from_groq_payload(ent_data: dict, content: str) -> ExtractedEntity | None:
    text = ent_data.get("text", "")
    etype = ent_data.get("type", "")
    if not text or not etype:
        return None
    idx = content.lower().find(text.lower()) if content else -1
    return ExtractedEntity(
        text=text,
        entity_type=etype,
        start=idx,
        end=idx + len(text) if idx >= 0 else -1,
        confidence=0.75,
        source="llm",
        entity_subtype=ent_data.get("entity_subtype"),
    )


def tier1_seed_and_tags(store: VectorStore, batch_size: int = 5000) -> dict:
    """Tier 1: Extract entities from seed matching + enrichment tags.

    Processes all enriched chunks without any API calls.
    """
    logger.info("=== Tier 1: Seed + Tag Extraction ===")
    cursor = store._read_cursor()

    total = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE summary IS NOT NULL AND summary != ''"))[0][0]
    logger.info("Total enriched chunks: %d", total)

    stats = {
        "chunks_processed": 0,
        "entities_found": 0,
        "entities_created": 0,
        "chunks_linked": 0,
        "errors": 0,
    }

    offset = 0
    while offset < total:
        rows = list(
            cursor.execute(
                """SELECT id, content, tags FROM chunks
               WHERE summary IS NOT NULL AND summary != ''
               ORDER BY id
               LIMIT ? OFFSET ?""",
                (batch_size, offset),
            )
        )
        if not rows:
            break

        for chunk_id, content, tags_str in rows:
            try:
                # Seed entity matching on content
                seed_results = extract_seed_entities(content or "", DEFAULT_SEED_ENTITIES)

                # Tag-based extraction
                tag_entities = []
                if tags_str:
                    try:
                        tags = json.loads(tags_str)
                        if isinstance(tags, list):
                            tag_entities = extract_entities_from_tags(tags)
                    except (json.JSONDecodeError, TypeError):
                        pass

                all_entities = seed_results + tag_entities
                if not all_entities:
                    stats["chunks_processed"] += 1
                    continue

                # Dedup by (name, type)
                seen = {}
                for e in all_entities:
                    key = (e.text.lower(), e.entity_type)
                    if key not in seen or e.confidence > seen[key].confidence:
                        seen[key] = e
                unique_entities = list(seen.values())

                stats["entities_found"] += len(unique_entities)

                # Process into KG
                result = ExtractionResult(
                    entities=unique_entities,
                    relations=[],
                    chunk_id=chunk_id,
                )
                kg_stats = process_extraction_result(store, result)
                stats["entities_created"] += kg_stats["entities_created"]
                stats["chunks_linked"] += kg_stats["chunks_linked"]
                stats["chunks_processed"] += 1

            except Exception:
                logger.exception("Error processing chunk %s", chunk_id)
                stats["errors"] += 1
                stats["chunks_processed"] += 1

        offset += batch_size
        logger.info(
            "Tier 1 progress: %d/%d chunks, %d entities found, %d linked",
            stats["chunks_processed"],
            total,
            stats["entities_found"],
            stats["chunks_linked"],
        )

    logger.info("=== Tier 1 Complete ===")
    logger.info("Stats: %s", json.dumps(stats, indent=2))
    return stats


def print_kg_stats(store: VectorStore):
    """Print current KG statistics."""
    cursor = store._read_cursor()
    ents = list(cursor.execute("SELECT COUNT(*) FROM kg_entities"))[0][0]
    rels = list(cursor.execute("SELECT COUNT(*) FROM kg_relations"))[0][0]
    links = list(cursor.execute("SELECT COUNT(*) FROM kg_entity_chunks"))[0][0]
    types = list(
        cursor.execute("SELECT entity_type, COUNT(*) FROM kg_entities GROUP BY entity_type ORDER BY COUNT(*) DESC")
    )
    print(f"\nKG Stats: {ents} entities, {rels} relations, {links} entity-chunk links")
    print("Entity types:", {t: c for t, c in types})


def main():
    parser = argparse.ArgumentParser(description="Batch KG rebuild")
    parser.add_argument("--tier1", action="store_true", help="Run Tier 1 (seed + tag extraction)")
    parser.add_argument("--stats", action="store_true", help="Print KG stats and exit")
    args = parser.parse_args()

    db_path = get_db_path()
    logger.info("Using DB: %s", db_path)
    store = VectorStore(db_path)

    if args.stats:
        print_kg_stats(store)
        store.close()
        return

    if not args.tier1:
        parser.print_help()
        print("\nSpecify --tier1 or --stats.")
        store.close()
        return

    print_kg_stats(store)

    if args.tier1:
        tier1_stats = tier1_seed_and_tags(store)
        print_kg_stats(store)

    store.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
