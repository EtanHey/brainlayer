#!/usr/bin/env python3
"""Batch KG rebuild — two-tier entity extraction over enriched chunks.

Tier 1: Local extraction (seed matching + tag parsing) — zero API calls
Tier 2: Groq NER with multi-chunk batching — rate-limited API calls

Usage:
    # Tier 1 only (fast, no API calls):
    python3 scripts/kg_rebuild.py --tier1

    # Tier 2 only (Groq NER, rate-limited):
    python3 scripts/kg_rebuild.py --tier2 --limit 5000

    # Both tiers:
    python3 scripts/kg_rebuild.py --tier1 --tier2 --limit 5000

    # Resume Tier 2 from where it left off:
    python3 scripts/kg_rebuild.py --tier2 --resume
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from brainlayer.paths import get_db_path
from brainlayer.pipeline.batch_extraction import DEFAULT_SEED_ENTITIES
from brainlayer.pipeline.entity_extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    extract_entities_from_tags,
    extract_seed_entities,
)
from brainlayer.pipeline.kg_extraction import process_extraction_result
from brainlayer.pipeline.kg_extraction_groq import (
    RateLimiter,
    build_multi_chunk_ner_prompt,
    call_groq_ner,
    parse_multi_chunk_response,
)
from brainlayer.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Progress tracking file
PROGRESS_FILE = Path(__file__).parent / ".kg_rebuild_progress.json"


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"tier1_done": False, "tier2_last_offset": 0, "tier2_processed": 0}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


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


def tier2_groq_ner(
    store: VectorStore,
    limit: int = 5000,
    chunks_per_call: int = 5,
    resume: bool = False,
) -> dict:
    """Tier 2: Groq NER extraction for high-importance chunks.

    Batches multiple chunks per API call for efficiency.
    """
    logger.info("=== Tier 2: Groq NER Extraction ===")

    progress = load_progress() if resume else {"tier2_processed": 0}

    cursor = store._read_cursor()
    # Use conservative rate limit — enrichment pipeline shares the 30 RPM quota
    rate_limiter = RateLimiter(max_per_minute=10)

    # Get high-importance enriched chunks not yet KG-extracted
    # Use a LEFT JOIN to skip chunks already linked to entities
    query = """
        SELECT c.id, c.content
        FROM chunks c
        LEFT JOIN kg_entity_chunks ec ON c.id = ec.chunk_id
        WHERE c.summary IS NOT NULL AND c.summary != ''
          AND c.importance >= 6
          AND ec.chunk_id IS NULL
          AND c.content IS NOT NULL
          AND LENGTH(c.content) > 50
        ORDER BY c.importance DESC, c.id
        LIMIT ?
    """

    stats = {
        "api_calls": 0,
        "chunks_processed": 0,
        "entities_found": 0,
        "relations_found": 0,
        "errors": 0,
    }

    while stats["chunks_processed"] < limit:
        rows = list(cursor.execute(query, (chunks_per_call,)))
        if not rows:
            logger.info("No more unprocessed chunks")
            break

        chunks = [{"id": r[0], "content": r[1]} for r in rows]

        try:
            rate_limiter.wait_if_needed()
            prompt = build_multi_chunk_ner_prompt(chunks)
            response = call_groq_ner(prompt)

            if not response:
                logger.warning("Empty Groq response, skipping batch")
                stats["errors"] += 1
                continue

            parsed_results = parse_multi_chunk_response(response)
            stats["api_calls"] += 1

            for chunk_result in parsed_results:
                chunk_id = chunk_result["chunk_id"]
                # Find the original content for span matching
                content = ""
                for c in chunks:
                    if c["id"] == chunk_id:
                        content = c["content"]
                        break

                entities = []
                for ent_data in chunk_result.get("entities", []):
                    text = ent_data.get("text", "")
                    etype = ent_data.get("type", "")
                    if not text or not etype:
                        continue
                    # Find span in content
                    idx = content.lower().find(text.lower()) if content else -1
                    entities.append(
                        ExtractedEntity(
                            text=text,
                            entity_type=etype,
                            start=idx,
                            end=idx + len(text) if idx >= 0 else -1,
                            confidence=0.75,
                            source="llm",
                        )
                    )

                relations = []
                for rel_data in chunk_result.get("relations", []):
                    source = rel_data.get("source", "")
                    target = rel_data.get("target", "")
                    rtype = rel_data.get("type", "")
                    if source and target and rtype:
                        relations.append(
                            ExtractedRelation(
                                source_text=source,
                                target_text=target,
                                relation_type=rtype,
                                confidence=0.70,
                            )
                        )

                if entities or relations:
                    result = ExtractionResult(
                        entities=entities,
                        relations=relations,
                        chunk_id=chunk_id,
                    )
                    kg_stats = process_extraction_result(store, result)
                    stats["entities_found"] += kg_stats["entities_created"]
                    stats["relations_found"] += kg_stats["relations_created"]

            stats["chunks_processed"] += len(chunks)

        except Exception:
            logger.exception("Error in Groq NER batch")
            stats["errors"] += 1

        # Save progress every batch
        progress["tier2_processed"] = stats["chunks_processed"]
        save_progress(progress)

        if stats["api_calls"] % 10 == 0:
            logger.info(
                "Tier 2 progress: %d chunks, %d API calls, %d entities, %d relations",
                stats["chunks_processed"],
                stats["api_calls"],
                stats["entities_found"],
                stats["relations_found"],
            )

    logger.info("=== Tier 2 Complete ===")
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
    parser.add_argument("--tier2", action="store_true", help="Run Tier 2 (Groq NER)")
    parser.add_argument("--limit", type=int, default=5000, help="Max chunks for Tier 2")
    parser.add_argument("--chunks-per-call", type=int, default=5, help="Chunks per Groq API call")
    parser.add_argument("--resume", action="store_true", help="Resume Tier 2 from last checkpoint")
    parser.add_argument("--stats", action="store_true", help="Print KG stats and exit")
    args = parser.parse_args()

    db_path = get_db_path()
    logger.info("Using DB: %s", db_path)
    store = VectorStore(db_path)

    if args.stats:
        print_kg_stats(store)
        store.close()
        return

    if not args.tier1 and not args.tier2:
        parser.print_help()
        print("\nSpecify --tier1, --tier2, or both.")
        store.close()
        return

    print_kg_stats(store)

    if args.tier1:
        tier1_stats = tier1_seed_and_tags(store)
        print_kg_stats(store)

    if args.tier2:
        tier2_stats = tier2_groq_ner(
            store,
            limit=args.limit,
            chunks_per_call=args.chunks_per_call,
            resume=args.resume,
        )
        print_kg_stats(store)

    store.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
