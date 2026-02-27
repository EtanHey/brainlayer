"""Batch entity extraction pipeline.

Processes chunks through entity extraction and stores results in the KG.
Ties together: entity_extraction (NER) + entity_resolution (dedup) + VectorStore (storage).
"""

import logging
import uuid
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

from ..vector_store import VectorStore
from .entity_extraction import (
    ExtractionResult,
    extract_entities_combined,
)
from .entity_resolution import resolve_entity

# Default seed entities for Etan's ecosystem
DEFAULT_SEED_ENTITIES: dict[str, list[str]] = {
    "person": [
        "Etan Heyman",
        "Dor Zohar",
        "Shachar Gerby",
        "Maor Noah",
        "Avi Simon",
        "Yuval Nir",
        "Daniel Munk",
    ],
    "company": ["Cantaloupe AI", "Domica", "MeHayom", "ProductDZ", "Weby"],
    "project": [
        "brainlayer",
        "voicelayer",
        "golems",
        "songscript",
        "domica",
        "rudy-monorepo",
        "union",
    ],
    "golem": [
        "golemsClaude",
        "brainClaude",
        "voiceClaude",
        "coachClaude",
        "contentClaude",
    ],
}


def process_chunk(
    chunk: dict[str, Any],
    seed_entities: Optional[dict[str, list[str]]] = None,
    llm_caller: Optional[Callable] = None,
) -> ExtractionResult:
    """Process a single chunk through entity extraction.

    Returns ExtractionResult with entities and relations found in the chunk.
    """
    if seed_entities is None:
        seed_entities = DEFAULT_SEED_ENTITIES

    text = chunk.get("content", "")
    chunk_id = chunk.get("id", "")

    result = extract_entities_combined(
        text,
        seed_entities,
        llm_caller=llm_caller,
        use_llm=llm_caller is not None,
    )
    result.chunk_id = chunk_id

    return result


def _dedup_entities(entities: list) -> list:
    """Deduplicate extracted entities by (normalized_name, entity_type).

    Keeps the highest-confidence mention for each unique (name, type) pair.
    """
    seen: dict[tuple[str, str], Any] = {}
    for entity in entities:
        key = (entity.text.strip().lower(), entity.entity_type)
        if key not in seen or entity.confidence > seen[key].confidence:
            seen[key] = entity
    return list(seen.values())


def store_extraction_result(
    result: ExtractionResult,
    store: VectorStore,
) -> dict[str, str]:
    """Store extraction results into the KG.

    Returns mapping of entity text -> entity_id for all stored entities.
    Deduplicates entities by (normalized_name, entity_type) before resolution.
    """
    entity_ids: dict[str, str] = {}

    # 1. Deduplicate entities before resolution
    unique_entities = _dedup_entities(result.entities)

    # 2. Resolve and store entities
    for entity in unique_entities:
        entity_id = resolve_entity(
            entity.text,
            entity.entity_type,
            "",  # context
            store,
        )
        entity_ids[entity.text] = entity_id

        # Link entity to source chunk
        if result.chunk_id:
            store.link_entity_chunk(entity_id, result.chunk_id, relevance=entity.confidence)

    # 3. Store relations
    for relation in result.relations:
        source_id = entity_ids.get(relation.source_text)
        target_id = entity_ids.get(relation.target_text)

        if source_id and target_id:
            rel_id = f"rel-{uuid.uuid4().hex[:12]}"
            store.add_relation(
                rel_id,
                source_id,
                target_id,
                relation.relation_type,
                properties=relation.properties,
            )

    return entity_ids


def process_batch(
    chunks: list[dict[str, Any]],
    store: VectorStore,
    seed_entities: Optional[dict[str, list[str]]] = None,
    llm_caller: Optional[Callable] = None,
) -> dict[str, int]:
    """Process a batch of chunks through extraction and store to KG.

    Returns stats dict with processing metrics.
    """
    stats = {
        "chunks_processed": 0,
        "entities_found": 0,
        "relations_found": 0,
        "errors": 0,
    }

    for chunk in chunks:
        try:
            result = process_chunk(chunk, seed_entities, llm_caller)
            entity_ids = store_extraction_result(result, store)

            stats["chunks_processed"] += 1
            stats["entities_found"] += len(result.entities)
            stats["relations_found"] += len(result.relations)
        except Exception:
            logger.exception("Error processing chunk %s", chunk.get("id", "unknown"))
            stats["chunks_processed"] += 1
            stats["errors"] += 1

    return stats
