"""KG extraction pipeline — wires entity extraction to KG standard tables.

After a chunk is embedded/enriched, this module:
1. Extracts entities from chunk content (seed + LLM + GLiNER)
2. Resolves each entity against existing kg_entities
3. Creates relations between entities
4. Links entities to the source chunk via kg_entity_chunks

Designed to run as a post-enrichment step in enrich_batch.
"""

import logging
import uuid
from typing import Any, Optional

from ..vector_store import VectorStore
from .entity_extraction import ExtractionResult, extract_entities_combined
from .entity_resolution import resolve_entity

logger = logging.getLogger(__name__)


def _mention_type_from_source(source: str) -> str:
    """Map extraction source to mention_type."""
    if source in ("seed",):
        return "explicit"
    if source in ("gliner",):
        return "explicit"
    if source in ("llm",):
        return "inferred"
    return "inferred"


def process_extraction_result(
    store: VectorStore,
    result: ExtractionResult,
) -> dict[str, int]:
    """Process an ExtractionResult into the KG tables.

    Takes extracted entities and relations, resolves them against the KG,
    and creates/links entries.

    Returns:
        Stats dict with counts of entities_created, relations_created, chunks_linked.
    """
    stats = {"entities_created": 0, "relations_created": 0, "chunks_linked": 0}

    if not result.entities and not result.relations:
        return stats

    chunk_id = result.chunk_id

    # Map extracted text → resolved entity ID for relation linking
    text_to_entity_id: dict[str, str] = {}

    for ext_entity in result.entities:
        # Check if entity already exists before resolution
        pre_existing = store.get_entity_by_name(ext_entity.entity_type, ext_entity.text)

        # Resolve against existing KG (creates new if not found)
        entity_id = resolve_entity(
            name=ext_entity.text,
            entity_type=ext_entity.entity_type,
            context=chunk_id,
            store=store,
        )
        text_to_entity_id[ext_entity.text.lower()] = entity_id

        existing = store.get_entity(entity_id)
        if existing:
            if pre_existing:
                # Entity existed — only update if extraction confidence is higher
                new_confidence = max(existing.get("confidence", 0) or 0, ext_entity.confidence)
            else:
                # Newly created — set confidence from extraction source
                new_confidence = ext_entity.confidence

            store.upsert_entity(
                entity_id,
                ext_entity.entity_type,
                existing["name"],
                confidence=new_confidence,
            )

        stats["entities_created"] += 1

        # Link entity to chunk
        if chunk_id:
            mention_type = _mention_type_from_source(ext_entity.source)
            store.link_entity_chunk(
                entity_id=entity_id,
                chunk_id=chunk_id,
                relevance=ext_entity.confidence,
                mention_type=mention_type,
            )
            stats["chunks_linked"] += 1

    # Process relations
    for ext_rel in result.relations:
        source_id = text_to_entity_id.get(ext_rel.source_text.lower())
        target_id = text_to_entity_id.get(ext_rel.target_text.lower())

        if not source_id or not target_id:
            logger.debug(
                "Skipping relation %s→%s: entity not resolved",
                ext_rel.source_text,
                ext_rel.target_text,
            )
            continue

        rel_id = f"rel-{uuid.uuid4().hex[:12]}"
        store.add_relation(
            relation_id=rel_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=ext_rel.relation_type,
            properties=ext_rel.properties,
            confidence=ext_rel.confidence,
            source_chunk_id=chunk_id,
        )
        stats["relations_created"] += 1

    return stats


def extract_kg_from_chunk(
    store: VectorStore,
    chunk_id: str,
    seed_entities: Optional[dict[str, list[str]]] = None,
    use_llm: bool = True,
    use_gliner: bool = False,
    llm_caller: Optional[Any] = None,
) -> dict[str, int]:
    """Full KG extraction for a single chunk.

    1. Reads chunk content from store
    2. Runs entity extraction (seed + optional LLM/GLiNER)
    3. Processes results into KG tables

    Args:
        store: VectorStore with KG tables
        chunk_id: ID of the chunk to extract from
        seed_entities: Known entities to match by string
        use_llm: Whether to use LLM for extraction
        use_gliner: Whether to use GLiNER model
        llm_caller: Optional LLM callable for testing

    Returns:
        Stats dict with counts.
    """
    empty_stats = {"entities_created": 0, "relations_created": 0, "chunks_linked": 0}

    # Get chunk content
    cursor = store._read_cursor()
    rows = list(cursor.execute("SELECT content FROM chunks WHERE id = ?", (chunk_id,)))
    if not rows:
        return empty_stats

    content = rows[0][0]
    if not content or not content.strip():
        return empty_stats

    # Run extraction
    extraction = extract_entities_combined(
        text=content,
        seed_entities=seed_entities or {},
        llm_caller=llm_caller,
        use_llm=use_llm,
        use_gliner=use_gliner,
    )
    extraction.chunk_id = chunk_id

    # Process into KG
    return process_extraction_result(store, extraction)
