"""KG extraction pipeline — wires entity extraction to KG standard tables.

After a chunk is embedded/enriched, this module:
1. Extracts entities from chunk content (seed + LLM + GLiNER)
2. Validates and corrects entity types and relation directions
3. Resolves each entity against existing kg_entities
4. Creates relations between entities
5. Links entities to the source chunk via kg_entity_chunks

Designed to run as a post-enrichment step in enrich_batch.
"""

import logging
import math
import re
import uuid
from typing import Any, Optional

from ..vector_store import VectorStore
from .entity_extraction import (
    KNOWN_PROJECT_TAGS,
    KNOWN_TECH_TAGS,
    ExtractedRelation,
    ExtractionResult,
    extract_entities_combined,
)
from .entity_resolution import resolve_entity

logger = logging.getLogger(__name__)

# Canonical relation types — everything else maps to 'related_to'
CANONICAL_RELATION_TYPES = {
    "works_at",
    "owns",
    "builds",
    "uses",
    "client_of",
    "affiliated_with",
    "coaches",
    "related_to",
}

# Known agent names (beyond *Claude/*Golem pattern matching)
_KNOWN_AGENTS = {"ralph", "claudegolem"}

# Relation direction constraints: relation_type → (valid_source_types, valid_target_types)
# If extracted direction is wrong, we swap source/target.
_RELATION_DIRECTION_RULES: dict[str, tuple[set[str], set[str]]] = {
    "works_at": ({"person", "agent"}, {"company"}),
    "owns": ({"person"}, {"company", "project", "agent"}),
    "builds": ({"person", "agent"}, {"project", "tool", "technology"}),
    "uses": ({"person", "agent", "project", "company"}, {"tool", "technology"}),
    "coaches": ({"agent"}, {"person"}),
}


def validate_extraction_result(result: ExtractionResult) -> ExtractionResult:
    """Validate and correct extracted entities and relations.

    Applies heuristic guardrails:
    1. Entity type coercion (agent patterns, known projects/tech)
    2. Relation direction validation and correction
    3. Self-referential relation removal
    4. Relation type normalization to canonical types
    """
    # Build entity type lookup from (possibly corrected) entities
    entity_types: dict[str, str] = {}

    # 1. Entity type coercion
    for entity in result.entities:
        name = entity.text
        name_lower = name.lower().replace("-", "").replace("_", "").replace(" ", "")

        # *Claude or *Golem pattern → agent
        if re.search(r"(?i)(claude|golem)$", name):
            entity.entity_type = "agent"
        # Known agent names
        elif name_lower in _KNOWN_AGENTS:
            entity.entity_type = "agent"
        # Known project tags
        elif name_lower in {p.lower().replace("-", "").replace("_", "") for p in KNOWN_PROJECT_TAGS}:
            entity.entity_type = "project"
        # Known tech tags
        elif name_lower in {t.lower().replace("-", "").replace("_", "") for t in KNOWN_TECH_TAGS}:
            entity.entity_type = "technology"

        entity_types[name.lower()] = entity.entity_type

    # 2-4. Validate relations
    validated_relations: list[ExtractedRelation] = []
    for rel in result.relations:
        # 3. Self-referential filter
        if rel.source_text.lower() == rel.target_text.lower():
            logger.debug(
                "Dropping self-referential relation: %s --%s--> %s", rel.source_text, rel.relation_type, rel.target_text
            )
            continue

        # 4. Normalize relation type
        if rel.relation_type not in CANONICAL_RELATION_TYPES:
            rel.relation_type = "related_to"

        # 2. Direction validation
        source_type = entity_types.get(rel.source_text.lower(), "unknown")
        target_type = entity_types.get(rel.target_text.lower(), "unknown")

        if rel.relation_type in _RELATION_DIRECTION_RULES:
            valid_src, valid_tgt = _RELATION_DIRECTION_RULES[rel.relation_type]
            if source_type not in valid_src and target_type in valid_src:
                # Wrong direction — swap
                rel.source_text, rel.target_text = rel.target_text, rel.source_text
                logger.debug(
                    "Swapped relation direction: %s --%s--> %s",
                    rel.source_text,
                    rel.relation_type,
                    rel.target_text,
                )

        validated_relations.append(rel)

    return ExtractionResult(
        entities=result.entities,
        relations=validated_relations,
        chunk_id=result.chunk_id,
        metadata=result.metadata,
    )


def compute_entity_importance(store: VectorStore) -> int:
    """Compute entity importance from chunk links and relations.

    Formula: importance = log2(1 + chunk_count) * avg_chunk_importance * (1 + 0.1 * relation_count)
    Normalized to 0-10 range.

    Returns number of entities updated.
    """
    cursor = store._read_cursor()

    # Get chunk link counts and avg importance per entity
    rows = list(
        cursor.execute(
            """SELECT ec.entity_id,
                      COUNT(*) as link_count,
                      AVG(COALESCE(c.importance, 5)) as avg_imp
               FROM kg_entity_chunks ec
               JOIN chunks c ON ec.chunk_id = c.id
               GROUP BY ec.entity_id"""
        )
    )

    if not rows:
        return 0

    # Get relation counts per entity
    rel_counts: dict[str, int] = {}
    for row in cursor.execute(
        """SELECT entity_id, COUNT(*) FROM (
               SELECT source_id as entity_id FROM kg_relations WHERE expired_at IS NULL
               UNION ALL
               SELECT target_id as entity_id FROM kg_relations WHERE expired_at IS NULL
           ) GROUP BY entity_id"""
    ):
        rel_counts[row[0]] = row[1]

    # Compute raw scores
    scores: dict[str, float] = {}
    for entity_id, link_count, avg_imp in rows:
        n_rels = rel_counts.get(entity_id, 0)
        raw = math.log2(1 + link_count) * (avg_imp / 10.0) * (1 + 0.1 * n_rels)
        scores[entity_id] = raw

    if not scores:
        return 0

    # Normalize to 0.5-9.5 range (leave room at edges)
    max_score = max(scores.values())
    min_score = min(scores.values())
    score_range = max_score - min_score if max_score > min_score else 1.0

    updated = 0
    write_cursor = store.conn.cursor()
    for entity_id, raw in scores.items():
        normalized = 0.5 + 9.0 * (raw - min_score) / score_range
        write_cursor.execute(
            "UPDATE kg_entities SET importance = ? WHERE id = ?",
            (round(normalized, 2), entity_id),
        )
        updated += 1

    logger.info("Updated importance for %d entities (range %.2f-%.2f)", updated, min_score, max_score)
    return updated


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

    # Validate and correct entity types + relation directions
    result = validate_extraction_result(result)

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
        # Extract fact from properties or dedicated field
        fact = getattr(ext_rel, "fact", None) or ext_rel.properties.get("fact")
        store.add_relation(
            relation_id=rel_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=ext_rel.relation_type,
            properties=ext_rel.properties,
            confidence=ext_rel.confidence,
            fact=fact,
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
