"""Brain Digest pipeline — structured content ingestion.

Takes raw content (transcripts, documents, articles), extracts:
- Entities and relations (via Phase 2 extraction pipeline)
- Sentiment (via Phase 6 rule-based analyzer)
- Action items, decisions, questions (regex-based extraction)
- Confidence tiers for entity quality

Creates a new chunk with source="digest" and links extracted entities.
"""

import logging
import re
import uuid
from typing import Any, Callable, Dict, List, Optional

from ..vector_store import VectorStore
from .batch_extraction import DEFAULT_SEED_ENTITIES, _dedup_entities, process_chunk, store_extraction_result
from .sentiment import analyze_sentiment

logger = logging.getLogger(__name__)

# Confidence tier thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.5

# Action item patterns
ACTION_PATTERNS = [
    re.compile(r"(?:action items?|todo|to-do|tasks?)[:]\s*(.*?)(?:\n\n|\Z)", re.I | re.S),
    re.compile(r"(?:^|\n)\s*\d+\.\s+(.*?)(?=\n\s*\d+\.|\n\n|\Z)", re.S),
    re.compile(r"(?:^|\n)\s*[-*]\s+(?:TODO|ACTION|TASK)[:.]?\s*(.*?)(?:\n|$)", re.I),
]

# Decision patterns
DECISION_PATTERNS = [
    re.compile(r"(?:decided|decision|agreed|chose|chosen|going with)[:.]?\s*(.*?)(?:\.|$)", re.I),
    re.compile(r"(?:we(?:'ll| will)|let(?:'s| us))\s+(use|go with|switch to|implement)\s+(.*?)(?:\.|$)", re.I),
]

# Question patterns
QUESTION_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:[-*])?\s*(.*?\?)\s*$", re.M),
    re.compile(r"\b((?:how|what|why|when|where|should|could|would|can|is|are|do|does)\s+.*?\?)", re.I),
]


def _extract_action_items(text: str) -> List[str]:
    """Extract action items from text using pattern matching."""
    items = []
    for pattern in ACTION_PATTERNS:
        for match in pattern.finditer(text):
            item = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if item and len(item) > 5 and item not in items:
                items.append(item)
    return items[:20]  # Cap at 20


def _extract_decisions(text: str) -> List[str]:
    """Extract decisions from text using pattern matching."""
    decisions = []
    for pattern in DECISION_PATTERNS:
        for match in pattern.finditer(text):
            decision = match.group(0).strip()
            if decision and len(decision) > 5 and decision not in decisions:
                decisions.append(decision)
    return decisions[:10]


def _extract_questions(text: str) -> List[str]:
    """Extract questions from text using pattern matching."""
    questions = []
    for pattern in QUESTION_PATTERNS:
        for match in pattern.finditer(text):
            q = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if q and len(q) > 5 and q not in questions:
                questions.append(q)
    return questions[:10]


def _build_seed_entities(participants: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Build seed entities dict, adding participants as people."""
    seeds = {k: list(v) for k, v in DEFAULT_SEED_ENTITIES.items()}
    if participants:
        for p in participants:
            if p not in seeds["person"]:
                seeds["person"].append(p)
    return seeds


def _classify_confidence(entities: list) -> Dict[str, int]:
    """Classify entities into confidence tiers."""
    high = 0
    needs_review = 0
    low = 0
    for entity in entities:
        conf = entity.get("confidence", 0)
        if conf >= HIGH_CONFIDENCE_THRESHOLD:
            high += 1
        elif conf >= MEDIUM_CONFIDENCE_THRESHOLD:
            needs_review += 1
        else:
            low += 1
    return {"high_confidence": high, "needs_review": needs_review, "low_confidence": low}


def digest_content(
    content: str,
    store: VectorStore,
    embed_fn: Callable[[str], List[float]],
    title: Optional[str] = None,
    project: Optional[str] = None,
    participants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Digest raw content into structured knowledge.

    Creates a new chunk, extracts entities/relations, analyzes sentiment,
    and extracts action items/decisions/questions.

    Args:
        content: Raw text to digest
        store: VectorStore instance
        embed_fn: Function to generate embeddings
        title: Optional title for the content
        project: Optional project name
        participants: Optional list of known participant names

    Returns:
        Structured result dict with digest_id, summary, entities, relations,
        action_items, decisions, questions, sentiment, stats.

    Raises:
        ValueError: If content is empty.
    """
    if not content or not content.strip():
        raise ValueError("content must be non-empty")

    # 1. Create chunk with source="digest"
    chunk_id = f"digest-{uuid.uuid4().hex[:12]}"
    embedding = embed_fn(content)

    chunks = [
        {
            "id": chunk_id,
            "content": content,
            "metadata": {"title": title} if title else {},
            "source_file": "digest",
            "project": project,
            "content_type": "user_message",
            "char_count": len(content),
            "source": "digest",
        }
    ]
    store.upsert_chunks(chunks, [embedding])

    # 2. Entity extraction (Phase 2 pipeline)
    seed_entities = _build_seed_entities(participants)
    chunk_dict = {"id": chunk_id, "content": content}
    extraction_result = process_chunk(chunk_dict, seed_entities=seed_entities)
    entity_ids = store_extraction_result(extraction_result, store)

    # Build entity list for response (deduplicated)
    unique_entities = _dedup_entities(extraction_result.entities)
    entities = []
    for ext_entity in unique_entities:
        eid = entity_ids.get(ext_entity.text)
        entity_data = store.get_entity(eid) if eid else None
        entities.append(
            {
                "name": entity_data["name"] if entity_data else ext_entity.text,
                "entity_type": entity_data["entity_type"] if entity_data else ext_entity.entity_type,
                "confidence": ext_entity.confidence,
                "entity_id": eid,
            }
        )

    # Build relation list for response
    relations = []
    for ext_rel in extraction_result.relations:
        relations.append(
            {
                "source": ext_rel.source_text,
                "target": ext_rel.target_text,
                "relation_type": ext_rel.relation_type,
                "confidence": ext_rel.confidence,
            }
        )

    # 3. Sentiment analysis (Phase 6)
    sentiment = analyze_sentiment(content)

    # Update chunk sentiment in DB
    store.update_sentiment(
        chunk_id,
        label=sentiment["label"],
        score=sentiment["score"],
        signals=sentiment.get("signals"),
    )

    # 4. Extract action items, decisions, questions
    action_items = _extract_action_items(content)
    decisions = _extract_decisions(content)
    questions = _extract_questions(content)

    # 5. Generate summary
    summary = title or content[:200].strip()
    if len(summary) > 200:
        summary = summary[:197] + "..."

    # 6. Confidence tier stats
    tier_stats = _classify_confidence(entities)

    return {
        "digest_id": chunk_id,
        "summary": summary,
        "entities": entities,
        "relations": relations,
        "action_items": action_items,
        "decisions": decisions,
        "questions": questions,
        "sentiment": sentiment,
        "stats": {
            "entities_found": len(entities),
            "relations_found": len(relations),
            "action_items": len(action_items),
            "decisions": len(decisions),
            "questions": len(questions),
            **tier_stats,
        },
    }


def entity_lookup(
    query: str,
    store: VectorStore,
    embed_fn: Callable[[str], List[float]],
    entity_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Look up an entity by name, returning structured info with relations and evidence.

    Tries FTS first, falls back to semantic search.

    Returns:
        Dict with name, entity_type, relations, evidence, or None if not found.
    """
    # Try FTS search first
    results = store.search_entities(query, entity_type=entity_type, limit=5)

    if not results:
        # Fall back to semantic search
        query_embedding = embed_fn(query)
        results = store.search_entities_semantic(query_embedding, entity_type=entity_type, limit=5)

    if not results:
        return None

    # Take best match
    entity = results[0]
    entity_id = entity["id"]

    # Get relations
    relations_raw = store.get_entity_relations(entity_id)
    relations = [
        {
            "relation_type": r["relation_type"],
            "target_name": r.get("target_name") or r.get("source_name", ""),
            "target_type": r.get("target_type") or r.get("source_type", ""),
            "direction": r.get("direction", "outgoing"),
            "confidence": r.get("confidence", 1.0),
        }
        for r in relations_raw
    ]

    # Get evidence chunks
    evidence_raw = store.get_entity_chunks(entity_id, limit=10)
    evidence = [
        {
            "chunk_id": e["chunk_id"],
            "content": e["content"][:300] if e.get("content") else "",
            "relevance": e["relevance"],
            "context": e.get("context", ""),
            "project": e.get("project"),
        }
        for e in evidence_raw
    ]

    return {
        "id": entity_id,
        "name": entity["name"],
        "entity_type": entity["entity_type"],
        "metadata": entity.get("metadata", {}),
        "relations": relations,
        "evidence": evidence,
    }
