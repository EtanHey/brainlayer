"""Brain Digest pipeline — structured content ingestion.

Takes raw content (transcripts, documents, articles), extracts:
- Entities and relations (via Phase 2 extraction pipeline)
- Sentiment (via Phase 6 rule-based analyzer)
- Action items, decisions, questions (regex-based extraction)
- Confidence tiers for entity quality

Creates a new chunk with source="digest" and links extracted entities.
"""

import logging
import os
import random
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from ..vector_store import VectorStore
from .batch_extraction import DEFAULT_SEED_ENTITIES, _dedup_entities, process_chunk, store_extraction_result
from .enrichment import VALID_INTENTS, build_external_prompt
from .sanitize import Sanitizer
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

FACETED_DIGEST_PROMPT = """You are a knowledge base tagger for BrainLayer digests.

Classify the digested content by specific topic, one activity tag, and technology/domain tags.
Return ONLY valid JSON with this exact schema:
{{
  "topics": ["specific-topic", "another-topic"],
  "activity": "act:designing",
  "domains": ["dom:mcp", "dom:python"],
  "confidence": 0.0
}}

Rules:
- topics: 1-3 specific, hyphenated tags about the subject, not generic workflow labels
- activity: exactly one act:* tag
- domains: 0-3 dom:* tags
- confidence: float 0.0-1.0

CHUNK (project: {project}, type: {content_type}):
---
{content}
---

{context_section}
"""

DEFAULT_FACETED_MODEL = os.environ.get("BRAINLAYER_DIGEST_GEMINI_MODEL", "gemini-2.5-flash-lite")
DEFAULT_FACETED_MAX_RETRIES = int(os.environ.get("BRAINLAYER_DIGEST_GEMINI_RETRIES", "12"))
DEFAULT_FACETED_BASE_DELAY = float(os.environ.get("BRAINLAYER_DIGEST_GEMINI_BASE_DELAY", "1.0"))
DEFAULT_FACETED_MAX_DELAY = float(os.environ.get("BRAINLAYER_DIGEST_GEMINI_MAX_DELAY", "120.0"))


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


def _build_faceted_gemini_config() -> Dict[str, Any]:
    """Build Gemini config with thinking disabled for flash models."""
    return {
        "response_mime_type": "application/json",
        "thinking_config": {"thinking_budget": 0},
    }


def _parse_faceted_enrichment(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse digest faceted-tag enrichment output."""
    if not text:
        return None

    try:
        payload = json_like = None
        for start in range(len(text)):
            if text[start] != "{":
                continue
            for end in range(len(text) - 1, start, -1):
                if text[end] != "}":
                    continue
                try:
                    import json

                    json_like = json.loads(text[start : end + 1])
                    break
                except Exception:
                    continue
            if json_like:
                payload = json_like
                break

        if not payload:
            return None

        topics = payload.get("topics", [])
        if not isinstance(topics, list):
            return None
        clean_topics = [str(topic).strip().lower() for topic in topics if isinstance(topic, str) and topic.strip()][:3]

        activity = payload.get("activity")
        if not isinstance(activity, str) or not activity.startswith("act:"):
            return None
        activity = activity.strip().lower()

        domains = payload.get("domains", [])
        if not isinstance(domains, list):
            return None
        clean_domains = [
            str(domain).strip().lower()
            for domain in domains
            if isinstance(domain, str) and domain.strip().startswith("dom:")
        ][:3]

        confidence = payload.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0

        return {
            "topics": clean_topics,
            "activity": activity,
            "domains": clean_domains,
            "confidence": max(0.0, min(1.0, float(confidence))),
        }
    except Exception:
        return None


def _default_faceted_enrich(
    *,
    content: str,
    project: Optional[str],
    title: Optional[str],
    participants: Optional[List[str]],
) -> Dict[str, Any]:
    """Run digest-time faceted tag enrichment through Gemini with sanitization."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {
            "status": "skipped",
            "reason": "GOOGLE_API_KEY not set",
            "provider": "gemini",
            "model": DEFAULT_FACETED_MODEL,
        }

    from google import genai

    chunk = {
        "content": content,
        "project": project or "unknown",
        "content_type": "digest",
        "source": "digest",
        "metadata": {"title": title, "participants": participants or []},
    }
    sanitizer = Sanitizer.from_env()
    prompt, sanitize_result = build_external_prompt(
        chunk,
        sanitizer,
        prompt_template=FACETED_DIGEST_PROMPT,
    )

    client = genai.Client(api_key=api_key)
    last_error: Exception | None = None

    for attempt in range(DEFAULT_FACETED_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=DEFAULT_FACETED_MODEL,
                contents=prompt,
                config=_build_faceted_gemini_config(),
            )
            parsed = _parse_faceted_enrichment(getattr(response, "text", None))
            if not parsed:
                return {
                    "status": "failed",
                    "reason": "invalid_json_response",
                    "provider": "gemini",
                    "model": DEFAULT_FACETED_MODEL,
                    "pii_detected": sanitize_result.pii_detected,
                }
            return {
                **parsed,
                "status": "enriched",
                "provider": "gemini",
                "model": DEFAULT_FACETED_MODEL,
                "pii_detected": sanitize_result.pii_detected,
            }
        except Exception as exc:
            last_error = exc
            if attempt == DEFAULT_FACETED_MAX_RETRIES - 1:
                break
            delay = min(DEFAULT_FACETED_BASE_DELAY * (2**attempt), DEFAULT_FACETED_MAX_DELAY)
            time.sleep(delay + random.uniform(0, delay * 0.2))

    return {
        "status": "failed",
        "reason": str(last_error) if last_error else "unknown_error",
        "provider": "gemini",
        "model": DEFAULT_FACETED_MODEL,
    }


def _activity_to_intent(activity_tag: Optional[str]) -> Optional[str]:
    """Convert act:* tags into the plain intent field used by chunk enrichment."""
    if not activity_tag or not activity_tag.startswith("act:"):
        return None
    intent = activity_tag.split(":", 1)[1].strip().lower()
    return intent if intent in VALID_INTENTS else None


def _is_successful_faceted_result(result: Dict[str, Any]) -> bool:
    """Accept both explicit status markers and minimal successful payloads."""
    if result.get("status") == "enriched":
        return True
    return bool(result.get("topics") or result.get("activity") or result.get("domains"))


def digest_content(
    content: str,
    store: VectorStore,
    embed_fn: Callable[[str], List[float]],
    title: Optional[str] = None,
    project: Optional[str] = None,
    participants: Optional[List[str]] = None,
    faceted_enrich_fn: Optional[Callable[..., Dict[str, Any]]] = None,
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

    faceted_enrich_fn = faceted_enrich_fn or _default_faceted_enrich
    faceted_result = faceted_enrich_fn(
        content=content,
        project=project,
        title=title,
        participants=participants,
    )

    merged_tags: List[str] = []
    if _is_successful_faceted_result(faceted_result):
        merged_tags = (
            faceted_result.get("topics", []) + [faceted_result.get("activity", "")] + faceted_result.get("domains", [])
        )
        merged_tags = [tag for tag in merged_tags if tag]
        faceted_result = {**faceted_result, "status": "enriched"}  # status last to prevent override
        store.update_enrichment(
            chunk_id=chunk_id,
            summary=summary,
            tags=merged_tags,
            intent=_activity_to_intent(faceted_result.get("activity")),
        )

    # 6. Confidence tier stats
    tier_stats = _classify_confidence(entities)

    return {
        "digest_id": chunk_id,
        "summary": summary,
        "tags": merged_tags,
        "entities": entities,
        "relations": relations,
        "action_items": action_items,
        "decisions": decisions,
        "questions": questions,
        "sentiment": sentiment,
        "enrichment": faceted_result,
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
