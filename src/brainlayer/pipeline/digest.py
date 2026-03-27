"""Brain Digest pipeline — structured content ingestion.

Takes raw content (transcripts, documents, articles), extracts:
- Entities and relations (via Phase 2 extraction pipeline)
- Sentiment (via Phase 6 rule-based analyzer)
- Action items, decisions, questions (regex-based extraction)
- Confidence tiers for entity quality

Creates a new chunk with source="digest" and links extracted entities.
"""

import json
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

# Feature flag: enable v2 digest pipeline enhancements
DIGEST_V2_ENABLED = os.environ.get("BRAINLAYER_DIGEST_V2", "true").lower() in ("1", "true", "yes")

# Confidence tier thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.5
SUPERSEDE_SIMILARITY_THRESHOLD = 0.85

# Length-tiered cosine similarity thresholds for dedup
# Research-validated: SemHash defaults to 0.90, strict 0.95 misses paraphrases
DEDUP_THRESHOLD_SHORT = 0.95   # <50 tokens: near-exact match required
DEDUP_THRESHOLD_MEDIUM = 0.90  # 50-200 tokens: moderate flexibility
DEDUP_THRESHOLD_LONG = 0.88    # 200+ tokens: more tolerance for paraphrases

def _estimate_tokens(text: str) -> int:
    """Estimate token count using whitespace split (~1.3 tokens per word)."""
    return max(1, int(len(text.split()) * 1.3))


def _get_dedup_threshold(token_count: int) -> float:
    """Return the cosine similarity threshold for dedup based on content length."""
    if token_count < 50:
        return DEDUP_THRESHOLD_SHORT
    elif token_count <= 200:
        return DEDUP_THRESHOLD_MEDIUM
    else:
        return DEDUP_THRESHOLD_LONG


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_duplicates(
    content: str,
    embedding: List[float],
    store: "VectorStore",
    project: Optional[str] = None,
    n_candidates: int = 5,
    exclude_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Find near-duplicate chunks using length-tiered cosine thresholds.

    Args:
        exclude_ids: Set of chunk IDs to exclude from results (e.g., the just-created chunk).

    Returns list of duplicate candidates with chunk_id, score, threshold, content_preview.
    """
    exclude_ids = exclude_ids or set()
    token_count = _estimate_tokens(content)
    threshold = _get_dedup_threshold(token_count)

    try:
        results = store.hybrid_search(
            query_embedding=embedding,
            query_text=content[:200],
            n_results=n_candidates,
            project_filter=project,
        )
    except Exception as e:
        logger.warning("Dedup search failed: %s", e)
        return []

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]

    duplicates = []
    for cid, doc, dist in zip(ids, docs, dists):
        if cid in exclude_ids:
            continue
        score = 1 - dist if dist is not None else 0
        if score >= threshold:
            duplicates.append({
                "chunk_id": cid,
                "score": round(score, 4),
                "threshold": threshold,
                "token_count": token_count,
                "content_preview": (doc or "")[:200],
            })

    return duplicates


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

    # 7. V2 enhancements: dedup detection + audit events
    duplicates: List[Dict[str, Any]] = []
    if DIGEST_V2_ENABLED:
        duplicates = find_duplicates(
            content=content,
            embedding=embedding,
            store=store,
            project=project,
            exclude_ids={chunk_id},
        )
        store.record_event(
            chunk_id=chunk_id,
            action="digest_created",
            by_whom="digest_pipeline_v2",
            reason=f"Digested {len(content)} chars, {len(entities)} entities, {len(duplicates)} duplicates found",
        )

    result: Dict[str, Any] = {
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
            "chunks_created": 1,
            "connections_made": 0,
            "supersedes_proposed": 0,
            **tier_stats,
        },
    }

    if DIGEST_V2_ENABLED and duplicates:
        result["duplicates"] = duplicates
        result["stats"]["duplicates_found"] = len(duplicates)

    return result


def digest_connect(
    content: str,
    store: VectorStore,
    embed_fn: Callable[[str], List[float]],
    title: Optional[str] = None,
    project: Optional[str] = None,
    participants: Optional[List[str]] = None,
    max_related: int = 5,
) -> Dict[str, Any]:
    """Search-connect-propose pipeline for intelligent content integration.

    Unlike digest_content() which stores immediately, this function returns
    a PROPOSAL that the calling agent decides whether to accept. This is the
    domain agent pattern: heavy analysis here, execution decision by caller.

    Pipeline:
    1. Extract key facts, entities, decisions from new content
    2. Search existing knowledge for each extracted entity/topic
    3. Find connections — related chunks that discuss the same topics
    4. Find contradictions — existing chunks that conflict with new content
    5. Propose supersedes — stale chunks this new content should replace
    6. Return structured proposal with all findings

    Args:
        content: New content to analyze and connect
        store: VectorStore instance for searching existing knowledge
        embed_fn: Embedding function for semantic search
        title: Optional title for the content
        project: Optional project filter for search
        participants: Optional list of known participant names
        max_related: Max related chunks to return per search (default 5)

    Returns:
        DigestProposal dict with: extracted facts, connections found,
        contradictions detected, supersede proposals, and suggested stores.
    """
    if not content or not content.strip():
        raise ValueError("content must be non-empty")

    # Step 1: Extract key facts from the new content
    seed_entities = _build_seed_entities(participants)
    chunk_dict = {"id": "connect-preview", "content": content}
    extraction_result = process_chunk(chunk_dict, seed_entities=seed_entities)

    unique_entities = _dedup_entities(extraction_result.entities)
    entities = [
        {
            "name": e.text,
            "entity_type": e.entity_type,
            "confidence": e.confidence,
        }
        for e in unique_entities
    ]

    decisions = _extract_decisions(content)
    action_items = _extract_action_items(content)
    questions = _extract_questions(content)

    # Step 2: Search existing knowledge for each entity/topic
    search_queries = []
    # Use entity names as search queries
    for entity in entities:
        if entity["confidence"] >= MEDIUM_CONFIDENCE_THRESHOLD:
            search_queries.append(entity["name"])
    # Use the title or first line as a topic query
    topic_query = title or content.split("\n")[0][:200]
    if topic_query not in search_queries:
        search_queries.insert(0, topic_query)
    # Use decisions as search queries (may contradict existing)
    for decision in decisions[:3]:
        if decision not in search_queries:
            search_queries.append(decision)

    # Step 3: Find connections via search
    connections: List[Dict[str, Any]] = []
    seen_chunk_ids: set = set()

    for query in search_queries[:10]:  # Cap at 10 searches
        try:
            query_embedding = embed_fn(query)
            results = store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                n_results=max_related,
                project_filter=project,
            )
            # hybrid_search returns {"ids": [[...]], "documents": [[...]], "metadatas": [[...]], "distances": [[...]]}
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                if cid in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(cid)
                score = 1 - dist if dist is not None else 0
                tags = meta.get("tags") or []
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                connections.append({
                    "chunk_id": cid,
                    "content_preview": (doc or "")[:300],
                    "score": score,
                    "matched_query": query,
                    "project": meta.get("project"),
                    "content_type": meta.get("content_type"),
                    "date": meta.get("created_at"),
                    "tags": tags,
                    "importance": meta.get("importance"),
                })
        except Exception as e:
            logger.warning("digest_connect search failed for query '%s': %s", query, e)

    # Step 4: Find contradictions — existing chunks with deprecation language
    # that share keywords with the new decisions
    contradictions: List[Dict[str, Any]] = []
    seen_contradiction_chunks: set = set()
    for conn in connections:
        preview = conn["content_preview"].lower()
        if not any(
            word in preview
            for word in ["instead", "replaced", "deprecated", "switched from", "no longer"]
        ):
            continue
        # Find which decision(s) relate to this chunk by keyword overlap
        for decision in decisions:
            decision_words = set(decision.lower().split()) - {"to", "the", "a", "an", "we", "i", "is", "are", "was"}
            if len(decision_words) < 2:
                continue
            overlap = sum(1 for w in decision_words if w in preview)
            if overlap >= 2 and conn["chunk_id"] not in seen_contradiction_chunks:
                seen_contradiction_chunks.add(conn["chunk_id"])
                contradictions.append({
                    "new_decision": decision,
                    "existing_chunk_id": conn["chunk_id"],
                    "existing_preview": conn["content_preview"],
                    "reason": "Existing chunk contains replacement/deprecation language with keyword overlap",
                })

    # Step 5: Propose supersedes — chunks that this new content should replace
    supersede_proposals: List[Dict[str, Any]] = []
    content_lower = content.lower()
    for conn in connections:
        preview_lower = conn["content_preview"].lower()
        # If the connected chunk discusses the same topic with lower importance
        # or is older content about the same decision, propose supersede
        is_same_topic = conn["score"] > SUPERSEDE_SIMILARITY_THRESHOLD
        is_old_decision = conn["content_type"] == "decision" and any(
            d.lower() in preview_lower for d in decisions
        )
        if is_same_topic or is_old_decision:
            supersede_proposals.append({
                "chunk_id": conn["chunk_id"],
                "content_preview": conn["content_preview"],
                "reason": "high_similarity" if is_same_topic else "updated_decision",
                "score": conn["score"],
            })

    # Step 6: V2 dedup detection
    duplicates: List[Dict[str, Any]] = []
    if DIGEST_V2_ENABLED:
        content_embedding = embed_fn(content)
        duplicates = find_duplicates(
            content=content,
            embedding=content_embedding,
            store=store,
            project=project,
        )

    # Step 7: Build the proposed store actions
    suggested_stores: List[Dict[str, Any]] = []
    # Suggest storing the new content as a chunk
    suggested_stores.append({
        "action": "store",
        "content": content,
        "title": title,
        "project": project,
        "tags": [e["name"].lower().replace(" ", "-") for e in entities[:5]],
        "importance": _auto_propose_importance(entities, decisions, action_items),
        "supersedes": [s["chunk_id"] for s in supersede_proposals],
    })

    result: Dict[str, Any] = {
        "status": "proposal",
        "content_preview": (title or content[:200]).strip(),
        "extracted": {
            "entities": entities,
            "decisions": decisions,
            "action_items": action_items,
            "questions": questions,
        },
        "related_chunks": connections[:20],
        "connections": connections[:20],  # backward compat alias
        "contradictions": contradictions,
        "duplicates": duplicates,
        "supersede_proposals": supersede_proposals,
        "suggested_stores": suggested_stores,
        "stats": {
            "entities_found": len(entities),
            "searches_performed": min(len(search_queries), 10),
            "connections_found": len(connections),
            "connections_made": len(connections),
            "contradictions_found": len(contradictions),
            "duplicates_found": len(duplicates),
            "supersedes_proposed": len(supersede_proposals),
            "chunks_created": 0,
        },
    }

    return result


def _auto_propose_importance(
    entities: List[Dict],
    decisions: List[str],
    action_items: List[str],
) -> int:
    """Heuristic importance score for proposed store action."""
    score = 5  # baseline
    if decisions:
        score += 2
    if action_items:
        score += 1
    high_conf = sum(1 for e in entities if e.get("confidence", 0) >= HIGH_CONFIDENCE_THRESHOLD)
    if high_conf >= 3:
        score += 1
    return min(score, 10)


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
