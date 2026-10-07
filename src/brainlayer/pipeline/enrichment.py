"""Batch enrichment pipeline — add LLM-generated metadata to BrainLayer chunks.

Processes unenriched chunks through the configured backend (Groq by default; Gemini, MLX, or
Ollama) to add 15 fields:
- summary: 2-4 dense sentences extracting key facts, decisions, names, numbers, outcomes
- key_facts: verbatim specific values (PR numbers, dates, paths, versions, error codes)
- tags: structured tags from fixed taxonomy
- importance: 1-10 score
- intent: debugging | designing | configuring | discussing | deciding | implementing | reviewing
- primary_symbols: classes, functions, files mentioned
- resolved_queries: HyDE-style question, keyword-dense query, and hypothetical answer snippet
- epistemic_level: hypothesis | substantiated | validated
- version_scope: version or system state discussed
- debt_impact: introduction | resolution | none
- external_deps: libraries or external APIs used
- entities: name/type/subtype/relation records fed into the knowledge graph
- sentiment_label, sentiment_score, sentiment_signals: chunk-level sentiment

Usage:
    python -m brainlayer.pipeline.enrichment                    # Process 100 chunks
    python -m brainlayer.pipeline.enrichment --batch-size=50    # Smaller batches
    python -m brainlayer.pipeline.enrichment --max=5000         # Process up to 5000
    python -m brainlayer.pipeline.enrichment --parallel=3       # 3 concurrent workers (MLX)
    python -m brainlayer.pipeline.enrichment --stats            # Show progress

AIDEV-NOTE: Two prompt paths exist:
  1. build_prompt()          — for LOCAL LLM enrichment (Ollama/MLX). No sanitization needed.
  2. build_external_prompt() — for ANY external API (Gemini, Groq, etc). Sanitization is MANDATORY.
     This function requires a Sanitizer instance — you literally cannot call it without one.
     cloud_backfill.py and any future external backend MUST use build_external_prompt().
"""

if __name__ == "__main__":
    import sys

    print("Cloud enrichment has been retired. Local checkpoint replay remains available.", file=sys.stderr)
    raise SystemExit(2)

import json
import logging
import math
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from .cloud_scrub import scrub_llm_output
from .entity_extraction import normalize_entity_type

# Thread-local storage for per-thread VectorStore connections.
# APSW connections are not safe for concurrent use from multiple threads.


# AIDEV-NOTE: Uses local LLM only — never sends chunk content to cloud APIs
# Backend selection: auto-detect by default
# - arm64 Mac → mlx (lighter, faster on Apple Silicon)
# - Other → ollama (universal fallback)
# Override with BRAINLAYER_ENRICH_BACKEND=ollama|mlx


# MLX URL: scripts also check MLX_URL for health, so accept both env vars

# Groq cloud API (for NON-PRIVATE content only — sanitization enforced in _enrich_one)
# Rate limiting: Groq free tier allows ~30 req/min. 2s delay = ~30/min max.

# Stall detection: max seconds a single chunk can take before being considered stalled
# Heartbeat: log progress every N chunks (min 1 to avoid ZeroDivisionError)
# Retry: per-chunk retry with exponential backoff
# Circuit breaker: abort batch after N consecutive failures (backend probably dead)
# MLX default timeout (shorter than Ollama — MLX should respond faster)
# Batch fail ratio: pause if more than this fraction of a batch fails
# Health check pause: seconds to wait before retrying after backend detected dead
# MLX restart: allow Python to restart MLX server if it dies

# Supabase usage logging — track GLM calls even though they're free


# High-value content types worth enriching

# Fixed tag taxonomy for coding conversations


def build_prompt(chunk: Dict[str, Any], context_chunks: Optional[List[Dict[str, Any]]] = None) -> str:
    """Build enrichment prompt with optional surrounding context.

    For LOCAL LLM enrichment only (Ollama/MLX). For external APIs, use
    build_external_prompt() which enforces PII sanitization.
    """
    if context_chunks is None:
        context_chunks = []

    content = chunk["content"]
    # Truncate very long chunks to preserve both early and late facts.
    if len(content) > 8000:
        head = content[:4800]
        tail = content[-3200:]
        content = head + "\n[...truncated middle...]\n" + tail

    context_section = ""
    if context_chunks:
        ctx_parts = []
        for ctx in context_chunks[:3]:  # Max 3 context chunks
            ctx_content = ctx["content"][:1000]
            ctx_parts.append(f"[{ctx.get('content_type', '?')}] {ctx_content}")
        context_section = "SURROUNDING CONTEXT:\n" + "\n---\n".join(ctx_parts)

    # Escape braces in content to avoid str.format() crash on code chunks
    safe_content = content.replace("{", "{{").replace("}", "}}")
    if context_section:
        context_section = context_section.replace("{", "{{").replace("}", "}}")

    _emit_prompt_signature_once()

    return ENRICHMENT_PROMPT.format(
        project=chunk.get("project", "unknown"),
        content_type=chunk.get("content_type", "unknown"),
        content=safe_content,
        context_section=context_section,
        tag_rules=_tag_rules_for_prompt(),
    )


# Mid-run fallback state — tracks consecutive failures for automatic backend switching.
# When the primary backend crashes mid-run (e.g., MLX "Abort trap: 6"), the pipeline
# automatically retries failed chunks on the fallback backend instead of losing the entire batch.


def parse_enrichment(text: str) -> Optional[Dict[str, Any]]:
    """Parse GLM's JSON response into enrichment metadata."""
    if not text:
        return None
    try:
        # Find JSON in response
        match = None
        for start in range(len(text)):
            if text[start] == "{":
                for end in range(len(text) - 1, start, -1):
                    if text[end] == "}":
                        try:
                            match = json.loads(text[start : end + 1])
                            break
                        except json.JSONDecodeError:
                            continue
                if match:
                    break

        if not match:
            return None

        # The model may echo a secret from its prompt into any field. Scrub the
        # whole payload before normalizing; a scrub failure lands in the except
        # below and returns None, so nothing unscrubbed is ever persisted.
        match = scrub_llm_output(match)

        # Validate and normalize
        result: Dict[str, Any] = {}

        summary = match.get("summary", "")
        if isinstance(summary, str) and len(summary) > 5:
            result["summary"] = summary[:500]  # Cap at 500 chars

        result["tags"] = normalize_enrichment_tags(match.get("tags", []))
        result["enrichment_metadata"] = enrichment_version_metadata()

        importance = match.get("importance")
        if isinstance(importance, (int, float)):
            result["importance"] = max(1.0, min(10.0, float(importance)))

        intent = match.get("intent", "")
        if isinstance(intent, str) and intent.lower().strip() in VALID_INTENTS:
            result["intent"] = intent.lower().strip()

        # Extended fields (graceful — missing is OK)
        primary_symbols = match.get("primary_symbols", [])
        if isinstance(primary_symbols, list):
            cleaned = [str(s).strip() for s in primary_symbols if isinstance(s, str) and s.strip()][:20]
            if cleaned:
                result["primary_symbols"] = cleaned

        resolved_query = match.get("resolved_query", "")
        if isinstance(resolved_query, str) and len(resolved_query) > 10:
            result["resolved_query"] = resolved_query[:500]

        key_facts = match.get("key_facts", [])
        if isinstance(key_facts, list):
            cleaned = [str(f).strip() for f in key_facts if isinstance(f, str) and f.strip()][:30]
            if cleaned:
                result["key_facts"] = cleaned

        resolved_queries = match.get("resolved_queries", [])
        if isinstance(resolved_queries, list):
            cleaned = [str(q).strip() for q in resolved_queries if isinstance(q, str) and len(q.strip()) > 10][:3]
            if cleaned:
                result["resolved_queries"] = cleaned

        epistemic_level = match.get("epistemic_level", "")
        if isinstance(epistemic_level, str) and epistemic_level.lower().strip() in VALID_EPISTEMIC:
            result["epistemic_level"] = epistemic_level.lower().strip()

        version_scope = match.get("version_scope")
        if isinstance(version_scope, str) and version_scope.strip() and version_scope.lower() != "null":
            result["version_scope"] = version_scope.strip()[:200]

        debt_impact = match.get("debt_impact", "")
        if isinstance(debt_impact, str) and debt_impact.lower().strip() in VALID_DEBT_IMPACT:
            result["debt_impact"] = debt_impact.lower().strip()

        sentiment_label = match.get("sentiment_label", "")
        if isinstance(sentiment_label, str) and sentiment_label.lower().strip() in VALID_SENTIMENTS:
            result["sentiment_label"] = sentiment_label.lower().strip()

        sentiment_score = match.get("sentiment_score")
        if isinstance(sentiment_score, (int, float)) and not isinstance(sentiment_score, bool):
            try:
                normalized_score = float(sentiment_score)
            except (OverflowError, ValueError):
                normalized_score = None
            if normalized_score is not None and math.isfinite(normalized_score):
                result["sentiment_score"] = max(-1.0, min(1.0, normalized_score))

        sentiment_signals = match.get("sentiment_signals", [])
        if isinstance(sentiment_signals, list):
            cleaned = []
            seen = set()
            for signal in sentiment_signals:
                if not isinstance(signal, str):
                    continue
                normalized = signal.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                cleaned.append(normalized)
                if len(cleaned) == 10:
                    break
            if cleaned:
                result["sentiment_signals"] = cleaned

        external_deps = match.get("external_deps", [])
        if isinstance(external_deps, list):
            cleaned = [str(d).strip().lower() for d in external_deps if isinstance(d, str) and d.strip()][:15]
            if cleaned:
                result["external_deps"] = cleaned

        VALID_ENTITY_TYPES = {
            "person",
            "agent",
            "company",
            "project",
            "technology",
            "tool",
            "concept",
            "topic",
            "source",
        }
        entities = match.get("entities", [])
        if isinstance(entities, list):
            cleaned_entities = []
            for e in entities:
                if isinstance(e, dict):
                    name = e.get("name", "")
                    etype = e.get("type", "")
                    if isinstance(name, str) and name.strip() and isinstance(etype, str) and etype.strip():
                        relation = e.get("relation")
                        entity_type, entity_subtype = normalize_entity_type(
                            name,
                            etype,
                            e.get("entity_subtype") or e.get("subtype"),
                        )
                        if entity_type not in VALID_ENTITY_TYPES:
                            continue
                        entity = {"name": name.strip(), "type": entity_type}
                        if entity_subtype:
                            entity["entity_subtype"] = entity_subtype
                        if isinstance(relation, str) and relation.strip() and relation.lower() != "null":
                            entity["relation"] = relation.strip()
                        cleaned_entities.append(entity)
            if cleaned_entities:
                result["entities"] = cleaned_entities[:20]

        # Must have at least summary + tags to be valid
        if "summary" in result and "tags" in result:
            return result
        return None

    except Exception as e:
        logger.debug("Enrichment result validation failed: %s", e)
        return None


from .enrichment_prompts import ENRICHMENT_PROMPT as ENRICHMENT_PROMPT
from .enrichment_prompts import _emit_prompt_signature_once as _emit_prompt_signature_once
from .enrichment_prompts import _prompt_signature_emitted as _prompt_signature_emitted
from .enrichment_prompts import _prompt_signature_lock as _prompt_signature_lock
from .enrichment_prompts import _tag_rules_for_prompt as _tag_rules_for_prompt
from .enrichment_prompts import build_external_prompt as build_external_prompt
from .enrichment_results import ENRICH_BACKEND as ENRICH_BACKEND
from .enrichment_results import ENRICHMENT_PROMPT_VERSION as ENRICHMENT_PROMPT_VERSION
from .enrichment_results import HIGH_VALUE_TYPES as HIGH_VALUE_TYPES
from .enrichment_results import MODEL as MODEL
from .enrichment_results import VALID_DEBT_IMPACT as VALID_DEBT_IMPACT
from .enrichment_results import VALID_EPISTEMIC as VALID_EPISTEMIC
from .enrichment_results import VALID_INTENTS as VALID_INTENTS
from .enrichment_results import VALID_SENTIMENTS as VALID_SENTIMENTS
from .enrichment_results import _detect_default_backend as _detect_default_backend
from .enrichment_results import enrichment_version_metadata as enrichment_version_metadata
from .enrichment_results import normalize_enrichment_tags as normalize_enrichment_tags
