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

import logging
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
from .enrichment_results import parse_enrichment as parse_enrichment
