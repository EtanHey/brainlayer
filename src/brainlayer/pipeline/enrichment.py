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
import os
import sys
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_prompt_signature_emitted = False
_prompt_signature_lock = threading.Lock()

from ..tag_normalization import (
    enrichment_tag_mode,
)
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


def _tag_rules_for_prompt() -> str:
    if enrichment_tag_mode() == "taxonomy":
        return """TAG RULES — TAXONOMY WHITELIST MODE:
- Use ONLY the faceted taxonomy labels from src/brainlayer/taxonomy.json (examples: "tech/debug/investigation", "tech/testing", "pm/decision", "project/brainlayer", "platform/github", "meta/noise")
- Do NOT invent free-form singleton tags, language tags, framework tags, names, or issue-specific labels
- Prefer 1-4 high-signal taxonomy labels per chunk"""
    return """TAG RULES — HYBRID TAG MODE:
- Prefer curated base/facet labels from src/brainlayer/taxonomy.json when they fit (examples: "tech/debug/investigation", "tech/testing", "pm/decision", "project/brainlayer", "platform/github", "meta/noise")
- ALSO include retrieval-useful specific leaf tags for concrete tools, libraries, projects, workflows, bugs, and concepts that the base vocabulary does not capture
- Use 2-7 total tags; lowercase; short hyphenated specifics; no sentence fragments or vague tags like "misc", "update", or "general"
- Normalize aliases in the output when obvious: React.js/reactjs/React -> react; Node.js/nodejs -> node; Code Rabbit -> coderabbit
- Keep specific tags when they help future search (examples: "tdd-guard", "context-gating", "gemini-batch", "voice-picker", "brainbar", "sqlite-fts5")
- Do not force a coarse label if a specific tag is the retrieval hook"""


ENRICHMENT_PROMPT = """You are a knowledge extraction engine for a personal knowledge graph. Your summaries will be embedded for vector search AND indexed for full-text keyword search. Write dense, fact-rich extractions — not descriptions.

RULES:
1. Write as if you ARE the expert stating facts — never as a librarian cataloging a document
2. Preserve ALL specific values verbatim: names, numbers, dates, PR numbers, costs, file paths, URLs, error messages, version numbers
3. Summary must be standalone — a reader with no context should learn the key facts
4. Front-load the most important/unique information

BANNED — never start or include these patterns:
- "This chunk/message describes/details/outlines/provides/contains..."
- "The user/assistant is asking/instructing/discussing/explaining..."
- "This is a conversation/discussion about..."
- "The conversation covers/revolves around..."
- Any sentence ABOUT the text rather than FROM the text
If you catch yourself writing about the source, DELETE it and write the actual facts instead.

META-RESEARCH DETECTION:
- If the chunk contains literal tool invocations such as brain_search(...) or brain_entity(...), treat it as meta-research noise
- Set importance to 2
- Add the tag "meta-research"

SUMMARY STYLE BY CONTENT TYPE:
- Decisions/conclusions: State the decision verbatim, who made it, when, why, and what was rejected
- Corrections/instructions: State what changed (old → new), who corrected it, what prior knowledge is now superseded
- Code/technical: Preserve all symbol names, file paths, config values, error messages verbatim; summarize intent around them
- Conversation: Identify speakers, key contributions, agreements, open questions
- Entity/biographical: Full name with aliases, role, relationships to other entities, key facts with dates
- SHORT/CONVERSATIONAL CHUNK: If the chunk is informal conversation, extract ACTIONABLE ITEMS and COMMITMENTS, not discussion flow or filler phrasing

FEW-SHOT EXAMPLES:

BAD (score 1/5 — meta-description, no facts):
  Input: [conversation about a sample project architecture decision]
  Output: "The conversation discussed architecture."
  WHY BAD: Says nothing about WHAT knowledge. Zero retrievable facts.

BAD (score 2/5 — vague, loses specifics):
  Input: [message about two /yash missions fixing nativewind and MCP SDK bugs]
  Output: "This message details two recent /yash missions that fixed issues in nativewind and MCP SDK"
  WHY BAD: Which missions? What issues? What PRs? No dates, no specifics.

GOOD (score 5/5 — dense, specific, retrievable):
  Input: [same /yash missions conversation]
  Output: "Two /yash missions completed (April 5, 2026): (1) nativewind PR #1722 — fixed style resolution bug in CSS property inheritance, (2) MCP SDK — patched auth token refresh race condition causing 401 errors on reconnect. Both PRs merged and deployed."
  WHY GOOD: Dates, PR numbers, specific bug descriptions, outcomes — all preserved.

GOOD (score 5/5 — decision with rationale):
  Input: [conversation deciding on database choice]
  Output: "Database decision (March 12, 2026): Chose SQLite + sqlite-vec over Postgres+pgvector for BrainLayer storage. Rationale: local-first architecture requirement, no server dependency, FTS5 for hybrid search. Rejected: Postgres (requires daemon), Pinecone (cloud-only, latency), ChromaDB (immature FTS)."
  WHY GOOD: Decision, date, what was chosen, why, what was rejected with reasons.

CHUNK (from project: {project}, type: {content_type}):
---
{content}
---

{context_section}

Return this exact JSON structure:
{{
  "summary": "<2-4 dense sentences extracting the key facts, decisions, names, numbers, and outcomes from this chunk — NOT a description of what the chunk is about>",
  "key_facts": ["<extract every specific value: PR numbers, dollar amounts, dates, file paths, entity names, URLs, version numbers, error codes, config values — verbatim strings only>"],
  "tags": ["<tag1>", "<tag2>"],
  "importance": <1-10 integer>,
  "intent": "<one of: debugging, designing, configuring, discussing, deciding, implementing, reviewing>",
  "primary_symbols": ["<class/function/file names mentioned>"],
  "resolved_queries": [
    "<natural-language question a user would ask that this chunk answers — use everyday search vocabulary>",
    "<keyword-dense query optimized for text search — pack in key terms, names, identifiers>",
    "<hypothetical answer snippet — 1-2 sentences written as if answering the question, preserving key terms for embedding similarity>"
  ],
  "epistemic_level": "<one of: hypothesis, substantiated, validated>",
  "version_scope": "<version or system state discussed, or null>",
  "debt_impact": "<one of: introduction, resolution, none>",
  "external_deps": ["<libraries or external APIs used>"],
  "entities": [
    {{"name": "<entity name>", "type": "<person|agent|company|project|technology|tool|concept|topic|source>", "entity_subtype": "<channel|podcast|brand|newsletter or null>", "relation": "<relationship to other entities in this chunk, e.g. 'developer of BrainLayer', 'dependency of MCP SDK', or null>"}}
  ],
  "sentiment_label": "<one of: frustration, confusion, positive, satisfaction, neutral>",
  "sentiment_score": <float from -1.0 to 1.0>,
  "sentiment_signals": ["<words/phrases that indicate the sentiment>"]
}}

{tag_rules}

IMPORTANCE RULES:
- 1-3: Trivial (greetings, short confirmations, file listings)
- 4-6: Moderate (standard code, config changes, routine discussions)
- 7-9: High (bug fixes with root cause, architecture decisions, novel patterns)
- 10: Critical (security fixes, production incidents, key architectural choices)

ENTITIES:
- Extract non-code entities only — people, agents, projects, technologies, companies, tools, concepts, topics, sources
- Sources are content you consume FROM: YouTube channels, podcasts, blogs, newsletters. The human host remains a person.
- Do NOT extract variable names, function names, file paths, or code symbols as entities
- Use the "relation" field to capture how the entity connects to other entities in this chunk

SUMMARY QUALITY CHECK — before returning, verify your summary:
✓ Contains at least one specific name, number, or date from the chunk
✓ Does NOT start with "This chunk" / "This message" / "The user"
✓ A person reading ONLY the summary would learn something concrete
✓ Key decisions include the WHY, not just the WHAT
✓ All PR numbers, costs, dates, file paths, and URLs from the chunk appear in either summary or key_facts

EPISTEMIC RUBRIC: hypothesis = proposed or unverified; substantiated = backed by concrete evidence in the chunk; validated = outcome explicitly confirmed by execution, merge, deploy, or user confirmation
DEBT IMPACT RUBRIC: introduction = creates debt, workaround, TODO, or blocker; resolution = removes debt or closes a blocker; none = no clear debt change
SENTIMENT RUBRIC: frustration = blocked/annoyed/failing; confusion = uncertainty/questioning; positive = upbeat/encouraging; satisfaction = relief/resolution/completion; neutral = factual/no strong affect

Return ONLY the JSON object, no other text."""


def _emit_prompt_signature_once() -> None:
    """Write a single prompt signature line per process for daemon verification."""
    global _prompt_signature_emitted
    if _prompt_signature_emitted:
        return
    with _prompt_signature_lock:
        if _prompt_signature_emitted:
            return
        _prompt_signature_emitted = True
        try:
            os.write(
                2,
                b"ENRICHMENT_PROMPT_LOADED truncation=8000 split=4800/3200 rubrics=epistemic_level,debt_impact,sentiment_label\n",
            )
        except OSError as exc:
            logger.debug("ENRICHMENT_PROMPT_LOADED emit failed: %s", exc)


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


def build_external_prompt(
    chunk: Dict[str, Any],
    sanitizer: "Sanitizer",
    context_chunks: Optional[List[Dict[str, Any]]] = None,
    prompt_template: Optional[str] = None,
) -> tuple[str, "SanitizeResult"]:
    """Build enrichment prompt with MANDATORY PII sanitization for external APIs.

    AIDEV-NOTE: This is THE function for sending content to any external LLM
    (Gemini, Groq, etc). Sanitization is not optional — it's coupled into
    the function signature. You cannot call this without a Sanitizer.

    Args:
        chunk: Chunk dict with at least 'content', 'project', 'content_type'.
        sanitizer: A Sanitizer instance (from Sanitizer.from_env() or custom).
        context_chunks: Optional surrounding chunks for enrichment context.

    Returns:
        Tuple of (prompt_string, sanitize_result). The prompt uses sanitized
        content. The result tracks what was replaced (for audit/mapping).
    """

    if context_chunks is None:
        context_chunks = []

    content = chunk["content"]
    # Truncate very long chunks to preserve both early and late facts.
    if len(content) > 8000:
        head = content[:4800]
        tail = content[-3200:]
        content = head + "\n[...truncated middle...]\n" + tail

    # Sanitize the main content
    metadata = {
        "source": chunk.get("source"),
        "sender": chunk.get("sender"),
        "project": chunk.get("project"),
    }
    result = sanitizer.sanitize(content, metadata)
    sanitized_content = result.sanitized

    # Sanitize context chunks too — merge their replacements into the main result
    context_section = ""
    if context_chunks:
        ctx_parts = []
        for ctx in context_chunks[:3]:
            ctx_content = ctx["content"][:1000]
            ctx_result = sanitizer.sanitize(ctx_content)
            # Merge context PII replacements into main result for full audit trail
            result.replacements.extend(ctx_result.replacements)
            if ctx_result.pii_detected:
                result.pii_detected = True
            ctx_parts.append(f"[{ctx.get('content_type', '?')}] {ctx_result.sanitized}")
        context_section = "SURROUNDING CONTEXT:\n" + "\n---\n".join(ctx_parts)

    # Escape braces for str.format()
    safe_content = sanitized_content.replace("{", "{{").replace("}", "}}")
    if context_section:
        context_section = context_section.replace("{", "{{").replace("}", "}}")

    template = prompt_template or ENRICHMENT_PROMPT
    _emit_prompt_signature_once()

    prompt = template.format(
        project=chunk.get("project", "unknown"),
        content_type=chunk.get("content_type", "unknown"),
        content=safe_content,
        context_section=context_section,
        tag_rules=_tag_rules_for_prompt(),
    )

    return prompt, result


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
