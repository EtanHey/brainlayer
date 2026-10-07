"""Offline legacy prompt rendering for compatibility and frozen evaluation fixtures.

This module contains no model transport or producer entry point.
"""

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from ..tag_normalization import enrichment_tag_mode

logger = logging.getLogger(__name__)


_prompt_signature_emitted = False


_prompt_signature_lock = threading.Lock()


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
