"""Offline multi-chunk KG prompts and credential-scrubbed response parsing.

The historical module path remains for local consumers. Direct Groq NER is retired.
"""

import json
import re
from typing import Any, Optional

from .cloud_scrub import scrub_llm_output
from .entity_extraction import normalize_entity_type

# Multi-chunk NER prompt — processes N chunks in one API call
_MULTI_CHUNK_NER_PROMPT = """Extract named entities and relationships from developer conversation chunks.

Entity types (choose carefully):
- person: Human names only (First Last). NOT project names, repos, or tools.
- agent: AI agents and autonomous tools (*Claude, *Golem, Ralph, ClaudeGolem).
- company: Business entities (Example Corp, Anthropic, OpenAI).
- project: Code repos, apps, products (brainlayer, golems, voicelayer).
- tool: Developer tools and services (CodeRabbit, Railway, Vercel).
- technology: Languages, frameworks, libraries (Python, React, SQLite, Convex).
- topic: Abstract concepts only when not fitting above types.
- source: Content sources you consume FROM: YouTube channels, podcasts, blogs, newsletters (t3.gg, Huberman Lab, Lex Fridman Podcast). NOT the human host — the host is a person.

Relation types and DIRECTION rules (source → target):
- works_at: person → company (person works at company)
- owns: person → project/company/source (person owns the project or content source)
- hosts: person → source (person hosts the channel, podcast, blog, or newsletter)
- appears_on: person → source (person appears as a guest on the source)
- builds: person/agent → project (who builds what)
- uses: entity → tool/technology (who uses what tool)
- client_of: person/company → person/company (A is a client OF B, meaning B serves A)
- affiliated_with: person → company (generic association)
- coaches: agent → person (agent coaches person, e.g. coachClaude coaches Etan)
- related_to: any → any (generic, use only when no specific type fits)

Return JSON with this exact structure:
{{"chunks": [{{"chunk_id": "id", "entities": [{{"text": "exact text", "type": "entity_type", "entity_subtype": "channel|podcast|brand|newsletter or null"}}], "relations": [{{"source": "entity text", "target": "entity text", "type": "relation_type", "fact": "natural language description"}}]}}]}}

Rules:
- Only extract entities that appear verbatim in the text
- Use the exact text from the input (preserve casing)
- If a chunk has no entities, use empty arrays
- Relations must reference entities that exist in the same chunk
- ALWAYS provide a fact: a clear natural-language sentence describing the relationship
- Direction matters: source is the actor/owner, target is the object/owned

{chunks_text}"""


def build_multi_chunk_ner_prompt(chunks: list[dict[str, Any]]) -> str:
    """Build a multi-chunk NER prompt.

    Args:
        chunks: List of dicts with 'id' and 'content' keys.

    Returns:
        Formatted prompt string.
    """
    parts = []
    for chunk in chunks:
        content = chunk.get("content", "")
        # Truncate very long chunks
        if len(content) > 1500:
            content = content[:1500] + "..."
        chunk_id = chunk.get("id", "unknown")
        parts.append(f"CHUNK {chunk_id}:\n{content}")

    chunks_text = "\n---\n".join(parts)
    return _MULTI_CHUNK_NER_PROMPT.format(chunks_text=chunks_text)


def parse_multi_chunk_response(response: str) -> list[dict[str, Any]]:
    """Parse a multi-chunk NER response from Groq.

    Returns list of dicts with chunk_id, entities, relations.
    """
    if not response:
        return []

    parsed = _extract_json(response)
    if not parsed:
        return []
    parsed = scrub_llm_output(parsed)

    results = []
    for chunk_data in parsed.get("chunks", []):
        if not isinstance(chunk_data, dict):
            continue
        chunk_id = chunk_data.get("chunk_id", "")
        if not chunk_id:
            continue
        entities = []
        for entity in chunk_data.get("entities", []):
            if not isinstance(entity, dict):
                continue
            text = entity.get("text", "")
            raw_type = entity.get("type", "")
            if not isinstance(text, str) or not text.strip() or not isinstance(raw_type, str) or not raw_type.strip():
                continue
            entity_type, entity_subtype = normalize_entity_type(
                text,
                raw_type,
                entity.get("entity_subtype") or entity.get("subtype"),
            )
            normalized_entity = {**entity, "type": entity_type}
            if entity_subtype:
                normalized_entity["entity_subtype"] = entity_subtype
                normalized_entity.pop("subtype", None)
            else:
                normalized_entity.pop("entity_subtype", None)
                normalized_entity.pop("subtype", None)
            entities.append(normalized_entity)
        relations = chunk_data.get("relations", [])
        results.append(
            {
                "chunk_id": chunk_id,
                "entities": entities,
                "relations": relations,
            }
        )

    return results


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Extract JSON object from LLM response."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return None
