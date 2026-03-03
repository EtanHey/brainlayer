"""Groq-backed KG extraction with multi-chunk batching.

Sends multiple chunks in a single API call for efficient entity+relation extraction.
Uses Groq's JSON mode for structured output.

Rate limit: 30 RPM on free tier. Multi-chunk batching multiplies throughput.
"""

import json
import logging
import re
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Multi-chunk NER prompt — processes N chunks in one API call
_MULTI_CHUNK_NER_PROMPT = """Extract named entities and relationships from developer conversation chunks.

Entity types (choose carefully):
- person: Human names only (First Last). NOT project names, repos, or tools.
- agent: AI agents and autonomous tools (*Claude, *Golem, Ralph, ClaudeGolem).
- company: Business entities (Cantaloupe AI, Domica, MeHayom, Weby, Union).
- project: Code repos, apps, products (brainlayer, golems, voicelayer, 6pm, soltome).
- tool: Developer tools and services (CodeRabbit, Railway, Vercel).
- technology: Languages, frameworks, libraries (Python, React, SQLite, Convex).
- topic: Abstract concepts only when not fitting above types.

Relation types and DIRECTION rules (source → target):
- works_at: person → company (person works at company)
- owns: person → project/company (person owns the project)
- builds: person/agent → project (who builds what)
- uses: entity → tool/technology (who uses what tool)
- client_of: person/company → person/company (A is a client OF B, meaning B serves A)
- affiliated_with: person → company (generic association)
- coaches: agent → person (agent coaches person, e.g. coachClaude coaches Etan)
- related_to: any → any (generic, use only when no specific type fits)

Return JSON with this exact structure:
{{"chunks": [{{"chunk_id": "id", "entities": [{{"text": "exact text", "type": "entity_type"}}], "relations": [{{"source": "entity text", "target": "entity text", "type": "relation_type", "fact": "natural language description"}}]}}]}}

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

    results = []
    for chunk_data in parsed.get("chunks", []):
        if not isinstance(chunk_data, dict):
            continue
        chunk_id = chunk_data.get("chunk_id", "")
        if not chunk_id:
            continue
        entities = chunk_data.get("entities", [])
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


def call_groq_ner(prompt: str, timeout: int = 60, max_retries: int = 5) -> Optional[str]:
    """Call Groq API for NER extraction with 429 retry backoff.

    Separate from the enrichment call_groq to avoid interfering with
    the enrichment pipeline's rate limiting and logging.
    """
    import os
    import random

    import requests

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        # Try loading from .env
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("GROQ_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        break

    if not api_key:
        logger.error("GROQ_API_KEY not set")
        return None

    url = os.environ.get(
        "BRAINLAYER_GROQ_URL",
        "https://api.groq.com/openai/v1/chat/completions",
    )
    model = os.environ.get("BRAINLAYER_GROQ_MODEL", "llama-3.3-70b-versatile")

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
                timeout=timeout,
            )
            if resp.status_code == 429:
                # Rate limited — extract retry-after or use exponential backoff
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    wait = float(retry_after) + random.uniform(0.5, 2.0)
                else:
                    wait = min(30 * (2**attempt), 120) + random.uniform(1, 5)
                logger.info("Rate limited (429), waiting %.1fs (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return None

        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                wait = min(30 * (2**attempt), 120) + random.uniform(1, 5)
                logger.info("Rate limited, waiting %.1fs (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            logger.error("Groq NER HTTP error: %s", e)
            return None
        except Exception as e:
            logger.error("Groq NER error: %s", e)
            return None

    logger.error("Groq NER: max retries (%d) exhausted", max_retries)
    return None


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_per_minute: int = 28):
        self.max_per_minute = max_per_minute
        self.calls: list[float] = []

    def wait_if_needed(self):
        """Block until we're under the rate limit."""
        now = time.time()
        # Remove calls older than 60s
        self.calls = [t for t in self.calls if now - t < 60]
        if len(self.calls) >= self.max_per_minute:
            wait_time = 60 - (now - self.calls[0]) + 0.5
            if wait_time > 0:
                logger.info("Rate limit: waiting %.1fs", wait_time)
                time.sleep(wait_time)
        self.calls.append(time.time())
