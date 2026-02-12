"""Batch enrichment pipeline — add LLM-generated metadata to Zikaron chunks.

Processes unenriched chunks through local GLM-4.7-Flash (Ollama) to add:
- summary: 1-sentence description
- tags: structured tags from fixed taxonomy
- importance: 1-10 score
- intent: debugging | designing | configuring | discussing | deciding

Usage:
    python -m zikaron.pipeline.enrichment                    # Process 100 chunks
    python -m zikaron.pipeline.enrichment --batch-size=50    # Smaller batches
    python -m zikaron.pipeline.enrichment --max=5000         # Process up to 5000
    python -m zikaron.pipeline.enrichment --stats            # Show progress
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..vector_store import VectorStore

# AIDEV-NOTE: Uses local Ollama GLM only — never sends chunk content to cloud APIs
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "glm-4.7-flash"
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"

# High-value content types worth enriching
HIGH_VALUE_TYPES = ["ai_code", "stack_trace", "user_message", "assistant_text"]

# Fixed tag taxonomy for coding conversations
VALID_INTENTS = ["debugging", "designing", "configuring", "discussing", "deciding", "implementing", "reviewing"]

ENRICHMENT_PROMPT = """You are a metadata extraction assistant. Analyze this conversation chunk and return ONLY a JSON object.

CHUNK (from project: {project}, type: {content_type}):
---
{content}
---

{context_section}

Return this exact JSON structure:
{{
  "summary": "<one sentence describing what this chunk is about>",
  "tags": ["<tag1>", "<tag2>", ...],
  "importance": <1-10 integer>,
  "intent": "<one of: debugging, designing, configuring, discussing, deciding, implementing, reviewing>"
}}

TAG RULES:
- Use lowercase, hyphenated tags (e.g., "bug-fix", "api-design")
- Include: language tags (python, typescript, sql), framework tags (react, fastapi, bun)
- Include: concept tags (error-handling, authentication, deployment, testing, refactoring)
- Include: action tags (bug-fix, feature-dev, code-review, documentation)
- 3-7 tags per chunk

IMPORTANCE RULES:
- 1-3: Trivial (greetings, short confirmations, file listings)
- 4-6: Moderate (standard code, config changes, routine discussions)
- 7-9: High (bug fixes with root cause, architecture decisions, novel patterns)
- 10: Critical (security fixes, production incidents, key architectural choices)

Return ONLY the JSON object, no other text."""


def build_prompt(chunk: Dict[str, Any], context_chunks: Optional[List[Dict[str, Any]]] = None) -> str:
    """Build enrichment prompt with optional surrounding context."""
    if context_chunks is None:
        context_chunks = []

    content = chunk["content"]
    # Truncate very long chunks to stay within GLM context
    if len(content) > 4000:
        content = content[:4000] + "\n... [truncated]"

    context_section = ""
    if context_chunks:
        ctx_parts = []
        for ctx in context_chunks[:3]:  # Max 3 context chunks
            ctx_content = ctx["content"][:1000]
            ctx_parts.append(f"[{ctx.get('content_type', '?')}] {ctx_content}")
        context_section = "SURROUNDING CONTEXT:\n" + "\n---\n".join(ctx_parts)

    return ENRICHMENT_PROMPT.format(
        project=chunk.get("project", "unknown"),
        content_type=chunk.get("content_type", "unknown"),
        content=content,
        context_section=context_section,
    )


def call_glm(prompt: str, timeout: int = 120) -> Optional[str]:
    """Call local GLM via Ollama HTTP API."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        print(f"  GLM error: {e}", file=sys.stderr)
        return None


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

        # Validate and normalize
        result: Dict[str, Any] = {}

        summary = match.get("summary", "")
        if isinstance(summary, str) and len(summary) > 5:
            result["summary"] = summary[:500]  # Cap at 500 chars

        tags = match.get("tags", [])
        if isinstance(tags, list):
            result["tags"] = [str(t).lower().strip() for t in tags if isinstance(t, str)][:10]

        importance = match.get("importance")
        if isinstance(importance, (int, float)):
            result["importance"] = max(1.0, min(10.0, float(importance)))

        intent = match.get("intent", "")
        if isinstance(intent, str) and intent.lower().strip() in VALID_INTENTS:
            result["intent"] = intent.lower().strip()

        # Must have at least summary + tags to be valid
        if "summary" in result and "tags" in result:
            return result
        return None

    except Exception:
        return None


def enrich_batch(
    store: VectorStore,
    batch_size: int = 50,
    content_types: Optional[List[str]] = None,
    with_context: bool = True,
) -> Dict[str, int]:
    """Process one batch of unenriched chunks. Returns counts."""
    types = content_types or HIGH_VALUE_TYPES
    chunks = store.get_unenriched_chunks(batch_size=batch_size, content_types=types)

    if not chunks:
        return {"processed": 0, "success": 0, "failed": 0}

    success = 0
    failed = 0

    for chunk in chunks:
        # Optionally get surrounding context
        context_chunks = []
        if with_context and chunk.get("conversation_id") and chunk.get("position") is not None:
            ctx = store.get_context(chunk["id"], before=2, after=1)
            context_chunks = [
                c for c in ctx.get("context", [])
                if not c.get("is_target")
            ]

        prompt = build_prompt(chunk, context_chunks)
        start = time.time()
        response = call_glm(prompt)
        duration = time.time() - start

        enrichment = parse_enrichment(response)
        if enrichment:
            store.update_enrichment(
                chunk_id=chunk["id"],
                summary=enrichment.get("summary"),
                tags=enrichment.get("tags"),
                importance=enrichment.get("importance"),
                intent=enrichment.get("intent"),
            )
            success += 1
        else:
            # Leave enriched_at NULL so chunk is retried on next run
            failed += 1

        if (success + failed) % 10 == 0:
            elapsed = time.time() - start
            print(f"  [{success + failed}/{len(chunks)}] {duration:.1f}s | ok={success} fail={failed}")

    return {"processed": len(chunks), "success": success, "failed": failed}


def run_enrichment(
    db_path: Optional[Path] = None,
    batch_size: int = 50,
    max_chunks: int = 0,
    content_types: Optional[List[str]] = None,
    with_context: bool = True,
) -> None:
    """Run the enrichment pipeline until done or max reached."""
    path = db_path or DEFAULT_DB_PATH
    store = VectorStore(path)

    try:
        # Check Ollama is running
        try:
            resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
            resp.raise_for_status()
        except Exception:
            raise RuntimeError("Ollama is not running. Start it with: ollama serve")

        stats = store.get_enrichment_stats()
        print(f"Enrichment status: {stats['enriched']}/{stats['total_chunks']} ({stats['percent']}%)")
        print(f"Remaining: {stats['remaining']}")
        if stats['by_intent']:
            print(f"Intent distribution: {stats['by_intent']}")
        print(f"Batch size: {batch_size}, Max: {max_chunks or 'unlimited'}")
        print("---")

        total_processed = 0
        total_success = 0
        total_failed = 0
        start_time = time.time()

        while True:
            result = enrich_batch(
                store,
                batch_size=batch_size,
                content_types=content_types,
                with_context=with_context,
            )

            if result["processed"] == 0:
                print("No more chunks to enrich.")
                break

            total_processed += result["processed"]
            total_success += result["success"]
            total_failed += result["failed"]

            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            print(f"Batch done: +{result['success']} ok, +{result['failed']} fail | "
                  f"Total: {total_processed} ({rate:.1f}/s)")

            if max_chunks > 0 and total_processed >= max_chunks:
                print(f"Reached max ({max_chunks}).")
                break

        elapsed = time.time() - start_time
        print(f"\n--- Enrichment Complete ---")
        print(f"Processed: {total_processed} ({total_success} ok, {total_failed} fail)")
        print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

        final_stats = store.get_enrichment_stats()
        print(f"Progress: {final_stats['enriched']}/{final_stats['total_chunks']} ({final_stats['percent']}%)")
    finally:
        store.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enrich Zikaron chunks with LLM metadata")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max", type=int, default=0, help="Max chunks to process (0=unlimited)")
    parser.add_argument("--no-context", action="store_true", help="Skip surrounding context")
    parser.add_argument("--stats", action="store_true", help="Show enrichment stats and exit")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    args = parser.parse_args()

    db = Path(args.db) if args.db else None

    if args.stats:
        store = VectorStore(db or DEFAULT_DB_PATH)
        stats = store.get_enrichment_stats()
        print(f"Total: {stats['total_chunks']}")
        print(f"Enriched: {stats['enriched']} ({stats['percent']}%)")
        print(f"Remaining: {stats['remaining']}")
        if stats['by_intent']:
            print(f"Intent: {stats['by_intent']}")
        store.close()
    else:
        run_enrichment(
            db_path=db,
            batch_size=args.batch_size,
            max_chunks=args.max,
            with_context=not args.no_context,
        )
