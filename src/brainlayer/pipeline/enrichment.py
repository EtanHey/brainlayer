"""Batch enrichment pipeline — add LLM-generated metadata to BrainLayer chunks.

Processes unenriched chunks through local GLM-4.7-Flash (Ollama) to add:
- summary: 1-sentence description
- tags: structured tags from fixed taxonomy
- importance: 1-10 score
- intent: debugging | designing | configuring | discussing | deciding
- primary_symbols: classes, functions, files mentioned
- resolved_query: hypothetical question this chunk answers (HyDE-style)
- epistemic_level: hypothesis | substantiated | validated
- version_scope: version or system state discussed
- debt_impact: introduction | resolution | none
- external_deps: libraries or external APIs used

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

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..vector_store import VectorStore

# Thread-local storage for per-thread VectorStore connections.
# APSW connections are not safe for concurrent use from multiple threads.
_thread_local = threading.local()


def _get_thread_store(db_path: Path) -> VectorStore:
    """Get or create a thread-local VectorStore instance."""
    if not hasattr(_thread_local, "store"):
        _thread_local.store = VectorStore(db_path)
    return _thread_local.store

# AIDEV-NOTE: Uses local LLM only — never sends chunk content to cloud APIs
# Backend selection: ollama (default) or mlx
ENRICH_BACKEND = os.environ.get("BRAINLAYER_ENRICH_BACKEND", "ollama")
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# MLX URL: scripts also check MLX_URL for health, so accept both env vars
MLX_URL = os.environ.get("BRAINLAYER_MLX_URL", os.environ.get("MLX_URL", "http://127.0.0.1:8080/v1/chat/completions"))
MLX_BASE_URL = MLX_URL.rsplit("/v1/", 1)[0] if "/v1/" in MLX_URL else MLX_URL.rstrip("/")
MODEL = os.environ.get("BRAINLAYER_ENRICH_MODEL", "glm-4.7-flash")
MLX_MODEL = os.environ.get("BRAINLAYER_MLX_MODEL", "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit")
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "brainlayer" / "brainlayer.db"

# Supabase usage logging — track GLM calls even though they're free
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


def _sync_stats_to_supabase(store: "VectorStore") -> None:
    """Sync enrichment stats to Supabase for dashboard visibility. Best-effort."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return
    try:
        stats = store.get_enrichment_stats()
        # Get detailed field counts from DB
        cursor = store.conn.cursor()
        total = stats["total_chunks"]
        has_tags = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE tags IS NOT NULL AND tags != ''"))[0][0]
        has_summary = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE summary IS NOT NULL AND summary != ''"))[0][0]
        has_importance = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE importance IS NOT NULL"))[0][0]
        has_intent = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE intent IS NOT NULL AND intent != ''"))[0][0]
        try:
            has_embeddings = list(cursor.execute("SELECT COUNT(*) FROM chunk_vectors_rowids"))[0][0]
        except Exception:
            has_embeddings = 0
        projects = list(cursor.execute(
            "SELECT project, COUNT(*) FROM chunks WHERE project IS NOT NULL GROUP BY project ORDER BY COUNT(*) DESC LIMIT 20"
        ))

        row = {
            "total_chunks": total,
            "embedded": has_embeddings,
            "tagged": has_tags,
            "summarized": has_summary,
            "importance_scored": has_importance,
            "intent_classified": has_intent,
            "projects": [{"project": p, "chunks": c} for p, c in projects],
            "by_intent": stats.get("by_intent", {}),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Upsert — use user_id=null for service-role inserts
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/enrichment_stats",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal,resolution=merge-duplicates",
            },
            json={**row, "user_id": None},
            timeout=5,
        )
        resp.raise_for_status()
    except Exception:
        pass  # Never let sync failure affect enrichment


def _log_glm_usage(prompt_tokens: int, completion_tokens: int, duration_ms: int, model: str = "") -> None:
    """Log LLM usage to Supabase llm_usage table. Best-effort, never blocks enrichment."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/llm_usage",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json={
                "model": model or MODEL,
                "source": "enrichment",
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "cost_usd": 0,
                "tier": "free",
                "duration_ms": duration_ms,
            },
            timeout=2,
        )
        resp.raise_for_status()
    except Exception:
        pass  # Never let logging failure affect enrichment

# High-value content types worth enriching
HIGH_VALUE_TYPES = ["ai_code", "stack_trace", "user_message", "assistant_text"]

# Fixed tag taxonomy for coding conversations
VALID_INTENTS = ["debugging", "designing", "configuring", "discussing", "deciding", "implementing", "reviewing"]
VALID_EPISTEMIC = ["hypothesis", "substantiated", "validated"]
VALID_DEBT_IMPACT = ["introduction", "resolution", "none"]

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
  "intent": "<one of: debugging, designing, configuring, discussing, deciding, implementing, reviewing>",
  "primary_symbols": ["<class/function/file names mentioned>"],
  "resolved_query": "<hypothetical question this chunk answers>",
  "epistemic_level": "<one of: hypothesis, substantiated, validated>",
  "version_scope": "<version or system state discussed, or null>",
  "debt_impact": "<one of: introduction, resolution, none>",
  "external_deps": ["<libraries or external APIs used>"]
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

PRIMARY_SYMBOLS: Extract class names, function names, file paths, and variable names that are central to this chunk. Empty array if none.

RESOLVED_QUERY: Write a natural question that someone would ask to find this chunk. E.g., "How do I fix EADDRINUSE errors in Bun?" or "What's the SQLite busy_timeout fix for concurrent access?"

EPISTEMIC_LEVEL:
- hypothesis: Exploring ideas, asking questions, speculating
- substantiated: Implementing with evidence, citing docs, testing
- validated: Confirmed working, merged, production-tested

DEBT_IMPACT:
- introduction: Adding workarounds, TODOs, known shortcuts
- resolution: Fixing tech debt, removing hacks, cleaning up
- none: Neither introducing nor resolving debt

EXTERNAL_DEPS: Libraries, APIs, services referenced (e.g., "ollama", "supabase", "react-force-graph-3d"). Empty array if none.

Return ONLY the JSON object, no other text."""


def build_prompt(chunk: Dict[str, Any], context_chunks: Optional[List[Dict[str, Any]]] = None) -> str:
    """Build enrichment prompt with optional surrounding context.

    For LOCAL LLM enrichment only (Ollama/MLX). For external APIs, use
    build_external_prompt() which enforces PII sanitization.
    """
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

    # Escape braces in content to avoid str.format() crash on code chunks
    safe_content = content.replace("{", "{{").replace("}", "}}")
    if context_section:
        context_section = context_section.replace("{", "{{").replace("}", "}}")

    return ENRICHMENT_PROMPT.format(
        project=chunk.get("project", "unknown"),
        content_type=chunk.get("content_type", "unknown"),
        content=safe_content,
        context_section=context_section,
    )


def build_external_prompt(
    chunk: Dict[str, Any],
    sanitizer: "Sanitizer",
    context_chunks: Optional[List[Dict[str, Any]]] = None,
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
    from .sanitize import Sanitizer, SanitizeResult  # noqa: F811 — type narrowing

    if context_chunks is None:
        context_chunks = []

    content = chunk["content"]
    # Truncate very long chunks to stay within context
    if len(content) > 4000:
        content = content[:4000] + "\n... [truncated]"

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

    prompt = ENRICHMENT_PROMPT.format(
        project=chunk.get("project", "unknown"),
        content_type=chunk.get("content_type", "unknown"),
        content=safe_content,
        context_section=context_section,
    )

    return prompt, result


def call_glm(prompt: str, timeout: int = 240) -> Optional[str]:
    """Call local GLM via Ollama HTTP API. Logs usage to Supabase."""
    try:
        start_ms = int(time.time() * 1000)
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False, "think": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        duration_ms = int(time.time() * 1000) - start_ms

        # Extract token counts from Ollama response
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0

        # Log to Supabase (best-effort)
        _log_glm_usage(prompt_tokens, completion_tokens, duration_ms)

        return data.get("response", "")
    except Exception as e:
        print(f"  GLM error: {e}", file=sys.stderr)
        return None


def call_mlx(prompt: str, timeout: int = 240) -> Optional[str]:
    """Call local MLX server via OpenAI-compatible API. Logs usage to Supabase."""
    try:
        start_ms = int(time.time() * 1000)
        resp = requests.post(
            MLX_URL,
            json={
                "model": MLX_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        duration_ms = int(time.time() * 1000) - start_ms

        # Extract token counts from OpenAI-compatible response
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Log to Supabase (best-effort) — use MLX model name, not Ollama's
        _log_glm_usage(prompt_tokens, completion_tokens, duration_ms, model=f"mlx:{MLX_MODEL}")

        # Extract response text
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return None
    except Exception as e:
        print(f"  MLX error: {e}", file=sys.stderr)
        return None


def call_llm(prompt: str, timeout: int = 240) -> Optional[str]:
    """Call local LLM using configured backend (ollama or mlx)."""
    if ENRICH_BACKEND == "mlx":
        return call_mlx(prompt, timeout=timeout)
    return call_glm(prompt, timeout=timeout)


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

        # Extended fields (graceful — missing is OK)
        primary_symbols = match.get("primary_symbols", [])
        if isinstance(primary_symbols, list):
            cleaned = [str(s).strip() for s in primary_symbols if isinstance(s, str) and s.strip()][:20]
            if cleaned:
                result["primary_symbols"] = cleaned

        resolved_query = match.get("resolved_query", "")
        if isinstance(resolved_query, str) and len(resolved_query) > 10:
            result["resolved_query"] = resolved_query[:500]

        epistemic_level = match.get("epistemic_level", "")
        if isinstance(epistemic_level, str) and epistemic_level.lower().strip() in VALID_EPISTEMIC:
            result["epistemic_level"] = epistemic_level.lower().strip()

        version_scope = match.get("version_scope")
        if isinstance(version_scope, str) and version_scope.strip() and version_scope.lower() != "null":
            result["version_scope"] = version_scope.strip()[:200]

        debt_impact = match.get("debt_impact", "")
        if isinstance(debt_impact, str) and debt_impact.lower().strip() in VALID_DEBT_IMPACT:
            result["debt_impact"] = debt_impact.lower().strip()

        external_deps = match.get("external_deps", [])
        if isinstance(external_deps, list):
            cleaned = [str(d).strip().lower() for d in external_deps if isinstance(d, str) and d.strip()][:15]
            if cleaned:
                result["external_deps"] = cleaned

        # Must have at least summary + tags to be valid
        if "summary" in result and "tags" in result:
            return result
        return None

    except Exception:
        return None


def _enrich_one(
    store_or_path,
    chunk: Dict[str, Any],
    with_context: bool = True,
) -> bool:
    """Enrich a single chunk. Returns True on success, False on failure.

    Args:
        store_or_path: VectorStore instance (sequential) or Path (parallel, uses thread-local store).
    """
    # In parallel mode, each thread gets its own VectorStore connection.
    if isinstance(store_or_path, Path):
        store = _get_thread_store(store_or_path)
    else:
        store = store_or_path

    context_chunks = []
    if with_context and chunk.get("conversation_id") and chunk.get("position") is not None:
        ctx = store.get_context(chunk["id"], before=2, after=1)
        context_chunks = [
            c for c in ctx.get("context", [])
            if not c.get("is_target")
        ]

    prompt = build_prompt(chunk, context_chunks)
    response = call_llm(prompt)

    enrichment = parse_enrichment(response)
    if enrichment:
        store.update_enrichment(
            chunk_id=chunk["id"],
            summary=enrichment.get("summary"),
            tags=enrichment.get("tags"),
            importance=enrichment.get("importance"),
            intent=enrichment.get("intent"),
            primary_symbols=enrichment.get("primary_symbols"),
            resolved_query=enrichment.get("resolved_query"),
            epistemic_level=enrichment.get("epistemic_level"),
            version_scope=enrichment.get("version_scope"),
            debt_impact=enrichment.get("debt_impact"),
            external_deps=enrichment.get("external_deps"),
        )
        return True
    return False


def enrich_batch(
    store: VectorStore,
    batch_size: int = 50,
    content_types: Optional[List[str]] = None,
    with_context: bool = True,
    parallel: int = 1,
) -> Dict[str, int]:
    """Process one batch of unenriched chunks. Returns counts.

    Args:
        parallel: Number of concurrent workers (1=sequential, >1=ThreadPoolExecutor).
                  MLX server supports concurrent requests. Ollama may not benefit.
    """
    types = content_types or HIGH_VALUE_TYPES
    chunks = store.get_unenriched_chunks(batch_size=batch_size, content_types=types)

    if not chunks:
        return {"processed": 0, "success": 0, "failed": 0}

    success = 0
    failed = 0

    if parallel > 1:
        # Parallel: pass db_path so each thread gets its own VectorStore connection.
        # APSW connections are not safe for concurrent use from multiple threads.
        db_path = store.db_path
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(_enrich_one, db_path, chunk, with_context): chunk
                for chunk in chunks
            }
            for future in as_completed(futures):
                try:
                    if future.result():
                        success += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"  Worker error: {e}", file=sys.stderr)
                    failed += 1

                done = success + failed
                if done % 10 == 0:
                    print(f"  [{done}/{len(chunks)}] ok={success} fail={failed}")
    else:
        # Sequential: one chunk at a time (original behavior)
        for chunk in chunks:
            start = time.time()
            ok = _enrich_one(store, chunk, with_context)
            duration = time.time() - start

            if ok:
                success += 1
            else:
                failed += 1

            done = success + failed
            if done % 10 == 0:
                print(f"  [{done}/{len(chunks)}] {duration:.1f}s | ok={success} fail={failed}")

    return {"processed": len(chunks), "success": success, "failed": failed}


def run_enrichment(
    db_path: Optional[Path] = None,
    batch_size: int = 50,
    max_chunks: int = 0,
    content_types: Optional[List[str]] = None,
    with_context: bool = True,
    parallel: int = 1,
) -> None:
    """Run the enrichment pipeline until done or max reached."""
    path = db_path or DEFAULT_DB_PATH
    store = VectorStore(path)

    try:
        # Check LLM backend is running
        if ENRICH_BACKEND == "mlx":
            try:
                resp = requests.get(f"{MLX_BASE_URL}/v1/models", timeout=5)
                resp.raise_for_status()
                print(f"Backend: MLX ({MLX_BASE_URL})")
            except Exception:
                raise RuntimeError(
                    f"MLX server not running at {MLX_BASE_URL}. Start with: "
                    "python3 -m mlx_lm.server --model <model> --port 8080"
                )
        else:
            try:
                resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
                resp.raise_for_status()
                print(f"Backend: Ollama ({MODEL})")
            except Exception:
                raise RuntimeError("Ollama is not running. Start it with: ollama serve")

        stats = store.get_enrichment_stats()
        print(f"Enrichment status: {stats['enriched']}/{stats['total_chunks']} ({stats['percent']}%)")
        print(f"Remaining: {stats['remaining']}")
        if stats['by_intent']:
            print(f"Intent distribution: {stats['by_intent']}")
        print(f"Batch size: {batch_size}, Max: {max_chunks or 'unlimited'}, Parallel: {parallel}")
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
                parallel=parallel,
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

            # Sync stats to Supabase every 5 batches
            if total_processed % (batch_size * 5) < batch_size:
                _sync_stats_to_supabase(store)

            if max_chunks > 0 and total_processed >= max_chunks:
                print(f"Reached max ({max_chunks}).")
                break

        elapsed = time.time() - start_time
        print(f"\n--- Enrichment Complete ---")
        print(f"Processed: {total_processed} ({total_success} ok, {total_failed} fail)")
        print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

        final_stats = store.get_enrichment_stats()
        print(f"Progress: {final_stats['enriched']}/{final_stats['total_chunks']} ({final_stats['percent']}%)")

        # Final sync to Supabase
        _sync_stats_to_supabase(store)
    finally:
        store.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enrich BrainLayer chunks with LLM metadata")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max", type=int, default=0, help="Max chunks to process (0=unlimited)")
    parser.add_argument("--parallel", type=int, default=1, help="Concurrent workers (1=sequential, 3=recommended for MLX)")
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
            parallel=args.parallel,
        )
