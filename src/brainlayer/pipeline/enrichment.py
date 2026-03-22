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
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

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
# Backend selection: auto-detect by default
# - arm64 Mac → mlx (lighter, faster on Apple Silicon)
# - Other → ollama (universal fallback)
# Override with BRAINLAYER_ENRICH_BACKEND=ollama|mlx


def _detect_default_backend() -> str:
    """Auto-detect the best enrichment backend for this platform.

    arm64 Mac → mlx (native Apple Silicon, no Docker overhead)
    Everything else → ollama (universal, works everywhere)
    """
    import platform

    explicit = os.environ.get("BRAINLAYER_ENRICH_BACKEND")
    if explicit:
        return explicit

    if platform.machine() == "arm64" and platform.system() == "Darwin":
        return "mlx"
    return "ollama"


ENRICH_BACKEND = _detect_default_backend()
OLLAMA_URL = os.environ.get("BRAINLAYER_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_BASE_URL = OLLAMA_URL.rsplit("/api/", 1)[0] if "/api/" in OLLAMA_URL else OLLAMA_URL.rstrip("/")
# MLX URL: scripts also check MLX_URL for health, so accept both env vars
MLX_URL = os.environ.get("BRAINLAYER_MLX_URL", os.environ.get("MLX_URL", "http://127.0.0.1:8080/v1/chat/completions"))
MLX_BASE_URL = MLX_URL.rsplit("/v1/", 1)[0] if "/v1/" in MLX_URL else MLX_URL.rstrip("/")
MODEL = os.environ.get("BRAINLAYER_ENRICH_MODEL", "glm-4.7-flash")
MLX_MODEL = os.environ.get("BRAINLAYER_MLX_MODEL", "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit")

# Groq cloud API (for NON-PRIVATE content only — sanitization enforced in _enrich_one)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = os.environ.get("BRAINLAYER_GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.environ.get("BRAINLAYER_GROQ_MODEL", "llama-3.3-70b-versatile")
# Rate limiting: Groq free tier allows ~30 req/min. 2s delay = ~30/min max.
GROQ_RATE_LIMIT_DELAY = float(os.environ.get("BRAINLAYER_GROQ_RATE_DELAY", "2.0"))
_groq_last_call: float = 0.0  # monotonic timestamp of last Groq API call
_groq_rate_lock = threading.Lock()  # serialize rate-limit checks across threads

# Stall detection: max seconds a single chunk can take before being considered stalled
STALL_TIMEOUT = int(os.environ.get("BRAINLAYER_STALL_TIMEOUT", "300"))  # 5 minutes default
# Heartbeat: log progress every N chunks (min 1 to avoid ZeroDivisionError)
HEARTBEAT_INTERVAL = max(1, int(os.environ.get("BRAINLAYER_HEARTBEAT_INTERVAL", "25")))
# Retry: per-chunk retry with exponential backoff
MAX_RETRIES = int(os.environ.get("BRAINLAYER_MAX_RETRIES", "2"))  # 0=no retry, 2=up to 3 attempts
RETRY_BASE_DELAY = float(os.environ.get("BRAINLAYER_RETRY_BASE_DELAY", "2.0"))  # seconds
RETRY_MAX_DELAY = float(os.environ.get("BRAINLAYER_RETRY_MAX_DELAY", "30.0"))  # cap
# Circuit breaker: abort batch after N consecutive failures (backend probably dead)
CIRCUIT_BREAKER_THRESHOLD = int(os.environ.get("BRAINLAYER_CIRCUIT_BREAKER", "10"))
# MLX default timeout (shorter than Ollama — MLX should respond faster)
MLX_DEFAULT_TIMEOUT = int(os.environ.get("BRAINLAYER_MLX_TIMEOUT", "60"))
# Batch fail ratio: pause if more than this fraction of a batch fails
BATCH_FAIL_RATIO_THRESHOLD = float(os.environ.get("BRAINLAYER_BATCH_FAIL_RATIO", "0.8"))
# Health check pause: seconds to wait before retrying after backend detected dead
HEALTH_CHECK_PAUSE = int(os.environ.get("BRAINLAYER_HEALTH_PAUSE", "15"))
# MLX restart: allow Python to restart MLX server if it dies
MLX_AUTO_RESTART = os.environ.get("BRAINLAYER_MLX_AUTO_RESTART", "1") == "1"
MLX_RESTART_WAIT = int(os.environ.get("BRAINLAYER_MLX_RESTART_WAIT", "60"))
from ..paths import DEFAULT_DB_PATH


def check_backend_health(backend: str) -> bool:
    """Check if the LLM backend is reachable. Returns True if healthy."""
    try:
        if backend == "mlx":
            resp = requests.get(f"{MLX_BASE_URL}/v1/models", timeout=5)
            resp.raise_for_status()
        elif backend == "ollama":
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            resp.raise_for_status()
        elif backend == "groq":
            return bool(GROQ_API_KEY)
        return True
    except Exception:
        return False


def _try_restart_mlx() -> bool:
    """Attempt to restart the MLX server. Returns True if successful."""
    import subprocess

    print("Attempting MLX server restart...", file=sys.stderr)
    try:
        # Parse port from MLX_BASE_URL
        port = "8080"
        if ":" in MLX_BASE_URL.split("//")[-1]:
            port = MLX_BASE_URL.split(":")[-1].rstrip("/")

        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlx_lm.server",
                "--model",
                MLX_MODEL,
                "--port",
                port,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait for server to be ready
        for i in range(MLX_RESTART_WAIT):
            time.sleep(1)
            if check_backend_health("mlx"):
                print(f"MLX server restarted successfully after {i + 1}s", file=sys.stderr)
                return True
        print(f"MLX server failed to restart after {MLX_RESTART_WAIT}s", file=sys.stderr)
        return False
    except Exception as e:
        print(f"MLX restart error: {e}", file=sys.stderr)
        return False


def _recover_backend(backend: str) -> bool:
    """Try to recover a dead backend. Returns True if recovered."""
    if backend == "mlx" and MLX_AUTO_RESTART:
        return _try_restart_mlx()
    return False


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
        has_summary = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE summary IS NOT NULL AND summary != ''"))[
            0
        ][0]
        has_importance = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE importance IS NOT NULL"))[0][0]
        has_intent = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE intent IS NOT NULL AND intent != ''"))[0][0]
        try:
            has_embeddings = list(cursor.execute("SELECT COUNT(*) FROM chunk_vectors_rowids"))[0][0]
        except Exception:
            has_embeddings = 0
        projects = list(
            cursor.execute(
                "SELECT project, COUNT(*) FROM chunks WHERE project IS NOT NULL GROUP BY project ORDER BY COUNT(*) DESC LIMIT 20"
            )
        )

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
VALID_INTENTS = [
    "debugging",
    "designing",
    "configuring",
    "discussing",
    "deciding",
    "implementing",
    "reviewing",
]
VALID_EPISTEMIC = ["hypothesis", "substantiated", "validated"]
VALID_DEBT_IMPACT = ["introduction", "resolution", "none"]
VALID_SENTIMENTS = ["frustration", "confusion", "positive", "satisfaction", "neutral"]

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
  "external_deps": ["<libraries or external APIs used>"],
  "sentiment_label": "<one of: frustration, confusion, positive, satisfaction, neutral>",
  "sentiment_score": <float from -1.0 (max frustration) to 1.0 (max positive)>,
  "sentiment_signals": ["<words/phrases that indicate the sentiment>"]
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

SENTIMENT_LABEL: Detect the emotional tone of the human user in this chunk.
- frustration: Anger, annoyance, things not working ("damn", "wtf", "broken again")
- confusion: Not understanding, asking for clarification ("I don't understand", "wait what?")
- positive: Excitement, praise, amazement ("amazing", "incredible", "wow")
- satisfaction: Task done, gratitude, approval ("thanks", "perfect", "exactly what I needed")
- neutral: No strong emotion (most code/config chunks)
Note: Only user_message chunks have meaningful sentiment. For ai_code/assistant_text, use "neutral".

SENTIMENT_SCORE: -1.0 = maximum frustration, 0.0 = neutral, 1.0 = maximum positive.

SENTIMENT_SIGNALS: List the specific words or phrases that indicate the sentiment. Empty array if neutral.

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

    template = prompt_template or ENRICHMENT_PROMPT

    prompt = template.format(
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


def call_mlx(prompt: str, timeout: int = MLX_DEFAULT_TIMEOUT) -> Optional[str]:
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
    except requests.exceptions.ConnectionError as e:
        print(f"  MLX connection error (server dead?): {e}", file=sys.stderr)
        return None
    except requests.exceptions.Timeout as e:
        print(f"  MLX timeout ({timeout}s): {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  MLX error: {e}", file=sys.stderr)
        return None


def call_groq(prompt: str, timeout: int = 60) -> Optional[str]:
    """Call Groq cloud API via OpenAI-compatible endpoint. Logs usage to Supabase.

    PRIVACY: This sends content to Groq's cloud. Callers MUST sanitize content
    before calling this function. The _enrich_one() function enforces this by
    using build_external_prompt() with a Sanitizer when backend='groq'.

    Rate limiting: enforces GROQ_RATE_LIMIT_DELAY between consecutive calls
    to stay under free tier limits (~30 req/min).
    """
    global _groq_last_call
    if not GROQ_API_KEY:
        print("  Groq error: GROQ_API_KEY not set", file=sys.stderr)
        return None
    try:
        # Rate limit: serialize timestamp check/update across threads
        with _groq_rate_lock:
            now = time.monotonic()
            elapsed = now - _groq_last_call
            if _groq_last_call > 0 and elapsed < GROQ_RATE_LIMIT_DELAY:
                time.sleep(GROQ_RATE_LIMIT_DELAY - elapsed)
            _groq_last_call = time.monotonic()

        start_ms = int(time.time() * 1000)
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
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

        # Log to Supabase (best-effort)
        _log_glm_usage(prompt_tokens, completion_tokens, duration_ms, model=f"groq:{GROQ_MODEL}")

        # Extract response text
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return None
    except Exception as e:
        print(f"  Groq error: {e}", file=sys.stderr)
        return None


# Mid-run fallback state — tracks consecutive failures for automatic backend switching.
# When the primary backend crashes mid-run (e.g., MLX "Abort trap: 6"), the pipeline
# automatically retries failed chunks on the fallback backend instead of losing the entire batch.
_consecutive_failures = 0
_FALLBACK_THRESHOLD = 3  # Switch after 3 consecutive failures
_fallback_active = False
_fallback_available: Optional[bool] = None  # None = not checked yet


def _check_fallback_available(primary: str) -> bool:
    """Check if the fallback backend is reachable. Cached for the run."""
    global _fallback_available
    if _fallback_available is not None:
        return _fallback_available

    fallback = "ollama" if primary == "mlx" else "mlx"
    try:
        if fallback == "ollama":
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        else:
            resp = requests.get(f"{MLX_BASE_URL}/v1/models", timeout=3)
        resp.raise_for_status()
        _fallback_available = True
    except Exception:
        _fallback_available = False
    return _fallback_available


def call_llm(prompt: str, timeout: int = 240, backend: Optional[str] = None) -> Optional[str]:
    """Call local LLM using configured backend (ollama or mlx).

    Mid-run fallback: if the primary backend fails consecutively (e.g., MLX crash),
    automatically retries on the fallback backend. This prevents losing entire
    enrichment batches when MLX dies with "Abort trap: 6".

    Args:
        backend: Override backend for this call. If None, uses ENRICH_BACKEND.
    """
    global _consecutive_failures, _fallback_active, _fallback_available

    effective = backend or ENRICH_BACKEND

    # If fallback is already active, use it directly
    if _fallback_active:
        fallback = "ollama" if effective == "mlx" else "mlx"
        result = call_mlx(prompt, timeout=timeout) if fallback == "mlx" else call_glm(prompt, timeout=timeout)
        if result is not None:
            return result
        return None

    # Try primary backend
    if effective == "groq":
        # Groq is cloud-only, no local fallback mechanism
        return call_groq(prompt, timeout=timeout)
    elif effective == "mlx":
        result = call_mlx(prompt, timeout=timeout)
    else:
        result = call_glm(prompt, timeout=timeout)

    if result is not None:
        _consecutive_failures = 0
        return result

    # Primary failed — track consecutive failures
    _consecutive_failures += 1

    if _consecutive_failures >= _FALLBACK_THRESHOLD:
        if _check_fallback_available(effective):
            fallback = "ollama" if effective == "mlx" else "mlx"
            print(
                f"  FALLBACK: {_consecutive_failures} consecutive failures on {effective}, switching to {fallback}",
                file=sys.stderr,
            )
            _fallback_active = True
            # Retry this chunk on fallback
            if fallback == "mlx":
                return call_mlx(prompt, timeout=timeout)
            return call_glm(prompt, timeout=timeout)
        elif _consecutive_failures == _FALLBACK_THRESHOLD:
            # Only log once when threshold is first hit
            print(
                f"  WARNING: {_consecutive_failures} consecutive failures on {effective}, no fallback available",
                file=sys.stderr,
            )

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

    except Exception as e:
        logger.debug("Enrichment result validation failed: %s", e)
        return None


def _enrich_one(
    store_or_path,
    chunk: Dict[str, Any],
    with_context: bool = True,
    backend: Optional[str] = None,
) -> bool:
    """Enrich a single chunk. Returns True on success, False on failure.

    Retries up to MAX_RETRIES times with exponential backoff + jitter on LLM failure.
    This absorbs transient MLX crashes and connection blips without losing the chunk.

    Args:
        store_or_path: VectorStore instance (sequential) or Path (parallel, uses thread-local store).
        backend: Override backend for LLM calls.
    """
    # In parallel mode, each thread gets its own VectorStore connection.
    if isinstance(store_or_path, Path):
        store = _get_thread_store(store_or_path)
    else:
        store = store_or_path

    context_chunks = []
    if with_context and chunk.get("conversation_id") and chunk.get("position") is not None:
        ctx = store.get_context(chunk["id"], before=2, after=1)
        context_chunks = [c for c in ctx.get("context", []) if not c.get("is_target")]

    # External backends (groq) MUST use sanitized prompts — no raw PII to cloud
    effective_backend = backend or ENRICH_BACKEND
    if effective_backend == "groq":
        from .sanitize import Sanitizer

        sanitizer = Sanitizer.from_env()
        prompt, _sanitize_result = build_external_prompt(chunk, sanitizer, context_chunks)
    else:
        prompt = build_prompt(chunk, context_chunks)

    # Retry loop with exponential backoff + jitter
    response = None
    for attempt in range(1 + MAX_RETRIES):
        chunk_start = time.time()
        response = call_llm(prompt, backend=backend)
        chunk_duration = time.time() - chunk_start

        # Stall detection
        if chunk_duration > STALL_TIMEOUT:
            print(
                f"  STALL: chunk {chunk['id'][:12]} took {chunk_duration:.0f}s "
                f"(threshold: {STALL_TIMEOUT}s, chars: {chunk.get('char_count', '?')})",
                file=sys.stderr,
            )

        if response is not None:
            break

        # LLM returned None — retry with backoff
        if attempt < MAX_RETRIES:
            delay = min(RETRY_BASE_DELAY * (2**attempt), RETRY_MAX_DELAY)
            jitter = random.uniform(0, delay * 0.3)
            total_delay = delay + jitter
            print(
                f"  RETRY: chunk {chunk['id'][:12]} attempt {attempt + 2}/{1 + MAX_RETRIES} in {total_delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(total_delay)

    enrichment = parse_enrichment(response)
    if enrichment:
        # Only set sentiment from LLM if rule-based hasn't already set it
        # (rule-based is high-confidence for obvious cases)
        sentiment_label = enrichment.get("sentiment_label")
        sentiment_score = enrichment.get("sentiment_score")
        sentiment_signals = enrichment.get("sentiment_signals")
        if sentiment_label and sentiment_label not in VALID_SENTIMENTS:
            sentiment_label = None
        existing_sentiment = chunk.get("sentiment_label")
        if existing_sentiment:
            # Rule-based already classified — don't overwrite
            sentiment_label = None
            sentiment_score = None
            sentiment_signals = None

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
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            sentiment_signals=sentiment_signals,
        )

        # KG extraction: extract entities from enriched chunk into KG tables
        try:
            from .batch_extraction import DEFAULT_SEED_ENTITIES
            from .entity_extraction import extract_entities_from_tags
            from .kg_extraction import extract_kg_from_chunk

            # Seed + tag extraction (no API calls, always enabled)
            extract_kg_from_chunk(
                store=store,
                chunk_id=chunk["id"],
                seed_entities=DEFAULT_SEED_ENTITIES,
                use_llm=False,
                use_gliner=False,
            )

            # Tag-based extraction from enrichment tags
            if enrichment.get("tags"):
                import json as _json

                from .entity_extraction import ExtractionResult
                from .kg_extraction import process_extraction_result

                tags = enrichment["tags"]
                if isinstance(tags, str):
                    try:
                        tags = _json.loads(tags)
                    except (ValueError, TypeError):
                        tags = []
                if isinstance(tags, list):
                    tag_entities = extract_entities_from_tags(tags)
                    if tag_entities:
                        result = ExtractionResult(
                            entities=tag_entities,
                            relations=[],
                            chunk_id=chunk["id"],
                        )
                        process_extraction_result(store, result)
        except Exception:
            # KG extraction is non-critical — don't fail the enrichment
            logger.warning("KG extraction failed for chunk %s", chunk["id"], exc_info=True)

        return True
    return False


def enrich_batch(
    store: VectorStore,
    batch_size: int = 50,
    content_types: Optional[List[str]] = None,
    with_context: bool = True,
    parallel: int = 1,
    backend: Optional[str] = None,
    since_hours: Optional[int] = None,
) -> Dict[str, int]:
    """Process one batch of unenriched chunks. Returns counts.

    Circuit breaker: if CIRCUIT_BREAKER_THRESHOLD consecutive chunks fail,
    the batch is aborted early (backend is probably dead). This prevents
    wasting hours of compute on connection-refused errors.

    Args:
        parallel: Number of concurrent workers (1=sequential, >1=ThreadPoolExecutor).
                  MLX server supports concurrent requests. Ollama may not benefit.
        backend: Override backend for LLM calls (used when run_enrichment detects fallback).
        since_hours: Only enrich chunks from the last N hours (for on-demand enrichment).
    """
    types = content_types or HIGH_VALUE_TYPES
    chunks = store.get_unenriched_chunks(batch_size=batch_size, content_types=types, since_hours=since_hours)

    if not chunks:
        return {"processed": 0, "success": 0, "failed": 0}

    success = 0
    failed = 0
    consecutive_failures = 0
    circuit_broken = False

    batch_start = time.time()
    last_heartbeat = batch_start

    if parallel > 1:
        # Parallel: pass db_path so each thread gets its own VectorStore connection.
        # APSW connections are not safe for concurrent use from multiple threads.
        db_path = store.db_path
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_enrich_one, db_path, chunk, with_context, backend): chunk for chunk in chunks}
            for future in as_completed(futures):
                try:
                    if future.result():
                        success += 1
                        consecutive_failures = 0
                    else:
                        failed += 1
                        consecutive_failures += 1
                except Exception as e:
                    # DB lock errors are transient contention, not backend failures.
                    # Don't count them toward circuit breaker.
                    is_db_lock = "database is locked" in str(e) or "BusyError" in type(e).__name__
                    if is_db_lock:
                        print(f"  Worker DB lock (transient): {e}", file=sys.stderr)
                        failed += 1
                        # Don't increment consecutive_failures for DB locks
                    else:
                        print(f"  Worker error: {e}", file=sys.stderr)
                        failed += 1
                        consecutive_failures += 1

                # Circuit breaker: abort if backend is clearly dead
                if consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
                    circuit_broken = True
                    print(
                        f"  CIRCUIT BREAK: {consecutive_failures} consecutive failures, "
                        f"aborting batch ({success} ok, {failed} fail)",
                        file=sys.stderr,
                    )
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

                done = success + failed
                now = time.time()
                if done % HEARTBEAT_INTERVAL == 0 or now - last_heartbeat > 60:
                    rate = done / (now - batch_start) if now > batch_start else 0
                    print(f"  HEARTBEAT [{done}/{len(chunks)}] ok={success} fail={failed} rate={rate:.1f}/s")
                    last_heartbeat = now
    else:
        # Sequential: one chunk at a time (original behavior)
        for chunk in chunks:
            start = time.time()
            ok = _enrich_one(store, chunk, with_context, backend=backend)
            duration = time.time() - start

            if ok:
                success += 1
                consecutive_failures = 0
            else:
                failed += 1
                consecutive_failures += 1

            # Circuit breaker
            if consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
                circuit_broken = True
                print(
                    f"  CIRCUIT BREAK: {consecutive_failures} consecutive failures, "
                    f"aborting batch ({success} ok, {failed} fail)",
                    file=sys.stderr,
                )
                break

            done = success + failed
            now = time.time()
            if done % HEARTBEAT_INTERVAL == 0 or now - last_heartbeat > 60:
                rate = done / (now - batch_start) if now > batch_start else 0
                print(
                    f"  HEARTBEAT [{done}/{len(chunks)}] {duration:.1f}s | ok={success} fail={failed} rate={rate:.1f}/s"
                )
                last_heartbeat = now

    return {
        "processed": success + failed,
        "success": success,
        "failed": failed,
        "circuit_broken": circuit_broken,
    }


def mark_unenrichable(store: VectorStore) -> int:
    """Tag chunks that are too short for their source as 'skipped:too_short'.

    Uses source-aware thresholds: 15 chars for WhatsApp/Telegram, 50 for everything else.
    Returns the number of newly tagged chunks.
    """
    cursor = store.conn.cursor()
    # Tag chunks below their source-specific threshold
    # WhatsApp/Telegram: 15 chars. Everything else: 50 chars.
    cursor.execute("""
        UPDATE chunks SET enriched_at = 'skipped:too_short'
        WHERE enriched_at IS NULL
        AND (
            (source IN ('whatsapp', 'telegram') AND char_count < 15)
            OR (source NOT IN ('whatsapp', 'telegram') AND char_count < 50)
            OR (source IS NULL AND char_count < 50)
        )
    """)
    # apsw doesn't have rowcount, count via separate query
    tagged = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE enriched_at = 'skipped:too_short'"))[0][0]
    return tagged


def run_enrichment(
    db_path: Optional[Path] = None,
    batch_size: int = 50,
    max_chunks: int = 0,
    content_types: Optional[List[str]] = None,
    with_context: bool = True,
    parallel: int = 1,
    since_hours: Optional[int] = None,
) -> None:
    """Run the enrichment pipeline until done or max reached."""
    global _consecutive_failures, _fallback_active, _fallback_available
    _consecutive_failures = 0
    _fallback_active = False
    _fallback_available = None

    path = db_path or DEFAULT_DB_PATH
    store = VectorStore(path)

    try:
        # Check LLM backend is running — with auto-fallback
        active_backend = ENRICH_BACKEND
        if active_backend == "groq":
            if not GROQ_API_KEY:
                raise RuntimeError(
                    "GROQ_API_KEY not set. Get one from https://console.groq.com/keys\n"
                    "Or use: op read 'op://development/GROQ_API_KEY/password'"
                )
            print(f"Backend: Groq ({GROQ_MODEL}) [cloud — sanitization enforced]")
        elif active_backend == "mlx":
            try:
                resp = requests.get(f"{MLX_BASE_URL}/v1/models", timeout=5)
                resp.raise_for_status()
                print(f"Backend: MLX ({MLX_BASE_URL})")
            except Exception:
                # MLX not running — try falling back to Ollama
                print(f"MLX not available at {MLX_BASE_URL}, trying Ollama fallback...", file=sys.stderr)
                try:
                    ollama_base = OLLAMA_BASE_URL
                    resp = requests.get(f"{ollama_base}/api/tags", timeout=5)
                    resp.raise_for_status()
                    active_backend = "ollama"
                    print(f"Backend: Ollama ({MODEL}) [fallback from MLX]")
                except Exception:
                    raise RuntimeError(
                        f"Neither MLX ({MLX_BASE_URL}) nor Ollama is running.\n"
                        f"Start MLX: python3 -m mlx_lm.server --model {MLX_MODEL} --port 8080\n"
                        f"Start Ollama: ollama serve"
                    )
        else:
            try:
                ollama_base = OLLAMA_BASE_URL
                resp = requests.get(f"{ollama_base}/api/tags", timeout=5)
                resp.raise_for_status()
                print(f"Backend: Ollama ({MODEL})")
            except Exception:
                raise RuntimeError("Ollama is not running. Start it with: ollama serve")

        # Override module-level backend for this run if fallback was used
        _run_backend = active_backend

        # Auto-tag unenrichable chunks (too short for their source)
        mark_unenrichable(store)

        stats = store.get_enrichment_stats()
        print(f"Enrichment: {stats['enriched']:,}/{stats['enrichable']:,} ({stats['percent']}%)")
        print(f"Remaining: {stats['remaining']:,} | Skipped: {stats['skipped']:,} (too short)")
        if stats["by_intent"]:
            print(f"Intent distribution: {stats['by_intent']}")
        print(f"Batch size: {batch_size}, Max: {max_chunks or 'unlimited'}, Parallel: {parallel}")
        if since_hours:
            print(f"Recent only: last {since_hours} hours")
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
                backend=_run_backend,
                since_hours=since_hours,
            )

            if result["processed"] == 0:
                print("No more chunks to enrich.")
                break

            total_processed += result["processed"]
            total_success += result["success"]
            total_failed += result["failed"]

            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            print(
                f"Batch done: +{result['success']} ok, +{result['failed']} fail | "
                f"Total: {total_processed} ({rate:.1f}/s)"
            )

            # Circuit breaker tripped — try to recover before giving up
            if result.get("circuit_broken"):
                print(
                    f"\nCircuit breaker tripped on {_run_backend}. Checking health...",
                    file=sys.stderr,
                )
                if not check_backend_health(_run_backend):
                    # Backend is dead — try to recover
                    if _recover_backend(_run_backend):
                        # Reset fallback state and continue
                        _consecutive_failures = 0
                        _fallback_active = False
                        print("Backend recovered. Continuing enrichment.", file=sys.stderr)
                        continue
                    else:
                        print(
                            f"Backend {_run_backend} is dead and could not be recovered. Stopping enrichment.",
                            file=sys.stderr,
                        )
                        break
                else:
                    # Backend is alive but returning errors — stop to avoid wasting time
                    print(
                        "Backend is reachable but returning errors. Stopping.",
                        file=sys.stderr,
                    )
                    break

            # High fail ratio detection: if >80% of batch failed, check health
            # before burning through more chunks on a dying backend
            if result["processed"] > 0:
                fail_ratio = result["failed"] / result["processed"]
                if fail_ratio >= BATCH_FAIL_RATIO_THRESHOLD:
                    print(
                        f"\nHigh fail ratio ({fail_ratio:.0%}) — checking backend health...",
                        file=sys.stderr,
                    )
                    if not check_backend_health(_run_backend):
                        if _recover_backend(_run_backend):
                            _consecutive_failures = 0
                            _fallback_active = False
                            print("Backend recovered after high-fail batch.", file=sys.stderr)
                        else:
                            print(
                                f"Backend {_run_backend} is dead. Stopping enrichment.",
                                file=sys.stderr,
                            )
                            break
                    else:
                        # Backend is alive but producing errors — pause briefly
                        print(
                            f"Backend alive but struggling. Pausing {HEALTH_CHECK_PAUSE}s...",
                            file=sys.stderr,
                        )
                        time.sleep(HEALTH_CHECK_PAUSE)

            # Sync stats to Supabase every 5 batches
            if total_processed % (batch_size * 5) < batch_size:
                _sync_stats_to_supabase(store)

            if max_chunks > 0 and total_processed >= max_chunks:
                print(f"Reached max ({max_chunks}).")
                break

        elapsed = time.time() - start_time
        print("\n--- Enrichment Complete ---")
        print(f"Processed: {total_processed} ({total_success} ok, {total_failed} fail)")
        print(f"Time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

        final_stats = store.get_enrichment_stats()
        print(f"Progress: {final_stats['enriched']:,}/{final_stats['enrichable']:,} ({final_stats['percent']}%)")
        print(f"Skipped: {final_stats['skipped']:,} | Remaining: {final_stats['remaining']:,}")

        # Final sync to Supabase
        _sync_stats_to_supabase(store)
    finally:
        store.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enrich BrainLayer chunks with LLM metadata")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max", type=int, default=0, help="Max chunks to process (0=unlimited)")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Concurrent workers (1=sequential, 3=recommended for MLX)",
    )
    parser.add_argument("--no-context", action="store_true", help="Skip surrounding context")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ollama", "mlx", "groq"],
        default=None,
        help="LLM backend (default: auto-detect). groq sends to cloud with sanitization.",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=None,
        metavar="HOURS",
        help="Only enrich chunks from the last N hours (on-demand mode)",
    )
    parser.add_argument("--stats", action="store_true", help="Show enrichment stats and exit")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    args = parser.parse_args()

    db = Path(args.db) if args.db else None

    if args.stats:
        store = VectorStore(db or DEFAULT_DB_PATH)
        stats = store.get_enrichment_stats()
        print(f"Total chunks:  {stats['total_chunks']:,}")
        print(f"Skipped:       {stats['skipped']:,} (too short)")
        print(f"Enrichable:    {stats['enrichable']:,}")
        print(f"Enriched:      {stats['enriched']:,} ({stats['percent']}%)")
        print(f"Remaining:     {stats['remaining']:,}")
        if stats["by_intent"]:
            print(f"Intent: {stats['by_intent']}")
        store.close()
    else:
        # Set backend via env var if specified on CLI (affects module-level detection)
        if args.backend:
            os.environ["BRAINLAYER_ENRICH_BACKEND"] = args.backend
            import brainlayer.pipeline.enrichment as _self

            _self.ENRICH_BACKEND = args.backend

        run_enrichment(
            db_path=db,
            batch_size=args.batch_size,
            max_chunks=args.max,
            with_context=not args.no_context,
            parallel=args.parallel,
            since_hours=args.recent,
        )
