"""Unified enrichment controller for realtime, batch, and local modes.

Replaces scattered scripts (enrichment-window.sh, enrichment-lazy.sh, enrich.sh)
with a single controller. Three backends:
  - realtime: Gemini 2.5 Flash-Lite, single chunk, <600ms target
  - batch: Gemini Batch API, backlog processing, thinkingBudget=0
  - local: MLX/Ollama backend, offline/privacy mode
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from .pipeline.enrichment import (
    build_external_prompt,
    build_prompt,
    parse_enrichment,
)
from .pipeline.rate_limiter import TokenBucket
from .pipeline.sanitize import Sanitizer
from .pipeline.write_queue import WriteQueue

logger = logging.getLogger(__name__)

GEMINI_REALTIME_MODEL = os.environ.get("BRAINLAYER_GEMINI_REALTIME_MODEL", "gemini-2.5-flash-lite")

# Auto-enrichment on brain_store: set to "0" or "false" to disable
AUTO_ENRICH_ENABLED = os.environ.get("BRAINLAYER_AUTO_ENRICH", "1").lower() not in ("0", "false", "no")

# Per-backend rate limits (requests per second). Override via env vars.
RATE_LIMITS = {
    "realtime": float(
        os.environ.get("BRAINLAYER_ENRICH_RATE", "5.0")
    ),  # 300 RPM default (AI Pro verified 500+ RPM Apr 2026)
    "local": float(os.environ.get("BRAINLAYER_LOCAL_RATE", "0")),  # no limit
    "batch": float(os.environ.get("BRAINLAYER_BATCH_RATE", "0")),  # no limit (async)
}
ENRICH_CONCURRENCY = int(os.environ.get("BRAINLAYER_ENRICH_CONCURRENCY", "10"))
WRITE_QUEUE_MAXSIZE = int(os.environ.get("BRAINLAYER_WRITE_QUEUE_MAXSIZE", "1000"))
RATE_LIMIT_BURST = int(os.environ.get("BRAINLAYER_ENRICH_BURST", "10"))

_WRITE_QUEUE_REGISTRY: dict[str, WriteQueue] = {}
_WRITE_QUEUE_LOCK = threading.Lock()
_ENRICHMENT_COLUMN_READY: set[str] = set()
_ENRICHMENT_COLUMN_LOCK = threading.Lock()
_RATE_LIMITER_REGISTRY: dict[tuple[str, float, int], TokenBucket] = {}
_RATE_LIMITER_LOCK = threading.Lock()
_STORE_OPERATION_COUNTS: dict[str, int] = {}
_STORE_OPERATION_LOCK = threading.Lock()
_STORE_OPERATION_CONDITION = threading.Condition(_STORE_OPERATION_LOCK)
_STORE_CLOSING: set[str] = set()

_META_RESEARCH_PATTERNS = [
    re.compile(r"brain_search\s*\(", re.IGNORECASE),
    re.compile(r"brain_search query=", re.IGNORECASE),
    re.compile(r"search results? for ['\"]", re.IGNORECASE),
    re.compile(r"Query \d+ for ['\"].*['\"] (degraded|scored|returned)", re.IGNORECASE),
    re.compile(r"(eval|baseline|pilot) score:?\s*\d+(?:[./]\d+)*/5", re.IGNORECASE),
    re.compile(r"Grade:?\s*\d/5", re.IGNORECASE),
    re.compile(r"\[BrainLayer (auto|deep)\] Memories matching", re.IGNORECASE),
    re.compile(r"['\"]?additionalContext['\"]?\s*:", re.IGNORECASE),
]


@dataclass
class EnrichmentResult:
    mode: str
    attempted: int
    enriched: int
    skipped: int
    failed: int
    errors: list[str] = field(default_factory=list)


def _load_cloud_backfill_module():
    """Load scripts/cloud_backfill.py without turning scripts into a package."""
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "cloud_backfill.py"
    spec = importlib.util.spec_from_file_location("brainlayer_cloud_backfill", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load cloud_backfill module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_checkpoint_table(store) -> None:
    return _load_cloud_backfill_module().ensure_checkpoint_table(store)


def get_pending_jobs(store):
    return _load_cloud_backfill_module().get_pending_jobs(store)


def get_unsubmitted_export_files(*args, **kwargs):
    return _load_cloud_backfill_module().get_unsubmitted_export_files(*args, **kwargs)


# ── Gemini client ──────────────────────────────────────────────────────────────


def _store_queue_key(store) -> str:
    db_path = getattr(store, "db_path", None)
    return str(db_path) if db_path is not None else f"store:{id(store)}"


def _get_store_write_queue(store) -> WriteQueue:
    key = _store_queue_key(store)
    with _WRITE_QUEUE_LOCK:
        write_queue = _WRITE_QUEUE_REGISTRY.get(key)
        if write_queue is None:
            write_queue = WriteQueue(maxsize=WRITE_QUEUE_MAXSIZE)
            write_queue.start()
            _WRITE_QUEUE_REGISTRY[key] = write_queue
        return write_queue


def _begin_store_operation(store) -> str:
    key = _store_queue_key(store)
    with _STORE_OPERATION_CONDITION:
        while key in _STORE_CLOSING:
            _STORE_OPERATION_CONDITION.wait()
        _STORE_OPERATION_COUNTS[key] = _STORE_OPERATION_COUNTS.get(key, 0) + 1
    return key


def _end_store_operation(store) -> None:
    key = _store_queue_key(store)
    cleanup = False
    with _STORE_OPERATION_CONDITION:
        remaining = _STORE_OPERATION_COUNTS.get(key, 0) - 1
        if remaining <= 0:
            _STORE_OPERATION_COUNTS.pop(key, None)
            _STORE_CLOSING.add(key)
            cleanup = True
        else:
            _STORE_OPERATION_COUNTS[key] = remaining

    if not cleanup:
        return

    try:
        with _WRITE_QUEUE_LOCK:
            write_queue = _WRITE_QUEUE_REGISTRY.pop(key, None)
        if write_queue is not None:
            write_queue.stop(timeout=1.0)

        with _ENRICHMENT_COLUMN_LOCK:
            _ENRICHMENT_COLUMN_READY.discard(key)
    finally:
        with _STORE_OPERATION_CONDITION:
            _STORE_CLOSING.discard(key)
            _STORE_OPERATION_CONDITION.notify_all()


def _submit_write(store, name: str, callback) -> Any:
    return _get_store_write_queue(store).submit(name, callback).result()


def _ensure_enrichment_columns(store) -> None:
    key = _store_queue_key(store)
    with _ENRICHMENT_COLUMN_LOCK:
        if key in _ENRICHMENT_COLUMN_READY:
            return

    def _ensure() -> None:
        _ensure_content_hash_column(store)
        _ensure_raw_entities_json_column(store)

    _submit_write(store, "ensure-enrichment-columns", _ensure)

    with _ENRICHMENT_COLUMN_LOCK:
        _ENRICHMENT_COLUMN_READY.add(key)


def _get_store_rate_limiter(
    store, rate_per_second: float | None = None, burst: int | None = None
) -> TokenBucket | None:
    if rate_per_second is None:
        rate_per_second = RATE_LIMITS["realtime"]
    if rate_per_second <= 0:
        return None

    burst = RATE_LIMIT_BURST if burst is None else burst
    key = (_store_queue_key(store), rate_per_second, burst)
    with _RATE_LIMITER_LOCK:
        limiter = _RATE_LIMITER_REGISTRY.get(key)
        if limiter is None:
            limiter = TokenBucket(rate_per_sec=rate_per_second, burst=burst)
            _RATE_LIMITER_REGISTRY[key] = limiter
        return limiter


def _get_chunk_readonly(store, chunk_id: str) -> dict[str, Any] | None:
    if not hasattr(store, "_read_cursor"):
        return store.get_chunk(chunk_id)

    row = (
        store._read_cursor()
        .execute(
            """SELECT id, content, metadata, source_file, project, content_type,
                      value_type, tags, importance, created_at, summary,
                      superseded_by, aggregated_into, archived_at
               FROM chunks WHERE id = ?""",
            (chunk_id,),
        )
        .fetchone()
    )
    if not row:
        return None
    return {
        "id": row[0],
        "content": row[1],
        "metadata": row[2],
        "source_file": row[3],
        "project": row[4],
        "content_type": row[5],
        "value_type": row[6],
        "tags": row[7],
        "importance": row[8],
        "created_at": row[9],
        "summary": row[10],
        "superseded_by": row[11],
        "aggregated_into": row[12],
        "archived_at": row[13],
    }


def _get_gemini_client():
    """Create Gemini client. Uses regional endpoint when GOOGLE_CLOUD_REGION is set."""
    try:
        from google import genai
    except ImportError:
        raise RuntimeError("google-genai package not installed. Install with: pip install google-genai")

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY not set")

    # Regional endpoint reduces latency from 11-12s (global) to 0.7s.
    # Set GOOGLE_CLOUD_REGION=us-central1 (or europe-west1, etc.) to enable.
    region = os.environ.get("GOOGLE_CLOUD_REGION")
    http_options: dict[str, Any] = {"retry_options": {"attempts": 1}}
    if region:
        http_options.update({"api_version": "v1beta", "url": f"https://{region}-aiplatform.googleapis.com"})
    return genai.Client(api_key=api_key, http_options=http_options)


GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "importance": {"type": "number"},
        "intent": {"type": "string"},
        "primary_symbols": {"type": "array", "items": {"type": "string"}},
        "resolved_query": {"type": "string"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "resolved_queries": {"type": "array", "items": {"type": "string"}},
        "epistemic_level": {"type": "string"},
        "version_scope": {"type": "string"},
        "debt_impact": {"type": "string"},
        "external_deps": {"type": "array", "items": {"type": "string"}},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["person", "company", "project", "technology", "tool", "concept"],
                    },
                    "relation": {"type": "string"},
                },
                "required": ["name", "type"],
            },
        },
        "sentiment_label": {"type": "string"},
        "sentiment_score": {"type": "number"},
        "sentiment_signals": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "tags", "importance", "intent", "entities"],
}


def _build_gemini_config() -> dict[str, Any]:
    return {
        "response_mime_type": "application/json",
        "response_schema": GEMINI_RESPONSE_SCHEMA,
        "thinking_config": {"thinking_budget": 0},
        "http_options": _build_gemini_http_options(),
    }


def _get_gemini_service_tier() -> str:
    return os.environ.get("BRAINLAYER_GEMINI_SERVICE_TIER", "flex")


def _build_gemini_http_options(timeout_ms: int | None = None) -> dict[str, Any]:
    http_options: dict[str, Any] = {
        "extra_body": {"serviceTier": _get_gemini_service_tier()},
    }
    if timeout_ms is not None:
        http_options["timeout"] = timeout_ms
    return http_options


# ── Entity extraction via Gemini ───────────────────────────────────────────────

GEMINI_EXTRACTION_MODEL = os.environ.get("BRAINLAYER_GEMINI_EXTRACTION_MODEL", "gemini-2.5-flash-lite")


def call_gemini_for_extraction(prompt: str) -> Optional[str]:
    """Call Gemini for entity/relation extraction. Returns raw text response.

    Rate-limited by BRAINLAYER_ENRICH_RATE (default 0.2 = 12 RPM).
    Timeout: 30 seconds per call.
    """
    try:
        client = _get_gemini_client()
    except RuntimeError:
        logger.debug("Gemini not available for extraction")
        return None

    try:
        response = client.models.generate_content(
            model=GEMINI_EXTRACTION_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "thinking_config": {"thinking_budget": 0},
                "http_options": _build_gemini_http_options(timeout_ms=30_000),
            },
        )
        return response.text if response and response.text else None
    except Exception:
        logger.warning("Gemini extraction call failed", exc_info=True)
        return None


# ── Content-hash dedup ─────────────────────────────────────────────────────────


def _content_hash(content: str) -> str:
    """SHA256 hash of content for dedup. Strips whitespace for normalization."""
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def is_meta_research(content: str) -> bool:
    if not content:
        return False
    return any(pattern.search(content) for pattern in _META_RESEARCH_PATTERNS)


def _is_duplicate_content(store, content: str) -> bool:
    """Check if content with the same hash already exists and is enriched.

    Returns True if a chunk with identical content hash exists and has been enriched,
    meaning re-enriching would be a no-op.
    """
    content_h = _content_hash(content)
    try:
        cursor = store._read_cursor()
        row = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE content_hash = ? AND enriched_at IS NOT NULL AND summary IS NOT NULL",
            (content_h,),
        ).fetchone()
        return row[0] > 0 if row else False
    except Exception:
        # content_hash column may not exist yet — fall back to no dedup
        return False


def _ensure_content_hash_column(store) -> bool:
    """Ensure the content_hash column exists on chunks table. Returns True if it exists."""
    try:
        cursor = store.conn.cursor()
        cursor.execute("SELECT content_hash FROM chunks LIMIT 0")
        return True
    except Exception:
        try:
            cursor.execute("ALTER TABLE chunks ADD COLUMN content_hash TEXT")
            return True
        except Exception:
            return False


def _ensure_raw_entities_json_column(store) -> bool:
    """Ensure the raw_entities_json staging column exists on chunks."""
    try:
        store.conn.cursor().execute("SELECT raw_entities_json FROM chunks LIMIT 0")
        return True
    except Exception:
        try:
            store.conn.cursor().execute("ALTER TABLE chunks ADD COLUMN raw_entities_json TEXT")
            return True
        except Exception:
            return False


def _normalize_chunk_tags(tags: Any) -> list[str]:
    if isinstance(tags, str):
        try:
            decoded = json.loads(tags)
        except json.JSONDecodeError:
            decoded = [tags]
        else:
            tags = decoded
    if isinstance(tags, list):
        return [str(tag) for tag in tags if str(tag).strip()]
    return []


def _mark_meta_research(store, chunk: dict[str, Any]) -> None:
    cursor = store.conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    tags = _normalize_chunk_tags(chunk.get("tags"))
    if "meta-research" not in tags:
        tags.append("meta-research")
    cursor.execute(
        "UPDATE chunks SET tags = ?, summary = NULL, enriched_at = ? WHERE id = ?",
        (json.dumps(tags), now, chunk["id"]),
    )
    content = chunk.get("content", "")
    if content:
        try:
            cursor.execute("UPDATE chunks SET content_hash = ? WHERE id = ?", (_content_hash(content), chunk["id"]))
        except Exception:
            pass


def _backfill_content_hashes(store, limit: int = 1000) -> int:
    """Backfill content_hash for chunks that don't have one yet. Returns count updated."""
    try:
        cursor = store.conn.cursor()
        rows = list(
            cursor.execute(
                "SELECT id, content FROM chunks WHERE content_hash IS NULL LIMIT ?",
                (limit,),
            )
        )
        count = 0
        for chunk_id, content in rows:
            if content:
                h = _content_hash(content)
                cursor.execute("UPDATE chunks SET content_hash = ? WHERE id = ?", (h, chunk_id))
                count += 1
        return count
    except Exception:
        return 0


# ── Retry / apply helpers ──────────────────────────────────────────────────────


def _retry_with_backoff(
    fn,
    max_retries: int = 12,
    base_delay: float = 1.0,
    max_delay: float = 120.0,
    retryable_errors: tuple = (Exception,),
):
    """Retry transient failures with exponential backoff and capped jitter."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retryable_errors as exc:
            if attempt >= max_retries:
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            sleep_for = min(delay + jitter, max_delay)
            logger.warning(
                "Retrying enrichment call after error %s (attempt %d/%d) in %.2fs",
                exc,
                attempt + 2,
                max_retries + 1,
                sleep_for,
            )
            time.sleep(sleep_for)


def _generate_content_with_rate_limit(
    client, model: str, prompt: str, config: dict[str, Any], limiter: TokenBucket | None
):
    if limiter is not None:
        limiter.acquire()
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )


def _apply_enrichment(store, chunk: dict[str, Any], enrichment: dict[str, Any]) -> None:
    resolved_queries = enrichment.get("resolved_queries")
    legacy_resolved_query = enrichment.get("resolved_query")
    if not legacy_resolved_query and isinstance(resolved_queries, list) and resolved_queries:
        legacy_resolved_query = resolved_queries[0]

    store.update_enrichment(
        chunk_id=chunk["id"],
        summary=enrichment.get("summary"),
        tags=enrichment.get("tags"),
        importance=enrichment.get("importance"),
        intent=enrichment.get("intent"),
        primary_symbols=enrichment.get("primary_symbols"),
        resolved_query=legacy_resolved_query,
        epistemic_level=enrichment.get("epistemic_level"),
        version_scope=enrichment.get("version_scope"),
        debt_impact=enrichment.get("debt_impact"),
        external_deps=enrichment.get("external_deps"),
        key_facts=enrichment.get("key_facts"),
        resolved_queries=resolved_queries,
        sentiment_label=enrichment.get("sentiment_label"),
        sentiment_score=enrichment.get("sentiment_score"),
        sentiment_signals=enrichment.get("sentiment_signals"),
    )
    entities = enrichment.get("entities", [])
    # AIDEV-NOTE: raw entities persisted to chunks.raw_entities_json staging column;
    # R84b canonicalization pipeline will consume and populate kg_entities downstream.
    if _ensure_raw_entities_json_column(store):
        store.conn.cursor().execute(
            "UPDATE chunks SET raw_entities_json = ? WHERE id = ?",
            (json.dumps(entities), chunk["id"]),
        )
    # Set content_hash after enrichment so dedup works next time
    content = chunk.get("content", "")
    if content:
        try:
            h = _content_hash(content)
            store.conn.cursor().execute("UPDATE chunks SET content_hash = ? WHERE id = ?", (h, chunk["id"]))
        except Exception:
            pass  # Non-critical — dedup still works on next index


def _enrich_single_chunk(
    client,
    model: str,
    config: dict[str, Any],
    chunk: dict[str, Any],
    sanitizer,
    *,
    is_duplicate,
    rate_limiter: TokenBucket | None,
    max_retries: int,
) -> tuple[dict[str, Any], str, Any]:
    """Run dedup, prompt build, and API call for one chunk.

    Returns `(chunk, status, data)` where status is one of:
      - `"skip"`: content hash already enriched
      - `"meta"`: chunk tagged as meta-research without a Gemini call
      - `"error"`: data is an error string
      - `"ok"`: data is the parsed enrichment dict
    """
    if is_meta_research(chunk.get("content", "")):
        return (chunk, "meta", None)
    if is_duplicate(chunk.get("content", "")):
        return (chunk, "skip", None)

    try:
        prompt, _sanitize_result = build_external_prompt(chunk, sanitizer)
    except Exception as exc:  # noqa: BLE001
        return (chunk, "error", f"prompt_build_error: {exc}")

    try:
        response = _retry_with_backoff(
            lambda: _generate_content_with_rate_limit(client, model, prompt, config, rate_limiter),
            max_retries=max_retries,
        )
        raw_response = getattr(response, "text", response)
        enrichment = parse_enrichment(raw_response)
        if not enrichment:
            return (chunk, "error", "invalid_enrichment")
        return (chunk, "ok", enrichment)
    except Exception as exc:  # noqa: BLE001
        return (chunk, "error", str(exc))


def enrich_single(store, chunk_id: str, max_retries: int = 2) -> dict[str, Any] | None:
    """Enrich a single chunk by ID via Gemini 2.5 Flash Lite.

    Designed for post-store auto-enrichment (R47 two-pass pattern):
    Pass 1 = sync embedding (immediate, searchable)
    Pass 2 = this function (async Gemini tagging, ~600ms target)

    Bypasses get_enrichment_candidates — works on any chunk regardless
    of enriched_at status. Overwrites stub enrichment with Gemini output.

    Returns the parsed enrichment dict on success, None on failure.
    """
    if not AUTO_ENRICH_ENABLED:
        return None

    _begin_store_operation(store)
    try:
        _ensure_enrichment_columns(store)

        chunk = _get_chunk_readonly(store, chunk_id)
        if not chunk:
            logger.warning("enrich_single: chunk not found: %s", chunk_id)
            return None

        if is_meta_research(chunk.get("content", "")):
            _submit_write(store, f"mark-meta:{chunk_id}", lambda: _mark_meta_research(store, chunk))
            logger.info("enrich_single: tagged %s as meta-research without Gemini", chunk_id)
            return None

        try:
            client = _get_gemini_client()
        except RuntimeError:
            logger.debug("enrich_single: no Gemini API key, skipping enrichment for %s", chunk_id)
            return None

        sanitizer = Sanitizer.from_env()
        try:
            prompt, _sanitize_result = build_external_prompt(chunk, sanitizer)
        except Exception as exc:
            logger.warning("enrich_single: prompt build failed for %s: %s", chunk_id, exc)
            return None

        config = _build_gemini_config()
        rate_limiter = _get_store_rate_limiter(store)

        def _call():
            response = _generate_content_with_rate_limit(client, GEMINI_REALTIME_MODEL, prompt, config, rate_limiter)
            return getattr(response, "text", None)

        try:
            raw_response = _retry_with_backoff(
                _call,
                max_retries=max_retries,
                base_delay=0.3,
                max_delay=5.0,
            )
        except Exception as exc:
            logger.warning("enrich_single: Gemini call failed for %s: %s", chunk_id, exc)
            return None

        enrichment = parse_enrichment(raw_response)
        if not enrichment:
            logger.warning("enrich_single: invalid enrichment response for %s", chunk_id)
            return None

        try:
            _submit_write(store, f"apply-enrichment:{chunk_id}", lambda: _apply_enrichment(store, chunk, enrichment))
        except Exception as exc:
            logger.warning("enrich_single: apply failed for %s: %s", chunk_id, exc)
            return None

        logger.info("enrich_single: enriched %s with %d tags", chunk_id, len(enrichment.get("tags", [])))
        return enrichment
    finally:
        _end_store_operation(store)


def _call_local_backend(prompt: str, backend: str = "mlx") -> str | None:
    from .pipeline.enrichment import call_llm

    return call_llm(prompt, backend=backend)


# ── Axiom telemetry ────────────────────────────────────────────────────────────

_DATASET_ENRICHMENT = "brainlayer-enrichment"


def _emit_enrichment_event(event: dict[str, Any]) -> bool:
    """Emit a single enrichment telemetry event to Axiom."""
    try:
        from .telemetry import emit

        return emit(_DATASET_ENRICHMENT, event)
    except Exception:
        return False


def _emit_enrichment_start(mode: str, limit: int) -> bool:
    if mode == "realtime":
        try:
            os.write(
                2,
                b"ENRICHMENT_RUNTIME_LOADED mode=realtime prompt=r81 truncation=8000 split=4800/3200 rubrics=epistemic_level,debt_impact,sentiment_label\n",
            )
        except OSError as exc:
            logger.debug("ENRICHMENT_RUNTIME_LOADED emit failed: %s", exc)
    return _emit_enrichment_event(
        {
            "_type": "start",
            "mode": mode,
            "limit": limit,
            "pid": os.getpid(),
            "hostname": os.uname().nodename,
        }
    )


def _emit_enrichment_complete(result: EnrichmentResult, duration_ms: float) -> bool:
    return _emit_enrichment_event(
        {
            "_type": "complete",
            "mode": result.mode,
            "attempted": result.attempted,
            "enriched": result.enriched,
            "skipped": result.skipped,
            "failed": result.failed,
            "duration_ms": round(duration_ms, 1),
            "error_count": len(result.errors),
        }
    )


def _emit_enrichment_error(mode: str, chunk_id: str, error: str) -> bool:
    return _emit_enrichment_event(
        {
            "_type": "error",
            "mode": mode,
            "chunk_id": chunk_id,
            "error": error[:300],
        }
    )


# ── Enrichment modes ───────────────────────────────────────────────────────────


def enrich_realtime(
    store,
    limit: int = 500,
    since_hours: int = 8760,
    rate_per_second: float | None = None,
    max_retries: int = 12,
    chunk_ids: list[str] | None = None,
) -> EnrichmentResult:
    """Enrich recent chunks via Gemini 2.5 Flash-Lite API."""
    if rate_per_second is None:
        rate_per_second = RATE_LIMITS["realtime"]

    _begin_store_operation(store)
    try:
        start_time = time.monotonic()
        _emit_enrichment_start("realtime", limit)

        candidates = store.get_enrichment_candidates(limit=limit, since_hours=since_hours, chunk_ids=chunk_ids)
        result = EnrichmentResult(mode="realtime", attempted=len(candidates), enriched=0, skipped=0, failed=0)
        if not candidates:
            _emit_enrichment_complete(result, 0)
            return result

        _ensure_enrichment_columns(store)

        client = _get_gemini_client()
        sanitizer = Sanitizer.from_env()
        config = _build_gemini_config()
        rate_limiter = _get_store_rate_limiter(store, rate_per_second=rate_per_second)

        def is_duplicate(content: str) -> bool:
            return _is_duplicate_content(store, content)

        with ThreadPoolExecutor(max_workers=ENRICH_CONCURRENCY) as executor:
            futures = []
            for chunk in candidates:
                futures.append(
                    executor.submit(
                        _enrich_single_chunk,
                        client,
                        GEMINI_REALTIME_MODEL,
                        config,
                        chunk,
                        sanitizer,
                        is_duplicate=is_duplicate,
                        rate_limiter=rate_limiter,
                        max_retries=max_retries,
                    )
                )

            for future in as_completed(futures):
                chunk, status, data = future.result()
                if status == "skip":
                    result.skipped += 1
                    continue
                if status == "meta":
                    _submit_write(
                        store, f"mark-meta:{chunk['id']}", lambda chunk=chunk: _mark_meta_research(store, chunk)
                    )
                    result.skipped += 1
                    continue
                if status == "error":
                    result.failed += 1
                    result.errors.append(f"{chunk['id']}: {data}")
                    _emit_enrichment_error("realtime", chunk["id"], str(data))
                    continue

                _submit_write(
                    store,
                    f"apply-enrichment:{chunk['id']}",
                    lambda chunk=chunk, data=data: _apply_enrichment(store, chunk, data),
                )
                result.enriched += 1

        duration_ms = (time.monotonic() - start_time) * 1000
        _emit_enrichment_complete(result, duration_ms)
        return result
    finally:
        _end_store_operation(store)


def enrich_batch(
    store,
    phase: str = "run",
    limit: int = 5000,
    max_retries: int = 3,
) -> EnrichmentResult:
    """Process backlog via realtime Gemini enrichment (one chunk at a time).

    Falls back to per-chunk realtime enrichment since the Gemini Batch API
    submit/poll/import workflow is not yet wired. This ensures unenriched
    chunks actually get processed instead of returning enriched=0.
    """
    _begin_store_operation(store)
    try:
        start_time = time.monotonic()
        _emit_enrichment_start("batch", limit)

        candidates = store.get_enrichment_candidates(limit=limit, chunk_ids=None)
        result = EnrichmentResult(mode="batch", attempted=len(candidates), enriched=0, skipped=0, failed=0, errors=[])

        if not candidates:
            duration_ms = (time.monotonic() - start_time) * 1000
            _emit_enrichment_complete(result, duration_ms)
            return result

        _ensure_enrichment_columns(store)

        try:
            client = _get_gemini_client()
        except RuntimeError as exc:
            result.errors.append(f"No Gemini client: {exc}")
            duration_ms = (time.monotonic() - start_time) * 1000
            _emit_enrichment_complete(result, duration_ms)
            return result

        sanitizer = Sanitizer.from_env()
        config = _build_gemini_config()
        rate_limiter = _get_store_rate_limiter(store, rate_per_second=RATE_LIMITS.get("realtime", 0.2))

        for chunk in candidates:
            if is_meta_research(chunk.get("content", "")):
                _submit_write(store, f"mark-meta:{chunk['id']}", lambda chunk=chunk: _mark_meta_research(store, chunk))
                result.skipped += 1
                continue
            if _is_duplicate_content(store, chunk.get("content", "")):
                result.skipped += 1
                continue

            try:
                prompt, _sanitize_result = build_external_prompt(chunk, sanitizer)
            except Exception as exc:
                result.failed += 1
                result.errors.append(f"{chunk['id']}: prompt_build_error: {exc}")
                continue

            try:
                response = _retry_with_backoff(
                    lambda: _generate_content_with_rate_limit(
                        client,
                        GEMINI_REALTIME_MODEL,
                        prompt,
                        config,
                        rate_limiter,
                    ),
                    max_retries=max_retries,
                )
                enrichment = parse_enrichment(response.text)
                if not enrichment:
                    result.failed += 1
                    result.errors.append(f"{chunk['id']}: invalid_enrichment")
                    _emit_enrichment_error("batch", chunk["id"], "invalid_enrichment")
                    continue
                _submit_write(
                    store,
                    f"apply-enrichment:{chunk['id']}",
                    lambda chunk=chunk, enrichment=enrichment: _apply_enrichment(store, chunk, enrichment),
                )
                result.enriched += 1
            except Exception as exc:
                result.failed += 1
                result.errors.append(f"{chunk['id']}: {exc}")
                _emit_enrichment_error("batch", chunk["id"], str(exc))

        duration_ms = (time.monotonic() - start_time) * 1000
        _emit_enrichment_complete(result, duration_ms)
        return result
    finally:
        _end_store_operation(store)


def enrich_local(
    store,
    limit: int = 100,
    parallel: int = 2,  # noqa: ARG001
    backend: str = "mlx",
) -> EnrichmentResult:
    """Enrich via local MLX/Ollama backend."""
    _begin_store_operation(store)
    try:
        start_time = time.monotonic()
        _emit_enrichment_start("local", limit)

        candidates = store.get_enrichment_candidates(limit=limit, chunk_ids=None)
        result = EnrichmentResult(mode="local", attempted=len(candidates), enriched=0, skipped=0, failed=0)

        _ensure_enrichment_columns(store)

        for chunk in candidates:
            # Content-hash dedup
            if _is_duplicate_content(store, chunk.get("content", "")):
                result.skipped += 1
                continue

            try:
                prompt = build_prompt(chunk)
                raw_response = _retry_with_backoff(lambda: _call_local_backend(prompt, backend=backend), max_retries=2)
                enrichment = parse_enrichment(raw_response)
                if not enrichment:
                    result.failed += 1
                    result.errors.append(f"{chunk['id']}: invalid_enrichment")
                    _emit_enrichment_error("local", chunk["id"], "invalid_enrichment")
                    continue
                _submit_write(
                    store,
                    f"apply-enrichment:{chunk['id']}",
                    lambda chunk=chunk, enrichment=enrichment: _apply_enrichment(store, chunk, enrichment),
                )
                result.enriched += 1
            except Exception as exc:  # noqa: BLE001
                result.failed += 1
                result.errors.append(f"{chunk['id']}: {exc}")
                _emit_enrichment_error("local", chunk["id"], str(exc))

        duration_ms = (time.monotonic() - start_time) * 1000
        _emit_enrichment_complete(result, duration_ms)
        return result
    finally:
        _end_store_operation(store)
