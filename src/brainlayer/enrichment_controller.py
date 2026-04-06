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
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from .pipeline.enrichment import (
    build_external_prompt,
    build_prompt,
    parse_enrichment,
)
from .pipeline.sanitize import Sanitizer

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
    if region:
        return genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta", "url": f"https://{region}-aiplatform.googleapis.com"},
        )
    return genai.Client(api_key=api_key)


GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "importance": {"type": "number"},
        "intent": {"type": "string"},
        "primary_symbols": {"type": "array", "items": {"type": "string"}},
        "resolved_query": {"type": "string"},
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
    }


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
                "http_options": {"timeout": 30_000},
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


def _is_duplicate_content(store, content: str) -> bool:
    """Check if content with the same hash already exists and is enriched.

    Returns True if a chunk with identical content hash exists and has been enriched,
    meaning re-enriching would be a no-op.
    """
    content_h = _content_hash(content)
    try:
        cursor = store._read_cursor()
        row = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE content_hash = ? AND enriched_at IS NOT NULL",
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
            time.sleep(min(delay + jitter, max_delay))


def _apply_enrichment(store, chunk: dict[str, Any], enrichment: dict[str, Any]) -> None:
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
    # Set content_hash after enrichment so dedup works next time
    content = chunk.get("content", "")
    if content:
        try:
            h = _content_hash(content)
            store.conn.cursor().execute("UPDATE chunks SET content_hash = ? WHERE id = ?", (h, chunk["id"]))
        except Exception:
            pass  # Non-critical — dedup still works on next index


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

    chunk = store.get_chunk(chunk_id)
    if not chunk:
        logger.warning("enrich_single: chunk not found: %s", chunk_id)
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

    def _call():
        response = client.models.generate_content(
            model=GEMINI_REALTIME_MODEL,
            contents=prompt,
            config=config,
        )
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
        _apply_enrichment(store, chunk, enrichment)
    except Exception as exc:
        logger.warning("enrich_single: apply failed for %s: %s", chunk_id, exc)
        return None

    logger.info("enrich_single: enriched %s with %d tags", chunk_id, len(enrichment.get("tags", [])))
    return enrichment


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

    start_time = time.monotonic()
    _emit_enrichment_start("realtime", limit)

    candidates = store.get_enrichment_candidates(limit=limit, since_hours=since_hours, chunk_ids=chunk_ids)
    result = EnrichmentResult(mode="realtime", attempted=len(candidates), enriched=0, skipped=0, failed=0)
    if not candidates:
        _emit_enrichment_complete(result, 0)
        return result

    # Ensure content_hash column exists for dedup
    _ensure_content_hash_column(store)

    client = _get_gemini_client()
    sanitizer = Sanitizer.from_env()
    per_chunk_delay = (1.0 / rate_per_second) if rate_per_second > 0 else 0.0

    for index, chunk in enumerate(candidates):
        # Content-hash dedup: skip if identical content already enriched
        if _is_duplicate_content(store, chunk.get("content", "")):
            result.skipped += 1
            continue

        try:
            prompt, _sanitize_result = build_external_prompt(chunk, sanitizer)

            def _call():
                response = client.models.generate_content(
                    model=GEMINI_REALTIME_MODEL,
                    contents=prompt,
                    config=_build_gemini_config(),
                )
                return getattr(response, "text", None)

            raw_response = _retry_with_backoff(_call, max_retries=max_retries)
            enrichment = parse_enrichment(raw_response)
            if not enrichment:
                result.failed += 1
                result.errors.append(f"{chunk['id']}: invalid_enrichment")
                _emit_enrichment_error("realtime", chunk["id"], "invalid_enrichment")
            else:
                _apply_enrichment(store, chunk, enrichment)
                result.enriched += 1
        except Exception as exc:  # noqa: BLE001
            result.failed += 1
            result.errors.append(f"{chunk['id']}: {exc}")
            _emit_enrichment_error("realtime", chunk["id"], str(exc))

        if per_chunk_delay > 0 and index < len(candidates) - 1:
            time.sleep(per_chunk_delay)

    duration_ms = (time.monotonic() - start_time) * 1000
    _emit_enrichment_complete(result, duration_ms)
    return result


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
    start_time = time.monotonic()
    _emit_enrichment_start("batch", limit)

    candidates = store.get_enrichment_candidates(limit=limit, chunk_ids=None)
    result = EnrichmentResult(mode="batch", attempted=len(candidates), enriched=0, skipped=0, failed=0, errors=[])

    if not candidates:
        duration_ms = (time.monotonic() - start_time) * 1000
        _emit_enrichment_complete(result, duration_ms)
        return result

    _ensure_content_hash_column(store)

    try:
        client = _get_gemini_client()
    except RuntimeError as exc:
        result.errors.append(f"No Gemini client: {exc}")
        duration_ms = (time.monotonic() - start_time) * 1000
        _emit_enrichment_complete(result, duration_ms)
        return result

    sanitizer = Sanitizer.from_env()
    config = _build_gemini_config()
    rate_limit = RATE_LIMITS.get("realtime", 0.2)

    for chunk in candidates:
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
                lambda: client.models.generate_content(
                    model=GEMINI_REALTIME_MODEL,
                    contents=prompt,
                    config=config,
                ),
                max_retries=max_retries,
            )
            enrichment = parse_enrichment(response.text)
            if not enrichment:
                result.failed += 1
                result.errors.append(f"{chunk['id']}: invalid_enrichment")
                _emit_enrichment_error("batch", chunk["id"], "invalid_enrichment")
                continue
            _apply_enrichment(store, chunk, enrichment)
            result.enriched += 1
        except Exception as exc:
            result.failed += 1
            result.errors.append(f"{chunk['id']}: {exc}")
            _emit_enrichment_error("batch", chunk["id"], str(exc))

        if rate_limit > 0:
            time.sleep(1.0 / rate_limit)

    duration_ms = (time.monotonic() - start_time) * 1000
    _emit_enrichment_complete(result, duration_ms)
    return result


def enrich_local(
    store,
    limit: int = 100,
    parallel: int = 2,  # noqa: ARG001
    backend: str = "mlx",
) -> EnrichmentResult:
    """Enrich via local MLX/Ollama backend."""
    start_time = time.monotonic()
    _emit_enrichment_start("local", limit)

    candidates = store.get_enrichment_candidates(limit=limit, chunk_ids=None)
    result = EnrichmentResult(mode="local", attempted=len(candidates), enriched=0, skipped=0, failed=0)

    # Ensure content_hash column exists for dedup
    _ensure_content_hash_column(store)

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
            _apply_enrichment(store, chunk, enrichment)
            result.enriched += 1
        except Exception as exc:  # noqa: BLE001
            result.failed += 1
            result.errors.append(f"{chunk['id']}: {exc}")
            _emit_enrichment_error("local", chunk["id"], str(exc))

    duration_ms = (time.monotonic() - start_time) * 1000
    _emit_enrichment_complete(result, duration_ms)
    return result
