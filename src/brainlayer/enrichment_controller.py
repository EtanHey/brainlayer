"""Retiring legacy producers; historical replay lives in enrichment_replay."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from .pipeline.cloud_scrub import CloudScrubError, scrub_for_cloud
from .pipeline.rate_limiter import TokenBucket
from .pipeline.write_queue import WriteQueue

logger = logging.getLogger(__name__)
_sleep = time.sleep

GEMINI_REALTIME_MODEL = os.environ.get("BRAINLAYER_GEMINI_REALTIME_MODEL", "gemini-2.5-flash-lite")
DEFAULT_MAX_COMMIT_INTERVAL_MS = 250.0
DEFAULT_POST_WRITE_YIELD_MS = 20.0
DEFAULT_ENRICH_SUPERVISOR_LIMIT = 200_000
DEFAULT_ENRICH_SUPERVISOR_SINCE_HOURS = 87_600
DEFAULT_ENRICH_IDLE_POLL_SECONDS = 30.0
DEFAULT_ENRICH_WATCHER_IDLE_SECONDS = 5.0
DEFAULT_ENRICH_DAILY_USD_CAP = 5.0
ENRICH_DAILY_COST_COUNTER_FILENAME = "enrich-daily-cost.json"
_ENRICHMENT_QUEUE_WRITES_OVERRIDE = threading.local()

# Gemini Developer API paid-tier text prices per 1M tokens for gemini-2.5-flash-lite.
GEMINI_FLASH_LITE_TEXT_PRICES_USD_PER_1M = {
    "standard": {"input": 0.10, "output": 0.40},
    "batch": {"input": 0.05, "output": 0.20},
    "flex": {"input": 0.05, "output": 0.20},
    "priority": {"input": 0.18, "output": 0.72},
}


def _bounded_nonnegative_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return max(0.0, default)
    return max(0.0, parsed)


def _bounded_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return max(1, default)
    return max(1, parsed)


# Auto-enrichment on brain_store: set to "0" or "false" to disable
AUTO_ENRICH_ENABLED = os.environ.get("BRAINLAYER_AUTO_ENRICH", "1").lower() not in ("0", "false", "no")

# Per-mode rate limits (requests per second). Override via env vars.
RATE_LIMITS = {
    "realtime": float(
        os.environ.get("BRAINLAYER_ENRICH_RATE", "5.0")
    ),  # 300 RPM default (AI Pro verified 500+ RPM Apr 2026)
    "batch": float(os.environ.get("BRAINLAYER_BATCH_RATE", "0")),  # no limit (async)
}
ENRICH_CONCURRENCY = int(os.environ.get("BRAINLAYER_ENRICH_CONCURRENCY", "10"))
MAX_COMMIT_BATCH = _bounded_positive_int(os.environ.get("BRAINLAYER_MAX_COMMIT_BATCH"), 25)
MAX_COMMIT_INTERVAL_SECONDS = (
    _bounded_nonnegative_float(os.environ.get("BRAINLAYER_MAX_COMMIT_INTERVAL_MS"), DEFAULT_MAX_COMMIT_INTERVAL_MS)
    / 1000.0
)
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
_ENRICH_COST_LOCK = threading.Lock()


@dataclass
class EnrichmentResult:
    mode: str
    attempted: int
    enriched: int
    skipped: int
    failed: int
    errors: list[str] = field(default_factory=list)


@dataclass
class EnrichmentSupervisorResult:
    mode: str = "supervisor"
    cycles: int = 0
    attempted: int = 0
    enriched: int = 0
    skipped: int = 0
    failed: int = 0
    failed_cycles: int = 0
    errors: list[str] = field(default_factory=list)
    exit_code: int = 0


class EnrichmentDailyCapReached(RuntimeError):
    """Raised when the local daily enrichment spend counter reaches its cap."""


def run_enrich_supervisor(
    db_path: Path | str,
    *,
    limit: int = DEFAULT_ENRICH_SUPERVISOR_LIMIT,
    since_hours: int = DEFAULT_ENRICH_SUPERVISOR_SINCE_HOURS,
    idle_poll_seconds: float | None = None,
    max_cycles: int | None = None,
    stop_event: threading.Event | None = None,
    vector_store_cls=None,
    enrich_fn=None,
    idle_backlog_ready_fn=None,
    sleep_fn=time.sleep,
) -> EnrichmentSupervisorResult:
    """Retired compatibility entrypoint; fails before using arguments."""
    raise RuntimeError("Cloud enrichment has been retired. Local checkpoint replay remains available.")


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
                        "enum": [
                            "person",
                            "agent",
                            "company",
                            "project",
                            "technology",
                            "tool",
                            "concept",
                            "topic",
                            "source",
                        ],
                    },
                    "entity_subtype": {
                        "type": "string",
                        "enum": ["channel", "podcast", "brand", "newsletter"],
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
    "required": [
        "summary",
        "tags",
        "importance",
        "intent",
        "entities",
        "sentiment_label",
        "sentiment_score",
        "sentiment_signals",
    ],
}


def _build_gemini_config() -> dict[str, Any]:
    return {
        "response_mime_type": "application/json",
        "response_schema": GEMINI_RESPONSE_SCHEMA,
        "thinking_config": {"thinking_budget": 0},
        "http_options": _build_gemini_http_options(),
    }


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

    Rate-limited by BRAINLAYER_ENRICH_RATE (default 5.0 req/s = 300 RPM).
    Timeout: 30 seconds per call.
    """
    raise RuntimeError("Cloud enrichment has been retired. Local checkpoint replay remains available.")
    try:
        client = _get_gemini_client()
    except RuntimeError:
        logger.debug("Gemini not available for extraction")
        return None

    try:
        response = _generate_content_with_rate_limit(
            client,
            GEMINI_EXTRACTION_MODEL,
            prompt,
            {
                "response_mime_type": "application/json",
                "thinking_config": {"thinking_budget": 0},
                "http_options": _build_gemini_http_options(timeout_ms=30_000),
            },
            None,
        )
        return response.text if response and response.text else None
    except Exception:
        logger.warning("Gemini extraction call failed", exc_info=True)
        return None


# ── Content-hash dedup ─────────────────────────────────────────────────────────


# The content_hash contract lives in ONE place. A second implementation of this
# function -- even a byte-identical one -- is precisely how four hash schemes got
# into the column, so the UPDATE paths below import it rather than redefine it.
from .chunk_write import canonical_content_hash as _content_hash  # noqa: E402


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


def _mark_duplicate_content(store, chunk: dict[str, Any]) -> None:
    cursor = store.conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    content = chunk.get("content", "")
    content_hash = _content_hash(content) if content else None
    if content_hash:
        cursor.execute(
            "UPDATE chunks SET enriched_at = ?, enrich_status = 'duplicate', content_hash = ? WHERE id = ?",
            (now, content_hash, chunk["id"]),
        )
    else:
        cursor.execute(
            "UPDATE chunks SET enriched_at = ?, enrich_status = 'duplicate' WHERE id = ?",
            (now, chunk["id"]),
        )


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
            if isinstance(exc, (EnrichmentDailyCapReached, CloudScrubError)) or _is_monthly_spending_cap_error(exc):
                raise
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
            _sleep(sleep_for)


def _generate_content_with_rate_limit(
    client, model: str, prompt: str, config: dict[str, Any], limiter: TokenBucket | None
):
    # Scrub before anything else: a scrub failure must never reach the network.
    prompt = scrub_for_cloud(prompt)
    _raise_if_enrich_daily_cap_reached()
    if limiter is not None:
        limiter.acquire()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    _record_enrich_response_usage(response)
    return response


def enrich_single(store, chunk_id: str, max_retries: int = 2) -> dict[str, Any] | None:
    """Retired compatibility entrypoint; fails before using arguments."""
    raise RuntimeError("Cloud enrichment has been retired. Local checkpoint replay remains available.")


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
    """Retired compatibility entrypoint; fails before using arguments."""
    raise RuntimeError("Cloud enrichment has been retired. Local checkpoint replay remains available.")


def enrich_batch(
    store,
    phase: str = "run",
    limit: int = 5000,
    max_retries: int = 3,
) -> EnrichmentResult:
    """Retired compatibility entrypoint; never opens a store or cloud client."""
    raise RuntimeError("Cloud enrichment has been retired. Local checkpoint replay remains available.")


def enrich_local(
    store,
    limit: int = 100,
    parallel: int = 2,
    backend: str = "mlx",
) -> EnrichmentResult:
    """Disabled legacy entrypoint kept only to fail loudly for stale callers."""
    del store, limit, parallel, backend
    raise RuntimeError("Local enrichment has been removed. Use Gemini realtime or batch modes.")


from .enrichment_replay import _apply_enrichment as _apply_enrichment
from .enrichment_replay import _apply_enrichment_impl as _apply_enrichment_impl
from .enrichment_replay import _backfill_content_hashes as _backfill_content_hashes
from .enrichment_replay import _chunk_columns as _chunk_columns
from .enrichment_replay import _current_auto_supersede_dry_run as _current_auto_supersede_dry_run
from .enrichment_replay import _current_enrichment_backend as _current_enrichment_backend
from .enrichment_replay import _derive_chunk_provenance_class as _derive_chunk_provenance_class
from .enrichment_replay import _enrichment_update_payload as _enrichment_update_payload
from .enrichment_replay import _ensure_content_hash_column as _ensure_content_hash_column
from .enrichment_replay import _ensure_provenance_class_column as _ensure_provenance_class_column
from .enrichment_replay import _ensure_raw_entities_json_column as _ensure_raw_entities_json_column
from .enrichment_replay import _entity_name_from_payload as _entity_name_from_payload
from .enrichment_replay import _get_chunk_readonly as _get_chunk_readonly
from .enrichment_replay import _get_gemini_service_tier as _get_gemini_service_tier
from .enrichment_replay import _is_duplicate_content as _is_duplicate_content
from .enrichment_replay import _maybe_auto_supersede_ingested_chunk as _maybe_auto_supersede_ingested_chunk
from .enrichment_replay import _previous_assistant_text as _previous_assistant_text
from .enrichment_replay import _savepoint as _savepoint
from .enrichment_replay import _with_enriched_by as _with_enriched_by
from .enrichment_replay import is_meta_research as is_meta_research
