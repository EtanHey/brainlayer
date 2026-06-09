"""Unified enrichment controller for Gemini-backed realtime and batch modes.

Replaces scattered enrichment scripts with a single controller:
  - realtime: Gemini 2.5 Flash-Lite, single chunk, <600ms target
  - batch: Gemini backlog processing, thinkingBudget=0
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
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from .chunk_origin import CHUNK_ORIGIN_GEMINI_FLASH_LITE
from .pipeline.enrichment import build_external_prompt, parse_enrichment
from .pipeline.rate_limiter import TokenBucket
from .pipeline.sanitize import Sanitizer
from .pipeline.write_queue import WriteQueue
from .provenance import derive_provenance_class
from .provenance_autosupersede import auto_supersede
from .provenance_integration import enqueue_provenance_resolution_for_entities

logger = logging.getLogger(__name__)

GEMINI_REALTIME_MODEL = os.environ.get("BRAINLAYER_GEMINI_REALTIME_MODEL", "gemini-2.5-flash-lite")
DEFAULT_MAX_COMMIT_INTERVAL_MS = 250.0
DEFAULT_POST_WRITE_YIELD_MS = 20.0
DEFAULT_ENRICH_SUPERVISOR_LIMIT = 200_000
DEFAULT_ENRICH_SUPERVISOR_SINCE_HOURS = 87_600
DEFAULT_ENRICH_IDLE_POLL_SECONDS = 30.0
DEFAULT_ENRICH_DAILY_USD_CAP = 5.0
ENRICH_DAILY_COST_COUNTER_FILENAME = "enrich-daily-cost.json"
_DEFAULT_CHUNK_ORIGIN = object()

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


def _enrich_daily_usd_cap() -> float:
    return _bounded_nonnegative_float(os.environ.get("BRAINLAYER_ENRICH_DAILY_USD_CAP"), DEFAULT_ENRICH_DAILY_USD_CAP)


def _local_cost_counter_date(now: datetime | None = None) -> str:
    current = now or datetime.now().astimezone()
    return current.date().isoformat()


def _enrich_cost_counter_path() -> Path:
    override_dir = os.environ.get("BRAINLAYER_ENRICH_COST_DIR")
    if override_dir:
        return Path(override_dir).expanduser() / ENRICH_DAILY_COST_COUNTER_FILENAME

    from .paths import get_db_path

    return get_db_path().parent / ENRICH_DAILY_COST_COUNTER_FILENAME


@contextmanager
def _locked_enrich_cost_counter(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _ENRICH_COST_LOCK:
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            try:
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass


def _read_enrich_cost_record(path: Path | None = None, today: str | None = None) -> dict[str, Any]:
    path = path or _enrich_cost_counter_path()
    today = today or _local_cost_counter_date()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"date": today, "spent_usd": 0.0}

    if data.get("date") != today:
        return {"date": today, "spent_usd": 0.0}

    try:
        spent = float(data.get("spent_usd", 0.0))
    except (TypeError, ValueError):
        spent = 0.0
    return {"date": today, "spent_usd": max(0.0, spent)}


def _write_enrich_cost_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _daily_cap_message(spent_usd: float, cap_usd: float) -> str:
    return f"ENRICH_DAILY_CAP_REACHED spent=${spent_usd:.6f} cap=${cap_usd:.2f}"


def _raise_if_enrich_daily_cap_reached() -> None:
    cap_usd = _enrich_daily_usd_cap()
    path = _enrich_cost_counter_path()
    today = _local_cost_counter_date()
    with _locked_enrich_cost_counter(path):
        record = _read_enrich_cost_record(path, today)
        spent_usd = float(record["spent_usd"])
        if spent_usd >= cap_usd:
            message = _daily_cap_message(spent_usd, cap_usd)
            logger.warning(message)
            raise EnrichmentDailyCapReached(message)


def _usage_value(usage: Any, *names: str) -> int:
    if usage is None:
        return 0
    for name in names:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0
    return 0


def _estimate_gemini_response_cost_usd(response: Any, service_tier: str | None = None) -> float:
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
    input_tokens = _usage_value(usage, "prompt_token_count", "promptTokenCount")
    candidate_tokens = _usage_value(usage, "candidates_token_count", "candidatesTokenCount")
    thoughts_tokens = _usage_value(usage, "thoughts_token_count", "thoughtsTokenCount")
    total_tokens = _usage_value(usage, "total_token_count", "totalTokenCount")
    output_tokens = candidate_tokens + thoughts_tokens
    if total_tokens and input_tokens:
        output_tokens = max(output_tokens, total_tokens - input_tokens)

    tier = (service_tier or _get_gemini_service_tier()).lower()
    prices = GEMINI_FLASH_LITE_TEXT_PRICES_USD_PER_1M.get(
        tier,
        GEMINI_FLASH_LITE_TEXT_PRICES_USD_PER_1M["standard"],
    )
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


def _record_enrich_response_usage(response: Any) -> float:
    cost_usd = _estimate_gemini_response_cost_usd(response)
    if cost_usd <= 0:
        return 0.0

    cap_usd = _enrich_daily_usd_cap()
    path = _enrich_cost_counter_path()
    today = _local_cost_counter_date()
    with _locked_enrich_cost_counter(path):
        record = _read_enrich_cost_record(path, today)
        spent_usd = float(record["spent_usd"]) + cost_usd
        _write_enrich_cost_record(path, {"date": today, "spent_usd": spent_usd})

    if spent_usd >= cap_usd:
        logger.warning(_daily_cap_message(spent_usd, cap_usd))
    return cost_usd


def _is_monthly_spending_cap_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "resource_exhausted" in text and "monthly" in text and "spending cap" in text


def _result_hit_daily_cap(result: EnrichmentResult) -> bool:
    return any(str(error).startswith("ENRICH_DAILY_CAP_REACHED") for error in result.errors)


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


def _current_idle_poll_seconds() -> float:
    return _bounded_nonnegative_float(
        os.environ.get("BRAINLAYER_ENRICH_IDLE_POLL_SECONDS"),
        DEFAULT_ENRICH_IDLE_POLL_SECONDS,
    )


def _sleep_or_wait_for_stop(stop_event: threading.Event | None, seconds: float, sleep_fn) -> None:
    if seconds <= 0:
        return
    if stop_event is not None and sleep_fn is time.sleep:
        stop_event.wait(seconds)
        return
    sleep_fn(seconds)


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
    sleep_fn=time.sleep,
) -> EnrichmentSupervisorResult:
    """Run realtime enrichment as a long-lived supervisor around one VectorStore.

    PR-alpha's writer pidfile mutex is acquired by the writable VectorStore
    constructor. Keeping that store alive here keeps the pidfile for the whole
    supervisor lifetime instead of reacquiring it on every launchd respawn.
    """
    from .vector_store import VectorStore

    if vector_store_cls is None:
        vector_store_cls = VectorStore
    if enrich_fn is None:
        enrich_fn = enrich_realtime
    if idle_poll_seconds is None:
        idle_poll_seconds = _current_idle_poll_seconds()

    db_path = Path(db_path)
    stats = EnrichmentSupervisorResult()
    store = _open_enrich_supervisor_store(vector_store_cls, db_path)
    logger.info("VectorStore initialized for enrich supervisor: %s", db_path)
    try:
        while stop_event is None or not stop_event.is_set():
            if max_cycles is not None and stats.cycles >= max_cycles:
                break

            try:
                result = enrich_fn(store, limit=limit, since_hours=since_hours)
            except EnrichmentDailyCapReached as exc:
                stats.cycles += 1
                stats.errors.append(str(exc))
                logger.warning("Enrich supervisor stopping: %s", exc)
                break
            except Exception as exc:  # noqa: BLE001
                stats.cycles += 1
                stats.failed_cycles += 1
                error = f"supervisor: {exc}"
                stats.errors.append(error)
                logger.exception("Enrich supervisor cycle failed; continuing")
                if max_cycles is None or stats.cycles < max_cycles:
                    _sleep_or_wait_for_stop(stop_event, idle_poll_seconds, sleep_fn)
                continue

            stats.cycles += 1
            stats.attempted += result.attempted
            stats.enriched += result.enriched
            stats.skipped += result.skipped
            stats.failed += result.failed
            stats.errors.extend(result.errors)

            if _result_hit_daily_cap(result):
                break

            if result.attempted == 0 and (max_cycles is None or stats.cycles < max_cycles):
                logger.info("Enrich supervisor queue empty; sleeping %.1fs", idle_poll_seconds)
                _sleep_or_wait_for_stop(stop_event, idle_poll_seconds, sleep_fn)
    finally:
        store.close()
    return stats


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


def _submit_write(store, name: str, callback, *, yield_after: bool = True) -> Any:
    result = _get_store_write_queue(store).submit(name, callback).result()
    yield_seconds = _current_post_write_yield_seconds() if yield_after else 0.0
    if yield_seconds > 0:
        time.sleep(yield_seconds)
    return result


def _arbitrated_writes_enabled() -> bool:
    return os.environ.get("BRAINLAYER_ARBITRATED") == "1"


def _open_enrich_supervisor_store(vector_store_cls, db_path: Path):
    if not _arbitrated_writes_enabled():
        return vector_store_cls(db_path)
    try:
        return vector_store_cls(db_path, readonly=True)
    except TypeError:
        logger.debug("VectorStore class does not accept readonly=True; opening with default constructor")
        return vector_store_cls(db_path)


def _current_max_commit_batch() -> int:
    return _bounded_positive_int(os.environ.get("BRAINLAYER_MAX_COMMIT_BATCH"), MAX_COMMIT_BATCH)


def _current_post_write_yield_seconds() -> float:
    return (
        _bounded_nonnegative_float(
            os.environ.get("BRAINLAYER_ENRICH_POST_WRITE_YIELD_MS"),
            DEFAULT_POST_WRITE_YIELD_MS,
        )
        / 1000.0
    )


def _current_enrichment_chunk_origin() -> str:
    return str(GEMINI_REALTIME_MODEL).strip() or CHUNK_ORIGIN_GEMINI_FLASH_LITE


def _current_auto_supersede_dry_run() -> bool | None:
    raw = str(os.environ.get("BRAINLAYER_AUTO_SUPERSEDE") or "").strip().lower()
    if raw in {"", "0", "false", "off", "no"}:
        return None
    return raw != "apply"


def _entity_name_from_payload(entity: Any) -> str:
    if isinstance(entity, dict):
        for key in ("name", "text", "entity", "label"):
            value = entity.get(key)
            if value:
                return str(value).strip()
        return ""
    return str(entity or "").strip()


def _maybe_auto_supersede_ingested_chunk(
    store,
    chunk: dict[str, Any],
    entities: list[Any],
    *,
    provenance_class: str,
) -> None:
    dry_run = _current_auto_supersede_dry_run()
    if dry_run is None:
        return

    for entity in entities:
        entity_name = _entity_name_from_payload(entity)
        if not entity_name:
            continue
        new_chunk = dict(chunk)
        new_chunk["entity"] = entity_name
        new_chunk["provenance_class"] = provenance_class
        try:
            report = auto_supersede(store, new_chunk, dry_run=dry_run)
        except Exception:
            logger.exception("auto_supersede failed for chunk=%s entity=%s", chunk.get("id"), entity_name)
            continue
        mode = "dry_run" if dry_run else "apply"
        if (
            report.candidate_count
            or report.contradiction_count
            or report.would_supersede_count
            or report.pending_confirm_count
            or report.skipped_count
        ):
            logger.info(
                "auto_supersede %s entity=%s candidates=%s contradictions=%s would_supersede=%s superseded=%s "
                "pending_confirm=%s skipped=%s",
                mode,
                report.entity,
                report.candidate_count,
                report.contradiction_count,
                report.would_supersede_count,
                report.superseded_count,
                report.pending_confirm_count,
                report.skipped_reason or report.skipped_count,
            )


def _enrichment_update_payload(
    chunk: dict[str, Any],
    enrichment: dict[str, Any],
    *,
    chunk_origin: str | None | object = _DEFAULT_CHUNK_ORIGIN,
) -> dict[str, Any]:
    content = chunk.get("content", "")
    if chunk_origin is _DEFAULT_CHUNK_ORIGIN:
        normalized_origin = _current_enrichment_chunk_origin()
    else:
        normalized_origin = str(chunk_origin or "").strip() or None
    return {
        "chunk_id": chunk["id"],
        "enrichment": enrichment,
        "content_hash": _content_hash(content) if content else None,
        "entities": enrichment.get("entities", []),
        "chunk_origin": normalized_origin,
        "provenance_class": derive_provenance_class(
            content_type=chunk.get("content_type"),
            sender=chunk.get("sender"),
            text=content,
            prev_assistant_text=chunk.get("prev_assistant_text"),
        ),
    }


def _enqueue_enrichment_write(
    chunk: dict[str, Any],
    enrichment: dict[str, Any],
    *,
    chunk_origin: str | None | object = _DEFAULT_CHUNK_ORIGIN,
) -> None:
    from .queue_io import enqueue_enrichment_updates

    try:
        enqueue_enrichment_updates([_enrichment_update_payload(chunk, enrichment, chunk_origin=chunk_origin)])
    except Exception:
        logger.exception("Failed to enqueue enrichment update for chunk %s", chunk.get("id"))
        raise


def _enqueue_enrichment_write_batch(items: list[tuple[dict[str, Any], dict[str, Any], str | None]]) -> None:
    if not items:
        return

    from .queue_io import enqueue_enrichment_updates

    try:
        enqueue_enrichment_updates(
            [
                _enrichment_update_payload(
                    chunk,
                    enrichment,
                    chunk_origin=None if counted_as == "skipped" else _DEFAULT_CHUNK_ORIGIN,
                )
                for chunk, enrichment, counted_as in items
            ]
        )
    except Exception:
        chunk_ids = ",".join(str(chunk.get("id")) for chunk, _, _ in items)
        logger.exception("Failed to enqueue enrichment update batch for chunks %s", chunk_ids)
        raise


class _EnrichmentWriteBatcher:
    def __init__(
        self,
        *,
        max_batch: int | None = None,
        max_interval_seconds: float = MAX_COMMIT_INTERVAL_SECONDS,
    ) -> None:
        self.max_batch = max(1, max_batch if max_batch is not None else _current_max_commit_batch())
        self.max_interval_seconds = max(0.0, max_interval_seconds)
        self._pending: list[tuple[dict[str, Any], dict[str, Any], str | None]] = []
        self._last_flush: float | None = None

    def enqueue(self, chunk: dict[str, Any], enrichment: dict[str, Any], *, counted_as: str | None = None) -> None:
        now = time.monotonic()
        if self._pending and self._last_flush is not None:
            elapsed = now - self._last_flush
            if elapsed >= self.max_interval_seconds:
                try:
                    # Flush overdue writes before appending the next result so drain gets smaller DB lock windows.
                    self.flush()
                    now = time.monotonic()
                except Exception:
                    logger.exception("Deferred overdue enrichment batch flush; retaining pending writes")
        if not self._pending:
            self._last_flush = now
        self._pending.append((chunk, enrichment, counted_as))
        if len(self._pending) >= self.max_batch:
            try:
                self.flush()
            except Exception:
                logger.exception("Deferred full enrichment batch flush; retaining pending writes")

    def flush(self) -> None:
        if not self._pending:
            return
        pending = self._pending
        _enqueue_enrichment_write_batch(pending)
        self._pending = []
        self._last_flush = None

    def pending_items(self) -> list[tuple[dict[str, Any], dict[str, Any], str | None]]:
        return list(self._pending)


def _flush_enrichment_batcher(
    write_batcher: _EnrichmentWriteBatcher,
    result: EnrichmentResult,
    mode: str,
) -> None:
    try:
        write_batcher.flush()
    except Exception as exc:
        pending = write_batcher.pending_items()
        if not pending:
            raise
        result.failed += len(pending)
        for chunk, _, counted_as in pending:
            if counted_as == "enriched" and result.enriched > 0:
                result.enriched -= 1
            elif counted_as == "skipped" and result.skipped > 0:
                result.skipped -= 1
            chunk_id = chunk.get("id")
            result.errors.append(f"{chunk_id}: {exc}")
            _emit_enrichment_error(mode, str(chunk_id), str(exc))


def _meta_research_enrichment(chunk: dict[str, Any]) -> dict[str, Any]:
    tags = _normalize_chunk_tags(chunk.get("tags"))
    if "meta-research" not in tags:
        tags.append("meta-research")
    return {
        "summary": None,
        "tags": tags,
    }


def _enqueue_meta_research_write(chunk: dict[str, Any]) -> None:
    _enqueue_enrichment_write(chunk, _meta_research_enrichment(chunk), chunk_origin=None)


def _ensure_enrichment_columns(store, *, yield_after: bool = True) -> None:
    if _arbitrated_writes_enabled():
        return

    key = _store_queue_key(store)
    with _ENRICHMENT_COLUMN_LOCK:
        if key in _ENRICHMENT_COLUMN_READY:
            return

    def _ensure() -> None:
        _ensure_content_hash_column(store)
        _ensure_raw_entities_json_column(store)
        _ensure_provenance_class_column(store)

    _submit_write(store, "ensure-enrichment-columns", _ensure, yield_after=yield_after)

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
    cursor = store.conn.cursor()
    try:
        cursor.execute("SELECT content_hash FROM chunks LIMIT 0")
    except Exception:
        try:
            cursor.execute("ALTER TABLE chunks ADD COLUMN content_hash TEXT")
        except Exception:
            return False

    try:
        indexes = list(cursor.execute("PRAGMA index_list(chunks)"))
        has_content_hash_index = False
        for row in indexes:
            index_name = row[1]
            is_unique = bool(row[2])
            quoted_name = index_name.replace('"', '""')
            columns = [info[2] for info in cursor.execute(f'PRAGMA index_info("{quoted_name}")')]
            if "content_hash" not in columns:
                continue
            if is_unique:
                cursor.execute(f'DROP INDEX IF EXISTS "{quoted_name}"')
                continue
            has_content_hash_index = True

        if not has_content_hash_index:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON chunks(content_hash)")
    except Exception:
        pass

    return True


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


def _ensure_provenance_class_column(store) -> bool:
    """Ensure the provenance_class staging column exists on chunks."""
    try:
        store.conn.cursor().execute("SELECT provenance_class FROM chunks LIMIT 0")
        setattr(store, "_has_provenance_class", True)
        return True
    except Exception:
        try:
            store.conn.cursor().execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
            setattr(store, "_has_provenance_class", True)
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
            if isinstance(exc, EnrichmentDailyCapReached) or _is_monthly_spending_cap_error(exc):
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
            time.sleep(sleep_for)


def _generate_content_with_rate_limit(
    client, model: str, prompt: str, config: dict[str, Any], limiter: TokenBucket | None
):
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


def _apply_enrichment(
    store,
    chunk: dict[str, Any],
    enrichment: dict[str, Any],
    *,
    chunk_origin: str | None = None,
) -> None:
    resolved_queries = enrichment.get("resolved_queries")
    legacy_resolved_query = enrichment.get("resolved_query")
    if not legacy_resolved_query and isinstance(resolved_queries, list) and resolved_queries:
        legacy_resolved_query = resolved_queries[0]
    normalized_origin = str(chunk_origin or _current_enrichment_chunk_origin()).strip()

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
        chunk_origin=normalized_origin or None,
    )
    entities = enrichment.get("entities", [])
    # AIDEV-NOTE: raw entities persisted to chunks.raw_entities_json staging column;
    # R84b canonicalization pipeline will consume and populate kg_entities downstream.
    if _ensure_raw_entities_json_column(store):
        store.conn.cursor().execute(
            "UPDATE chunks SET raw_entities_json = ? WHERE id = ?",
            (json.dumps(entities), chunk["id"]),
        )
        try:
            from .kg_promotion import promote_chunk_raw_entities
            from .vector_store import VectorStore

            if isinstance(store, VectorStore):
                promote_chunk_raw_entities(store, chunk["id"])
        except Exception:
            logger.debug("raw entity KG promotion skipped for %s", chunk["id"], exc_info=True)
    # Set content_hash after enrichment so dedup works next time
    content = chunk.get("content", "")
    if content:
        try:
            h = _content_hash(content)
            store.conn.cursor().execute("UPDATE chunks SET content_hash = ? WHERE id = ?", (h, chunk["id"]))
        except Exception:
            pass  # Non-critical — dedup still works on next index
    provenance_class = derive_provenance_class(
        content_type=chunk.get("content_type"),
        sender=chunk.get("sender"),
        text=content,
        prev_assistant_text=chunk.get("prev_assistant_text"),
    )
    if _ensure_provenance_class_column(store):
        store.conn.cursor().execute(
            "UPDATE chunks SET provenance_class = ? WHERE id = ?",
            (provenance_class, chunk["id"]),
        )
    _maybe_auto_supersede_ingested_chunk(store, chunk, entities, provenance_class=provenance_class)
    enqueue_provenance_resolution_for_entities(store, entities, chunk_id=chunk["id"])


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
    except EnrichmentDailyCapReached:
        raise
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
        if not _arbitrated_writes_enabled():
            _ensure_enrichment_columns(store)

        chunk = _get_chunk_readonly(store, chunk_id)
        if not chunk:
            logger.warning("enrich_single: chunk not found: %s", chunk_id)
            return None

        if is_meta_research(chunk.get("content", "")):
            if _arbitrated_writes_enabled():
                _enqueue_meta_research_write(chunk)
            else:
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
        except EnrichmentDailyCapReached as exc:
            logger.warning("enrich_single: %s", exc)
            return None
        except Exception as exc:
            logger.warning("enrich_single: Gemini call failed for %s: %s", chunk_id, exc)
            return None

        enrichment = parse_enrichment(raw_response)
        if not enrichment:
            logger.warning("enrich_single: invalid enrichment response for %s", chunk_id)
            return None

        try:
            if _arbitrated_writes_enabled():
                _enqueue_enrichment_write(chunk, enrichment)
            else:
                _submit_write(
                    store, f"apply-enrichment:{chunk_id}", lambda: _apply_enrichment(store, chunk, enrichment)
                )
        except Exception as exc:
            logger.warning("enrich_single: apply failed for %s: %s", chunk_id, exc)
            return None

        logger.info("enrich_single: enriched %s with %d tags", chunk_id, len(enrichment.get("tags", [])))
        return enrichment
    finally:
        _end_store_operation(store)


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

        if not _arbitrated_writes_enabled():
            _ensure_enrichment_columns(store, yield_after=rate_per_second > 0)

        client = _get_gemini_client()
        sanitizer = Sanitizer.from_env()
        config = _build_gemini_config()
        rate_limiter = _get_store_rate_limiter(store, rate_per_second=rate_per_second)
        write_batcher = _EnrichmentWriteBatcher() if _arbitrated_writes_enabled() else None

        def is_duplicate(content: str) -> bool:
            return _is_duplicate_content(store, content)

        try:
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
                    try:
                        chunk, status, data = future.result()
                    except EnrichmentDailyCapReached as exc:
                        result.errors.append(str(exc))
                        _emit_enrichment_error("realtime", "daily-cap", str(exc))
                        for pending in futures:
                            pending.cancel()
                        break
                    if status == "skip":
                        result.skipped += 1
                        continue
                    if status == "meta":
                        if write_batcher is not None:
                            write_batcher.enqueue(chunk, _meta_research_enrichment(chunk), counted_as="skipped")
                        else:
                            _submit_write(
                                store,
                                f"mark-meta:{chunk['id']}",
                                lambda chunk=chunk: _mark_meta_research(store, chunk),
                                yield_after=rate_per_second > 0,
                            )
                        result.skipped += 1
                        continue
                    if status == "error":
                        result.failed += 1
                        result.errors.append(f"{chunk['id']}: {data}")
                        _emit_enrichment_error("realtime", chunk["id"], str(data))
                        continue

                    if write_batcher is not None:
                        write_batcher.enqueue(chunk, data, counted_as="enriched")
                    else:
                        _submit_write(
                            store,
                            f"apply-enrichment:{chunk['id']}",
                            lambda chunk=chunk, data=data: _apply_enrichment(store, chunk, data),
                            yield_after=rate_per_second > 0,
                        )
                    result.enriched += 1
        finally:
            if write_batcher is not None:
                _flush_enrichment_batcher(write_batcher, result, "realtime")

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

        if not _arbitrated_writes_enabled():
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
        rate_limiter = _get_store_rate_limiter(store, rate_per_second=RATE_LIMITS["batch"])
        write_batcher = _EnrichmentWriteBatcher() if _arbitrated_writes_enabled() else None

        try:
            for chunk in candidates:
                if is_meta_research(chunk.get("content", "")):
                    if write_batcher is not None:
                        write_batcher.enqueue(chunk, _meta_research_enrichment(chunk), counted_as="skipped")
                    else:
                        _submit_write(
                            store, f"mark-meta:{chunk['id']}", lambda chunk=chunk: _mark_meta_research(store, chunk)
                        )
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
                    if write_batcher is not None:
                        write_batcher.enqueue(chunk, enrichment, counted_as="enriched")
                    else:
                        _submit_write(
                            store,
                            f"apply-enrichment:{chunk['id']}",
                            lambda chunk=chunk, enrichment=enrichment: _apply_enrichment(store, chunk, enrichment),
                        )
                    result.enriched += 1
                except EnrichmentDailyCapReached as exc:
                    result.errors.append(str(exc))
                    _emit_enrichment_error("batch", "daily-cap", str(exc))
                    break
                except Exception as exc:
                    result.failed += 1
                    result.errors.append(f"{chunk['id']}: {exc}")
                    _emit_enrichment_error("batch", chunk["id"], str(exc))
        finally:
            if write_batcher is not None:
                _flush_enrichment_batcher(write_batcher, result, "batch")

        duration_ms = (time.monotonic() - start_time) * 1000
        _emit_enrichment_complete(result, duration_ms)
        return result
    finally:
        _end_store_operation(store)


def enrich_local(
    store,
    limit: int = 100,
    parallel: int = 2,
    backend: str = "mlx",
) -> EnrichmentResult:
    """Disabled legacy entrypoint kept only to fail loudly for stale callers."""
    del store, limit, parallel, backend
    raise RuntimeError("Local enrichment has been removed. Use Gemini realtime or batch modes.")
