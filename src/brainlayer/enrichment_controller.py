"""Unified enrichment controller for realtime, batch, and local modes."""

from __future__ import annotations

import importlib.util
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pipeline.enrichment import (
    build_external_prompt,
    build_prompt,
    parse_enrichment,
)
from .pipeline.sanitize import Sanitizer

GEMINI_REALTIME_MODEL = os.environ.get("BRAINLAYER_GEMINI_REALTIME_MODEL", "gemini-2.5-flash-lite")


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


def _get_gemini_client():
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY not set")
    return genai.Client(api_key=api_key)


def _build_gemini_config() -> dict[str, Any]:
    return {
        "response_mime_type": "application/json",
        "thinking_config": {"thinking_budget": 0},
    }


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


def _call_local_backend(prompt: str, backend: str = "mlx") -> str | None:
    from .pipeline.enrichment import call_llm

    return call_llm(prompt, backend=backend)


def enrich_realtime(
    store,
    limit: int = 25,
    since_hours: int = 24,
    rate_per_second: float = float(
        os.environ.get("BRAINLAYER_ENRICH_RATE", "0.2")
    ),  # Default 12 RPM. Tier 1 allows 2000 RPM (~33/s)
    max_retries: int = 12,
    chunk_ids: list[str] | None = None,
) -> EnrichmentResult:
    """Enrich recent chunks via Gemini free-tier API."""
    candidates = store.get_enrichment_candidates(limit=limit, since_hours=since_hours, chunk_ids=chunk_ids)
    result = EnrichmentResult(mode="realtime", attempted=len(candidates), enriched=0, skipped=0, failed=0)
    if not candidates:
        return result

    client = _get_gemini_client()
    sanitizer = Sanitizer.from_env()
    per_chunk_delay = (1.0 / rate_per_second) if rate_per_second > 0 else 0.0

    for index, chunk in enumerate(candidates):
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
            else:
                _apply_enrichment(store, chunk, enrichment)
                result.enriched += 1
        except Exception as exc:  # noqa: BLE001
            result.failed += 1
            result.errors.append(f"{chunk['id']}: {exc}")

        if per_chunk_delay > 0 and index < len(candidates) - 1:
            time.sleep(per_chunk_delay)

    return result


def enrich_batch(
    store,
    phase: str = "run",
    limit: int = 5000,
    max_retries: int = 12,  # noqa: ARG001
) -> EnrichmentResult:
    """Process backlog via Gemini Batch API."""
    ensure_checkpoint_table(store)
    pending = get_pending_jobs(store) if phase in {"poll", "run"} else []
    export_files = (
        get_unsubmitted_export_files(db_path=getattr(store, "db_path", None)) if phase in {"submit", "run"} else []
    )
    attempted = len(pending) + len(export_files)
    return EnrichmentResult(mode="batch", attempted=attempted, enriched=0, skipped=0, failed=0, errors=[])


def enrich_local(
    store,
    limit: int = 100,
    parallel: int = 2,  # noqa: ARG001
    backend: str = "mlx",
) -> EnrichmentResult:
    """Enrich via local MLX backend."""
    candidates = store.get_enrichment_candidates(limit=limit, chunk_ids=None)
    result = EnrichmentResult(mode="local", attempted=len(candidates), enriched=0, skipped=0, failed=0)

    for chunk in candidates:
        try:
            prompt = build_prompt(chunk)
            raw_response = _retry_with_backoff(lambda: _call_local_backend(prompt, backend=backend), max_retries=2)
            enrichment = parse_enrichment(raw_response)
            if not enrichment:
                result.failed += 1
                result.errors.append(f"{chunk['id']}: invalid_enrichment")
                continue
            _apply_enrichment(store, chunk, enrichment)
            result.enriched += 1
        except Exception as exc:  # noqa: BLE001
            result.failed += 1
            result.errors.append(f"{chunk['id']}: {exc}")

    return result
