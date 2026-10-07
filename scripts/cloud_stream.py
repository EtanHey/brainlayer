#!/usr/bin/env python3
"""Stream enrichment — process pre-exported JSONL via regular Gemini API.

Workaround for Gemini Batch API 429 bug (Jan 2026). Reads the JSONL files
exported by cloud_backfill.py and processes them via regular API with
asyncio concurrency and rate limiting.

Usage:
    # Process all exported JSONL files (default: 50 concurrent workers)
    python3 scripts/cloud_stream.py

    # Custom concurrency (stay under 2000 RPM Tier 1 limit)
    python3 scripts/cloud_stream.py --workers 100

    # Process specific files
    python3 scripts/cloud_stream.py --files backfill_data/batch_*_001.jsonl

    # Dry run — count chunks, estimate cost
    python3 scripts/cloud_stream.py --dry-run

    # Resume from where we left off (skips already-enriched chunks)
    python3 scripts/cloud_stream.py  # Always resumes automatically
"""

import sys as _gate_sys

# Enrichment retired (2026-10-07). Keep this gate until final script deletion.
_gate_sys.exit("RETIRED: enrichment is removed; this script cannot run.")

import asyncio
import os
import signal
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.paths import get_db_path
from brainlayer.pipeline.enrichment import parse_enrichment
from brainlayer.vector_store import VectorStore

# ── Config ──────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = get_db_path()
EXPORT_DIR = Path(__file__).resolve().parent / "backfill_data"
MODEL = "models/gemini-2.5-flash"

# Rate limiting — Tier 1: 2000 RPM, 4M TPM
MAX_RPM = 1500  # Stay safely under 2000 RPM limit
RATE_LIMIT_DELAY = 60.0 / MAX_RPM  # ~0.04s between requests

# Gemini 2.5 Flash pricing
COST_PER_M_INPUT = 0.15  # $/1M input tokens
COST_PER_M_OUTPUT = 0.60  # $/1M output tokens

# ── Globals for graceful shutdown ───────────────────────────────────────

shutdown_requested = False
total_stats = {"success": 0, "failed": 0, "skipped": 0, "input_tokens": 0, "output_tokens": 0}


def handle_signal(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    print("\n[SIGINT] Graceful shutdown requested — finishing in-flight requests...")


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# ── Gemini client ──────────────────────────────────────────────────────


def get_genai_client():
    """Get a google.genai Client."""
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai not installed. Run: pip install google-genai")
        sys.exit(1)

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY not set")
        sys.exit(1)

    return genai.Client(api_key=api_key)


# ── Rate limiter ───────────────────────────────────────────────────────


class RateLimiter:
    """Token bucket rate limiter for RPM."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.lock = asyncio.Lock()
        self.last_request = 0.0

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            wait = self.interval - (now - self.last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_request = time.monotonic()


# ── Worker ─────────────────────────────────────────────────────────────


async def process_chunk(
    client,
    chunk_id: str,
    prompt: str,
    store: VectorStore,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
) -> str:
    """Process a single chunk via Gemini API. Returns 'success', 'failed', or 'skipped'."""
    global total_stats, shutdown_requested

    if shutdown_requested:
        return "skipped"

    async with semaphore:
        # Check if already enriched (auto-resume)
        cursor = store.conn.cursor()
        existing = list(cursor.execute("SELECT enriched_at FROM chunks WHERE id = ?", [chunk_id]))
        if existing and existing[0][0] is not None:
            total_stats["skipped"] += 1
            return "skipped"

        await rate_limiter.acquire()

        try:
            # Run the sync API call in a thread to not block the event loop
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                    "max_output_tokens": 512,
                },
            )

            response_text = response.text
            usage = response.usage_metadata

            if usage:
                total_stats["input_tokens"] += getattr(usage, "prompt_token_count", 0) or 0
                total_stats["output_tokens"] += getattr(usage, "candidates_token_count", 0) or 0

            enrichment = parse_enrichment(response_text)
            if enrichment:
                store.update_enrichment(
                    chunk_id=chunk_id,
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
                total_stats["success"] += 1
                return "success"
            else:
                total_stats["failed"] += 1
                return "failed"

        except Exception as e:
            err = str(e)
            if "429" in err:
                # Rate limited — wait and retry once
                await asyncio.sleep(5)
                try:
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=MODEL,
                        contents=prompt,
                        config={
                            "response_mime_type": "application/json",
                            "temperature": 0.1,
                            "max_output_tokens": 512,
                        },
                    )
                    enrichment = parse_enrichment(response.text)
                    if enrichment:
                        store.update_enrichment(
                            chunk_id=chunk_id,
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
                        total_stats["success"] += 1
                        return "success"
                except Exception:
                    pass

            total_stats["failed"] += 1
            return "failed"


# ── Main ───────────────────────────────────────────────────────────────


# ── CLI ─────────────────────────────────────────────────────────────────
