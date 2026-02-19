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

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.vector_store import VectorStore
from brainlayer.pipeline.enrichment import parse_enrichment

# ── Config ──────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "brainlayer" / "brainlayer.db"
EXPORT_DIR = Path(__file__).resolve().parent / "backfill_data"
MODEL = "models/gemini-2.5-flash"

# Rate limiting — Tier 1: 2000 RPM, 4M TPM
MAX_RPM = 1500  # Stay safely under 2000 RPM limit
RATE_LIMIT_DELAY = 60.0 / MAX_RPM  # ~0.04s between requests

# Gemini 2.5 Flash pricing
COST_PER_M_INPUT = 0.15   # $/1M input tokens
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
        existing = list(cursor.execute(
            "SELECT enriched_at FROM chunks WHERE id = ?", [chunk_id]
        ))
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

async def run_stream(
    db_path: Path,
    jsonl_files: List[Path],
    workers: int = 50,
    dry_run: bool = False,
) -> None:
    """Process all JSONL files via streaming API."""
    global total_stats, shutdown_requested

    store = VectorStore(db_path)

    # Load all chunks from JSONL files
    chunks = []
    for jsonl_path in jsonl_files:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    chunk_id = data["key"]
                    prompt = data["request"]["contents"][0]["parts"][0]["text"]
                    chunks.append((chunk_id, prompt))
                except (json.JSONDecodeError, KeyError):
                    continue

    # Filter already-enriched (fast DB check)
    cursor = store.conn.cursor()
    unenriched = []
    for chunk_id, prompt in chunks:
        existing = list(cursor.execute(
            "SELECT enriched_at FROM chunks WHERE id = ?", [chunk_id]
        ))
        if not existing or existing[0][0] is None:
            unenriched.append((chunk_id, prompt))

    print(f"Total chunks in JSONL files: {len(chunks):,}")
    print(f"Already enriched (skipping): {len(chunks) - len(unenriched):,}")
    print(f"To process: {len(unenriched):,}")

    # Cost estimate
    avg_prompt_tokens = 700
    avg_output_tokens = 200
    est_input = len(unenriched) * avg_prompt_tokens
    est_output = len(unenriched) * avg_output_tokens
    est_cost = (est_input * COST_PER_M_INPUT + est_output * COST_PER_M_OUTPUT) / 1_000_000
    est_minutes = len(unenriched) / MAX_RPM
    print(f"Estimated cost: ${est_cost:.2f}")
    print(f"Estimated time: {est_minutes:.0f} min ({est_minutes/60:.1f} hours) at {MAX_RPM} RPM")

    if dry_run:
        print("\n[DRY RUN] Not processing.")
        return

    if not unenriched:
        print("Nothing to process!")
        return

    print(f"\nStarting with {workers} workers, {MAX_RPM} RPM limit...")
    print("Press Ctrl+C for graceful shutdown.\n")

    client = get_genai_client()
    rate_limiter = RateLimiter(MAX_RPM)
    semaphore = asyncio.Semaphore(workers)

    start_time = time.time()
    last_report = start_time

    # Process in batches for progress reporting
    batch_size = 500
    for batch_start in range(0, len(unenriched), batch_size):
        if shutdown_requested:
            break

        batch = unenriched[batch_start:batch_start + batch_size]
        tasks = [
            process_chunk(client, chunk_id, prompt, store, rate_limiter, semaphore)
            for chunk_id, prompt in batch
        ]
        await asyncio.gather(*tasks)

        # Progress report
        processed = total_stats["success"] + total_stats["failed"] + total_stats["skipped"]
        elapsed = time.time() - start_time
        rpm = processed / (elapsed / 60) if elapsed > 0 else 0
        remaining = len(unenriched) - processed
        eta_min = remaining / rpm if rpm > 0 else 0
        cost_so_far = (
            total_stats["input_tokens"] * COST_PER_M_INPUT +
            total_stats["output_tokens"] * COST_PER_M_OUTPUT
        ) / 1_000_000

        print(
            f"  [{processed:,}/{len(unenriched):,}] "
            f"{total_stats['success']:,} ok, {total_stats['failed']:,} fail | "
            f"{rpm:.0f} RPM | ETA {eta_min:.0f}min | ${cost_so_far:.2f}"
        )

    # Final summary
    elapsed = time.time() - start_time
    cost = (
        total_stats["input_tokens"] * COST_PER_M_INPUT +
        total_stats["output_tokens"] * COST_PER_M_OUTPUT
    ) / 1_000_000

    print(f"\n{'='*60}")
    print("STREAM ENRICHMENT COMPLETE" + (" (interrupted)" if shutdown_requested else ""))
    print(f"  Success: {total_stats['success']:,}")
    print(f"  Failed:  {total_stats['failed']:,}")
    print(f"  Skipped: {total_stats['skipped']:,}")
    print(f"  Tokens:  {total_stats['input_tokens']:,} in / {total_stats['output_tokens']:,} out")
    print(f"  Cost:    ${cost:.2f}")
    print(f"  Time:    {elapsed/60:.1f} min")
    print(f"{'='*60}")

    # Log usage to Supabase
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if supabase_url and supabase_key and total_stats["success"] > 0:
        try:
            import requests
            requests.post(
                f"{supabase_url}/rest/v1/llm_usage",
                headers={
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json={
                    "model": MODEL.replace("models/", ""),
                    "source": "enrichment-stream",
                    "input_tokens": total_stats["input_tokens"],
                    "output_tokens": total_stats["output_tokens"],
                    "cost_usd": cost,
                    "tier": "paid",
                },
                timeout=5,
            )
            print("Usage logged to Supabase.")
        except Exception:
            pass

    store.close()


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stream enrichment via regular Gemini API")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument("--workers", type=int, default=50, help="Concurrent workers (default: 50)")
    parser.add_argument("--files", nargs="+", help="Specific JSONL files to process")
    parser.add_argument("--dry-run", action="store_true", help="Count chunks and estimate cost")
    parser.add_argument("--rpm", type=int, default=1500, help="Max requests per minute (default: 1500)")

    args = parser.parse_args()
    db = Path(args.db) if args.db else DEFAULT_DB_PATH

    global MAX_RPM, RATE_LIMIT_DELAY
    MAX_RPM = args.rpm
    RATE_LIMIT_DELAY = 60.0 / MAX_RPM

    # Find JSONL files
    if args.files:
        jsonl_files = [Path(f) for f in args.files]
    else:
        # Find the full-run files (largest set from most recent export)
        pattern = str(EXPORT_DIR / "batch_*.jsonl")
        all_files = sorted(glob(pattern))
        if not all_files:
            print(f"No JSONL files found in {EXPORT_DIR}")
            sys.exit(1)

        # Take files >1MB (full-run files are ~40MB each, test files are <1MB)
        large_files = [f for f in all_files if Path(f).stat().st_size > 1_000_000]
        if large_files:
            jsonl_files = [Path(f) for f in sorted(large_files)]
        else:
            # Fallback: use all files
            jsonl_files = [Path(f) for f in sorted(all_files)]

    print(f"JSONL files: {len(jsonl_files)}")
    for f in jsonl_files[:3]:
        print(f"  {f.name}")
    if len(jsonl_files) > 3:
        print(f"  ... and {len(jsonl_files) - 3} more")

    asyncio.run(run_stream(db, jsonl_files, workers=args.workers, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
