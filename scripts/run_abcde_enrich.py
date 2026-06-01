#!/usr/bin/env python3
"""Drive the ABCDE variant enrichment run against an OpenAI-compatible backend.

Read-only chunk sampling (never mutates the DB), mandatory sanitization (inside
the runner), live cost metering with a hard spend stop. The JUDGE leg is offline
and not invoked here.

Usage:
  # smoke: measure real per-item tokens/cost, write nothing permanent
  op read "op://Private/grok-content-golem/credential" | \
    python3 scripts/run_abcde_enrich.py --smoke 4 --out /tmp/abcde_smoke.jsonl

  # run: auto-size N to fit the budget, write judge-ready JSONL
  python3 scripts/run_abcde_enrich.py --run --max-usd 4.20 --avg-usd-per-call 0.019 \
    --out eval_results/abcde_enrich.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainlayer.eval.abcde_enrich_runner import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_TICK_USD,
    make_http_chat_fn,
    run_batch,
)
from brainlayer.eval.abcde_variants import ABCDE_VARIANTS  # noqa: E402
from brainlayer.paths import get_db_path  # noqa: E402
from brainlayer.pipeline.sanitize import Sanitizer  # noqa: E402

HIGH_VALUE_TYPES = ("ai_code", "user_message", "assistant_text", "stack_trace")


def sample_chunks(n: int, *, min_chars: int = 80, seed: int | None = None) -> list[dict]:
    """Read-only diverse sample of enrichable chunks. Never writes."""
    db = get_db_path()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" for _ in HIGH_VALUE_TYPES)
        # Deterministic-ish: order by a hash of id so the same N is stable across
        # smoke/run invocations without a writable RANDOM() seed table.
        rows = con.execute(
            f"""SELECT id, content, project, content_type, source
                FROM chunks
                WHERE content_type IN ({placeholders})
                  AND char_count >= ?
                  AND content IS NOT NULL
                ORDER BY substr(id, -6), id
                LIMIT ?""",
            (*HIGH_VALUE_TYPES, min_chars, n),
        ).fetchall()
    finally:
        con.close()
    return [
        {"id": r[0], "content": r[1], "project": r[2] or "unknown",
         "content_type": r[3] or "unknown", "source": r[4]}
        for r in rows
    ]


def resolve_api_key() -> str:
    key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if key:
        return key.strip()
    # Read piped stdin (op read "...| python ...").
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    raise SystemExit("No API key. Pipe `op read op://Private/grok-content-golem/credential` or set XAI_API_KEY.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", type=int, default=0, help="Smoke: sample this many chunks (×5 variants), measure, stop.")
    ap.add_argument("--run", action="store_true", help="Full run with budget-sized N.")
    ap.add_argument("--n", type=int, default=0, help="Explicit chunk count for --run (overrides auto-size).")
    ap.add_argument("--max-usd", type=float, default=4.20, help="Hard cumulative spend ceiling (USD).")
    ap.add_argument("--avg-usd-per-call", type=float, default=0.0,
                    help="Measured avg $/call from smoke; used to auto-size N for --run.")
    ap.add_argument("--base-url", default="https://api.x.ai/v1")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Backend model sent to the OpenAI-compatible API.")
    ap.add_argument("--tick-usd", type=float, default=DEFAULT_TICK_USD)
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--out", default="/tmp/abcde_enrich.jsonl")
    args = ap.parse_args()

    variants = list(ABCDE_VARIANTS)
    key = resolve_api_key()
    chat_fn = make_http_chat_fn(base_url=args.base_url, api_key=key, model_override=args.model)
    sanitizer = Sanitizer.from_env()

    if args.smoke > 0:
        chunks = sample_chunks(args.smoke, min_chars=args.min_chars)
        print(f"SMOKE: {len(chunks)} chunks × {len(variants)} variants = {len(chunks)*len(variants)} calls", file=sys.stderr)
        t0 = time.time()
        stats = run_batch(chunks, variants, sanitizer, chat_fn, output_path=args.out,
                          max_usd=args.max_usd, tick_usd=args.tick_usd)
        out = stats.as_dict()
        out["wall_seconds"] = round(time.time() - t0, 1)
        out["per_variant"] = {}  # filled below
        print(json.dumps(out, indent=2))
        return

    if args.run:
        if args.n > 0:
            n = args.n
        elif args.avg_usd_per_call > 0:
            # 5 variants per chunk; 85% safety margin on the budget.
            n = max(1, int((args.max_usd * 0.85) / (len(variants) * args.avg_usd_per_call)))
        else:
            raise SystemExit("--run needs --n or --avg-usd-per-call")
        chunks = sample_chunks(n, min_chars=args.min_chars)
        print(f"RUN: N={len(chunks)} chunks × {len(variants)} variants = {len(chunks)*len(variants)} calls "
              f"(max_usd={args.max_usd})", file=sys.stderr)
        t0 = time.time()
        stats = run_batch(chunks, variants, sanitizer, chat_fn, output_path=args.out,
                          max_usd=args.max_usd, tick_usd=args.tick_usd)
        out = stats.as_dict()
        out["wall_seconds"] = round(time.time() - t0, 1)
        out["output_jsonl"] = args.out
        print(json.dumps(out, indent=2))
        return

    raise SystemExit("Pass --smoke N or --run.")


if __name__ == "__main__":
    main()
