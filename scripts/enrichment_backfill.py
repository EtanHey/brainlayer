"""Backfill enrichment: run faceted tag prompt on chunks missing faceted tags.

Usage:
  GOOGLE_API_KEY=... python3 scripts/enrichment_backfill.py [--limit N] [--test]

  --test: run on 10 chunks, print results, don't commit to DB
  --limit N: process N chunks (default: all unfaceted)
"""

import warnings

warnings.warn(
    "scripts/enrichment_backfill.py is deprecated. Use 'brainlayer enrich --mode realtime' or brain_digest mode='enrich' instead.",
    DeprecationWarning,
    stacklevel=2,
)

import json
import os
import sys
import time
import sqlite3
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY required")
    sys.exit(1)

from google import genai

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash-lite"  # No thinking by design. Fast, cheap, good for classification.
DB_PATH = str(Path.home() / ".local/share/brainlayer/brainlayer.db")

PROMPT = """You are a knowledge base tagger for a personal multi-project development knowledge base. Your job is to identify WHAT SPECIFIC THING each chunk is about — not the kind of work being done.

## Critical distinction

GOOD tags describe the SUBJECT: "brainlayer-search-quality", "6pm-confirmation-flow", "importance-calibration"
BAD tags describe the FORMAT: "typescript", "debugging", "code-review", "feature-dev"

Ask yourself: "If someone searches for this topic in 6 months, what words would they use?" Tag with THOSE words.

## Output schema (JSON)

Return a JSON object with these fields in this exact order:

- **a_reasoning** (string): 1-2 sentences explaining what specific subject this chunk discusses.
- **b_topics** (string[]): 1-3 object tags — specific, hyphenated, 2-4 words.
- **c_activity** (string): Exactly one of: act:debugging, act:implementing, act:designing, act:reviewing, act:researching, act:planning, act:configuring, act:refactoring, act:testing, act:learning
- **d_domain** (string[]): 0-3 technology domains from: dom:typescript, dom:python, dom:swift, dom:sql, dom:react, dom:convex, dom:supabase, dom:mcp, dom:vertex-ai, dom:ollama, dom:mlx, dom:git, dom:telegram, dom:whatsapp, dom:macos, dom:cli, dom:css, dom:html, dom:docker, dom:railway, dom:linear, dom:obsidian. Empty array if no specific technology.
- **e_confidence** (number): 0.0-1.0 confidence in your tagging. Below 0.5 = low-content chunk.

## Now tag this chunk:

{chunk_content}"""


def get_unfaceted_chunks(db, limit=None):
    """Get chunks that don't have faceted tags yet, newest first."""
    sql = """
        SELECT rowid, id, content, source, tags
        FROM chunks
        WHERE (tags NOT LIKE '%dom:%' AND tags NOT LIKE '%act:%') OR tags IS NULL
        ORDER BY rowid DESC
    """
    if limit:
        sql += f" LIMIT {limit}"
    return db.execute(sql).fetchall()


def enrich_chunk(content):
    """Call Gemini to get faceted tags for a chunk."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=PROMPT.replace("{chunk_content}", content[:2000]),
            config={
                "response_mime_type": "application/json",
            }
        )
        return json.loads(response.text)
    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"  API error: {e}")
        return None


def merge_tags(existing_tags, new_result):
    """Merge new faceted tags with existing tags."""
    existing = []
    if existing_tags:
        try:
            existing = json.loads(existing_tags)
            if not isinstance(existing, list):
                existing = [existing_tags]
        except json.JSONDecodeError:
            existing = [existing_tags] if existing_tags else []

    # Remove old activity/domain tags if present
    existing = [t for t in existing if not t.startswith("dom:") and not t.startswith("act:")]

    new_tags = new_result.get("b_topics", []) + [new_result.get("c_activity", "")] + new_result.get("d_domain", [])
    new_tags = [t for t in new_tags if t]  # remove empties

    merged = list(set(existing + new_tags))
    return json.dumps(merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test on 10 chunks, don't write to DB")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to process")
    args = parser.parse_args()

    if args.test:
        args.limit = 10

    db = sqlite3.connect(DB_PATH, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=10000")

    chunks = get_unfaceted_chunks(db, args.limit)
    total = len(chunks)
    print(f"Chunks to process: {total}", flush=True)

    if total == 0:
        print("Nothing to enrich!")
        return

    done = 0
    errors = 0
    t0 = time.time()

    for rowid, chunk_id, content, source, existing_tags in chunks:
        result = enrich_chunk(content)

        if result and "b_topics" in result:
            merged = merge_tags(existing_tags, result)
            confidence = result.get("e_confidence", 0)

            if not args.test:
                db.execute(
                    "UPDATE chunks SET tags = ?, tag_confidence = ? WHERE rowid = ?",
                    (merged, confidence, rowid)
                )

            done += 1
            if done % 50 == 0:
                if not args.test:
                    db.commit()
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate / 60 if rate > 0 else 0
                print(f"  [{done}/{total}] {rate:.1f}/sec, {eta:.0f}min left", flush=True)

            if args.test:
                print(f"\n  Chunk {done}: {content[:80]}...")
                print(f"  Topics: {result.get('b_topics', [])}")
                print(f"  Activity: {result.get('c_activity', '')}")
                print(f"  Domain: {result.get('d_domain', [])}")
                print(f"  Confidence: {result.get('e_confidence', 0)}")
        else:
            errors += 1
            if args.test:
                print(f"\n  Chunk FAILED: {content[:60]}... → {result}")

        # Flash-Lite: 30 RPM free tier, 2000 RPM paid. Minimal sleep.
        time.sleep(0.1)

    if not args.test:
        db.commit()

    elapsed = time.time() - t0
    print(f"\nDone: {done}/{total} enriched, {errors} errors, {elapsed:.0f}s ({done/elapsed:.1f}/sec)", flush=True)
    db.close()


if __name__ == "__main__":
    main()
