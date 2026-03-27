#!/usr/bin/env python3
"""Paced batch submission — submits one batch at a time with delays to avoid 429s.
Bypasses VectorStore entirely to avoid DB lock issues with BrainBar.

Usage: GOOGLE_API_KEY=... python3 scripts/batch_submit_paced.py [--delay 45] [--max-retries 5]
"""
import glob, json, os, sys, time
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("pip install google-generativeai"); sys.exit(1)

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY required"); sys.exit(1)

genai.configure(api_key=API_KEY)

DELAY = int(sys.argv[sys.argv.index("--delay") + 1]) if "--delay" in sys.argv else 45
MAX_RETRIES = int(sys.argv[sys.argv.index("--max-retries") + 1]) if "--max-retries" in sys.argv else 10
MODEL = "gemini-2.5-flash-lite"

# Track what we've already submitted this run
STATE_FILE = Path(__file__).parent / "backfill_data" / ".paced_state.json"
submitted = {}
if STATE_FILE.exists():
    submitted = json.loads(STATE_FILE.read_text())

# Find all batch JSONL files
batch_files = sorted(glob.glob(str(Path(__file__).parent / "backfill_data" / "batch_*.jsonl")))
print(f"Total batch files: {len(batch_files)}")
print(f"Already submitted (this run): {len(submitted)}")
print(f"Delay between submissions: {DELAY}s")
print(f"Max retries per batch: {MAX_RETRIES}")
print(f"Model: {MODEL}")
print()

created = 0
skipped = 0
failed = 0

for i, fpath in enumerate(batch_files):
    fname = Path(fpath).name
    if fname in submitted:
        skipped += 1
        continue

    # Count chunks in this file
    with open(fpath) as f:
        chunks = sum(1 for _ in f)

    print(f"[{i+1}/{len(batch_files)}] {fname} ({chunks} chunks)")

    # Upload file
    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Uploading...", end="", flush=True)
            uploaded = genai.upload_file(fpath)
            print(f" ok ({uploaded.name})")
            break
        except Exception as e:
            wait = min(30 * (attempt + 1), 300)
            print(f" 429, waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
    else:
        print(f"  FAILED upload after {MAX_RETRIES} retries, skipping")
        failed += 1
        continue

    # Create batch job
    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Creating batch job...", end="", flush=True)
            job = genai.batches.create(
                model=f"models/{MODEL}",
                src=uploaded.uri,
                config={"display_name": fname},
            )
            print(f" ok ({job.name}, {job.state})")
            submitted[fname] = {"job_name": job.name, "chunks": chunks, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            STATE_FILE.write_text(json.dumps(submitted, indent=2))
            created += 1
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = min(60 * (attempt + 1), 600)
                print(f" 429, waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                print(f" ERROR: {e}")
                failed += 1
                break
    else:
        print(f"  FAILED create after {MAX_RETRIES} retries, skipping")
        failed += 1
        continue

    # Pace ourselves
    if i < len(batch_files) - 1:
        print(f"  Waiting {DELAY}s before next...")
        time.sleep(DELAY)

print(f"\nDone! Created: {created}, Skipped (already done): {skipped}, Failed: {failed}")
print(f"State saved to {STATE_FILE}")
