#!/usr/bin/env python3
"""Poll Vertex AI batch jobs and import results to BrainLayer DB.

Usage:
    python3 scripts/vertex_poll_import.py
    python3 scripts/vertex_poll_import.py --status   # Just check status, don't import
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.vector_store import VectorStore
from brainlayer.paths import DEFAULT_DB_PATH
from brainlayer.pipeline.enrichment import parse_enrichment

# ── Config ──────────────────────────────────────────────────────────────

DB_PATH = DEFAULT_DB_PATH
JOBS_FILE = Path(__file__).resolve().parent / "backfill_data" / "vertex_jobs.json"
GCS_OUTPUT = "gs://brainlayer-enrichment-batch/output"
GSUTIL = Path.home() / "google-cloud-sdk" / "bin" / "gsutil"
POLL_INTERVAL = 300  # 5 minutes
MAX_POLLS = 120  # 10 hours max

PROJECT = "angular-pipe-358301"
LOCATION = "us-central1"


def get_client():
    from google import genai
    return genai.Client(vertexai=True, project=PROJECT, location=LOCATION)


def check_all_jobs(client, job_ids):
    """Check status of all jobs. Returns dict of {job_id: state}."""
    states = {}
    for jid in job_ids:
        name = f"projects/58446144174/locations/{LOCATION}/batchPredictionJobs/{jid}"
        try:
            batch = client.batches.get(name=name)
            states[jid] = {
                "state": str(batch.state),
                "job": batch,
            }
        except Exception as e:
            states[jid] = {"state": "ERROR", "error": str(e)}
    return states


def download_results(job_id):
    """Download batch results from GCS output directory."""
    # Vertex AI writes results to output/prediction-model-*-of-*
    # List all output files for this job
    result = subprocess.run(
        [str(GSUTIL), "ls", f"{GCS_OUTPUT}/"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  Error listing GCS: {result.stderr[:200]}")
        return []

    # Find files related to this job
    # Vertex AI creates output in the dest directory with the job name
    all_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    print(f"  Found {len(all_files)} files in output dir")

    # Download all JSONL results
    results = []
    local_dir = Path(__file__).resolve().parent / "backfill_data" / "vertex_output"
    local_dir.mkdir(exist_ok=True)

    for gcs_file in all_files:
        if not gcs_file.endswith("/"):  # Skip directories
            local_path = local_dir / gcs_file.split("/")[-1]
            if not local_path.exists():
                subprocess.run(
                    [str(GSUTIL), "cp", gcs_file, str(local_path)],
                    capture_output=True
                )

            if local_path.exists():
                with open(local_path) as f:
                    for line in f:
                        try:
                            results.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue

    return results


def import_results_to_db(store, results):
    """Import batch results into BrainLayer DB."""
    success = 0
    failed = 0
    skipped = 0
    cursor = store.conn.cursor()

    for result in results:
        chunk_id = result.get("key")
        if not chunk_id:
            failed += 1
            continue

        # Skip already enriched
        existing = list(cursor.execute(
            "SELECT enriched_at FROM chunks WHERE id = ?", [chunk_id]
        ))
        if existing and existing[0][0] is not None:
            skipped += 1
            continue

        # Extract response text from Vertex AI batch output
        response_text = None
        try:
            response = result.get("response", result.get("output", {}))
            if isinstance(response, dict):
                candidates = response.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        response_text = parts[0].get("text", "")
            elif isinstance(response, str):
                response_text = response
        except (KeyError, IndexError, TypeError):
            pass

        if not response_text:
            failed += 1
            continue

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
            success += 1
        else:
            failed += 1

        if (success + failed + skipped) % 5000 == 0:
            print(f"  Progress: {success} ok, {failed} fail, {skipped} skip")

    return {"success": success, "failed": failed, "skipped": skipped}


def notify(title, body):
    """Send Telegram notification."""
    try:
        subprocess.run(["notify", title, body], capture_output=True, timeout=10)
    except Exception:
        pass


def main():
    status_only = "--status" in sys.argv

    with open(JOBS_FILE) as f:
        jobs_data = json.load(f)

    job_ids = [j["id"] for j in jobs_data["jobs"]]
    print(f"Tracking {len(job_ids)} batch jobs")

    client = get_client()
    store = VectorStore(DB_PATH)
    completed_jobs = set()
    total_imported = {"success": 0, "failed": 0, "skipped": 0}

    for poll_num in range(MAX_POLLS):
        states = check_all_jobs(client, job_ids)

        # Summary
        pending = sum(1 for s in states.values() if "PENDING" in s["state"])
        running = sum(1 for s in states.values() if "RUNNING" in s["state"])
        succeeded = sum(1 for s in states.values() if "SUCCEEDED" in s["state"])
        failed = sum(1 for s in states.values() if "FAILED" in s["state"])

        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now}] Poll #{poll_num + 1}: {pending} pending, {running} running, {succeeded} done, {failed} failed")

        if status_only:
            for jid, info in states.items():
                state = info["state"]
                short_id = jid[-6:]
                if "SUCCEEDED" in state:
                    job = info["job"]
                    cs = getattr(job, "completion_stats", None)
                    ok = getattr(cs, "successful_count", 0) if cs else 0
                    fail_count = getattr(cs, "failed_count", 0) if cs else 0
                    print(f"  {short_id}: SUCCEEDED (ok={ok}, fail={fail_count})")
                else:
                    print(f"  {short_id}: {state}")
            return

        # Process newly completed jobs
        for jid, info in states.items():
            if jid in completed_jobs:
                continue

            if "SUCCEEDED" in info["state"]:
                print(f"\n  Job {jid[-6:]} SUCCEEDED — downloading results...")
                results = download_results(jid)
                if results:
                    print(f"  Importing {len(results)} results...")
                    counts = import_results_to_db(store, results)
                    for k in total_imported:
                        total_imported[k] += counts[k]
                    print(f"  Imported: {counts['success']} ok, {counts['failed']} fail, {counts['skipped']} skip")
                completed_jobs.add(jid)

            elif "FAILED" in info["state"]:
                err = info.get("error", "unknown")
                job = info.get("job")
                if job and hasattr(job, "error"):
                    err = str(job.error)
                print(f"  Job {jid[-6:]} FAILED: {err[:200]}")
                completed_jobs.add(jid)

        # Check if all done
        all_done = len(completed_jobs) >= len(job_ids)
        if all_done:
            break

        # Wait before next poll
        print(f"  Waiting {POLL_INTERVAL}s before next poll...")
        time.sleep(POLL_INTERVAL)

    # Final summary
    stats = store.get_enrichment_stats()
    print(f"\n{'='*60}")
    print("VERTEX AI BATCH ENRICHMENT COMPLETE")
    print(f"  Imported: {total_imported['success']:,} ok, {total_imported['failed']:,} fail, {total_imported['skipped']:,} skip")
    print(f"  Enrichment: {stats['enriched']:,}/{stats['total_chunks']:,} ({stats['percent']}%)")
    print(f"{'='*60}")

    notify("Enrichment Done", f"Imported {total_imported['success']:,} chunks. DB at {stats['percent']}% enriched.")
    store.close()

    # Run brain-export for dashboard 3D view
    if not status_only and total_imported["success"] > 0:
        print("\nRunning brain-export for dashboard...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "brainlayer.cli", "brain-export"],
                capture_output=True, text=True, timeout=600,
                cwd=str(Path(__file__).resolve().parent.parent)
            )
            if result.returncode == 0:
                print(f"  Brain export done: {result.stdout.strip()[-200:]}")
                notify("Brain Export Done", "Dashboard brain view regenerated with enriched data.")
            else:
                print(f"  Brain export failed: {result.stderr[:200]}")
                notify("Brain Export Failed", f"Error: {result.stderr[:100]}")
        except Exception as e:
            print(f"  Brain export error: {e}")
            notify("Brain Export Error", str(e)[:100])


if __name__ == "__main__":
    main()
