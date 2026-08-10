#!/usr/bin/env python3
"""Run the approved full hybrid-taxonomy backlog backfill.

This script is intentionally operational, not a general-purpose CLI. It reuses
the completed 2,000-row sample where parsing succeeded, submits only the
remaining canonical enrichment candidates to Gemini Batch, parses with the
production hybrid parser, and imports successful enrichments into the prod DB.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path = [entry for entry in sys.path if entry != str(SRC)]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import hybrid_backfill_sample as sample  # noqa: E402
from brainlayer.cloud_backfill import (  # noqa: E402
    CHUNKS_PER_JOB,
    build_batch_request_line,
    download_gemini_results,
    estimate_batch_cost_usd,
    get_gemini_batch_state,
    submit_gemini_batch,
)
from brainlayer.paths import get_db_path  # noqa: E402
from brainlayer.pipeline.enrichment import (  # noqa: E402
    build_external_prompt,
    enrichment_version_metadata,
    parse_enrichment,
)
from brainlayer.pipeline.sanitize import Sanitizer  # noqa: E402

RUN_ID = "hybrid-backfill-full-2026-06-17-r1"
SAMPLE_RUN_ID = "hybrid-backfill-sample-2026-06-16-r1"
MODEL = "models/gemini-2.5-flash-lite"
BACKEND = "gemini-batch"
ARTIFACT_DIR = ROOT / "artifacts" / RUN_ID
SAMPLE_DIR = ROOT / "artifacts" / SAMPLE_RUN_ID
COLLAB_PATH = Path("/Users/etanheyman/Gits/orchestrator/collab/2026-06-10-gen16-channel.md")
DB_PATH = Path(get_db_path())

ENRICHMENT_COLUMNS = [
    "summary",
    "tags",
    "importance",
    "intent",
    "primary_symbols",
    "resolved_query",
    "epistemic_level",
    "version_scope",
    "debt_impact",
    "external_deps",
    "key_facts",
    "resolved_queries",
    "sentiment_label",
    "sentiment_score",
    "sentiment_signals",
    "raw_entities_json",
    "enriched_at",
    "enrich_status",
    "enrichment_version",
    "enrichment_model",
    "enrichment_backend",
    "metadata",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def append_collab(line: str) -> None:
    existing = COLLAB_PATH.read_text(encoding="utf-8") if COLLAB_PATH.exists() else ""
    header = "### enrich-36k-backfill"
    if header not in existing:
        COLLAB_PATH.write_text(existing.rstrip() + f"\n\n{header}\n\n{line.rstrip()}\n", encoding="utf-8")
        return
    match = re.search(r"(?m)^### enrich-36k-backfill\s*$", existing)
    if not match:
        COLLAB_PATH.write_text(existing.rstrip() + "\n" + line.rstrip() + "\n", encoding="utf-8")
        return
    insert_at = existing.find("\n", match.end())
    insert_at = len(existing) if insert_at == -1 else insert_at + 1
    if existing[insert_at : insert_at + 1] == "\n":
        insert_at += 1
    COLLAB_PATH.write_text(existing[:insert_at] + line.rstrip() + "\n" + existing[insert_at:], encoding="utf-8")


def set_run_env() -> None:
    os.environ["BRAINLAYER_ENRICHMENT_TAG_MODE"] = "hybrid"
    os.environ["BRAINLAYER_ENRICHMENT_RUN_ID"] = RUN_ID
    os.environ["BRAINLAYER_ENRICHMENT_MODEL_STAMP"] = MODEL
    os.environ["BRAINLAYER_ENRICHMENT_BACKEND_STAMP"] = BACKEND


def safe_job_name(batch_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", batch_name)


def candidate_counts() -> dict[str, int]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=30)
    try:
        cur = conn.cursor()
        raw_ge50 = cur.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE enriched_at IS NULL
              AND enrich_status IS NULL
              AND char_count >= 50
            """
        ).fetchone()[0]
        raw_any = cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE enriched_at IS NULL AND enrich_status IS NULL"
        ).fetchone()[0]
    finally:
        conn.close()
    store = sample.ReadOnlyCandidateStore(DB_PATH)
    try:
        dedup = store.get_enrichment_candidates(limit=1_000_000, min_content_length=50, order="newest")
    finally:
        store.close()
    return {
        "raw_selector_ge50": int(raw_ge50),
        "raw_null_status_any_length": int(raw_any),
        "dedup_ranked_candidates_ge50": len(dedup),
    }


def load_sample_successes() -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    successes: dict[str, dict[str, Any]] = {}
    failed = 0
    total = 0
    for row in read_jsonl(SAMPLE_DIR / "parsed-enrichments.jsonl"):
        total += 1
        parsed = row.get("parsed")
        if parsed and not row.get("error"):
            successes[str(row["chunk_id"])] = {
                "chunk_id": str(row["chunk_id"]),
                "source": "sample-reuse",
                "parsed": parsed,
                "raw_response_text": row.get("raw_response_text") or "",
                "error": None,
            }
        else:
            failed += 1
    return successes, {"sample_rows": total, "sample_parse_success": len(successes), "sample_parse_failed": failed}


def export_remaining() -> dict[str, Any]:
    set_run_env()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "batch-inputs").mkdir(parents=True, exist_ok=True)
    sample_successes, sample_stats = load_sample_successes()

    store = sample.ReadOnlyCandidateStore(DB_PATH)
    try:
        candidates = store.get_enrichment_candidates(limit=1_000_000, min_content_length=50, order="newest")
    finally:
        store.close()

    candidate_ids = {str(row["id"]) for row in candidates}
    reusable_ids = sorted(candidate_ids & set(sample_successes))
    remaining = [row for row in candidates if str(row["id"]) not in sample_successes]

    sanitizer = Sanitizer.from_env()
    sample_mapping = SAMPLE_DIR / "pii_mapping.json"
    if sample_mapping.exists():
        sanitizer.load_mapping(sample_mapping)

    selection_rows: list[dict[str, Any]] = []
    batch_lines: list[dict[str, Any]] = []
    sanitized = 0
    prompt_chars = 0
    for row in remaining:
        chunk = {
            "content": row.get("content") or "",
            "project": row.get("project"),
            "content_type": row.get("content_type"),
            "source": row.get("source"),
            "sender": row.get("sender"),
            "prev_assistant_text": row.get("prev_assistant_text"),
        }
        prompt, sanitize_result = build_external_prompt(chunk, sanitizer)
        if sanitize_result.pii_detected:
            sanitized += 1
        prompt_chars += len(prompt)
        batch_lines.append(build_batch_request_line(str(row["id"]), prompt))
        selection_rows.append(
            {
                "chunk_id": str(row["id"]),
                "content": row.get("content") or "",
                "project": row.get("project"),
                "content_type": row.get("content_type"),
                "source": row.get("source"),
                "sender": row.get("sender"),
                "char_count": row.get("char_count"),
                "created_at": row.get("created_at"),
                "conversation_id": row.get("conversation_id"),
                "position": row.get("position"),
                "source_file": row.get("source_file"),
            }
        )

    batch_paths: list[str] = []
    for index, batch in enumerate(sample.chunked(batch_lines, CHUNKS_PER_JOB), start=1):
        path = ARTIFACT_DIR / "batch-inputs" / f"batch_{index:03d}.jsonl"
        write_jsonl(path, batch)
        batch_paths.append(str(path))

    write_jsonl(ARTIFACT_DIR / "remaining-selection.jsonl", selection_rows)
    write_jsonl(ARTIFACT_DIR / "reused-sample-successes.jsonl", [sample_successes[i] for i in reusable_ids])
    sanitizer.save_mapping(ARTIFACT_DIR / "pii_mapping.json")
    summary = {
        "generated_at": now_iso(),
        "run_id": RUN_ID,
        "sample_run_id": SAMPLE_RUN_ID,
        "db_path": str(DB_PATH),
        "artifact_dir": str(ARTIFACT_DIR),
        "model": MODEL,
        "backend": BACKEND,
        "version_metadata": enrichment_version_metadata(model=MODEL, backend=BACKEND),
        "candidate_counts": candidate_counts(),
        "dedup_ranked_candidate_rows": len(candidates),
        "reused_sample_success_rows": len(reusable_ids),
        "remaining_rows_to_submit": len(remaining),
        "sample_stats": sample_stats,
        "batch_jsonl_files": batch_paths,
        "chunks_per_job": CHUNKS_PER_JOB,
        "pii_sanitized_remaining_prompts": sanitized,
        "avg_prompt_chars": round(prompt_chars / len(batch_lines), 1) if batch_lines else 0,
    }
    write_json(ARTIFACT_DIR / "export-summary.json", summary)
    append_collab(
        "@orc enrichCodex FULL BACKFILL PROGRESS "
        f"({now_iso()}): run `{RUN_ID}` exported current canonical backlog. "
        f"Dedup-ranked candidates {len(candidates)}; reusing {len(reusable_ids)} parsed sample successes; "
        f"submitting {len(remaining)} remaining rows in {len(batch_paths)} Gemini Batch jobs. "
        f"Raw selector ge50={summary['candidate_counts']['raw_selector_ge50']}. "
        "No DB import or live re-enable has happened yet."
    )
    return summary


def submit_jobs() -> dict[str, Any]:
    summary = read_json(ARTIFACT_DIR / "export-summary.json")
    if not summary:
        raise SystemExit("Missing export-summary.json; run export first")
    jobs_path = ARTIFACT_DIR / "jobs.json"
    jobs = read_json(jobs_path, {"jobs": []})
    existing = {job.get("jsonl_path") for job in jobs.get("jobs", [])}
    for jsonl in summary["batch_jsonl_files"]:
        if jsonl in existing:
            continue
        name = submit_gemini_batch(Path(jsonl), model=MODEL, store=None)
        jobs["jobs"].append(
            {
                "batch_name": name,
                "jsonl_path": jsonl,
                "model": MODEL,
                "submitted_at": now_iso(),
                "status": "submitted" if name else "submit_failed",
            }
        )
        write_json(jobs_path, jobs)
    append_collab(
        "@orc enrichCodex FULL BACKFILL PROGRESS "
        f"({now_iso()}): submitted {len(jobs.get('jobs', []))} remaining-backlog Gemini Batch jobs for `{RUN_ID}`. "
        "Will import only after all jobs parse successfully enough to write valid enrichments."
    )
    return jobs


def poll_jobs(poll_interval_seconds: int, timeout_hours: float) -> dict[str, Any]:
    deadline = time.time() + timeout_hours * 3600
    jobs_path = ARTIFACT_DIR / "jobs.json"
    jobs = read_json(jobs_path, {"jobs": []})
    if not jobs.get("jobs"):
        raise SystemExit("Missing jobs.json or no jobs")
    raw_dir = ARTIFACT_DIR / "raw-results"

    while time.time() < deadline:
        pending = 0
        failed: list[dict[str, Any]] = []
        for job in jobs["jobs"]:
            if job.get("status") == "succeeded" and job.get("raw_results_jsonl"):
                continue
            if job.get("status") in {"failed", "cancelled"}:
                failed.append(job)
                continue
            status = get_gemini_batch_state(job["batch_name"])
            job["status"] = status["state"]
            job["raw_state"] = status.get("raw_state")
            job["error"] = status.get("error")
            job["last_checked_at"] = now_iso()
            if status["state"] == "succeeded":
                results = download_gemini_results(status["job"])
                raw_path = raw_dir / f"{safe_job_name(job['batch_name'])}.jsonl"
                write_jsonl(raw_path, results)
                job["raw_results_jsonl"] = str(raw_path)
                job["result_count"] = len(results)
                job["completed_at"] = now_iso()
            elif status["state"] in {"failed", "cancelled"}:
                job["completed_at"] = now_iso()
                failed.append(job)
            else:
                pending += 1
        write_json(jobs_path, jobs)
        if failed:
            append_collab(
                "@orc enrichCodex FULL BACKFILL FAILED "
                f"({now_iso()}): Gemini Batch job failed/cancelled for `{RUN_ID}`: "
                + "; ".join(f"{j.get('batch_name')} {j.get('raw_state')} {j.get('error')}" for j in failed)
                + ". No import/live re-enable performed by this failure path."
            )
            raise SystemExit(2)
        if pending == 0:
            break
        log(f"{pending} Gemini Batch jobs still pending; sleeping {poll_interval_seconds}s")
        time.sleep(poll_interval_seconds)

    combined = combine_raw(jobs_path, ARTIFACT_DIR / "raw-results-combined.jsonl")
    log(f"combined remaining raw rows -> {combined}")
    return jobs


def combine_raw(jobs_path: Path, out_path: Path) -> Path:
    jobs = read_json(jobs_path, {"jobs": []})
    rows: list[dict[str, Any]] = []
    for job in jobs.get("jobs", []):
        raw = job.get("raw_results_jsonl")
        if raw:
            rows.extend(read_jsonl(Path(raw)))
    write_jsonl(out_path, rows)
    return out_path


def extract_response_text(result: dict[str, Any]) -> str:
    return sample.extract_response_text(result)


def analyze_remaining() -> dict[str, Any]:
    set_run_env()
    raw_rows = read_jsonl(ARTIFACT_DIR / "raw-results-combined.jsonl")
    parsed_rows: list[dict[str, Any]] = []
    errors = Counter()
    tags = Counter()
    for result in raw_rows:
        chunk_id = str(result.get("key") or "")
        text = extract_response_text(result)
        parsed = parse_enrichment(text) if text else None
        error = None
        if not chunk_id:
            error = "missing_key"
        elif not text:
            error = "missing_response_text"
        elif not parsed:
            error = "parse_failed"
        if error:
            errors[error] += 1
        else:
            tags.update(parsed.get("tags") or [])
        parsed_rows.append(
            {
                "chunk_id": chunk_id,
                "source": "remaining-batch",
                "parsed": parsed,
                "raw_response_text": text,
                "error": error,
            }
        )
    write_jsonl(ARTIFACT_DIR / "parsed-remaining-enrichments.jsonl", parsed_rows)

    reusable = read_jsonl(ARTIFACT_DIR / "reused-sample-successes.jsonl")
    importable = [row for row in reusable if row.get("parsed") and not row.get("error")]
    importable.extend(row for row in parsed_rows if row.get("parsed") and not row.get("error"))
    write_jsonl(ARTIFACT_DIR / "importable-enrichments.jsonl", importable)
    success = [row for row in parsed_rows if row.get("parsed") and not row.get("error")]
    summary = {
        "generated_at": now_iso(),
        "run_id": RUN_ID,
        "remaining_raw_result_rows": len(raw_rows),
        "remaining_parse_success": len(success),
        "remaining_parse_failed": len(raw_rows) - len(success),
        "remaining_parse_success_rate": round(len(success) / len(raw_rows), 4) if raw_rows else 0,
        "errors": dict(errors),
        "reused_sample_success_rows": len(reusable),
        "total_importable_rows": len(importable),
        "tag_stats_remaining": {
            "total_tag_assignments": sum(tags.values()),
            "unique_tags": len(tags),
            "singleton_tags": sum(1 for count in tags.values() if count == 1),
            "singleton_tag_rate": round(sum(1 for count in tags.values() if count == 1) / len(tags), 4) if tags else 0,
            "top_tags": dict(tags.most_common(30)),
        },
        "usage_remaining": sum_usage(ARTIFACT_DIR / "jobs.json"),
        "usage_sample_hybrid_reused": sum_usage(SAMPLE_DIR / "jobs.json"),
    }
    write_json(ARTIFACT_DIR / "parse-summary.json", summary)
    return summary


def row_usage(path: Path) -> dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    for row in read_jsonl(path):
        response = row.get("response") if isinstance(row, dict) else {}
        usage = response.get("usageMetadata") if isinstance(response, dict) else {}
        if isinstance(usage, dict):
            input_tokens += int(usage.get("promptTokenCount") or usage.get("prompt_token_count") or 0)
            output_tokens += int(usage.get("candidatesTokenCount") or usage.get("candidates_token_count") or 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def sum_usage(jobs_path: Path) -> dict[str, Any]:
    jobs = read_json(jobs_path, {"jobs": []})
    input_tokens = 0
    output_tokens = 0
    for job in jobs.get("jobs", []):
        raw = job.get("raw_results_jsonl")
        if raw:
            usage = row_usage(Path(raw))
            input_tokens += usage["input_tokens"]
            output_tokens += usage["output_tokens"]
    return {
        "jobs": len(jobs.get("jobs", [])),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(estimate_batch_cost_usd(input_tokens, output_tokens), 6),
    }


def normalize_for_import(parsed: dict[str, Any]) -> dict[str, Any]:
    enrichment = dict(parsed)
    enrichment["enrich_status"] = "success"
    return enrichment


def resolved_query_for(enrichment: dict[str, Any]) -> str | None:
    resolved = enrichment.get("resolved_query")
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()
    queries = enrichment.get("resolved_queries")
    if isinstance(queries, list) and queries:
        first = str(queries[0]).strip()
        return first or None
    return None


def metadata_with_stamp(raw: Any, metadata: dict[str, Any]) -> str:
    if raw:
        try:
            parsed = json.loads(str(raw))
            base = parsed if isinstance(parsed, dict) else {"_previous_metadata_raw": raw}
        except (TypeError, json.JSONDecodeError):
            base = {"_previous_metadata_raw": raw}
    else:
        base = {}
    base.update(metadata)
    return json.dumps(base, ensure_ascii=True, sort_keys=True)


def import_enrichments() -> dict[str, Any]:
    rows = read_jsonl(ARTIFACT_DIR / "importable-enrichments.jsonl")
    if not rows:
        raise SystemExit("No importable enrichments")
    ids = [str(row["chunk_id"]) for row in rows]
    id_set = set(ids)
    if len(id_set) != len(ids):
        raise SystemExit("Duplicate importable chunk ids")

    version = enrichment_version_metadata(model=MODEL, backend=BACKEND)
    stamp = {
        "enrichment_model": MODEL,
        "enrichment_backend": BACKEND,
        "enrichment_version": version["prompt_version"],
        "enrichment_run_id": RUN_ID,
        "taxonomy_git_sha": version["taxonomy_git_sha"],
        "taxonomy_content_sha": version["taxonomy_content_sha"],
        "tag_mode": version["tag_mode"],
    }

    conn = sqlite3.connect(str(DB_PATH), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.row_factory = sqlite3.Row
    before_counts = db_counts(conn)
    rollback_rows: list[dict[str, Any]] = []
    applied = 0
    skipped_not_eligible = 0
    missing = 0
    ts = now_iso()
    try:
        conn.execute("BEGIN IMMEDIATE")
        for row in rows:
            chunk_id = str(row["chunk_id"])
            current = conn.execute(
                f"SELECT id, content_hash, {', '.join(ENRICHMENT_COLUMNS)} FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
            if not current:
                missing += 1
                continue
            rollback_rows.append({key: current[key] for key in current.keys()})
            if current["enriched_at"] is not None or current["enrich_status"] is not None:
                skipped_not_eligible += 1
                continue
            enrichment = normalize_for_import(row["parsed"])
            entities = enrichment.get("entities")
            cursor = conn.execute(
                """
                UPDATE chunks
                SET summary = ?,
                    tags = ?,
                    importance = ?,
                    intent = ?,
                    primary_symbols = ?,
                    resolved_query = ?,
                    epistemic_level = ?,
                    version_scope = ?,
                    debt_impact = ?,
                    external_deps = ?,
                    key_facts = ?,
                    resolved_queries = ?,
                    sentiment_label = ?,
                    sentiment_score = ?,
                    sentiment_signals = ?,
                    raw_entities_json = ?,
                    enriched_at = ?,
                    enrich_status = 'success',
                    enrichment_version = ?,
                    enrichment_model = ?,
                    enrichment_backend = ?,
                    metadata = ?
                WHERE id = ?
                  AND enriched_at IS NULL
                  AND enrich_status IS NULL
                """,
                (
                    enrichment.get("summary"),
                    json.dumps(enrichment.get("tags") or [], ensure_ascii=True),
                    enrichment.get("importance"),
                    enrichment.get("intent"),
                    json.dumps(enrichment.get("primary_symbols") or [], ensure_ascii=True),
                    resolved_query_for(enrichment),
                    enrichment.get("epistemic_level"),
                    enrichment.get("version_scope"),
                    enrichment.get("debt_impact"),
                    json.dumps(enrichment.get("external_deps") or [], ensure_ascii=True),
                    json.dumps(enrichment.get("key_facts") or [], ensure_ascii=True),
                    json.dumps(enrichment.get("resolved_queries") or [], ensure_ascii=True),
                    enrichment.get("sentiment_label"),
                    enrichment.get("sentiment_score"),
                    json.dumps(enrichment.get("sentiment_signals") or [], ensure_ascii=True),
                    json.dumps(entities if isinstance(entities, list) else [], ensure_ascii=True),
                    ts,
                    version["prompt_version"],
                    MODEL,
                    BACKEND,
                    metadata_with_stamp(current["metadata"], stamp),
                    chunk_id,
                ),
            )
            if cursor.rowcount == 1:
                applied += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        write_jsonl(ARTIFACT_DIR / "rollback-target-rows.jsonl", rollback_rows)

    after_counts = db_counts(conn)
    verification = verify_import(conn, id_set)
    conn.close()
    summary = {
        "generated_at": now_iso(),
        "run_id": RUN_ID,
        "import_timestamp": ts,
        "requested_importable_rows": len(rows),
        "applied_rows": applied,
        "skipped_not_eligible": skipped_not_eligible,
        "missing_rows": missing,
        "db_counts_before": before_counts,
        "db_counts_after": after_counts,
        "verification": verification,
        "version_metadata": version,
    }
    write_json(ARTIFACT_DIR / "import-summary.json", summary)
    return summary


def db_counts(conn: sqlite3.Connection) -> dict[str, int]:
    cur = conn.cursor()
    return {
        "raw_selector_ge50": cur.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE enriched_at IS NULL
              AND enrich_status IS NULL
              AND char_count >= 50
            """
        ).fetchone()[0],
        "success": cur.execute("SELECT COUNT(*) FROM chunks WHERE enrich_status = 'success'").fetchone()[0],
        "summary_present": cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE summary IS NOT NULL AND trim(summary) != ''"
        ).fetchone()[0],
        "run_id_metadata": cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE metadata LIKE ?", (f'%"{RUN_ID}"%',)
        ).fetchone()[0],
        "model_stamped": cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE enrichment_model = ? AND enrichment_backend = ?",
            (MODEL, BACKEND),
        ).fetchone()[0],
    }


def verify_import(conn: sqlite3.Connection, ids: set[str]) -> dict[str, Any]:
    placeholders = ",".join("?" for _ in ids)
    if not ids:
        return {}
    row = conn.execute(
        f"""
        SELECT
          COUNT(*) AS rows_seen,
          SUM(CASE WHEN enrich_status = 'success' AND enriched_at IS NOT NULL THEN 1 ELSE 0 END) AS success_rows,
          SUM(CASE WHEN summary IS NOT NULL AND trim(summary) != '' THEN 1 ELSE 0 END) AS summary_rows,
          SUM(CASE WHEN tags IS NOT NULL AND json_valid(tags) = 1 THEN 1 ELSE 0 END) AS valid_tags_rows,
          SUM(CASE WHEN enrichment_version = 'r82-hybrid-taxonomy' THEN 1 ELSE 0 END) AS version_rows,
          SUM(CASE WHEN enrichment_model = ? AND enrichment_backend = ? THEN 1 ELSE 0 END) AS model_backend_rows,
          SUM(CASE WHEN metadata LIKE ? THEN 1 ELSE 0 END) AS metadata_run_rows
        FROM chunks
        WHERE id IN ({placeholders})
        """,
        (MODEL, BACKEND, f'%"{RUN_ID}"%', *sorted(ids)),
    ).fetchone()
    return dict(row)


def plaintext_key_scan(paths: list[Path]) -> dict[str, Any]:
    key = os.environ.get("GOOGLE_API_KEY", "")
    key_hits = 0
    ai_pattern_hits = 0
    google_key_regex = re.compile(r"AIza[0-9A-Za-z_-]{20,}")
    for path in paths:
        if not path.exists() or path.is_dir():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if key and key in text:
            key_hits += 1
        ai_pattern_hits += len(google_key_regex.findall(text))
    result = {
        "paths_scanned": [str(path) for path in paths if path.exists() and not path.is_dir()],
        "actual_google_api_key_hits": key_hits,
        "google_api_key_pattern_hits": ai_pattern_hits,
    }
    write_json(ARTIFACT_DIR / "plaintext-key-scan.json", result)
    return result


def final_report() -> dict[str, Any]:
    export = read_json(ARTIFACT_DIR / "export-summary.json", {})
    parse = read_json(ARTIFACT_DIR / "parse-summary.json", {})
    imp = read_json(ARTIFACT_DIR / "import-summary.json", {})
    scan = plaintext_key_scan(
        [
            ARTIFACT_DIR / "raw-results-combined.jsonl",
            ARTIFACT_DIR / "parsed-remaining-enrichments.jsonl",
            ARTIFACT_DIR / "importable-enrichments.jsonl",
            SAMPLE_DIR / "raw-results-combined.jsonl",
            SAMPLE_DIR / "parsed-enrichments.jsonl",
        ]
    )
    remaining_usage = parse.get("usage_remaining", {})
    sample_usage = parse.get("usage_sample_hybrid_reused", {})
    total_cost = round(
        float(remaining_usage.get("estimated_cost_usd") or 0)
        + float(sample_usage.get("estimated_cost_usd") or 0),
        6,
    )
    final = {
        "generated_at": now_iso(),
        "run_id": RUN_ID,
        "dedup_ranked_candidate_rows": export.get("dedup_ranked_candidate_rows"),
        "reused_sample_success_rows": export.get("reused_sample_success_rows"),
        "remaining_rows_submitted": export.get("remaining_rows_to_submit"),
        "remaining_parse_success": parse.get("remaining_parse_success"),
        "remaining_parse_failed": parse.get("remaining_parse_failed"),
        "total_importable_rows": parse.get("total_importable_rows"),
        "applied_rows": imp.get("applied_rows"),
        "db_counts_before": imp.get("db_counts_before"),
        "db_counts_after": imp.get("db_counts_after"),
        "plaintext_key_scan": scan,
        "cost": {
            "remaining_incremental": remaining_usage,
            "sample_hybrid_reused_prior_spend": sample_usage,
            "total_hybrid_generation_cost_including_reused_sample": total_cost,
        },
    }
    write_json(ARTIFACT_DIR / "final-summary.json", final)
    append_collab(
        "@orc enrichCodex FULL BACKFILL IMPORTED "
        f"({now_iso()}): run `{RUN_ID}` imported production hybrid enrichments. "
        f"Dedup-ranked candidates {final['dedup_ranked_candidate_rows']}; reused sample successes {final['reused_sample_success_rows']}; "
        f"submitted remaining {final['remaining_rows_submitted']}; remaining parse {final['remaining_parse_success']}/"
        f"{final['remaining_rows_submitted']} (failed {final['remaining_parse_failed']}); "
        f"total importable {final['total_importable_rows']}; DB applied {final['applied_rows']}. "
        f"Backlog raw ge50 before/after {imp.get('db_counts_before', {}).get('raw_selector_ge50')} -> "
        f"{imp.get('db_counts_after', {}).get('raw_selector_ge50')}; success rows "
        f"{imp.get('db_counts_before', {}).get('success')} -> {imp.get('db_counts_after', {}).get('success')}. "
        f"Cost: incremental remaining ${remaining_usage.get('estimated_cost_usd', 0):.6f}; "
        f"reused sample hybrid prior spend ${sample_usage.get('estimated_cost_usd', 0):.6f}; "
        f"hybrid generation total ${total_cost:.6f}. "
        f"Plaintext key scan actual-key hits={scan['actual_google_api_key_hits']} pattern hits={scan['google_api_key_pattern_hits']}. "
        f"Artifacts `{ARTIFACT_DIR}`. Next: re-enable live enrichment."
    )
    return final


def run_all(args: argparse.Namespace) -> None:
    export_remaining()
    submit_jobs()
    poll_jobs(args.poll_interval_seconds, args.timeout_hours)
    analyze_remaining()
    import_enrichments()
    final_report()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["export", "submit", "poll", "analyze", "import", "report", "run-all"],
    )
    parser.add_argument("--poll-interval-seconds", type=int, default=180)
    parser.add_argument("--timeout-hours", type=float, default=24)
    args = parser.parse_args()

    if args.command == "export":
        export_remaining()
    elif args.command == "submit":
        submit_jobs()
    elif args.command == "poll":
        poll_jobs(args.poll_interval_seconds, args.timeout_hours)
    elif args.command == "analyze":
        analyze_remaining()
    elif args.command == "import":
        import_enrichments()
    elif args.command == "report":
        final_report()
    elif args.command == "run-all":
        run_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
