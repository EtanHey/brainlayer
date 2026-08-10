#!/usr/bin/env python3
"""Stage the hybrid-taxonomy 36K backfill through Gemini Batch.

This script intentionally does not write to the BrainLayer DB. It exports a
diverse source-of-truth candidate sample, submits Gemini Batch jobs, parses the
completed outputs with the production hybrid parser, and judges a subset before
any full backlog import is attempted.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path = [entry for entry in sys.path if entry != str(SRC)]
sys.path.insert(0, str(SRC))

from brainlayer.cloud_backfill import (  # noqa: E402
    CHUNKS_PER_JOB,
    DEFAULT_BATCH_MODEL,
    _extract_usage_metadata,
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
from brainlayer.session_repo import SessionMixin  # noqa: E402

DEFAULT_RUN_ID = "hybrid-backfill-sample-2026-06-16-r1"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / DEFAULT_RUN_ID
DEFAULT_JUDGE_MODEL = "models/gemini-2.5-flash-lite"
REQUIRED_METADATA_KEYS = {
    "prompt_version",
    "taxonomy_git_sha",
    "taxonomy_content_sha",
    "tag_mode",
    "model",
    "backend",
    "run_id",
}


class ReadOnlyCandidateStore(SessionMixin):
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)

    def _read_cursor(self):
        return self.conn.cursor()

    def close(self) -> None:
        self.conn.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def chunked(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def char_bucket(char_count: int | None) -> str:
    count = int(char_count or 0)
    if count < 250:
        return "short"
    if count < 1000:
        return "medium"
    return "long"


def _stratum_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("content_type") or "unknown"),
        str(row.get("source") or "unknown"),
        char_bucket(row.get("char_count")),
    )


def diverse_sample(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)

    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_stratum_key(row)].append(row)

    for group_rows in groups.values():
        group_rows.sort(key=lambda item: str(item.get("id", "")))
        rng.shuffle(group_rows)

    ordered_groups = sorted(groups.values(), key=lambda group: (-len(group), _stratum_key(group[0])))
    sample: list[dict[str, Any]] = []

    for group_rows in ordered_groups:
        if len(sample) >= limit:
            break
        sample.append(group_rows.pop(0))

    while len(sample) < limit:
        progressed = False
        for group_rows in ordered_groups:
            if len(sample) >= limit:
                break
            if not group_rows:
                continue
            sample.append(group_rows.pop(0))
            progressed = True
        if not progressed:
            break

    sample.sort(key=lambda item: str(item.get("id", "")))
    return sample


def set_run_env(run_id: str, model: str, backend: str = "gemini-batch") -> None:
    os.environ["BRAINLAYER_ENRICHMENT_TAG_MODE"] = "hybrid"
    os.environ["BRAINLAYER_ENRICHMENT_RUN_ID"] = run_id
    os.environ["BRAINLAYER_ENRICHMENT_MODEL_STAMP"] = model
    os.environ["BRAINLAYER_ENRICHMENT_BACKEND_STAMP"] = backend


def build_sanitizer(artifact_dir: Path) -> Sanitizer:
    sanitizer = Sanitizer.from_env()
    mapping_path = artifact_dir / "pii_mapping.json"
    sanitizer.load_mapping(mapping_path)
    return sanitizer


def candidate_counts(db_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        cursor = conn.cursor()
        return {
            "total_chunks": cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "source_of_truth_eligible_ge50": cursor.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE enriched_at IS NULL
                  AND enrich_status IS NULL
                  AND char_count >= 50
                """
            ).fetchone()[0],
            "null_status_null_enriched_any_length": cursor.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE enriched_at IS NULL
                  AND enrich_status IS NULL
                """
            ).fetchone()[0],
            "null_status_null_enriched_ge1": cursor.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE enriched_at IS NULL
                  AND enrich_status IS NULL
                  AND char_count >= 1
                """
            ).fetchone()[0],
            "char_lt50_null_status": cursor.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE enriched_at IS NULL
                  AND enrich_status IS NULL
                  AND char_count < 50
                """
            ).fetchone()[0],
        }
    finally:
        conn.close()


def select_candidates(db_path: Path, *, limit: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    store = ReadOnlyCandidateStore(db_path)
    try:
        candidates = store.get_enrichment_candidates(limit=1_000_000, min_content_length=50, order="newest")
    finally:
        store.close()

    selected = diverse_sample(candidates, limit=limit, seed=seed)
    strata = Counter("|".join(_stratum_key(row)) for row in selected)
    content_types = Counter(str(row.get("content_type") or "unknown") for row in selected)
    sources = Counter(str(row.get("source") or "unknown") for row in selected)
    buckets = Counter(char_bucket(row.get("char_count")) for row in selected)
    summary = {
        "candidate_rows_after_duplicate_ranking": len(candidates),
        "selected_rows": len(selected),
        "seed": seed,
        "limit": limit,
        "content_type_counts": dict(content_types.most_common()),
        "source_counts": dict(sources.most_common()),
        "char_bucket_counts": dict(buckets.most_common()),
        "strata_count": len(strata),
        "top_strata": dict(strata.most_common(20)),
    }
    return selected, summary


def export_sample(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.artifact_dir)
    batch_dir = artifact_dir / "batch-inputs"
    set_run_env(args.run_id, args.model)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.db_path)
    counts = candidate_counts(db_path)
    selected, sample_summary = select_candidates(db_path, limit=args.limit, seed=args.seed)
    sanitizer = build_sanitizer(artifact_dir)

    selection_path = artifact_dir / "selection-sample.jsonl"
    selection_rows: list[dict[str, Any]] = []
    batch_lines: list[dict[str, Any]] = []
    sanitized_count = 0
    prompt_chars = 0

    for row in selected:
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
            sanitized_count += 1
        prompt_chars += len(prompt)
        batch_lines.append(build_batch_request_line(str(row["id"]), prompt))
        selection_rows.append(
            {
                "chunk_id": row["id"],
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
                "stratum": "|".join(_stratum_key(row)),
            }
        )

    write_jsonl(selection_path, selection_rows)
    sanitizer.save_mapping(artifact_dir / "pii_mapping.json")

    batch_paths: list[str] = []
    for index, batch in enumerate(chunked(batch_lines, CHUNKS_PER_JOB), start=1):
        path = batch_dir / f"batch_{index:03d}.jsonl"
        write_jsonl(path, batch)
        batch_paths.append(str(path))

    summary = {
        "generated_at": now_iso(),
        "run_id": args.run_id,
        "db_path": str(db_path),
        "artifact_dir": str(artifact_dir),
        "model": args.model,
        "backend": "gemini-batch",
        "version_metadata": enrichment_version_metadata(),
        "candidate_counts": counts,
        "sample": sample_summary,
        "selection_jsonl": str(selection_path),
        "batch_jsonl_files": batch_paths,
        "chunks_per_job": CHUNKS_PER_JOB,
        "pii_sanitized_chunks": sanitized_count,
        "avg_prompt_chars": round(prompt_chars / len(batch_lines), 1) if batch_lines else 0,
    }
    write_json(artifact_dir / "export-summary.json", summary)
    return summary


def submit_jobs(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.artifact_dir)
    export_summary = read_json(artifact_dir / "export-summary.json")
    if not export_summary:
        raise SystemExit("Missing export-summary.json; run export first")

    jobs_path = artifact_dir / "jobs.json"
    jobs = read_json(jobs_path, {"jobs": []})
    existing_files = {job.get("jsonl_path") for job in jobs.get("jobs", [])}

    for jsonl in export_summary["batch_jsonl_files"]:
        if jsonl in existing_files:
            continue
        batch_name = submit_gemini_batch(Path(jsonl), model=args.model, store=None)
        jobs["jobs"].append(
            {
                "batch_name": batch_name,
                "jsonl_path": jsonl,
                "model": args.model,
                "submitted_at": now_iso(),
                "status": "submitted" if batch_name else "submit_failed",
            }
        )
        write_json(jobs_path, jobs)

    return jobs


def safe_job_name(batch_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", batch_name)


def poll_jobs(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.artifact_dir)
    jobs_path = artifact_dir / "jobs.json"
    jobs = read_json(jobs_path, {"jobs": []})
    if not jobs.get("jobs"):
        raise SystemExit("Missing jobs.json or no submitted jobs")

    raw_dir = artifact_dir / "raw-results"
    deadline = time.time() + args.timeout_hours * 3600
    interval = args.poll_interval_seconds

    while time.time() < deadline:
        pending = 0
        for job in jobs["jobs"]:
            batch_name = job.get("batch_name")
            if not batch_name or job.get("status") in {"succeeded", "failed", "cancelled"}:
                continue
            status = get_gemini_batch_state(batch_name)
            job["status"] = status["state"]
            job["raw_state"] = status.get("raw_state")
            job["error"] = status.get("error")
            job["last_checked_at"] = now_iso()
            if status["state"] == "succeeded":
                usage = _extract_usage_metadata(status["job"])
                job.update(usage)
                results = download_gemini_results(status["job"])
                raw_path = raw_dir / f"{safe_job_name(batch_name)}.jsonl"
                write_jsonl(raw_path, results)
                job["raw_results_jsonl"] = str(raw_path)
                job["result_count"] = len(results)
                job["completed_at"] = now_iso()
            elif status["state"] in {"failed", "cancelled"}:
                job["completed_at"] = now_iso()
            else:
                pending += 1
        write_json(jobs_path, jobs)
        if pending == 0:
            break
        print(f"{pending} Gemini Batch jobs still pending; sleeping {interval}s", flush=True)
        time.sleep(interval)

    combine_raw_results(artifact_dir)
    return jobs


def combine_raw_results(artifact_dir: Path) -> Path:
    jobs = read_json(artifact_dir / "jobs.json", {"jobs": []})
    combined_path = artifact_dir / "raw-results-combined.jsonl"
    rows: list[dict[str, Any]] = []
    for job in jobs.get("jobs", []):
        raw_path = job.get("raw_results_jsonl")
        if raw_path:
            rows.extend(read_jsonl(Path(raw_path)))
    write_jsonl(combined_path, rows)
    return combined_path


def extract_response_text(result: dict[str, Any]) -> str:
    response = result.get("response", result.get("output", {}))
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        return ""
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return ""
    return str(parts[0].get("text") or "")


def analyze_results(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.artifact_dir)
    set_run_env(args.run_id, args.model)
    combined_path = combine_raw_results(artifact_dir)
    raw_rows = read_jsonl(combined_path)
    selection = {row["chunk_id"]: row for row in read_jsonl(artifact_dir / "selection-sample.jsonl")}
    jobs = read_json(artifact_dir / "jobs.json", {"jobs": []})

    parsed_rows: list[dict[str, Any]] = []
    errors = Counter()
    field_presence = Counter()
    tag_counter: Counter[str] = Counter()
    metadata_ok = 0

    for result in raw_rows:
        chunk_id = str(result.get("key") or "")
        response_text = extract_response_text(result)
        parsed = parse_enrichment(response_text) if response_text else None
        error = None
        if not chunk_id:
            error = "missing_key"
        elif not response_text:
            error = "missing_response_text"
        elif not parsed:
            error = "parse_failed"
        if error:
            errors[error] += 1
        else:
            for field in parsed.keys():
                field_presence[field] += 1
            tags = parsed.get("tags") if isinstance(parsed, dict) else []
            if isinstance(tags, list):
                tag_counter.update(str(tag) for tag in tags)
            metadata = parsed.get("enrichment_metadata") if isinstance(parsed, dict) else None
            if isinstance(metadata, dict) and REQUIRED_METADATA_KEYS.issubset(metadata.keys()):
                metadata_ok += 1
        parsed_rows.append(
            {
                "chunk_id": chunk_id,
                "source": selection.get(chunk_id, {}),
                "parsed": parsed,
                "raw_response_text": response_text,
                "error": error,
            }
        )

    parsed_path = artifact_dir / "parsed-enrichments.jsonl"
    write_jsonl(parsed_path, parsed_rows)

    success_rows = [row for row in parsed_rows if row["parsed"]]
    unique_tags = len(tag_counter)
    singleton_tags = sum(1 for count in tag_counter.values() if count == 1)
    input_tokens = sum(int(job.get("input_tokens") or 0) for job in jobs.get("jobs", []))
    output_tokens = sum(int(job.get("output_tokens") or 0) for job in jobs.get("jobs", []))
    batch_cost = estimate_batch_cost_usd(input_tokens, output_tokens)

    summary = {
        "generated_at": now_iso(),
        "run_id": args.run_id,
        "model": args.model,
        "backend": "gemini-batch",
        "raw_result_rows": len(raw_rows),
        "parse_success": len(success_rows),
        "parse_failed": len(raw_rows) - len(success_rows),
        "parse_success_rate": round(len(success_rows) / len(raw_rows), 4) if raw_rows else 0,
        "errors": dict(errors),
        "field_presence": dict(field_presence),
        "metadata_complete_rows": metadata_ok,
        "metadata_complete_rate": round(metadata_ok / len(success_rows), 4) if success_rows else 0,
        "tag_stats": {
            "total_tag_assignments": sum(tag_counter.values()),
            "unique_tags": unique_tags,
            "singleton_tags": singleton_tags,
            "singleton_tag_rate": round(singleton_tags / unique_tags, 4) if unique_tags else 0,
            "avg_tags_per_success": round(sum(tag_counter.values()) / len(success_rows), 3) if success_rows else 0,
            "top_tags": dict(tag_counter.most_common(30)),
        },
        "distributions": {
            "intent": dict(Counter(row["parsed"].get("intent") for row in success_rows if row["parsed"].get("intent"))),
            "epistemic_level": dict(
                Counter(row["parsed"].get("epistemic_level") for row in success_rows if row["parsed"].get("epistemic_level"))
            ),
            "debt_impact": dict(
                Counter(row["parsed"].get("debt_impact") for row in success_rows if row["parsed"].get("debt_impact"))
            ),
            "sentiment_label": dict(
                Counter(row["parsed"].get("sentiment_label") for row in success_rows if row["parsed"].get("sentiment_label"))
            ),
        },
        "batch_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(batch_cost, 6),
            "pricing_source": "brainlayer.cloud_backfill Gemini Batch constants",
        },
        "parsed_jsonl": str(parsed_path),
    }
    write_json(artifact_dir / "analysis-summary.json", summary)
    return summary


def judge_prompt(source: dict[str, Any], parsed: dict[str, Any]) -> str:
    fields = {
        "summary": parsed.get("summary"),
        "tags": parsed.get("tags"),
        "importance": parsed.get("importance"),
        "intent": parsed.get("intent"),
        "epistemic_level": parsed.get("epistemic_level"),
        "debt_impact": parsed.get("debt_impact"),
        "sentiment_label": parsed.get("sentiment_label"),
        "resolved_query": parsed.get("resolved_query"),
    }
    return (
        "Judge this BrainLayer enrichment against the source chunk. "
        "Score each field from 1 to 5 where 5 is accurate, specific, and retrieval-useful. "
        "Return strict JSON with keys field_scores, overall, issues.\n\n"
        f"SOURCE_METADATA:\n{json.dumps({k: source.get(k) for k in ['project', 'content_type', 'source', 'char_count']}, sort_keys=True)}\n\n"
        f"SOURCE_CHUNK:\n{source.get('sanitized_content', '')[:6000]}\n\n"
        f"ENRICHMENT:\n{json.dumps(fields, sort_keys=True)}"
    )


def judge_subset(args: argparse.Namespace) -> dict[str, Any]:
    from google import genai

    artifact_dir = Path(args.artifact_dir)
    parsed_rows = [row for row in read_jsonl(artifact_dir / "parsed-enrichments.jsonl") if row.get("parsed")]
    if not parsed_rows:
        raise SystemExit("No parsed rows found; run analyze first")

    sampled = diverse_sample(
        [
            {
                **row,
                "id": row["chunk_id"],
                "content_type": row.get("source", {}).get("content_type"),
                "source": row.get("source", {}).get("source"),
                "char_count": row.get("source", {}).get("char_count"),
            }
            for row in parsed_rows
        ],
        limit=args.limit,
        seed=args.seed,
    )

    sanitizer = build_sanitizer(artifact_dir)
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY"))
    output_path = artifact_dir / "judge-results.jsonl"
    if output_path.exists() and not args.resume:
        output_path.unlink()

    existing = {row["chunk_id"] for row in read_jsonl(output_path)}
    judgments: list[dict[str, Any]] = []
    prompt_tokens = 0
    output_tokens = 0

    for row in sampled:
        chunk_id = row["chunk_id"]
        if chunk_id in existing:
            continue
        source = dict(row.get("source") or {})
        sanitize_result = sanitizer.sanitize(
            str(source.get("content") or ""),
            {"source": source.get("source"), "sender": source.get("sender")},
        )
        source["sanitized_content"] = sanitize_result.sanitized
        started = time.monotonic()
        raw_response = ""
        parsed_judge = None
        error = None
        try:
            response = client.models.generate_content(
                model=args.model,
                contents=judge_prompt(source, row["parsed"]),
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0,
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            raw_response = getattr(response, "text", "") or ""
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)
            parsed_judge = json.loads(raw_response)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        out = {
            "chunk_id": chunk_id,
            "judge_model": args.model,
            "latency_seconds": round(time.monotonic() - started, 3),
            "parsed_judge": parsed_judge,
            "raw_response": raw_response,
            "error": error,
        }
        append_jsonl(output_path, [out])
        if parsed_judge and not error:
            judgments.append(out)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    all_rows = read_jsonl(output_path)
    valid = [row for row in all_rows if row.get("parsed_judge") and not row.get("error")]
    field_scores: dict[str, list[float]] = defaultdict(list)
    overall_scores: list[float] = []
    issue_counter: Counter[str] = Counter()

    for row in valid:
        parsed = row["parsed_judge"]
        scores = parsed.get("field_scores") if isinstance(parsed, dict) else {}
        if isinstance(scores, dict):
            for field, value in scores.items():
                if isinstance(value, (int, float)):
                    field_scores[str(field)].append(float(value))
        overall = parsed.get("overall") if isinstance(parsed, dict) else None
        if isinstance(overall, (int, float)):
            overall_scores.append(float(overall))
        issues = parsed.get("issues") if isinstance(parsed, dict) else []
        if isinstance(issues, list):
            issue_counter.update(str(issue)[:120] for issue in issues if str(issue).strip())

    realtime_in = float(os.environ.get("BRAINLAYER_GEMINI_REALTIME_INPUT_COST_PER_MILLION", "0") or 0)
    realtime_out = float(os.environ.get("BRAINLAYER_GEMINI_REALTIME_OUTPUT_COST_PER_MILLION", "0") or 0)
    estimated_cost = None
    if realtime_in > 0 or realtime_out > 0:
        estimated_cost = (prompt_tokens * realtime_in + output_tokens * realtime_out) / 1_000_000

    summary = {
        "generated_at": now_iso(),
        "judge_model": args.model,
        "requested_limit": args.limit,
        "judged_rows": len(all_rows),
        "valid_judgments": len(valid),
        "valid_rate": round(len(valid) / len(all_rows), 4) if all_rows else 0,
        "field_score_means": {
            field: round(sum(values) / len(values), 3) for field, values in sorted(field_scores.items()) if values
        },
        "overall_mean": round(sum(overall_scores) / len(overall_scores), 3) if overall_scores else None,
        "top_issues": dict(issue_counter.most_common(20)),
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(estimated_cost, 6) if estimated_cost is not None else None,
            "pricing_source": "env BRAINLAYER_GEMINI_REALTIME_*_COST_PER_MILLION"
            if estimated_cost is not None
            else "not configured",
        },
        "judge_jsonl": str(output_path),
    }
    write_json(artifact_dir / "judge-summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--db-path", default=str(get_db_path()))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--model", default=DEFAULT_BATCH_MODEL)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export")
    export.add_argument("--limit", type=int, default=2000)
    export.add_argument("--seed", type=int, default=20260616)

    sub.add_parser("submit")

    poll = sub.add_parser("poll")
    poll.add_argument("--timeout-hours", type=float, default=24)
    poll.add_argument("--poll-interval-seconds", type=int, default=60)

    sub.add_parser("analyze")

    judge = sub.add_parser("judge")
    judge.add_argument("--limit", type=int, default=150)
    judge.add_argument("--seed", type=int, default=20260616)
    judge.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    judge.add_argument("--sleep-seconds", type=float, default=0.1)
    judge.add_argument("--resume", action="store_true")

    run = sub.add_parser("run-sample")
    run.add_argument("--limit", type=int, default=2000)
    run.add_argument("--seed", type=int, default=20260616)
    run.add_argument("--timeout-hours", type=float, default=24)
    run.add_argument("--poll-interval-seconds", type=int, default=60)
    run.add_argument("--judge-limit", type=int, default=150)
    run.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    run.add_argument("--judge-sleep-seconds", type=float, default=0.1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "export":
        print(json.dumps(export_sample(args), indent=2, sort_keys=True))
    elif args.command == "submit":
        print(json.dumps(submit_jobs(args), indent=2, sort_keys=True))
    elif args.command == "poll":
        print(json.dumps(poll_jobs(args), indent=2, sort_keys=True))
    elif args.command == "analyze":
        print(json.dumps(analyze_results(args), indent=2, sort_keys=True))
    elif args.command == "judge":
        print(json.dumps(judge_subset(args), indent=2, sort_keys=True))
    elif args.command == "run-sample":
        export_sample(args)
        submit_jobs(args)
        poll_jobs(args)
        analyze_results(args)
        judge_args = argparse.Namespace(
            artifact_dir=args.artifact_dir,
            limit=args.judge_limit,
            seed=args.seed,
            model=args.judge_model,
            sleep_seconds=args.judge_sleep_seconds,
            resume=True,
        )
        print(json.dumps(judge_subset(judge_args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
