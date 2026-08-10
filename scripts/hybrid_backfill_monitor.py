#!/usr/bin/env python3
"""Detached monitor for the staged hybrid backfill sample.

Polls the existing hybrid Gemini Batch jobs, then runs same-sample baseline and
paired judge Batch jobs. No BrainLayer DB writes are performed.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path = [entry for entry in sys.path if entry != str(SRC)]
sys.path.insert(0, str(SRC))

import hybrid_backfill_sample as sample  # noqa: E402
from brainlayer.cloud_backfill import (  # noqa: E402
    CHUNKS_PER_JOB,
    _extract_usage_metadata,
    build_batch_request_line,
    download_gemini_results,
    estimate_batch_cost_usd,
    get_gemini_batch_state,
    submit_gemini_batch,
)
from brainlayer.eval.enrichment_llm_judge import (  # noqa: E402
    _build_prompt,
    aggregate_judgments,
    build_pair_requests,
    parse_judge_response,
)
from brainlayer.pipeline.enrichment import build_external_prompt  # noqa: E402
from brainlayer.pipeline.sanitize import Sanitizer  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "hybrid-backfill-sample-2026-06-16-r1"
PROCESS_CRASH_LOG = ARTIFACT_DIR / "logs" / "monitor-crash.log"
PROCESS_CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
_CRASH_HANDLE = PROCESS_CRASH_LOG.open("a", encoding="utf-8")
faulthandler.enable(file=_CRASH_HANDLE)
print(f"[{datetime.now(timezone.utc).isoformat()}] monitor process starting argv={sys.argv}", file=_CRASH_HANDLE, flush=True)
COLLAB_PATH = Path("/Users/etanheyman/Gits/orchestrator/collab/2026-06-10-gen16-channel.md")
HYBRID_RUN_ID = "hybrid-backfill-sample-2026-06-16-r1"
BASELINE_RUN_ID = "baseline-oldprompt-validation-off-sample-2026-06-17-r1"
JUDGE_RUN_ID = "judge-hybrid-vs-baseline-sample-2026-06-17-r1"
HYBRID_MODEL = "models/gemini-2.5-flash-lite"
BASELINE_MODEL = "models/gemini-2.5-flash-lite"
JUDGE_MODEL = "models/gemini-2.5-flash"

VALID_INTENTS = {"debugging", "designing", "configuring", "discussing", "deciding", "implementing", "reviewing"}
VALID_EPISTEMIC = {"hypothesis", "substantiated", "validated"}
VALID_DEBT = {"introduction", "resolution", "none"}
VALID_SENTIMENTS = {"frustration", "confusion", "positive", "satisfaction", "neutral"}


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


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
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def safe_job_name(batch_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", batch_name)


def append_collab(line: str) -> None:
    existing = COLLAB_PATH.read_text(encoding="utf-8") if COLLAB_PATH.exists() else ""
    header = "### enrich-36k-backfill"
    if header not in existing:
        with COLLAB_PATH.open("a", encoding="utf-8") as handle:
            handle.write("\n" + header + "\n\n" + line.rstrip() + "\n")
        return

    match = re.search(r"(?m)^### enrich-36k-backfill\s*$", existing)
    if not match:
        with COLLAB_PATH.open("a", encoding="utf-8") as handle:
            handle.write("\n" + line.rstrip() + "\n")
        return

    insert_at = existing.find("\n", match.end())
    insert_at = len(existing) if insert_at == -1 else insert_at + 1
    if existing[insert_at : insert_at + 1] == "\n":
        insert_at += 1
    updated = existing[:insert_at] + line.rstrip() + "\n" + existing[insert_at:]
    COLLAB_PATH.write_text(updated, encoding="utf-8")


def poll_jobs_file(jobs_path: Path, raw_dir: Path) -> dict[str, Any]:
    jobs = read_json(jobs_path, {"jobs": []})
    if not jobs.get("jobs"):
        return {"state": "missing", "jobs": jobs.get("jobs", [])}

    pending = 0
    failed: list[dict[str, Any]] = []
    succeeded = 0
    for job in jobs["jobs"]:
        batch_name = job.get("batch_name")
        if not batch_name:
            failed.append(job)
            continue
        if job.get("status") == "succeeded" and job.get("raw_results_jsonl"):
            succeeded += 1
            continue
        if job.get("status") in {"failed", "cancelled"}:
            failed.append(job)
            continue
        status = get_gemini_batch_state(batch_name)
        job["status"] = status["state"]
        job["raw_state"] = status.get("raw_state")
        job["error"] = status.get("error")
        job["last_checked_at"] = datetime.now(timezone.utc).isoformat()
        if status["state"] == "succeeded":
            usage = _extract_usage_metadata(status["job"])
            job.update(usage)
            results = download_gemini_results(status["job"])
            raw_path = raw_dir / f"{safe_job_name(batch_name)}.jsonl"
            write_jsonl(raw_path, results)
            job["raw_results_jsonl"] = str(raw_path)
            job["result_count"] = len(results)
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            succeeded += 1
        elif status["state"] in {"failed", "cancelled"}:
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            failed.append(job)
        else:
            pending += 1
    write_json(jobs_path, jobs)
    if failed:
        return {"state": "failed", "jobs": jobs["jobs"], "failed": failed}
    if pending:
        return {"state": "pending", "jobs": jobs["jobs"], "pending": pending}
    return {"state": "succeeded", "jobs": jobs["jobs"], "succeeded": succeeded}


def combine_raw(jobs_path: Path, out_path: Path) -> Path:
    jobs = read_json(jobs_path, {"jobs": []})
    rows: list[dict[str, Any]] = []
    for job in jobs.get("jobs", []):
        raw = job.get("raw_results_jsonl")
        if raw:
            rows.extend(read_jsonl(Path(raw)))
    write_jsonl(out_path, rows)
    return out_path


def sum_job_usage(*jobs_paths: Path) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    job_count = 0
    for jobs_path in jobs_paths:
        jobs = read_json(jobs_path, {"jobs": []})
        for job in jobs.get("jobs", []):
            job_count += 1
            job_input = int(job.get("input_tokens") or 0)
            job_output = int(job.get("output_tokens") or 0)
            if (job_input == 0 and job_output == 0) and job.get("raw_results_jsonl"):
                raw_usage = result_usage(Path(job["raw_results_jsonl"]))
                job_input = raw_usage["input_tokens"]
                job_output = raw_usage["output_tokens"]
            input_tokens += job_input
            output_tokens += job_output
    return {
        "jobs": job_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(estimate_batch_cost_usd(input_tokens, output_tokens), 6),
    }


def result_usage(path: Path) -> dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    for row in read_jsonl(path):
        response = row.get("response") if isinstance(row, dict) else {}
        usage = response.get("usageMetadata") if isinstance(response, dict) else {}
        if not isinstance(usage, dict):
            continue
        input_tokens += int(usage.get("promptTokenCount") or usage.get("prompt_token_count") or 0)
        output_tokens += int(usage.get("candidatesTokenCount") or usage.get("candidates_token_count") or 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def ensure_hybrid_analyzed() -> None:
    analysis_path = ARTIFACT_DIR / "analysis-summary.json"
    if analysis_path.exists() and (ARTIFACT_DIR / "hybrid-for-judge.jsonl").exists():
        return
    combined_path = combine_raw(ARTIFACT_DIR / "jobs.json", ARTIFACT_DIR / "raw-results-combined.jsonl")
    combined_rows = len(read_jsonl(combined_path))
    selection_rows = len(read_jsonl(ARTIFACT_DIR / "selection-sample.jsonl"))
    log(f"combined hybrid raw results: {combined_rows}/{selection_rows} rows -> {combined_path}")
    if combined_rows != selection_rows:
        raise RuntimeError(f"hybrid raw result count mismatch: {combined_rows}/{selection_rows}")
    log("analyzing hybrid Batch outputs")
    args = argparse.Namespace(artifact_dir=str(ARTIFACT_DIR), run_id=HYBRID_RUN_ID, model=HYBRID_MODEL)
    sample.analyze_results(args)
    rows = []
    for row in read_jsonl(ARTIFACT_DIR / "parsed-enrichments.jsonl"):
        rows.append(
            {
                "chunk_id": row["chunk_id"],
                "model": HYBRID_MODEL,
                "backend": "gemini-batch-hybrid",
                "enrichment": row.get("parsed") or {},
                "raw_response": row.get("raw_response_text") or "",
                "error": row.get("error"),
            }
        )
    write_jsonl(ARTIFACT_DIR / "hybrid-for-judge.jsonl", rows)


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    for start in range(len(text)):
        if text[start] != "{":
            continue
        for end in range(len(text) - 1, start, -1):
            if text[end] != "}":
                continue
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def parse_baseline_enrichment(text: str) -> dict[str, Any] | None:
    raw = extract_json_object(text)
    if not raw:
        return None
    result: dict[str, Any] = {}
    summary = raw.get("summary")
    if isinstance(summary, str) and len(summary.strip()) > 5:
        result["summary"] = summary.strip()[:500]
    tags = raw.get("tags")
    if isinstance(tags, list):
        cleaned = [str(tag).strip() for tag in tags if isinstance(tag, str) and str(tag).strip()][:10]
        if cleaned:
            result["tags"] = cleaned
    importance = raw.get("importance")
    if isinstance(importance, (int, float)) and not isinstance(importance, bool):
        result["importance"] = max(1.0, min(10.0, float(importance)))
    intent = raw.get("intent")
    if isinstance(intent, str) and intent.lower().strip() in VALID_INTENTS:
        result["intent"] = intent.lower().strip()
    for key in ("primary_symbols", "external_deps", "key_facts", "resolved_queries", "sentiment_signals"):
        value = raw.get(key)
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()][:30]
            if cleaned:
                result[key] = cleaned
    resolved_query = raw.get("resolved_query")
    if isinstance(resolved_query, str) and len(resolved_query.strip()) > 10:
        result["resolved_query"] = resolved_query.strip()[:500]
    epistemic = raw.get("epistemic_level")
    if isinstance(epistemic, str) and epistemic.lower().strip() in VALID_EPISTEMIC:
        result["epistemic_level"] = epistemic.lower().strip()
    version_scope = raw.get("version_scope")
    if isinstance(version_scope, str) and version_scope.strip() and version_scope.lower().strip() != "null":
        result["version_scope"] = version_scope.strip()[:200]
    debt = raw.get("debt_impact")
    if isinstance(debt, str) and debt.lower().strip() in VALID_DEBT:
        result["debt_impact"] = debt.lower().strip()
    sentiment = raw.get("sentiment_label")
    if isinstance(sentiment, str) and sentiment.lower().strip() in VALID_SENTIMENTS:
        result["sentiment_label"] = sentiment.lower().strip()
    sentiment_score = raw.get("sentiment_score")
    if isinstance(sentiment_score, (int, float)) and not isinstance(sentiment_score, bool):
        result["sentiment_score"] = max(-1.0, min(1.0, float(sentiment_score)))
    return result if result.get("summary") and result.get("tags") else None


def ensure_baseline_submitted() -> None:
    jobs_path = ARTIFACT_DIR / "baseline-jobs.json"
    if jobs_path.exists() and read_json(jobs_path, {"jobs": []}).get("jobs"):
        return
    log("exporting/submitting same-sample baseline Batch jobs")
    os.environ["BRAINLAYER_ENRICHMENT_TAG_MODE"] = "taxonomy"
    os.environ["BRAINLAYER_ENRICHMENT_RUN_ID"] = BASELINE_RUN_ID
    sanitizer = Sanitizer.from_env()
    sanitizer.load_mapping(ARTIFACT_DIR / "pii_mapping.json")
    selection = read_jsonl(ARTIFACT_DIR / "selection-sample.jsonl")
    batch_lines: list[dict[str, Any]] = []
    sanitized = 0
    for row in selection:
        chunk = {
            "content": row.get("content") or "",
            "project": row.get("project"),
            "content_type": row.get("content_type"),
            "source": row.get("source"),
            "sender": row.get("sender"),
        }
        prompt, sanitize_result = build_external_prompt(chunk, sanitizer)
        if sanitize_result.pii_detected:
            sanitized += 1
        batch_lines.append(build_batch_request_line(str(row["chunk_id"]), prompt))
    batch_dir = ARTIFACT_DIR / "baseline-batch-inputs"
    batch_paths: list[str] = []
    for index, chunk in enumerate(sample.chunked(batch_lines, CHUNKS_PER_JOB), start=1):
        path = batch_dir / f"baseline_batch_{index:03d}.jsonl"
        write_jsonl(path, chunk)
        batch_paths.append(str(path))
    write_json(
        ARTIFACT_DIR / "baseline-export-summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": BASELINE_RUN_ID,
            "model": BASELINE_MODEL,
            "backend": "gemini-batch-baseline-oldprompt-validation-off",
            "rows": len(selection),
            "batch_jsonl_files": batch_paths,
            "pii_sanitized_chunks": sanitized,
            "baseline_definition": "old production taxonomy prompt behavior with tag validation off in parser",
        },
    )
    jobs = {"jobs": []}
    for path in batch_paths:
        name = submit_gemini_batch(Path(path), model=BASELINE_MODEL, store=None)
        jobs["jobs"].append(
            {
                "batch_name": name,
                "jsonl_path": path,
                "model": BASELINE_MODEL,
                "status": "submitted" if name else "submit_failed",
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(jobs_path, jobs)


def ensure_baseline_parsed() -> None:
    out_path = ARTIFACT_DIR / "baseline-for-judge.jsonl"
    summary_path = ARTIFACT_DIR / "baseline-analysis-summary.json"
    if out_path.exists() and summary_path.exists():
        return
    raw_path = combine_raw(ARTIFACT_DIR / "baseline-jobs.json", ARTIFACT_DIR / "baseline-raw-results-combined.jsonl")
    rows = []
    errors = Counter()
    tags = Counter()
    for result in read_jsonl(raw_path):
        chunk_id = str(result.get("key") or "")
        text = sample.extract_response_text(result)
        parsed = parse_baseline_enrichment(text) if text else None
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
        rows.append(
            {
                "chunk_id": chunk_id,
                "model": BASELINE_MODEL,
                "backend": "gemini-batch-baseline-oldprompt-validation-off",
                "enrichment": parsed or {},
                "raw_response": text,
                "error": error,
            }
        )
    write_jsonl(out_path, rows)
    success = [row for row in rows if not row.get("error")]
    write_json(
        summary_path,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": BASELINE_RUN_ID,
            "raw_result_rows": len(rows),
            "parse_success": len(success),
            "parse_failed": len(rows) - len(success),
            "parse_success_rate": round(len(success) / len(rows), 4) if rows else 0,
            "errors": dict(errors),
            "tag_stats": {
                "total_tag_assignments": sum(tags.values()),
                "unique_tags": len(tags),
                "singleton_tags": sum(1 for count in tags.values() if count == 1),
                "singleton_tag_rate": round(
                    sum(1 for count in tags.values() if count == 1) / len(tags), 4
                )
                if tags
                else 0,
                "avg_tags_per_success": round(sum(tags.values()) / len(success), 3) if success else 0,
                "top_tags": dict(tags.most_common(30)),
            },
            "batch_usage": sum_job_usage(ARTIFACT_DIR / "baseline-jobs.json"),
        },
    )


def judge_selection_rows() -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(ARTIFACT_DIR / "selection-sample.jsonl"):
        rows.append(
            {
                "chunk_id": row["chunk_id"],
                "chunk_text": row.get("content") or "",
                "pure_metadata": {
                    "project": row.get("project"),
                    "content_type": row.get("content_type"),
                    "source": row.get("source"),
                    "char_count": row.get("char_count"),
                },
            }
        )
    return rows


def ensure_judge_submitted() -> None:
    jobs_path = ARTIFACT_DIR / "judge-batch-jobs.json"
    if jobs_path.exists() and read_json(jobs_path, {"jobs": []}).get("jobs"):
        return
    log("building/submitting paired judge Batch jobs")
    selection = judge_selection_rows()
    hybrid = {row["chunk_id"]: row for row in read_jsonl(ARTIFACT_DIR / "hybrid-for-judge.jsonl") if not row.get("error")}
    baseline = {
        row["chunk_id"]: row for row in read_jsonl(ARTIFACT_DIR / "baseline-for-judge.jsonl") if not row.get("error")
    }
    requests = build_pair_requests(selection, hybrid, baseline, seed=20260617)
    request_map = {
        request["chunk_id"]: {
            "label_to_system": request["label_to_system"],
            "source_available": True,
        }
        for request in requests
    }
    write_json(ARTIFACT_DIR / "judge-request-map.json", request_map)
    write_json(
        ARTIFACT_DIR / "judge-request-summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "requested": len(requests),
            "selection_rows": len(selection),
            "hybrid_rows": len(hybrid),
            "baseline_rows": len(baseline),
            "judge_model": JUDGE_MODEL,
            "local_label": "HYBRID production prompt + hybrid taxonomy",
            "flex_label": "BASELINE old prompt + validation off",
        },
    )
    batch_dir = ARTIFACT_DIR / "judge-batch-inputs"
    batch_lines = [build_batch_request_line(request["chunk_id"], _build_prompt(request)) for request in requests]
    batch_paths: list[str] = []
    for index, chunk in enumerate(sample.chunked(batch_lines, CHUNKS_PER_JOB), start=1):
        path = batch_dir / f"judge_batch_{index:03d}.jsonl"
        write_jsonl(path, chunk)
        batch_paths.append(str(path))
    jobs = {"jobs": []}
    for path in batch_paths:
        name = submit_gemini_batch(Path(path), model=JUDGE_MODEL, store=None)
        jobs["jobs"].append(
            {
                "batch_name": name,
                "jsonl_path": path,
                "model": JUDGE_MODEL,
                "status": "submitted" if name else "submit_failed",
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(jobs_path, jobs)


def ensure_judge_aggregated() -> dict[str, Any]:
    summary_path = ARTIFACT_DIR / "judge-hybrid-vs-baseline-summary.json"
    if summary_path.exists():
        return read_json(summary_path)
    raw_path = combine_raw(ARTIFACT_DIR / "judge-batch-jobs.json", ARTIFACT_DIR / "judge-raw-results-combined.jsonl")
    request_map = read_json(ARTIFACT_DIR / "judge-request-map.json", {})
    rows = []
    judgments = []
    for result in read_jsonl(raw_path):
        chunk_id = str(result.get("key") or "")
        text = sample.extract_response_text(result)
        parsed = None
        error = None
        try:
            parsed = parse_judge_response(text)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        mapping = (request_map.get(chunk_id) or {}).get("label_to_system") or {}
        row = {
            "chunk_id": chunk_id,
            "judge_model": JUDGE_MODEL,
            "label_to_system": mapping,
            "field_scores": parsed.get("field_scores") if parsed else None,
            "overall": parsed.get("overall") if parsed else None,
            "raw_response": text,
            "error": error,
        }
        rows.append(row)
        if not error and parsed and mapping:
            judgments.append(row)
    write_jsonl(ARTIFACT_DIR / "judge-hybrid-vs-baseline.jsonl", rows)
    aggregation = aggregate_judgments(judgments)
    total_usage = sum_job_usage(
        ARTIFACT_DIR / "jobs.json",
        ARTIFACT_DIR / "baseline-jobs.json",
        ARTIFACT_DIR / "judge-batch-jobs.json",
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": JUDGE_MODEL,
        "requested": len(request_map),
        "succeeded": len(judgments),
        "failed": len(request_map) - len(judgments),
        "aggregation": aggregation,
        "verdict": verdict(aggregation),
        "usage_total": total_usage,
        "usage_breakdown": {
            "hybrid": sum_job_usage(ARTIFACT_DIR / "jobs.json"),
            "baseline": sum_job_usage(ARTIFACT_DIR / "baseline-jobs.json"),
            "judge": sum_job_usage(ARTIFACT_DIR / "judge-batch-jobs.json"),
        },
    }
    write_json(summary_path, summary)
    return summary


def verdict(aggregation: Mapping[str, Any]) -> str:
    overall = aggregation.get("overall", {})
    delta = float(overall.get("score_delta_local_minus_flex", 0.0))
    counts = overall.get("winner_counts", {})
    if delta >= 0.25 and counts.get("local", 0) >= counts.get("flex", 0):
        return "hybrid-better-than-baseline"
    if delta <= -0.25 and counts.get("flex", 0) > counts.get("local", 0):
        return "hybrid-below-baseline"
    return "hybrid-roughly-matches-baseline"


def post_failure(stage: str, state: Mapping[str, Any]) -> None:
    failed = state.get("failed") or []
    details = "; ".join(
        f"{job.get('batch_name')} status={job.get('status')} raw={job.get('raw_state')} error={job.get('error')}"
        for job in failed
    )
    append_collab(
        f"@orc enrichCodex SAMPLE FAILED ({datetime.now(timezone.utc).isoformat()}): stage `{stage}` failed. {details}. "
        "Live enrichment stayed paused; no full backfill/import was run."
    )


def post_success(summary: Mapping[str, Any]) -> None:
    agg = summary["aggregation"]
    fields = agg["fields"]
    field_bits = []
    for field in ("summary", "tags", "importance", "intent", "epistemic_level", "debt_impact", "sentiment"):
        metric = fields[field]
        means = metric["mean_scores"]
        field_bits.append(
            f"{field}: baseline {means['flex']:.2f} -> hybrid {means['local']:.2f} "
            f"(delta {metric['score_delta_local_minus_flex']:+.2f})"
        )
    overall = agg["overall"]
    means = overall["mean_scores"]
    hybrid_analysis = read_json(ARTIFACT_DIR / "analysis-summary.json", {})
    baseline_analysis = read_json(ARTIFACT_DIR / "baseline-analysis-summary.json", {})
    request_summary = read_json(ARTIFACT_DIR / "judge-request-summary.json", {})
    append_collab(
        "@orc enrichCodex SAMPLE VERDICT "
        f"({datetime.now(timezone.utc).isoformat()}): run `{HYBRID_RUN_ID}` finished; "
        "NO full backfill/import run and live enrichment stayed paused. "
        f"Judge rows: requested {summary['requested']}, succeeded {summary['succeeded']}, failed {summary['failed']}; "
        f"verdict `{summary['verdict']}`. "
        f"Overall: baseline {means['flex']:.2f} -> hybrid {means['local']:.2f} "
        f"(delta {overall['score_delta_local_minus_flex']:+.2f}; wins hybrid/baseline/tie "
        f"{overall['winner_counts']['local']}/{overall['winner_counts']['flex']}/{overall['winner_counts']['tie']}). "
        + "; ".join(field_bits)
        + ". "
        f"Parse: hybrid {hybrid_analysis.get('parse_success')}/{hybrid_analysis.get('raw_result_rows')} "
        f"({hybrid_analysis.get('parse_success_rate')}), baseline {baseline_analysis.get('parse_success')}/"
        f"{baseline_analysis.get('raw_result_rows')} ({baseline_analysis.get('parse_success_rate')}). "
        f"Tag singleton rate: hybrid {hybrid_analysis.get('tag_stats', {}).get('singleton_tag_rate')}, "
        f"baseline {baseline_analysis.get('tag_stats', {}).get('singleton_tag_rate')}. "
        f"TOTAL estimated Gemini Batch cost from returned usage metadata: ${summary['usage_total']['estimated_cost_usd']:.6f} "
        f"({summary['usage_total']['input_tokens']} input tokens, {summary['usage_total']['output_tokens']} output tokens; "
        f"hybrid ${summary['usage_breakdown']['hybrid']['estimated_cost_usd']:.6f}, "
        f"baseline ${summary['usage_breakdown']['baseline']['estimated_cost_usd']:.6f}, "
        f"judge ${summary['usage_breakdown']['judge']['estimated_cost_usd']:.6f}). "
        f"Artifacts: `{ARTIFACT_DIR}`. Await Etan GO before remaining backlog. "
        f"Judge labels: local={request_summary.get('local_label')}; flex={request_summary.get('flex_label')}."
    )


def step_once() -> str:
    baseline_jobs_path = ARTIFACT_DIR / "baseline-jobs.json"
    if baseline_jobs_path.exists():
        baseline_pre_state = poll_jobs_file(baseline_jobs_path, ARTIFACT_DIR / "baseline-raw-results")
        if baseline_pre_state["state"] == "failed":
            post_failure("baseline-generation", baseline_pre_state)
            return "failed"

    hybrid_state = poll_jobs_file(ARTIFACT_DIR / "jobs.json", ARTIFACT_DIR / "raw-results")
    if hybrid_state["state"] == "failed":
        post_failure("hybrid-generation", hybrid_state)
        return "failed"
    if hybrid_state["state"] != "succeeded":
        log(f"hybrid jobs pending: {hybrid_state}")
        return "pending"

    ensure_hybrid_analyzed()
    ensure_baseline_submitted()
    baseline_state = poll_jobs_file(ARTIFACT_DIR / "baseline-jobs.json", ARTIFACT_DIR / "baseline-raw-results")
    if baseline_state["state"] == "failed":
        post_failure("baseline-generation", baseline_state)
        return "failed"
    if baseline_state["state"] != "succeeded":
        log(f"baseline jobs pending: {baseline_state}")
        return "pending"

    ensure_baseline_parsed()
    ensure_judge_submitted()
    judge_state = poll_jobs_file(ARTIFACT_DIR / "judge-batch-jobs.json", ARTIFACT_DIR / "judge-raw-results")
    if judge_state["state"] == "failed":
        post_failure("judge", judge_state)
        return "failed"
    if judge_state["state"] != "succeeded":
        log(f"judge jobs pending: {judge_state}")
        return "pending"

    summary = ensure_judge_aggregated()
    post_success(summary)
    return "done"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["step", "loop"])
    parser.add_argument("--sleep-seconds", type=int, default=180)
    args = parser.parse_args()
    try:
        if args.command == "step":
            return 0 if step_once() in {"pending", "done"} else 2
        while True:
            state = step_once()
            if state == "done":
                log("sample verdict posted")
                return 0
            if state == "failed":
                log("failure posted")
                return 2
            time.sleep(args.sleep_seconds)
    except Exception as exc:  # noqa: BLE001
        tb_path = ARTIFACT_DIR / "logs" / "monitor-exception.log"
        tb_path.write_text(traceback.format_exc(), encoding="utf-8")
        append_collab(
            f"@orc enrichCodex SAMPLE FAILED ({datetime.now(timezone.utc).isoformat()}): "
            f"stage `monitor-exception` raised {type(exc).__name__}: {exc}. "
            f"Traceback: `{tb_path}`. Live enrichment stayed paused; no full backfill/import was run."
        )
        log(f"monitor exception written to {tb_path}: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
