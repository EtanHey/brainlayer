"""Track A local-vs-Gemini enrichment quality benchmark utilities.

This module is intentionally separate from ``enrichment_controller``. It reads
BrainLayer chunks and writes JSONL artifacts for evaluation, but it does not
write enrichment results back to the live database.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from brainlayer.eval.enrichment_graders import validate_schema_gate
from brainlayer.paths import get_db_path
from brainlayer.pipeline.enrichment import parse_enrichment

ENRICHMENT_COLUMNS = (
    "summary",
    "tags",
    "importance",
    "intent",
    "primary_symbols",
    "resolved_query",
    "key_facts",
    "resolved_queries",
    "epistemic_level",
    "version_scope",
    "debt_impact",
    "external_deps",
    "raw_entities_json",
    "sentiment_label",
    "sentiment_score",
    "sentiment_signals",
)

CORRECTION_PATTERNS = (
    r"\bcorrection\b",
    r"\bactually\b",
    r"\bno,\s+that\b",
    r"\bwrong\b",
    r"\bnot\s+that\b",
    r"\bi told you\b",
    r"\bsupersede\b",
)
DECISION_PATTERNS = (
    r"\bdecision\b",
    r"\bdecided\b",
    r"\bruling\b",
    r"\blocked\b",
    r"\bapproved\b",
    r"\brecommend(?:ed|ation)?\b",
    r"\bmust\b",
    r"\bdefault\b",
)
CHECKPOINT_PATTERNS = (
    r"\bPR\s*#?\d+\b",
    r"\bcommit\s+[0-9a-f]{7,40}\b",
    r"\bbranch\b",
    r"\bmerged\b",
    r"\bpushed\b",
    r"\b\d+\s+passed\b",
    r"\bTASK_DONE\b",
    r"\bDONE\b",
    r"\bCLAIMED\b",
    r"\bBLOCKED\b",
)
NOISE_PATTERNS = (
    r"^\s*(ok|okay|thanks|thank you|sounds good|yep|yes|no)\s*[.!]*\s*$",
    r"\[BrainLayer (auto|deep)\] Memories matching",
    r"\bbrain_search\s*\(",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    limit: int = 2000
    since_days: int = 14
    min_score: int = 4
    candidate_multiplier: int = 6


@dataclass(frozen=True)
class MeaningfulScore:
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MemorySnapshot:
    total_gb: float
    available_gb: float
    used_gb: float
    swap_used_gb: float
    swap_total_gb: float


@dataclass(frozen=True)
class PairScore:
    overall: float
    schema_passed: bool
    schema_errors: tuple[str, ...]
    source_support: float
    baseline_alignment: float
    tag_overlap: float


def build_selection_query(config: BenchmarkConfig) -> str:
    """Return a pure-chunk selection query with no enrichment columns."""

    del config
    return """
        SELECT id, content, project, content_type, source, source_file, created_at, char_count
        FROM chunks
        WHERE content IS NOT NULL
          AND length(trim(content)) >= 20
          AND (
            created_at IS NULL
            OR datetime(created_at) >= datetime('now', ?)
          )
        ORDER BY datetime(COALESCE(created_at, '1970-01-01T00:00:00Z')) DESC
        LIMIT ?
    """


def select_meaningful_chunks(db_path: str | Path, config: BenchmarkConfig) -> list[dict[str, Any]]:
    """Select meaningful recent chunks using only pure content/metadata signals."""

    rows: list[dict[str, Any]] = []
    with _connect_readonly(db_path) as conn:
        conn.row_factory = sqlite3.Row
        query = build_selection_query(config)
        raw_rows = conn.execute(
            query,
            (f"-{max(1, config.since_days)} days", max(config.limit, 1) * max(config.candidate_multiplier, 1)),
        ).fetchall()
        scored: list[tuple[int, dict[str, Any], MeaningfulScore]] = []
        for row in raw_rows:
            pure = dict(row)
            score = score_meaningful_chunk(pure["content"], pure)
            if score.score >= config.min_score:
                scored.append((_created_at_sort_key(pure.get("created_at")), pure, score))

        scored.sort(key=lambda item: (item[2].score, item[0]), reverse=True)
        selected = scored[: config.limit]
        baselines = _load_existing_enrichments(conn, [pure["id"] for _, pure, _ in selected])

    for _, pure, score in selected:
        chunk_id = pure.pop("id")
        rows.append(
            {
                "chunk_id": chunk_id,
                "chunk_text": pure.pop("content"),
                "pure_metadata": pure,
                "meaningful_score": score.score,
                "meaningful_reasons": list(score.reasons),
                "gemini_existing": baselines.get(chunk_id, {}),
            }
        )
    return rows


def score_meaningful_chunk(content: str, metadata: Mapping[str, Any] | None = None) -> MeaningfulScore:
    """Heuristic score for Track A's meaningful-chunk sample."""

    metadata = metadata or {}
    text = content or ""
    lowered = text.lower()
    reasons: list[str] = []
    score = 0

    if len(text.strip()) < 20 or any(re.search(pattern, text, re.I) for pattern in NOISE_PATTERNS):
        return MeaningfulScore(0, ("noise",))

    if any(re.search(pattern, text, re.I) for pattern in CORRECTION_PATTERNS):
        score += 4
        reasons.append("correction")
    if any(re.search(pattern, text, re.I) for pattern in DECISION_PATTERNS):
        score += 3
        reasons.append("decision")
    if any(re.search(pattern, text, re.I) for pattern in CHECKPOINT_PATTERNS):
        score += 3
        reasons.append("checkpoint")
    if "etan" in lowered:
        score += 1
        reasons.append("etan")
    if re.search(r"\b(tests?|pytest|ci|verified|green|red|regression|gate)\b", lowered):
        score += 1
        reasons.append("verification")
    if re.search(r"\b(brainlayer|golems|orchestrator|cmux|voicelayer|flex|gemini|mlx)\b", lowered):
        score += 1
        reasons.append("ecosystem")
    if metadata.get("content_type") in {"user_text", "assistant_text"}:
        score += 1
        reasons.append("conversation")

    return MeaningfulScore(min(score, 10), tuple(dict.fromkeys(reasons)))


def enrichment_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Parse existing DB enrichment fields into the runtime enrichment shape."""

    result: dict[str, Any] = {}
    for key in (
        "summary",
        "intent",
        "resolved_query",
        "epistemic_level",
        "version_scope",
        "debt_impact",
        "sentiment_label",
    ):
        value = row.get(key)
        if value not in (None, ""):
            result[key] = value

    if row.get("importance") not in (None, ""):
        try:
            result["importance"] = int(round(float(row["importance"])))
        except (TypeError, ValueError):
            pass
    if row.get("sentiment_score") not in (None, ""):
        try:
            result["sentiment_score"] = float(row["sentiment_score"])
        except (TypeError, ValueError):
            pass

    for key in ("tags", "primary_symbols", "key_facts", "resolved_queries", "external_deps", "sentiment_signals"):
        parsed = _json_list(row.get(key))
        if parsed:
            result[key] = parsed

    entities = _json_list(row.get("raw_entities_json"))
    if entities:
        result["entities"] = entities
    return result


def current_memory_snapshot() -> MemorySnapshot:
    """Return host memory/swap in GiB."""

    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - dependency exists in local env
        raise RuntimeError("psutil is required for local MLX safety preflight") from exc

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    divisor = 1024**3
    return MemorySnapshot(
        total_gb=round(vm.total / divisor, 3),
        available_gb=round(vm.available / divisor, 3),
        used_gb=round(vm.used / divisor, 3),
        swap_used_gb=round(swap.used / divisor, 3),
        swap_total_gb=round(swap.total / divisor, 3),
    )


def is_memory_safe_for_model(
    snapshot: MemorySnapshot,
    *,
    model_id: str,
    min_available_gb: float | None = None,
    max_swap_used_gb: float = 16.0,
) -> tuple[bool, str]:
    """Conservative preflight for local MLX runs on the live M4 Max."""

    model_lower = model_id.lower()
    if min_available_gb is None:
        min_available_gb = 12.0 if "14b" in model_lower else 5.0 if "4b" in model_lower else 8.0
    if snapshot.available_gb < min_available_gb:
        return (
            False,
            f"available memory {snapshot.available_gb:.1f}GB below required {min_available_gb:.1f}GB for {model_id}",
        )
    if snapshot.swap_total_gb and snapshot.swap_used_gb > max_swap_used_gb:
        return (
            False,
            f"swap used {snapshot.swap_used_gb:.1f}GB above safety cap {max_swap_used_gb:.1f}GB",
        )
    return True, "memory preflight passed"


def run_local_mlx(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    model_id: str,
    limit: int,
    max_tokens: int = 900,
    min_available_gb: float | None = None,
    allow_tight_memory: bool = False,
) -> dict[str, Any]:
    """Run a benchmark-only MLX enrichment pass and write JSONL outputs."""

    snapshot = current_memory_snapshot()
    ok, reason = is_memory_safe_for_model(snapshot, model_id=model_id, min_available_gb=min_available_gb)
    if not ok and not allow_tight_memory:
        raise RuntimeError(f"LOCAL_MLX_PREFLIGHT_BLOCKED: {reason}; snapshot={asdict(snapshot)}")

    from mlx_lm import generate, load

    model, tokenizer = load(model_id)
    count = 0
    failures = 0
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_jsonl).open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for row in _iter_jsonl(source):
            if count >= limit:
                break
            started = time.monotonic()
            prompt = _local_prompt_for_row(row)
            try:
                generated = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=max_tokens)
                enrichment = _normalize_enrichment_for_benchmark(parse_enrichment(generated) or {})
                error = None if enrichment else "invalid_enrichment"
            except Exception as exc:  # noqa: BLE001
                generated = ""
                enrichment = {}
                error = str(exc)
            if error:
                failures += 1
            target.write(
                json.dumps(
                    {
                        "chunk_id": row["chunk_id"],
                        "model": model_id,
                        "backend": "local-mlx",
                        "latency_seconds": round(time.monotonic() - started, 3),
                        "enrichment": enrichment,
                        "raw_response": generated,
                        "error": error,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            count += 1
    return {"attempted": count, "failed": failures, "model": model_id, "memory_snapshot": asdict(snapshot)}


def run_gemini_flex_sample(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    limit: int,
    seed: int = 20260615,
) -> dict[str, Any]:
    """Re-run a deterministic sample through Gemini Flex for variance measurement."""

    from brainlayer.enrichment_controller import (
        GEMINI_REALTIME_MODEL,
        _build_gemini_config,
        _generate_content_with_rate_limit,
        _get_gemini_client,
    )
    from brainlayer.pipeline.enrichment import build_external_prompt
    from brainlayer.pipeline.sanitize import Sanitizer

    rows = list(_read_jsonl(input_jsonl))
    random.Random(seed).shuffle(rows)
    sample = rows[:limit]
    client = _get_gemini_client()
    config = _build_gemini_config()
    sanitizer = Sanitizer.from_env()
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with output_path.open("w", encoding="utf-8") as target:
        for row in sample:
            chunk = {
                "id": row["chunk_id"],
                "content": row["chunk_text"],
                "project": row.get("pure_metadata", {}).get("project"),
                "content_type": row.get("pure_metadata", {}).get("content_type"),
                "source": row.get("pure_metadata", {}).get("source"),
            }
            started = time.monotonic()
            try:
                prompt, _ = build_external_prompt(chunk, sanitizer)
                response = _generate_content_with_rate_limit(client, GEMINI_REALTIME_MODEL, prompt, config, None)
                raw_response = getattr(response, "text", "")
                enrichment = _normalize_enrichment_for_benchmark(parse_enrichment(raw_response) or {})
                error = None if enrichment else "invalid_enrichment"
            except Exception as exc:  # noqa: BLE001
                raw_response = ""
                enrichment = {}
                error = str(exc)
            if error:
                failures += 1
            target.write(
                json.dumps(
                    {
                        "chunk_id": row["chunk_id"],
                        "model": GEMINI_REALTIME_MODEL,
                        "backend": "gemini-flex",
                        "latency_seconds": round(time.monotonic() - started, 3),
                        "enrichment": enrichment,
                        "raw_response": raw_response,
                        "error": error,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return {"attempted": len(sample), "failed": failures, "model": GEMINI_REALTIME_MODEL}


def score_enrichment_pair(
    source_text: str,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> PairScore:
    """Score a candidate enrichment against the source and Gemini baseline."""

    schema = validate_schema_gate(candidate)
    candidate_text = _enrichment_text(candidate)
    baseline_text = _enrichment_text(baseline)
    source_support = _token_overlap(candidate_text, source_text)
    baseline_alignment = _token_overlap(candidate_text, baseline_text)
    tag_overlap = _jaccard(_string_list(candidate.get("tags")), _string_list(baseline.get("tags")))
    schema_score = 1.0 if schema.passed else 0.0
    overall = 0.2 * schema_score + 0.35 * source_support + 0.35 * baseline_alignment + 0.1 * tag_overlap
    return PairScore(
        overall=round(max(0.0, min(1.0, overall)), 4),
        schema_passed=schema.passed,
        schema_errors=tuple(schema.errors),
        source_support=round(source_support, 4),
        baseline_alignment=round(baseline_alignment, 4),
        tag_overlap=round(tag_overlap, 4),
    )


def grade_outputs(
    selection_jsonl: str | Path,
    local_jsonl: str | Path,
    report_json: str | Path,
    *,
    flex_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    """Grade local-vs-existing-Gemini and optional Flex-vs-existing-Gemini variance."""

    selection = {row["chunk_id"]: row for row in _read_jsonl(selection_jsonl)}
    local = {row["chunk_id"]: row for row in _read_jsonl(local_jsonl)}
    flex = {row["chunk_id"]: row for row in _read_jsonl(flex_jsonl)} if flex_jsonl else {}

    local_scores: list[dict[str, Any]] = []
    flex_scores: list[dict[str, Any]] = []
    for chunk_id, row in selection.items():
        baseline = row.get("gemini_existing") or {}
        if not baseline:
            continue
        if chunk_id in local:
            score = score_enrichment_pair(row["chunk_text"], local[chunk_id].get("enrichment") or {}, baseline)
            local_scores.append({"chunk_id": chunk_id, **asdict(score), "error": local[chunk_id].get("error")})
        if chunk_id in flex:
            score = score_enrichment_pair(row["chunk_text"], flex[chunk_id].get("enrichment") or {}, baseline)
            flex_scores.append({"chunk_id": chunk_id, **asdict(score), "error": flex[chunk_id].get("error")})

    local_summary = _score_summary(local_scores)
    flex_summary = _score_summary(flex_scores)
    verdict = "insufficient-flex-baseline"
    if local_summary["count"] and flex_summary["count"]:
        verdict = (
            "local-matches-flex-variance"
            if local_summary["mean_overall"] >= flex_summary["mean_overall"] - 0.08
            else "local-below-flex-variance"
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_count": len(selection),
        "local": local_summary,
        "flex_variance": flex_summary,
        "verdict": verdict,
        "local_scores": local_scores,
        "flex_scores": flex_scores,
    }
    Path(report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(report_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    count = 0
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
            count += 1
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track A local-vs-Gemini enrichment quality benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    select_p = sub.add_parser("select", help="select meaningful pure chunks from a read-only BrainLayer DB")
    select_p.add_argument("--db", default=str(get_db_path()))
    select_p.add_argument("--out", required=True)
    select_p.add_argument("--limit", type=int, default=2000)
    select_p.add_argument("--since-days", type=int, default=14)
    select_p.add_argument("--min-score", type=int, default=4)

    mem_p = sub.add_parser("memory", help="print local memory preflight JSON")
    mem_p.add_argument("--model", default="mlx-community/Qwen3-14B-4bit")
    mem_p.add_argument("--min-available-gb", type=float)

    local_p = sub.add_parser("local", help="run benchmark-only local MLX enrichment")
    local_p.add_argument("--in", dest="input", required=True)
    local_p.add_argument("--out", required=True)
    local_p.add_argument("--model", required=True)
    local_p.add_argument("--limit", type=int, default=50)
    local_p.add_argument("--max-tokens", type=int, default=900)
    local_p.add_argument("--min-available-gb", type=float)
    local_p.add_argument("--allow-tight-memory", action="store_true")

    flex_p = sub.add_parser("flex-sample", help="rerun a sample through Gemini Flex for natural variance")
    flex_p.add_argument("--in", dest="input", required=True)
    flex_p.add_argument("--out", required=True)
    flex_p.add_argument("--limit", type=int, default=50)
    flex_p.add_argument("--seed", type=int, default=20260615)

    grade_p = sub.add_parser("grade", help="grade local-vs-Gemini and optional Flex variance")
    grade_p.add_argument("--selection", required=True)
    grade_p.add_argument("--local", required=True)
    grade_p.add_argument("--out", required=True)
    grade_p.add_argument("--flex")

    args = parser.parse_args(argv)
    if args.command == "select":
        rows = select_meaningful_chunks(args.db, BenchmarkConfig(args.limit, args.since_days, args.min_score))
        count = write_jsonl(rows, args.out)
        print(json.dumps({"selected": count, "out": args.out}, sort_keys=True))
        return 0
    if args.command == "memory":
        snapshot = current_memory_snapshot()
        ok, reason = is_memory_safe_for_model(snapshot, model_id=args.model, min_available_gb=args.min_available_gb)
        print(json.dumps({"allowed": ok, "reason": reason, "snapshot": asdict(snapshot)}, sort_keys=True))
        return 0 if ok else 2
    if args.command == "local":
        result = run_local_mlx(
            args.input,
            args.out,
            model_id=args.model,
            limit=args.limit,
            max_tokens=args.max_tokens,
            min_available_gb=args.min_available_gb,
            allow_tight_memory=args.allow_tight_memory,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "flex-sample":
        print(
            json.dumps(run_gemini_flex_sample(args.input, args.out, limit=args.limit, seed=args.seed), sort_keys=True)
        )
        return 0
    if args.command == "grade":
        print(json.dumps(grade_outputs(args.selection, args.local, args.out, flex_jsonl=args.flex), sort_keys=True))
        return 0
    return 1


def _local_prompt_for_row(row: Mapping[str, Any]) -> str:
    content = str(row["chunk_text"])
    if len(content) > 4500:
        content = content[:2700] + "\n[...truncated middle...]\n" + content[-1600:]
    metadata = row.get("pure_metadata", {})
    project = metadata.get("project") or "unknown"
    content_type = metadata.get("content_type") or "unknown"
    return f"""You enrich one BrainLayer memory chunk for future retrieval.
Return ONLY one valid JSON object. No markdown fences. No prose before or after JSON.

Allowed intent: debugging, designing, configuring, discussing, deciding, implementing, reviewing.
Allowed epistemic_level: hypothesis, substantiated, validated.
Allowed debt_impact: introduction, resolution, none.
Allowed entity.type: person, agent, company, project, technology, tool, concept, topic, source.
Allowed sentiment_label: frustration, confusion, positive, satisfaction, neutral.

Required JSON keys:
summary: one specific sentence, <= 80 words
key_facts: 3-8 supported facts
tags: 3-7 lowercase tags
importance: integer 1-10
intent: one allowed intent
primary_symbols: list of important names/symbols
resolved_query: one likely future search query
resolved_queries: exactly 3 likely future search queries
epistemic_level: one allowed epistemic_level
version_scope: string or null
debt_impact: one allowed debt_impact
external_deps: list of external tools/services
entities: list of objects with name, type, entity_subtype, relation
sentiment_label: one allowed sentiment_label
sentiment_score: number from -1 to 1
sentiment_signals: list of short evidence strings

Project: {project}
Content type: {content_type}
BEGIN SOURCE CHUNK
{content}
END SOURCE CHUNK

Now output exactly one JSON object matching the required keys. Do not copy JSON examples from the source chunk unless they are inside string values.
"""


def _apply_chat_template(tokenizer: Any, prompt: str) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_template is None:
        return prompt


def _normalize_enrichment_for_benchmark(enrichment: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(enrichment)
    if isinstance(result.get("importance"), float) and math.isfinite(result["importance"]):
        result["importance"] = int(round(result["importance"]))
    if "sentiment_score" not in result:
        result["sentiment_score"] = 0.0
    if "sentiment_signals" not in result:
        result["sentiment_signals"] = []
    if "entities" not in result:
        result["entities"] = []
    if "external_deps" not in result:
        result["external_deps"] = []
    if "version_scope" not in result:
        result["version_scope"] = None
    return result
    try:
        rendered = apply_template(
            [
                {
                    "role": "system",
                    "content": "You are a strict JSON enrichment function. You output only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered if isinstance(rendered, str) else prompt
    except Exception:
        return prompt


def _connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    if path.exists():
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    return sqlite3.connect(path)


def _load_existing_enrichments(conn: sqlite3.Connection, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return {}
    columns = _available_columns(conn, "chunks")
    selected_cols = ["id"] + [col for col in ENRICHMENT_COLUMNS if col in columns]
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(
        f"SELECT {', '.join(selected_cols)} FROM chunks WHERE id IN ({placeholders})", tuple(chunk_ids)
    ).fetchall()
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_map = dict(zip(selected_cols, row, strict=True))
        chunk_id = row_map.pop("id")
        result[chunk_id] = enrichment_from_row(row_map)
    return result


def _available_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _created_at_sort_key(value: Any) -> int:
    if not value:
        return 0
    text = str(value).replace("Z", "+00:00")
    try:
        return int(datetime.fromisoformat(text).timestamp())
    except ValueError:
        return 0


def _json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).lower().strip() for item in value if str(item).strip()]


def _enrichment_text(enrichment: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("summary", "resolved_query", "intent", "epistemic_level", "debt_impact", "sentiment_label"):
        value = enrichment.get(key)
        if isinstance(value, str):
            parts.append(value)
    for key in ("key_facts", "resolved_queries", "tags", "primary_symbols", "external_deps", "sentiment_signals"):
        parts.extend(_string_list(enrichment.get(key)))
    entities = enrichment.get("entities")
    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, Mapping) and entity.get("name"):
                parts.append(str(entity["name"]))
    return " ".join(parts)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_.#/-]{1,}", text.lower())
        if token not in {"the", "and", "that", "this", "with", "from"}
    }


def _token_overlap(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _score_summary(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {"count": 0, "mean_overall": 0.0, "schema_pass_rate": 0.0}
    overall = [float(row["overall"]) for row in scores if row.get("error") in (None, "")]
    schema_passed = [bool(row["schema_passed"]) for row in scores if row.get("error") in (None, "")]
    return {
        "count": len(scores),
        "successful": len(overall),
        "mean_overall": round(sum(overall) / len(overall), 4) if overall else 0.0,
        "median_overall": round(sorted(overall)[len(overall) // 2], 4) if overall else 0.0,
        "schema_pass_rate": round(sum(schema_passed) / len(schema_passed), 4) if schema_passed else 0.0,
    }


def _read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(_iter_jsonl(handle))


def _iter_jsonl(handle: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in handle:
        if line.strip():
            yield json.loads(line)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
