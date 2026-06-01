"""Offline ABCDE enrichment LLM-judge harness.

This module deliberately does not call a metered model API. It prepares and
scores JSONL batches that can be judged inline by a subscription CLI/session or
by a local callable in tests. All writes go through ExperimentStore.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from brainlayer.eval.enrichment_graders import (
    find_banned_summary_pattern,
    grade_candidate,
    score_entities,
    validate_schema_gate,
)
from brainlayer.eval.experiment_store import VALID_VARIANT_IDS, ExperimentStore

DEFAULT_TEMPERATURE = 0
JUDGE_KAPPA_FLOOR = 0.6
JUDGE_SPEARMAN_FLOOR = 0.7
RESPONSE_SCHEMA = "two_step_reason_then_score_v1"
SCORE_KEYS = ("faithfulness", "usefulness", "entity_coverage")
SCORE_WEIGHTS = {"faithfulness": 0.4, "usefulness": 0.3, "entity_coverage": 0.3}

JUDGE_PROMPT = """You are judging BrainLayer enrichment label quality.

Evaluate exactly one variant enrichment against exactly one source chunk. Use
the deterministic pre-signals as weak evidence only; the source chunk is the
authority.

Rubric, 1-5 integer scores:
- faithfulness: 5 means every summary/key fact/entity/query is supported by the
  source chunk; 1 means major hallucination or contradiction.
- usefulness: 5 means the enrichment would materially improve future retrieval,
  recall, and operator understanding; 1 means generic, vague, or non-actionable.
- entity_coverage: 5 means all important people/projects/tools/concepts present
  in the source chunk are covered without unsupported additions; 1 means key
  entities are missed or mostly hallucinated.

Two-step strict JSON only:
1. reason: concise per-dimension reasoning, no hidden chain-of-thought.
2. score: integer scores for faithfulness, usefulness, and entity_coverage.

Few-shot:
Input: chunk says "BrainLayer uses SQLite FTS5 for exact search." enrichment
claims "BrainLayer uses Postgres and Pinecone."
Output: {"reason":{"faithfulness":"Contradicts SQLite FTS5 with unsupported
Postgres/Pinecone.","usefulness":"Misleads retrieval with wrong systems.",
"entity_coverage":"BrainLayer appears, but the storage entities are wrong."},
"score":{"faithfulness":1,"usefulness":1,"entity_coverage":2},"rationale":
"Unsupported storage claims make the label unsafe."}

Input: chunk says "PR-JUDGE adds offline JSONL scoring with kappa and Spearman
calibration gates." enrichment captures PR-JUDGE, offline JSONL, and both gates.
Output: {"reason":{"faithfulness":"All claims are supported by the chunk.",
"usefulness":"Captures the operational point and metrics for future lookup.",
"entity_coverage":"The central PR and metric concepts are covered."},"score":
{"faithfulness":5,"usefulness":5,"entity_coverage":5},"rationale":
"Faithful, specific, and retrieval-useful."}

Return only one JSON object with keys: reason, score, rationale.
"""

JUDGE_PROMPT_HASH = hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()

JudgeCallable = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class CalibrationResult:
    kappa: float
    spearman_rho: float
    kappa_floor: float
    spearman_floor: float
    sample_size: int
    quarantined: bool


class JudgeQuarantinedError(RuntimeError):
    """Raised when REC-08 calibration floors are not met."""

    def __init__(self, result: CalibrationResult) -> None:
        super().__init__(
            "LLM judge is quarantined: "
            f"kappa={result.kappa:.4f} floor={result.kappa_floor:.4f}, "
            f"spearman={result.spearman_rho:.4f} floor={result.spearman_floor:.4f}"
        )
        self.result = result


class JudgeResponseError(ValueError):
    """Raised when an inline/subscription judge returns non-strict JSON."""


def prepare_batch_jsonl(
    input_jsonl_path: str | Path,
    output_jsonl_path: str | Path,
    *,
    judge_model: str = "subscription-inline",
    judge_cli_version: str = "unknown",
) -> int:
    """Write prompt packages for inline/subscription batch judging."""

    count = 0
    with (
        Path(input_jsonl_path).open("r", encoding="utf-8") as source,
        Path(output_jsonl_path).open("w", encoding="utf-8") as target,
    ):
        for row in _iter_jsonl(source):
            request = build_judge_request(row, judge_model=judge_model, judge_cli_version=judge_cli_version)
            target.write(json.dumps(request, sort_keys=True) + "\n")
            count += 1
    return count


def score_jsonl_inline(
    input_jsonl_path: str | Path,
    output_jsonl_path: str | Path,
    *,
    judge: JudgeCallable,
    experiment_store: ExperimentStore | None = None,
    judge_model: str = "subscription-inline",
    judge_cli_version: str = "unknown",
) -> int:
    """Score input JSONL using an injected inline/subscription judge callable."""

    count = 0
    with (
        Path(input_jsonl_path).open("r", encoding="utf-8") as source,
        Path(output_jsonl_path).open("w", encoding="utf-8") as target,
    ):
        for row in _iter_jsonl(source):
            request = build_judge_request(row, judge_model=judge_model, judge_cli_version=judge_cli_version)
            response = _validate_response(judge(request))
            scored = _scored_row(row, request, response, judge_model=judge_model, judge_cli_version=judge_cli_version)
            target.write(json.dumps(scored, sort_keys=True) + "\n")
            if experiment_store is not None:
                _persist_judgment(experiment_store, scored)
            count += 1
    return count


def build_judge_request(
    row: Mapping[str, Any],
    *,
    judge_model: str = "subscription-inline",
    judge_cli_version: str = "unknown",
) -> dict[str, Any]:
    """Build the strict JSON prompt package for one chunk/variant row."""

    chunk_id = _required_str(row, "chunk_id")
    variant_id = _required_str(row, "variant_id")
    if variant_id not in VALID_VARIANT_IDS:
        raise ValueError(f"variant_id must be one of A..E, got {variant_id!r}")
    chunk_text = _required_str(row, "chunk_text")
    enrichment = _required_mapping(row, "enrichment")
    pre_signals = deterministic_pre_signals(row)

    return {
        "mode": "inline_or_subscription_batch",
        "chunk_id": chunk_id,
        "variant_id": variant_id,
        "system_instruction": (
            "Do not call a metered API. Judge this row inline or with a subscription CLI session. "
            "Return only strict JSON matching the contract."
        ),
        "judge_prompt": JUDGE_PROMPT,
        "judge_prompt_hash": JUDGE_PROMPT_HASH,
        "judge_model": judge_model,
        "judge_cli_version": judge_cli_version,
        "temperature": DEFAULT_TEMPERATURE,
        "response_schema": RESPONSE_SCHEMA,
        "strict_json_contract": {
            "required_top_level_keys": ["reason", "score", "rationale"],
            "score_keys": list(SCORE_KEYS),
            "score_range": [1, 5],
            "score_type": "integer",
        },
        "input": {
            "chunk_text": chunk_text,
            "enrichment": enrichment,
            "deterministic_pre_signals": pre_signals,
            "variant_model": row.get("model"),
            "variant_prompt_hash": row.get("prompt_hash"),
        },
    }


def deterministic_pre_signals(row: Mapping[str, Any]) -> dict[str, Any]:
    """Cheap deterministic grader signals fed into the judge prompt."""

    enrichment = _required_mapping(row, "enrichment")
    chunk_text = _required_str(row, "chunk_text")
    schema = validate_schema_gate(enrichment)
    summary = enrichment.get("summary", "")
    gold = row.get("gold")
    pre_signals: dict[str, Any] = {
        "schema": {"passed": schema.passed, "errors": list(schema.errors)},
        "banned_pattern_hit": False,
        "banned_pattern": None,
    }
    if isinstance(summary, str):
        banned_pattern = find_banned_summary_pattern(summary)
        pre_signals["banned_pattern_hit"] = banned_pattern is not None
        pre_signals["banned_pattern"] = banned_pattern
    if isinstance(gold, Mapping):
        grade = grade_candidate(enrichment, gold, chunk_text=chunk_text)
        pre_signals["grader_overall_score"] = grade.overall_score
        pre_signals["disqualified"] = grade.disqualified
        pre_signals["key_facts_recall"] = grade.key_facts.recall
        pre_signals["entity_name_f1"] = grade.entities.name_f1
        pre_signals["entity_type_strict_f1"] = grade.entities.type_strict_f1
        pre_signals["importance_band_accuracy"] = grade.importance.band_accuracy
        pre_signals["meta_research_passed"] = grade.meta_research.passed
    elif schema.passed:
        entities = score_entities(enrichment, {"gold_entities": []}, chunk_text)
        pre_signals["unsupported_entities"] = list(entities.hallucinated_entities)
    return pre_signals


def calibrate_judge(
    *,
    judge_scores: Sequence[Mapping[str, Any]],
    human_scores: Sequence[Mapping[str, Any]],
    kappa_floor: float = JUDGE_KAPPA_FLOOR,
    spearman_floor: float = JUDGE_SPEARMAN_FLOOR,
) -> CalibrationResult:
    """REC-08 hard gate for judge-vs-human agreement."""

    if len(judge_scores) != len(human_scores):
        raise ValueError("judge_scores and human_scores must have the same length")
    if not judge_scores:
        raise ValueError("calibration requires at least one paired score")

    judge_values = [_composite_from_score(score) for score in judge_scores]
    human_values = [_composite_from_score(score) for score in human_scores]
    kappa = _cohens_kappa(_score_labels(judge_values), _score_labels(human_values))
    spearman_rho = _spearman(judge_values, human_values)
    result = CalibrationResult(
        kappa=kappa,
        spearman_rho=spearman_rho,
        kappa_floor=kappa_floor,
        spearman_floor=spearman_floor,
        sample_size=len(judge_scores),
        quarantined=kappa < kappa_floor or spearman_rho < spearman_floor,
    )
    if result.quarantined:
        raise JudgeQuarantinedError(result)
    return result


def _scored_row(
    row: Mapping[str, Any],
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    judge_model: str,
    judge_cli_version: str,
) -> dict[str, Any]:
    score = response["score"]
    composite = _weighted_composite(score)
    return {
        "chunk_id": row["chunk_id"],
        "variant_id": row["variant_id"],
        "scores": {
            "faithfulness": score["faithfulness"],
            "usefulness": score["usefulness"],
            "entity_coverage": score["entity_coverage"],
            "composite": composite,
        },
        "reason": response["reason"],
        "rationale": response["rationale"],
        "deterministic_pre_signals": request["input"]["deterministic_pre_signals"],
        "request": {
            "judge_prompt_hash": JUDGE_PROMPT_HASH,
            "response_schema": RESPONSE_SCHEMA,
        },
        "judge_metadata": {
            "temperature": DEFAULT_TEMPERATURE,
            "judge_model": judge_model,
            "judge_cli_version": judge_cli_version,
            "judge_prompt_hash": JUDGE_PROMPT_HASH,
            "response_schema": RESPONSE_SCHEMA,
        },
    }


def _persist_judgment(store: ExperimentStore, scored: Mapping[str, Any]) -> None:
    scores = {
        "scores": scored["scores"],
        "reason": scored["reason"],
        "judge_metadata": scored["judge_metadata"],
        "deterministic_pre_signals": scored["deterministic_pre_signals"],
    }
    store.add_judgment(
        chunk_id=str(scored["chunk_id"]),
        variant_id=str(scored["variant_id"]),
        source="llm",
        scores=scores,
        better_option_flag=False,
        rationale=str(scored["rationale"]),
    )


def _validate_response(response: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise JudgeResponseError("judge response must be a JSON object")
    expected_keys = {"reason", "score", "rationale"}
    if set(response) != expected_keys:
        raise JudgeResponseError(f"judge response must contain exactly {sorted(expected_keys)}")
    reason = response["reason"]
    score = response["score"]
    rationale = response["rationale"]
    if not isinstance(reason, Mapping):
        raise JudgeResponseError("judge response reason must be an object")
    if not isinstance(score, Mapping):
        raise JudgeResponseError("judge response score must be an object")
    if not isinstance(rationale, str) or not rationale.strip():
        raise JudgeResponseError("judge response rationale must be a non-empty string")
    for key in SCORE_KEYS:
        if key not in reason or not isinstance(reason[key], str) or not reason[key].strip():
            raise JudgeResponseError(f"judge reason.{key} must be a non-empty string")
        value = score.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
            raise JudgeResponseError(f"judge score.{key} must be an integer from 1 to 5")
    if set(score) != set(SCORE_KEYS):
        raise JudgeResponseError(f"judge score keys must be exactly {list(SCORE_KEYS)}")
    return {"reason": dict(reason), "score": dict(score), "rationale": rationale}


def _weighted_composite(score: Mapping[str, int]) -> float:
    return round(sum(float(score[key]) * SCORE_WEIGHTS[key] for key in SCORE_KEYS), 4)


def _composite_from_score(score: Mapping[str, Any]) -> float:
    if "composite" in score:
        return _bounded_score(score["composite"])
    if all(key in score for key in SCORE_KEYS):
        return _weighted_composite({key: int(score[key]) for key in SCORE_KEYS})
    raise ValueError("score must include composite or faithfulness/usefulness/entity_coverage")


def _bounded_score(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"score must be a finite number, got {value!r}")
    numeric = float(value)
    if not 1.0 <= numeric <= 5.0:
        raise ValueError(f"score must be in [1, 5], got {numeric!r}")
    return numeric


def _score_labels(values: Sequence[float]) -> list[int]:
    return [min(5, max(1, int(round(value)))) for value in values]


def _cohens_kappa(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right):
        raise ValueError("kappa inputs must have the same length")
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / total
    labels = sorted(set(left) | set(right))
    expected = 0.0
    for label in labels:
        left_prob = sum(value == label for value in left) / total
        right_prob = sum(value == label for value in right) / total
        expected += left_prob * right_prob
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("spearman inputs must have the same length")
    if len(left) < 2:
        return 1.0
    return _pearson(_rank(left), _rank(right))


def _rank(values: Sequence[float]) -> list[float]:
    sorted_pairs = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and sorted_pairs[end][0] == sorted_pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for _, original_index in sorted_pairs[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_denominator = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    denominator = left_denominator * right_denominator
    if denominator == 0:
        return 0.0
    rho = numerator / denominator
    if math.isclose(rho, 1.0):
        return 1.0
    if math.isclose(rho, -1.0):
        return -1.0
    return rho


def _iter_jsonl(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        value = json.loads(stripped)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} must be an object")
        yield value


def _required_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value
