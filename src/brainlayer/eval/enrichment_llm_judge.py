"""Offline blinded requests, response parsing and aggregation for saved artifacts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FIELD_NAMES = (
    "summary",
    "tags",
    "importance",
    "intent",
    "epistemic_level",
    "debt_impact",
    "sentiment",
)

FIELD_RUBRIC = {
    "summary": "Specific, faithful, concise capture of the chunk's durable memory value.",
    "tags": "Useful retrieval tags that are specific, normalized, and not noisy.",
    "importance": "Importance score matches operational significance of the source chunk.",
    "intent": "Intent enum matches what the chunk is doing.",
    "epistemic_level": "Epistemic level reflects whether the source is hypothesis, substantiated, or validated.",
    "debt_impact": "Debt impact reflects whether the chunk introduces/resolves/no-ops technical or operational debt.",
    "sentiment": "Sentiment label/score/signals faithfully capture operator affect without over-reading.",
}


JUDGE_PROMPT = """You are judging BrainLayer enrichment quality.

Compare Candidate A and Candidate B against the source chunk only. The source
chunk is authoritative. Do not prefer either candidate because of style or model
identity; identities are hidden.

Score each field from 1 to 5:
1 = wrong/hallucinated/actively harmful for retrieval
2 = weak or materially incomplete
3 = acceptable but missing useful specificity
4 = good, faithful, useful
5 = excellent, faithful, specific, retrieval-useful

Fields:
- summary: {summary_rubric}
- tags: {tags_rubric}
- importance: {importance_rubric}
- intent: {intent_rubric}
- epistemic_level: {epistemic_level_rubric}
- debt_impact: {debt_impact_rubric}
- sentiment: {sentiment_rubric}

Return strict JSON only:
{{
  "field_scores": {{
    "summary": {{"A": 1-5, "B": 1-5, "winner": "A|B|tie", "reason": "short reason"}},
    "tags": {{"A": 1-5, "B": 1-5, "winner": "A|B|tie", "reason": "short reason"}},
    "importance": {{"A": 1-5, "B": 1-5, "winner": "A|B|tie", "reason": "short reason"}},
    "intent": {{"A": 1-5, "B": 1-5, "winner": "A|B|tie", "reason": "short reason"}},
    "epistemic_level": {{"A": 1-5, "B": 1-5, "winner": "A|B|tie", "reason": "short reason"}},
    "debt_impact": {{"A": 1-5, "B": 1-5, "winner": "A|B|tie", "reason": "short reason"}},
    "sentiment": {{"A": 1-5, "B": 1-5, "winner": "A|B|tie", "reason": "short reason"}}
  }},
  "overall": {{"A": 1-5, "B": 1-5, "winner": "A|B|tie", "reason": "short reason"}}
}}
"""


def build_pair_requests(
    selection_rows: Sequence[Mapping[str, Any]],
    local_rows: Mapping[str, Mapping[str, Any]],
    flex_rows: Mapping[str, Mapping[str, Any]],
    *,
    seed: int = 20260615,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    requests: list[dict[str, Any]] = []
    for row in selection_rows:
        chunk_id = str(row["chunk_id"])
        local = local_rows.get(chunk_id)
        flex = flex_rows.get(chunk_id)
        if not local or not flex:
            continue
        order = ["local", "flex"]
        rng.shuffle(order)
        candidates: dict[str, Any] = {}
        label_to_system: dict[str, str] = {}
        for label, system in zip(("A", "B"), order, strict=True):
            source_row = local if system == "local" else flex
            candidates[label] = {
                "system": system,
                "backend": source_row.get("backend"),
                "model": source_row.get("model"),
                "enrichment": source_row.get("enrichment") or {},
            }
            label_to_system[label] = system
        requests.append(
            {
                "chunk_id": chunk_id,
                "source": {
                    "chunk_text": row["chunk_text"],
                    "pure_metadata": row.get("pure_metadata") or {},
                },
                "fields": list(FIELD_NAMES),
                "candidates": candidates,
                "label_to_system": label_to_system,
            }
        )
    return requests


def parse_judge_response(text: str) -> dict[str, Any]:
    parsed = _extract_json_object(text)
    _validate_judge_payload(parsed)
    return parsed


def aggregate_judgments(judgments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {field: _empty_metric() for field in FIELD_NAMES}
    overall = _empty_metric()
    for judgment in judgments:
        mapping = judgment["label_to_system"]
        for field in FIELD_NAMES:
            _add_metric(fields[field], judgment["field_scores"][field], mapping)
        _add_metric(overall, judgment["overall"], mapping)

    return {
        "count": len(judgments),
        "fields": {field: _finalize_metric(metric) for field, metric in fields.items()},
        "overall": _finalize_metric(overall),
    }


def _build_prompt(request: Mapping[str, Any]) -> str:
    rubric = JUDGE_PROMPT.format(**{f"{field}_rubric": FIELD_RUBRIC[field] for field in FIELD_NAMES})
    payload = {
        "chunk_id": request["chunk_id"],
        "source": request["source"],
        "candidate_A": request["candidates"]["A"]["enrichment"],
        "candidate_B": request["candidates"]["B"]["enrichment"],
    }
    return rubric + "\n\nJUDGE_INPUT:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _empty_metric() -> dict[str, Any]:
    return {
        "scores": {"local": [], "flex": []},
        "winner_counts": {"local": 0, "flex": 0, "tie": 0},
    }


def _add_metric(metric: dict[str, Any], scores: Mapping[str, Any], label_to_system: Mapping[str, str]) -> None:
    for label in ("A", "B"):
        system = label_to_system[label]
        metric["scores"][system].append(float(scores[label]))
    winner = scores.get("winner")
    if winner == "tie":
        metric["winner_counts"]["tie"] += 1
    elif winner in ("A", "B"):
        metric["winner_counts"][label_to_system[winner]] += 1
    else:
        raise ValueError(f"invalid winner: {winner!r}")


def _finalize_metric(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean_scores": {
            system: round(sum(values) / len(values), 4) if values else 0.0
            for system, values in metric["scores"].items()
        },
        "winner_counts": dict(metric["winner_counts"]),
        "score_delta_local_minus_flex": round(
            _mean(metric["scores"]["local"]) - _mean(metric["scores"]["flex"]),
            4,
        ),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _verdict(summary: Mapping[str, Any]) -> str:
    overall = summary.get("overall", {})
    delta = float(overall.get("score_delta_local_minus_flex", 0.0))
    counts = overall.get("winner_counts", {})
    if delta >= 0.25 and counts.get("local", 0) >= counts.get("flex", 0):
        return "local-better-than-flex"
    if delta <= -0.25 and counts.get("flex", 0) > counts.get("local", 0):
        return "local-below-flex"
    return "local-roughly-matches-flex"


def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("empty judge response")
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
    raise ValueError("judge response did not contain a JSON object")


def _validate_judge_payload(payload: Mapping[str, Any]) -> None:
    field_scores = payload.get("field_scores")
    overall = payload.get("overall")
    if not isinstance(field_scores, Mapping) or not isinstance(overall, Mapping):
        raise ValueError("judge response must contain field_scores and overall")
    for field in FIELD_NAMES:
        if field not in field_scores:
            raise ValueError(f"missing field score: {field}")
        _validate_score_block(field_scores[field], f"field_scores.{field}")
    _validate_score_block(overall, "overall")


def _validate_score_block(block: Any, label: str) -> None:
    if not isinstance(block, Mapping):
        raise ValueError(f"{label} must be an object")
    for candidate in ("A", "B"):
        value = block.get(candidate)
        if not isinstance(value, (int, float)) or not 1 <= float(value) <= 5:
            raise ValueError(f"{label}.{candidate} must be a score from 1 to 5")
    if block.get("winner") not in {"A", "B", "tie"}:
        raise ValueError(f"{label}.winner must be A, B, or tie")
    if not isinstance(block.get("reason"), str) or not block["reason"].strip():
        raise ValueError(f"{label}.reason must be a non-empty string")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rows_by_chunk_id(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["chunk_id"]): row for row in rows}
