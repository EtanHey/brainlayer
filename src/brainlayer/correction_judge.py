"""LLM-backed adjudication for conflicting entity facts."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from .enrichment_controller import (
    GEMINI_REALTIME_MODEL,
    _build_gemini_config,
    _generate_content_with_rate_limit,
    _get_gemini_client,
    _get_store_rate_limiter,
)

JudgeAction = Literal["supersede", "merge", "noise"]


@dataclass(frozen=True)
class Verdict:
    action: JudgeAction
    confidence: float
    reasoning: str


class CorrectionJudge(ABC):
    @abstractmethod
    def judge(
        self,
        entity: str | dict[str, Any],
        new_fact: str | dict[str, Any],
        conflicting_fact: str | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> Verdict:
        """Return an adjudication verdict for two possibly conflicting facts."""


JUDGE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["supersede", "merge", "noise"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["action", "confidence", "reasoning"],
}


def _fact_text(fact: str | dict[str, Any]) -> str:
    if isinstance(fact, str):
        return fact
    value = fact.get("fact_text") or fact.get("text") or ""
    return str(value)


def _entity_name(entity: str | dict[str, Any]) -> str:
    if isinstance(entity, str):
        return entity
    value = entity.get("name") or entity.get("entity_id") or entity.get("id") or ""
    return str(value)


def _build_judge_prompt(
    entity: str | dict[str, Any],
    new_fact: str | dict[str, Any],
    conflicting_fact: str | dict[str, Any],
    context: dict[str, Any] | None,
) -> str:
    payload = {
        "entity": _entity_name(entity),
        "new_fact": _fact_text(new_fact),
        "existing_active_fact": _fact_text(conflicting_fact),
        "context": context or {},
    }
    return (
        "You adjudicate conflicts between active BrainLayer entity facts.\n"
        "Return strict JSON only with action, confidence, and reasoning.\n"
        "Choose supersede when the new fact corrects or replaces the existing fact.\n"
        "Choose merge when both facts should collapse into one memory.\n"
        "Choose noise when the new assertion is irrelevant, low-signal, or not a correction.\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def _build_judge_config() -> dict[str, Any]:
    config = dict(_build_gemini_config())
    config["response_schema"] = JUDGE_RESPONSE_SCHEMA
    return config


def _coerce_verdict(payload: Any) -> Verdict:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("correction judge response must be a JSON object")

    action = str(payload.get("action", "")).strip().lower()
    if action not in {"supersede", "merge", "noise"}:
        raise ValueError(f"invalid correction judge action: {action!r}")

    raw_confidence = payload.get("confidence")
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"correction judge confidence must be a number, got {raw_confidence!r}") from exc
    if confidence < 0 or confidence > 1:
        raise ValueError("correction judge confidence must be between 0 and 1")

    reasoning = str(payload.get("reasoning") or "").strip()
    if not reasoning:
        raise ValueError("correction judge reasoning is required")

    return Verdict(action=action, confidence=confidence, reasoning=reasoning)  # type: ignore[arg-type]


def _response_payload(response: Any) -> Any:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed
    text = getattr(response, "text", None)
    if text is not None:
        return text
    return response


class GeminiCorrectionJudge(CorrectionJudge):
    def __init__(self, *, client: Any | None = None, rate_limiter: Any | None = None) -> None:
        self._client = client
        self._rate_limiter = rate_limiter

    def judge(
        self,
        entity: str | dict[str, Any],
        new_fact: str | dict[str, Any],
        conflicting_fact: str | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> Verdict:
        client = self._client or _get_gemini_client()
        prompt = _build_judge_prompt(entity, new_fact, conflicting_fact, context)
        response = _generate_content_with_rate_limit(
            client,
            GEMINI_REALTIME_MODEL,
            prompt,
            _build_judge_config(),
            self._rate_limiter,
        )
        return _coerce_verdict(_response_payload(response))


class LocalCorrectionJudge(CorrectionJudge):
    def judge(
        self,
        entity: str | dict[str, Any],
        new_fact: str | dict[str, Any],
        conflicting_fact: str | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> Verdict:
        raise NotImplementedError("LocalCorrectionJudge is a swap-to-local stub.")


def get_correction_judge(*, store: Any | None = None) -> CorrectionJudge:
    backend = os.environ.get("BRAINLAYER_JUDGE_BACKEND", "gemini").strip().lower()
    if backend == "gemini":
        rate_limiter = _get_store_rate_limiter(store) if store is not None else None
        return GeminiCorrectionJudge(rate_limiter=rate_limiter)
    if backend == "local":
        return LocalCorrectionJudge()
    raise ValueError(f"Unsupported BRAINLAYER_JUDGE_BACKEND: {backend!r}")
