"""Caller-supplied adjudication contracts and offline verdict parsing."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

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


def get_correction_judge(*, store: Any | None = None) -> CorrectionJudge:
    """Fail loudly for legacy callers; no built-in model judge remains."""
    raise RuntimeError(
        "Automatic fact correction judging is retired. "
        "Pass a caller-supplied correction_judge to refresh_entity_facts instead."
    )
