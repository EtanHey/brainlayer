"""Reference-only Wave 5 benchmark harness.

This module deliberately lives under ``tests/``.  It specifies the contracts that
future production implementations must satisfy without adding a runtime feature.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

ETAN_DIGEST_REQUIREMENT = (
    "I should get a summarized version of what's superseding what, and be the human in the loop for some of it"
)
CORRECTION_GOLD_PATH = Path(__file__).parent / "fixtures" / "wave5_corrections_gold.json"


@dataclass(frozen=True)
class OccurrenceEvent:
    fingerprint: str
    scope: str
    session_id: str
    occurred_at: datetime
    severity: int


@dataclass(frozen=True)
class OccurrenceReceipt:
    occurrence_id: str
    alert: str | None
    event_count: int
    session_ids: tuple[str, ...]


@dataclass(frozen=True)
class DailyOccurrence:
    day: date
    occurrence_id: str
    event_count: int


@dataclass(frozen=True)
class CorrectionGoldCase:
    case_id: str
    source_pointer: str
    source_excerpt: str
    historical_claim: str
    corrected_claim: str
    scope: str
    entity: str
    attribute: str
    expected_decision: str
    expected_digest: str | None


@dataclass(frozen=True)
class CorrectionThresholds:
    min_auto_confidence: float
    min_precision: float
    min_recall: float
    max_false_positives: int
    min_digest_fidelity: float


@dataclass(frozen=True)
class CorrectionGold:
    cases: tuple[CorrectionGoldCase, ...]
    thresholds: CorrectionThresholds
    etan_digest_requirement: str
    rollback_rules: tuple[str, ...]


@dataclass(frozen=True)
class CandidateCorrection:
    case_id: str
    decision: str
    confidence: float
    digest: str | None


@dataclass(frozen=True)
class CorrectionBenchReport:
    precision: float
    recall: float
    false_positives: int
    digest_fidelity: float
    promote_to_auto: bool
    rollback_required: bool
    blockers: tuple[str, ...]
    etan_digest_requirement: str
    rollback_rules: tuple[str, ...]


class OccurrenceLedger:
    """In-memory oracle for the immutable occurrence-ledger contract."""

    def __init__(self) -> None:
        self.events: list[OccurrenceEvent] = []
        self.receipts: list[OccurrenceReceipt] = []

    @staticmethod
    def _occurrence_id(event: OccurrenceEvent) -> str:
        identity = f"{event.scope}\0{event.fingerprint}".encode()
        return hashlib.sha256(identity).hexdigest()

    def record(self, event: OccurrenceEvent) -> OccurrenceReceipt:
        if not event.fingerprint or not event.scope:
            raise ValueError("fingerprint and scope are required")
        if event.occurred_at.tzinfo is None or event.occurred_at.tzinfo != UTC:
            raise ValueError("occurred_at must be an explicit UTC timestamp")
        occurrence_id = self._occurrence_id(event)
        prior = [stored for stored in self.events if self._occurrence_id(stored) == occurrence_id]
        if not prior:
            alert = "new"
        elif event.severity > max(stored.severity for stored in prior):
            alert = "escalated"
        else:
            alert = None

        self.events.append(event)
        matching = [stored for stored in self.events if self._occurrence_id(stored) == occurrence_id]
        receipt = OccurrenceReceipt(
            occurrence_id=occurrence_id,
            alert=alert,
            event_count=len(matching),
            session_ids=tuple(dict.fromkeys(stored.session_id for stored in matching)),
        )
        self.receipts.append(receipt)
        return receipt

    def weave_accumulation(self, *, through: date) -> tuple[DailyOccurrence, ...]:
        daily_counts: dict[tuple[date, str], int] = {}
        for event in self.events:
            event_day = event.occurred_at.date()
            if event_day > through:
                continue
            key = (event_day, self._occurrence_id(event))
            daily_counts[key] = daily_counts.get(key, 0) + 1
        return tuple(
            DailyOccurrence(day=day, occurrence_id=occurrence_id, event_count=count)
            for (day, occurrence_id), count in sorted(daily_counts.items())
        )


def load_correction_gold(path: Path = CORRECTION_GOLD_PATH) -> CorrectionGold:
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = CorrectionThresholds(**payload["thresholds"])
    cases = tuple(CorrectionGoldCase(**case) for case in payload["cases"])
    return CorrectionGold(
        cases=cases,
        thresholds=thresholds,
        etan_digest_requirement=payload["etan_digest_requirement"],
        rollback_rules=tuple(payload["rollback_rules"]),
    )


def score_correction_gold(
    candidates: list[CandidateCorrection],
    *,
    gold: CorrectionGold | None = None,
) -> CorrectionBenchReport:
    benchmark = gold or load_correction_gold()
    by_id = {candidate.case_id: candidate for candidate in candidates}
    if set(by_id) != {case.case_id for case in benchmark.cases}:
        raise ValueError("candidate case IDs must exactly match the frozen correction gold set")
    if len(by_id) != len(candidates):
        raise ValueError("candidate case IDs must be unique")
    for candidate in candidates:
        if candidate.decision not in {"supersede", "keep_both"}:
            raise ValueError(f"invalid candidate decision: {candidate.decision!r}")
        if not 0.0 <= candidate.confidence <= 1.0:
            raise ValueError("candidate confidence must be between 0 and 1")

    true_positives = false_positives = false_negatives = 0
    exact_digests = 0
    for case in benchmark.cases:
        candidate = by_id[case.case_id]
        predicts_auto = (
            candidate.decision == "supersede" and candidate.confidence >= benchmark.thresholds.min_auto_confidence
        )
        expected_auto = case.expected_decision == "supersede"
        if predicts_auto and expected_auto:
            true_positives += 1
            exact_digests += candidate.digest == case.expected_digest
        elif predicts_auto:
            false_positives += 1
        elif expected_auto:
            false_negatives += 1

    predicted_positive = true_positives + false_positives
    actual_positive = true_positives + false_negatives
    precision = true_positives / predicted_positive if predicted_positive else 0.0
    recall = true_positives / actual_positive if actual_positive else 0.0
    digest_fidelity = exact_digests / true_positives if true_positives else 0.0

    blockers: list[str] = []
    thresholds = benchmark.thresholds
    if false_positives > thresholds.max_false_positives:
        blockers.append("false-positive budget")
    if precision < thresholds.min_precision:
        blockers.append("precision threshold")
    if recall < thresholds.min_recall:
        blockers.append("recall threshold")
    if digest_fidelity < thresholds.min_digest_fidelity:
        blockers.append("digest fidelity")

    promote = not blockers
    return CorrectionBenchReport(
        precision=precision,
        recall=recall,
        false_positives=false_positives,
        digest_fidelity=digest_fidelity,
        promote_to_auto=promote,
        rollback_required=not promote,
        blockers=tuple(blockers),
        etan_digest_requirement=benchmark.etan_digest_requirement,
        rollback_rules=benchmark.rollback_rules,
    )
