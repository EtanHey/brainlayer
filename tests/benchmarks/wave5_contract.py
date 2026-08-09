"""Reference-only Wave 5 benchmark harness.

This module deliberately lives under ``tests/``.  It specifies the contracts that
future production implementations must satisfy without adding a runtime feature.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Protocol

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
    session_ids: tuple[str, ...]


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
class CorrectionInputCase:
    case_id: str
    source_pointer: str
    source_excerpt: str
    historical_claim: str
    corrected_claim: str
    scope: str
    entity: str
    attribute: str


@dataclass(frozen=True)
class CorrectionThresholds:
    min_auto_confidence: float
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
    proposed_digests: tuple[str, ...]
    etan_digest_requirement: str
    rollback_rules: tuple[str, ...]


class Ledger(Protocol):
    """Seam implemented by the reference oracle and future production ledger."""

    def record(self, event: OccurrenceEvent) -> OccurrenceReceipt: ...

    def weave_accumulation(self, *, through: date) -> tuple[DailyOccurrence, ...]: ...


LedgerFactory = Callable[[], Ledger]
CandidateProducer = Callable[[tuple[CorrectionInputCase, ...]], list[CandidateCorrection]]


def correction_inputs(gold: CorrectionGold) -> tuple[CorrectionInputCase, ...]:
    return tuple(
        CorrectionInputCase(
            case_id=case.case_id,
            source_pointer=case.source_pointer,
            source_excerpt=case.source_excerpt,
            historical_claim=case.historical_claim,
            corrected_claim=case.corrected_claim,
            scope=case.scope,
            entity=case.entity,
            attribute=case.attribute,
        )
        for case in gold.cases
    )


def oracle_candidate_producer(
    cases: tuple[CorrectionInputCase, ...],
) -> list[CandidateCorrection]:
    gold_by_id = {case.case_id: case for case in load_correction_gold().cases}
    return [
        CandidateCorrection(
            case_id=case.case_id,
            decision=gold_by_id[case.case_id].expected_decision,
            confidence=0.99,
            digest=gold_by_id[case.case_id].expected_digest
            if gold_by_id[case.case_id].expected_decision == "supersede"
            else None,
        )
        for case in cases
    ]


class OccurrenceLedger:
    """In-memory oracle for the immutable occurrence-ledger contract."""

    def __init__(self) -> None:
        self.events: list[OccurrenceEvent] = []
        self.receipts: list[OccurrenceReceipt] = []
        self._weave_delivered: set[int] = set()

    @staticmethod
    def _occurrence_id(event: OccurrenceEvent) -> str:
        identity = json.dumps(
            [event.scope, event.fingerprint],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(identity).hexdigest()

    def record(self, event: OccurrenceEvent) -> OccurrenceReceipt:
        if (
            type(event.fingerprint) is not str
            or type(event.scope) is not str
            or not event.fingerprint.strip()
            or not event.scope.strip()
        ):
            raise ValueError("fingerprint and scope are required")
        if type(event.session_id) is not str or not event.session_id.strip():
            raise ValueError("session_id is required")
        if (
            type(event.occurred_at) is not datetime
            or event.occurred_at.tzinfo is None
            or event.occurred_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("occurred_at must be an explicit UTC timestamp")
        if type(event.severity) is not int or event.severity < 0:
            raise ValueError("severity must be a non-negative integer")
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
        daily_events: dict[tuple[date, str], list[OccurrenceEvent]] = {}
        delivered_now: list[int] = []
        for index, event in enumerate(self.events):
            if index in self._weave_delivered:
                continue
            event_day = event.occurred_at.date()
            if event_day > through:
                continue
            key = (event_day, self._occurrence_id(event))
            daily_events.setdefault(key, []).append(event)
            delivered_now.append(index)
        feed = tuple(
            DailyOccurrence(
                day=day,
                occurrence_id=occurrence_id,
                event_count=len(events),
                session_ids=tuple(dict.fromkeys(event.session_id for event in events)),
            )
            for (day, occurrence_id), events in sorted(daily_events.items())
        )
        self._weave_delivered.update(delivered_now)
        return feed


def load_correction_gold(path: Path = CORRECTION_GOLD_PATH) -> CorrectionGold:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported correction gold schema version")
    thresholds = CorrectionThresholds(**payload["thresholds"])
    fractional_thresholds = (
        thresholds.min_auto_confidence,
        thresholds.min_recall,
        thresholds.min_digest_fidelity,
    )
    if any(
        type(value) not in {int, float} or not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in fractional_thresholds
    ):
        raise ValueError("fractional correction thresholds must be finite values between 0 and 1")
    if type(thresholds.max_false_positives) is not int or thresholds.max_false_positives < 0:
        raise ValueError("max false-positive correction threshold must be a non-negative integer")
    cases = tuple(CorrectionGoldCase(**case) for case in payload["cases"])
    for case in cases:
        if case.expected_decision not in {"supersede", "keep_both"}:
            raise ValueError(f"invalid expected_decision for {case.case_id}")
        required_text = (
            case.case_id,
            case.source_pointer,
            case.source_excerpt,
            case.historical_claim,
            case.corrected_claim,
            case.scope,
            case.entity,
            case.attribute,
        )
        if not all(value.strip() for value in required_text):
            raise ValueError(f"non-empty case content required for {case.case_id}")
        source_path, separator, excerpt_digest = case.source_pointer.partition("#excerpt-sha256:")
        expected_excerpt_digest = hashlib.sha256(case.source_excerpt.encode()).hexdigest()
        if not source_path or not separator or excerpt_digest != expected_excerpt_digest:
            raise ValueError(f"stable source excerpt anchor required for {case.case_id}")
        if case.expected_decision == "supersede":
            if case.expected_digest is None:
                raise ValueError(f"supersede digest required for {case.case_id}")
            if not all(field in case.expected_digest for field in (case.entity, case.attribute, case.source_pointer)):
                raise ValueError(f"digest must name entity, attribute, and source for {case.case_id}")
        elif case.expected_digest is not None:
            raise ValueError(f"keep_both digest must be null for {case.case_id}")
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
    human_approved: bool = False,
) -> CorrectionBenchReport:
    if type(human_approved) is not bool:
        raise ValueError("human_approved must be a boolean")
    benchmark = gold or load_correction_gold()
    by_id = {candidate.case_id: candidate for candidate in candidates}
    if set(by_id) != {case.case_id for case in benchmark.cases}:
        raise ValueError("candidate case IDs must exactly match the frozen correction gold set")
    if len(by_id) != len(candidates):
        raise ValueError("candidate case IDs must be unique")
    for candidate in candidates:
        if candidate.decision not in {"supersede", "keep_both"}:
            raise ValueError(f"invalid candidate decision: {candidate.decision!r}")
        if (
            type(candidate.confidence) not in {int, float}
            or not math.isfinite(candidate.confidence)
            or not 0.0 <= candidate.confidence <= 1.0
        ):
            raise ValueError("candidate confidence must be a finite number between 0 and 1")

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

    metric_blockers: list[str] = []
    thresholds = benchmark.thresholds
    if false_positives > thresholds.max_false_positives:
        metric_blockers.append("false-positive budget")
    if recall < thresholds.min_recall:
        metric_blockers.append("recall threshold")
    if digest_fidelity < thresholds.min_digest_fidelity:
        metric_blockers.append("digest fidelity")

    blockers = [*metric_blockers]
    if not human_approved:
        blockers.append("human approval")
    promote = not blockers
    proposed_digests = tuple(
        by_id[case.case_id].digest
        for case in benchmark.cases
        if by_id[case.case_id].decision == "supersede" and by_id[case.case_id].digest is not None
    )
    return CorrectionBenchReport(
        precision=precision,
        recall=recall,
        false_positives=false_positives,
        digest_fidelity=digest_fidelity,
        promote_to_auto=promote,
        rollback_required=bool(metric_blockers),
        blockers=tuple(blockers),
        proposed_digests=proposed_digests,
        etan_digest_requirement=benchmark.etan_digest_requirement,
        rollback_rules=benchmark.rollback_rules,
    )
