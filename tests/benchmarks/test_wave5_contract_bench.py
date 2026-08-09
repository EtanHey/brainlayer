import hashlib
import json
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tests.benchmarks.wave5_contract import (
    CORRECTION_GOLD_PATH,
    ETAN_DIGEST_REQUIREMENT,
    CandidateCorrection,
    CandidateProducer,
    CorrectionInputCase,
    DailyOccurrence,
    Ledger,
    LedgerFactory,
    OccurrenceEvent,
    correction_inputs,
    load_correction_gold,
    oracle_candidate_producer,
    score_correction_gold,
)

FIXED_NOW = datetime(2026, 8, 9, 9, 0, tzinfo=UTC)


class _ExplodingExternalLedger:
    def record(self, event: OccurrenceEvent) -> None:
        raise AssertionError("external ledger factory selected")

    def weave_accumulation(self, *, through: date) -> tuple[DailyOccurrence, ...]:
        return ()


def _broken_candidate_producer(
    cases: tuple[CorrectionInputCase, ...],
) -> list[CandidateCorrection]:
    print("external candidate producer selected")
    return [CandidateCorrection(case.case_id, "keep_both", 0.99, None) for case in cases]


def test_external_ledger_factory_option_reaches_contract() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__)),
            "-q",
            "-k",
            "occurrence_identity_includes_semantic_fingerprint_and_scope",
            "--wave5-ledger-factory",
            "tests.benchmarks.test_wave5_contract_bench:_ExplodingExternalLedger",
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "external ledger factory selected" in result.stdout


def test_external_candidate_producer_option_reaches_promotion_benchmark() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__)),
            "-q",
            "-k",
            "candidate_producer_meets_promotion_thresholds_and_false_positive_budget",
            "--wave5-candidate-producer",
            "tests.benchmarks.test_wave5_contract_bench:_broken_candidate_producer",
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "external candidate producer selected" in result.stdout


def test_occurrence_identity_includes_semantic_fingerprint_and_scope(
    ledger_factory: LedgerFactory,
) -> None:
    ledger: Ledger = ledger_factory()

    first = ledger.record(
        OccurrenceEvent(
            fingerprint="sqlite-wal-checkpoint-starvation",
            scope="host:m2/service:brainlayer-watch",
            session_id="session-a",
            occurred_at=FIXED_NOW,
            severity=2,
        )
    )
    other_scope = ledger.record(
        OccurrenceEvent(
            fingerprint="sqlite-wal-checkpoint-starvation",
            scope="host:m1/service:brainlayer-watch",
            session_id="session-a",
            occurred_at=FIXED_NOW,
            severity=2,
        )
    )
    other_fingerprint = ledger.record(
        OccurrenceEvent(
            fingerprint="sqlite-busy-write-contention",
            scope="host:m2/service:brainlayer-watch",
            session_id="session-a",
            occurred_at=FIXED_NOW,
            severity=2,
        )
    )

    canonical_identity = json.dumps(
        ["host:m2/service:brainlayer-watch", "sqlite-wal-checkpoint-starvation"],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert first.occurrence_id == hashlib.sha256(canonical_identity).hexdigest()
    assert first.occurrence_id != other_scope.occurrence_id
    assert first.occurrence_id != other_fingerprint.occurrence_id
    assert other_scope.alert == "new"
    assert other_scope.event_count == 1
    assert other_scope.session_ids == ("session-a",)
    assert other_fingerprint.alert == "new"
    assert other_fingerprint.event_count == 1
    assert other_fingerprint.session_ids == ("session-a",)


def test_occurrence_identity_has_no_delimiter_collision(
    ledger_factory: LedgerFactory,
) -> None:
    ledger = ledger_factory()
    first = ledger.record(OccurrenceEvent("b\0c", "a", "session-a", FIXED_NOW, 1))
    second = ledger.record(OccurrenceEvent("c", "a\0b", "session-b", FIXED_NOW, 1))

    assert first.occurrence_id != second.occurrence_id
    assert (first.alert, first.event_count, first.session_ids) == (
        "new",
        1,
        ("session-a",),
    )
    assert (second.alert, second.event_count, second.session_ids) == (
        "new",
        1,
        ("session-b",),
    )


def test_cross_session_repeats_dedupe_without_losing_event_provenance(
    ledger_factory: LedgerFactory,
) -> None:
    ledger = ledger_factory()
    first = ledger.record(
        OccurrenceEvent(
            fingerprint="sqlite-wal-checkpoint-starvation",
            scope="host:m2/service:brainlayer-watch",
            session_id="session-a",
            occurred_at=FIXED_NOW,
            severity=2,
        )
    )
    repeated = ledger.record(
        OccurrenceEvent(
            fingerprint="sqlite-wal-checkpoint-starvation",
            scope="host:m2/service:brainlayer-watch",
            session_id="session-b",
            occurred_at=FIXED_NOW + timedelta(minutes=5),
            severity=2,
        )
    )

    assert first.alert == "new"
    assert repeated.alert is None
    assert repeated.occurrence_id == first.occurrence_id
    assert repeated.event_count == 2
    assert repeated.session_ids == ("session-a", "session-b")
    same_session = ledger.record(
        OccurrenceEvent(
            fingerprint="sqlite-wal-checkpoint-starvation",
            scope="host:m2/service:brainlayer-watch",
            session_id="session-b",
            occurred_at=FIXED_NOW + timedelta(minutes=10),
            severity=2,
        )
    )
    assert same_session.alert is None
    assert same_session.event_count == 3
    assert same_session.session_ids == ("session-a", "session-b")
    same_session_escalated = ledger.record(
        OccurrenceEvent(
            fingerprint="sqlite-wal-checkpoint-starvation",
            scope="host:m2/service:brainlayer-watch",
            session_id="session-b",
            occurred_at=FIXED_NOW + timedelta(minutes=15),
            severity=3,
        )
    )
    assert (
        same_session_escalated.alert,
        same_session_escalated.event_count,
        same_session_escalated.session_ids,
    ) == ("escalated", 4, ("session-a", "session-b"))


def test_only_new_or_escalating_occurrences_alert(ledger_factory: LedgerFactory) -> None:
    ledger = ledger_factory()
    base = OccurrenceEvent(
        fingerprint="sqlite-wal-checkpoint-starvation",
        scope="host:m2/service:brainlayer-watch",
        session_id="session-a",
        occurred_at=FIXED_NOW,
        severity=2,
    )

    new = ledger.record(base)
    assert new.alert == "new"
    assert new.event_count == 1
    assert new.session_ids == ("session-a",)
    lower = ledger.record(OccurrenceEvent(**{**base.__dict__, "session_id": "session-b", "severity": 1}))
    assert (lower.alert, lower.event_count, lower.session_ids) == (
        None,
        2,
        ("session-a", "session-b"),
    )
    equal = ledger.record(OccurrenceEvent(**{**base.__dict__, "session_id": "session-c", "severity": 2}))
    assert (equal.alert, equal.event_count, equal.session_ids) == (
        None,
        3,
        ("session-a", "session-b", "session-c"),
    )
    escalated = ledger.record(OccurrenceEvent(**{**base.__dict__, "session_id": "session-d", "severity": 3}))
    assert escalated.alert == "escalated"
    assert escalated.event_count == 4
    assert escalated.session_ids == (
        "session-a",
        "session-b",
        "session-c",
        "session-d",
    )
    repeated_max = ledger.record(OccurrenceEvent(**{**base.__dict__, "session_id": "session-e", "severity": 3}))
    assert (repeated_max.alert, repeated_max.event_count, repeated_max.session_ids) == (
        None,
        5,
        ("session-a", "session-b", "session-c", "session-d", "session-e"),
    )


def test_weave_feed_accumulates_by_event_day_until_weave_is_invoked(
    ledger_factory: LedgerFactory,
) -> None:
    ledger = ledger_factory()
    occurrence_id = ""
    cross_day_receipts = []
    for session_id, offset in (
        ("session-a", timedelta()),
        ("session-b", timedelta(days=1)),
        ("session-c", timedelta(days=2)),
    ):
        receipt = ledger.record(
            OccurrenceEvent(
                fingerprint="sqlite-wal-checkpoint-starvation",
                scope="host:m2/service:brainlayer-watch",
                session_id=session_id,
                occurred_at=FIXED_NOW + offset,
                severity=2,
            )
        )
        cross_day_receipts.append(receipt)
        occurrence_id = occurrence_id or receipt.occurrence_id
    assert cross_day_receipts[1].alert is None
    assert cross_day_receipts[1].event_count == 2
    assert cross_day_receipts[1].session_ids == ("session-a", "session-b")
    assert cross_day_receipts[2].alert is None
    assert cross_day_receipts[2].event_count == 3
    assert cross_day_receipts[2].session_ids == (
        "session-a",
        "session-b",
        "session-c",
    )
    ledger.record(
        OccurrenceEvent(
            fingerprint="sqlite-wal-checkpoint-starvation",
            scope="host:m2/service:brainlayer-watch",
            session_id="session-a",
            occurred_at=FIXED_NOW + timedelta(minutes=5),
            severity=2,
        )
    )
    ledger.record(
        OccurrenceEvent(
            fingerprint="sqlite-wal-checkpoint-starvation",
            scope="host:m2/service:brainlayer-watch",
            session_id="session-d",
            occurred_at=FIXED_NOW + timedelta(minutes=10),
            severity=2,
        )
    )
    other_occurrence = ledger.record(
        OccurrenceEvent(
            fingerprint="sqlite-busy-write-contention",
            scope="host:m2/service:brainlayer-watch",
            session_id="session-e",
            occurred_at=FIXED_NOW + timedelta(minutes=15),
            severity=2,
        )
    )
    assert (
        other_occurrence.alert,
        other_occurrence.event_count,
        other_occurrence.session_ids,
    ) == ("new", 1, ("session-e",))

    feed = ledger.weave_accumulation(through=date(2026, 8, 10))

    assert len(feed) == 3
    feed_by_key = {(bucket.day, bucket.occurrence_id): bucket for bucket in feed}
    assert set(feed_by_key) == {
        (date(2026, 8, 9), occurrence_id),
        (date(2026, 8, 9), other_occurrence.occurrence_id),
        (date(2026, 8, 10), occurrence_id),
    }
    primary_day = feed_by_key[(date(2026, 8, 9), occurrence_id)]
    assert (primary_day.event_count, primary_day.session_ids) == (
        3,
        ("session-a", "session-d"),
    )
    other_day = feed_by_key[(date(2026, 8, 9), other_occurrence.occurrence_id)]
    assert (other_day.event_count, other_day.session_ids) == (1, ("session-e",))
    next_day = feed_by_key[(date(2026, 8, 10), occurrence_id)]
    assert (next_day.event_count, next_day.session_ids) == (1, ("session-b",))
    assert ledger.weave_accumulation(through=date(2026, 8, 10)) == ()
    late_receipt = ledger.record(
        OccurrenceEvent(
            fingerprint="sqlite-wal-checkpoint-starvation",
            scope="host:m2/service:brainlayer-watch",
            session_id="session-late",
            occurred_at=FIXED_NOW + timedelta(minutes=20),
            severity=3,
        )
    )
    assert (
        late_receipt.occurrence_id,
        late_receipt.alert,
        late_receipt.event_count,
        late_receipt.session_ids,
    ) == (
        occurrence_id,
        "escalated",
        6,
        ("session-a", "session-b", "session-c", "session-d", "session-late"),
    )
    assert ledger.weave_accumulation(through=date(2026, 8, 10)) == (
        DailyOccurrence(
            day=date(2026, 8, 9),
            occurrence_id=occurrence_id,
            event_count=1,
            session_ids=("session-late",),
        ),
    )
    assert ledger.weave_accumulation(through=date(2026, 8, 10)) == ()
    future_feed = ledger.weave_accumulation(through=date(2026, 8, 11))
    assert future_feed == (
        DailyOccurrence(
            day=date(2026, 8, 11),
            occurrence_id=occurrence_id,
            event_count=1,
            session_ids=("session-c",),
        ),
    )
    assert ledger.weave_accumulation(through=date(2026, 8, 11)) == ()


def test_occurrence_ledger_rejects_ambiguous_timestamp_or_identity(
    ledger_factory: LedgerFactory,
) -> None:
    ledger = ledger_factory()
    valid = {
        "fingerprint": "sqlite-wal-checkpoint-starvation",
        "scope": "host:m2/service:brainlayer-watch",
        "session_id": "session-a",
        "occurred_at": FIXED_NOW,
        "severity": 2,
    }

    with pytest.raises(ValueError, match="UTC timestamp"):
        ledger.record(OccurrenceEvent(**{**valid, "occurred_at": FIXED_NOW.replace(tzinfo=None)}))
    with pytest.raises(ValueError, match="UTC timestamp"):
        ledger.record(
            OccurrenceEvent(
                **{
                    **valid,
                    "occurred_at": FIXED_NOW.astimezone(timezone(timedelta(hours=3))),
                }
            )
        )
    with pytest.raises(ValueError, match="fingerprint and scope"):
        ledger.record(OccurrenceEvent(**{**valid, "fingerprint": ""}))
    with pytest.raises(ValueError, match="fingerprint and scope"):
        ledger.record(OccurrenceEvent(**{**valid, "scope": ""}))
    with pytest.raises(ValueError, match="fingerprint and scope"):
        ledger.record(OccurrenceEvent(**{**valid, "fingerprint": " \t"}))
    with pytest.raises(ValueError, match="fingerprint and scope"):
        ledger.record(OccurrenceEvent(**{**valid, "scope": " \t"}))
    with pytest.raises(ValueError, match="session_id"):
        ledger.record(OccurrenceEvent(**{**valid, "session_id": ""}))
    with pytest.raises(ValueError, match="session_id"):
        ledger.record(OccurrenceEvent(**{**valid, "session_id": " \t"}))
    assert ledger.weave_accumulation(through=FIXED_NOW.date()) == ()


@pytest.mark.parametrize("severity", [True, "2", 2.0, float("nan"), -1])
def test_occurrence_ledger_rejects_malformed_severity_without_writing(
    ledger_factory: LedgerFactory,
    severity: object,
) -> None:
    ledger = ledger_factory()
    event = OccurrenceEvent(
        fingerprint="sqlite-wal-checkpoint-starvation",
        scope="host:m2/service:brainlayer-watch",
        session_id="session-a",
        occurred_at=FIXED_NOW,
        severity=severity,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="severity must be a non-negative integer"):
        ledger.record(event)
    assert ledger.weave_accumulation(through=FIXED_NOW.date()) == ()


def test_occurrence_ledger_accepts_equivalent_explicit_utc_timezone(
    ledger_factory: LedgerFactory,
) -> None:
    ledger = ledger_factory()

    receipt = ledger.record(
        OccurrenceEvent(
            fingerprint="sqlite-wal-checkpoint-starvation",
            scope="host:m2/service:brainlayer-watch",
            session_id="session-a",
            occurred_at=FIXED_NOW.astimezone(ZoneInfo("UTC")),
            severity=2,
        )
    )

    assert receipt.alert == "new"


def _oracle_candidates() -> list[CandidateCorrection]:
    gold = load_correction_gold()
    return oracle_candidate_producer(correction_inputs(gold))


def test_correction_gold_is_real_history_with_source_pointers_and_exact_etan_digest() -> None:
    gold = load_correction_gold()

    assert tuple(case.case_id for case in gold.cases) == (
        "workflow-exclusion-scope",
        "project-is-repo",
        "brain-worker-class",
        "store-update-not-duplicate",
        "merged-is-not-deployed",
        "stored-data-check-on-copy",
        "keep-archive-preconditions",
        "keep-law-and-context-roles",
        "keep-agent-authorship-rulings",
        "keep-worker-read-write-rules",
    )
    assert hashlib.sha256(CORRECTION_GOLD_PATH.read_bytes()).hexdigest() == (
        "1a8bf7077c0d6c6638e6472bde0f417e6d5f64514c25ac6d7ecd15977adb3806"
    )
    repo_root = Path(__file__).resolve().parents[2]
    for case in gold.cases:
        source_path, excerpt_digest = case.source_pointer.split("#excerpt-sha256:", 1)
        source_text = (repo_root / source_path).read_text(encoding="utf-8")
        assert hashlib.sha256(case.source_excerpt.encode()).hexdigest() == excerpt_digest
        assert case.source_excerpt in source_text
    producer_inputs = correction_inputs(gold)
    assert tuple(case.case_id for case in producer_inputs) == tuple(case.case_id for case in gold.cases)
    assert all(not hasattr(case, "expected_decision") for case in producer_inputs)
    assert all(not hasattr(case, "expected_digest") for case in producer_inputs)
    assert gold.etan_digest_requirement == ETAN_DIGEST_REQUIREMENT
    assert ETAN_DIGEST_REQUIREMENT == (
        "I should get a summarized version of what's superseding what, and be the human in the loop for some of it"
    )


def test_candidate_producer_meets_promotion_thresholds_and_false_positive_budget(
    candidate_producer: CandidateProducer,
) -> None:
    gold = load_correction_gold()
    candidates = candidate_producer(correction_inputs(gold))
    unapproved = score_correction_gold(candidates, gold=gold)
    report = score_correction_gold(candidates, gold=gold, human_approved=True)

    assert gold.thresholds.min_auto_confidence == 0.98
    assert gold.thresholds.min_recall == 1.0
    assert gold.thresholds.max_false_positives == 0
    assert gold.thresholds.min_digest_fidelity == 1.0
    assert gold.rollback_rules == (
        "false-positive budget exceeded",
        "recall below threshold",
        "digest fidelity below threshold",
    )
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.false_positives == 0
    assert unapproved.promote_to_auto is False
    assert unapproved.rollback_required is False
    assert "human approval" in unapproved.blockers
    candidates_by_id = {candidate.case_id: candidate for candidate in candidates}
    assert unapproved.proposed_digests == tuple(
        candidates_by_id[case.case_id].digest
        for case in gold.cases
        if candidates_by_id[case.case_id].digest is not None
    )
    assert report.promote_to_auto is True
    assert report.rollback_required is False
    assert report.etan_digest_requirement == ETAN_DIGEST_REQUIREMENT
    assert report.rollback_rules == gold.rollback_rules


def test_one_false_positive_exhausts_budget_and_requires_rollback() -> None:
    candidates = _oracle_candidates()
    negative_index = next(index for index, candidate in enumerate(candidates) if candidate.decision == "keep_both")
    negative = candidates[negative_index]
    candidates[negative_index] = CandidateCorrection(
        case_id=negative.case_id,
        decision="supersede",
        confidence=0.99,
        digest="Incorrect destructive suggestion",
    )

    report = score_correction_gold(candidates)

    assert report.false_positives == 1
    assert report.promote_to_auto is False
    assert report.rollback_required is True
    assert "false-positive budget" in report.blockers


def test_missing_or_changed_digest_blocks_promotion() -> None:
    candidates = _oracle_candidates()
    positive_index = next(index for index, candidate in enumerate(candidates) if candidate.decision == "supersede")
    positive = candidates[positive_index]
    candidates[positive_index] = CandidateCorrection(
        case_id=positive.case_id,
        decision=positive.decision,
        confidence=positive.confidence,
        digest="A vague digest without the approved before/after sample.",
    )

    report = score_correction_gold(candidates)

    assert report.digest_fidelity < 1.0
    assert report.promote_to_auto is False
    assert report.rollback_required is True
    assert "digest fidelity" in report.blockers


def test_below_threshold_supersession_stays_suggest_only() -> None:
    candidates = _oracle_candidates()
    positive_index = next(index for index, candidate in enumerate(candidates) if candidate.decision == "supersede")
    positive = candidates[positive_index]
    candidates[positive_index] = CandidateCorrection(
        case_id=positive.case_id,
        decision=positive.decision,
        confidence=0.97,
        digest=positive.digest,
    )

    report = score_correction_gold(candidates)

    assert report.recall < 1.0
    assert report.promote_to_auto is False
    assert report.rollback_required is True
    assert "recall threshold" in report.blockers


def test_gold_scorer_rejects_unknown_decisions_and_out_of_range_confidence() -> None:
    candidates = _oracle_candidates()
    first = candidates[0]

    candidates[0] = CandidateCorrection(first.case_id, "archive", 0.99, first.digest)
    with pytest.raises(ValueError, match="decision"):
        score_correction_gold(candidates)

    candidates[0] = CandidateCorrection(first.case_id, first.decision, 1.01, first.digest)
    with pytest.raises(ValueError, match="confidence"):
        score_correction_gold(candidates)

    with pytest.raises(ValueError, match="unique"):
        score_correction_gold([*candidates, candidates[-1]])


@pytest.mark.parametrize("confidence", [True, "0.99", float("nan"), float("inf")])
def test_gold_scorer_rejects_malformed_candidate_confidence(
    confidence: object,
) -> None:
    candidates = _oracle_candidates()
    first = candidates[0]
    candidates[0] = CandidateCorrection(
        first.case_id,
        first.decision,
        confidence,  # type: ignore[arg-type]
        first.digest,
    )

    with pytest.raises(ValueError, match="candidate confidence"):
        score_correction_gold(candidates)


def test_auto_confidence_threshold_is_inclusive() -> None:
    candidates = _oracle_candidates()
    first = candidates[0]
    candidates[0] = CandidateCorrection(
        first.case_id,
        first.decision,
        0.98,
        first.digest,
    )

    assert score_correction_gold(candidates, human_approved=True).promote_to_auto is True


@pytest.mark.parametrize("approval", ["false", 1, None])
def test_gold_scorer_rejects_non_boolean_human_approval(approval: object) -> None:
    with pytest.raises(ValueError, match="human_approved must be a boolean"):
        score_correction_gold(
            _oracle_candidates(),
            human_approved=approval,  # type: ignore[arg-type]
        )


def test_correction_gold_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    payload = json.loads(CORRECTION_GOLD_PATH.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    fixture_path = tmp_path / "future-gold.json"
    fixture_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported correction gold schema"):
        load_correction_gold(fixture_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_auto_confidence", float("nan")),
        ("min_recall", float("inf")),
        ("min_recall", True),
        ("min_recall", -0.01),
        ("min_digest_fidelity", 1.01),
        ("max_false_positives", -1),
        ("max_false_positives", True),
        ("max_false_positives", 0.5),
    ],
)
def test_correction_gold_loader_rejects_unsafe_thresholds(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = json.loads(CORRECTION_GOLD_PATH.read_text(encoding="utf-8"))
    payload["thresholds"][field] = value
    fixture_path = tmp_path / f"invalid-threshold-{field}.json"
    fixture_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="correction threshold"):
        load_correction_gold(fixture_path)


@pytest.mark.parametrize(
    ("case_index", "field", "value", "message"),
    [
        (0, "expected_decision", "archive", "expected_decision"),
        (0, "expected_digest", None, "supersede digest"),
        (0, "expected_digest", "missing structured fields", "entity, attribute, and source"),
        (6, "expected_digest", "must stay null", "keep_both digest"),
        (0, "corrected_claim", "", "non-empty case content"),
        (0, "source_pointer", "AGENTS.md:1", "stable source excerpt anchor"),
    ],
)
def test_correction_gold_loader_rejects_inert_case_contracts(
    tmp_path: Path,
    case_index: int,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = json.loads(CORRECTION_GOLD_PATH.read_text(encoding="utf-8"))
    payload["cases"][case_index][field] = value
    fixture_path = tmp_path / f"invalid-{case_index}-{field}.json"
    fixture_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_correction_gold(fixture_path)
