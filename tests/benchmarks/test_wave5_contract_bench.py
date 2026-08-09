import hashlib
import json
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tests.benchmarks.wave5_contract import (
    CORRECTION_GOLD_PATH,
    ETAN_DIGEST_REQUIREMENT,
    CandidateCorrection,
    Ledger,
    LedgerFactory,
    OccurrenceEvent,
    OccurrenceLedger,
    load_correction_gold,
    score_correction_gold,
)

FIXED_NOW = datetime(2026, 8, 9, 9, 0, tzinfo=UTC)


@pytest.fixture
def ledger_factory() -> LedgerFactory:
    """Override this fixture to run the same contract against production."""
    return OccurrenceLedger


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

    assert first.occurrence_id != other_scope.occurrence_id


def test_occurrence_identity_has_no_delimiter_collision(
    ledger_factory: LedgerFactory,
) -> None:
    ledger = ledger_factory()
    first = ledger.record(OccurrenceEvent("b\0c", "a", "session-a", FIXED_NOW, 1))
    second = ledger.record(OccurrenceEvent("c", "a\0b", "session-b", FIXED_NOW, 1))

    assert first.occurrence_id != second.occurrence_id


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
    assert same_session.event_count == 3
    assert same_session.session_ids == ("session-a", "session-b")


def test_only_new_or_escalating_occurrences_alert(ledger_factory: LedgerFactory) -> None:
    ledger = ledger_factory()
    base = OccurrenceEvent(
        fingerprint="sqlite-wal-checkpoint-starvation",
        scope="host:m2/service:brainlayer-watch",
        session_id="session-a",
        occurred_at=FIXED_NOW,
        severity=2,
    )

    assert ledger.record(base).alert == "new"
    assert ledger.record(OccurrenceEvent(**{**base.__dict__, "session_id": "session-b", "severity": 1})).alert is None
    assert (
        ledger.record(OccurrenceEvent(**{**base.__dict__, "session_id": "session-c", "severity": 3})).alert
        == "escalated"
    )
    assert ledger.record(OccurrenceEvent(**{**base.__dict__, "session_id": "session-d", "severity": 3})).alert is None


def test_weave_feed_accumulates_by_event_day_until_weave_is_invoked(
    ledger_factory: LedgerFactory,
) -> None:
    ledger = ledger_factory()
    occurrence_id = ""
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
        occurrence_id = occurrence_id or receipt.occurrence_id

    feed = ledger.weave_accumulation(through=date(2026, 8, 10))

    assert [bucket.day for bucket in feed] == [date(2026, 8, 9), date(2026, 8, 10)]
    assert [bucket.event_count for bucket in feed] == [1, 1]
    assert {bucket.occurrence_id for bucket in feed} == {occurrence_id}
    assert [bucket.session_ids for bucket in feed] == [("session-a",), ("session-b",)]
    assert ledger.weave_accumulation(through=date(2026, 8, 10)) == ()
    future_feed = ledger.weave_accumulation(through=date(2026, 8, 11))
    assert [(bucket.day, bucket.event_count) for bucket in future_feed] == [(date(2026, 8, 11), 1)]


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
    return [
        CandidateCorrection(
            case_id=case.case_id,
            decision=case.expected_decision,
            confidence=0.99,
            digest=case.expected_digest if case.expected_decision == "supersede" else None,
        )
        for case in gold.cases
    ]


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
    assert gold.etan_digest_requirement == ETAN_DIGEST_REQUIREMENT
    assert ETAN_DIGEST_REQUIREMENT == (
        "I should get a summarized version of what's superseding what, and be the human in the loop for some of it"
    )


def test_gold_oracle_meets_promotion_thresholds_and_false_positive_budget() -> None:
    gold = load_correction_gold()
    candidates = _oracle_candidates()
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
    assert unapproved.proposed_digests == tuple(
        candidate.digest for candidate in candidates if candidate.digest is not None
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


def test_correction_gold_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    payload = json.loads(CORRECTION_GOLD_PATH.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    fixture_path = tmp_path / "future-gold.json"
    fixture_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported correction gold schema"):
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
