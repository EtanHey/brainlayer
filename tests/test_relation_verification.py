import json
from datetime import datetime

import pytest

from brainlayer.pipeline.relation_verification import EvidenceWindow
from brainlayer.pipeline.relation_verification import verify_relation as review


def verify_relation(*args, on_response=None):
    return review(*args, on_response=on_response or (lambda event: None))


@pytest.fixture
def source():
    return {
        "chunk_id": "source-id",
        "content": "Atlas depends on SQLite for its runtime.",
        "entities": [
            {"id": "a", "name": "Atlas", "type": "project"},
            {"id": "s", "name": "SQLite", "type": "technology"},
        ],
        "origin": "session-a",
        "created_at": "2026-01-01T00:00:00+00:00",
        "source_class": "cli-agent",
    }


@pytest.fixture
def relation(source):
    return dict(source_id="a", target_id="s", type="depends_on", quote=source["content"], temporal_status="current")


def reference(**changes):
    return EvidenceWindow(
        **dict(
            dict(
                chunk_id="reference-id",
                content="SQLite is required by Atlas at runtime.",
                origin="session-b",
                created_at="2026-01-02T00:00:00+00:00",
                source_class="cli-agent",
            ),
            **changes,
        )
    )


def caller(source, refs, primary="supports", verdict="supports", mutate=None):
    response = {
        "primary": {"verdict": primary, "quote": source["content"]},
        "references": [{"verdict": verdict, "quote": r.content} for r in refs],
    }
    if mutate:
        mutate(response)
    return lambda messages: json.dumps(response)


@pytest.mark.parametrize("source_class", ["cli-agent", "subagent", "fleet-coordination"])
def test_corroborated_is_source_assertion_not_current_graph_permission(source, relation, source_class):
    source["source_class"] = source_class
    refs = [reference(source_class=source_class)]
    result = verify_relation(source, relation, refs, caller(source, refs))
    assert result["status"] == "CORROBORATED_SOURCE_ASSERTION"
    assert result["current_truth"] == "UNVERIFIED"
    assert result["canonical_write_authorized"] is False
    assert result["independent_supports"] == ["reference-id"]


@pytest.mark.parametrize("changes", [{"origin": "session-a"}, {"origin": None}, {"created_at": None}])
def test_repeated_or_untraceable_evidence_cannot_corroborate(source, relation, changes):
    refs = [reference(**changes)]
    assert verify_relation(source, relation, refs, caller(source, refs))["status"] == "UNKNOWN"


def test_no_hits_is_unknown(source, relation):
    assert verify_relation(source, relation, [], caller(source, []))["status"] == "UNKNOWN"


def test_unclear_reference_prevents_corroboration_despite_other_support(source, relation):
    refs = [reference(), reference(chunk_id="unclear-id", origin="session-c")]
    response = caller(source, refs, mutate=lambda r: r["references"][1].update(verdict="unclear", quote=""))
    assert verify_relation(source, relation, refs, response)["status"] == "UNKNOWN"


def test_recording_failure_prevents_interpreting_model_output(source, relation):
    def failed_record(event):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        verify_relation(source, relation, [], lambda messages: "not JSON", on_response=failed_record)


def test_identical_quote_from_other_session_is_not_independent_support(source, relation):
    refs = [reference(content="Forwarded assertion: " + source["content"])]
    response = caller(source, refs, mutate=lambda r: r["references"][0].update(quote=source["content"]))
    assert verify_relation(source, relation, refs, response)["status"] == "UNKNOWN"


def test_another_copied_span_from_primary_is_not_independent_support(source, relation):
    copied = "Atlas requires SQLite for storage."
    source["content"] += " " + copied
    refs = [reference(content="Forwarded excerpt: " + copied)]

    def quotes(response):
        response["primary"]["quote"] = relation["quote"]
        response["references"][0]["quote"] = copied

    assert verify_relation(source, relation, refs, caller(source, refs, mutate=quotes))["status"] == "UNKNOWN"


def test_policy_is_preserved_without_retyping_a_service_or_making_an_id(source, relation):
    source["content"] = "Atlas agents MUST search SQLite before answering."
    relation["quote"] = source["content"]
    result = verify_relation(source, relation, [], caller(source, [], primary="mandatory_policy"))
    assert result["status"] == "GOVERNED_BY_UNBOUND"
    assert result["policy_quote"] == source["content"]
    assert result["proposed_relation"]["type"] == "depends_on"
    assert "policy_id" not in result


@pytest.mark.parametrize("verdict", ["negated", "planned", "question", "co_mention", "wrong_relation"])
def test_source_rejection_cannot_be_overridden_by_external_support(source, relation, verdict):
    refs = [reference()]
    result = verify_relation(source, relation, refs, caller(source, refs, primary=verdict))
    assert result["status"] == "REJECTED"


@pytest.mark.parametrize(
    "verdict,date,status",
    [
        ("contradicts", "2026-01-02T00:00:00+00:00", "REJECTED"),
        ("contradicts", "2025-12-31T00:00:00+00:00", "UNKNOWN"),
        ("ended", "2026-01-02T00:00:00+00:00", "HISTORICAL_ONLY"),
        ("ended", None, "UNKNOWN"),
        ("supports", "2026-01-02T00:00:00", "UNKNOWN"),
        ("contradicts", "2026-01-02T00:00:00", "UNKNOWN"),
    ],
)
def test_chronology_distinguishes_correction_expiry_and_prior_denial(source, relation, verdict, date, status):
    refs = [reference(created_at=date)]
    assert verify_relation(source, relation, refs, caller(source, refs, verdict=verdict))["status"] == status


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["references"].clear(),
        lambda r: r["references"][0].update(quote="invented quote"),
        lambda r: r["primary"].update(quote="runtime"),
        lambda r: r.update(extra="not allowed"),
        lambda r: r["primary"].update(verdict="accept"),
    ],
)
def test_raw_evidence_is_retained_before_bad_verdict_rejects(source, relation, mutate):
    refs = [reference()]
    trace = []
    with pytest.raises(ValueError):
        verify_relation(source, relation, refs, caller(source, refs, mutate=mutate), on_response=trace.append)
    assert len(trace) == 1 and trace[0]["raw"]
    evidence = trace[0]["evidence"]
    assert evidence["primary"]["chunk_id"] == source["chunk_id"]
    assert evidence["primary"]["content"] == source["content"]
    assert evidence["references"] == [vars(r) for r in refs]


def test_policy_outcome_cannot_replace_the_original_proposal_quote(source, relation):
    policy_quote = "Atlas agents MUST search SQLite before answering."
    source["content"] += " " + policy_quote
    response = caller(source, [], primary="mandatory_policy", mutate=lambda r: r["primary"].update(quote=policy_quote))
    with pytest.raises(ValueError, match="original quote"):
        verify_relation(source, relation, [], response)


def test_hidden_reference_and_invalid_structural_quote_never_reach_model(source, relation):
    def forbidden(messages):
        pytest.fail("ineligible source reached model")

    with pytest.raises(ValueError):
        verify_relation(source, relation, [reference(source_class="desktop")], forbidden)
    relation["quote"] = "not in source"
    with pytest.raises(ValueError):
        verify_relation(source, relation, [], forbidden)


@pytest.mark.parametrize(
    "source_class", ["desktop", "brain-worker", "session-miner", "weave", None, "", "unknown", "codex-session"]
)
@pytest.mark.parametrize("position", ["primary", "reference"])
def test_excluded_evidence_classes_never_reach_model(source, relation, source_class, position):
    def forbidden(messages):
        pytest.fail("excluded evidence reached model")

    refs = []
    if position == "primary":
        source["source_class"] = source_class
    else:
        refs = [reference(source_class=source_class)]
    with pytest.raises(ValueError, match="eligible complete bounded evidence"):
        verify_relation(source, relation, refs, forbidden)


@pytest.mark.parametrize("date", [datetime(2026, 1, 1), b"2026-01-01T00:00:00Z", 1767225600])
@pytest.mark.parametrize("position", ["primary", "reference"])
def test_invalid_date_shape_is_rejected_before_transport(source, relation, date, position):
    def forbidden(messages):
        pytest.fail("invalid timestamp reached model")

    refs = []
    if position == "primary":
        source["created_at"] = date
    else:
        refs = [reference(created_at=date)]
    with pytest.raises(ValueError, match="eligible complete bounded evidence"):
        verify_relation(source, relation, refs, forbidden)


@pytest.mark.parametrize("stage", ["caller", "on_response"])
@pytest.mark.parametrize("target", ["relation", "references"])
def test_callback_mutation_cannot_change_reviewed_inputs(source, relation, stage, target):
    refs = [reference()]
    response = caller(source, refs)
    expected = verify_relation(source, relation, refs, response)

    def mutate():
        if target == "relation":
            relation["source_id"] = "unreviewed-id"
        else:
            refs[0] = reference(chunk_id="unreviewed-id", origin="unreviewed-origin")

    def transport(messages):
        if stage == "caller":
            mutate()
        return response(messages)

    def record(event):
        if stage == "on_response":
            mutate()

    result = verify_relation(source, relation, refs, transport, on_response=record)
    assert result == expected


def test_model_receives_names_and_source_data_separate_from_instructions(source, relation):
    refs = [reference()]

    def inspect(messages):
        assert [m["role"] for m in messages] == ["system", "user"]
        assert "software/runtime" in messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        assert payload["proposal"]["source_name"] == "Atlas"
        assert "source-id" not in messages[1]["content"]
        assert "reference-id" not in messages[1]["content"]
        return caller(source, refs)(messages)

    assert verify_relation(source, relation, refs, inspect)["status"] == "CORROBORATED_SOURCE_ASSERTION"
