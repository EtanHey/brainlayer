"""Tests for provenance-aware memory resolution (T4, gen-14).

RED→GREEN eval for the provenance class-gate resolver. Encodes the flip-cases
validated by hand today (nanoClaw 4/4, controlLayer per-attribute) as executable
fixtures so the resolver's logic is regression-locked.

The whole point of this layer: rank contradictory facts by SOURCE AUTHORITY, not
recency. Recency is only a tiebreaker WITHIN the top provenance class; a newer
agent-inference must never beat an older raw-Etan-direct fact (the exact failure
mode Graphiti/recency-only resolvers get wrong).

Provenance entrenchment lattice (AGM epistemic entrenchment):
    RAW-ETAN-DIRECT (3) > ETAN-ENDORSEMENT (2) > AGENT-PARAPHRASE (1) > AGENT-INFERENCE (0)

Pure-function unit tests only — no DB, no live BrainLayer, no writes.
"""

from __future__ import annotations

from brainlayer.provenance import (
    PROVENANCE_RANK,
    Claim,
    derive_provenance_class,
    normalize_entity,
    resolve,
    resolve_entity,
)

OPERATIONAL_EVIDENCE = "OPERATIONAL-EVIDENCE"

# ---------------------------------------------------------------------------
# 1. CLASS DERIVATION (the mechanical gate that rides enrichment)
#    content_type + sender + quote-vs-assertion -> provenance class.
# ---------------------------------------------------------------------------


def test_user_message_is_raw_etan_direct():
    assert derive_provenance_class(content_type="user_message", sender="user", text="x" * 200) == "RAW-ETAN-DIRECT"


def test_short_user_echoing_prior_assistant_is_endorsement():
    # A short user turn that restates the immediately-preceding assistant framing
    # is an ENDORSEMENT of that framing, NOT independent raw-Etan-direct.
    cls = derive_provenance_class(
        content_type="user_message",
        sender="user",
        text="yeah exactly, that",
        prev_assistant_text="controlLayer is an umbrella for the file schema + policies, not a new running thing",
    )
    assert cls == "ETAN-ENDORSEMENT"


def test_short_user_correction_without_echo_is_raw_direct():
    # Short user turns are not automatically endorsements: a correction that
    # does not echo the assistant is Etan stating a direct replacement fact.
    cls = derive_provenance_class(
        content_type="user_message",
        sender="user",
        text="no, it's X",
        prev_assistant_text="The entity's active mode is Y.",
    )
    assert cls == "RAW-ETAN-DIRECT"


def test_assistant_quoting_user_is_paraphrase():
    cls = derive_provenance_class(
        content_type="assistant_text",
        sender="assistant",
        text='Etan said: "use Codex heavily as a puppet" — storing that.',
    )
    assert cls == "AGENT-PARAPHRASE"


def test_assistant_asserting_new_conclusion_is_inference():
    cls = derive_provenance_class(
        content_type="assistant_text",
        sender="assistant",
        text="Arbitration split: VoiceBar daemon enforces, controlLayer decides.",
    )
    assert cls == "AGENT-INFERENCE"


def test_tool_result_envelope_is_not_etan_direct():
    # role:user envelopes that are actually tool_result / file_read content must
    # never be classified as Etan speaking (expA false-positive trap).
    cls = derive_provenance_class(content_type="file_read", sender="user", text="<file contents>")
    assert cls != "RAW-ETAN-DIRECT"


# ---------------------------------------------------------------------------
# 2. ALIAS NORMALIZATION must run BEFORE classification
#    "nano claw" (voice-spaced) and "nanoClaw" are the same entity — the single
#    most load-bearing turn was grep-missed because of this.
# ---------------------------------------------------------------------------


def test_alias_normalization_collapses_spaced_and_camelcase():
    assert normalize_entity("nano claw") == normalize_entity("nanoClaw")
    assert normalize_entity("Nano Claw") == normalize_entity("nanoclaw")
    assert normalize_entity("control layer") == normalize_entity("controlLayer")


def test_alias_normalization_collapses_hyphen_space_and_missing_separator():
    assert normalize_entity("co-pilot") == normalize_entity("copilot")
    assert normalize_entity("co pilot") == normalize_entity("copilot")


def test_alias_normalization_collapses_digit_version_separators():
    # Version punctuation is treated as a separator/noise boundary, not as a
    # semantic token. This intentionally collapses 5.5 -> 55 for alias lookup.
    assert normalize_entity("gpt-5.5") == "gpt55"
    assert normalize_entity("gpt 5.5") == normalize_entity("GPT5.5")


def test_alias_normalization_strips_trailing_punctuation_and_repeated_whitespace():
    assert normalize_entity("  Co   Pilot!!! ") == normalize_entity("co-pilot")


def test_alias_normalization_handles_hebrew_unicode_idempotently():
    normalized = normalize_entity("  איתן  היימן!!! ")
    assert normalized
    assert normalize_entity(normalized) == normalized


def test_alias_normalization_degenerate_input_does_not_throw():
    assert normalize_entity("") == ""
    assert normalize_entity("!!!...---") == ""


# ---------------------------------------------------------------------------
# 3. THE CORE RESOLVER — class-gate, not naive latest-wins
# ---------------------------------------------------------------------------


def _c(value, cls, ts, *, anchored=True, cid=None, attribute="attr"):
    return Claim(
        id=cid or f"{cls}:{value}:{ts}",
        entity="e",
        attribute=attribute,
        value=value,
        provenance_class=cls,
        timestamp=ts,
        user_anchored=anchored,
        text=f"{cls} says {value}",
    )


def test_recency_never_crosses_class_boundary():
    """The Graphiti failure mode: a NEW agent-inference must NOT beat an OLD
    raw-Etan-direct fact. Source authority outranks recency."""
    old_direct = _c("X", "RAW-ETAN-DIRECT", "2026-01-01T00:00:00Z")
    new_inference = _c("Y", "AGENT-INFERENCE", "2026-06-08T00:00:00Z", anchored=False)
    out = resolve([old_direct, new_inference])
    assert out.authoritative.value == "X"
    assert out.authoritative.provenance_class == "RAW-ETAN-DIRECT"
    # the newer inference is NOT authoritative and is routed to confirm, not crowned
    assert new_inference in out.flagged_pending_user_confirm


def test_recency_tiebreaks_within_top_class_only():
    """Two raw-Etan-direct claims that disagree: the MORE RECENT one wins —
    but only because they are the SAME (top) class."""
    older = _c("A", "RAW-ETAN-DIRECT", "2026-06-08T14:31:46Z")
    newer = _c("B", "RAW-ETAN-DIRECT", "2026-06-08T16:37:12Z")
    out = resolve([older, newer])
    assert out.authoritative.value == "B"
    assert older in out.superseded  # older same-class -> superseded (revised by Etan himself)


def test_identical_timestamps_tiebreak_by_lowest_claim_id():
    a = _c("A", "RAW-ETAN-DIRECT", "2026-06-08T14:31:46Z", cid="claim-a")
    b = _c("B", "RAW-ETAN-DIRECT", "2026-06-08T14:31:46Z", cid="claim-b")
    out = resolve([b, a])
    assert out.authoritative.id == "claim-a"


def test_malformed_and_empty_timestamps_sort_last_without_crashing():
    malformed = _c("BAD_TS", "RAW-ETAN-DIRECT", "not-a-date", cid="claim-bad")
    empty = _c("EMPTY_TS", "RAW-ETAN-DIRECT", "", cid="claim-empty")
    valid = _c("VALID_TS", "RAW-ETAN-DIRECT", "2026-06-08T14:31:46Z", cid="claim-valid")
    out = resolve([malformed, empty, valid])
    assert out.authoritative.id == "claim-valid"


def test_mixed_iso_offsets_compare_by_absolute_time_not_raw_text():
    older_same_day_local = _c("OLDER", "RAW-ETAN-DIRECT", "2026-06-08T11:00:00+02:00", cid="older-local")
    newer_utc = _c("NEWER", "RAW-ETAN-DIRECT", "2026-06-08T10:00:00Z", cid="newer-utc")
    out = resolve([older_same_day_local, newer_utc])
    assert out.authoritative.id == "newer-utc"


def test_nanoclaw_flip_case():
    """The validated 4/4 case. Three raw-Etan-direct turns (Etan revises himself
    from '≈ Hermes' to 'DISTINCT') plus an agent-inference that drifted to
    'grows toward Hermes'. Authoritative = the latest raw-Etan-direct (DISTINCT);
    the agent-inference is flagged, never authoritative."""
    d2 = _c("EQUALS_HERMES", "RAW-ETAN-DIRECT", "2026-06-08T14:31:46Z", cid="d2")
    d3 = _c("DISTINCT", "RAW-ETAN-DIRECT", "2026-06-08T16:37:12Z", cid="d3")
    d4 = _c("DISTINCT", "RAW-ETAN-DIRECT", "2026-06-08T18:50:44Z", cid="d4")
    i1 = _c("GROWS_TOWARD_HERMES", "AGENT-INFERENCE", "2026-06-08T15:50:38Z", anchored=False, cid="i1")
    out = resolve([d2, d3, d4, i1])
    assert out.authoritative.value == "DISTINCT"
    assert out.authoritative.id in {"d3", "d4"}
    # Etan's own earlier '≈ Hermes' snapshot is superseded (older same-class)
    assert d2 in out.superseded
    # the agent's never-said-directly inference is flagged for confirmation, NOT superseded as a peer fact
    assert i1 in out.flagged_pending_user_confirm
    assert i1 not in out.superseded


def test_unanchored_inference_only_routes_to_pending_confirm():
    """An attribute asserted ONLY by agent-inference (no user anchor anywhere)
    must not be crowned as fact — disposition is PENDING-USER-CONFIRM."""
    i1 = _c("CONTROLLAYER_DECIDES", "AGENT-INFERENCE", "2026-06-05T21:47:27Z", anchored=False)
    out = resolve([i1])
    assert out.disposition == "PENDING-USER-CONFIRM"
    assert out.authoritative is None
    assert i1 in out.flagged_pending_user_confirm


def test_paraphrase_loses_to_direct_but_is_not_flagged():
    """A faithful agent-paraphrase that AGREES with the direct fact just yields;
    a paraphrase is not an inference, so it is not routed to confirm."""
    direct = _c("DISTINCT", "RAW-ETAN-DIRECT", "2026-06-08T16:37:12Z")
    para = _c("DISTINCT", "AGENT-PARAPHRASE", "2026-06-08T16:40:00Z")
    out = resolve([direct, para])
    assert out.authoritative.provenance_class == "RAW-ETAN-DIRECT"
    assert not out.flagged_pending_user_confirm  # agreeing paraphrase, nothing to confirm


# ---------------------------------------------------------------------------
# 4. PER-(ENTITY, ATTRIBUTE) AUTHORITY — single-fact resolution is WRONG
#    controlLayer: DEFINITION converges; ARBITRATION is agent-only.
#    A foundational fact on one attribute must not be buried by chatter on another.
# ---------------------------------------------------------------------------


def test_per_attribute_resolution_keeps_attributes_independent():
    claims = [
        # DEFINITION attribute — Etan-direct + agreeing paraphrase
        _c("FILE_SCHEMA_AND_POLICIES", "RAW-ETAN-DIRECT", "2026-06-05T22:32:03Z", attribute="DEFINITION"),
        _c("FILE_SCHEMA_AND_POLICIES", "AGENT-PARAPHRASE", "2026-06-05T22:25:08Z", attribute="DEFINITION"),
        # ARBITRATION attribute — ONLY an agent-inference, no user anchor
        _c("CONTROLLAYER_DECIDES", "AGENT-INFERENCE", "2026-06-05T21:47:27Z", anchored=False, attribute="ARBITRATION"),
    ]
    by_attr = resolve_entity(claims)
    assert by_attr["DEFINITION"].authoritative.value == "FILE_SCHEMA_AND_POLICIES"
    assert by_attr["DEFINITION"].authoritative.provenance_class == "RAW-ETAN-DIRECT"
    # ARBITRATION resolved INDEPENDENTLY — not crowned, routed to confirm
    assert by_attr["ARBITRATION"].disposition == "PENDING-USER-CONFIRM"
    assert by_attr["ARBITRATION"].authoritative is None


def test_three_attributes_resolve_to_different_top_classes_in_one_pass():
    claims = [
        _c("DIRECT_DEF", "RAW-ETAN-DIRECT", "2026-01-01T00:00:00Z", attribute="DEFINITION"),
        _c("ENDORSED_STATUS", "ETAN-ENDORSEMENT", "2026-02-01T00:00:00Z", attribute="STATUS"),
        _c("AGENT_ONLY", "AGENT-INFERENCE", "2026-03-01T00:00:00Z", anchored=False, attribute="ARBITRATION"),
    ]
    by_attr = resolve_entity(claims)
    assert by_attr["DEFINITION"].authoritative.provenance_class == "RAW-ETAN-DIRECT"
    assert by_attr["STATUS"].authoritative.provenance_class == "ETAN-ENDORSEMENT"
    assert by_attr["ARBITRATION"].disposition == "PENDING-USER-CONFIRM"


def test_endorsement_wins_without_direct_and_agreeing_paraphrase_is_not_superseded():
    endorsement = _c("CURRENT", "ETAN-ENDORSEMENT", "2026-06-08T10:00:00Z", cid="endorsement")
    paraphrase = _c("CURRENT", "AGENT-PARAPHRASE", "2026-06-08T11:00:00Z", cid="paraphrase")
    out = resolve([endorsement, paraphrase])
    assert out.authoritative.id == "endorsement"
    assert not out.superseded


def test_endorsement_wins_without_direct_and_disagreeing_paraphrase_is_superseded():
    endorsement = _c("CURRENT", "ETAN-ENDORSEMENT", "2026-06-08T10:00:00Z", cid="endorsement")
    paraphrase = _c("STALE", "AGENT-PARAPHRASE", "2026-06-08T11:00:00Z", cid="paraphrase")
    out = resolve([endorsement, paraphrase])
    assert out.authoritative.id == "endorsement"
    assert paraphrase in out.superseded


def test_controllayer_definition_converges_but_arbitration_is_inference_only():
    """Flip-case: controlLayer's DEFINITION can converge cleanly while a separate
    ARBITRATION claim remains inference-only and must not borrow authority from
    the definition attribute."""
    claims = [
        _c("FILE_SCHEMA_AND_POLICIES", "RAW-ETAN-DIRECT", "2026-06-05T22:32:03Z", attribute="DEFINITION"),
        _c("FILE_SCHEMA_AND_POLICIES", "ETAN-ENDORSEMENT", "2026-06-05T22:33:00Z", attribute="DEFINITION"),
        _c("FILE_SCHEMA_AND_POLICIES", "AGENT-PARAPHRASE", "2026-06-05T22:34:00Z", attribute="DEFINITION"),
        _c("CONTROLLAYER_DECIDES", "AGENT-INFERENCE", "2026-06-08T09:00:00Z", anchored=False, attribute="ARBITRATION"),
        _c(
            "VOICEBAR_DAEMON_ENFORCES",
            "AGENT-INFERENCE",
            "2026-06-08T09:05:00Z",
            anchored=False,
            attribute="ARBITRATION",
        ),
    ]
    by_attr = resolve_entity(claims)
    assert by_attr["DEFINITION"].authoritative.value == "FILE_SCHEMA_AND_POLICIES"
    assert by_attr["DEFINITION"].authoritative.provenance_class == "RAW-ETAN-DIRECT"
    assert by_attr["ARBITRATION"].disposition == "PENDING-USER-CONFIRM"
    assert by_attr["ARBITRATION"].authoritative is None
    assert {c.value for c in by_attr["ARBITRATION"].flagged_pending_user_confirm} == {
        "CONTROLLAYER_DECIDES",
        "VOICEBAR_DAEMON_ENFORCES",
    }


def test_codex_budget_shape_newer_agent_budget_inference_cannot_beat_old_direct_budget():
    """Flip-case: a recent Codex-budget inference must not cross the class
    boundary and replace an older direct budget statement."""
    direct_budget = _c(
        "TOKEN_BUDGET_IS_USER_CONTROLLED",
        "RAW-ETAN-DIRECT",
        "2026-04-12T10:00:00Z",
        cid="budget-direct",
        attribute="CODEX_BUDGET_POLICY",
    )
    recent_agent_budget = _c(
        "CODEX_CAN_SPEND_FREELY_UNTIL_CONTEXT_COMPACTION",
        "AGENT-INFERENCE",
        "2026-06-08T23:20:00Z",
        anchored=False,
        cid="budget-infer",
        attribute="CODEX_BUDGET_POLICY",
    )
    by_attr = resolve_entity([direct_budget, recent_agent_budget])
    out = by_attr["CODEX_BUDGET_POLICY"]
    assert out.authoritative.id == "budget-direct"
    assert out.authoritative.value == "TOKEN_BUDGET_IS_USER_CONTROLLED"
    assert recent_agent_budget in out.flagged_pending_user_confirm


def test_foundational_old_fact_not_buried_by_recent_other_attribute():
    """A months-old foundational RAW-ETAN-DIRECT on attribute A stays authoritative
    for A even when there is very recent chatter on a DIFFERENT attribute B."""
    foundational = _c("CORE_DEF", "RAW-ETAN-DIRECT", "2026-01-15T00:00:00Z", attribute="DEFINITION")
    recent_other = _c("SOME_STATUS", "AGENT-PARAPHRASE", "2026-06-08T00:00:00Z", attribute="STATUS")
    by_attr = resolve_entity([foundational, recent_other])
    assert by_attr["DEFINITION"].authoritative.value == "CORE_DEF"
    assert by_attr["DEFINITION"].authoritative.timestamp == "2026-01-15T00:00:00Z"


def test_operational_evidence_default_off_loses_to_agent_paraphrase():
    operational = _c(
        "CHUNK_ORIGIN_COUNTS_FROM_DB",
        OPERATIONAL_EVIDENCE,
        "2026-06-08T11:00:00Z",
        cid="operational",
        attribute="CHUNK_ORIGIN_COUNTS",
    )
    stale_paraphrase = _c(
        "OLDER_SUMMARY",
        "AGENT-PARAPHRASE",
        "2026-06-01T11:00:00Z",
        cid="paraphrase",
        attribute="CHUNK_ORIGIN_COUNTS",
    )
    out = resolve([operational, stale_paraphrase])
    assert out.authoritative.id == "paraphrase"


def test_operational_evidence_opt_in_wins_for_system_state_attribute():
    operational = _c(
        "CHUNK_ORIGIN_COUNTS_FROM_DB",
        OPERATIONAL_EVIDENCE,
        "2026-06-08T11:00:00Z",
        cid="operational",
        attribute="CHUNK_ORIGIN_COUNTS",
    )
    stale_paraphrase = _c(
        "OLDER_SUMMARY",
        "AGENT-PARAPHRASE",
        "2026-06-01T11:00:00Z",
        cid="paraphrase",
        attribute="CHUNK_ORIGIN_COUNTS",
    )
    out = resolve([operational, stale_paraphrase], enable_operational_evidence=True)
    assert out.authoritative.id == "operational"


def test_operational_evidence_opt_in_treats_primary_backend_as_system_state():
    operational = _c(
        "GEMINI_FROM_AGGREGATE_ORIGIN_COUNTS",
        OPERATIONAL_EVIDENCE,
        "2026-06-08T11:00:00Z",
        cid="operational",
        attribute="PRIMARY_BACKEND",
    )
    stale_paraphrase = _c(
        "GROQ_FROM_STALE_STATUS_SUMMARY",
        "AGENT-PARAPHRASE",
        "2026-06-01T11:00:00Z",
        cid="paraphrase",
        attribute="PRIMARY_BACKEND",
    )

    out = resolve([operational, stale_paraphrase], enable_operational_evidence=True)

    assert out.authoritative.id == "operational"
    assert [claim.id for claim in out.superseded] == ["paraphrase"]


def test_operational_evidence_opt_in_does_not_apply_to_personal_attributes():
    operational = _c(
        "SYSTEM_OBSERVED_PREFERENCE",
        OPERATIONAL_EVIDENCE,
        "2026-06-08T11:00:00Z",
        cid="operational",
        attribute="PREFERENCE",
    )
    paraphrase = _c(
        "ETAN_SAID_PREFERENCE",
        "AGENT-PARAPHRASE",
        "2026-06-01T11:00:00Z",
        cid="paraphrase",
        attribute="PREFERENCE",
    )
    out = resolve([operational, paraphrase], enable_operational_evidence=True)
    assert out.authoritative.id == "paraphrase"


# ---------------------------------------------------------------------------
# 5. RANK TABLE sanity
# ---------------------------------------------------------------------------


def test_rank_ordering():
    assert (
        PROVENANCE_RANK["RAW-ETAN-DIRECT"]
        > PROVENANCE_RANK["ETAN-ENDORSEMENT"]
        > PROVENANCE_RANK["AGENT-PARAPHRASE"]
        > PROVENANCE_RANK["AGENT-INFERENCE"]
    )
