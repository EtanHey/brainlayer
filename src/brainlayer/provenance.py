"""Provenance-aware memory resolution (T4, gen-14).

Rank contradictory facts by SOURCE AUTHORITY, not recency. This is the third
layer of the 3-stage memory pipeline (SCOPE filter -> RRF fusion -> PROVENANCE
class-gate) — it brackets RRF on the output side, deciding which of several
conflicting chunks about the same (entity, attribute) is authoritative.

Entrenchment lattice (AGM epistemic entrenchment; Atlas/XTrace anti-recency
stance — user-stated beliefs outrank system inferences):

    RAW-ETAN-DIRECT (3) > ETAN-ENDORSEMENT (2) > AGENT-PARAPHRASE (1) > AGENT-INFERENCE (0)

Hard rules (enforced in code, not prose — every recency-only system gets these
wrong):
  * Resolution is per (entity, ATTRIBUTE). A single-fact resolver is actively
    wrong: it would crown an agent's inference as part of "the entity fact"
    just because it shares the entity string.
  * Recency is a TIEBREAKER ONLY, and only WITHIN the top class. A newer
    AGENT-INFERENCE never beats an older RAW-ETAN-DIRECT.
  * A foundational fact months old stays authoritative for ITS attribute;
    recent narrow chatter on a different attribute must never bury it (falls out
    of per-attribute grouping).
  * No auto-delete. An unanchored agent-inference that contradicts (or stands
    alone for) an attribute routes to PENDING-USER-CONFIRM, never silent
    retraction (TMS STALE-not-retract; global NEVER-AUTO-DELETE-PERSONAL-DATA).
    The loser of a genuine class conflict routes to brain_supersede (reversible)
    by the caller — this module only decides, it does not write.

This module is pure logic: no DB, no LLM, no I/O. The mechanical class gate
(derive_provenance_class) is what rides the enrichment pass; the LLM half
(PARAPHRASE-vs-INFERENCE nuance, contradiction detection) happens upstream in
enrichment and is passed in as already-classified Claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

# --- entrenchment lattice ---------------------------------------------------

RAW_ETAN_DIRECT = "RAW-ETAN-DIRECT"
ETAN_ENDORSEMENT = "ETAN-ENDORSEMENT"
AGENT_PARAPHRASE = "AGENT-PARAPHRASE"
AGENT_INFERENCE = "AGENT-INFERENCE"
OPERATIONAL_EVIDENCE = "OPERATIONAL-EVIDENCE"

PROVENANCE_RANK: dict[str, int] = {
    RAW_ETAN_DIRECT: 3,
    ETAN_ENDORSEMENT: 2,
    AGENT_PARAPHRASE: 1,
    AGENT_INFERENCE: 0,
}

# Pending Etan's bless of the 5th class — default off. When explicitly enabled
# for system-state attributes, operational evidence outranks stale paraphrases
# while remaining below direct user statements and user endorsements.
_OPERATIONAL_EVIDENCE_RANK = PROVENANCE_RANK[AGENT_PARAPHRASE] + 0.5
_DEFAULT_SYSTEM_STATE_ATTRIBUTES = {
    "CHUNK_ORIGIN_COUNTS",
    "AGGREGATE_CHUNK_ORIGIN_COUNTS",
    "SYSTEM_STATE",
    "SERVICE_STATUS",
    "MCP_STATUS",
    "DEPLOYMENT_STATUS",
    "PRIMARY_BACKEND",
}

# content_type values the indexer assigns to genuine Etan turns vs everything
# else. user_message is already separated from tool_result / file_read by the
# indexer, so the expA false-positive trap (role:user envelopes that are really
# tool output) is filtered for us at this layer.
_ETAN_CONTENT_TYPES = {"user_message"}
_ASSISTANT_CONTENT_TYPES = {"assistant_text"}

# a short user turn under this length, when it echoes the prior assistant turn,
# is read as an ENDORSEMENT of the agent's framing rather than independent
# raw-Etan-direct (length is a heuristic, not a rule).
_ENDORSEMENT_MAX_LEN = 80

# markers that an assistant turn is QUOTING the user (paraphrase) rather than
# asserting its own conclusion (inference).
_QUOTE_MARKERS = ('"', "etan said", "etan:", "you said", "> ", "quote", "storing that")


def normalize_entity(name: str) -> str:
    """Collapse entity-name spelling drift to a single key BEFORE classification.

    Voice dictation produces 'nano claw' / 'Nano Claw' / 'nanoClaw' for the same
    entity; the single most load-bearing turn in the nanoClaw case was grep-missed
    purely because of this. Normalization MUST run before classify, not after.

    Rule: case-fold, split camelCase, then keep Unicode alphanumerics only.
    Separators and punctuation are alias noise, so version punctuation collapses:
    `gpt-5.5`, `gpt 5.5`, and `GPT5.5` all normalize to `gpt55`.
    """
    # Split camelCase, then strip punctuation/separators while preserving
    # non-ASCII letters such as Hebrew.
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name or ""))
    return "".join(ch for ch in spaced.casefold() if ch.isalnum())


def derive_provenance_class(
    *,
    content_type: str | None,
    sender: str | None,
    text: str,
    prev_assistant_text: str | None = None,
) -> str:
    """Mechanical provenance-class gate — the cheap part that rides enrichment.

    Derives the class from the originating turn's content_type + sender + a
    quote-vs-assertion check. The PARAPHRASE-vs-INFERENCE split is refined by the
    LLM in the enrichment pass; this gives the mechanical default.
    """
    ct = (content_type or "").strip().lower()
    sdr = (sender or "").strip().lower()

    # Etan-direct candidates: genuine user_message turns only. A role:user
    # envelope carrying tool_result / file_read content is NOT Etan speaking.
    if ct in _ETAN_CONTENT_TYPES or (
        sdr == "user" and ct not in _ASSISTANT_CONTENT_TYPES and ct not in {"file_read", "tool_result"}
    ):
        if (
            prev_assistant_text
            and len(text.strip()) <= _ENDORSEMENT_MAX_LEN
            and not _is_short_correction(text)
            and _echoes(text, prev_assistant_text)
        ):
            return ETAN_ENDORSEMENT
        return RAW_ETAN_DIRECT

    # assistant turns: paraphrase (quoting user) vs inference (own conclusion)
    if ct in _ASSISTANT_CONTENT_TYPES or sdr == "assistant":
        low = text.lower()
        if any(m in low for m in _QUOTE_MARKERS):
            return AGENT_PARAPHRASE
        return AGENT_INFERENCE

    # tool_result / file_read / unknown -> not a dialogue assertion; treat as
    # lowest authority so it can never be crowned.
    return AGENT_INFERENCE


# ack / filler words carry no propositional content: a short user turn made up
# only of these is a bare agreement to whatever was just said -> ENDORSEMENT.
_ACK_FILLER = {
    "yeah",
    "yep",
    "yes",
    "okay",
    "sure",
    "exactly",
    "that",
    "this",
    "right",
    "true",
    "agree",
    "agreed",
    "correct",
    "good",
    "great",
    "nice",
    "perfect",
    "love",
    "like",
    "fine",
    "cool",
    "totally",
    "absolutely",
    "indeed",
    "yup",
}


def _echoes(short_user_text: str, prev_assistant_text: str) -> bool:
    """Cheap lexical overlap: does the short user turn restate the assistant's
    framing? Used to demote echo-acks to ENDORSEMENT."""
    user_toks = {t for t in re.findall(r"[a-z0-9]{4,}", short_user_text.lower()) if t not in _ACK_FILLER}
    if not user_toks:
        # pure ack like "yeah exactly, that" with no content words -> it can only
        # be endorsing the thing just said.
        return True
    asst_toks = set(re.findall(r"[a-z0-9]{4,}", prev_assistant_text.lower()))
    overlap = user_toks & asst_toks
    return len(overlap) >= max(1, len(user_toks) // 2)


def _is_short_correction(short_user_text: str) -> bool:
    """Return True for terse user corrections that should not be endorsements."""
    low = short_user_text.strip().casefold()
    return bool(re.search(r"\b(no|nope|nah|wrong|actually|correction|instead|not)\b", low))


# --- claims + resolution ----------------------------------------------------


@dataclass
class Claim:
    """One mention of an (entity, attribute) = value, already classified.

    value is opaque: two claims CONTRADICT iff they assert different `value`
    strings for the same attribute. Contradiction detection (which is semantic)
    happens upstream; here value-equality is the contradiction signal.
    """

    id: str
    entity: str
    attribute: str
    value: str
    provenance_class: str
    timestamp: str  # ISO-8601; lexicographic compare == chronological for Z-stamps
    user_anchored: bool = True
    text: str = ""

    @property
    def rank(self) -> int:
        return PROVENANCE_RANK.get(self.provenance_class, 0)


@dataclass
class Resolution:
    entity: str
    attribute: str
    authoritative: Claim | None
    superseded: list[Claim] = field(default_factory=list)
    flagged_pending_user_confirm: list[Claim] = field(default_factory=list)
    disposition: str = "RESOLVED"  # RESOLVED | PENDING-USER-CONFIRM


def _is_unanchored_inference(claim: Claim) -> bool:
    return claim.provenance_class == AGENT_INFERENCE and not claim.user_anchored


def resolve(
    claims: list[Claim],
    *,
    enable_operational_evidence: bool = False,
    system_state_attributes: set[str] | None = None,
) -> Resolution:
    """Resolve a SINGLE (entity, attribute) group to one authoritative claim.

    Ordering: by entrenchment rank first, recency only as a tiebreaker within the
    top rank. Losers that assert a different value are superseded; unanchored
    agent-inferences that contradict (or stand alone) route to PENDING-USER-CONFIRM
    and are NEVER crowned, regardless of recency.
    """
    if not claims:
        return Resolution(entity="", attribute="", authoritative=None, disposition="RESOLVED")

    entity = claims[0].entity
    attribute = claims[0].attribute

    # anchored claims are eligible to be authoritative; unanchored inferences are not.
    eligible = [c for c in claims if not _is_unanchored_inference(c)]

    if not eligible:
        # only unanchored agent-inference exists for this attribute -> nothing to
        # crown; route every inference to confirmation.
        return Resolution(
            entity=entity,
            attribute=attribute,
            authoritative=None,
            flagged_pending_user_confirm=list(claims),
            disposition="PENDING-USER-CONFIRM",
        )

    top_rank = max(
        _claim_rank(
            c,
            enable_operational_evidence=enable_operational_evidence,
            system_state_attributes=system_state_attributes,
        )
        for c in eligible
    )
    top_class = [
        c
        for c in eligible
        if _claim_rank(
            c,
            enable_operational_evidence=enable_operational_evidence,
            system_state_attributes=system_state_attributes,
        )
        == top_rank
    ]
    # Recency tiebreak WITHIN the top class only. ISO strings are parsed and
    # compared as UTC instants; malformed/empty timestamps sort last. Identical
    # instants use lowest claim id as a stable deterministic tiebreak.
    authoritative = _select_authoritative(top_class)

    superseded: list[Claim] = []
    flagged: list[Claim] = []
    for c in claims:
        if c is authoritative or c.id == authoritative.id:
            continue
        if c.value == authoritative.value:
            # agrees with the winner — not a contradiction; nothing to supersede
            # or confirm (a faithful paraphrase/endorsement that concurs).
            continue
        if _is_unanchored_inference(c):
            # contradicting inference Etan never said directly -> confirm, not supersede
            flagged.append(c)
        else:
            # genuine lower-or-equal-standing competing fact -> supersede (reversible)
            superseded.append(c)

    return Resolution(
        entity=entity,
        attribute=attribute,
        authoritative=authoritative,
        superseded=superseded,
        flagged_pending_user_confirm=flagged,
        disposition="RESOLVED",
    )


def _claim_rank(
    claim: Claim,
    *,
    enable_operational_evidence: bool,
    system_state_attributes: set[str] | None,
) -> float:
    if (
        enable_operational_evidence
        and claim.provenance_class == OPERATIONAL_EVIDENCE
        and _is_system_state_attribute(claim.attribute, system_state_attributes)
    ):
        return _OPERATIONAL_EVIDENCE_RANK
    return float(claim.rank)


def _is_system_state_attribute(attribute: str, system_state_attributes: set[str] | None) -> bool:
    allowed = system_state_attributes or _DEFAULT_SYSTEM_STATE_ATTRIBUTES
    normalized_allowed = {_normalize_attribute_key(value) for value in allowed}
    return _normalize_attribute_key(attribute) in normalized_allowed


def _normalize_attribute_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9_]+", "_", str(value or "").strip().upper()).strip("_")


def _select_authoritative(claims: list[Claim]) -> Claim:
    parsed_by_id = {c.id: _parse_timestamp(c.timestamp) for c in claims}
    valid_times = [dt for dt in parsed_by_id.values() if dt is not None]
    if valid_times:
        best_dt = max(valid_times)
        latest = [c for c in claims if parsed_by_id[c.id] == best_dt]
    else:
        latest = list(claims)
    return min(latest, key=lambda c: c.id)


def _parse_timestamp(timestamp: str) -> datetime | None:
    raw = str(timestamp or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def resolve_entity(
    claims: list[Claim],
    *,
    enable_operational_evidence: bool = False,
    system_state_attributes: set[str] | None = None,
) -> dict[str, Resolution]:
    """Resolve ALL attributes of an entity independently (per-attribute authority).

    Groups claims by attribute and resolves each group on its own, so a
    foundational fact on one attribute is never buried by chatter on another.
    """
    groups: dict[str, list[Claim]] = {}
    for c in claims:
        groups.setdefault(c.attribute, []).append(c)
    return {
        attr: resolve(
            group,
            enable_operational_evidence=enable_operational_evidence,
            system_state_attributes=system_state_attributes,
        )
        for attr, group in groups.items()
    }
