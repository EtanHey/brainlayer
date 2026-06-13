"""Ingest-time autosupersession detector for provenance-resolved chunks.

This module is intentionally standalone from the enrichment hook. It only
detects same-entity contradictions, delegates authority to the provenance
class-gate, and applies reversible supersedes through the existing integration
helper when explicitly run with dry_run=False.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .content_class import _PERSONAL_RE, _PERSONAL_RISK_RE
from .provenance import AGENT_INFERENCE, Claim, derive_provenance_class, normalize_entity, resolve
from .provenance_integration import (
    _attribute_value,
    _brain_supersede,
    _commit_if_supported,
    _conn,
    _enqueue_pending_user_confirm,
    _entity_chunk_rows,
    _entity_ids,
    _row_has_actionable_fact,
)


@dataclass(frozen=True)
class AutoSupersedeDecision:
    action: str
    entity: str
    attribute: str | None
    old_chunk_id: str | None
    new_chunk_id: str
    reason: str


@dataclass
class AutoSupersedeReport:
    entity: str
    dry_run: bool
    candidate_count: int = 0
    contradiction_count: int = 0
    would_supersede_count: int = 0
    superseded_count: int = 0
    pending_confirm_count: int = 0
    skipped_count: int = 0
    skipped_reason: str | None = None
    attribute_dispositions: dict[str, str] = field(default_factory=dict)
    authoritative_by_attribute: dict[str, str | None] = field(default_factory=dict)
    decisions: list[AutoSupersedeDecision] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def gather_same_entity(conn, entity: str) -> list[dict[str, Any]]:
    """Return active chunk rows for entity ids matching the normalized entity."""
    db = _conn(conn)
    entity_ids, _canonical_name = _normalized_entity_ids(db, entity)
    if not entity_ids:
        return []
    rows = _entity_chunk_rows(db, entity_ids)
    return sorted(rows, key=lambda row: (str(row.get("created_at") or ""), str(row.get("id") or "")))


def detect_contradiction(new_chunk: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, str]:
    """V1 contradiction: same normalized entity and attribute, different value."""
    if not _same_chunk_entity(new_chunk, candidate):
        return False, ""
    if not _chunk_has_actionable_fact(candidate):
        return False, ""

    candidate_attribute, candidate_value = _chunk_attribute_value(candidate)
    for claim_chunk in _chunk_claim_chunks(new_chunk):
        if not _chunk_has_actionable_fact(claim_chunk):
            continue
        new_attribute, new_value = _chunk_attribute_value(claim_chunk)
        if new_attribute == candidate_attribute:
            return new_value != candidate_value, new_attribute
    return False, ""


def auto_supersede(
    store,
    new_chunk: dict[str, Any],
    *,
    dry_run: bool = True,
    enable_operational_evidence: bool = False,
    system_state_attributes: set[str] | None = None,
) -> AutoSupersedeReport:
    """Detect and optionally apply reversible supersedes for one newly-ingested chunk."""
    db = _conn(store)
    entity = _new_chunk_entity(new_chunk)
    entity_ids, canonical_entity = _normalized_entity_ids(db, entity)
    report = AutoSupersedeReport(entity=canonical_entity, dry_run=dry_run)

    if _is_personal_entity_or_chunk(entity, new_chunk):
        return _skip_personal(report, new_chunk)

    if not entity_ids:
        report.notes.append("No kg_entities row matched normalized entity")
        return report

    all_candidates = [
        {**dict(row), "entity": canonical_entity}
        for row in _entity_chunk_rows(db, entity_ids)
        if str(row.get("id")) != _chunk_id(new_chunk)
    ]
    report.candidate_count = len(all_candidates)

    if any(_is_personal_entity_or_chunk(canonical_entity, candidate) for candidate in all_candidates):
        return _skip_personal(report, new_chunk)

    candidates = [candidate for candidate in all_candidates if _chunk_has_actionable_fact(candidate)]
    skipped_unstructured = len(all_candidates) - len(candidates)
    if skipped_unstructured:
        report.notes.append(f"Skipped {skipped_unstructured} unstructured candidate row(s)")

    new_claim_chunks = [
        claim_chunk for claim_chunk in _chunk_claim_chunks(new_chunk) if _chunk_has_actionable_fact(claim_chunk)
    ]
    if not new_claim_chunks:
        report.notes.append("Skipped unstructured new chunk")
        return report
    new_claims_by_attribute = _chunks_by_attribute(new_claim_chunks)
    candidates_by_attribute = _chunks_by_attribute(candidates)

    for candidate in candidates:
        is_contradiction, attribute = detect_contradiction(new_chunk, candidate)
        if is_contradiction:
            report.contradiction_count += 1

    for attribute, new_rows in new_claims_by_attribute.items():
        if attribute == "MENTION":
            report.notes.append("Skipped unstructured MENTION autosupersede group")
            continue
        rows = [*new_rows, *candidates_by_attribute.get(attribute, [])]
        claims = [_chunk_to_claim(row, canonical_entity) for row in rows]
        resolution = resolve(
            claims,
            enable_operational_evidence=enable_operational_evidence,
            system_state_attributes=system_state_attributes,
        )

        authoritative = resolution.authoritative
        report.attribute_dispositions[attribute] = resolution.disposition
        report.authoritative_by_attribute[attribute] = authoritative.id if authoritative is not None else None
        if authoritative is None:
            for claim in resolution.flagged_pending_user_confirm:
                _record_pending(report, db, claim, dry_run=dry_run)
            continue

        new_claim_ids = {_chunk_id(row) for row in new_rows}
        if authoritative.id not in new_claim_ids:
            report.notes.append(f"Existing chunk {authoritative.id} remains authoritative for {attribute}")

        for loser in resolution.superseded:
            _record_supersede(report, store, db, loser, authoritative, dry_run=dry_run)
        for claim in resolution.flagged_pending_user_confirm:
            _record_pending(report, db, claim, dry_run=dry_run)

    if not dry_run:
        _commit_if_supported(db)
    return report


def _normalized_entity_ids(conn, entity: str) -> tuple[list[str], str]:
    entity_ids, canonical_name = _entity_ids(conn, entity)
    target = normalize_entity(entity)
    seen = set(entity_ids)
    matched_names: dict[str, str] = {entity_id: canonical_name for entity_id in entity_ids}

    for entity_id, name in _iter_entity_names(conn):
        if normalize_entity(name) == target and entity_id not in seen:
            seen.add(entity_id)
            entity_ids.append(entity_id)
            matched_names[entity_id] = name

    for entity_id, name, alias in _iter_entity_aliases(conn):
        if normalize_entity(alias) == target and entity_id not in seen:
            seen.add(entity_id)
            entity_ids.append(entity_id)
            matched_names[entity_id] = name

    if entity_ids:
        return entity_ids, matched_names.get(entity_ids[0], canonical_name)
    return [], entity


def _iter_entity_names(conn):
    if not _table_exists(conn, "kg_entities"):
        return
    for row in conn.execute("SELECT id, name FROM kg_entities"):
        yield str(row[0]), str(row[1])


def _iter_entity_aliases(conn):
    if not _table_exists(conn, "kg_entity_aliases"):
        return
    rows = conn.execute(
        """
        SELECT e.id, e.name, a.alias
        FROM kg_entity_aliases a
        JOIN kg_entities e ON e.id = a.entity_id
        """
    )
    for row in rows:
        yield str(row[0]), str(row[1]), str(row[2])


def _table_exists(conn, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    return row is not None


def _same_chunk_entity(new_chunk: dict[str, Any], candidate: dict[str, Any]) -> bool:
    new_entity = _chunk_entity(new_chunk)
    candidate_entity = _chunk_entity(candidate)
    if not new_entity or not candidate_entity:
        return False
    return normalize_entity(new_entity) == normalize_entity(candidate_entity)


def _new_chunk_entity(new_chunk: dict[str, Any]) -> str:
    return _chunk_entity(new_chunk) or ""


def _chunk_entity(chunk: dict[str, Any]) -> str:
    for key in ("entity", "entity_name", "name"):
        value = chunk.get(key)
        if value:
            return str(value)
    return ""


def _chunk_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or "")


def _chunk_attribute_value(chunk: dict[str, Any]) -> tuple[str, str]:
    if chunk.get("attribute") is not None and chunk.get("value") is not None:
        return _attribute_value(
            f"{chunk.get('attribute')}: {chunk.get('value')}",
            None,
            None,
        )
    return _attribute_value(
        str(chunk.get("content") or ""),
        chunk.get("context"),
        chunk.get("mention_type"),
    )


def _chunk_has_actionable_fact(chunk: dict[str, Any]) -> bool:
    if chunk.get("attribute") is not None and chunk.get("value") is not None:
        return True
    return _row_has_actionable_fact(chunk)


def _chunk_claim_chunks(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    claims = chunk.get("claims")
    if not isinstance(claims, list) or not claims:
        return [chunk]

    claim_chunks: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        merged = dict(chunk)
        merged.update(claim)
        if not merged.get("content") and merged.get("attribute") is not None and merged.get("value") is not None:
            merged["content"] = f"{merged['attribute']}: {merged['value']}"
        claim_chunks.append(merged)
    return claim_chunks or [chunk]


def _chunks_by_attribute(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_attribute: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        attribute, _value = _chunk_attribute_value(chunk)
        by_attribute.setdefault(attribute, []).append(chunk)
    return by_attribute


def _chunk_to_claim(chunk: dict[str, Any], entity: str) -> Claim:
    attribute, value = _chunk_attribute_value(chunk)
    provenance_class = str(chunk.get("provenance_class") or "").strip() or derive_provenance_class(
        content_type=chunk.get("content_type"),
        sender=chunk.get("sender"),
        text=str(chunk.get("content") or ""),
    )
    return Claim(
        id=_chunk_id(chunk),
        entity=entity,
        attribute=attribute,
        value=value,
        provenance_class=provenance_class,
        timestamp=str(chunk.get("created_at") or "1970-01-01T00:00:00Z"),
        user_anchored=provenance_class != AGENT_INFERENCE,
        text=str(chunk.get("content") or ""),
    )


def _record_supersede(
    report: AutoSupersedeReport,
    store,
    conn,
    loser: Claim,
    authoritative: Claim,
    *,
    dry_run: bool,
) -> None:
    report.would_supersede_count += 1
    report.decisions.append(
        AutoSupersedeDecision(
            action="SUPERSEDE",
            entity=authoritative.entity,
            attribute=authoritative.attribute,
            old_chunk_id=loser.id,
            new_chunk_id=authoritative.id,
            reason=f"{loser.provenance_class} loses to {authoritative.provenance_class}",
        )
    )
    if not dry_run and _brain_supersede(store, conn, loser.id, authoritative.id):
        report.superseded_count += 1


def _record_pending(report: AutoSupersedeReport, conn, claim: Claim, *, dry_run: bool) -> None:
    report.decisions.append(
        AutoSupersedeDecision(
            action="PENDING-USER-CONFIRM",
            entity=claim.entity,
            attribute=claim.attribute,
            old_chunk_id=claim.id,
            new_chunk_id="",
            reason="unanchored agent inference requires user confirmation",
        )
    )
    if dry_run:
        report.pending_confirm_count += 1
    elif _enqueue_pending_user_confirm(conn, claim):
        report.pending_confirm_count += 1


def _skip_personal(report: AutoSupersedeReport, new_chunk: dict[str, Any]) -> AutoSupersedeReport:
    report.skipped_count = 1
    report.skipped_reason = "skipped: personal"
    report.decisions.append(
        AutoSupersedeDecision(
            action="SKIP",
            entity=report.entity,
            attribute=None,
            old_chunk_id=None,
            new_chunk_id=_chunk_id(new_chunk),
            reason=report.skipped_reason,
        )
    )
    return report


def _is_personal_entity_or_chunk(entity: str, chunk: dict[str, Any]) -> bool:
    if _PERSONAL_RISK_RE.search(str(entity or "")):
        return True
    content_type = str(chunk.get("content_type") or "").strip().lower()
    if content_type in {"journal", "personal", "health", "medical", "therapy"}:
        return True
    text = " ".join(
        str(chunk.get(key) or "")
        for key in (
            "content",
            "context",
            "mention_type",
        )
    )
    return bool(_PERSONAL_RE.search(text))
