"""DB-facing provenance resolution helpers.

The pure resolver lives in provenance.py. This module is the small integration
surface that reads chunk-backed KG mentions, builds Claims, and optionally
applies reversible supersede / pending-confirm writes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .provenance import (
    AGENT_INFERENCE,
    RAW_ETAN_DIRECT,
    Claim,
    Resolution,
    derive_provenance_class,
    normalize_entity,
    resolve_entity,
)


@dataclass
class ProvenanceConflictReport:
    entity: str
    entity_ids: list[str]
    resolutions: dict[str, Resolution]
    dry_run: bool = True
    superseded_count: int = 0
    pending_confirm_count: int = 0
    claim_count: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class ProvenanceSweepReport:
    swept: int = 0
    superseded_count: int = 0
    pending_confirm_count: int = 0
    skipped_personal_count: int = 0
    entities: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def resolve_entity_conflicts(
    store,
    entity: str,
    *,
    dry_run: bool = True,
    enable_operational_evidence: bool = False,
    system_state_attributes: set[str] | None = None,
) -> ProvenanceConflictReport:
    """Resolve conflicting chunk claims for one KG entity.

    V1 attribute heuristic: prefer structured `kg_entity_chunks.context`
    attributes when present; otherwise parse deterministic `ATTRIBUTE: value`
    content prefixes; otherwise group as `MENTION`. This intentionally avoids
    LLM calls in the resolver consumer.
    """
    conn = _conn(store)
    entity_ids, canonical_name = _entity_ids(conn, entity)
    if not entity_ids:
        return ProvenanceConflictReport(
            entity=entity,
            entity_ids=[],
            resolutions={},
            dry_run=dry_run,
            notes=["No kg_entities row matched entity id/name/alias"],
        )

    rows = _entity_chunk_rows(conn, entity_ids)
    claims = _apply_confirmed_claim_overrides(conn, [_row_to_claim(row, canonical_name) for row in rows])
    actionable_claim_ids = {claim.id for row, claim in zip(rows, claims, strict=True) if _row_has_actionable_fact(row)}
    resolutions = resolve_entity(
        claims,
        enable_operational_evidence=enable_operational_evidence,
        system_state_attributes=system_state_attributes,
    )
    report = ProvenanceConflictReport(
        entity=canonical_name,
        entity_ids=entity_ids,
        resolutions=resolutions,
        dry_run=dry_run,
        claim_count=len(claims),
    )
    if dry_run:
        return report

    for resolution in resolutions.values():
        authoritative = resolution.authoritative
        if authoritative is not None:
            for loser in resolution.superseded:
                if authoritative.id not in actionable_claim_ids or loser.id not in actionable_claim_ids:
                    report.notes.append(
                        f"Skipped unstructured provenance supersede for attribute {resolution.attribute}"
                    )
                    continue
                if _brain_supersede(store, conn, loser.id, authoritative.id):
                    report.superseded_count += 1
                elif _is_personal_chunk(conn, loser.id):
                    report.notes.append(f"Skipped personal-data supersede for chunk {loser.id}")
        for claim in resolution.flagged_pending_user_confirm:
            if claim.id not in actionable_claim_ids:
                report.notes.append(f"Skipped unstructured pending-confirm for attribute {resolution.attribute}")
                continue
            if _enqueue_pending_user_confirm(conn, claim):
                report.pending_confirm_count += 1
    _commit_if_supported(conn)
    return report


def confirm_pending(store, claim_id: str) -> ProvenanceConflictReport:
    """Confirm a pending inference and rerun resolution for its entity.

    Confirmation promotes only the pending entity/attribute claim to
    RAW-ETAN-DIRECT, removes its queue row, and lets the normal reversible
    supersede path resolve the affected attribute. The caller controls when
    this user-confirmed write is allowed.
    """
    conn = _conn(store)
    _ensure_pending_user_confirm_table(conn)
    row = _find_pending_confirmation(conn, claim_id)
    if isinstance(row, ProvenanceConflictReport):
        return row

    pending_id = str(row[0])
    entity = str(row[1])
    attribute = str(row[2])
    chunk_id = str(row[3])
    value = str(row[4])
    _ensure_confirmed_claims_table(conn)
    conn.execute(
        """
        INSERT OR REPLACE INTO provenance_confirmed_claims
        (id, entity, attribute, chunk_id, value, provenance_class, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"{entity}:{attribute}:{chunk_id}:{value}",
            entity,
            attribute,
            chunk_id,
            value,
            RAW_ETAN_DIRECT,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.execute("DELETE FROM provenance_pending_user_confirm WHERE id = ?", (pending_id,))
    _commit_if_supported(conn)
    return resolve_entity_conflicts(store, entity, dry_run=False)


def _find_pending_confirmation(conn, claim_id: str):
    row = conn.execute(
        """
        SELECT id, entity, attribute, chunk_id, value
        FROM provenance_pending_user_confirm
        WHERE id = ?
        """,
        (claim_id,),
    ).fetchone()
    if row is not None:
        return row

    rows = list(
        conn.execute(
            """
            SELECT id, entity, attribute, chunk_id, value
            FROM provenance_pending_user_confirm
            WHERE chunk_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (claim_id,),
        )
    )
    if len(rows) != 1:
        qualifier = "No" if not rows else "Ambiguous"
        return ProvenanceConflictReport(
            entity="",
            entity_ids=[],
            resolutions={},
            dry_run=False,
            notes=[f"{qualifier} pending provenance confirmation matched claim_id={claim_id}"],
        )
    return rows[0]


def enqueue_provenance_resolution(
    store,
    entity: str,
    *,
    chunk_id: str | None = None,
    reason: str = "enrichment",
    commit: bool = True,
) -> bool:
    """Debounced enqueue for an entity whose claims should be re-resolved."""
    normalized = str(entity or "").strip()
    if not normalized:
        return False
    conn = _conn(store)
    _ensure_provenance_resolve_queue(conn)
    now = datetime.now(timezone.utc).isoformat()
    inserted = not conn.execute("SELECT 1 FROM provenance_resolve_queue WHERE entity = ?", (normalized,)).fetchone()
    conn.execute(
        """
        INSERT INTO provenance_resolve_queue (entity, chunk_id, reason, created_at, updated_at, attempts)
        VALUES (?, ?, ?, ?, ?, 0)
        ON CONFLICT(entity) DO UPDATE SET
            chunk_id = COALESCE(excluded.chunk_id, provenance_resolve_queue.chunk_id),
            reason = excluded.reason,
            updated_at = excluded.updated_at
        """,
        (normalized, chunk_id, reason, now, now),
    )
    if commit:
        _commit_if_supported(conn)
    return inserted


def enqueue_provenance_resolution_for_entities(
    store,
    entities: list[Any] | None,
    *,
    chunk_id: str | None = None,
    reason: str = "enrichment",
    commit: bool = True,
) -> int:
    """Extract entity names from enrichment payloads and enqueue each once."""
    count = 0
    for entity in entities or []:
        name = _entity_name_from_payload(entity)
        if name and enqueue_provenance_resolution(
            store,
            name,
            chunk_id=chunk_id,
            reason=reason,
            commit=commit,
        ):
            count += 1
    return count


def sweep_provenance_queue(
    store,
    *,
    limit: int = 100,
    enable_operational_evidence: bool = False,
) -> ProvenanceSweepReport:
    """Drain queued entities and apply reversible provenance resolution."""
    conn = _conn(store)
    _ensure_provenance_resolve_queue(conn)
    rows = list(
        conn.execute(
            """
            SELECT entity
            FROM provenance_resolve_queue
            ORDER BY updated_at ASC, entity ASC
            LIMIT ?
            """,
            (limit,),
        )
    )
    report = ProvenanceSweepReport()
    for row in rows:
        entity = str(row[0])
        entity_report = resolve_entity_conflicts(
            store,
            entity,
            dry_run=False,
            enable_operational_evidence=enable_operational_evidence,
        )
        report.swept += 1
        report.entities.append(entity)
        report.superseded_count += entity_report.superseded_count
        report.pending_confirm_count += entity_report.pending_confirm_count
        personal_notes = [note for note in entity_report.notes if "personal-data" in note]
        report.skipped_personal_count += len(personal_notes)
        report.notes.extend(entity_report.notes)
        if _should_retry_provenance_queue_entity(entity_report):
            _bump_provenance_queue_retry(conn, entity)
            continue
        conn.execute("DELETE FROM provenance_resolve_queue WHERE entity = ?", (entity,))
    _commit_if_supported(conn)
    return report


def list_pending_confirm(store) -> list[dict[str, Any]]:
    conn = _conn(store)
    _ensure_pending_user_confirm_table(conn)
    rows = conn.execute(
        """
        SELECT id, entity, attribute, chunk_id, value, provenance_class, reason, created_at
        FROM provenance_pending_user_confirm
        ORDER BY created_at ASC, entity ASC, attribute ASC
        """
    ).fetchall()
    keys = ["id", "entity", "attribute", "chunk_id", "value", "provenance_class", "reason", "created_at"]
    return [dict(zip(keys, tuple(row), strict=True)) for row in rows]


def reject_pending(store, claim_id: str) -> bool:
    """Reject a pending inference by reversibly archiving its source chunk."""
    conn = _conn(store)
    _ensure_pending_user_confirm_table(conn)
    row = _find_pending_confirmation(conn, claim_id)
    if isinstance(row, ProvenanceConflictReport):
        return False
    pending_id = str(row[0])
    chunk_id = str(row[3])
    archived = _archive_chunk(store, conn, chunk_id)
    if archived:
        conn.execute("DELETE FROM provenance_pending_user_confirm WHERE id = ?", (pending_id,))
        _commit_if_supported(conn)
    return archived


def get_entity_provenance_annotations(store, entity: str) -> dict[str, dict[str, Any]]:
    """Return authoritative/superseded/pending provenance by attribute for brain_entity."""
    conn = _conn(store)
    entity_ids, canonical_name = _entity_ids(conn, entity)
    if not entity_ids:
        return {}

    rows = _entity_chunk_rows(conn, entity_ids, include_archived=True)
    active_claims = _apply_confirmed_claim_overrides(
        conn, [_row_to_claim(row, canonical_name) for row in rows if _is_active_row(row)]
    )
    resolutions = resolve_entity(active_claims)
    all_claims = _apply_confirmed_claim_overrides(conn, [_row_to_claim(row, canonical_name) for row in rows])
    all_pairs = list(zip(rows, all_claims, strict=True))

    annotations: dict[str, dict[str, Any]] = {}
    for attribute, resolution in resolutions.items():
        # "MENTION" is the catch-all attribute for chunks with no parseable
        # (attribute: value) claim — i.e. the entity is merely referenced. These
        # are not resolved facts and must never be crowned AUTHORITATIVE, or a
        # stale chunk that a correction already superseded at the entity_facts
        # layer gets re-surfaced here. Only structured attributes are real.
        if attribute == "MENTION":
            continue
        authoritative = resolution.authoritative
        if authoritative is None:
            continue
        superseded = [
            _claim_annotation(claim)
            for row, claim in all_pairs
            if claim.attribute == attribute and _is_superseded_row(row) and claim.id != authoritative.id
        ]
        annotations[attribute] = {
            "authoritative": _claim_annotation(authoritative),
            "superseded": superseded,
            "pending": [],
        }

    for pending in _pending_for_entity(conn, canonical_name):
        attribute = pending["attribute"]
        annotations.setdefault(attribute, {"authoritative": None, "superseded": [], "pending": []})
        annotations[attribute]["pending"].append(pending)

    return annotations


def _conn(store):
    return getattr(store, "conn", store)


def _columns(conn, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _entity_ids(conn, entity: str) -> tuple[list[str], str]:
    rows = list(
        conn.execute("SELECT id, name FROM kg_entities WHERE id = ? OR lower(name) = lower(?)", (entity, entity))
    )
    if rows:
        return [str(row[0]) for row in rows], str(rows[0][1])

    if "kg_entity_aliases" in _tables(conn):
        alias_rows = list(
            conn.execute(
                """
                SELECT e.id, e.name
                FROM kg_entity_aliases a
                JOIN kg_entities e ON e.id = a.entity_id
                WHERE lower(a.alias) = lower(?)
                """,
                (entity,),
            )
        )
        if alias_rows:
            return [str(row[0]) for row in alias_rows], str(alias_rows[0][1])

    target = normalize_entity(entity)
    normalized_rows = [
        (str(row[0]), str(row[1]))
        for row in conn.execute("SELECT id, name FROM kg_entities")
        if normalize_entity(str(row[1])) == target
    ]
    if normalized_rows:
        return [row[0] for row in normalized_rows], normalized_rows[0][1]
    return [], entity


def _tables(conn) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _ensure_provenance_resolve_queue(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS provenance_resolve_queue (
            entity TEXT PRIMARY KEY,
            chunk_id TEXT,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def _should_retry_provenance_queue_entity(report: ProvenanceConflictReport) -> bool:
    return not report.entity_ids or any(note.startswith("No kg_entities row matched") for note in report.notes)


def _bump_provenance_queue_retry(conn, entity: str) -> None:
    conn.execute(
        """
        UPDATE provenance_resolve_queue
        SET attempts = attempts + 1,
            updated_at = ?
        WHERE entity = ?
        """,
        (datetime.now(timezone.utc).isoformat(), entity),
    )


def _entity_name_from_payload(entity: Any) -> str:
    if isinstance(entity, dict):
        for key in ("name", "text", "entity", "label"):
            value = entity.get(key)
            if value:
                return str(value).strip()
        return ""
    return str(entity or "").strip()


def _entity_chunk_rows(conn, entity_ids: list[str], *, include_archived: bool = False) -> list[dict[str, Any]]:
    chunk_cols = _columns(conn, "chunks")
    ec_cols = _columns(conn, "kg_entity_chunks")

    def chunk_expr(col: str, fallback: str = "NULL") -> str:
        return f"c.{col}" if col in chunk_cols else f"{fallback} AS {col}"

    def ec_expr(col: str, fallback: str = "NULL") -> str:
        return f"ec.{col}" if col in ec_cols else f"{fallback} AS {col}"

    placeholders = ",".join("?" for _ in entity_ids)
    status_filter = ""
    lifecycle_filters: list[str] = []
    if not include_archived:
        if "status" in chunk_cols:
            lifecycle_filters.append("COALESCE(c.status, 'active') NOT IN ('archived', 'superseded')")
        if "superseded_by" in chunk_cols:
            lifecycle_filters.append("c.superseded_by IS NULL")
        if "archived_at" in chunk_cols:
            lifecycle_filters.append("c.archived_at IS NULL")
        if "archived" in chunk_cols:
            lifecycle_filters.append("COALESCE(c.archived, 0) = 0")
    if lifecycle_filters:
        status_filter = "AND " + " AND ".join(lifecycle_filters)

    sql = f"""
        SELECT
            c.id,
            c.content,
            {chunk_expr("content_type")},
            {chunk_expr("sender")},
            {chunk_expr("created_at", "'1970-01-01T00:00:00Z'")},
            {chunk_expr("provenance_class")},
            {chunk_expr("source")},
            {chunk_expr("source_file")},
            {chunk_expr("status", "'active'")},
            {chunk_expr("superseded_by")},
            {chunk_expr("archived", "0")},
            {chunk_expr("archived_at")},
            {ec_expr("context")},
            {ec_expr("mention_type")}
        FROM kg_entity_chunks ec
        JOIN chunks c ON c.id = ec.chunk_id
        WHERE ec.entity_id IN ({placeholders})
        {status_filter}
    """
    keys = [
        "id",
        "content",
        "content_type",
        "sender",
        "created_at",
        "provenance_class",
        "source",
        "source_file",
        "status",
        "superseded_by",
        "archived",
        "archived_at",
        "context",
        "mention_type",
    ]
    return [dict(zip(keys, tuple(row), strict=True)) for row in conn.execute(sql, entity_ids)]


def _row_to_claim(row: dict[str, Any], entity: str) -> Claim:
    attribute, value = _attribute_value(row.get("content") or "", row.get("context"), row.get("mention_type"))
    provenance_class = str(row.get("provenance_class") or "").strip()
    if not provenance_class:
        source = str(row.get("source") or "").strip().lower()
        source_file = str(row.get("source_file") or "").strip().lower()
        if source == "manual" and source_file in {"brainlayer-store", "brainbar-store", "brainlayer-queue"}:
            provenance_class = RAW_ETAN_DIRECT
        else:
            provenance_class = derive_provenance_class(
                content_type=row.get("content_type"),
                sender=row.get("sender"),
                text=str(row.get("content") or ""),
            )
    return Claim(
        id=str(row["id"]),
        entity=entity,
        attribute=attribute,
        value=value,
        provenance_class=provenance_class,
        timestamp=str(row.get("created_at") or "1970-01-01T00:00:00Z"),
        user_anchored=provenance_class != AGENT_INFERENCE,
        text=str(row.get("content") or ""),
    )


def _apply_confirmed_claim_overrides(conn, claims: list[Claim]) -> list[Claim]:
    if not claims or "provenance_confirmed_claims" not in _tables(conn):
        return claims
    rows = conn.execute(
        """
        SELECT entity, attribute, chunk_id, value, provenance_class
        FROM provenance_confirmed_claims
        """
    ).fetchall()
    confirmed = {(str(row[0]), str(row[1]), str(row[2]), str(row[3])): str(row[4] or RAW_ETAN_DIRECT) for row in rows}
    for claim in claims:
        provenance_class = confirmed.get((claim.entity, claim.attribute, claim.id, claim.value))
        if provenance_class:
            claim.provenance_class = provenance_class
            claim.user_anchored = True
    return claims


_ATTRIBUTE_VALUE_RE = re.compile(r"\s*([A-Za-z][A-Za-z0-9 _-]{1,40})\s*:\s*(.+?)\s*$", flags=re.S)


def _row_has_actionable_fact(row: dict[str, Any]) -> bool:
    structured = _parse_context(row.get("context"))
    if structured:
        attr = structured.get("attribute") or structured.get("relation_type") or structured.get("relation")
        value = structured.get("value") or structured.get("fact") or structured.get("summary")
        if attr and value:
            return True

    return any(
        bool(_ATTRIBUTE_VALUE_RE.match(candidate))
        for candidate in (str(row.get("context") or ""), str(row.get("content") or ""))
    )


def _attribute_value(content: str, context: Any, mention_type: Any) -> tuple[str, str]:
    structured = _parse_context(context)
    if structured:
        attr = structured.get("attribute") or structured.get("relation_type") or structured.get("relation")
        value = structured.get("value") or structured.get("fact") or structured.get("summary")
        if attr and value:
            return _normalize_attribute(str(attr)), _normalize_value(str(value))

    for candidate in (str(context or ""), content):
        match = _ATTRIBUTE_VALUE_RE.match(candidate)
        if match:
            return _normalize_attribute(match.group(1)), _normalize_value(match.group(2))

    if mention_type:
        return _normalize_attribute(str(mention_type)), _normalize_value(content)
    return "MENTION", _normalize_value(content)


def _parse_context(context: Any) -> dict[str, Any]:
    if isinstance(context, dict):
        return context
    if not isinstance(context, str) or not context.strip():
        return {}
    try:
        parsed = json.loads(context)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_attribute(value: str) -> str:
    return re.sub(r"[^A-Z0-9_]+", "_", value.strip().upper()).strip("_") or "MENTION"


def _normalize_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _brain_supersede(store, conn, old_chunk_id: str, new_chunk_id: str) -> bool:
    if _is_personal_chunk(conn, old_chunk_id):
        return False

    if hasattr(store, "supersede_chunk"):
        return bool(store.supersede_chunk(old_chunk_id, new_chunk_id))

    cols = _columns(conn, "chunks")
    if "superseded_by" not in cols:
        return False
    if not conn.execute("SELECT 1 FROM chunks WHERE id = ?", (old_chunk_id,)).fetchone():
        return False
    if not conn.execute("SELECT 1 FROM chunks WHERE id = ?", (new_chunk_id,)).fetchone():
        return False
    updates = "superseded_by = ?"
    if "status" in cols:
        updates += ", status = 'superseded'"
    conn.execute(f"UPDATE chunks SET {updates} WHERE id = ?", (new_chunk_id, old_chunk_id))
    return True


_PERSONAL_TYPES = {"journal"}
_PERSONAL_KEYWORDS = ("health", "family", "relationship", "finance", "financial", "personal", "therapy", "medical")


def _is_personal_chunk(conn, chunk_id: str) -> bool:
    cols = _columns(conn, "chunks")
    if not conn.execute("SELECT 1 FROM chunks WHERE id = ?", (chunk_id,)).fetchone():
        return False
    select_cols = ["content"]
    select_cols.append("content_type" if "content_type" in cols else "NULL AS content_type")
    row = conn.execute(f"SELECT {', '.join(select_cols)} FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if row is None:
        return False
    content = str(row[0] or "")
    content_type = str(row[1] or "").strip().lower()
    return content_type in _PERSONAL_TYPES or any(keyword in content.lower() for keyword in _PERSONAL_KEYWORDS)


def _archive_chunk(store, conn, chunk_id: str) -> bool:
    if hasattr(store, "archive_chunk"):
        return bool(store.archive_chunk(chunk_id))

    if not conn.execute("SELECT 1 FROM chunks WHERE id = ?", (chunk_id,)).fetchone():
        return False
    cols = _columns(conn, "chunks")
    updates = []
    params: list[Any] = []
    if "status" in cols:
        updates.append("status = 'archived'")
    if "archived" in cols:
        updates.append("archived = 1")
    if "archived_at" in cols:
        updates.append("archived_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
    if not updates:
        updates.append("status = 'archived'")
        conn.execute("ALTER TABLE chunks ADD COLUMN status TEXT DEFAULT 'active'")
    params.append(chunk_id)
    conn.execute(f"UPDATE chunks SET {', '.join(updates)} WHERE id = ?", params)
    return True


def _is_active_row(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "active").lower()
    return (
        status not in {"archived", "superseded"}
        and not row.get("superseded_by")
        and not row.get("archived_at")
        and not bool(row.get("archived"))
    )


def _is_superseded_row(row: dict[str, Any]) -> bool:
    return str(row.get("status") or "").lower() == "superseded" or bool(row.get("superseded_by"))


def _claim_annotation(claim: Claim) -> dict[str, Any]:
    return {
        "chunk_id": claim.id,
        "value": claim.value,
        "provenance_class": claim.provenance_class,
        "evidence": claim.id,
        "text": claim.text,
    }


def _pending_for_entity(conn, entity: str) -> list[dict[str, Any]]:
    if "provenance_pending_user_confirm" not in _tables(conn):
        return []
    rows = conn.execute(
        """
        SELECT id, entity, attribute, chunk_id, value, provenance_class, reason, created_at
        FROM provenance_pending_user_confirm
        WHERE entity = ?
        ORDER BY created_at ASC
        """,
        (entity,),
    ).fetchall()
    keys = ["id", "entity", "attribute", "chunk_id", "value", "provenance_class", "reason", "created_at"]
    return [dict(zip(keys, tuple(row), strict=True)) for row in rows]


def _ensure_pending_user_confirm_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS provenance_pending_user_confirm (
            id TEXT PRIMARY KEY,
            entity TEXT NOT NULL,
            attribute TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            value TEXT NOT NULL,
            provenance_class TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )


def _ensure_confirmed_claims_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS provenance_confirmed_claims (
            id TEXT PRIMARY KEY,
            entity TEXT NOT NULL,
            attribute TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            value TEXT NOT NULL,
            provenance_class TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(entity, attribute, chunk_id, value)
        )
        """
    )


def _enqueue_pending_user_confirm(conn, claim: Claim) -> bool:
    _ensure_pending_user_confirm_table(conn)
    pending_id = f"{claim.entity}:{claim.attribute}:{claim.id}"
    if conn.execute("SELECT 1 FROM provenance_pending_user_confirm WHERE id = ?", (pending_id,)).fetchone():
        return False
    conn.execute(
        """
        INSERT OR IGNORE INTO provenance_pending_user_confirm
        (id, entity, attribute, chunk_id, value, provenance_class, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pending_id,
            claim.entity,
            claim.attribute,
            claim.id,
            claim.value,
            claim.provenance_class,
            "unanchored agent inference requires user confirmation",
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    return True


def _commit_if_supported(conn) -> None:
    commit = getattr(conn, "commit", None)
    if callable(commit):
        commit()
