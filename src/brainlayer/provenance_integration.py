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
    claims = [_row_to_claim(row, canonical_name) for row in rows]
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
                if _brain_supersede(store, conn, loser.id, authoritative.id):
                    report.superseded_count += 1
        for claim in resolution.flagged_pending_user_confirm:
            if _enqueue_pending_user_confirm(conn, claim):
                report.pending_confirm_count += 1
    _commit_if_supported(conn)
    return report


def confirm_pending(store, claim_id: str) -> ProvenanceConflictReport:
    """Confirm a pending inference and rerun resolution for its entity.

    Confirmation promotes the chunk to RAW-ETAN-DIRECT, removes its queue row,
    and lets the normal reversible supersede path resolve the affected
    attribute. The caller controls when this user-confirmed write is allowed.
    """
    conn = _conn(store)
    _ensure_pending_user_confirm_table(conn)
    row = conn.execute(
        """
        SELECT entity, attribute, chunk_id
        FROM provenance_pending_user_confirm
        WHERE chunk_id = ? OR id = ?
        """,
        (claim_id, claim_id),
    ).fetchone()
    if row is None:
        return ProvenanceConflictReport(
            entity="",
            entity_ids=[],
            resolutions={},
            dry_run=False,
            notes=[f"No pending provenance confirmation matched claim_id={claim_id}"],
        )

    entity = str(row[0])
    chunk_id = str(row[2])
    if "provenance_class" not in _columns(conn, "chunks"):
        conn.execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
    conn.execute("UPDATE chunks SET provenance_class = ? WHERE id = ?", (RAW_ETAN_DIRECT, chunk_id))
    conn.execute("DELETE FROM provenance_pending_user_confirm WHERE chunk_id = ? OR id = ?", (chunk_id, claim_id))
    _commit_if_supported(conn)
    return resolve_entity_conflicts(store, entity, dry_run=False)


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


def _entity_chunk_rows(conn, entity_ids: list[str]) -> list[dict[str, Any]]:
    chunk_cols = _columns(conn, "chunks")
    ec_cols = _columns(conn, "kg_entity_chunks")

    def chunk_expr(col: str, fallback: str = "NULL") -> str:
        return f"c.{col}" if col in chunk_cols else f"{fallback} AS {col}"

    def ec_expr(col: str, fallback: str = "NULL") -> str:
        return f"ec.{col}" if col in ec_cols else f"{fallback} AS {col}"

    placeholders = ",".join("?" for _ in entity_ids)
    status_filter = ""
    if "status" in chunk_cols:
        status_filter = "AND COALESCE(c.status, 'active') NOT IN ('archived', 'superseded')"

    sql = f"""
        SELECT
            c.id,
            c.content,
            {chunk_expr("content_type")},
            {chunk_expr("sender")},
            {chunk_expr("created_at", "'1970-01-01T00:00:00Z'")},
            {chunk_expr("provenance_class")},
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
        "context",
        "mention_type",
    ]
    return [dict(zip(keys, tuple(row), strict=True)) for row in conn.execute(sql, entity_ids)]


def _row_to_claim(row: dict[str, Any], entity: str) -> Claim:
    attribute, value = _attribute_value(row.get("content") or "", row.get("context"), row.get("mention_type"))
    provenance_class = str(row.get("provenance_class") or "").strip() or derive_provenance_class(
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


def _attribute_value(content: str, context: Any, mention_type: Any) -> tuple[str, str]:
    structured = _parse_context(context)
    if structured:
        attr = structured.get("attribute") or structured.get("relation_type") or structured.get("relation")
        value = structured.get("value") or structured.get("fact") or structured.get("summary")
        if attr and value:
            return _normalize_attribute(str(attr)), _normalize_value(str(value))

    for candidate in (str(context or ""), content):
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9 _-]{1,40})\s*:\s*(.+?)\s*$", candidate, flags=re.S)
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
