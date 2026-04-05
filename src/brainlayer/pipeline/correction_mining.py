"""Mine correction-tagged chunks into structured correction pairs and KG facts."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from ..vector_store import VectorStore

CORRECTION_TAGS = ("correction", "user-correction", "clarification")


@dataclass(frozen=True)
class PatternSpec:
    regex: re.Pattern[str]
    pattern_type: str


@dataclass(frozen=True)
class CorrectionMatch:
    pattern_type: str
    entity_name: str
    attribute: str | None
    old_value: str | None
    new_value: str | None
    confidence: float


PATTERNS = [
    PatternSpec(
        re.compile(
            r"(?:^|\bno[,.]?\s+)(?P<entity>[\w][\w\s]+?)\s+is\s+not\s+(?P<value>.+?)(?:[.,]|$)",
            re.IGNORECASE,
        ),
        "negation",
    ),
    PatternSpec(
        re.compile(
            r"(?P<entity>[\w][\w\s]+?)\s+(?P<verb>works?\s+at|from|at)\s+(?P<value>.+?)(?:[.,]|$)",
            re.IGNORECASE,
        ),
        "association",
    ),
    PatternSpec(
        re.compile(
            r"(?P<left>[\w][\w\s]+?)\s+(?:also\s+known\s+as|aka|=)\s+(?P<right>.+?)(?:[.,]|$)",
            re.IGNORECASE,
        ),
        "alias",
    ),
    PatternSpec(
        re.compile(
            r"(?:^|no[,.]?\s+)?(?P<entity>[\w][\w\s]+?)\s+is\s+(?:actually\s+)?(?:a\s+)?(?P<value>.+?)(?:[.,]|$)",
            re.IGNORECASE,
        ),
        "identity",
    ),
    PatternSpec(
        re.compile(r"(?P<entity>[\u0590-\u05FF][\u0590-\u05FF\s]+?)\s+(?:זה|הוא|היא)\s+(?P<value>.+?)(?:[.,]|$)"),
        "hebrew_identity",
    ),
]

NON_ENTITY_SUBJECTS = {
    "he",
    "her",
    "here",
    "i",
    "it",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "those",
    "we",
    "what",
    "who",
    "you",
}
NON_ENTITY_PREFIXES = ("working ", "making ", "being ")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value.strip(" \t\r\n.,"))
    return cleaned or None


def _looks_like_hebrew(value: str) -> bool:
    return bool(re.search(r"[\u0590-\u05FF]", value))


def _is_likely_entity_name(value: str | None) -> bool:
    cleaned = _clean(value)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if lowered in NON_ENTITY_SUBJECTS:
        return False
    if any(lowered.startswith(prefix) for prefix in NON_ENTITY_PREFIXES):
        return False
    if _looks_like_hebrew(cleaned):
        return True
    tokens = cleaned.split()
    return cleaned[0].isupper() or len(tokens) > 1


def _association_attribute(verb: str) -> str:
    normalized = verb.lower().strip()
    if normalized.startswith("work"):
        return "works_at"
    if normalized == "from":
        return "from"
    return "at"


def _build_match(spec: PatternSpec, match: re.Match[str]) -> CorrectionMatch | None:
    groups = {key: _clean(value) for key, value in match.groupdict().items()}

    if spec.pattern_type == "identity":
        entity = groups["entity"]
        value = groups["value"]
        if not _is_likely_entity_name(entity):
            return None
        return CorrectionMatch(
            pattern_type="identity",
            entity_name=entity,
            attribute=value,
            old_value=None,
            new_value=value,
            confidence=0.85,
        )

    if spec.pattern_type == "hebrew_identity":
        entity = groups["entity"]
        value = groups["value"]
        if not _is_likely_entity_name(entity):
            return None
        return CorrectionMatch(
            pattern_type="hebrew_identity",
            entity_name=entity,
            attribute=value,
            old_value=None,
            new_value=value,
            confidence=0.88,
        )

    if spec.pattern_type == "negation":
        entity = groups["entity"]
        value = groups["value"]
        if not _is_likely_entity_name(entity):
            return None
        return CorrectionMatch(
            pattern_type="negation",
            entity_name=entity,
            attribute="negated_fact",
            old_value=value,
            new_value=None,
            confidence=0.7,
        )

    if spec.pattern_type == "association":
        entity = groups["entity"]
        value = groups["value"]
        verb = groups["verb"] or ""
        if not _is_likely_entity_name(entity) or not _is_likely_entity_name(value):
            return None
        return CorrectionMatch(
            pattern_type="association",
            entity_name=entity,
            attribute=_association_attribute(verb),
            old_value=None,
            new_value=value,
            confidence=0.9,
        )

    if spec.pattern_type == "alias":
        left = groups["left"]
        right = groups["right"]
        canonical = right if _is_likely_entity_name(right) else left
        alias = left if canonical == right else right
        if not _is_likely_entity_name(canonical) or not alias:
            return None
        return CorrectionMatch(
            pattern_type="alias",
            entity_name=canonical,
            attribute="alias",
            old_value=alias,
            new_value=canonical,
            confidence=0.95,
        )

    return None


def extract_corrections(text: str) -> list[CorrectionMatch]:
    """Extract structured correction pairs from free text."""
    matches: list[CorrectionMatch] = []
    spans: list[tuple[int, int]] = []
    for spec in PATTERNS:
        for raw_match in spec.regex.finditer(text):
            span = raw_match.span()
            if any(not (span[1] <= seen[0] or span[0] >= seen[1]) for seen in spans):
                continue
            candidate = _build_match(spec, raw_match)
            if candidate is not None:
                matches.append(candidate)
                spans.append(span)
    return matches


def mine_corrections(store: VectorStore) -> dict[str, Any]:
    """Extract correction pairs from correction-tagged chunks into a staging table."""
    cursor = store.conn.cursor()
    placeholders = ", ".join("?" for _ in CORRECTION_TAGS)
    rows = list(
        store._read_cursor().execute(
            f"""
            SELECT DISTINCT c.id, c.content
            FROM chunks c
            JOIN chunk_tags ct ON ct.chunk_id = c.id
            WHERE lower(ct.tag) IN ({placeholders})
            ORDER BY c.id
            """,
            tuple(tag.lower() for tag in CORRECTION_TAGS),
        )
    )

    stats: dict[str, Any] = {
        "chunks_processed": len(rows),
        "pairs_extracted": 0,
        "by_pattern_type": {},
    }

    for chunk_id, content in rows:
        cursor.execute("DELETE FROM correction_pairs WHERE chunk_id = ?", (chunk_id,))
        extracted = extract_corrections(content)
        for item in extracted:
            cursor.execute(
                """
                INSERT INTO correction_pairs (
                    chunk_id, pattern_type, entity_name, attribute, old_value, new_value, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    item.pattern_type,
                    item.entity_name,
                    item.attribute,
                    item.old_value,
                    item.new_value,
                    item.confidence,
                ),
            )
            stats["pairs_extracted"] += 1
            stats["by_pattern_type"][item.pattern_type] = stats["by_pattern_type"].get(item.pattern_type, 0) + 1

    stats["by_pattern_type"] = dict(sorted(stats["by_pattern_type"].items()))
    return stats


def _person_entity_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9\u0590-\u05ff]+", "-", name.lower()).strip("-")
    return f"corr-person-{slug or uuid.uuid4().hex[:8]}"


def _org_entity_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9\u0590-\u05ff]+", "-", name.lower()).strip("-")
    return f"corr-org-{slug or uuid.uuid4().hex[:8]}"


def _resolve_person_name(store: VectorStore, name: str) -> tuple[str, dict[str, Any] | None]:
    exact = store.get_entity_by_name("person", name)
    if exact is not None:
        return name, exact

    if len(name.split()) != 1:
        return name, None

    rows = list(
        store._read_cursor().execute(
            """
            SELECT id, entity_type, name, metadata, created_at, updated_at,
                   canonical_name, description, confidence, importance,
                   valid_from, valid_until, group_id
            FROM kg_entities
            WHERE entity_type = 'person' AND lower(name) LIKE lower(?)
            ORDER BY length(name) ASC
            """,
            (f"{name} %",),
        )
    )
    if len(rows) != 1:
        return name, None

    row = rows[0]
    return row[2], {
        "id": row[0],
        "entity_type": row[1],
        "name": row[2],
        "metadata": json.loads(row[3]) if row[3] else {},
        "created_at": row[4],
        "updated_at": row[5],
        "canonical_name": row[6],
        "description": row[7],
        "confidence": row[8],
        "importance": row[9],
        "valid_from": row[10],
        "valid_until": row[11],
        "group_id": row[12],
    }


def _ensure_person(store: VectorStore, name: str, *, identity: str | None = None, confidence: float = 0.8) -> str:
    resolved_name, existing = _resolve_person_name(store, name)
    metadata = dict(existing["metadata"]) if existing else {}
    if identity:
        metadata["identity"] = identity
    entity_id = existing["id"] if existing else _person_entity_id(resolved_name)
    return store.upsert_entity(
        entity_id,
        "person",
        resolved_name,
        metadata=metadata,
        confidence=confidence,
        importance=0.7,
        description=identity if identity and not _looks_like_hebrew(identity) else None,
    )


def _ensure_organization(store: VectorStore, name: str, confidence: float = 0.85) -> str:
    existing = store.get_entity_by_name("organization", name)
    entity_id = existing["id"] if existing else _org_entity_id(name)
    return store.upsert_entity(
        entity_id,
        "organization",
        name,
        metadata={"source": "correction-mining"},
        confidence=confidence,
        importance=0.6,
    )


def promote_corrections(store: VectorStore, min_confidence: float = 0.8) -> dict[str, int]:
    """Promote high-confidence correction pairs into KG entities, relations, and aliases."""
    cursor = store.conn.cursor()
    rows = list(
        store._read_cursor().execute(
            """
            SELECT id, chunk_id, pattern_type, entity_name, attribute, old_value, new_value, confidence
            FROM correction_pairs
            ORDER BY id
            """,
        )
    )

    stats = {
        "entities_upserted": 0,
        "relations_created": 0,
        "aliases_created": 0,
        "manual_review": 0,
    }
    seen_entities: set[str] = set()

    for _, chunk_id, pattern_type, entity_name, attribute, old_value, new_value, confidence in rows:
        if pattern_type == "negation":
            stats["manual_review"] += 1
            if confidence < min_confidence:
                continue

        if confidence < min_confidence:
            continue

        if pattern_type in {"identity", "hebrew_identity"}:
            entity_id = _ensure_person(store, entity_name, identity=new_value, confidence=confidence)
            if entity_id not in seen_entities:
                stats["entities_upserted"] += 1
                seen_entities.add(entity_id)
            store.link_entity_chunk(
                entity_id, chunk_id, relevance=confidence, context="correction-mining", mention_type="explicit"
            )
            continue

        if pattern_type == "association":
            source_id = _ensure_person(store, entity_name, confidence=confidence)
            target_id = _ensure_organization(store, new_value, confidence=confidence)
            for entity_id in (source_id, target_id):
                if entity_id not in seen_entities:
                    stats["entities_upserted"] += 1
                    seen_entities.add(entity_id)
            relation_id = f"corr-rel-{uuid.uuid4().hex[:12]}"
            store.add_relation(
                relation_id,
                source_id,
                target_id,
                attribute or "related_to",
                confidence=confidence,
                fact=f"{entity_name} {attribute.replace('_', ' ')} {new_value}" if attribute else None,
                source_chunk_id=chunk_id,
            )
            store.link_entity_chunk(
                source_id, chunk_id, relevance=confidence, context="correction-mining", mention_type="explicit"
            )
            store.link_entity_chunk(
                target_id, chunk_id, relevance=confidence, context="correction-mining", mention_type="explicit"
            )
            stats["relations_created"] += 1
            continue

        if pattern_type == "alias":
            entity_id = _ensure_person(store, entity_name, confidence=confidence)
            if entity_id not in seen_entities:
                stats["entities_upserted"] += 1
                seen_entities.add(entity_id)
            cursor.execute(
                """
                INSERT OR IGNORE INTO kg_entity_aliases (alias, entity_id, alias_type, created_at, valid_from)
                VALUES (?, ?, 'correction', strftime('%Y-%m-%dT%H:%M:%fZ','now'), strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                """,
                (old_value, entity_id),
            )
            stats["aliases_created"] += store.conn.changes()
            store.link_entity_chunk(
                entity_id, chunk_id, relevance=confidence, context="correction-mining", mention_type="explicit"
            )
            continue

    return stats
