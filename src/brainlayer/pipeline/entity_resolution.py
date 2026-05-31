"""Entity resolution — map extracted names to canonical KG entities.

Cascading approach:
1. Exact match (name or alias, case-insensitive)
2. Vector similarity fallback (cosine on entity embeddings, threshold 0.92)
3. Create new entity if no match above 0.75

Hebrew prefix stripping handles morphological prefixes (ב,ל,מ,ש,ה,ו,כ).
"""

import logging
import re
import unicodedata
from typing import Callable, List, Optional

from ..vector_store import VectorStore

logger = logging.getLogger(__name__)

# Hebrew prefix characters that fuse with nouns (prepositions, articles, conjunctions)
_HEBREW_PREFIXES = set("בלמשהוכ")

# Hebrew Unicode range
_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


def strip_hebrew_prefix(text: str) -> str:
    """Strip common Hebrew morphological prefixes.

    Only strips if:
    - Text starts with a Hebrew character
    - First char is a known prefix
    - Remaining text is at least 2 chars
    """
    if not text or len(text) < 3:
        return text

    # Only strip from Hebrew text
    if not _HEBREW_RE.match(text[0]):
        return text

    if text[0] in _HEBREW_PREFIXES and len(text) - 1 >= 2:
        return text[1:]

    return text


def _normalize_name(name: str) -> str:
    """Normalize a name for comparison: lowercase, strip whitespace, remove accents."""
    name = name.strip().lower()
    # Remove Unicode accents/diacritics (nikud for Hebrew)
    name = "".join(c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn")
    return name


def resolve_entity(
    name: str,
    entity_type: str,
    context: str,
    store: VectorStore,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
) -> str:
    """Resolve an extracted name to an existing entity ID, or create a new one.

    Cascading resolution:
    1. Exact name match (case-insensitive, via get_entity_by_name)
    2. Alias match (case-insensitive)
    3. Hebrew prefix-stripped match
    4. Vector similarity fallback (if embed_fn provided, threshold 0.92 auto-match)
    5. Create new entity (only if nothing matches above 0.75)
    """
    # 1. Exact name match (UNIQUE(entity_type, name) — case-sensitive in DB)
    existing = store.get_entity_by_name(entity_type, name)
    if existing:
        return existing["id"]

    # Try case-insensitive via FTS5
    candidates = store.search_entities(name, entity_type=entity_type, limit=3)
    for c in candidates:
        if _normalize_name(c["name"]) == _normalize_name(name):
            return c["id"]

    # 2. Alias match
    alias_match = store.get_entity_by_alias(name)
    if alias_match and alias_match["entity_type"] == entity_type:
        return alias_match["id"]

    # 3. Hebrew prefix-stripped match
    stripped = strip_hebrew_prefix(name)
    if stripped != name:
        alias_match = store.get_entity_by_alias(stripped)
        if alias_match and alias_match["entity_type"] == entity_type:
            return alias_match["id"]
        existing = store.get_entity_by_name(entity_type, stripped)
        if existing:
            store.add_entity_alias(name, existing["id"], alias_type="hebrew_prefix")
            return existing["id"]

    # 4. Vector similarity fallback
    if embed_fn is not None:
        try:
            query_embedding = embed_fn(f"{entity_type}: {name}")
            vec_candidates = store.search_entities_semantic(query_embedding, entity_type=entity_type, limit=5)
            if vec_candidates:
                best = vec_candidates[0]
                # sqlite-vec distance is cosine distance (0=identical, 2=opposite)
                similarity = 1.0 - best.get("distance", 1.0)
                if similarity >= 0.92:
                    # High confidence — auto-match and store alias
                    logger.debug(
                        "Vector match: '%s' → '%s' (sim=%.3f)",
                        name,
                        best["name"],
                        similarity,
                    )
                    store.add_entity_alias(name, best["id"], alias_type="vector_match")
                    return best["id"]
                elif similarity >= 0.75:
                    # Moderate confidence — still match but log for review
                    logger.info(
                        "Fuzzy vector match: '%s' → '%s' (sim=%.3f, needs review)",
                        name,
                        best["name"],
                        similarity,
                    )
                    store.add_entity_alias(name, best["id"], alias_type="fuzzy_vector")
                    return best["id"]
        except Exception as e:
            logger.debug("Vector similarity fallback failed for '%s': %s", name, e)

    # 5. No match — create new entity with auto-generated ID
    import uuid

    entity_id = f"{entity_type}-{uuid.uuid4().hex[:12]}"
    embedding = embed_fn(f"{entity_type}: {name}") if embed_fn else None
    return store.upsert_entity(entity_id, entity_type, name, metadata={}, embedding=embedding)


def _max_present(left: float | None, right: float | None) -> float | None:
    """Return the larger non-null numeric value."""
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _min_present(left: int | None, right: int | None) -> int | None:
    """Return the smaller non-null integer value."""
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _earliest_present(left: str | None, right: str | None) -> str | None:
    """Return the earliest non-null comparable timestamp."""
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _latest_valid_until(left: str | None, right: str | None) -> str | None:
    """Return the widest valid-until bound; None represents open-ended validity."""
    if left is None or right is None:
        return None
    return max(left, right)


def _merge_mention_type(left: str | None, right: str | None) -> str | None:
    """Preserve explicit mentions over inferred/implicit support."""
    if left == "explicit" or right == "explicit":
        return "explicit"
    return left or right


def merge_entities_preserving_links(store: VectorStore, keep_id: str, merge_id: str) -> dict[str, int]:
    """Merge merge_id into keep_id while preserving stronger duplicate evidence.

    Steps:
    1. Move chunk links from merge_id to keep_id
    2. Merge duplicate chunk-link support instead of dropping richer conflicts
    3. Move relations (both directions) from merge_id to keep_id
    4. Merge duplicate relation support instead of dropping richer conflicts
    5. Store merged entity's name as alias on keep_id
    6. Delete merge_id entity
    """
    stats = {
        "chunk_links_moved": 0,
        "chunk_conflicts_merged": 0,
        "relations_moved": 0,
        "relation_conflicts_merged": 0,
        "aliases_moved": 0,
        "entities_deleted": 0,
    }
    if keep_id == merge_id:
        return stats

    cursor = store.conn.cursor()
    merged = store.get_entity(merge_id)
    if not merged:
        return stats

    # 1. Move chunk links, merging same-chunk conflicts into the canonical row.
    chunk_rows = list(
        cursor.execute(
            """
            SELECT chunk_id, relevance, context, mention_type, relation_tier, weight
            FROM kg_entity_chunks
            WHERE entity_id = ?
            """,
            (merge_id,),
        )
    )
    for chunk_id, relevance, context, mention_type, relation_tier, weight in chunk_rows:
        existing = cursor.execute(
            """
            SELECT relevance, context, mention_type, relation_tier, weight
            FROM kg_entity_chunks
            WHERE entity_id = ? AND chunk_id = ?
            """,
            (keep_id, chunk_id),
        ).fetchone()
        if existing:
            existing_relevance, existing_context, existing_mention_type, existing_tier, existing_weight = existing
            merged_relevance = _max_present(existing_relevance, relevance)
            merged_context = existing_context
            if context and (not existing_context or (relevance or 0.0) >= (existing_relevance or 0.0)):
                merged_context = context
            cursor.execute(
                """
                UPDATE kg_entity_chunks
                SET relevance = ?,
                    context = ?,
                    mention_type = ?,
                    relation_tier = ?,
                    weight = ?
                WHERE entity_id = ? AND chunk_id = ?
                """,
                (
                    merged_relevance,
                    merged_context,
                    _merge_mention_type(existing_mention_type, mention_type),
                    _min_present(existing_tier, relation_tier),
                    _max_present(existing_weight, weight),
                    keep_id,
                    chunk_id,
                ),
            )
            stats["chunk_conflicts_merged"] += 1
        else:
            cursor.execute(
                "UPDATE kg_entity_chunks SET entity_id = ? WHERE entity_id = ? AND chunk_id = ?",
                (keep_id, merge_id, chunk_id),
            )
            stats["chunk_links_moved"] += 1
    cursor.execute("DELETE FROM kg_entity_chunks WHERE entity_id = ?", (merge_id,))

    # 2. Move relations, merging UNIQUE(source,target,type) conflicts into the canonical edge.
    relation_rows = list(
        cursor.execute(
            """
            SELECT id, source_id, target_id, relation_type, properties, confidence,
                   user_verified, fact, importance, valid_from, valid_until,
                   expired_at, source_chunk_id
            FROM kg_relations
            WHERE source_id = ? OR target_id = ?
            """,
            (merge_id, merge_id),
        )
    )
    for (
        relation_id,
        source_id,
        target_id,
        relation_type,
        properties,
        confidence,
        user_verified,
        fact,
        importance,
        valid_from,
        valid_until,
        expired_at,
        source_chunk_id,
    ) in relation_rows:
        new_source_id = keep_id if source_id == merge_id else source_id
        new_target_id = keep_id if target_id == merge_id else target_id
        existing = cursor.execute(
            """
            SELECT id, properties, confidence, user_verified, fact, importance,
                   valid_from, valid_until, expired_at, source_chunk_id
            FROM kg_relations
            WHERE source_id = ? AND target_id = ? AND relation_type = ? AND id != ?
            LIMIT 1
            """,
            (new_source_id, new_target_id, relation_type, relation_id),
        ).fetchone()
        if existing:
            (
                existing_id,
                existing_properties,
                existing_confidence,
                existing_user_verified,
                existing_fact,
                existing_importance,
                existing_valid_from,
                existing_valid_until,
                existing_expired_at,
                existing_source_chunk_id,
            ) = existing
            merged_properties = existing_properties
            if (not existing_properties or existing_properties == "{}") and properties:
                merged_properties = properties
            if existing_expired_at is None or expired_at is None:
                merged_expired_at = None
            else:
                merged_expired_at = max(existing_expired_at, expired_at)
            cursor.execute(
                """
                UPDATE kg_relations
                SET properties = ?,
                    confidence = ?,
                    user_verified = ?,
                    fact = ?,
                    importance = ?,
                    valid_from = ?,
                    valid_until = ?,
                    expired_at = ?,
                    source_chunk_id = ?
                WHERE id = ?
                """,
                (
                    merged_properties,
                    _max_present(existing_confidence, confidence),
                    max(existing_user_verified or 0, user_verified or 0),
                    existing_fact or fact,
                    _max_present(existing_importance, importance),
                    _earliest_present(existing_valid_from, valid_from),
                    _latest_valid_until(existing_valid_until, valid_until),
                    merged_expired_at,
                    existing_source_chunk_id or source_chunk_id,
                    existing_id,
                ),
            )
            cursor.execute("DELETE FROM kg_relations WHERE id = ?", (relation_id,))
            stats["relation_conflicts_merged"] += 1
        else:
            cursor.execute(
                "UPDATE kg_relations SET source_id = ?, target_id = ? WHERE id = ?",
                (new_source_id, new_target_id, relation_id),
            )
            stats["relations_moved"] += 1

    # 3. Move aliases from merged entity to kept entity.
    alias_rows = list(cursor.execute("SELECT alias FROM kg_entity_aliases WHERE entity_id = ?", (merge_id,)))
    cursor.execute(
        "UPDATE OR IGNORE kg_entity_aliases SET entity_id = ? WHERE entity_id = ?",
        (keep_id, merge_id),
    )
    cursor.execute("DELETE FROM kg_entity_aliases WHERE entity_id = ?", (merge_id,))
    stats["aliases_moved"] += len(alias_rows)

    # Store merged entity's name as alias.
    store.add_entity_alias(merged["name"], keep_id, alias_type="merged")

    # 4. Delete vector embedding for merged entity.
    cursor.execute("DELETE FROM kg_vec_entities WHERE entity_id = ?", (merge_id,))

    # 5. Delete the merged entity (FTS5 trigger handles cleanup).
    cursor.execute("DELETE FROM kg_entities WHERE id = ?", (merge_id,))
    stats["entities_deleted"] = 1
    return stats


def merge_entities(store: VectorStore, keep_id: str, merge_id: str) -> None:
    """Merge merge_id into keep_id. Preserves all links and relations.

    Steps:
    1. Move chunk links from merge_id to keep_id
    2. Move relations (both directions) from merge_id to keep_id
    3. Store merged entity's name as alias on keep_id
    4. Delete merge_id entity
    5. Clean up duplicate links/relations
    """
    merge_entities_preserving_links(store, keep_id, merge_id)
