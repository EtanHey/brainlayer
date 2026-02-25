"""Entity resolution — map extracted names to canonical KG entities.

Three-layer approach:
1. Exact match (name or alias, case-insensitive)
2. Fuzzy match (Jaro-Winkler on short names)
3. Create new entity if no match

Hebrew prefix stripping handles morphological prefixes (ב,ל,מ,ש,ה,ו,כ).
"""

import re
import unicodedata

from ..vector_store import VectorStore

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
) -> str:
    """Resolve an extracted name to an existing entity ID, or create a new one.

    Resolution order:
    1. Exact name match (case-insensitive, via get_entity_by_name)
    2. Alias match (case-insensitive)
    3. Hebrew prefix-stripped match
    4. Create new entity
    """
    # 1. Exact name match (UNIQUE(entity_type, name) — case-sensitive in DB)
    # Try original and lowered
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
        # Also try as a direct name
        existing = store.get_entity_by_name(entity_type, stripped)
        if existing:
            # Store the prefixed form as an alias for future lookups
            store.add_entity_alias(name, existing["id"], alias_type="hebrew_prefix")
            return existing["id"]

    # 4. No match — create new entity with auto-generated ID
    import uuid

    entity_id = f"{entity_type}-{uuid.uuid4().hex[:12]}"
    return store.upsert_entity(entity_id, entity_type, name, metadata={})


def merge_entities(store: VectorStore, keep_id: str, merge_id: str) -> None:
    """Merge merge_id into keep_id. Preserves all links and relations.

    Steps:
    1. Move chunk links from merge_id to keep_id
    2. Move relations (both directions) from merge_id to keep_id
    3. Store merged entity's name as alias on keep_id
    4. Delete merge_id entity
    5. Clean up duplicate links/relations
    """
    if keep_id == merge_id:
        return

    cursor = store.conn.cursor()

    # Get the merged entity's name before deleting it
    merged = store.get_entity(merge_id)
    if not merged:
        return

    # 1. Move chunk links
    cursor.execute(
        "UPDATE OR IGNORE kg_entity_chunks SET entity_id = ? WHERE entity_id = ?",
        (keep_id, merge_id),
    )
    # Delete any remaining (duplicates that couldn't be updated due to UNIQUE constraint)
    cursor.execute("DELETE FROM kg_entity_chunks WHERE entity_id = ?", (merge_id,))

    # 2. Move relations (source side)
    cursor.execute(
        "UPDATE OR IGNORE kg_relations SET source_id = ? WHERE source_id = ?",
        (keep_id, merge_id),
    )
    cursor.execute("DELETE FROM kg_relations WHERE source_id = ?", (merge_id,))

    # 2b. Move relations (target side)
    cursor.execute(
        "UPDATE OR IGNORE kg_relations SET target_id = ? WHERE target_id = ?",
        (keep_id, merge_id),
    )
    cursor.execute("DELETE FROM kg_relations WHERE target_id = ?", (merge_id,))

    # 3. Move aliases from merged entity to kept entity
    cursor.execute(
        "UPDATE OR IGNORE kg_entity_aliases SET entity_id = ? WHERE entity_id = ?",
        (keep_id, merge_id),
    )
    cursor.execute("DELETE FROM kg_entity_aliases WHERE entity_id = ?", (merge_id,))

    # Store merged entity's name as alias
    store.add_entity_alias(merged["name"], keep_id, alias_type="merged")

    # 4. Delete vector embedding for merged entity
    cursor.execute("DELETE FROM kg_vec_entities WHERE entity_id = ?", (merge_id,))

    # 5. Delete the merged entity (FTS5 trigger handles cleanup)
    cursor.execute("DELETE FROM kg_entities WHERE id = ?", (merge_id,))
