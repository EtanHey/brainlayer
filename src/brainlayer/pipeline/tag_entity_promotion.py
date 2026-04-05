"""Promote high-frequency concept tags into KG entities."""

from __future__ import annotations

import re
from typing import Any

from ..vector_store import VectorStore

ACTIVITY_TAG_PREFIXES = ("act:", "dom:", "meta/")
ACTIVITY_TAGS = {
    "debugging",
    "testing",
    "refactoring",
    "code-review",
    "bug-fix",
    "feature-dev",
    "configuration",
    "documentation",
    "project-management",
    "error-handling",
    "task-management",
    "deployment",
    "workflow",
    "automation",
    "verification",
    "command-line",
    "planning",
    "tooling",
    "file-system",
    "file-management",
    "scripting",
    "monitoring",
    "assistant-action",
    "status-update",
    "version-control",
    "implementation",
    "collaboration",
    "discussion",
    "code-analysis",
    "metadata",
    "architecture",
    "styling",
    "confirmation",
    "troubleshooting",
    "design",
    "frontend",
    "backend",
    "command",
    "shell",
    "bash",
    "grep",
    "json",
    "regex",
    "html",
    "css",
    "svg",
}

PERSON_TAGS = {
    "andrew-huberman",
    "avi-simon",
    "daniel-munk",
    "dor-zohar",
    "etan-heyman",
    "joshua-anderson",
    "maor-noah",
    "shachar-gerby",
    "theo-browne",
    "yuval-nir",
}

TECHNOLOGY_TAGS = {
    "1password",
    "convex",
    "docker",
    "javascript",
    "nextjs",
    "openai",
    "postgres",
    "python",
    "railway",
    "react",
    "sqlite",
    "supabase",
    "telegram",
    "typescript",
    "whatsapp",
}

TOPIC_TAGS = {
    "cold-exposure",
    "dopamine",
    "exercise",
    "fitness",
    "metabolism",
    "neuroscience",
    "nutrition",
    "psychology",
    "supplements",
    "wellness",
}

HEBREW_MARKERS = {"hebrew", "rtl", "right-to-left"}
COMMUNITY_MARKERS = {"community", "collective", "crew", "forum", "group", "guild", "network"}


def _slugify_tag(tag: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")
    return re.sub(r"-{2,}", "-", normalized)


def classify_tag_entity_type(tag: str) -> str:
    """Infer an entity type from a promoted tag."""
    normalized = _slugify_tag(tag)

    if normalized in PERSON_TAGS:
        return "person"
    if normalized in TECHNOLOGY_TAGS:
        return "technology"
    if any(marker in normalized for marker in COMMUNITY_MARKERS):
        return "community"
    if normalized in TOPIC_TAGS or any(marker in normalized for marker in HEBREW_MARKERS):
        return "topic"
    return "topic"


def find_promotion_candidates(
    store: VectorStore, min_count: int = 500, limit: int | None = None
) -> list[dict[str, Any]]:
    """Find high-frequency concept tags worth promoting to entities."""
    cursor = store._read_cursor()
    placeholders = ", ".join("?" for _ in ACTIVITY_TAGS)
    query = f"""
        SELECT ct.tag, COUNT(*) as cnt
        FROM chunk_tags ct
        LEFT JOIN kg_entities e ON lower(e.name) = lower(ct.tag)
        WHERE e.id IS NULL
          AND ct.tag IS NOT NULL
          AND ct.tag != ''
          AND ct.tag NOT LIKE 'act:%'
          AND ct.tag NOT LIKE 'dom:%'
          AND ct.tag NOT LIKE 'meta/%'
          AND lower(ct.tag) NOT IN ({placeholders})
        GROUP BY ct.tag
        HAVING COUNT(*) >= ?
        ORDER BY cnt DESC, ct.tag ASC
    """
    params: list[Any] = [tag.lower() for tag in sorted(ACTIVITY_TAGS)]
    params.append(min_count)
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    rows = list(cursor.execute(query, params))
    return [
        {
            "tag": row[0],
            "count": row[1],
            "entity_type": classify_tag_entity_type(row[0]),
        }
        for row in rows
    ]


def promote_tag_entities(
    store: VectorStore,
    min_count: int = 500,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Promote high-frequency tags into KG entities and link matching chunks."""
    candidates = find_promotion_candidates(store, min_count=min_count, limit=limit)
    stats = {
        "candidates": len(candidates),
        "entities_created": 0,
        "links_created": 0,
        "promoted_tags": [candidate["tag"] for candidate in candidates],
    }
    if dry_run:
        return stats

    cursor = store.conn.cursor()
    kg_entity_chunk_cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entity_chunks)")}
    has_mention_type = "mention_type" in kg_entity_chunk_cols

    for candidate in candidates:
        tag = candidate["tag"]
        entity_type = candidate["entity_type"]
        entity_id = f"auto-tag-{_slugify_tag(tag)}"
        existing = store.get_entity_by_name(entity_type, tag)
        if existing is None:
            store.upsert_entity(
                entity_id,
                entity_type,
                tag,
                metadata={"source": "tag-promotion", "tag_count": candidate["count"]},
                confidence=0.8,
                importance=0.6,
            )
            stats["entities_created"] += 1
        else:
            entity_id = existing["id"]

        if has_mention_type:
            cursor.execute(
                """
                INSERT OR IGNORE INTO kg_entity_chunks (entity_id, chunk_id, relevance, context, mention_type)
                SELECT ?, ct.chunk_id, 0.8, 'tag-promotion', 'tag'
                FROM chunk_tags ct
                WHERE ct.tag = ?
                """,
                (entity_id, tag),
            )
        else:
            cursor.execute(
                """
                INSERT OR IGNORE INTO kg_entity_chunks (entity_id, chunk_id, relevance, context)
                SELECT ?, ct.chunk_id, 0.8, 'tag-promotion'
                FROM chunk_tags ct
                WHERE ct.tag = ?
                """,
                (entity_id, tag),
            )
        stats["links_created"] += store.conn.changes()

    return stats
