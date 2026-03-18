"""Phase 1: Importance calibration — heuristic SQL fix.

Deflates importance inflation via content-type-aware caps.
41.1% of chunks are importance >= 7 (target: <20%).
96.4% of high-importance chunks are raw transcript.

Research: ~/Gits/orchestrator/docs.local/research/research-24-importance-calibration-results.md
"""

import logging
from typing import Any, Dict

from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Content types considered "conversation" — verbose, contextual, rarely actionable alone
CONVERSATION_TYPES = (
    "assistant_text",
    "user_message",
    "file_read",
    "git_diff",
    "build_log",
    "dir_listing",
    "config",
    "noise",
)

# Content types from brain_store (manually curated, high trust)
BRAIN_STORE_TYPES = (
    "idea",
    "mistake",
    "decision",
    "learning",
    "todo",
    "bookmark",
    "note",
    "journal",
    "issue",
)

# Curated knowledge (high value from pipeline classification)
CURATED_TYPES = ("learning", "skill", "project_config", "research")

# Media sources
MEDIA_SOURCES = ("youtube", "podcast")

# Keywords that indicate operational importance — exempt from capping
DECISION_KEYWORDS = (
    "decided",
    "decision",
    "learned",
    "learning",
    "mistake",
    "critical",
    "never",
    "always",
    "must not",
    "architecture",
)

# NULL importance defaults by content type / source
NULL_DEFAULTS = {
    # By source (checked first)
    "_source_manual": 8,
    "_source_youtube": 4,
    "_source_podcast": 4,
    # By content_type
    "ai_code": 5,
    "stack_trace": 5,
    # Curated types
    "learning": 7,
    "skill": 7,
    "project_config": 7,
    "research": 7,
    # brain_store types
    "decision": 7,
    "mistake": 7,
    "idea": 5,
    "todo": 5,
    "bookmark": 4,
    "note": 5,
    "journal": 4,
    "issue": 6,
    # Conversation types
    "assistant_text": 3,
    "user_message": 3,
    "file_read": 2,
    "git_diff": 2,
    "build_log": 2,
    "dir_listing": 2,
    "config": 3,
    "noise": 1,
}


def _has_decision_keywords(content: str) -> bool:
    """Check if content contains keywords indicating operational importance."""
    lower = content.lower()
    return any(kw in lower for kw in DECISION_KEYWORDS)


def _null_default(content_type: str | None, source: str | None) -> int:
    """Determine default importance for a NULL-importance chunk."""
    # Source takes priority
    if source == "manual":
        return NULL_DEFAULTS["_source_manual"]
    if source in MEDIA_SOURCES:
        return NULL_DEFAULTS.get(f"_source_{source}", 4)
    # Then content_type
    if content_type and content_type in NULL_DEFAULTS:
        return NULL_DEFAULTS[content_type]
    return 3  # fallback


def get_distribution(cursor) -> Dict[str, Any]:
    """Get importance distribution as bucket counts."""
    rows = cursor.execute("""
        SELECT
            CASE
                WHEN importance IS NULL THEN 'NULL'
                WHEN importance <= 2 THEN '1-2'
                WHEN importance <= 4 THEN '3-4'
                WHEN importance <= 6 THEN '5-6'
                WHEN importance <= 8 THEN '7-8'
                ELSE '9-10'
            END as bucket,
            COUNT(*) as cnt
        FROM chunks
        GROUP BY bucket
        ORDER BY bucket
    """).fetchall()
    dist = dict(rows)
    total = sum(dist.values())
    high = dist.get("7-8", 0) + dist.get("9-10", 0)
    return {
        "buckets": dist,
        "total": total,
        "high_count": high,
        "high_pct": round(high / total * 100, 1) if total > 0 else 0,
    }


def calibrate_importance(store: VectorStore, dry_run: bool = False) -> Dict[str, Any]:
    """Apply heuristic importance calibration.

    Rules (applied in priority order):
    1. NULL importance → default by content_type/source
    2. source='manual' (brain_store) → floor at 8
    3. source in media → cap at 5
    4. Conversation types with importance >= 7:
       - If has decision keywords → exempt (no change)
       - Else → min(importance - 2, 5)
    5. General deflation for remaining:
       - importance >= 8 → importance - 2
       - importance >= 6 → importance - 1

    Args:
        store: VectorStore instance
        dry_run: If True, report distribution changes without modifying DB

    Returns:
        Dict with before/after distribution stats.
    """
    cursor = store.conn.cursor()
    before = get_distribution(cursor)

    if dry_run:
        after = _simulate(cursor)
        return {"before": before, "after_simulated": after}

    # --- Step 1: Default NULL importance ---
    # Get all NULL-importance chunks and assign defaults
    null_chunks = cursor.execute("SELECT id, content_type, source FROM chunks WHERE importance IS NULL").fetchall()
    for chunk_id, content_type, source in null_chunks:
        default = _null_default(content_type, source)
        cursor.execute(
            "UPDATE chunks SET importance = ? WHERE id = ?",
            (default, chunk_id),
        )
    logger.info("Defaulted %d NULL-importance chunks", len(null_chunks))

    # --- Step 2: Floor brain_store entries at 8 ---
    cursor.execute("UPDATE chunks SET importance = 8 WHERE source = 'manual' AND importance < 8")

    # --- Step 3: Cap media transcripts at 5 ---
    placeholders = ",".join("?" for _ in MEDIA_SOURCES)
    cursor.execute(
        f"UPDATE chunks SET importance = 5 WHERE source IN ({placeholders}) AND importance > 5",
        MEDIA_SOURCES,
    )

    # --- Step 4: Cap conversation types at 5 (unless decision keywords) ---
    conv_placeholders = ",".join("?" for _ in CONVERSATION_TYPES)
    inflated_conv = cursor.execute(
        f"SELECT id, content, importance FROM chunks WHERE content_type IN ({conv_placeholders}) AND importance >= 7",
        CONVERSATION_TYPES,
    ).fetchall()
    for chunk_id, content, imp in inflated_conv:
        if _has_decision_keywords(content):
            continue  # exempt
        new_imp = min(imp - 2, 5)
        cursor.execute(
            "UPDATE chunks SET importance = ? WHERE id = ?",
            (new_imp, chunk_id),
        )
    logger.info("Capped %d conversation chunks (exempted decision-keyword chunks)", len(inflated_conv))

    # --- Step 5: General deflation for remaining high scores ---
    # Exclude already-handled: manual source, media sources, conversation types,
    # curated types, and brain_store types (they have floors, not caps)
    protected_types = CONVERSATION_TYPES + CURATED_TYPES + BRAIN_STORE_TYPES
    prot_placeholders = ",".join("?" for _ in protected_types)
    cursor.execute(
        f"UPDATE chunks SET importance = CASE "
        f"  WHEN importance >= 8 THEN importance - 2 "
        f"  WHEN importance >= 6 THEN importance - 1 "
        f"  ELSE importance END "
        f"WHERE importance >= 6 "
        f"AND (source IS NULL OR (source != 'manual' "
        f"  AND source NOT IN ({placeholders}))) "
        f"AND (content_type IS NULL OR content_type NOT IN ({prot_placeholders}))",
        (*MEDIA_SOURCES, *protected_types),
    )

    after = get_distribution(cursor)
    return {"before": before, "after": after}


def _simulate(cursor) -> Dict[str, Any]:
    """Simulate calibration without modifying data.

    Reads all chunks and computes what the distribution WOULD be.
    """
    rows = cursor.execute("SELECT id, content, content_type, source, importance FROM chunks").fetchall()

    new_importances = []
    for chunk_id, content, content_type, source, importance in rows:
        new_imp = _compute_new_importance(content, content_type, source, importance)
        new_importances.append(new_imp)

    # Build distribution from simulated values
    buckets: Dict[str, int] = {}
    for imp in new_importances:
        if imp is None:
            bucket = "NULL"
        elif imp <= 2:
            bucket = "1-2"
        elif imp <= 4:
            bucket = "3-4"
        elif imp <= 6:
            bucket = "5-6"
        elif imp <= 8:
            bucket = "7-8"
        else:
            bucket = "9-10"
        buckets[bucket] = buckets.get(bucket, 0) + 1

    total = sum(buckets.values())
    high = buckets.get("7-8", 0) + buckets.get("9-10", 0)
    return {
        "buckets": buckets,
        "total": total,
        "high_count": high,
        "high_pct": round(high / total * 100, 1) if total > 0 else 0,
    }


def _compute_new_importance(
    content: str,
    content_type: str | None,
    source: str | None,
    importance: float | None,
) -> float:
    """Compute what a chunk's importance WOULD be after calibration."""
    # Step 1: NULL defaults
    if importance is None:
        return _null_default(content_type, source)

    imp = importance

    # Step 2: brain_store floor
    if source == "manual":
        return max(imp, 8)

    # Step 3: media cap
    if source in MEDIA_SOURCES and imp > 5:
        return 5

    # Step 4: conversation cap (with decision keyword exemption)
    if content_type in CONVERSATION_TYPES and imp >= 7:
        if _has_decision_keywords(content or ""):
            return imp  # exempt
        return min(imp - 2, 5)

    # Step 5: general deflation (exclude curated + brain_store types)
    if content_type in CURATED_TYPES or content_type in BRAIN_STORE_TYPES:
        return imp
    if imp >= 8:
        return imp - 2
    if imp >= 6:
        return imp - 1

    return imp
