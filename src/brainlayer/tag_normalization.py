"""Tag normalization and tombstoning utilities."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import apsw

_VALID_TAXONOMY_TAGS: frozenset[str] | None = None
_TAXONOMY_CONTENT_SHA: str | None = None
_TAXONOMY_GIT_SHA: str | None = None

TAG_MODE_ENV = "BRAINLAYER_ENRICHMENT_TAG_MODE"
TAG_MODE_HYBRID = "hybrid"
TAG_MODE_TAXONOMY = "taxonomy"

TAG_ALIASES: dict[str, str] = {
    "reactjs": "react",
    "react-js": "react",
    "react.js": "react",
    "nodejs": "node",
    "node-js": "node",
    "node.js": "node",
    "nextjs": "nextjs",
    "next-js": "nextjs",
    "next.js": "nextjs",
    "typescriptlang": "typescript",
    "ts": "typescript",
    "js": "javascript",
    "py": "python",
    "github-actions": "github-actions",
    "gh-actions": "github-actions",
    "code-rabbit": "coderabbit",
    "code rabbit": "coderabbit",
    "brain-layer": "brainlayer",
    "voice-layer": "voicelayer",
    "voicebar": "voicebar",
    "brainbar": "brainbar",
}


@dataclass(frozen=True)
class TagTombstoneResult:
    tombstoned: int
    updated_chunks: int


def valid_taxonomy_tags() -> frozenset[str]:
    global _VALID_TAXONOMY_TAGS
    if _VALID_TAXONOMY_TAGS is None:
        taxonomy_path = Path(__file__).resolve().parent / "taxonomy.json"
        data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        labels: set[str] = set()
        for category in data.get("categories", {}).values():
            if isinstance(category, dict):
                labels.update(str(label).strip().lower() for label in category.get("labels", {}) if str(label).strip())
        _VALID_TAXONOMY_TAGS = frozenset(labels)
    return _VALID_TAXONOMY_TAGS


def enrichment_tag_mode() -> str:
    """Return the active tag normalization mode.

    ``hybrid`` is the default because Phase-1 showed the coarse taxonomy-only
    whitelist regressed retrieval-useful tags. ``taxonomy`` remains available
    for A/B rollback via ``BRAINLAYER_ENRICHMENT_TAG_MODE=taxonomy``.
    """

    value = os.environ.get(TAG_MODE_ENV, TAG_MODE_HYBRID).strip().lower()
    if value in {"taxonomy", "whitelist", "taxonomy-only", "faceted"}:
        return TAG_MODE_TAXONOMY
    if value in {"hybrid", "freeform", "hybrid-taxonomy"}:
        return TAG_MODE_HYBRID
    return TAG_MODE_HYBRID


def taxonomy_content_sha() -> str:
    global _TAXONOMY_CONTENT_SHA
    if _TAXONOMY_CONTENT_SHA is None:
        taxonomy_path = Path(__file__).resolve().parent / "taxonomy.json"
        _TAXONOMY_CONTENT_SHA = sha256(taxonomy_path.read_bytes()).hexdigest()[:12]
    return _TAXONOMY_CONTENT_SHA


def taxonomy_git_sha() -> str:
    global _TAXONOMY_GIT_SHA
    if _TAXONOMY_GIT_SHA is not None:
        return _TAXONOMY_GIT_SHA
    override = os.environ.get("BRAINLAYER_ENRICHMENT_TAXONOMY_GIT_SHA", "").strip()
    if override:
        _TAXONOMY_GIT_SHA = override
        return _TAXONOMY_GIT_SHA
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        _TAXONOMY_GIT_SHA = result.stdout.strip() or "unknown"
    except Exception:
        _TAXONOMY_GIT_SHA = "unknown"
    return _TAXONOMY_GIT_SHA


def normalize_taxonomy_tags(tags: Any, *, limit: int = 10) -> list[str]:
    if not isinstance(tags, list):
        return []
    valid_tags = valid_taxonomy_tags()
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        value = _basic_normalize_tag(tag)
        if value not in valid_tags or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
        if len(normalized) >= limit:
            break
    return normalized


def normalize_hybrid_tags(tags: Any, *, limit: int = 10) -> list[str]:
    """Normalize curated taxonomy tags plus retrieval-useful free-form leaves."""

    if not isinstance(tags, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        value = normalize_hybrid_tag(tag, existing=normalized)
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
        if len(normalized) >= limit:
            break
    return normalized


def normalize_hybrid_tag(tag: str, *, existing: list[str] | None = None) -> str | None:
    value = _basic_normalize_tag(tag)
    if not value:
        return None
    if len(value) > 64:
        return None
    if value in valid_taxonomy_tags():
        return value

    alias = TAG_ALIASES.get(value) or TAG_ALIASES.get(_alias_key(value))
    if alias:
        value = alias

    # Keep faceted leaf specifics like tech/debug/resolution instead of dropping
    # them when the curated base lacks that exact leaf.
    if "/" in value:
        parts = [part for part in value.split("/") if part]
        if len(parts) < 2 or any(len(part) < 2 for part in parts):
            return None
        value = "/".join(parts)

    if not _looks_like_useful_leaf(value):
        return None

    for prior in existing or []:
        if _same_or_near_duplicate(value, prior):
            return prior
    return value


def normalize_enrichment_tag_values(tags: Any, *, limit: int = 10) -> list[str]:
    if enrichment_tag_mode() == TAG_MODE_TAXONOMY:
        return normalize_taxonomy_tags(tags, limit=limit)
    return normalize_hybrid_tags(tags, limit=limit)


def _basic_normalize_tag(tag: str) -> str:
    value = tag.strip().lower()
    value = value.replace("_", "-")
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    value = re.sub(r"/{2,}", "/", value)
    value = value.strip(" -./")
    return value


def _alias_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _looks_like_useful_leaf(value: str) -> bool:
    if not value or value in {"misc", "other", "general", "update", "status"}:
        return False
    if len(value) < 2:
        return False
    return bool(re.search(r"[a-z]", value))


def _same_or_near_duplicate(left: str, right: str) -> bool:
    if left == right:
        return True
    if _alias_key(left) == _alias_key(right):
        return True
    if min(len(left), len(right)) < 5:
        return False
    distance = _levenshtein(_alias_key(left), _alias_key(right))
    denominator = max(len(_alias_key(left)), len(_alias_key(right)), 1)
    return distance / denominator <= 0.16


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def ensure_tag_tombstone_schema(conn: apsw.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tag_tombstones (
            tag TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            occurrence_count INTEGER NOT NULL,
            tombstoned_at TEXT NOT NULL
        )
        """
    )


def _loads_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        decoded = raw
    else:
        try:
            decoded = json.loads(str(raw))
        except json.JSONDecodeError:
            return []
    if not isinstance(decoded, list):
        return []
    return [str(tag).strip() for tag in decoded if str(tag).strip()]


def _batches(values: list[str], size: int = 500) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def tombstone_singleton_tags(conn: apsw.Connection) -> TagTombstoneResult:
    """Tombstone non-taxonomy tags that occur once and remove them from chunks."""
    ensure_tag_tombstone_schema(conn)
    taxonomy_tags = valid_taxonomy_tags()
    rows = list(
        conn.execute(
            """
            SELECT tag, COUNT(DISTINCT chunk_id) AS occurrences
            FROM chunk_tags
            GROUP BY tag
            HAVING occurrences = 1
            """
        )
    )
    tombstones = [str(tag) for tag, occurrences in rows if str(tag).strip().lower() not in taxonomy_tags]
    if not tombstones:
        return TagTombstoneResult(tombstoned=0, updated_chunks=0)

    now = datetime.now(timezone.utc).isoformat()
    for tag in tombstones:
        conn.execute(
            """
            INSERT OR REPLACE INTO tag_tombstones(tag, reason, occurrence_count, tombstoned_at)
            VALUES (?, 'singleton-non-taxonomy', 1, ?)
            """,
            (tag, now),
        )

    tombstone_set = set(tombstones)
    chunk_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    for batch in _batches(tombstones):
        for row in conn.execute(
            f"""
            SELECT DISTINCT chunk_id
            FROM chunk_tags
            WHERE tag IN ({", ".join("?" for _ in batch)})
            """,
            batch,
        ):
            chunk_id = str(row[0])
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                chunk_ids.append(chunk_id)

    updated = 0
    for chunk_id in chunk_ids:
        row = conn.execute("SELECT tags FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            continue
        tags = _loads_tags(row[0])
        filtered = [tag for tag in tags if tag not in tombstone_set]
        if filtered == tags:
            continue
        conn.execute(
            "UPDATE chunks SET tags = ? WHERE id = ?",
            (json.dumps(filtered, ensure_ascii=True, separators=(",", ":")), chunk_id),
        )
        updated += 1

    return TagTombstoneResult(tombstoned=len(tombstones), updated_chunks=updated)
