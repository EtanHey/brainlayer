"""Deterministic KG promotion from staged raw entity enrichment."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .vector_store import VectorStore


class _ReadOnlyPromotionStore:
    """Small read-only adapter for dry-runs against a live BrainLayer DB."""

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    def _read_cursor(self) -> sqlite3.Cursor:
        return self.conn.cursor()

    def close(self) -> None:
        self.conn.close()


_IDENTITY_TAG_RE = re.compile(r"^[a-z][a-z0-9-]+-[a-z][a-z0-9-]+-identification$")
_SLUG_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_KNOWN_GIVEN_NAME_ALIASES = {
    "michal": {"מיכל"},
}


@dataclass
class PromotionCandidate:
    canonical_name: str
    entity_type: str
    reason: str
    chunk_ids: set[str] = field(default_factory=set)
    surfaces: set[str] = field(default_factory=set)
    identity_tags: set[str] = field(default_factory=set)


def _slugify(value: str) -> str:
    return _SLUG_TOKEN_RE.sub("-", value.casefold()).strip("-")


def _canonical_from_identity_tag(tag: str) -> str:
    base = tag[: -len("-identification")]
    return " ".join(part.capitalize() for part in base.split("-") if part)


def _load_raw_entities(raw_json: Any) -> list[dict[str, Any]]:
    if not raw_json:
        return []
    if isinstance(raw_json, list):
        decoded = raw_json
    else:
        try:
            decoded = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(decoded, list):
        return []
    return [item for item in decoded if isinstance(item, dict)]


def _load_tags(tags_json: Any) -> list[str]:
    if not tags_json:
        return []
    if isinstance(tags_json, list):
        decoded = tags_json
    else:
        try:
            decoded = json.loads(tags_json)
        except (json.JSONDecodeError, TypeError):
            decoded = [tags_json]
    if not isinstance(decoded, list):
        return []
    return [str(tag).strip() for tag in decoded if str(tag).strip()]


def _raw_entity_name(entity: dict[str, Any]) -> str:
    name = entity.get("name") or entity.get("text")
    return str(name).strip() if name else ""


def _raw_entity_type(entity: dict[str, Any]) -> str:
    entity_type = entity.get("type") or entity.get("entity_type")
    return str(entity_type).strip().casefold() if entity_type else ""


def _valid_surface(surface: str) -> bool:
    if not surface:
        return False
    folded = surface.casefold()
    if "person_" in folded or "[person" in folded:
        return False
    return True


def _select_canonical_surface(surfaces: set[str]) -> str:
    latin = [surface for surface in surfaces if surface and surface.isascii() and " " in surface]
    if latin:
        return max(latin, key=lambda value: (len(value), value))
    spaced = [surface for surface in surfaces if " " in surface]
    if spaced:
        return max(spaced, key=lambda value: (len(value), value))
    return max(surfaces, key=lambda value: (len(value), value))


def _matching_rows(store: Any, limit: int | None = None) -> list[dict[str, Any]]:
    cursor = store._read_cursor()
    params: list[Any] = []
    limit_sql = ""
    if limit is not None:
        limit_sql = " LIMIT ?"
        params.append(int(limit))

    rows = list(
        cursor.execute(
            f"""
            SELECT c.id, c.raw_entities_json, c.tags
            FROM chunks c
            WHERE (
                (c.raw_entities_json IS NOT NULL AND c.raw_entities_json != '')
                OR c.id IN (
                    SELECT chunk_id FROM chunk_tags WHERE tag GLOB '*-identification'
                )
            )
              AND COALESCE(c.status, 'active') = 'active'
              AND c.superseded_by IS NULL
              AND c.aggregated_into IS NULL
              AND c.archived_at IS NULL
            ORDER BY c.created_at DESC, c.id
            {limit_sql}
            """,
            params,
        )
    )
    return [{"chunk_id": row[0], "raw_entities_json": row[1], "tags": row[2]} for row in rows]


def _collect_candidates(rows: list[dict[str, Any]], entity_type: str) -> list[PromotionCandidate]:
    by_identity_tag: dict[str, PromotionCandidate] = {}
    by_surface_key: dict[str, PromotionCandidate] = defaultdict(
        lambda: PromotionCandidate(canonical_name="", entity_type=entity_type, reason="stable_raw_entity")
    )
    raw_surface_entries: list[tuple[str, str]] = []

    for row in rows:
        chunk_id = row["chunk_id"]
        tags = _load_tags(row.get("tags"))
        identity_tags = [tag for tag in tags if _IDENTITY_TAG_RE.fullmatch(tag)]
        raw_entities = [
            entity
            for entity in _load_raw_entities(row.get("raw_entities_json"))
            if _raw_entity_type(entity) == entity_type
        ]

        surfaces = {surface for entity in raw_entities if _valid_surface(surface := _raw_entity_name(entity))}
        for surface in surfaces:
            raw_surface_entries.append((chunk_id, surface))

        for tag in identity_tags:
            candidate = by_identity_tag.setdefault(
                tag,
                PromotionCandidate(
                    canonical_name=_canonical_from_identity_tag(tag),
                    entity_type=entity_type,
                    reason="identity_tag",
                ),
            )
            candidate.chunk_ids.add(chunk_id)
            candidate.surfaces.update(surfaces)
            candidate.identity_tags.add(tag)

        if not raw_entities:
            continue

        for surface in surfaces:
            key = f"{entity_type}:{surface.casefold()}"
            candidate = by_surface_key[key]
            candidate.chunk_ids.add(chunk_id)
            candidate.surfaces.add(surface)

    candidates: list[PromotionCandidate] = []
    for candidate in by_identity_tag.values():
        for chunk_id, surface in raw_surface_entries:
            if _surface_matches_identity(candidate.canonical_name, surface):
                candidate.chunk_ids.add(chunk_id)
                candidate.surfaces.add(surface)
        if len(candidate.chunk_ids) >= 2:
            candidate.surfaces.add(candidate.canonical_name)
            candidates.append(candidate)

    for candidate in by_surface_key.values():
        if len(candidate.chunk_ids) >= 3 and candidate.surfaces:
            candidate.canonical_name = _select_canonical_surface(candidate.surfaces)
            candidates.append(candidate)

    return candidates


def _surface_matches_identity(canonical_name: str, surface: str) -> bool:
    canonical_parts = [part.casefold() for part in canonical_name.split() if part]
    if len(canonical_parts) < 2:
        return False
    surface_folded = surface.casefold()
    given = canonical_parts[0]
    family = canonical_parts[-1]
    if surface_folded in _KNOWN_GIVEN_NAME_ALIASES.get(given, set()):
        return True
    if given not in surface_folded:
        return False
    normalized_surface = _SLUG_TOKEN_RE.sub("", surface_folded).replace("h", "")
    normalized_family = _SLUG_TOKEN_RE.sub("", family).replace("h", "")
    return normalized_family in normalized_surface or len(surface.split()) == 1


def _promote_candidate(store: VectorStore, candidate: PromotionCandidate, *, dry_run: bool = False) -> dict[str, int]:
    stats = {"entities_promoted": 0, "aliases_created": 0, "chunks_linked": 0}
    if dry_run:
        stats["entities_promoted"] = 1
        stats["aliases_created"] = len(candidate.surfaces | candidate.identity_tags)
        stats["chunks_linked"] = len(candidate.chunk_ids)
        return stats

    entity_id = f"promoted-{candidate.entity_type}-{_slugify(candidate.canonical_name)}"
    existing = store.get_entity_by_name(candidate.entity_type, candidate.canonical_name)
    if existing:
        entity_id = existing["id"]
    else:
        entity_id = store.upsert_entity(
            entity_id,
            candidate.entity_type,
            candidate.canonical_name,
            metadata={"source": "raw_entities_json_promotion", "reason": candidate.reason},
            canonical_name=_slugify(candidate.canonical_name).replace("-", "_"),
            confidence=0.9,
            importance=0.7,
        )
        stats["entities_promoted"] = 1

    before_aliases = {(row["alias"].casefold(), row["alias_type"]) for row in store.get_entity_aliases(entity_id)}
    for surface in sorted(candidate.surfaces):
        if surface != candidate.canonical_name:
            store.add_entity_alias(surface, entity_id, alias_type="raw_surface")
    for tag in sorted(candidate.identity_tags):
        store.add_entity_alias(tag, entity_id, alias_type="identity_tag")
    after_aliases = {(row["alias"].casefold(), row["alias_type"]) for row in store.get_entity_aliases(entity_id)}
    stats["aliases_created"] = len(after_aliases - before_aliases)

    cursor = store.conn.cursor()
    for chunk_id in sorted(candidate.chunk_ids):
        before = cursor.execute(
            "SELECT 1 FROM kg_entity_chunks WHERE entity_id = ? AND chunk_id = ?",
            (entity_id, chunk_id),
        ).fetchone()
        store.link_entity_chunk(
            entity_id=entity_id,
            chunk_id=chunk_id,
            relevance=0.95 if candidate.reason == "identity_tag" else 0.8,
            context=f"raw_entities_json promotion: {candidate.reason}",
            mention_type="explicit" if candidate.reason == "identity_tag" else "inferred",
        )
        if before is None:
            stats["chunks_linked"] += 1

    return stats


def promote_raw_entity_identities(
    store: VectorStore,
    *,
    entity_type: str = "person",
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Promote high-signal staged raw entities into canonical KG identities."""
    rows = _matching_rows(store, limit=limit)
    candidates = _collect_candidates(rows, entity_type=entity_type)
    totals = {
        "chunks_scanned": len(rows),
        "candidates": len(candidates),
        "entities_promoted": 0,
        "aliases_created": 0,
        "chunks_linked": 0,
    }
    for candidate in candidates:
        stats = _promote_candidate(store, candidate, dry_run=dry_run)
        for key in ("entities_promoted", "aliases_created", "chunks_linked"):
            totals[key] += stats[key]
    return totals


def promote_chunk_raw_entities(
    store: VectorStore,
    chunk_id: str,
    *,
    entity_type: str = "person",
    dry_run: bool = False,
) -> dict[str, int]:
    """Promote identities relevant to one chunk after realtime enrichment."""
    cursor = store._read_cursor()
    row = cursor.execute("SELECT tags FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if row is None:
        return {
            "chunks_scanned": 0,
            "candidates": 0,
            "entities_promoted": 0,
            "aliases_created": 0,
            "chunks_linked": 0,
        }

    identity_tags = [tag for tag in _load_tags(row[0]) if _IDENTITY_TAG_RE.fullmatch(tag)]
    if not identity_tags:
        return promote_raw_entity_identities(store, entity_type=entity_type, limit=500, dry_run=dry_run)

    placeholders = ", ".join("?" for _ in identity_tags)
    rows = list(
        cursor.execute(
            f"""
            SELECT c.id, c.raw_entities_json, c.tags
            FROM chunks c
            JOIN chunk_tags ct ON c.id = ct.chunk_id
            WHERE ct.tag IN ({placeholders})
            """,
            identity_tags,
        )
    )
    candidates = _collect_candidates(
        [{"chunk_id": item[0], "raw_entities_json": item[1], "tags": item[2]} for item in rows],
        entity_type=entity_type,
    )
    totals = {
        "chunks_scanned": len(rows),
        "candidates": len(candidates),
        "entities_promoted": 0,
        "aliases_created": 0,
        "chunks_linked": 0,
    }
    for candidate in candidates:
        stats = _promote_candidate(store, candidate, dry_run=dry_run)
        for key in ("entities_promoted", "aliases_created", "chunks_linked"):
            totals[key] += stats[key]
    return totals


def main() -> None:
    from .paths import get_db_path

    parser = argparse.ArgumentParser(description="Backfill KG promotion from chunks.raw_entities_json")
    parser.add_argument("--db-path", type=Path, default=get_db_path())
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview promotions without writing. This is the default."
    )
    parser.add_argument("--apply", action="store_true", help="Write promotions. Default is dry-run.")
    args = parser.parse_args()

    dry_run = not args.apply
    store = _ReadOnlyPromotionStore(args.db_path) if dry_run else VectorStore(args.db_path)
    try:
        stats = promote_raw_entity_identities(store, limit=args.limit, dry_run=dry_run)
        stats["mode"] = "apply" if args.apply else "dry-run"
        stats["timestamp"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
