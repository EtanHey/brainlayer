#!/usr/bin/env python3
"""Apply approved P2 KG safe cleanup buckets.

Approved scope:
- Named fixes: brainClaude -> brainlayerClaude, voiceClaude -> voicelayerClaude,
  GLiNER tool -> GLiNER technology, clod no-op.
- Soft archive low-support junk labels.
- Merge duplicate PERSON_xxxxxxxx families into the person canonical.

This intentionally does not touch cross-type ontology families such as
BrainLayer/VoiceLayer/Golems project-vs-tool-vs-technology rows.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.paths import get_db_path
from brainlayer.pipeline.entity_resolution import merge_entities_preserving_links
from brainlayer.vector_store import VectorStore

BRAIN_TARGET_ID = "173733c921d4e817"
VOICE_TARGET_ID = "golem-224489db81b9"
GLINER_TARGET_ID = "5cc471c76ffdab6d"
GLINER_TOOL_ID = "fb6b880f-de32-5a7c-b1ba-a8657c68af98"

EXPECTED = {
    "brain_sources": 10,
    "voice_sources": 9,
    "gliner_sources": 1,
    "person_groups": 585,
    "person_sources": 887,
    "junk": 491,
}

VOICE_VARIANT_KEYS = {
    "voiceclaude",
    "voicelayerclaude",
    "voicelayerclaudeworkeremail1",
}


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def person_token_key(value: str) -> str | None:
    match = re.fullmatch(r"\[?(PERSON_[0-9a-f]{8})\]?", value, re.I)
    return match.group(1).lower() if match else None


def fetch_entities(store: VectorStore) -> list[dict[str, Any]]:
    cursor = store._read_cursor()
    columns = ["id", "name", "entity_type", "status", "metadata", "rels", "chunks", "canonical_name"]
    return [
        dict(zip(columns, row))
        for row in cursor.execute(
            """
            WITH rc AS (
              SELECT source_id AS entity_id, count(*) AS c FROM kg_relations GROUP BY source_id
              UNION ALL
              SELECT target_id AS entity_id, count(*) AS c FROM kg_relations GROUP BY target_id
            ), rs AS (SELECT entity_id, sum(c) AS rels FROM rc GROUP BY entity_id),
            cc AS (SELECT entity_id, count(*) AS chunks FROM kg_entity_chunks GROUP BY entity_id)
            SELECT e.id, e.name, e.entity_type, e.status, e.metadata,
                   coalesce(rs.rels, 0) AS rels,
                   coalesce(cc.chunks, 0) AS chunks,
                   coalesce(e.canonical_name, '') AS canonical_name
            FROM kg_entities e
            LEFT JOIN rs ON rs.entity_id = e.id
            LEFT JOIN cc ON cc.entity_id = e.id
            ORDER BY lower(e.name), e.entity_type, e.id
            """
        )
    ]


def is_local_path_or_file(name: str) -> bool:
    lower = name.lower()
    if (
        name.startswith("/Users/")
        or name.startswith("/private/")
        or name.startswith("/tmp/")
        or name.startswith("./")
        or name.startswith("~/")
    ):
        return True
    # Deliberately excludes .js because names like Node.js/Next.js are real entities.
    return "/" not in name and re.search(r"\.(md|json|py|ts|tsx|swift|toml|ya?ml|sh|db|sqlite|mov)$", lower) is not None


def junk_reasons(row: dict[str, Any]) -> list[str]:
    name = str(row["name"])
    lower = name.lower()
    reasons: list[str] = []
    if re.fullmatch(r"[-_/\\.:@#~]+", name) or len(name.strip()) <= 1:
        reasons.append("fragment")
    if "http://" in lower or "https://" in lower:
        reasons.append("url-label")
    if is_local_path_or_file(name):
        reasons.append("path-or-file-label")
    if re.fullmatch(r"[0-9a-f]{8,}", lower):
        reasons.append("hex-id-fragment")
    if "\x08" in name or len(name) > 120:
        reasons.append("control/long-fragment")
    return reasons


def build_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["id"]: row for row in rows}

    brain_named = [row for row in rows if normalize_name(row["name"]) in {"brainclaude", "brainlayerclaude"}]
    voice_named = [row for row in rows if normalize_name(row["name"]) in VOICE_VARIANT_KEYS]
    gliner_sources = [by_id[GLINER_TOOL_ID]] if GLINER_TOOL_ID in by_id else []

    person_groups: list[dict[str, Any]] = []
    person_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = person_token_key(row["name"])
        if key:
            person_by_key[key].append(row)
    for key, family in person_by_key.items():
        persons = [row for row in family if row["entity_type"] == "person"]
        if not persons or len(family) <= 1:
            continue
        target = max(persons, key=lambda row: (row["chunks"], row["rels"], row["name"]))
        # Approved P2 scope intentionally absorbs all PERSON_token placeholder rows,
        # including rows that KG extraction mistyped as project/tool/etc. These are not
        # real cross-type ontology entities.
        sources = [row for row in family if row["id"] != target["id"]]
        person_groups.append({"key": key, "target": target, "sources": sources})
    person_groups.sort(key=lambda group: (-sum(row["chunks"] for row in group["sources"]), group["key"]))

    junk = []
    for row in rows:
        reasons = junk_reasons(row)
        if reasons and row["rels"] == 0 and row["chunks"] <= 10:
            row = dict(row)
            row["reason"] = ",".join(reasons)
            junk.append(row)
    junk.sort(key=lambda row: (row["reason"], -row["chunks"], row["name"].lower(), row["id"]))

    return {
        "brain_target": by_id.get(BRAIN_TARGET_ID),
        "brain_sources": [row for row in brain_named if row["id"] != BRAIN_TARGET_ID],
        "voice_target": by_id.get(VOICE_TARGET_ID),
        "voice_sources": [row for row in voice_named if row["id"] != VOICE_TARGET_ID],
        "gliner_target": by_id.get(GLINER_TARGET_ID),
        "gliner_sources": gliner_sources,
        "person_groups": person_groups,
        "junk": junk,
    }


def plan_counts(plan: dict[str, Any]) -> dict[str, int]:
    return {
        "brain_sources": len(plan["brain_sources"]),
        "voice_sources": len(plan["voice_sources"]),
        "gliner_sources": len(plan["gliner_sources"]),
        "person_groups": len(plan["person_groups"]),
        "person_sources": sum(len(group["sources"]) for group in plan["person_groups"]),
        "junk": len(plan["junk"]),
    }


def validate_plan(plan: dict[str, Any], *, allow_count_drift: bool = False) -> None:
    if not plan["brain_target"]:
        raise RuntimeError(f"brain target missing: {BRAIN_TARGET_ID}")
    if not plan["voice_target"]:
        raise RuntimeError(f"voice target missing: {VOICE_TARGET_ID}")
    if not plan["gliner_target"]:
        raise RuntimeError(f"GLiNER target missing: {GLINER_TARGET_ID}")
    counts = plan_counts(plan)
    mismatches = {key: (counts[key], expected) for key, expected in EXPECTED.items() if counts[key] != expected}
    if mismatches and not allow_count_drift:
        raise RuntimeError(f"candidate counts drifted: {mismatches}")


def set_entity_name(store: VectorStore, entity_id: str, name: str, canonical_name: str) -> None:
    store.conn.cursor().execute(
        """
        UPDATE kg_entities
        SET name = ?,
            canonical_name = ?,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
        WHERE id = ?
        """,
        (name, canonical_name, entity_id),
    )


def archive_entity(store: VectorStore, entity_id: str, reason: str, archived_at: str) -> bool:
    row = store.conn.cursor().execute("SELECT metadata FROM kg_entities WHERE id = ?", (entity_id,)).fetchone()
    if not row:
        return False
    try:
        metadata = json.loads(row[0]) if row[0] else {}
    except json.JSONDecodeError:
        metadata = {"_previous_metadata_raw": row[0]}
    metadata["p2_cleanup"] = {
        "action": "archive-junk",
        "reason": reason,
        "archived_at": archived_at,
    }
    cursor = store.conn.cursor()
    cursor.execute(
        """
        UPDATE kg_entities
        SET status = 'archived',
            expired_at = ?,
            metadata = ?,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
        WHERE id = ? AND status = 'active'
        """,
        (archived_at, json.dumps(metadata, sort_keys=True), entity_id),
    )
    changes = getattr(store.conn, "changes", None)
    if callable(changes):
        return changes() > 0
    return getattr(cursor, "rowcount", 0) > 0


def merge_many(store: VectorStore, target_id: str, sources: list[dict[str, Any]], stats: Counter[str]) -> None:
    for source in sources:
        if not store.get_entity(source["id"]):
            continue
        merge_stats = merge_entities_preserving_links(store, target_id, source["id"])
        stats.update({f"merge_{key}": value for key, value in merge_stats.items()})


def apply_plan(store: VectorStore, plan: dict[str, Any]) -> Counter[str]:
    stats: Counter[str] = Counter()
    archived_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    set_entity_name(store, BRAIN_TARGET_ID, "brainlayerClaude", "brainlayerclaude")
    store.add_entity_alias("brainClaude", BRAIN_TARGET_ID, alias_type="merged")
    store.add_entity_alias("BrainLayer Claude", BRAIN_TARGET_ID, alias_type="merged")
    store.add_entity_alias("brainlayer Claude", BRAIN_TARGET_ID, alias_type="merged")
    merge_many(store, BRAIN_TARGET_ID, plan["brain_sources"], stats)
    stats["named_brain_sources"] = len(plan["brain_sources"])

    set_entity_name(store, VOICE_TARGET_ID, "voicelayerClaude", "voicelayerclaude")
    store.add_entity_alias("voiceClaude", VOICE_TARGET_ID, alias_type="merged")
    merge_many(store, VOICE_TARGET_ID, plan["voice_sources"], stats)
    stats["named_voice_sources"] = len(plan["voice_sources"])

    merge_many(store, GLINER_TARGET_ID, plan["gliner_sources"], stats)
    stats["named_gliner_sources"] = len(plan["gliner_sources"])

    for group in plan["person_groups"]:
        merge_many(store, group["target"]["id"], group["sources"], stats)
    stats["person_groups"] = len(plan["person_groups"])
    stats["person_sources"] = sum(len(group["sources"]) for group in plan["person_groups"])

    archived = 0
    for row in plan["junk"]:
        if archive_entity(store, row["id"], row["reason"], archived_at):
            archived += 1
    stats["junk_archived"] = archived
    return stats


def orphan_counts(store: VectorStore) -> dict[str, int]:
    cursor = store._read_cursor()
    return {
        "entity_chunk_orphans": cursor.execute(
            """
            SELECT count(*)
            FROM kg_entity_chunks ec
            LEFT JOIN kg_entities e ON e.id = ec.entity_id
            WHERE e.id IS NULL
            """
        ).fetchone()[0],
        "relation_source_orphans": cursor.execute(
            """
            SELECT count(*)
            FROM kg_relations r
            LEFT JOIN kg_entities e ON e.id = r.source_id
            WHERE e.id IS NULL
            """
        ).fetchone()[0],
        "relation_target_orphans": cursor.execute(
            """
            SELECT count(*)
            FROM kg_relations r
            LEFT JOIN kg_entities e ON e.id = r.target_id
            WHERE e.id IS NULL
            """
        ).fetchone()[0],
    }


def print_plan(plan: dict[str, Any]) -> None:
    counts = plan_counts(plan)
    print("P2 safe cleanup plan")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print("  clod: no-op (0 entity/alias rows in audit)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply approved safe cleanup buckets")
    parser.add_argument("--allow-count-drift", action="store_true", help="Do not abort if audited counts changed")
    args = parser.parse_args()

    store = VectorStore(get_db_path(), readonly=not args.apply)
    try:
        plan = build_plan(fetch_entities(store))
        validate_plan(plan, allow_count_drift=args.allow_count_drift)
        print_plan(plan)

        if not args.apply:
            print("DRY-RUN: no DB writes performed")
            return 0

        cursor = store.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            stats = apply_plan(store, plan)
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise

        print("APPLIED")
        for key in sorted(stats):
            print(f"  {key}: {stats[key]}")
        print("Orphan counts:")
        for key, value in orphan_counts(store).items():
            print(f"  {key}: {value}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
