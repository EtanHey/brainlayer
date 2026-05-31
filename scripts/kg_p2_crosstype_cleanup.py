#!/usr/bin/env python3
"""Apply the approved P2 cross-type ontology cleanup.

This is intentionally scoped to the Etan-approved cross-type bucket:
- Merge repo/tool/technology/concept fragments into pinned project hubs.
- Promote pinned Codex tool rows to agent rows, then merge fragments.
- Keep BrainLayer MCP tools as child tool entities.
- Keep Domica company and repo separate, with the repo parented to company.
- Reroute and delete launcher-only domicaClaude rows.

Dry-run is the default. Use --apply only after commander gate approval.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.paths import get_db_path
from brainlayer.pipeline.entity_resolution import merge_entities_preserving_links
from brainlayer.vector_store import VectorStore

BRAINLAYER_PROJECT = "62c5d7d77039273c"
VOICELAYER_PROJECT = "28284387ab6c9d06"
GOLEMS_PROJECT = "d3ecc33fefa4e77c"
DOMICA_COMPANY = "company-1ebe7e404092"
DOMICA_REPO = "caa1ccfcb6f24932"

PROMOTION_IDS = {
    "0f29b289-c55b-5eb4-ac39-876576dcf710",
    "da0c00a1-52cf-5053-8958-be08d849f22e",
    "6f6aa18b-d464-50af-8e8b-2a4f216efd95",
}

MERGE_GROUPS: list[dict[str, Any]] = [
    {
        "label": "BrainLayer project",
        "target": BRAINLAYER_PROJECT,
        "sources": [
            "64933700-1fce-5859-8368-03f2328a97b1",
            "2ded5812-ccf0-5320-b63e-c1ed488a6431",
            "f21fc53a-3100-5105-96b7-1a8b6b090a5e",
            "c789efde-d060-54c7-95d0-4b8f7985861f",
            "e8bc81d8-0a57-5b82-a71a-d51124532239",
        ],
    },
    {
        "label": "VoiceLayer project",
        "target": VOICELAYER_PROJECT,
        "sources": [
            "e1f091ec-f874-5ec9-bdd9-cfb096619e6c",
            "2d1472e6-37c7-59ee-85dc-b51d111d0107",
            "e0d70d13-596e-54d1-bdd1-21ac23ae783e",
            "7113b6e6-7fae-56cf-8239-181bf2b98e5e",
            "e5a9e2ed-0442-5e61-93c5-c1519ac5d5fd",
        ],
    },
    {
        "label": "Golems project",
        "target": GOLEMS_PROJECT,
        "sources": [
            "6980d016-cde6-5d62-ae43-8f69fbd115bd",
            "351c5e07-8242-57a4-83d9-ee1dbe607eca",
            "dceb66e6-a877-515c-ad25-1be3afc0d04e",
            "76e8a3b5-b4f6-538c-8ce9-2a553a401ae3",
        ],
    },
    {
        "label": "golemsClaude agent",
        "target": "4c406b6a63221673",
        "sources": [
            "15a3d139-d3a7-5a83-a6ff-aecc0dab71a0",
            "78816112-6108-5cc0-8b05-f5cef51bf215",
            "93024a9e-afe8-5ab7-9a4a-3c2a42efe926",
            "person-b81b47445f58",
            "47dba758-01db-5eb1-8170-e19e466e1e89",
            "ac45b600-a2eb-5ecc-93f8-529822697fbc",
            "4ca4e86a-d4c9-5bae-970f-d0b51177ddea",
        ],
    },
    {
        "label": "brainlayerCodex agent",
        "target": "0f29b289-c55b-5eb4-ac39-876576dcf710",
        "sources": [
            "6c7b76f7-a11d-5748-8d66-c6f15499f3cf",
            "a0d23a80-a205-5e06-94c4-cd07748aa6a7",
            "9f35caf1-3747-54d2-9146-4ae6fdad83d4",
            "29ad4499-9ffa-5e9e-a1a6-82fc02f51925",
            "a803ebf9-fd5b-515d-9220-4d5aa995f8c3",
        ],
    },
    {
        "label": "voicelayerCodex agent",
        "target": "da0c00a1-52cf-5053-8958-be08d849f22e",
        "sources": [
            "d6b7b511-de1c-5038-9da6-a6c81c0e8b53",
            "ab657a42-ffac-52f0-acf6-fb9cf767b9fa",
        ],
    },
    {
        "label": "golemsCodex agent",
        "target": "6f6aa18b-d464-50af-8e8b-2a4f216efd95",
        "sources": ["14e320ab-e797-5ed3-a20e-d9589730f139"],
    },
    {
        "label": "brain_search tool child",
        "target": "af04c8f8-f307-5100-b289-b31f0481daed",
        "sources": ["ddd87796-e610-5f34-b5a2-203e7563d153"],
    },
    {
        "label": "brain_store tool child",
        "target": "d41ffaf3-a523-57dd-ad50-d738b1d0cb3a",
        "sources": ["9ff99459-08a2-5bd4-ab7f-7d0a1b66162e"],
    },
    {
        "label": "brain_recall tool child",
        "target": "c5e35273-6e71-5848-a76c-de31d8280669",
        "sources": ["6e542bd2-8877-508a-8de3-5ee34a0af52a"],
    },
    {
        "label": "brain_digest tool child",
        "target": "182281ea-1e39-544d-a133-b7511154aa3e",
        "sources": [
            "5978fb87-b7d2-5731-b720-60feb5640165",
            "29507fb1-d62b-50a1-b3ca-e20776ee3d8b",
            "4b1c6161-9af9-5e4c-88e4-f86c15b36bef",
        ],
    },
    {
        "label": "Domica company",
        "target": DOMICA_COMPANY,
        "sources": [
            "af6e7ea4-bbe9-5aa6-a148-9afa628108d5",
            "3f140ee3-827e-5cd8-957f-7328b1c50bad",
        ],
    },
    {
        "label": "Domica repo",
        "target": DOMICA_REPO,
        "sources": [
            "59e14d7d-5260-5cb4-bcb8-79c8d7d8a453",
            "5832aa42-4814-533f-98a5-9884dde193ce",
        ],
    },
    {
        "label": "brainlayerCursor parked stub",
        "target": "d8b16528-4ee8-5eea-b542-8c8d1d723569",
        "sources": ["f544548c-7fd8-5d7c-bd46-aaf479b1c065"],
    },
]

DOMICACLAUDE_SOURCE_IDS = [
    "5002583c-d978-593b-8c3f-2fa6213d42e7",
    "2cf0afb8-f5ad-53b9-8c2a-887bfafbc0dd",
    "20e0793a-98fb-57b7-93eb-6382bbafd5ed",
    "b40df362-d209-5632-a427-c2a55406d561",
    "60b52dec-c88f-5aca-9555-c8d98b78c39b",
]

DOMICACLAUDE_REPO_CHUNKS = {
    "/Users/etanheyman/.claude/projects/-Users-etanheyman-Gits-orchestrator/133cc8a5-9a3e-43c3-a28f-16e14cc587f2.jsonl:3135",
    "rt-36549975-126282fd97760b10",
    "rt-36549975-56769be1226e3356",
    "rt-9bc494d1-c8976cadd31e4354",
    "rt-aa647023-13b5386ce981bbf9",
    "rt-aa647023-56ca2299106c5a29",
    "rt-aa647023-7c8399bc00005c18",
    "rt-aa647023-9abb410c4025a8ef",
    "rt-aa647023-cd7c5ac8b9bf2a27",
    "rt-aa647023-cdc7480d827692d5",
    "rt-aa647023-ce94ff0086334d0c",
    "rt-aa647023-e510437775757611",
}

DOMICACLAUDE_COMPANY_CHUNKS: set[str] = set()

DOMICACLAUDE_DROP_CHUNKS = {
    "rt-36549975-621419b7e1921ad3",
    "rt-36549975-bc2c6940da4d8667",
    "rt-36549975-f80670637d1b55fb",
}

BRAIN_TOOL_CHILD_IDS = [
    "af04c8f8-f307-5100-b289-b31f0481daed",
    "d41ffaf3-a523-57dd-ad50-d738b1d0cb3a",
    "c5e35273-6e71-5848-a76c-de31d8280669",
    "182281ea-1e39-544d-a133-b7511154aa3e",
    "4589a616-dab0-50ad-aad1-9912f8070d03",
    "9fd3af9f-cf22-5788-b885-fb2c4d100d3d",
    "6c057ec9-5fe0-5343-9f2a-04b45ffa71ce",
    "0969612d-ab05-5372-963c-7fb3bc047e4f",
]

CURSOR_STUB_IDS = [
    "d8b16528-4ee8-5eea-b542-8c8d1d723569",
    "ffd47e32-0283-5eb7-976f-114513983a02",
]

EXPECTED_COUNTS = {
    "merge_sources": 40,
    "launcher_delete_rows": 5,
    "promotions": 3,
}


def all_merge_source_ids() -> list[str]:
    return [source for group in MERGE_GROUPS for source in group["sources"]]


def validate_counts(counts: dict[str, int]) -> None:
    mismatches = {
        key: (counts.get(key), expected) for key, expected in EXPECTED_COUNTS.items() if counts.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"count gate mismatch: {mismatches}")


def relation_count(store: VectorStore, entity_id: str) -> int:
    cursor = store._read_cursor()
    return cursor.execute(
        "SELECT count(*) FROM kg_relations WHERE source_id = ? OR target_id = ?",
        (entity_id, entity_id),
    ).fetchone()[0]


def collect_counts(store: VectorStore) -> dict[str, int]:
    cursor = store._read_cursor()
    source_ids = all_merge_source_ids()
    launcher_ids = DOMICACLAUDE_SOURCE_IDS
    placeholders = ",".join("?" for _ in source_ids)
    launcher_placeholders = ",".join("?" for _ in launcher_ids)
    promotion_placeholders = ",".join("?" for _ in PROMOTION_IDS)
    merge_sources = cursor.execute(
        f"SELECT count(*) FROM kg_entities WHERE id IN ({placeholders}) AND status = 'active'",
        source_ids,
    ).fetchone()[0]
    launcher_delete_rows = cursor.execute(
        f"SELECT count(*) FROM kg_entities WHERE id IN ({launcher_placeholders}) AND status = 'active'",
        launcher_ids,
    ).fetchone()[0]
    promotions = cursor.execute(
        f"SELECT count(*) FROM kg_entities WHERE id IN ({promotion_placeholders}) AND entity_type = 'tool'",
        tuple(PROMOTION_IDS),
    ).fetchone()[0]
    source_relation_rows = sum(relation_count(store, entity_id) for entity_id in source_ids + launcher_ids)
    return {
        "merge_sources": merge_sources,
        "launcher_delete_rows": launcher_delete_rows,
        "promotions": promotions,
        "source_relation_rows": source_relation_rows,
    }


def validate_live_shape(store: VectorStore) -> None:
    counts = collect_counts(store)
    validate_counts(counts)
    if counts["source_relation_rows"] != 0:
        raise RuntimeError(f"approved source rows gained relations: {counts['source_relation_rows']}")

    target_ids = {group["target"] for group in MERGE_GROUPS}
    target_ids.update({DOMICA_COMPANY, DOMICA_REPO, *BRAIN_TOOL_CHILD_IDS, *CURSOR_STUB_IDS})
    missing_targets = sorted(entity_id for entity_id in target_ids if store.get_entity(entity_id) is None)
    if missing_targets:
        raise RuntimeError(f"target rows missing: {missing_targets}")


def promote_to_agent(store: VectorStore, entity_id: str) -> None:
    cursor = store.conn.cursor()
    row = cursor.execute("SELECT name, entity_type FROM kg_entities WHERE id = ?", (entity_id,)).fetchone()
    if not row:
        raise RuntimeError(f"promotion row missing: {entity_id}")
    name, entity_type = row
    if entity_type == "agent":
        return
    if entity_type != "tool":
        raise RuntimeError(f"cannot promote {entity_id}: expected tool, found {entity_type}")
    existing = cursor.execute(
        "SELECT id FROM kg_entities WHERE entity_type = 'agent' AND name = ? AND id != ?",
        (name, entity_id),
    ).fetchone()
    if existing:
        raise RuntimeError(f"agent named {name!r} already exists: {existing[0]}")
    cursor.execute(
        """
        UPDATE kg_entities
        SET entity_type = 'agent',
            canonical_name = coalesce(nullif(canonical_name, ''), lower(?)),
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
        WHERE id = ?
        """,
        (name, entity_id),
    )


def _merge_mention_type(left: str | None, right: str | None) -> str | None:
    if left == "explicit" or right == "explicit":
        return "explicit"
    return left or right


def _coalesce_max(left: float | int | None, right: float | int | None) -> float | int | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _coalesce_min(left: float | int | None, right: float | int | None) -> float | int | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def move_chunk_link(store: VectorStore, source_id: str, target_id: str, chunk_id: str) -> bool:
    cursor = store.conn.cursor()
    row = cursor.execute(
        """
        SELECT relevance, context, mention_type, relation_tier, weight
        FROM kg_entity_chunks
        WHERE entity_id = ? AND chunk_id = ?
        """,
        (source_id, chunk_id),
    ).fetchone()
    if not row:
        return False

    relevance, context, mention_type, relation_tier, weight = row
    existing = cursor.execute(
        """
        SELECT relevance, context, mention_type, relation_tier, weight
        FROM kg_entity_chunks
        WHERE entity_id = ? AND chunk_id = ?
        """,
        (target_id, chunk_id),
    ).fetchone()
    if existing:
        existing_relevance, existing_context, existing_mention_type, existing_tier, existing_weight = existing
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
                _coalesce_max(existing_relevance, relevance),
                merged_context,
                _merge_mention_type(existing_mention_type, mention_type),
                _coalesce_min(existing_tier, relation_tier),
                _coalesce_max(existing_weight, weight),
                target_id,
                chunk_id,
            ),
        )
        return True

    cursor.execute(
        """
        INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance, context, mention_type, relation_tier, weight)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (target_id, chunk_id, relevance, context, mention_type, relation_tier, weight),
    )
    return True


def cleanup_domica_launcher_rows(
    store: VectorStore,
    *,
    source_ids: list[str],
    repo_chunk_ids: set[str],
    company_chunk_ids: set[str],
    drop_chunk_ids: set[str],
    repo_id: str,
    company_id: str,
) -> Counter[str]:
    cursor = store.conn.cursor()
    stats: Counter[str] = Counter()
    expected_chunks = repo_chunk_ids | company_chunk_ids | drop_chunk_ids

    for source_id in source_ids:
        if relation_count(store, source_id) != 0:
            raise RuntimeError(f"domicaClaude launcher row unexpectedly has relations: {source_id}")
        rows = list(
            cursor.execute(
                "SELECT chunk_id FROM kg_entity_chunks WHERE entity_id = ? ORDER BY chunk_id",
                (source_id,),
            )
        )
        for (chunk_id,) in rows:
            if chunk_id not in expected_chunks:
                raise RuntimeError(f"unclassified domicaClaude chunk link: {source_id} -> {chunk_id}")
            if chunk_id in repo_chunk_ids:
                if move_chunk_link(store, source_id, repo_id, chunk_id):
                    stats["domica_launcher_repo_links"] += 1
            elif chunk_id in company_chunk_ids:
                if move_chunk_link(store, source_id, company_id, chunk_id):
                    stats["domica_launcher_company_links"] += 1
            else:
                stats["domica_launcher_dropped_links"] += 1
            cursor.execute(
                "DELETE FROM kg_entity_chunks WHERE entity_id = ? AND chunk_id = ?",
                (source_id, chunk_id),
            )
        cursor.execute("DELETE FROM kg_entity_aliases WHERE entity_id = ?", (source_id,))
        cursor.execute("DELETE FROM kg_vec_entities WHERE entity_id = ?", (source_id,))
        cursor.execute("DELETE FROM kg_entities WHERE id = ?", (source_id,))
        stats["domica_launcher_rows_deleted"] += 1
    return stats


def set_parent(store: VectorStore, entity_id: str, parent_id: str) -> None:
    store.conn.cursor().execute(
        """
        UPDATE kg_entities
        SET parent_id = ?,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
        WHERE id = ?
        """,
        (parent_id, entity_id),
    )


def mark_low_importance(store: VectorStore, entity_ids: list[str]) -> None:
    placeholders = ",".join("?" for _ in entity_ids)
    store.conn.cursor().execute(
        f"""
        UPDATE kg_entities
        SET importance = min(coalesce(importance, 0.5), 0.25),
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
        WHERE id IN ({placeholders})
        """,
        entity_ids,
    )


def merge_group(store: VectorStore, target_id: str, source_ids: list[str]) -> Counter[str]:
    stats: Counter[str] = Counter()
    for source_id in source_ids:
        merge_stats = merge_entities_preserving_links(store, target_id, source_id)
        stats.update({f"merge_{key}": value for key, value in merge_stats.items()})
    return stats


def apply_plan(store: VectorStore) -> Counter[str]:
    validate_live_shape(store)
    stats: Counter[str] = Counter()

    for entity_id in sorted(PROMOTION_IDS):
        promote_to_agent(store, entity_id)
        stats["promotions"] += 1

    for group in MERGE_GROUPS:
        group_stats = merge_group(store, group["target"], group["sources"])
        stats.update(group_stats)
        stats[f"group_{group['label']}_sources"] = len(group["sources"])

    for tool_id in BRAIN_TOOL_CHILD_IDS:
        set_parent(store, tool_id, BRAINLAYER_PROJECT)
        stats["brain_tool_parent_updates"] += 1
    set_parent(store, DOMICA_REPO, DOMICA_COMPANY)
    stats["domica_repo_parent_updates"] += 1
    mark_low_importance(store, CURSOR_STUB_IDS)
    stats["cursor_stubs_low_importance"] = len(CURSOR_STUB_IDS)

    stats.update(
        cleanup_domica_launcher_rows(
            store,
            source_ids=DOMICACLAUDE_SOURCE_IDS,
            repo_chunk_ids=DOMICACLAUDE_REPO_CHUNKS,
            company_chunk_ids=DOMICACLAUDE_COMPANY_CHUNKS,
            drop_chunk_ids=DOMICACLAUDE_DROP_CHUNKS,
            repo_id=DOMICA_REPO,
            company_id=DOMICA_COMPANY,
        )
    )
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


def print_plan(store: VectorStore) -> None:
    counts = collect_counts(store)
    print("P2 cross-type cleanup plan")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print("Groups:")
    for group in MERGE_GROUPS:
        print(f"  {group['label']}: target={group['target']} sources={len(group['sources'])}")
    print(f"  domicaClaude launcher rows: {len(DOMICACLAUDE_SOURCE_IDS)}")
    print(f"  domicaClaude repo chunks: {len(DOMICACLAUDE_REPO_CHUNKS)}")
    print(f"  domicaClaude dropped chunks: {len(DOMICACLAUDE_DROP_CHUNKS)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply the approved cleanup")
    args = parser.parse_args()

    store = VectorStore(get_db_path(), readonly=not args.apply)
    try:
        validate_live_shape(store)
        print_plan(store)
        if not args.apply:
            print("DRY-RUN: no DB writes performed")
            return 0

        cursor = store.conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            stats = apply_plan(store)
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
