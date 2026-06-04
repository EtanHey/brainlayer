#!/usr/bin/env python3
"""Reviewable KG entity deduplication and path pseudo-entity purge.

Dry-run is the default. Use --apply only during a supervised bulk-DB window:
stop enrichment workers first, checkpoint WAL before/after, and review the
printed diff before applying.

Approved merges are deliberately explicit. Do not turn --suggest output into
automatic apply behavior.

Optional mapping JSON:
{
  "merges": [
    {
      "label": "canonical by id",
      "canonical": {"id": "person-etan", "entity_type": "person", "name": "Etan Heyman"},
      "sources": ["person-etan-fragment"],
      "aliases": ["Etan"]
    }
  ],
  "name_merges": [
    {
      "label": "canonical by exact reviewed names",
      "canonical": {"entity_type": "person", "name": "Etan Heyman"},
      "source_names": [{"entity_type": "person", "name": "Etan"}],
      "aliases": ["Etan"]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from brainlayer.paths import get_db_path
from brainlayer.pipeline.entity_resolution import merge_entities_preserving_links
from brainlayer.vector_store import VectorStore

try:
    from kg_p2_safe_cleanup import normalize_name as normalize_entity_name
except Exception:  # pragma: no cover - fallback only if the sibling script is unavailable

    def normalize_entity_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())


# Human-approved merge section. Keep this empty unless BL-LEAD has reviewed the
# exact canonical/source mapping. --apply never consumes --suggest output.
APPROVED_MERGES: list[dict[str, Any]] = []
APPROVED_NAME_MERGES: list[dict[str, Any]] = []

# Explicit hard stop for names that must never be merged by this tooling, even if
# a later mapping accidentally includes them.
NEVER_MERGE_NAMES = {"David Heyman"}
NEVER_MERGE_NAME_KEYS = {name.casefold() for name in NEVER_MERGE_NAMES}

KG_TABLES = ("kg_entities", "kg_relations", "kg_entity_chunks", "kg_entity_aliases")
FILE_PATH_PREFIXES = (
    "src/",
    "scripts/",
    "tests/",
    "docs/",
    "migrations/",
    "hooks/",
    "brain-bar/",
    "site/",
    "launchd/",
)
PATH_EXTENSION_RE = re.compile(
    r"\.(md|json|py|pyi|ts|tsx|js|jsx|swift|toml|ya?ml|sh|sql|db|sqlite|txt|html|css|plist)$",
    re.IGNORECASE,
)


class MergeStoreAdapter:
    """Minimal adapter for entity_resolution.merge_entities_preserving_links."""

    def __init__(self, conn: Any):
        self.conn = conn

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        rows = fetch_dicts(
            self.conn,
            """
            SELECT *
            FROM kg_entities
            WHERE id = ?
            LIMIT 1
            """,
            (entity_id,),
        )
        return rows[0] if rows else None

    def add_entity_alias(self, alias: str, entity_id: str, alias_type: str = "name") -> None:
        if not alias:
            return
        execute(
            self.conn,
            """
            INSERT OR IGNORE INTO kg_entity_aliases (alias, entity_id, alias_type, created_at)
            VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            """,
            (alias, entity_id, alias_type),
        )


def execute(conn: Any, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> Any:
    return conn.cursor().execute(sql, params)


def _cursor_description(cursor: Any) -> list[Any]:
    try:
        getdescription = getattr(cursor, "getdescription", None)
        if callable(getdescription):
            return list(getdescription() or [])
        return list(getattr(cursor, "description", None) or [])
    except Exception as exc:
        if exc.__class__.__name__ == "ExecutionCompleteError" and exc.__class__.__module__ == "apsw":
            return []
        raise


def fetch_dicts(conn: Any, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> list[dict[str, Any]]:
    cursor = conn.cursor()
    rows = cursor.execute(sql, params)
    columns = [description[0] for description in _cursor_description(cursor)]
    if not columns:
        return []
    return [dict(zip(columns, row)) for row in rows]


def fetch_one(conn: Any, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> tuple[Any, ...] | None:
    return execute(conn, sql, params).fetchone()


def table_exists(conn: Any, table: str) -> bool:
    return (
        fetch_one(conn, "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (table,))
        is not None
    )


def table_columns(conn: Any, table: str) -> set[str]:
    if not table_exists(conn, table):
        return set()
    return {row[1] for row in execute(conn, f"PRAGMA table_info({table})")}


def table_counts(conn: Any) -> dict[str, int]:
    counts = {}
    for table in KG_TABLES:
        counts[table] = fetch_one(conn, f"SELECT count(*) FROM {table}")[0] if table_exists(conn, table) else 0
    return counts


def _json_metadata(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _canonical_id(group: dict[str, Any]) -> str:
    canonical = group.get("canonical") or {}
    canonical_id = group.get("canonical_id") or group.get("target") or canonical.get("id")
    if not isinstance(canonical_id, str) or not canonical_id:
        raise ValueError(f"merge group {group.get('label', '<unnamed>')} missing canonical id")
    return canonical_id


def _source_ids(group: dict[str, Any]) -> list[str]:
    raw_sources = group.get("sources", group.get("source_ids", group.get("fragments", [])))
    if not isinstance(raw_sources, list) or not all(isinstance(source, str) for source in raw_sources):
        raise ValueError(f"merge group {group.get('label', '<unnamed>')} sources must be a list of ids")
    return raw_sources


def _aliases(group: dict[str, Any]) -> list[str]:
    raw_aliases = group.get("aliases", [])
    if raw_aliases is None:
        return []
    if not isinstance(raw_aliases, list) or not all(isinstance(alias, str) for alias in raw_aliases):
        raise ValueError(f"merge group {group.get('label', '<unnamed>')} aliases must be a list of strings")
    return raw_aliases


def _blocked_name(name: str | None) -> bool:
    return isinstance(name, str) and name.casefold() in NEVER_MERGE_NAME_KEYS


def _raise_if_blocked_name(name: str | None) -> None:
    if _blocked_name(name):
        raise RuntimeError(f"blocked never-merge name in reviewed mapping: {name}")


def ensure_canonical_entity(conn: Any, group: dict[str, Any]) -> str:
    canonical_id = _canonical_id(group)
    adapter = MergeStoreAdapter(conn)
    if adapter.get_entity(canonical_id):
        return canonical_id

    canonical = group.get("canonical") or {}
    name = canonical.get("name")
    entity_type = canonical.get("entity_type")
    if not isinstance(name, str) or not isinstance(entity_type, str):
        raise RuntimeError(
            f"canonical entity {canonical_id} is missing; mapping must define canonical.name and canonical.entity_type"
        )

    existing = fetch_one(
        conn,
        "SELECT id FROM kg_entities WHERE entity_type = ? AND name = ? LIMIT 1",
        (entity_type, name),
    )
    if existing and existing[0] != canonical_id:
        raise RuntimeError(
            f"canonical name/type already exists with id {existing[0]}; refusing to create duplicate {canonical_id}"
        )

    columns = table_columns(conn, "kg_entities")
    insert_columns = ["id", "entity_type", "name"]
    values: list[Any] = [canonical_id, entity_type, name]
    optional_values = {
        "metadata": _json_metadata(canonical.get("metadata")),
        "canonical_name": canonical.get("canonical_name") or name.casefold(),
        "status": canonical.get("status", "active"),
    }
    for column, value in optional_values.items():
        if column in columns:
            insert_columns.append(column)
            values.append(value)
    placeholders = ", ".join("?" for _ in insert_columns)
    execute(
        conn,
        f"INSERT INTO kg_entities ({', '.join(insert_columns)}) VALUES ({placeholders})",
        values,
    )
    return canonical_id


def apply_approved_mapping(conn: Any, mapping: list[dict[str, Any]]) -> Counter[str]:
    """Apply only human-approved merge groups."""
    stats: Counter[str] = Counter()
    adapter = MergeStoreAdapter(conn)

    for group in mapping:
        canonical_id = ensure_canonical_entity(conn, group)
        canonical = adapter.get_entity(canonical_id)
        if not canonical:
            raise RuntimeError(f"canonical entity unavailable after ensure: {canonical_id}")
        _raise_if_blocked_name(canonical.get("name"))

        adapter.add_entity_alias(canonical["name"], canonical_id, alias_type="canonical")
        for alias in _aliases(group):
            adapter.add_entity_alias(alias, canonical_id, alias_type="approved")

        group_sources = [source_id for source_id in _source_ids(group) if source_id != canonical_id]
        for source_id in group_sources:
            source = adapter.get_entity(source_id)
            if not source:
                stats["missing_sources"] += 1
                continue
            _raise_if_blocked_name(source.get("name"))
            adapter.add_entity_alias(source["name"], canonical_id, alias_type="approved_merge")
            merge_stats = merge_entities_preserving_links(adapter, canonical_id, source_id)
            stats.update({f"merge_{key}": value for key, value in merge_stats.items()})
            stats["merge_sources"] += 1
        stats["merge_groups"] += 1
    return stats


def _source_name_specs(group: dict[str, Any]) -> list[dict[str, Any]]:
    raw_specs = group.get("source_names", [])
    if raw_specs is None:
        return []
    if not isinstance(raw_specs, list):
        raise ValueError(f"merge group {group.get('label', '<unnamed>')} source_names must be a list")
    specs = []
    for raw_spec in raw_specs:
        if isinstance(raw_spec, str):
            specs.append({"name": raw_spec})
            continue
        if not isinstance(raw_spec, dict) or not isinstance(raw_spec.get("name"), str):
            raise ValueError("source_names entries must be strings or objects with a string name")
        specs.append(dict(raw_spec))
    return specs


def _entity_type_filter(spec: dict[str, Any]) -> list[str]:
    if isinstance(spec.get("entity_type"), str):
        return [spec["entity_type"]]
    entity_types = spec.get("entity_types")
    if entity_types is None:
        return []
    if not isinstance(entity_types, list) or not all(isinstance(entity_type, str) for entity_type in entity_types):
        raise ValueError("entity_types must be a list of strings")
    return entity_types


def _resolve_canonical_by_name(conn: Any, group: dict[str, Any]) -> dict[str, Any]:
    canonical = group.get("canonical") or {}
    if not isinstance(canonical, dict):
        raise ValueError(f"merge group {group.get('label', '<unnamed>')} canonical must be an object")
    if isinstance(canonical.get("id"), str):
        rows = fetch_dicts(conn, "SELECT id, name, entity_type FROM kg_entities WHERE id = ?", (canonical["id"],))
    else:
        name = canonical.get("name")
        entity_type = canonical.get("entity_type")
        if not isinstance(name, str) or not isinstance(entity_type, str):
            raise ValueError("name-based canonical must define canonical.name and canonical.entity_type")
        _raise_if_blocked_name(name)
        rows = fetch_dicts(
            conn,
            "SELECT id, name, entity_type FROM kg_entities WHERE entity_type = ? AND name = ?",
            (entity_type, name),
        )
    if len(rows) != 1:
        raise RuntimeError(f"canonical exact-name lookup returned {len(rows)} rows for {canonical!r}")
    _raise_if_blocked_name(rows[0]["name"])
    return rows[0]


def _resolve_source_name_spec(conn: Any, spec: dict[str, Any], canonical_id: str) -> list[dict[str, Any]]:
    name = spec["name"]
    _raise_if_blocked_name(name)
    entity_types = _entity_type_filter(spec)
    params: list[Any] = [name]
    type_clause = ""
    if entity_types:
        type_clause = f" AND entity_type IN ({_placeholders(entity_types)})"
        params.extend(entity_types)
    rows = fetch_dicts(
        conn,
        f"""
        SELECT id, name, entity_type
        FROM kg_entities
        WHERE name = ?{type_clause}
        ORDER BY entity_type, id
        """,
        params,
    )
    resolved = []
    for row in rows:
        _raise_if_blocked_name(row["name"])
        if row["id"] != canonical_id:
            resolved.append(row)
    if not resolved and spec.get("required", True):
        raise RuntimeError(f"source exact-name lookup returned no mergeable rows for {spec!r}")
    return resolved


def reviewed_name_mapping_to_id_mapping(conn: Any, mapping: list[dict[str, Any]]) -> list[dict[str, Any]]:
    id_mapping = []
    for group in mapping:
        canonical = _resolve_canonical_by_name(conn, group)
        source_ids: list[str] = []
        for spec in _source_name_specs(group):
            for source in _resolve_source_name_spec(conn, spec, canonical["id"]):
                if source["id"] not in source_ids:
                    source_ids.append(source["id"])
        id_mapping.append(
            {
                "label": group.get("label", canonical["id"]),
                "canonical": {
                    "id": canonical["id"],
                    "entity_type": canonical["entity_type"],
                    "name": canonical["name"],
                },
                "sources": source_ids,
                "aliases": _aliases(group)
                + [spec["name"] for spec in _source_name_specs(group) if not _blocked_name(spec["name"])],
            }
        )
    return id_mapping


def apply_reviewed_name_mapping(conn: Any, mapping: list[dict[str, Any]]) -> Counter[str]:
    """Resolve exact reviewed source_names, then apply the same ID merge path."""
    stats = apply_approved_mapping(conn, reviewed_name_mapping_to_id_mapping(conn, mapping))
    if mapping:
        stats["name_merge_groups"] += len(mapping)
    return stats


def path_pseudo_entity_reason(name: str) -> str | None:
    value = name.strip()
    if not value:
        return None
    if value.startswith("/"):
        return "absolute-path"
    if value.startswith(("./", "../", "~/")):
        return "relative-path"
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return "windows-path"
    if "/" in value or "\\" in value:
        normalized = value.replace("\\", "/")
        if normalized.startswith(FILE_PATH_PREFIXES):
            return "repo-path"
        if PATH_EXTENSION_RE.search(normalized):
            return "file-path"
    return None


def find_path_pseudo_entities(conn: Any) -> list[dict[str, Any]]:
    rows = fetch_dicts(
        conn,
        """
        WITH rc AS (
          SELECT source_id AS entity_id, count(*) AS c FROM kg_relations GROUP BY source_id
          UNION ALL
          SELECT target_id AS entity_id, count(*) AS c FROM kg_relations GROUP BY target_id
        ), rs AS (SELECT entity_id, sum(c) AS rels FROM rc GROUP BY entity_id),
        cc AS (SELECT entity_id, count(*) AS chunks FROM kg_entity_chunks GROUP BY entity_id)
        SELECT e.id, e.name, e.entity_type, coalesce(rs.rels, 0) AS relations, coalesce(cc.chunks, 0) AS chunks
        FROM kg_entities e
        LEFT JOIN rs ON rs.entity_id = e.id
        LEFT JOIN cc ON cc.entity_id = e.id
        ORDER BY e.name, e.id
        """,
    )
    candidates = []
    for row in rows:
        reason = path_pseudo_entity_reason(str(row["name"]))
        if reason:
            row = dict(row)
            row["reason"] = reason
            candidates.append(row)
    return candidates


def _entity_ids(rows_or_ids: list[dict[str, Any]] | list[str]) -> list[str]:
    ids = []
    for row_or_id in rows_or_ids:
        if isinstance(row_or_id, str):
            ids.append(row_or_id)
        else:
            ids.append(str(row_or_id["id"]))
    return ids


def _placeholders(values: list[str]) -> str:
    return ", ".join("?" for _ in values)


def purge_entities(conn: Any, rows_or_ids: list[dict[str, Any]] | list[str]) -> Counter[str]:
    ids = _entity_ids(rows_or_ids)
    stats: Counter[str] = Counter()
    if not ids:
        return stats

    placeholders = _placeholders(ids)
    relation_params = [*ids, *ids]
    stats["purged_chunk_links"] += fetch_one(
        conn,
        f"SELECT count(*) FROM kg_entity_chunks WHERE entity_id IN ({placeholders})",
        ids,
    )[0]
    stats["purged_relations"] += fetch_one(
        conn,
        f"SELECT count(*) FROM kg_relations WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
        relation_params,
    )[0]
    stats["purged_aliases"] += fetch_one(
        conn,
        f"SELECT count(*) FROM kg_entity_aliases WHERE entity_id IN ({placeholders})",
        ids,
    )[0]

    execute(conn, f"DELETE FROM kg_entity_chunks WHERE entity_id IN ({placeholders})", ids)
    execute(
        conn,
        f"DELETE FROM kg_relations WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
        relation_params,
    )
    execute(conn, f"DELETE FROM kg_entity_aliases WHERE entity_id IN ({placeholders})", ids)
    if table_exists(conn, "kg_vec_entities"):
        execute(conn, f"DELETE FROM kg_vec_entities WHERE entity_id IN ({placeholders})", ids)
    execute(conn, f"DELETE FROM kg_entities WHERE id IN ({placeholders})", ids)
    stats["purged_entities"] += len(ids)
    return stats


def suggest_candidate_clusters(conn: Any, *, min_shared_chunks: int = 2) -> list[dict[str, Any]]:
    rows = fetch_dicts(
        conn,
        """
        WITH cc AS (SELECT entity_id, count(*) AS chunks FROM kg_entity_chunks GROUP BY entity_id)
        SELECT e.id, e.name, e.entity_type, coalesce(cc.chunks, 0) AS chunks
        FROM kg_entities e
        LEFT JOIN cc ON cc.entity_id = e.id
        ORDER BY lower(e.name), e.entity_type, e.id
        """,
    )
    suggestions: list[dict[str, Any]] = []
    by_normalized: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_normalized_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = normalize_entity_name(str(row["name"]))
        if key and not path_pseudo_entity_reason(str(row["name"])):
            by_normalized[(str(row["entity_type"]), key)].append(row)
            by_normalized_all[key].append(row)
    for (entity_type, key), family in sorted(by_normalized.items()):
        if len(family) > 1:
            suggestions.append(
                {
                    "reason": "normalized-name",
                    "key": key,
                    "entity_type": entity_type,
                    "entities": family,
                }
            )
    for key, family in sorted(by_normalized_all.items()):
        entity_types = {str(row["entity_type"]) for row in family}
        if len(family) > 1 and len(entity_types) > 1:
            suggestions.append(
                {
                    "reason": "normalized-name-cross-type",
                    "key": key,
                    "entity_types": sorted(entity_types),
                    "entities": family,
                }
            )

    if min_shared_chunks > 0:
        shared_rows = fetch_dicts(
            conn,
            """
            SELECT a.entity_id AS left_id,
                   le.name AS left_name,
                   le.entity_type AS left_type,
                   b.entity_id AS right_id,
                   re.name AS right_name,
                   re.entity_type AS right_type,
                   count(*) AS shared_chunks
            FROM kg_entity_chunks a
            JOIN kg_entity_chunks b ON b.chunk_id = a.chunk_id AND b.entity_id > a.entity_id
            JOIN kg_entities le ON le.id = a.entity_id
            JOIN kg_entities re ON re.id = b.entity_id
            GROUP BY a.entity_id, b.entity_id
            HAVING count(*) >= ?
            ORDER BY shared_chunks DESC, lower(left_name), lower(right_name)
            """,
            (min_shared_chunks,),
        )
        for row in shared_rows:
            suggestions.append(
                {
                    "reason": "shared-chunks",
                    "shared_chunks": row["shared_chunks"],
                    "entities": [
                        {"id": row["left_id"], "name": row["left_name"], "entity_type": row["left_type"]},
                        {"id": row["right_id"], "name": row["right_name"], "entity_type": row["right_type"]},
                    ],
                }
            )
    return suggestions


def _create_clone_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE kg_entities (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            canonical_name TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT,
            updated_at TEXT,
            expired_at TEXT,
            UNIQUE(entity_type, name)
        );
        CREATE TABLE kg_relations (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            properties TEXT DEFAULT '{}',
            confidence REAL DEFAULT 1.0,
            user_verified INTEGER DEFAULT 0,
            fact TEXT,
            importance REAL DEFAULT 0.5,
            valid_from TEXT,
            valid_until TEXT,
            expired_at TEXT,
            source_chunk_id TEXT,
            UNIQUE(source_id, target_id, relation_type)
        );
        CREATE TABLE kg_entity_chunks (
            entity_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            relevance REAL DEFAULT 1.0,
            context TEXT,
            mention_type TEXT,
            relation_tier INTEGER DEFAULT 4,
            weight REAL DEFAULT 0.25,
            PRIMARY KEY (entity_id, chunk_id)
        );
        CREATE TABLE kg_entity_aliases (
            alias TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            alias_type TEXT DEFAULT 'name',
            created_at TEXT,
            valid_from TEXT,
            valid_to TEXT,
            PRIMARY KEY (alias, entity_id)
        );
        CREATE TABLE kg_vec_entities (
            entity_id TEXT PRIMARY KEY,
            embedding BLOB
        );
        """
    )


def clone_kg_subset(conn: Any) -> sqlite3.Connection:
    clone = sqlite3.connect(":memory:")
    _create_clone_schema(clone)
    for table in KG_TABLES:
        if not table_exists(conn, table):
            continue
        source_columns = table_columns(conn, table)
        target_columns = table_columns(clone, table)
        columns = [column for column in target_columns if column in source_columns]
        if not columns:
            continue
        column_list = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        rows = list(execute(conn, f"SELECT {column_list} FROM {table}"))
        clone.executemany(f"INSERT OR IGNORE INTO {table} ({column_list}) VALUES ({placeholders})", rows)
    return clone


def preview_merge_groups(conn: Any, mapping: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previews = []
    for group in mapping:
        canonical_id = _canonical_id(group)
        canonical = fetch_dicts(conn, "SELECT id, name, entity_type FROM kg_entities WHERE id = ?", (canonical_id,))
        sources = []
        for source_id in _source_ids(group):
            rows = fetch_dicts(
                conn,
                """
                WITH rc AS (
                  SELECT source_id AS entity_id, count(*) AS c FROM kg_relations GROUP BY source_id
                  UNION ALL
                  SELECT target_id AS entity_id, count(*) AS c FROM kg_relations GROUP BY target_id
                ), rs AS (SELECT entity_id, sum(c) AS rels FROM rc GROUP BY entity_id),
                cc AS (SELECT entity_id, count(*) AS chunks FROM kg_entity_chunks GROUP BY entity_id)
                SELECT e.id, e.name, e.entity_type, coalesce(rs.rels, 0) AS relations, coalesce(cc.chunks, 0) AS chunks
                FROM kg_entities e
                LEFT JOIN rs ON rs.entity_id = e.id
                LEFT JOIN cc ON cc.entity_id = e.id
                WHERE e.id = ?
                """,
                (source_id,),
            )
            sources.append(
                rows[0] if rows else {"id": source_id, "name": "<missing>", "entity_type": "", "missing": True}
            )
        previews.append(
            {
                "label": group.get("label", canonical_id),
                "canonical": canonical[0]
                if canonical
                else (group.get("canonical") or {"id": canonical_id, "missing": True}),
                "sources": sources,
                "aliases": _aliases(group),
            }
        )
    return previews


def build_dry_run_diff(
    conn: Any, mapping: list[dict[str, Any]], name_mapping: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    name_mapping = name_mapping or []
    before = table_counts(conn)
    purge_candidates = find_path_pseudo_entities(conn)
    clone = clone_kg_subset(conn)
    merge_stats = apply_approved_mapping(clone, mapping)
    name_merge_stats = apply_reviewed_name_mapping(clone, name_mapping)
    purge_stats = purge_entities(clone, find_path_pseudo_entities(clone))
    after = table_counts(clone)
    clone.close()
    return {
        "before": before,
        "after": after,
        "merge_previews": preview_merge_groups(conn, mapping)
        + preview_merge_groups(conn, reviewed_name_mapping_to_id_mapping(conn, name_mapping)),
        "purge_candidates": purge_candidates,
        "merge_stats": dict(merge_stats + name_merge_stats),
        "purge_stats": dict(purge_stats),
    }


def print_diff(diff: dict[str, Any]) -> None:
    print("KG entity dedup dry-run diff")
    print("Counts:")
    for table in KG_TABLES:
        before = diff["before"][table]
        after = diff["after"][table]
        print(f"  {table}: {before} -> {after} ({after - before:+d})")

    print("Approved merges:")
    if not diff["merge_previews"]:
        print("  (none)")
    for group in diff["merge_previews"]:
        canonical = group["canonical"]
        print(
            f"  {group['label']}: canonical={canonical.get('id')} "
            f"name={canonical.get('name', '<new>')!r} type={canonical.get('entity_type', '<new>')}"
        )
        if group["aliases"]:
            print(f"    aliases: {', '.join(group['aliases'])}")
        for source in group["sources"]:
            print(
                f"    merge {source['id']} name={source.get('name')!r} type={source.get('entity_type')} "
                f"chunks={source.get('chunks', 0)} relations={source.get('relations', 0)}"
            )

    print("Path pseudo-entity purges:")
    if not diff["purge_candidates"]:
        print("  (none)")
    for row in diff["purge_candidates"]:
        print(
            f"  purge {row['id']} name={row['name']!r} reason={row['reason']} "
            f"chunks={row['chunks']} relations={row['relations']}"
        )
    print("Simulated stats:")
    for key, value in sorted({**diff["merge_stats"], **diff["purge_stats"]}.items()):
        print(f"  {key}: {value}")
    print("DRY-RUN: no DB writes performed")


def print_suggestions(suggestions: list[dict[str, Any]]) -> None:
    print("KG entity dedup suggestions")
    if not suggestions:
        print("  (none)")
        return
    for index, suggestion in enumerate(suggestions, start=1):
        if suggestion["reason"] == "normalized-name":
            print(
                f"  {index}. normalized-name key={suggestion['key']!r} "
                f"type={suggestion['entity_type']} entities={len(suggestion['entities'])}"
            )
        elif suggestion["reason"] == "normalized-name-cross-type":
            print(
                f"  {index}. normalized-name-cross-type key={suggestion['key']!r} "
                f"types={','.join(suggestion['entity_types'])} entities={len(suggestion['entities'])}"
            )
        else:
            print(f"  {index}. shared-chunks count={suggestion['shared_chunks']}")
        for entity in suggestion["entities"]:
            print(f"     - {entity['id']} {entity['entity_type']} {entity['name']!r}")
    print("SUGGEST ONLY: copy reviewed candidates into APPROVED_MERGES or --mapping JSON before --apply")


def checkpoint_wal(conn: Any, mode: str = "PASSIVE") -> tuple[Any, ...] | None:
    try:
        return fetch_one(conn, f"PRAGMA wal_checkpoint({mode})")
    except Exception as exc:
        print(f"WAL checkpoint {mode} skipped/failed: {exc}")
        return None


def _batches(values: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def apply_with_safety(
    conn: Any,
    mapping: list[dict[str, Any]],
    *,
    name_mapping: list[dict[str, Any]] | None = None,
    batch_size: int,
    checkpoint_every: int,
) -> Counter[str]:
    name_mapping = name_mapping or []
    stats: Counter[str] = Counter()
    print("Safety: verify enrichment workers are stopped before running this bulk operation.")
    print("Safety: script checkpoints WAL before apply, after configured batches, and after apply.")
    checkpoint_wal(conn, "FULL")

    for group in mapping:
        execute(conn, "BEGIN IMMEDIATE")
        try:
            stats.update(apply_approved_mapping(conn, [group]))
            execute(conn, "COMMIT")
        except Exception:
            execute(conn, "ROLLBACK")
            raise
    for group in name_mapping:
        execute(conn, "BEGIN IMMEDIATE")
        try:
            stats.update(apply_reviewed_name_mapping(conn, [group]))
            execute(conn, "COMMIT")
        except Exception:
            execute(conn, "ROLLBACK")
            raise

    purge_candidates = find_path_pseudo_entities(conn)
    for batch_number, batch in enumerate(_batches(purge_candidates, batch_size), start=1):
        execute(conn, "BEGIN IMMEDIATE")
        try:
            stats.update(purge_entities(conn, batch))
            execute(conn, "COMMIT")
        except Exception:
            execute(conn, "ROLLBACK")
            raise
        if checkpoint_every > 0 and batch_number % checkpoint_every == 0:
            checkpoint_wal(conn, "PASSIVE")

    checkpoint_wal(conn, "FULL")
    return stats


def load_mapping(path: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if path is None:
        return list(APPROVED_MERGES), list(APPROVED_NAME_MERGES)
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        id_mapping = data.get("merges", data.get("id_merges", []))
        name_mapping = data.get("name_merges", [])
    else:
        id_mapping = data
        name_mapping = []
    if not isinstance(id_mapping, list) or not all(isinstance(group, dict) for group in id_mapping):
        raise ValueError("--mapping merges must be a JSON list")
    if not isinstance(name_mapping, list) or not all(isinstance(group, dict) for group in name_mapping):
        raise ValueError("--mapping name_merges must be a JSON list")
    return id_mapping, name_mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=get_db_path(), help="SQLite DB path")
    parser.add_argument("--mapping", type=Path, help="Reviewed JSON mapping; defaults to APPROVED_MERGES in this file")
    parser.add_argument("--apply", action="store_true", help="Apply approved mapping and path purge")
    parser.add_argument("--suggest", action="store_true", help="Suggest candidate clusters only; never applies")
    parser.add_argument("--min-shared-chunks", type=int, default=2, help="Minimum shared chunks for --suggest pairs")
    parser.add_argument("--batch-size", type=int, default=5000, help="Entities to purge per write batch")
    parser.add_argument(
        "--checkpoint-every", type=int, default=3, help="Run PASSIVE WAL checkpoint every N purge batches"
    )
    args = parser.parse_args()

    if args.apply and args.suggest:
        parser.error("--suggest is review-only and cannot be combined with --apply")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    mapping, name_mapping = load_mapping(args.mapping)
    store = VectorStore(args.db, readonly=not args.apply)
    try:
        if args.suggest:
            print_suggestions(suggest_candidate_clusters(store.conn, min_shared_chunks=args.min_shared_chunks))
            return 0

        diff = build_dry_run_diff(store.conn, mapping, name_mapping)
        print_diff(diff)
        if not args.apply:
            print("To apply: review the mapping, stop enrichment, then rerun with --apply")
            return 0

        stats = apply_with_safety(
            store.conn,
            mapping,
            name_mapping=name_mapping,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
        )
        print("APPLIED")
        for key, value in sorted(stats.items()):
            print(f"  {key}: {value}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
