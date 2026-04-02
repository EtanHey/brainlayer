"""Agent-as-Entity enrichment — populate KG agent entities with structured profiles.

Reads agents_registry.yaml and updates kg_entities with descriptions,
importance, metadata, and typed relations. Fixes known data issues
(self-referential edges, missing entities).

Usage:
    python -m brainlayer.pipeline.agent_enrichment [--dry-run]
"""

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).parent.parent / "data" / "agents_registry.yaml"


def load_registry() -> dict[str, Any]:
    """Load agent registry YAML."""
    try:
        import yaml
    except ImportError:
        # Fallback: parse simple YAML manually for environments without PyYAML
        raise ImportError("PyYAML required for agent registry. pip install pyyaml")

    with open(REGISTRY_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("agents", {})


def enrich_agents(db_path: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Enrich agent entities in the KG from the registry.

    Returns stats dict with counts of created/updated/fixed entities and relations.
    """
    if db_path is None:
        from ..paths import get_db_path

        db_path = get_db_path()

    registry = load_registry()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")

    stats = {"entities_created": 0, "entities_updated": 0, "relations_added": 0, "relations_fixed": 0}

    try:
        # Fix 1: Remove self-referential edges
        self_refs = conn.execute(
            "SELECT id, source_id FROM kg_relations WHERE source_id = target_id"
        ).fetchall()
        for rel_id, source_id in self_refs:
            name = conn.execute(
                "SELECT name FROM kg_entities WHERE id = ?", (source_id,)
            ).fetchone()
            logger.info("Removing self-referential edge: %s → %s", name, name)
            if not dry_run:
                conn.execute("DELETE FROM kg_relations WHERE id = ?", (rel_id,))
            stats["relations_fixed"] += 1

        # Process each agent in registry
        for agent_name, agent_data in registry.items():
            _process_agent(conn, agent_name, agent_data, stats, dry_run)

        if not dry_run:
            conn.commit()

    finally:
        conn.close()

    return stats


def _process_agent(
    conn: sqlite3.Connection,
    name: str,
    data: dict[str, Any],
    stats: dict[str, int],
    dry_run: bool,
) -> None:
    """Create or update a single agent entity."""
    # Check if entity exists
    row = conn.execute(
        "SELECT id, description, metadata FROM kg_entities WHERE name = ? AND entity_type = 'agent'",
        (name,),
    ).fetchone()

    metadata = {
        "repo": data.get("repo"),
        "skill_path": data.get("skill_path"),
        "model": data.get("model"),
        "capabilities": data.get("capabilities", []),
    }
    metadata_json = json.dumps(metadata)
    description = data.get("description", "")
    importance = 7.0  # Agents are high-importance entities

    if row:
        entity_id = row[0]
        # Update existing entity
        if not dry_run:
            conn.execute(
                """UPDATE kg_entities
                   SET description = ?, metadata = ?, importance = ?,
                       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id = ?""",
                (description, metadata_json, importance, entity_id),
            )
        stats["entities_updated"] += 1
        logger.info("Updated agent: %s (id=%s)", name, entity_id)
    else:
        entity_id = f"agent-{uuid.uuid4().hex[:12]}"
        if not dry_run:
            conn.execute(
                """INSERT INTO kg_entities (id, entity_type, name, description, metadata, importance, created_at, updated_at)
                   VALUES (?, 'agent', ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'), strftime('%Y-%m-%dT%H:%M:%fZ','now'))""",
                (entity_id, name, description, metadata_json, importance),
            )
        stats["entities_created"] += 1
        logger.info("Created agent: %s (id=%s)", name, entity_id)

    # Add relations
    for rel in data.get("relations", []):
        _add_relation(conn, entity_id, name, rel, stats, dry_run)


def _add_relation(
    conn: sqlite3.Connection,
    source_id: str,
    source_name: str,
    rel: dict[str, str],
    stats: dict[str, int],
    dry_run: bool,
) -> None:
    """Add a typed relation from the registry."""
    target_name = rel["target"]
    rel_type = rel["type"]
    description = rel.get("description", "")

    # Find target entity
    target_row = conn.execute(
        "SELECT id FROM kg_entities WHERE name = ?", (target_name,)
    ).fetchone()

    if not target_row:
        logger.debug("Target entity %r not found for relation %s -> %s", target_name, source_name, target_name)
        return

    target_id = target_row[0]

    # Check if relation already exists
    existing = conn.execute(
        "SELECT id FROM kg_relations WHERE source_id = ? AND target_id = ? AND relation_type = ?",
        (source_id, target_id, rel_type),
    ).fetchone()

    if existing:
        return

    rel_id = f"rel-{uuid.uuid4().hex[:12]}"
    props = json.dumps({"description": description, "source": "agents_registry"})

    if not dry_run:
        conn.execute(
            """INSERT OR IGNORE INTO kg_relations
               (id, source_id, target_id, relation_type, properties, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, 0.95, strftime('%Y-%m-%dT%H:%M:%fZ','now'))""",
            (rel_id, source_id, target_id, rel_type, props),
        )
    stats["relations_added"] += 1
    logger.info("Added relation: %s --%s--> %s", source_name, rel_type, target_name)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Enrich agent entities from registry")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    args = parser.parse_args()

    result = enrich_agents(dry_run=args.dry_run)
    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{prefix}Agent enrichment complete:")
    print(f"  Entities created: {result['entities_created']}")
    print(f"  Entities updated: {result['entities_updated']}")
    print(f"  Relations added:  {result['relations_added']}")
    print(f"  Relations fixed:  {result['relations_fixed']}")
