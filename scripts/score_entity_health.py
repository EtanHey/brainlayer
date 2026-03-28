"""R49 Phase 1 — Score entity health against type contracts.

Evaluates every entity against its type contract and populates the
entity_health table with completeness scores.

Scoring algorithm (5 dimensions, weighted):
  40% — Required field coverage
  25% — Expected field coverage
  15% — Relationship density (relative to type contract minimum)
  10% — Chunk density (relative to type contract minimum)
  10% — Recency (exponential decay, 90-day half-life)

Health levels (Wikidata 5-tier):
  5 — Very Detailed (0.85+)
  4 — Detailed (0.65–0.84)
  3 — Moderate (0.45–0.64)
  2 — Basic (0.25–0.44)
  1 — Stub (<0.25)

Usage:
  python scripts/score_entity_health.py [--db PATH] [--contracts PATH]
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_contracts(contracts_path: str) -> Dict[str, Dict[str, Any]]:
    """Load entity type contracts from YAML file.

    Returns:
        Dict mapping entity_type -> contract spec (required, expected, health_criteria, etc.)
    """
    with open(contracts_path) as f:
        data = yaml.safe_load(f)
    return data.get("entity_types", {})


def classify_health_level(score: float) -> int:
    """Classify a completeness score into a 1-5 health level.

    5 — Very Detailed (0.85+)
    4 — Detailed (0.65–0.84)
    3 — Moderate (0.45–0.64)
    2 — Basic (0.25–0.44)
    1 — Stub (<0.25)
    """
    if score >= 0.85:
        return 5
    elif score >= 0.65:
        return 4
    elif score >= 0.45:
        return 3
    elif score >= 0.25:
        return 2
    else:
        return 1


def _get_entity_field_values(store, entity_id: str) -> Dict[str, Any]:
    """Extract all known field values for an entity from its row + metadata."""
    cursor = store._read_cursor()
    row = list(cursor.execute(
        """SELECT id, entity_type, name, metadata, description,
                  canonical_name, confidence, importance, entity_subtype, status
           FROM kg_entities WHERE id = ?""",
        (entity_id,),
    ))
    if not row:
        return {}

    r = row[0]
    metadata = json.loads(r[3]) if r[3] else {}

    # Merge top-level columns and metadata into a flat field map
    fields = {
        "name": r[2],
        "type": r[1],
        "description": r[4] or metadata.get("description"),
        "entity_subtype": r[8],
        "status": r[9],
    }

    # Add all metadata keys as fields
    for k, v in metadata.items():
        if k not in fields:
            fields[k] = v

    return fields


def _count_relationships(store, entity_id: str) -> int:
    """Count total relationships (incoming + outgoing) for an entity."""
    cursor = store._read_cursor()
    out_count = list(cursor.execute(
        "SELECT COUNT(*) FROM kg_relations WHERE source_id = ?", (entity_id,)
    ))[0][0]
    in_count = list(cursor.execute(
        "SELECT COUNT(*) FROM kg_relations WHERE target_id = ?", (entity_id,)
    ))[0][0]
    return out_count + in_count


def _count_chunks(store, entity_id: str) -> int:
    """Count chunks linked to an entity."""
    cursor = store._read_cursor()
    return list(cursor.execute(
        "SELECT COUNT(*) FROM kg_entity_chunks WHERE entity_id = ?", (entity_id,)
    ))[0][0]


def _get_entity_updated_at(store, entity_id: str) -> Optional[str]:
    """Get the most recent updated_at for an entity."""
    cursor = store._read_cursor()
    row = list(cursor.execute(
        "SELECT updated_at FROM kg_entities WHERE id = ?", (entity_id,)
    ))
    if row and row[0][0]:
        return row[0][0]
    return None


def score_entity(
    store,
    entity_id: str,
    entity_type: str,
    contract: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a single entity against its type contract.

    Returns dict with: completeness_score, health_level, missing_required,
    missing_expected, chunk_count, relationship_count.
    """
    fields = _get_entity_field_values(store, entity_id)
    required_fields = contract.get("required", [])
    expected_fields = contract.get("expected", [])
    health_criteria = contract.get("health_criteria", {})

    # 1. Required field coverage (40%)
    if required_fields:
        populated_required = sum(
            1 for f in required_fields
            if fields.get(f) is not None and fields.get(f) != ""
        )
        required_score = populated_required / len(required_fields)
    else:
        required_score = 1.0

    missing_required = [
        f for f in required_fields
        if fields.get(f) is None or fields.get(f) == ""
    ]

    # 2. Expected field coverage (25%)
    if expected_fields:
        populated_expected = sum(
            1 for f in expected_fields
            if fields.get(f) is not None and fields.get(f) != ""
        )
        expected_score = populated_expected / len(expected_fields)
    else:
        expected_score = 1.0

    missing_expected = [
        f for f in expected_fields
        if fields.get(f) is None or fields.get(f) == ""
    ]

    # 3. Relationship density (15%) — relative to contract minimum
    relationship_count = _count_relationships(store, entity_id)
    min_rels = health_criteria.get("min_relationships", 1)
    rel_score = min(relationship_count / max(min_rels, 1), 1.0)

    # 4. Chunk density (10%) — relative to contract minimum
    chunk_count = _count_chunks(store, entity_id)
    min_chunks = health_criteria.get("min_chunks", 1)
    chunk_score = min(chunk_count / max(min_chunks, 1), 1.0)

    # 5. Recency (10%) — exponential decay, 90-day half-life
    updated_at = _get_entity_updated_at(store, entity_id)
    if updated_at:
        try:
            # Handle both formats: with and without 'Z'
            ts = updated_at.rstrip("Z")
            updated_dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_stale = max((now - updated_dt).total_seconds() / 86400, 0)
            recency_score = math.pow(0.5, days_stale / 90.0)
        except (ValueError, TypeError):
            recency_score = 0.5  # Unknown date, assume moderate
    else:
        recency_score = 0.5  # No date, assume moderate

    # Weighted combination
    completeness_score = (
        0.40 * required_score
        + 0.25 * expected_score
        + 0.15 * rel_score
        + 0.10 * chunk_score
        + 0.10 * recency_score
    )

    # Clamp to [0, 1]
    completeness_score = max(0.0, min(1.0, completeness_score))

    return {
        "completeness_score": round(completeness_score, 4),
        "health_level": classify_health_level(completeness_score),
        "missing_required": missing_required,
        "missing_expected": missing_expected,
        "chunk_count": chunk_count,
        "relationship_count": relationship_count,
    }


def score_all_entities(store, contracts: Dict[str, Dict[str, Any]]) -> int:
    """Score all entities and persist results to entity_health table.

    Returns number of entities scored.
    """
    cursor = store._read_cursor()
    entities = list(cursor.execute(
        "SELECT id, entity_type, name FROM kg_entities"
    ))

    scored = 0
    write_cursor = store.conn.cursor()

    for entity_id, entity_type, entity_name in entities:
        # Find the contract — check direct type, then check parent type via hierarchy
        contract = contracts.get(entity_type)
        if contract is None:
            # Try parent type from hierarchy
            parent_row = list(cursor.execute(
                "SELECT parent_type FROM entity_type_hierarchy WHERE child_type = ?",
                (entity_type,),
            ))
            if parent_row:
                contract = contracts.get(parent_row[0][1] if len(parent_row[0]) > 1 else parent_row[0][0])
        if contract is None:
            # Default: score with minimal contract
            contract = {
                "required": ["name", "type", "description"],
                "expected": [],
                "health_criteria": {"min_chunks": 1, "max_stale_days": 180, "min_relationships": 0},
            }

        result = score_entity(store, entity_id, entity_type, contract)

        write_cursor.execute(
            """INSERT INTO entity_health
               (entity_name, completeness_score, health_level,
                missing_required, missing_expected, chunk_count,
                relationship_count, last_scored_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
               ON CONFLICT(entity_name) DO UPDATE SET
                 completeness_score = excluded.completeness_score,
                 health_level = excluded.health_level,
                 missing_required = excluded.missing_required,
                 missing_expected = excluded.missing_expected,
                 chunk_count = excluded.chunk_count,
                 relationship_count = excluded.relationship_count,
                 last_scored_at = excluded.last_scored_at
            """,
            (
                entity_name,
                result["completeness_score"],
                result["health_level"],
                json.dumps(result["missing_required"]),
                json.dumps(result["missing_expected"]),
                result["chunk_count"],
                result["relationship_count"],
            ),
        )
        scored += 1

    return scored


def main():
    """CLI entry point: score all entities in the production database."""
    from brainlayer.paths import get_db_path
    from brainlayer.vector_store import VectorStore

    db_path = get_db_path()
    contracts_path = Path(__file__).parent.parent / "contracts" / "entity-types.yaml"

    if not contracts_path.exists():
        print(f"ERROR: Contracts file not found at {contracts_path}")
        sys.exit(1)

    print(f"Loading contracts from {contracts_path}")
    contracts = load_contracts(str(contracts_path))
    print(f"Loaded {len(contracts)} entity type contracts")

    print(f"Opening database at {db_path}")
    store = VectorStore(db_path)

    try:
        scored = score_all_entities(store, contracts)
        print(f"Scored {scored} entities")

        # Print summary
        cursor = store._read_cursor()
        for level in range(5, 0, -1):
            count = list(cursor.execute(
                "SELECT COUNT(*) FROM entity_health WHERE health_level = ?", (level,)
            ))[0][0]
            labels = {5: "Very Detailed", 4: "Detailed", 3: "Moderate", 2: "Basic", 1: "Stub"}
            print(f"  Level {level} ({labels[level]}): {count} entities")
    finally:
        store.close()


if __name__ == "__main__":
    main()
