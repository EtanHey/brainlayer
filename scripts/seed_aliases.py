"""Seed known KG aliases and phonetic keys."""

from __future__ import annotations

import json
import os
from pathlib import Path

from brainlayer.phonetic import phonetic_key
from brainlayer.vector_store import VectorStore


def _load_known_aliases() -> list[tuple[str, str, str]]:
    """Load optional, user-owned alias seeds without shipping personal defaults."""
    path = os.environ.get("BRAINLAYER_KG_ALIAS_SEED_PATH")
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(
        not isinstance(item, list)
        or len(item) != 3
        or not all(isinstance(value, str) and value.strip() for value in item)
        for item in payload
    ):
        raise ValueError("KG alias seed must be a list of [alias, entity_name, alias_type] triples")
    return [tuple(item) for item in payload]


def _iter_entities(store: VectorStore) -> list[tuple[str, str, str]]:
    """Return entity id, name, and canonical name for all KG entities."""
    return list(
        store._read_cursor().execute(
            """
            SELECT id, name, canonical_name
            FROM kg_entities
            ORDER BY name
            """
        )
    )


def seed_aliases(db_path: str | Path) -> int:
    """Populate known aliases and BMPM phonetic aliases. Returns rows inserted."""
    store = VectorStore(Path(db_path))
    inserted = 0
    try:
        entities_by_name = {name: entity_id for entity_id, name, _canonical_name in _iter_entities(store)}

        for alias, entity_name, alias_type in _load_known_aliases():
            entity_id = entities_by_name.get(entity_name)
            if entity_id is None:
                continue
            before = len(store.get_entity_aliases(entity_id))
            store.add_entity_alias(alias, entity_id, alias_type=alias_type)
            after = len(store.get_entity_aliases(entity_id))
            inserted += max(0, after - before)

        for entity_id, name, canonical_name in _iter_entities(store):
            for candidate in {name, canonical_name or ""}:
                key = phonetic_key(candidate)
                if not key:
                    continue
                before = len(store.get_entity_aliases(entity_id))
                store.add_entity_alias(key, entity_id, alias_type="phonetic")
                after = len(store.get_entity_aliases(entity_id))
                inserted += max(0, after - before)
    finally:
        store.close()

    return inserted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed known and phonetic KG aliases.")
    parser.add_argument("db_path", help="Path to BrainLayer SQLite DB")
    args = parser.parse_args()
    print(seed_aliases(args.db_path))
