"""Tests for BMPM-backed Hebrew alias resolution."""

import importlib.util
import sqlite3
import time
from pathlib import Path

import pytest

from brainlayer.phonetic import phonetic_key
from brainlayer.pipeline.digest import entity_lookup
from brainlayer.vector_store import VectorStore

HOOKS_DIR = Path(__file__).parent.parent / "hooks"


def load_hook_module(filename: str):
    """Import a hook script as a module."""
    spec = importlib.util.spec_from_file_location(
        filename.replace("-", "_").replace(".py", ""),
        HOOKS_DIR / filename,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore with KG tables."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def prompt_search():
    return load_hook_module("brainlayer-prompt-search.py")


def _upsert_person(store: VectorStore, entity_id: str = "person-etan", name: str = "Etan Heyman") -> str:
    return store.upsert_entity(entity_id, "person", name, metadata={})


def test_phonetic_key_generates():
    from brainlayer.phonetic import phonetic_key

    english = phonetic_key("Etan")
    hebrew = phonetic_key("איתן")

    assert isinstance(english, str)
    assert isinstance(hebrew, str)
    assert english
    assert hebrew


def test_phonetic_match_hebrew_english():
    from brainlayer.phonetic import phonetic_match

    assert phonetic_match("Etan", "איתן")


def test_phonetic_match_handles(store):
    entity_id = _upsert_person(store)
    store.add_entity_alias("EtanHey", entity_id, alias_type="handle")

    resolved = store.resolve_entity("EtanHey")

    assert resolved is not None
    assert resolved["id"] == entity_id


def test_seed_aliases_populates(tmp_path):
    db_path = tmp_path / "seed.db"
    store = VectorStore(db_path)
    entity_id = store.upsert_entity("person-etan", "person", "Etan Heyman", metadata={})
    store.close()

    from scripts.seed_aliases import seed_aliases

    inserted = seed_aliases(db_path)

    conn = sqlite3.connect(db_path)
    aliases = conn.execute(
        "SELECT alias, alias_type, entity_id FROM kg_entity_aliases WHERE entity_id = ?",
        (entity_id,),
    ).fetchall()
    conn.close()

    assert inserted >= 2
    assert ("איתן", "hebrew", entity_id) in aliases
    assert any(alias_type == "phonetic" for _, alias_type, _ in aliases)


def test_resolve_entity_exact(store):
    entity_id = _upsert_person(store)

    resolved = store.resolve_entity("Etan Heyman")

    assert resolved is not None
    assert resolved["id"] == entity_id


def test_resolve_entity_alias(store):
    entity_id = _upsert_person(store)
    store.add_entity_alias("איתן", entity_id, alias_type="hebrew")

    resolved = store.resolve_entity("איתן")

    assert resolved is not None
    assert resolved["id"] == entity_id


def test_resolve_entity_phonetic(store):
    _upsert_person(store)
    from scripts.seed_aliases import seed_aliases

    seed_aliases(store.db_path)

    resolved = entity_lookup(
        query="איתן",
        store=store,
        embed_fn=lambda _: [0.0] * 1024,
        entity_type="person",
    )

    assert resolved is not None
    assert resolved["name"] == "Etan Heyman"


def test_hook_detects_hebrew_entity(prompt_search, tmp_path):
    db_path = tmp_path / "hook.db"
    store = VectorStore(db_path)
    entity_id = _upsert_person(store)
    store.add_entity_alias("איתן", entity_id, alias_type="hebrew")

    from scripts.seed_aliases import seed_aliases

    seed_aliases(store.db_path)
    store.close()

    conn = sqlite3.connect(db_path)
    matches = prompt_search.detect_entities_in_prompt("מה איתן מעדיף לפגישות?", conn)
    conn.close()

    assert any(match["id"] == entity_id for match in matches)


def test_hook_runs_phonetic_fallback_even_with_exact_match(prompt_search, tmp_path):
    db_path = tmp_path / "hook-mixed.db"
    store = VectorStore(db_path)
    person_id = _upsert_person(store)
    project_id = store.upsert_entity("project-brainlayer", "project", "BrainLayer", metadata={})
    store.add_entity_alias(phonetic_key("Etan"), person_id, alias_type="phonetic")
    store.close()

    conn = sqlite3.connect(db_path)
    matches = prompt_search.detect_entities_in_prompt("Tell me about BrainLayer and what does איתן think?", conn)
    conn.close()

    assert {match["id"] for match in matches} == {person_id, project_id}


def test_alias_lookup_performance(store):
    entity_id = _upsert_person(store)
    store.add_entity_alias("איתן", entity_id, alias_type="hebrew")

    # Warm cache and import cost before timing.
    store.resolve_entity("איתן")

    start = time.perf_counter()
    iterations = 200
    for _ in range(iterations):
        resolved = store.resolve_entity("איתן")
        assert resolved is not None
        assert resolved["id"] == entity_id
    avg_ms = ((time.perf_counter() - start) / iterations) * 1000

    assert avg_ms < 5, f"Alias resolution averaged {avg_ms:.3f}ms per call"
