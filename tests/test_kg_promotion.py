import json
from pathlib import Path
from unittest.mock import patch

import pytest

from brainlayer import kg_promotion
from brainlayer.pipeline.digest import entity_lookup
from brainlayer.vector_store import VectorStore

# Repair (c): promotion skips archived_at/lineage, not status='archived'.


@pytest.fixture(autouse=True)
def synthetic_given_name_alias(monkeypatch):
    monkeypatch.setattr(kg_promotion, "_KNOWN_GIVEN_NAME_ALIASES", {"alex": {"אלכס"}})


def test_given_name_aliases_ship_empty_and_load_optional_user_config(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv("BRAINLAYER_GIVEN_NAME_ALIASES_PATH", raising=False)
    assert kg_promotion._load_known_given_name_aliases() == {}

    config = tmp_path / "given-name-aliases.json"
    config.write_text(json.dumps({"Alex": ["אלכס"]}), encoding="utf-8")
    monkeypatch.setenv("BRAINLAYER_GIVEN_NAME_ALIASES_PATH", str(config))
    assert kg_promotion._load_known_given_name_aliases() == {"alex": {"אלכס"}}

    config.write_text("[]", encoding="utf-8")
    with caplog.at_level("WARNING", logger="brainlayer.kg_promotion"):
        assert kg_promotion._load_known_given_name_aliases() == {}
    assert [record.message for record in caplog.records] == ["given-name alias config ignored: ValueError"]


def test_given_name_aliases_ignore_malformed_json_without_logging_payload(monkeypatch, tmp_path, caplog):
    config = tmp_path / "given-name-aliases.json"
    config.write_text("{private-payload", encoding="utf-8")
    monkeypatch.setenv("BRAINLAYER_GIVEN_NAME_ALIASES_PATH", str(config))

    with caplog.at_level("WARNING", logger="brainlayer.kg_promotion"):
        assert kg_promotion._load_known_given_name_aliases() == {}
    assert [record.message for record in caplog.records] == ["given-name alias config ignored: JSONDecodeError"]
    assert "private-payload" not in caplog.text


def test_given_name_aliases_ignore_unreadable_file_without_logging_path(monkeypatch, tmp_path, caplog):
    config = tmp_path / "given-name-aliases.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("BRAINLAYER_GIVEN_NAME_ALIASES_PATH", str(config))

    with patch.object(Path, "read_text", side_effect=PermissionError("private-payload")):
        with caplog.at_level("WARNING", logger="brainlayer.kg_promotion"):
            assert kg_promotion._load_known_given_name_aliases() == {}
    assert [record.message for record in caplog.records] == ["given-name alias config ignored: PermissionError"]
    assert str(config) not in caplog.text
    assert "private-payload" not in caplog.text


def _insert_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    content: str,
    raw_entities: list[dict],
    tags: list[str],
) -> None:
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, raw_entities_json, tags
        ) VALUES (?, ?, '{}', 'kg-promotion-test.jsonl', 'brainlayer',
                  'assistant_text', ?, 'test', ?, ?)""",
        (chunk_id, content, len(content), json.dumps(raw_entities), json.dumps(tags)),
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO chunk_tags (chunk_id, tag) VALUES (?, ?)",
        [(chunk_id, tag) for tag in tags],
    )


def test_promotes_identification_tagged_person_surfaces_to_canonical_entity(tmp_path):
    from brainlayer.kg_promotion import promote_raw_entity_identities

    store = VectorStore(tmp_path / "kg-promotion.db")
    try:
        tag = "alex-sample-identification"
        _insert_chunk(
            store,
            chunk_id="chunk-en",
            content="Alex Sample coached Noa for the speakers workshop.",
            raw_entities=[
                {"name": "Alex Sample", "type": "person", "relation": "coach"},
                {"name": "TechGym Speakers Workshop", "type": "concept", "relation": "context"},
            ],
            tags=[tag, "workshop-coaching"],
        )
        _insert_chunk(
            store,
            chunk_id="chunk-he",
            content="היי אלכס, אשמח להשתתף בסדנא",
            raw_entities=[{"name": "אלכס", "type": "person", "relation": "recipient"}],
            tags=[tag, "hebrew"],
        )

        stats = promote_raw_entity_identities(store, entity_type="person")

        entity = store.resolve_entity("Alex Sample")
        assert entity is not None
        assert entity["entity_type"] == "person"
        assert entity["name"] == "Alex Sample"
        hebrew_entity = store.resolve_entity("אלכס")
        assert hebrew_entity is not None
        assert hebrew_entity["id"] == entity["id"]

        aliases = {(row["alias"], row["alias_type"]) for row in store.get_entity_aliases(entity["id"])}
        assert ("אלכס", "raw_surface") in aliases
        assert ("alex-sample-identification", "identity_tag") in aliases

        linked = {row["chunk_id"] for row in store.get_entity_chunks(entity["id"], limit=10, include_audit=True)}
        assert linked == {"chunk-en", "chunk-he"}
        assert stats["entities_promoted"] == 1
        assert stats["aliases_created"] >= 2
        assert stats["chunks_linked"] == 2
    finally:
        store.close()


def test_identity_tag_does_not_alias_co_mentioned_people(tmp_path):
    from brainlayer.kg_promotion import promote_raw_entity_identities

    store = VectorStore(tmp_path / "kg-promotion-co-mentioned.db")
    try:
        tag = "alex-sample-identification"
        _insert_chunk(
            store,
            chunk_id="chunk-co-mentioned",
            content="Alex Sample coached Noa Hey for the speakers workshop.",
            raw_entities=[
                {"name": "Alex Sample", "type": "person", "relation": "coach"},
                {"name": "Noa Hey", "type": "person", "relation": "student"},
            ],
            tags=[tag],
        )
        _insert_chunk(
            store,
            chunk_id="chunk-he",
            content="היי אלכס, אשמח להשתתף בסדנא",
            raw_entities=[{"name": "אלכס", "type": "person", "relation": "recipient"}],
            tags=[tag],
        )

        promote_raw_entity_identities(store, entity_type="person")

        entity = store.resolve_entity("Alex Sample")
        assert entity is not None
        aliases = {row["alias"] for row in store.get_entity_aliases(entity["id"])}
        assert "Noa Hey" not in aliases
        assert store.resolve_entity("Noa Hey") is None
    finally:
        store.close()


def test_identity_tag_does_not_match_untagged_bare_given_name(tmp_path):
    from brainlayer.kg_promotion import promote_raw_entity_identities

    store = VectorStore(tmp_path / "kg-promotion-given-name.db")
    try:
        tag = "john-smith-identification"
        _insert_chunk(
            store,
            chunk_id="identity",
            content="John Smith is the account owner.",
            raw_entities=[{"name": "John Smith", "type": "person", "relation": "owner"}],
            tags=[tag],
        )
        _insert_chunk(
            store,
            chunk_id="identity-tag-only",
            content="Follow-up about John Smith.",
            raw_entities=[],
            tags=[tag],
        )
        _insert_chunk(
            store,
            chunk_id="untagged-john",
            content="John asked a separate unrelated question.",
            raw_entities=[{"name": "John", "type": "person", "relation": "speaker"}],
            tags=["inbox"],
        )

        promote_raw_entity_identities(store, entity_type="person")

        entity = store.resolve_entity("John Smith")
        assert entity is not None
        aliases = {row["alias"] for row in store.get_entity_aliases(entity["id"])}
        linked = {row["chunk_id"] for row in store.get_entity_chunks(entity["id"], limit=10, include_audit=True)}
        assert "John" not in aliases
        assert "untagged-john" not in linked
    finally:
        store.close()


def test_promoted_alias_resolves_through_entity_lookup_for_person(tmp_path):
    from brainlayer.kg_promotion import promote_raw_entity_identities

    store = VectorStore(tmp_path / "kg-promotion-lookup.db")
    try:
        tag = "alex-sample-identification"
        _insert_chunk(
            store,
            chunk_id="chunk-en",
            content="Alex Sample coached Noa for the speakers workshop.",
            raw_entities=[{"name": "Alex Sample", "type": "person", "relation": "coach"}],
            tags=[tag],
        )
        _insert_chunk(
            store,
            chunk_id="chunk-he",
            content="היי אלכס, אשמח להשתתף בסדנא",
            raw_entities=[{"name": "אלכס", "type": "person", "relation": "recipient"}],
            tags=[tag],
        )

        promote_raw_entity_identities(store, entity_type="person")

        by_full_name = entity_lookup(
            query="Alex Sample",
            store=store,
            embed_fn=lambda _: [0.0] * 1024,
            entity_type="person",
        )
        by_hebrew_alias = entity_lookup(
            query="אלכס",
            store=store,
            embed_fn=lambda _: [0.0] * 1024,
            entity_type="person",
        )

        assert by_full_name is not None
        assert by_hebrew_alias is not None
        assert by_hebrew_alias["id"] == by_full_name["id"]
        assert by_hebrew_alias["name"] == "Alex Sample"
    finally:
        store.close()


def test_skips_single_mention_person_without_identity_tag(tmp_path):
    from brainlayer.kg_promotion import promote_raw_entity_identities

    store = VectorStore(tmp_path / "kg-promotion-skip.db")
    try:
        _insert_chunk(
            store,
            chunk_id="chunk-one",
            content="A one-off mention of Dana should not become a KG person.",
            raw_entities=[{"name": "Dana", "type": "person", "relation": "mentioned"}],
            tags=["casual-note"],
        )

        stats = promote_raw_entity_identities(store, entity_type="person")

        assert store.resolve_entity("Dana") is None
        assert stats["entities_promoted"] == 0
        assert stats["chunks_linked"] == 0
    finally:
        store.close()


def test_promotes_stable_person_surface_seen_in_three_chunks(tmp_path):
    from brainlayer.kg_promotion import promote_raw_entity_identities

    store = VectorStore(tmp_path / "kg-promotion-stable.db")
    try:
        for index in range(3):
            _insert_chunk(
                store,
                chunk_id=f"chunk-{index}",
                content=f"Ofir Levi appeared in planning note {index}.",
                raw_entities=[{"name": "Ofir Levi", "type": "person", "relation": "participant"}],
                tags=["planning"],
            )

        stats = promote_raw_entity_identities(store, entity_type="person")

        entity = store.resolve_entity("Ofir Levi")
        assert entity is not None
        assert entity["name"] == "Ofir Levi"
        assert stats["entities_promoted"] == 1
        assert stats["chunks_linked"] == 3
    finally:
        store.close()


def test_identity_tag_can_promote_matching_raw_surfaces_from_other_chunks(tmp_path):
    from brainlayer.kg_promotion import promote_raw_entity_identities

    store = VectorStore(tmp_path / "kg-promotion-real-shape.db")
    try:
        tag = "alex-sample-identification"
        _insert_chunk(
            store,
            chunk_id="identity-tag-only",
            content="Can't find Alex Sample anywhere, but she is the speakers workshop coach.",
            raw_entities=[],
            tags=[tag, "workshop-coaching"],
        )
        _insert_chunk(
            store,
            chunk_id="raw-spelling-drift",
            content="Slide example uses Alex Herskovits as the search target.",
            raw_entities=[{"name": "Alex Herskovits", "type": "person", "relation": "search target"}],
            tags=["presentation-dev"],
        )
        _insert_chunk(
            store,
            chunk_id="raw-hebrew",
            content="היי אלכס, אשמח להשתתף בסדנא",
            raw_entities=[{"name": "אלכס", "type": "person", "relation": "recipient"}],
            tags=["hebrew"],
        )

        stats = promote_raw_entity_identities(store, entity_type="person")

        entity = store.resolve_entity("Alex Sample")
        assert entity is not None
        spelling_drift_entity = store.resolve_entity("Alex Herskovits")
        assert spelling_drift_entity is not None
        assert spelling_drift_entity["id"] == entity["id"]
        hebrew_entity = store.resolve_entity("אלכס")
        assert hebrew_entity is not None
        assert hebrew_entity["id"] == entity["id"]
        assert stats["entities_promoted"] == 1
    finally:
        store.close()


def test_chunk_promotion_includes_untagged_matching_raw_surfaces(tmp_path):
    from brainlayer.kg_promotion import promote_chunk_raw_entities

    store = VectorStore(tmp_path / "kg-promotion-chunk-shape.db")
    try:
        tag = "alex-sample-identification"
        _insert_chunk(
            store,
            chunk_id="identity-tag-only",
            content="Alex Sample is the speakers workshop coach.",
            raw_entities=[],
            tags=[tag],
        )
        _insert_chunk(
            store,
            chunk_id="raw-spelling-drift",
            content="Slide example uses Alex Herskovits as the search target.",
            raw_entities=[{"name": "Alex Herskovits", "type": "person", "relation": "search target"}],
            tags=["presentation-dev"],
        )
        _insert_chunk(
            store,
            chunk_id="raw-hebrew",
            content="היי אלכס, אשמח להשתתף בסדנא",
            raw_entities=[{"name": "אלכס", "type": "person", "relation": "recipient"}],
            tags=["hebrew"],
        )

        stats = promote_chunk_raw_entities(store, "identity-tag-only", entity_type="person")

        entity = store.resolve_entity("Alex Sample")
        assert entity is not None
        spelling_drift_entity = store.resolve_entity("Alex Herskovits")
        assert spelling_drift_entity is not None
        assert spelling_drift_entity["id"] == entity["id"]
        hebrew_entity = store.resolve_entity("אלכס")
        assert hebrew_entity is not None
        assert hebrew_entity["id"] == entity["id"]
        assert stats["entities_promoted"] == 1
    finally:
        store.close()


def test_chunk_promotion_without_identity_tag_is_noop(tmp_path):
    from brainlayer.kg_promotion import promote_chunk_raw_entities

    store = VectorStore(tmp_path / "kg-promotion-no-tag.db")
    try:
        for index in range(3):
            _insert_chunk(
                store,
                chunk_id=f"chunk-{index}",
                content=f"Ofir Levi appeared in planning note {index}.",
                raw_entities=[{"name": "Ofir Levi", "type": "person", "relation": "participant"}],
                tags=["planning"],
            )

        stats = promote_chunk_raw_entities(store, "chunk-0", entity_type="person")

        assert stats == {
            "chunks_scanned": 1,
            "candidates": 0,
            "entities_promoted": 0,
            "aliases_created": 0,
            "chunks_linked": 0,
        }
        assert store.resolve_entity("Ofir Levi") is None
    finally:
        store.close()


def test_chunk_promotion_excludes_archived_identity_tag_chunks(tmp_path):
    from brainlayer.kg_promotion import promote_chunk_raw_entities

    store = VectorStore(tmp_path / "kg-promotion-archived.db")
    try:
        tag = "alex-sample-identification"
        _insert_chunk(
            store,
            chunk_id="active",
            content="Alex Sample is the speakers workshop coach.",
            raw_entities=[{"name": "Alex Sample", "type": "person", "relation": "coach"}],
            tags=[tag],
        )
        _insert_chunk(
            store,
            chunk_id="archived",
            content="היי אלכס, archived note",
            raw_entities=[{"name": "אלכס", "type": "person", "relation": "recipient"}],
            tags=[tag],
        )
        store.conn.cursor().execute("UPDATE chunks SET archived_at = '2026-05-18T00:00:00Z' WHERE id = 'archived'")

        stats = promote_chunk_raw_entities(store, "active", entity_type="person")

        entity = store.resolve_entity("Alex Sample")
        assert entity is None
        assert stats["entities_promoted"] == 0
        assert stats["chunks_scanned"] == 1
    finally:
        store.close()


def test_non_latin_promoted_entity_id_uses_hash_fallback(tmp_path):
    from brainlayer.kg_promotion import promote_raw_entity_identities

    store = VectorStore(tmp_path / "kg-promotion-non-latin.db")
    try:
        for index in range(3):
            _insert_chunk(
                store,
                chunk_id=f"chunk-{index}",
                content=f"משה לוי appeared in planning note {index}.",
                raw_entities=[{"name": "משה לוי", "type": "person", "relation": "participant"}],
                tags=["planning"],
            )

        promote_raw_entity_identities(store, entity_type="person")

        entity = store.resolve_entity("משה לוי")
        assert entity is not None
        assert entity["id"].startswith("promoted-person-")
        assert entity["id"] != "promoted-person-"
    finally:
        store.close()


def test_dry_run_alias_count_excludes_canonical_name(tmp_path):
    from brainlayer.kg_promotion import promote_raw_entity_identities

    store = VectorStore(tmp_path / "kg-promotion-dry-run.db")
    try:
        tag = "alex-sample-identification"
        _insert_chunk(
            store,
            chunk_id="chunk-en",
            content="Alex Sample coached Noa.",
            raw_entities=[{"name": "Alex Sample", "type": "person", "relation": "coach"}],
            tags=[tag],
        )
        _insert_chunk(
            store,
            chunk_id="chunk-he",
            content="היי אלכס",
            raw_entities=[{"name": "אלכס", "type": "person", "relation": "recipient"}],
            tags=[tag],
        )

        stats = promote_raw_entity_identities(store, entity_type="person", dry_run=True)

        assert stats["aliases_created"] == 2
    finally:
        store.close()


def test_raw_entity_promotion_is_idempotent(tmp_path):
    from brainlayer.kg_promotion import promote_raw_entity_identities

    store = VectorStore(tmp_path / "kg-promotion-idempotent.db")
    try:
        tag = "alex-sample-identification"
        for chunk_id, surface in (("chunk-en", "Alex Sample"), ("chunk-he", "אלכס")):
            _insert_chunk(
                store,
                chunk_id=chunk_id,
                content=f"{surface} workshop context",
                raw_entities=[{"name": surface, "type": "person", "relation": "participant"}],
                tags=[tag],
            )

        first = promote_raw_entity_identities(store, entity_type="person")
        second = promote_raw_entity_identities(store, entity_type="person")
        entity = store.resolve_entity("Alex Sample")

        assert first["entities_promoted"] == 1
        assert first["chunks_linked"] == 2
        assert second["entities_promoted"] == 0
        assert second["aliases_created"] == 0
        assert second["chunks_linked"] == 0
        assert len(store.get_entity_chunks(entity["id"], limit=10, include_audit=True)) == 2
    finally:
        store.close()
