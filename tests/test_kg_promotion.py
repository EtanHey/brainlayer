import json

from brainlayer.pipeline.digest import entity_lookup
from brainlayer.vector_store import VectorStore


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
        tag = "michal-hershkovits-identification"
        _insert_chunk(
            store,
            chunk_id="chunk-en",
            content="Michal Hershkovits coached Etan for the speakers workshop.",
            raw_entities=[
                {"name": "Michal Hershkovits", "type": "person", "relation": "coach"},
                {"name": "TechGym Speakers Workshop", "type": "concept", "relation": "context"},
            ],
            tags=[tag, "workshop-coaching"],
        )
        _insert_chunk(
            store,
            chunk_id="chunk-he",
            content="היי מיכל, אשמח להשתתף בסדנא",
            raw_entities=[{"name": "מיכל", "type": "person", "relation": "recipient"}],
            tags=[tag, "hebrew"],
        )

        stats = promote_raw_entity_identities(store, entity_type="person")

        entity = store.resolve_entity("Michal Hershkovits")
        assert entity is not None
        assert entity["entity_type"] == "person"
        assert entity["name"] == "Michal Hershkovits"
        hebrew_entity = store.resolve_entity("מיכל")
        assert hebrew_entity is not None
        assert hebrew_entity["id"] == entity["id"]

        aliases = {(row["alias"], row["alias_type"]) for row in store.get_entity_aliases(entity["id"])}
        assert ("מיכל", "raw_surface") in aliases
        assert ("michal-hershkovits-identification", "identity_tag") in aliases

        linked = {row["chunk_id"] for row in store.get_entity_chunks(entity["id"], limit=10, include_audit=True)}
        assert linked == {"chunk-en", "chunk-he"}
        assert stats["entities_promoted"] == 1
        assert stats["aliases_created"] >= 2
        assert stats["chunks_linked"] == 2
    finally:
        store.close()


def test_promoted_alias_resolves_through_entity_lookup_for_person(tmp_path):
    from brainlayer.kg_promotion import promote_raw_entity_identities

    store = VectorStore(tmp_path / "kg-promotion-lookup.db")
    try:
        tag = "michal-hershkovits-identification"
        _insert_chunk(
            store,
            chunk_id="chunk-en",
            content="Michal Hershkovits coached Etan for the speakers workshop.",
            raw_entities=[{"name": "Michal Hershkovits", "type": "person", "relation": "coach"}],
            tags=[tag],
        )
        _insert_chunk(
            store,
            chunk_id="chunk-he",
            content="היי מיכל, אשמח להשתתף בסדנא",
            raw_entities=[{"name": "מיכל", "type": "person", "relation": "recipient"}],
            tags=[tag],
        )

        promote_raw_entity_identities(store, entity_type="person")

        by_full_name = entity_lookup(
            query="Michal Hershkovits",
            store=store,
            embed_fn=lambda _: [0.0] * 1024,
            entity_type="person",
        )
        by_hebrew_alias = entity_lookup(
            query="מיכל",
            store=store,
            embed_fn=lambda _: [0.0] * 1024,
            entity_type="person",
        )

        assert by_full_name is not None
        assert by_hebrew_alias is not None
        assert by_hebrew_alias["id"] == by_full_name["id"]
        assert by_hebrew_alias["name"] == "Michal Hershkovits"
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
        tag = "michal-hershkovits-identification"
        _insert_chunk(
            store,
            chunk_id="identity-tag-only",
            content="Can't find Michal Hershkovits anywhere, but she is the speakers workshop coach.",
            raw_entities=[],
            tags=[tag, "workshop-coaching"],
        )
        _insert_chunk(
            store,
            chunk_id="raw-spelling-drift",
            content="Slide example uses Michal Herskovits as the search target.",
            raw_entities=[{"name": "Michal Herskovits", "type": "person", "relation": "search target"}],
            tags=["presentation-dev"],
        )
        _insert_chunk(
            store,
            chunk_id="raw-hebrew",
            content="היי מיכל, אשמח להשתתף בסדנא",
            raw_entities=[{"name": "מיכל", "type": "person", "relation": "recipient"}],
            tags=["hebrew"],
        )

        stats = promote_raw_entity_identities(store, entity_type="person")

        entity = store.resolve_entity("Michal Hershkovits")
        assert entity is not None
        spelling_drift_entity = store.resolve_entity("Michal Herskovits")
        assert spelling_drift_entity is not None
        assert spelling_drift_entity["id"] == entity["id"]
        hebrew_entity = store.resolve_entity("מיכל")
        assert hebrew_entity is not None
        assert hebrew_entity["id"] == entity["id"]
        assert stats["entities_promoted"] == 1
    finally:
        store.close()


def test_raw_entity_promotion_is_idempotent(tmp_path):
    from brainlayer.kg_promotion import promote_raw_entity_identities

    store = VectorStore(tmp_path / "kg-promotion-idempotent.db")
    try:
        tag = "michal-hershkovits-identification"
        for chunk_id, surface in (("chunk-en", "Michal Hershkovits"), ("chunk-he", "מיכל")):
            _insert_chunk(
                store,
                chunk_id=chunk_id,
                content=f"{surface} workshop context",
                raw_entities=[{"name": surface, "type": "person", "relation": "participant"}],
                tags=[tag],
            )

        first = promote_raw_entity_identities(store, entity_type="person")
        second = promote_raw_entity_identities(store, entity_type="person")
        entity = store.resolve_entity("Michal Hershkovits")

        assert first["entities_promoted"] == 1
        assert first["chunks_linked"] == 2
        assert second["entities_promoted"] == 0
        assert second["aliases_created"] == 0
        assert second["chunks_linked"] == 0
        assert len(store.get_entity_chunks(entity["id"], limit=10, include_audit=True)) == 2
    finally:
        store.close()
