"""Tests for 6PM entity upgrades — per-person memory scoping, brain_get_person, entity-tagged store.

4 features tested:
1. Person profile schema convention on metadata JSON
2. Per-person memory scoping (search(entity_id=...))
3. brain_get_person composite logic (entity_lookup + scoped memories)
4. Entity-tagged brain_store (store_memory(entity_id=...))
"""

import json

import pytest

from brainlayer.vector_store import VectorStore, serialize_f32

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore for testing."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def mock_embed():
    """Mock embedding function that returns a deterministic 1024-dim vector."""

    def _embed(text: str) -> list[float]:
        seed = sum(ord(c) for c in text[:50]) % 100
        return [float(seed + i) / 1000.0 for i in range(1024)]

    return _embed


@pytest.fixture
def person_entity(store, mock_embed):
    """Create a person entity with 6PM-style profile metadata."""
    entity_id = store.upsert_entity(
        entity_id="person-avi-simon",
        entity_type="person",
        name="Avi Simon",
        metadata={
            "hard_constraints": {
                "blocked_weekdays": ["SAT"],
                "not_before": "09:00",
                "not_after": "18:00",
            },
            "preferences": {
                "preferred_time_of_day": "MORNING",
                "duration_minutes": 30,
            },
            "contact_info": {
                "email": "avi@6pm.ai",
                "phone": "+972-54-1234567",
            },
        },
    )
    return entity_id


@pytest.fixture
def person_with_chunks(store, mock_embed, person_entity):
    """Create a person entity with linked chunks (messages/memories)."""
    entity_id = person_entity

    # Create some chunks and link them to the person
    chunks = [
        ("Mondays are impossible for me, I have team standup all morning", "user_message"),
        ("Best time for me is after 15:00 on weekdays", "user_message"),
        ("I prefer video calls over phone calls", "user_message"),
        ("Meeting with Avi went well, he confirmed Thursday 2pm works", "assistant_text"),
    ]

    chunk_ids = []
    now = "2026-02-26T10:00:00Z"
    for content, content_type in chunks:
        chunk_id = f"test-{len(chunk_ids)}"
        embedding = mock_embed(content)
        cursor = store.conn.cursor()
        cursor.execute(
            """INSERT INTO chunks (id, content, metadata, source_file, project,
               content_type, value_type, char_count, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk_id,
                content,
                json.dumps({}),
                "test.jsonl",
                "6pm-mini",
                content_type,
                "HIGH",
                len(content),
                "manual",
                now,
            ),
        )
        cursor.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, serialize_f32(embedding)),
        )
        store.link_entity_chunk(entity_id, chunk_id, relevance=1.0, context="test message")
        chunk_ids.append(chunk_id)

    # Also create an unlinked chunk (should NOT appear in entity-scoped searches)
    cursor = store.conn.cursor()
    unlinked_embedding = mock_embed("Random unrelated message about weather")
    cursor.execute(
        """INSERT INTO chunks (id, content, metadata, source_file, project,
           content_type, value_type, char_count, source, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "test-unlinked",
            "Random unrelated message about weather",
            json.dumps({}),
            "test.jsonl",
            "6pm-mini",
            "user_message",
            "HIGH",
            38,
            "manual",
            now,
        ),
    )
    cursor.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        ("test-unlinked", serialize_f32(unlinked_embedding)),
    )

    return entity_id, chunk_ids


# ── 1. Person Profile Schema Convention ────────────────────────


class TestPersonProfileSchema:
    """Verify the person profile metadata schema convention."""

    def test_person_entity_stores_hard_constraints(self, store, person_entity):
        """Person entity metadata includes hard_constraints."""
        entity = store.get_entity(person_entity)
        assert entity is not None
        meta = entity["metadata"]
        assert "hard_constraints" in meta
        assert meta["hard_constraints"]["blocked_weekdays"] == ["SAT"]
        assert meta["hard_constraints"]["not_before"] == "09:00"
        assert meta["hard_constraints"]["not_after"] == "18:00"

    def test_person_entity_stores_preferences(self, store, person_entity):
        """Person entity metadata includes preferences."""
        entity = store.get_entity(person_entity)
        meta = entity["metadata"]
        assert "preferences" in meta
        assert meta["preferences"]["preferred_time_of_day"] == "MORNING"
        assert meta["preferences"]["duration_minutes"] == 30

    def test_person_entity_stores_contact_info(self, store, person_entity):
        """Person entity metadata includes contact_info."""
        entity = store.get_entity(person_entity)
        meta = entity["metadata"]
        assert "contact_info" in meta
        assert meta["contact_info"]["email"] == "avi@6pm.ai"

    def test_person_entity_type_is_person(self, store, person_entity):
        """Entity type is 'person'."""
        entity = store.get_entity(person_entity)
        assert entity["entity_type"] == "person"
        assert entity["name"] == "Avi Simon"

    def test_person_entity_upsert_updates_metadata(self, store, person_entity):
        """Upserting same entity updates metadata without creating duplicate."""
        # Upsert with updated constraints
        updated_id = store.upsert_entity(
            entity_id="person-avi-simon",
            entity_type="person",
            name="Avi Simon",
            metadata={
                "hard_constraints": {
                    "blocked_weekdays": ["SAT", "FRI"],
                    "not_before": "10:00",
                    "not_after": "17:00",
                },
                "preferences": {
                    "preferred_time_of_day": "AFTERNOON",
                },
            },
        )

        # Should be the same entity (upsert, not create)
        entity = store.get_entity(updated_id)
        assert entity["metadata"]["hard_constraints"]["blocked_weekdays"] == ["SAT", "FRI"]
        assert entity["metadata"]["preferences"]["preferred_time_of_day"] == "AFTERNOON"


# ── 2. Per-Person Memory Scoping ──────────────────────────────


class TestPerPersonMemoryScoping:
    """Test entity_id filtering in search and hybrid_search."""

    def test_search_with_entity_id_filters_results(self, store, mock_embed, person_with_chunks):
        """search(entity_id=...) only returns chunks linked to that entity."""
        entity_id, chunk_ids = person_with_chunks
        embedding = mock_embed("what time works for meeting")

        results = store.search(
            query_embedding=embedding,
            n_results=20,
            entity_id=entity_id,
        )

        # Should only get the 4 linked chunks, not the unlinked one
        result_ids = results["ids"][0]
        assert "test-unlinked" not in result_ids
        assert len(result_ids) <= 4

    def test_search_without_entity_id_returns_all(self, store, mock_embed, person_with_chunks):
        """search() without entity_id returns all chunks (including unlinked)."""
        entity_id, chunk_ids = person_with_chunks
        embedding = mock_embed("what time works for meeting")

        results = store.search(
            query_embedding=embedding,
            n_results=20,
        )

        result_ids = results["ids"][0]
        # Should include both linked and unlinked chunks
        assert len(result_ids) == 5  # 4 linked + 1 unlinked

    def test_hybrid_search_with_entity_id(self, store, mock_embed, person_with_chunks):
        """hybrid_search(entity_id=...) scopes both semantic and FTS to entity chunks."""
        entity_id, chunk_ids = person_with_chunks
        embedding = mock_embed("meeting schedule")

        results = store.hybrid_search(
            query_embedding=embedding,
            query_text="meeting",
            n_results=20,
            entity_id=entity_id,
        )

        result_ids = results["ids"][0]
        assert "test-unlinked" not in result_ids

    def test_entity_id_with_no_linked_chunks_returns_empty(self, store, mock_embed):
        """Searching with entity_id that has no linked chunks returns empty."""
        # Create entity with no chunks
        entity_id = store.upsert_entity(
            entity_id="person-nobody",
            entity_type="person",
            name="Nobody",
            metadata={},
        )

        embedding = mock_embed("test query")
        results = store.search(
            query_embedding=embedding,
            n_results=10,
            entity_id=entity_id,
        )

        assert len(results["ids"][0]) == 0

    def test_text_search_with_entity_id(self, store, mock_embed, person_with_chunks):
        """Text search with entity_id only returns entity-linked chunks."""
        entity_id, chunk_ids = person_with_chunks

        results = store.search(
            query_text="impossible",
            n_results=20,
            entity_id=entity_id,
        )

        # "Mondays are impossible" is linked; "Random unrelated" is not
        result_ids = results["ids"][0]
        assert "test-unlinked" not in result_ids
        assert len(result_ids) >= 1


# ── 3. Entity-Tagged Store ────────────────────────────────────


class TestEntityTaggedStore:
    """Test store_memory(entity_id=...) auto-linking."""

    def test_store_with_entity_id_links_chunk(self, store, mock_embed, person_entity):
        """store_memory with entity_id links the new chunk to the entity."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Avi mentioned he can't do Mondays at all",
            memory_type="note",
            entity_id=person_entity,
        )

        chunk_id = result["id"]

        # Verify the chunk is linked to the entity
        entity_chunks = store.get_entity_chunks(person_entity)
        chunk_ids = [c["chunk_id"] for c in entity_chunks]
        assert chunk_id in chunk_ids

    def test_store_without_entity_id_no_link(self, store, mock_embed, person_entity):
        """store_memory without entity_id does NOT link to any entity."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="General note about something unrelated",
            memory_type="note",
        )

        chunk_id = result["id"]

        # Should NOT be linked to the person entity
        entity_chunks = store.get_entity_chunks(person_entity)
        chunk_ids = [c["chunk_id"] for c in entity_chunks]
        assert chunk_id not in chunk_ids

    def test_stored_entity_linked_chunk_is_searchable_by_entity(self, store, mock_embed, person_entity):
        """After storing with entity_id, the chunk appears in entity-scoped search."""
        from brainlayer.store import store_memory

        store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Avi prefers Tuesday afternoons for long meetings",
            memory_type="note",
            entity_id=person_entity,
        )

        # Search scoped to entity
        embedding = mock_embed("when does Avi prefer meetings")
        results = store.search(
            query_embedding=embedding,
            n_results=10,
            entity_id=person_entity,
        )

        # Should find the stored memory
        assert len(results["ids"][0]) >= 1
        found_contents = results["documents"][0]
        assert any("Tuesday afternoons" in c for c in found_contents)

    def test_store_entity_link_has_context(self, store, mock_embed, person_entity):
        """Entity link from store has descriptive context."""
        from brainlayer.store import store_memory

        store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Test memory",
            memory_type="learning",
            entity_id=person_entity,
        )

        entity_chunks = store.get_entity_chunks(person_entity)
        assert len(entity_chunks) >= 1
        assert "brain_store" in entity_chunks[0]["context"]


# ── 4. brain_get_person Composite Logic (Unit-Level) ──────────


class TestBrainGetPersonLogic:
    """Test the entity_lookup + scoped memories logic used by brain_get_person."""

    def test_entity_lookup_finds_person(self, store, mock_embed, person_entity):
        """entity_lookup finds a person by name."""
        from brainlayer.pipeline.digest import entity_lookup

        # Need to add embedding to the entity for semantic search
        entity_embedding = mock_embed("Avi Simon person")
        store.conn.cursor().execute(
            "INSERT INTO kg_vec_entities (entity_id, embedding) VALUES (?, ?)",
            (person_entity, serialize_f32(entity_embedding)),
        )

        result = entity_lookup(
            query="Avi Simon",
            store=store,
            embed_fn=mock_embed,
            entity_type="person",
        )

        assert result is not None
        assert result["name"] == "Avi Simon"
        assert result["entity_type"] == "person"
        assert "hard_constraints" in result["metadata"]

    def test_entity_lookup_returns_metadata_with_constraints(self, store, mock_embed, person_entity):
        """entity_lookup result includes structured profile fields."""
        from brainlayer.pipeline.digest import entity_lookup

        entity_embedding = mock_embed("Avi Simon person")
        store.conn.cursor().execute(
            "INSERT INTO kg_vec_entities (entity_id, embedding) VALUES (?, ?)",
            (person_entity, serialize_f32(entity_embedding)),
        )

        result = entity_lookup(
            query="Avi Simon",
            store=store,
            embed_fn=mock_embed,
            entity_type="person",
        )

        assert result["metadata"]["hard_constraints"]["not_before"] == "09:00"
        assert result["metadata"]["preferences"]["preferred_time_of_day"] == "MORNING"

    def test_entity_chunks_ordered_by_relevance(self, store, mock_embed, person_with_chunks):
        """get_entity_chunks returns chunks ordered by relevance score."""
        entity_id, chunk_ids = person_with_chunks

        chunks = store.get_entity_chunks(entity_id, limit=10)
        assert len(chunks) == 4

        # All should have relevance 1.0 (set in fixture)
        for chunk in chunks:
            assert chunk["relevance"] == 1.0
            assert chunk["content"] is not None

    def test_get_entity_chunks_with_content(self, store, mock_embed, person_with_chunks):
        """get_entity_chunks returns full chunk content."""
        entity_id, chunk_ids = person_with_chunks

        chunks = store.get_entity_chunks(entity_id, limit=10)
        contents = [c["content"] for c in chunks]
        assert any("impossible" in c for c in contents)
        assert any("15:00" in c for c in contents)


# ── 5. Multiple Persons Isolation ─────────────────────────────


class TestMultiplePersonIsolation:
    """Verify that per-person scoping correctly isolates different people."""

    def test_two_persons_isolated_search(self, store, mock_embed):
        """Searching by entity_id for person A doesn't return person B's chunks."""
        # Create person A
        person_a_id = store.upsert_entity(
            entity_id="person-a",
            entity_type="person",
            name="Person A",
            metadata={"preferences": {"preferred_time_of_day": "MORNING"}},
        )

        # Create person B
        person_b_id = store.upsert_entity(
            entity_id="person-b",
            entity_type="person",
            name="Person B",
            metadata={"preferences": {"preferred_time_of_day": "EVENING"}},
        )

        # Create chunks for each
        now = "2026-02-26T10:00:00Z"
        for person_id, content, chunk_id in [
            (person_a_id, "Person A likes mornings", "chunk-a"),
            (person_b_id, "Person B prefers evenings", "chunk-b"),
        ]:
            embedding = mock_embed(content)
            cursor = store.conn.cursor()
            cursor.execute(
                """INSERT INTO chunks (id, content, metadata, source_file, project,
                   content_type, value_type, char_count, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (chunk_id, content, "{}", "test.jsonl", "test", "note", "HIGH", len(content), "manual", now),
            )
            cursor.execute(
                "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, serialize_f32(embedding)),
            )
            store.link_entity_chunk(person_id, chunk_id, relevance=1.0)

        # Search for person A's memories
        embedding = mock_embed("time preference")
        results_a = store.search(
            query_embedding=embedding,
            n_results=10,
            entity_id=person_a_id,
        )

        # Search for person B's memories
        results_b = store.search(
            query_embedding=embedding,
            n_results=10,
            entity_id=person_b_id,
        )

        # Person A should only see chunk-a
        assert "chunk-a" in results_a["ids"][0]
        assert "chunk-b" not in results_a["ids"][0]

        # Person B should only see chunk-b
        assert "chunk-b" in results_b["ids"][0]
        assert "chunk-a" not in results_b["ids"][0]
