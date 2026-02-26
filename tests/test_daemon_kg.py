"""Tests for Knowledge Graph HTTP API endpoints in the daemon."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked vector_store and embedding_model."""
    import brainlayer.daemon as daemon_mod

    # Mock the global state
    mock_store = MagicMock()
    mock_store.count.return_value = 100
    mock_model = MagicMock()
    mock_model.embed_query.return_value = [0.1] * 1024

    # Set globals
    daemon_mod.vector_store = mock_store
    daemon_mod.embedding_model = mock_model

    client = TestClient(daemon_mod.app, raise_server_exceptions=False)
    yield client, mock_store, mock_model

    # Cleanup
    daemon_mod.vector_store = None
    daemon_mod.embedding_model = None


class TestDigestEndpoint:
    """POST /digest tests."""

    def test_digest_success(self, client):
        client, mock_store, mock_model = client

        with patch("brainlayer.pipeline.digest.digest_content") as mock_digest:
            mock_digest.return_value = {
                "digest_id": "digest-abc123",
                "entities": [{"name": "Alice", "entity_type": "person"}],
                "relations": [],
                "sentiment": {"polarity": "neutral"},
                "action_items": [],
                "decisions": [],
                "questions": [],
                "stats": {"entities": 1},
            }
            resp = client.post(
                "/digest",
                json={"content": "Alice can't do Sundays", "participants": ["Alice"]},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["digest_id"] == "digest-abc123"
            assert len(data["entities"]) == 1

    def test_digest_empty_content(self, client):
        client, _, _ = client

        with patch("brainlayer.pipeline.digest.digest_content") as mock_digest:
            mock_digest.side_effect = ValueError("content must be non-empty")
            resp = client.post("/digest", json={"content": ""})
            assert resp.status_code == 400


class TestPersonEndpoint:
    """GET /person/{name} tests."""

    def test_person_found(self, client):
        client, mock_store, mock_model = client

        with patch("brainlayer.pipeline.digest.entity_lookup") as mock_lookup:
            mock_lookup.return_value = {
                "id": "ent-123",
                "name": "Alice",
                "entity_type": "person",
                "metadata": {"role": "client"},
            }
            mock_store.get_entity_chunks.return_value = [
                {
                    "chunk_id": "c1",
                    "content": "Alice prefers mornings",
                    "content_type": "note",
                    "relevance": 0.9,
                }
            ]
            resp = client.get("/person/Alice")
            assert resp.status_code == 200
            data = resp.json()
            assert data["entity"]["name"] == "Alice"
            assert len(data["memories"]) == 1

    def test_person_not_found(self, client):
        client, _, _ = client

        with patch("brainlayer.pipeline.digest.entity_lookup") as mock_lookup:
            mock_lookup.return_value = None
            resp = client.get("/person/Nobody")
            assert resp.status_code == 404

    def test_person_with_context(self, client):
        client, mock_store, mock_model = client

        with patch("brainlayer.pipeline.digest.entity_lookup") as mock_lookup:
            mock_lookup.return_value = {
                "id": "ent-123",
                "name": "Alice",
                "entity_type": "person",
            }
            mock_store.hybrid_search.return_value = {
                "documents": [["Alice said she's free Tuesday"]],
                "metadatas": [[{"content_type": "user_message", "created_at": "2026-02-20", "summary": "scheduling"}]],
            }
            resp = client.get("/person/Alice?context=scheduling")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["memories"]) == 1


class TestStoreEndpoint:
    """POST /store tests."""

    def test_store_success(self, client):
        client, _, _ = client

        with patch("brainlayer.store.store_memory") as mock_store_fn:
            mock_store_fn.return_value = {
                "chunk_id": "manual-abc",
                "related": [],
            }
            resp = client.post(
                "/store",
                json={"content": "Alice prefers morning meetings", "type": "learning"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["chunk_id"] == "manual-abc"


class TestEntityEndpoints:
    """GET/PATCH /entity/{id} tests."""

    def test_get_entity(self, client):
        client, mock_store, _ = client

        mock_store.get_entity.return_value = {
            "id": "ent-123",
            "name": "Alice",
            "entity_type": "person",
            "metadata": '{"role": "client"}',
        }
        mock_store.get_entity_relations.return_value = [
            {
                "relation_type": "works_with",
                "target_name": "Bob",
                "target_type": "person",
                "direction": "outgoing",
                "confidence": 0.9,
            }
        ]
        mock_store.get_entity_chunks.return_value = [
            {"chunk_id": "c1", "content": "Alice meeting notes", "relevance": 0.8}
        ]

        resp = client.get("/entity/ent-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity"]["name"] == "Alice"
        assert len(data["relations"]) == 1
        assert len(data["chunks"]) == 1

    def test_get_entity_not_found(self, client):
        client, mock_store, _ = client
        mock_store.get_entity.return_value = None

        resp = client.get("/entity/nonexistent")
        assert resp.status_code == 404

    def test_update_entity_metadata(self, client):
        client, mock_store, _ = client

        mock_store.get_entity.return_value = {
            "id": "ent-123",
            "name": "Alice",
            "entity_type": "person",
            "metadata": '{"role": "client"}',
        }

        resp = client.patch(
            "/entity/ent-123",
            json={"metadata": {"blocked_weekdays": ["Sunday"], "preferred_time": "morning"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Sunday" in data["metadata"]["blocked_weekdays"]
        assert data["metadata"]["role"] == "client"  # preserved
        assert data["metadata"]["preferred_time"] == "morning"  # added
