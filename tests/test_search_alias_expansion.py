import json
from unittest.mock import MagicMock, patch

import pytest

from brainlayer._helpers import serialize_f32
from brainlayer.mcp.search_handler import _brain_search
from brainlayer.vector_store import VectorStore


def _embed(seed_text: str) -> list[float]:
    seed = (sum(ord(ch) for ch in seed_text[:40]) % 97) / 1000.0
    return [seed + (i / 10000.0) for i in range(1024)]


def _insert_chunk(store: VectorStore, *, chunk_id: str, content: str) -> None:
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, summary, tags, created_at
        ) VALUES (?, ?, '{}', 'test.jsonl', 'brainlayer', 'assistant_text', ?, 'manual', ?, ?, ?)""",
        (
            chunk_id,
            content,
            len(content),
            content,
            json.dumps(["aliases"]),
            "2026-04-30T10:00:00Z",
        ),
    )
    cursor.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_f32(_embed("distant vector"))),
    )


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.embed_query.return_value = _embed("query vector")
    return model


@pytest.mark.asyncio
async def test_brain_search_expands_lexical_defense_variants(tmp_path, mock_model):
    store = VectorStore(tmp_path / "lexical-defense.db")
    try:
        _insert_chunk(store, chunk_id="chunk-hershkovits", content="Met with Hershkovits about the release plan.")
        store.build_binary_index()
        cursor = store.conn.cursor()
        cursor.execute("DELETE FROM chunk_vectors")
        cursor.execute("DELETE FROM chunk_vectors_binary")

        with (
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=store),
            patch("brainlayer.mcp.search_handler._get_embedding_model", return_value=mock_model),
        ):
            _, structured = await _brain_search(query="Hershkovitz", project="brainlayer", detail="compact")

        assert structured["total"] == 1
        assert structured["results"][0]["chunk_id"] == "chunk-hershkovits"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_brain_search_expands_kg_aliases_by_normalized_surface(tmp_path, mock_model):
    store = VectorStore(tmp_path / "kg-aliases.db")
    try:
        _insert_chunk(store, chunk_id="chunk-stalker", content="stalker_golem pipeline note for overnight run.")
        store.build_binary_index()
        cursor = store.conn.cursor()
        cursor.execute("DELETE FROM chunk_vectors")
        cursor.execute("DELETE FROM chunk_vectors_binary")

        store.upsert_entity("entity-stalker", "project", "stalker-golem")
        store.add_entity_alias("stalker_golem", "entity-stalker", alias_type="normalized")

        with (
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=store),
            patch("brainlayer.mcp.search_handler._get_embedding_model", return_value=mock_model),
        ):
            _, structured = await _brain_search(query="stalkerGolem", project="brainlayer", detail="compact")

        assert structured["total"] == 1
        assert structured["results"][0]["chunk_id"] == "chunk-stalker"
    finally:
        store.close()
