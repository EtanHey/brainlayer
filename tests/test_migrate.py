from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from brainlayer import migrate
from brainlayer.embeddings import MAX_PASSAGE_CHARS
from brainlayer.runtime_store import ReadonlyStore


@pytest.mark.parametrize("content", ["ordinary migration passage", (" " * 32 + "return value\n") * 1000])
def test_chromadb_reembedding_bounds_model_input_without_changing_stored_content(tmp_path, monkeypatch, content):
    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir()
    collection = MagicMock(name="collection")
    collection.name = "source"
    collection.count.return_value = 1
    collection.get.return_value = {
        "ids": ["migrated"],
        "documents": [content],
        "metadatas": [{}],
        "embeddings": [None],
    }
    client = SimpleNamespace(list_collections=lambda: [collection])
    monkeypatch.setattr(migrate, "CHROMADB_AVAILABLE", True)
    monkeypatch.setattr(migrate, "CHROMADB_PATH", chroma_path)
    monkeypatch.setattr(migrate, "chromadb", SimpleNamespace(PersistentClient=lambda **kwargs: client), raising=False)
    monkeypatch.setattr(migrate, "Settings", lambda **kwargs: None, raising=False)
    model = MagicMock()
    model._load_model.return_value.encode.return_value = np.array([[0.25] * 1024])
    monkeypatch.setattr(migrate, "get_embedding_model", lambda: model)

    target = tmp_path / "migrated.db"
    assert migrate.migrate_from_chromadb(target) is True

    assert model._load_model.return_value.encode.call_args.args[0] == [content[:MAX_PASSAGE_CHARS]]
    store = ReadonlyStore(target)
    try:
        assert store.conn.execute("SELECT content FROM chunks WHERE id='migrated'").fetchone() == (content,)
        for table in ("chunk_vectors", "chunk_vectors_binary"):
            assert store.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE chunk_id='migrated'").fetchone() == (1,)
    finally:
        store.close()
