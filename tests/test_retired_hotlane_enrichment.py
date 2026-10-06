"""Legacy enrichment options cannot stop local hotlane vector progress."""

import importlib
from unittest.mock import MagicMock

import pytest

from brainlayer.vector_store import VectorStore

pytestmark = pytest.mark.retired_enrichment


@pytest.mark.parametrize("split", [False, True])
def test_hotlane_retired_enrichment_preserves_local_vectors(tmp_path, caplog, split):
    hotlane = importlib.import_module("scripts.hotlane_brainbar_daemon")
    db = tmp_path / "synthetic.db"
    store = VectorStore(db)
    store.conn.cursor().execute(
        "INSERT INTO chunks (id, content, metadata, source_file, source, created_at) "
        "VALUES ('synthetic-hot', 'Local embedding fixture', '{}', 'brainbar-store', 'mcp', '2099-01-01')"
    )
    callback = MagicMock()
    options = dict(
        embed_fn=lambda _text: [0.125] * 1024,
        recent_limit=5,
        backlog_batch=0,
        enrich_limit=25,
        enrich_since_hours=8760,
        enrich_fn=callback,
    )
    try:
        if split:
            store.close()
            result = hotlane._run_split_cycle(db_path=db, vector_store_cls=VectorStore, **options)
            store = VectorStore(db)
        else:
            result = hotlane.run_cycle(store=store, **options)
        callback.assert_not_called()
        assert result == hotlane.CycleResult(embedded=1)
        assert list(store.conn.cursor().execute("SELECT id FROM chunk_vectors_rowids")) == [("synthetic-hot",)]
        assert list(store.conn.cursor().execute("SELECT summary, tags, enriched_at FROM chunks")) == [
            (None, None, None)
        ]
        assert "enrichment has been retired" in caplog.text
    finally:
        store.close()
