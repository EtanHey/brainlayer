"""Retiring digest callbacks preserves old metadata and local new-chunk fields."""

from unittest.mock import MagicMock

from brainlayer.pipeline.digest import digest_content
from brainlayer.vector_store import VectorStore


def test_brain_digest_retired_callback_keeps_historical_metadata_and_new_local_content(tmp_path):
    """Exercise real fixture rows; the old callback must never run or write tags."""
    store = VectorStore(tmp_path / "digest.db")
    try:
        store.upsert_chunks(
            [
                {
                    "id": "historical-fixture",
                    "content": "Synthetic historical enrichment retained for retrieval.",
                    "metadata": {},
                    "source_file": "fixture.jsonl",
                    "content_type": "user_message",
                    "char_count": 54,
                    "source": "digest",
                }
            ],
            [[0.0, 1.0] + [0.0] * 1022],
        )
        store.update_enrichment(
            "historical-fixture",
            summary="Historical summary",
            tags=["historical-model-tag"],
            importance=7,
            intent="designing",
        )
        historical_query = "SELECT summary,tags,importance,intent,enriched_at,enrich_status FROM chunks WHERE id = 'historical-fixture'"
        historical = store.conn.execute(historical_query).fetchone()
        callback = MagicMock(
            return_value={
                "topics": ["new-model-tag"],
                "activity": "act:designing",
                "domains": ["dom:python"],
                "confidence": 1.0,
                "status": "enriched",
            }
        )
        result = digest_content(
            content="Person Alpha chose BrainLayer. TODO: write local tests.",
            store=store,
            embed_fn=lambda _text: [1.0] + [0.0] * 1023,
            title="Synthetic local decision",
            participants=["Person Alpha"],
            faceted_enrich_fn=callback,
        )
        callback.assert_not_called()
        assert result["enrichment"] == {"status": "retired", "reason": "cloud_enrichment_retired"}
        assert result["tags"] == []
        assert result["summary"] == "Synthetic local decision"
        assert result["entities"] and result["decisions"] and result["action_items"]
        row = store.conn.execute(
            "SELECT tags,intent,summary,enriched_at,sentiment_label FROM chunks WHERE id = ?", (result["digest_id"],)
        ).fetchone()
        assert row[:4] == (None, None, None, None)
        assert row[4] is not None
        assert store.conn.execute(historical_query).fetchone() == historical
    finally:
        store.close()
