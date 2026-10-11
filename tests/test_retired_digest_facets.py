"""The built-in digest tagger cannot construct a cloud client."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from brainlayer.pipeline import digest
from brainlayer.pipeline.sanitize import Sanitizer
from brainlayer.vector_store import VectorStore

pytestmark = pytest.mark.retired_enrichment


@pytest.fixture
def legacy_settings(monkeypatch):
    """Old activation flags and synthetic credentials must not reactivate Gemini."""
    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-retirement-key")
    monkeypatch.setenv("BRAINLAYER_LLM_ENTITY_EXTRACTION", "0")
    monkeypatch.setenv("BRAINLAYER_DIGEST_GEMINI_MODEL", "synthetic-legacy-model")
    monkeypatch.setattr(Sanitizer, "from_env", MagicMock(return_value=object()))
    monkeypatch.setattr(
        digest,
        "build_external_prompt",
        MagicMock(return_value=("synthetic prompt", SimpleNamespace(pii_detected=False))),
        raising=False,
    )


def test_brain_digest_retired_faceted_helper_never_constructs_client(legacy_settings):
    """Direct legacy calls return retirement before building a model client."""
    result = digest._default_faceted_enrich(content="synthetic content", project="fixture", title=None, participants=[])
    assert result == {"status": "retired", "reason": "cloud_enrichment_retired"}


def test_brain_digest_retired_facets_keep_local_persistence(legacy_settings, tmp_path):
    """The actual default pipeline stores local content/vectors and no model metadata."""
    store = VectorStore(tmp_path / "digest.db")
    try:
        result = digest.digest_content(
            content="Person Alpha chose BrainLayer. TODO: write local tests. Why keep local embeddings?",
            store=store,
            embed_fn=lambda _text: [1.0] + [0.0] * 1023,
            title="Synthetic decision",
            participants=["Person Alpha"],
        )
        assert result["summary"] == "Synthetic decision"
        assert result["enrichment"] == {"status": "retired", "reason": "cloud_enrichment_retired"}
        assert result["decisions"]
        assert result["action_items"]
        assert result["questions"]
        assert {entity["name"].lower() for entity in result["entities"]} >= {"person alpha", "brainlayer"}
        row = store.conn.execute(
            "SELECT tags, intent, summary, enriched_at FROM chunks WHERE id = ?", (result["digest_id"],)
        ).fetchone()
        assert row == (None, None, None, None)
        assert store.conn.execute("SELECT 1 FROM chunk_vectors WHERE chunk_id = ?", (result["digest_id"],)).fetchone()
    finally:
        store.close()


def test_dead_faceted_parser_is_absent():
    assert not hasattr(digest, "_parse_faceted_enrichment")
