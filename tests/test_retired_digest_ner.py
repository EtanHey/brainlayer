"""Digest and connect retain local entities without choosing an implicit LLM."""

from unittest.mock import MagicMock

import pytest

from brainlayer.pipeline import entity_extraction
from brainlayer.pipeline.batch_extraction import process_chunk
from brainlayer.pipeline.digest import digest_connect, digest_content
from brainlayer.vector_store import VectorStore

TEXT = "Person Alpha chose BrainLayer. TODO: write local tests. Why keep local embeddings?"
SEEDS = {"person": ["Person Alpha"], "project": ["BrainLayer"]}


@pytest.fixture(params=[None, "0", "1"])
def implicit_factory(request, monkeypatch):
    """Legacy flags and synthetic credentials cannot reactivate a producer."""
    if request.param is None:
        monkeypatch.delenv("BRAINLAYER_LLM_ENTITY_EXTRACTION", raising=False)
    else:
        monkeypatch.setenv("BRAINLAYER_LLM_ENTITY_EXTRACTION", request.param)
    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-retirement-key")
    caller = MagicMock(return_value='{"entities": [], "relations": []}')
    factory = MagicMock(return_value=caller)
    monkeypatch.setattr(entity_extraction, "_get_default_llm_caller", factory, raising=False)
    yield factory
    factory.assert_not_called()
    caller.assert_not_called()


@pytest.mark.parametrize("surface", ["combined", "batch"])
def test_brain_digest_retired_ner_keeps_seeds_and_cooccurrence(implicit_factory, surface):
    """Normal extraction stays useful with no caller supplied."""
    result = (
        entity_extraction.extract_entities_combined(TEXT, SEEDS)
        if surface == "combined"
        else process_chunk({"id": "synthetic-chunk", "content": TEXT}, seed_entities=SEEDS)
    )
    assert {entity.text.lower() for entity in result.entities} == {"person alpha", "brainlayer"}
    assert all(entity.source == "seed" for entity in result.entities)
    assert any(relation.relation_type == "co_occurs_with" for relation in result.relations)


def test_brain_digest_retired_ner_direct_implicit_call_errors(implicit_factory):
    """The compatibility helper reports retirement instead of selecting Gemini."""
    with pytest.raises(RuntimeError, match="Implicit LLM entity extraction has been retired"):
        entity_extraction.extract_entities_llm(TEXT)


def test_brain_digest_retired_ner_preserves_explicit_local_gliner(implicit_factory, monkeypatch):
    """Dispatch to a synthetic local model result without loading a real model."""
    start = TEXT.index("local tests")
    entity = entity_extraction.ExtractedEntity("local tests", "tool", start, start + 11, 0.8, "gliner")
    extractor = MagicMock(return_value=[entity])
    monkeypatch.setattr(entity_extraction, "extract_entities_gliner", extractor)
    result = entity_extraction.extract_entities_combined(TEXT, SEEDS, use_gliner=True)
    assert {item.text.lower() for item in result.entities} == {"person alpha", "brainlayer", "local tests"}
    extractor.assert_called_once_with(TEXT)


@pytest.mark.parametrize("surface", ["digest", "connect"])
def test_brain_digest_retired_ner_real_temporary_pipeline_keeps_local_entities(implicit_factory, tmp_path, surface):
    """Exercise actual local KG persistence and a connect proposal on fixture data."""
    store = VectorStore(tmp_path / "local-digest.db")
    try:
        kwargs = dict(
            content=TEXT,
            store=store,
            embed_fn=lambda _text: [1.0] + [0.0] * 1023,
            participants=["Person Alpha"],
            project="synthetic-project",
        )
        result = (
            digest_content(**kwargs, faceted_enrich_fn=lambda **_kwargs: {"status": "skipped"})
            if surface == "digest"
            else digest_connect(**kwargs)
        )
        entities = result["entities"] if surface == "digest" else result["extracted"]["entities"]
        assert {entity["name"].lower() for entity in entities} >= {"person alpha", "brainlayer"}
        if surface == "digest":
            assert any(relation["relation_type"] == "co_occurs_with" for relation in result["relations"])
            row = store.conn.execute("SELECT content FROM chunks WHERE id = ?", (result["digest_id"],)).fetchone()
            assert row[0] == TEXT
            assert store.conn.execute(
                "SELECT 1 FROM chunk_vectors WHERE chunk_id = ?", (result["digest_id"],)
            ).fetchone()
        else:
            assert result["status"] == "proposal"
            assert result["suggested_stores"][0]["content"] == TEXT
            assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    finally:
        store.close()


def test_explicit_fixture_callback_and_empty_input_remain_supported():
    """Offline JSON parsing callbacks survive removal of the built-in transport."""
    caller = MagicMock(return_value='{"entities": [], "relations": []}')
    assert entity_extraction.extract_entities_llm("", llm_caller=caller) == ([], [])
    caller.assert_not_called()
    assert entity_extraction.extract_entities_llm(TEXT, llm_caller=caller) == ([], [])
    caller.assert_called_once()
