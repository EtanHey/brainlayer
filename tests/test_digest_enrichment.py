"""Tests for digest-time faceted enrichment."""

from brainlayer.pipeline.digest import _build_faceted_gemini_config


def test_faceted_gemini_config_disables_thinking():
    """Flash models must always force thinkingBudget=0."""
    config = _build_faceted_gemini_config()

    assert config["response_mime_type"] == "application/json"
    assert config["thinking_config"]["thinking_budget"] == 0


def test_missing_pii_ner_returns_failed_after_one_persist(tmp_path, monkeypatch):
    """A post-persist PII failure returns a receipt rather than inviting a retry."""
    from unittest.mock import MagicMock

    import spacy
    from google import genai

    from brainlayer.pipeline.digest import digest_content
    from brainlayer.vector_store import VectorStore

    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("BRAINLAYER_LLM_ENTITY_EXTRACTION", "0")
    monkeypatch.setenv("BRAINLAYER_MCP_SOCKET", str(tmp_path / "unused.sock"))
    monkeypatch.setenv("BRAINLAYER_FORBID_BRAINBAR_SOCKET", "1")
    monkeypatch.setattr(spacy, "load", MagicMock(side_effect=OSError("private-loader-value")))
    client = MagicMock()
    monkeypatch.setattr(genai, "Client", client)

    with VectorStore(tmp_path / "digest.db") as store:
        result = digest_content(
            content="Synthetic digest content for missing NER regression.",
            store=store,
            embed_fn=lambda _: [0.1] * 1024,
        )
        assert result["enrichment"]["status"] == "failed"
        assert result["enrichment"]["reason"] == "pii_ner_unavailable"
        assert result["enrichment"]["stored"] is True
        assert result["enrichment"]["chunk_id"] == result["digest_id"]
        assert "private-loader-value" not in str(result)
        assert store.get_chunk(result["digest_id"]) is not None
        assert list(store.conn.execute("SELECT COUNT(*) FROM chunks"))[0][0] == 1
        client.assert_not_called()
