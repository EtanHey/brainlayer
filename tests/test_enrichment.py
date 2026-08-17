"""Local enrichment pipeline must not stamp backend names onto chunk_origin."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_local_enrichment_pipeline_does_not_stamp_backend_as_chunk_origin():
    from brainlayer.pipeline import enrichment

    store = MagicMock()
    store.get_context.return_value = {"context": []}
    chunk = {
        "id": "chunk-mlx",
        "content": "content that should be enriched",
        "content_type": "user_message",
        "project": "brainlayer",
        "conversation_id": None,
        "position": None,
    }

    with (
        patch.object(enrichment, "build_prompt", return_value="prompt"),
        patch.object(enrichment, "call_llm", return_value='{"summary":"ok summary","tags":["test"]}'),
        patch.object(enrichment, "parse_enrichment", return_value={"summary": "ok summary", "tags": ["test"]}),
    ):
        result = enrichment._enrich_one(store, chunk, with_context=False, backend="mlx")

    assert result is True
    kwargs = store.update_enrichment.call_args.kwargs
    assert kwargs.get("chunk_origin") in (None, "")
    assert kwargs.get("enrichment_model")
