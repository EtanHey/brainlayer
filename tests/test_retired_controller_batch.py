"""Retired controller batch phases must fail before touching data or a model."""

from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("phase", ["run", "submit", "poll", "import"])
@pytest.mark.parametrize("candidates", [[], [{"id": "synthetic-existing", "content": "Synthetic historical content"}]])
def test_controller_batch_is_retired_before_store_access(monkeypatch, request, phase, candidates):
    request.getfixturevalue("isolate_brainlayer_runtime_paths")
    from brainlayer import enrichment_controller as controller

    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-retirement-key")
    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_QUEUE_WRITES", "1")
    store = MagicMock()
    store.get_enrichment_candidates.return_value = candidates
    factory = MagicMock(side_effect=AssertionError("cloud factory reached"))
    monkeypatch.setattr(controller, "_get_gemini_client", factory)

    with pytest.raises(RuntimeError, match="enrichment has been retired"):
        controller.enrich_batch(store, phase=phase, limit=5, max_retries=3)

    factory.assert_not_called()
    assert store.mock_calls == []
