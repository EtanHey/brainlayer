"""Tests for on-demand enrichment — --recent flag for enriching new chunks."""

from unittest.mock import MagicMock

from brainlayer.pipeline import enrichment


class TestRecentEnrichment:
    """Test --recent flag passes since_hours through the pipeline."""

    def test_enrich_batch_passes_since_hours(self):
        """enrich_batch passes since_hours to get_unenriched_chunks."""
        store = MagicMock()
        store.get_unenriched_chunks.return_value = []

        result = enrichment.enrich_batch(store, batch_size=10, since_hours=6)

        store.get_unenriched_chunks.assert_called_once()
        call_kwargs = store.get_unenriched_chunks.call_args
        assert call_kwargs[1].get("since_hours") == 6 or (
            len(call_kwargs[0]) > 0 and 6 in call_kwargs[0]
        )

    def test_enrich_batch_none_since_hours_by_default(self):
        """enrich_batch passes since_hours=None by default (all chunks)."""
        store = MagicMock()
        store.get_unenriched_chunks.return_value = []

        enrichment.enrich_batch(store, batch_size=10)

        call_kwargs = store.get_unenriched_chunks.call_args
        assert call_kwargs[1].get("since_hours") is None

    def test_run_enrichment_accepts_since_hours(self):
        """run_enrichment accepts since_hours parameter."""
        import inspect

        sig = inspect.signature(enrichment.run_enrichment)
        assert "since_hours" in sig.parameters
