"""Retirement compatibility for digest-time faceted enrichment."""

from brainlayer.pipeline.digest import _default_faceted_enrich


def test_brain_digest_retired_faceted_compatibility_receipt():
    """Legacy helper options receive a retirement receipt without a transport."""
    result = _default_faceted_enrich(content="synthetic content", project=None, title=None, participants=None)
    assert result == {"status": "retired", "reason": "cloud_enrichment_retired"}
