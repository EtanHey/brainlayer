"""Historical session metadata must not retain a model-producing entry point."""


def test_session_model_producer_is_absent():
    from brainlayer.pipeline import session_enrichment

    assert not hasattr(session_enrichment, "enrich_session")
