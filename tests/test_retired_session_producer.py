"""Historical session metadata must not retain a model-producing entry point."""


def test_session_model_producer_is_absent():
    from brainlayer.pipeline import session_history

    assert not hasattr(session_history, "enrich_session")


def test_session_model_prompt_is_absent():
    from brainlayer.pipeline import session_history

    assert not hasattr(session_history, "build_session_prompt")
    assert not hasattr(session_history, "SESSION_ANALYSIS_PROMPT")


def test_retired_session_module_is_absent():
    import importlib.util
    from pathlib import Path

    assert importlib.util.find_spec("brainlayer.pipeline.session_enrichment") is None
    assert not (Path(__file__).resolve().parents[1] / "src/brainlayer/pipeline/session_enrichment.py").exists()
