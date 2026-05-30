import os
from types import SimpleNamespace

import pytest


def test_gemini_correction_judge_parses_strict_json_and_uses_enrichment_backend(monkeypatch):
    from brainlayer import correction_judge
    from brainlayer.correction_judge import GeminiCorrectionJudge
    from brainlayer.enrichment_controller import GEMINI_REALTIME_MODEL

    captured = {}

    def fake_get_client():
        return SimpleNamespace(models=SimpleNamespace())

    def fake_generate(client, model, prompt, config, rate_limiter):
        captured["client"] = client
        captured["model"] = model
        captured["prompt"] = prompt
        captured["config"] = config
        captured["rate_limiter"] = rate_limiter
        return SimpleNamespace(
            text='{"action":"supersede","confidence":0.91,"reasoning":"The new fact corrects the stale backend."}'
        )

    monkeypatch.setattr(correction_judge, "_get_gemini_client", fake_get_client)
    monkeypatch.setattr(correction_judge, "_generate_content_with_rate_limit", fake_generate)

    verdict = GeminiCorrectionJudge(rate_limiter="limiter").judge(
        entity="BrainLayer",
        new_fact={"fact_text": "BrainLayer uses sqlite-vec."},
        conflicting_fact={"fact_text": "BrainLayer uses ChromaDB."},
        context={"source": "unit-test"},
    )

    assert verdict.action == "supersede"
    assert verdict.confidence == pytest.approx(0.91)
    assert verdict.reasoning == "The new fact corrects the stale backend."
    assert captured["model"] == GEMINI_REALTIME_MODEL
    assert captured["rate_limiter"] == "limiter"
    assert captured["config"]["response_mime_type"] == "application/json"
    assert captured["config"]["response_schema"]["properties"]["action"]["enum"] == [
        "supersede",
        "merge",
        "noise",
    ]
    assert "BrainLayer uses sqlite-vec." in captured["prompt"]
    assert "BrainLayer uses ChromaDB." in captured["prompt"]


def test_correction_judge_factory_defaults_to_gemini_and_keeps_local_seam(monkeypatch):
    from brainlayer.correction_judge import GeminiCorrectionJudge, LocalCorrectionJudge, get_correction_judge

    monkeypatch.delenv("BRAINLAYER_JUDGE_BACKEND", raising=False)
    assert isinstance(get_correction_judge(), GeminiCorrectionJudge)

    monkeypatch.setenv("BRAINLAYER_JUDGE_BACKEND", "local")
    judge = get_correction_judge()
    assert isinstance(judge, LocalCorrectionJudge)
    with pytest.raises(NotImplementedError):
        judge.judge("BrainLayer", "new", "old", {})


def test_verdict_parser_rejects_missing_confidence_with_value_error():
    from brainlayer.correction_judge import _coerce_verdict

    with pytest.raises(ValueError, match="confidence must be a number"):
        _coerce_verdict({"action": "supersede", "reasoning": "missing confidence"})


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("BRAINLAYER_RUN_LIVE_JUDGE") != "1", reason="set BRAINLAYER_RUN_LIVE_JUDGE=1")
def test_live_gemini_correction_judge_can_adjudicate_real_conflict():
    from brainlayer.correction_judge import GeminiCorrectionJudge

    verdict = GeminiCorrectionJudge().judge(
        entity="BrainLayer",
        new_fact={"fact_text": "BrainLayer uses sqlite-vec for vector search."},
        conflicting_fact={"fact_text": "BrainLayer uses ChromaDB for vector search."},
        context={"purpose": "manual live smoke test"},
    )

    assert verdict.action in {"supersede", "merge", "noise"}
    assert 0 <= verdict.confidence <= 1
    assert verdict.reasoning
