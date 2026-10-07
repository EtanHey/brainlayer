"""Fact correction remains caller-supplied; cloud adjudication is retired."""

import ast
import inspect

import pytest

pytestmark = pytest.mark.retired_enrichment


@pytest.mark.parametrize("backend", [None, "gemini", "groq", "local"])
def test_correction_judge_factory_is_retired(monkeypatch, backend):
    from brainlayer import correction_judge

    if backend is None:
        monkeypatch.delenv("BRAINLAYER_JUDGE_BACKEND", raising=False)
    else:
        monkeypatch.setenv("BRAINLAYER_JUDGE_BACKEND", backend)
    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-retirement-key")
    with pytest.raises(RuntimeError, match="retired"):
        correction_judge.get_correction_judge()


def test_correction_judge_has_no_cloud_sender_or_import():
    from brainlayer import correction_judge

    assert not hasattr(correction_judge, "GeminiCorrectionJudge")
    tree = ast.parse(inspect.getsource(correction_judge))
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("enrichment" in name or "google" in name for name in imports)


def test_verdict_parser_rejects_missing_confidence_with_value_error():
    from brainlayer.correction_judge import _coerce_verdict

    with pytest.raises(ValueError, match="confidence must be a number"):
        _coerce_verdict({"action": "supersede", "reasoning": "missing confidence"})
