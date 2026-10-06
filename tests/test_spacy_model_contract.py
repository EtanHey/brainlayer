"""Unit regressions for missing configured NER; real-keg proof lives in CI."""

import json
import sys
from types import SimpleNamespace

import pytest

from brainlayer.pipeline.enrichment import build_external_prompt
from brainlayer.pipeline.sanitize import SanitizeConfig, Sanitizer
from scripts import ci_ratchet_table as ratchet


@pytest.mark.parametrize("error", [ImportError("private-value"), OSError("private-value"), ValueError("private-value")])
def test_configured_ner_failure_blocks_every_attempt(monkeypatch, capsys, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=fail))
    sanitizer = Sanitizer(SanitizeConfig())
    for _ in range(2):
        with pytest.raises(RuntimeError, match="PII NER model unavailable") as caught:
            sanitizer.sanitize("John Smith sent private-value")
        assert type(caught.value).__name__ == "PIINERUnavailableError"
        assert caught.value.__cause__ is None
        assert "private-value" not in str(caught.value)
    assert capsys.readouterr().err == ""


def test_missing_spacy_import_blocks_prompt(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacy", None)
    with pytest.raises(RuntimeError, match="PII NER model unavailable"):
        build_external_prompt({"content": "John Smith is here"}, Sanitizer(SanitizeConfig()))


@pytest.mark.parametrize("parallel", [1, 4])
def test_missing_model_blocks_batch(monkeypatch, parallel):
    monkeypatch.setitem(sys.modules, "spacy", None)
    with pytest.raises(RuntimeError, match="PII NER model unavailable"):
        Sanitizer(SanitizeConfig()).sanitize_batch([{"content": "John Smith"}], parallel=parallel)


def test_explicit_local_ner_opt_out(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacy", None)
    result = Sanitizer(SanitizeConfig(use_spacy_ner=False)).sanitize("a@example.com")
    assert "a@example.com" not in result.sanitized


@pytest.mark.parametrize(
    "payload", [None, {}, {"scope": "published", "loaded": False}, {"scope": "candidate", "loaded": True}]
)
def test_spacy_row_missing_or_incomplete_report_is_red(tmp_path, payload):
    path = tmp_path / "report.json"
    if payload is not None:
        path.write_text(json.dumps(payload))
    probe = SimpleNamespace(
        spacy_report=path,
        signature_unavailable=None,
        measured_sha="a" * 40,
        head_sha="a" * 40,
    )
    assert ratchet.row_spacy_model(probe, {}).status == ratchet.RED


def test_realtime_missing_ner_never_calls_remote(monkeypatch):
    from unittest.mock import Mock

    from brainlayer import enrichment_controller as controller

    monkeypatch.setitem(sys.modules, "spacy", None)
    send = Mock()
    monkeypatch.setattr(controller, "_generate_content_with_rate_limit", send)
    _, status, error = controller._enrich_single_chunk(
        object(),
        "synthetic-model",
        {},
        {"id": "synthetic", "content": "John Smith lives in London."},
        Sanitizer(SanitizeConfig()),
        is_duplicate=lambda _: False,
        rate_limiter=None,
        max_retries=0,
    )
    assert status == "error" and "PII NER model unavailable" in error
    send.assert_not_called()


def test_real_probe_missing_binary_and_bad_wheel_fail(tmp_path):
    from unittest.mock import patch

    from scripts.ci_spacy_model import main, probe

    assert probe(tmp_path / "absent-python") is False
    wheel = tmp_path / "bad.whl"
    wheel.write_bytes(b"not-a-wheel")
    out = tmp_path / "report.json"
    with patch.object(
        sys,
        "argv",
        [
            "probe",
            "--python",
            str(tmp_path / "absent-python"),
            "--fix-sha",
            "a" * 40,
            "--candidate-wheel",
            str(wheel),
            "--out",
            str(out),
        ],
    ):
        assert main() == 1
    assert json.loads(out.read_text())["loaded"] is False


@pytest.mark.parametrize("scope", ["published", "candidate"])
def test_spacy_row_measured_pass_and_failed_published_are_distinct(tmp_path, scope):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            dict(
                bug_sha=ratchet.SPACY_BUG_SHA,
                fix_sha="a" * 40,
                python="/fixture/keg/bin/python",
                published_loaded=scope == "published",
                loaded=True,
                scope=scope,
            )
        )
    )
    probe = SimpleNamespace(spacy_report=path, measured_sha="a" * 40, head_sha="a" * 40)
    row = ratchet.row_spacy_model(probe, {})
    assert row.status == ratchet.GREEN
    assert f"{scope} · published {'PASS' if scope == 'published' else 'FAIL'}" in row.value
    assert ratchet.SPACY_BUG_SHA in row.notes and "a" * 40 in row.value
    payload = json.loads(path.read_text())
    payload.update(scope="published", published_loaded=False)
    path.write_text(json.dumps(payload))
    assert ratchet.row_spacy_model(probe, {}).status == ratchet.RED


def test_spacy_workflow_defaults_to_published_and_opt_in_is_explicit():
    from pathlib import Path

    import yaml

    workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github/workflows/ratchet.yml").read_text())
    job = workflow["jobs"]["signatures"]
    step = next(s for s in job["steps"] if s.get("id") == "spacy")
    assert "ratchet:spacy-candidate" in step["env"]["CANDIDATE_MODEL"]
    assert "--candidate-wheel" in step["run"] and '"$CANDIDATE_MODEL" == "true"' in step["run"]
    assert (
        "--spacy-report"
        in next(
            s
            for s in workflow["jobs"]["table"]["steps"]
            if s.get("name") == "Hand the signature measurement to the collector"
        )["run"]
    )
