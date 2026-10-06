"""Unit regressions for missing configured NER; real-keg proof lives in CI."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

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
        measured_sha="a" * 40,
    )
    assert ratchet.row_spacy_model(probe, {}).status == ratchet.RED


def test_realtime_missing_ner_never_calls_remote(monkeypatch):
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


def test_real_probe_missing_binary_and_bad_wheel_fail(tmp_path, monkeypatch):
    from scripts.ci_spacy_model import main, probe

    python = tmp_path / "absent-python"
    assert probe(python) is False
    wheel = tmp_path / "bad.whl"
    wheel.write_bytes(b"not-a-wheel")
    out = tmp_path / "report.json"
    argv = ["probe", "--python", str(python), "--fix-sha", "a" * 40]
    argv += ["--candidate-wheel", str(wheel), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 1
    assert json.loads(out.read_text())["loaded"] is False


@pytest.mark.parametrize("scope", ["published", "candidate"])
def test_spacy_row_measured_pass_and_failed_published_are_distinct(tmp_path, scope):
    path = tmp_path / "report.json"
    payload = dict(bug_sha=ratchet.SPACY_BUG_SHA, fix_sha="a" * 40, python="/fixture/keg/bin/python", loaded=True)
    payload.update(scope=scope, published_loaded=scope == "published")
    path.write_text(json.dumps(payload))
    probe = SimpleNamespace(spacy_report=path, measured_sha="a" * 40, head_sha="a" * 40)
    row = ratchet.row_spacy_model(probe, {})
    assert row.status == ratchet.GREEN
    assert f"{scope} · published {'PASS' if scope == 'published' else 'FAIL'}" in row.value
    assert ratchet.SPACY_BUG_SHA in row.notes and "a" * 40 in row.value
    payload.update(scope="published", published_loaded=False)
    path.write_text(json.dumps(payload))
    assert ratchet.row_spacy_model(probe, {}).status == ratchet.RED


def test_spacy_workflow_defaults_to_published_and_opt_in_is_explicit():
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


def test_keg_probe_disables_user_site(monkeypatch):
    from scripts import ci_spacy_model as model

    monkeypatch.setattr(model, "PROBE", "import site; raise SystemExit(0 if site.ENABLE_USER_SITE is False else 1)")
    assert model.probe(Path(sys.executable))


@pytest.mark.parametrize("pipe_names", [[], ["ner"]])
def test_optimized_probe_still_checks_ner_and_redaction(tmp_path, pipe_names):
    from scripts.ci_spacy_model import PROBE

    (tmp_path / "spacy.py").write_text(
        "from types import SimpleNamespace\n"
        f"class Model:\n    pipe_names = {pipe_names!r}\n"
        "    def __call__(self, text): return SimpleNamespace(ents=[])\n"
        "def load(*args, **kwargs): return Model()\n"
    )
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join((str(tmp_path), str(Path(__file__).resolve().parents[1] / "src")))
    )
    result = subprocess.run([sys.executable, "-O", "-c", PROBE], env=env, capture_output=True, timeout=30)
    assert result.returncode != 0
