"""Local pipeline imports must not eagerly load model transports."""

import os
import subprocess
import sys
from pathlib import Path


def test_local_pipeline_import_does_not_load_cloud_transport():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    code = r"""
import sys
import brainlayer.pipeline as pipeline
assert pipeline.Sanitizer is not None
assert pipeline.chunk_content is not None
assert pipeline.classify_content is not None
assert pipeline.analyze_semantic_style is not None
for name in ('brainlayer.pipeline.enrichment', 'brainlayer.pipeline.groq', 'groq', 'requests', 'google.genai'):
    assert name not in sys.modules, f'Eager cloud transport imported: {name}'
"""
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_external_prompt_import_keeps_four_argument_result():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    code = r"""
from brainlayer.pipeline import build_external_prompt, Sanitizer, SanitizeConfig
from brainlayer.pipeline.enrichment import build_external_prompt as original
assert build_external_prompt is original
sanitizer = Sanitizer(SanitizeConfig(owner_names=("Jane Developer",), use_spacy_ner=False))
prompt, result = build_external_prompt(
    {"project": "compat", "content_type": "user_message", "content": "Jane Developer fixed {bug}"},
    sanitizer,
    [{"content_type": "ai_code", "content": "Jane Developer confirmed"}],
    "{project}|{content_type}|{content}|{context_section}",
)
assert prompt == "compat|user_message|[OWNER] fixed {{bug}}|SURROUNDING CONTEXT:\n[ai_code] [OWNER] confirmed", prompt
assert result.sanitized == "[OWNER] fixed {bug}"
assert result.pii_detected is True
assert len(result.replacements) == 2
"""
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_pipeline_star_import_remains_transport_free():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    code = r"""
import sys
from brainlayer.pipeline import *
assert Sanitizer is not None
assert chunk_content is not None
assert 'build_external_prompt' not in globals()
for name in ('brainlayer.pipeline.enrichment', 'brainlayer.pipeline.groq', 'groq', 'requests', 'google.genai'):
    assert name not in sys.modules, f'Eager cloud transport imported: {name}'
"""
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
