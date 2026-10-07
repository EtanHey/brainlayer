"""Historical result parsing must not import the retired producer pipeline."""

import os
import subprocess
import sys
from pathlib import Path


def test_saved_result_support_imports_without_model_transport(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = r"""
import builtins, sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def local_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in ('brainlayer.pipeline.enrichment', 'brainlayer.enrichment_controller') or name.startswith(('requests', 'google.genai')):
        raise AssertionError(f'Saved-result helper imported transport: {name}')
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = local_import
from brainlayer.pipeline.enrichment_results import HIGH_VALUE_TYPES, normalize_enrichment_tags, enrichment_version_metadata
assert HIGH_VALUE_TYPES == ['ai_code', 'stack_trace', 'user_message', 'assistant_text']
assert normalize_enrichment_tags(['React.js', 'reactjs']) == ['react']
assert enrichment_version_metadata()['prompt_version']
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root / "src")],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_prompt_template_is_a_local_offline_artifact():
    from brainlayer.pipeline.enrichment_prompts import ENRICHMENT_PROMPT

    assert "{content}" in ENRICHMENT_PROMPT
    assert "sentiment_label" in ENRICHMENT_PROMPT


def test_public_prompt_builder_lives_outside_the_retired_producer():
    from brainlayer.pipeline import build_external_prompt

    assert build_external_prompt.__module__ == "brainlayer.pipeline.enrichment_prompts"
