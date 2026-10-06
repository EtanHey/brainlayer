"""Local pipeline imports must not eagerly load model transports."""

import os
import subprocess
import sys
from pathlib import Path


def test_local_pipeline_import_does_not_load_cloud_transport():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    code = """
import sys
import brainlayer.pipeline as pipeline
assert pipeline.Sanitizer is not None
assert pipeline.chunk_content is not None
assert pipeline.classify_content is not None
assert pipeline.analyze_semantic_style is not None
for name in ('brainlayer.pipeline.enrichment', 'brainlayer.pipeline.groq', 'google.genai'):
    assert name not in sys.modules, f'Eager cloud transport imported: {name}'
"""
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
