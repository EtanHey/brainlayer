"""Legacy module execution cannot open a database or a model transport."""

import os
import subprocess
import sys
from pathlib import Path


def test_legacy_module_execution_is_retired_before_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = r"""
import builtins, runpy, sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'requests' or name.endswith('vector_store'):
        raise AssertionError(f'Retired module loaded runtime dependency: {name}')
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded_import
sys.argv = ['brainlayer.pipeline.enrichment', '--stats', '--backend', 'groq']
runpy.run_module('brainlayer.pipeline.enrichment', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root / "src")],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "enrichment has been retired" in result.stderr
    assert "AssertionError" not in result.stderr


def test_legacy_runner_is_absent_and_saved_parser_remains_available():
    from brainlayer.pipeline import enrichment

    assert not hasattr(enrichment, "run_enrichment")
    saved = enrichment.parse_enrichment('{"summary":"Historical saved result stays readable", "importance":7}')
    assert saved["summary"] == "Historical saved result stays readable"


def test_legacy_batch_producer_is_absent():
    from brainlayer.pipeline import enrichment

    assert not hasattr(enrichment, "enrich_batch")


def test_legacy_single_chunk_producer_is_absent():
    from brainlayer.pipeline import enrichment

    assert not hasattr(enrichment, "_enrich_one")


def test_legacy_fallback_health_and_restart_are_absent():
    from brainlayer.pipeline import enrichment

    for name in (
        "call_llm",
        "_check_fallback_available",
        "check_backend_health",
        "_try_restart_mlx",
        "_recover_backend",
    ):
        assert not hasattr(enrichment, name), name


def test_legacy_groq_sender_is_absent():
    from brainlayer.pipeline import enrichment

    assert not hasattr(enrichment, "call_groq")


def test_legacy_http_model_senders_are_absent():
    from brainlayer.pipeline import enrichment

    assert not hasattr(enrichment, "call_glm")
    assert not hasattr(enrichment, "call_mlx")


def test_legacy_producer_telemetry_and_controls_are_absent():
    from brainlayer.pipeline import enrichment

    for name in ("_sync_stats_to_supabase", "_log_glm_usage", "GROQ_API_KEY", "OLLAMA_URL", "MLX_AUTO_RESTART"):
        assert not hasattr(enrichment, name), name
