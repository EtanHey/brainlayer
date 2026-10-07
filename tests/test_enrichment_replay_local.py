"""Historical replay support cannot depend on a model producer or transport."""

import os
import subprocess
import sys
from pathlib import Path


def test_replay_support_cold_import_is_transport_free(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = r"""
import builtins, sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def local_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in ('brainlayer.pipeline.enrichment', 'brainlayer.enrichment_controller') or name.startswith(('requests', 'google.genai')):
        raise AssertionError(f'Replay support imported transport: {name}')
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = local_import
from brainlayer import enrichment_replay as replay
from brainlayer.chunk_write import canonical_content_hash
assert replay._content_hash is canonical_content_hash
assert replay._derive_chunk_provenance_class({'source':'mcp', 'source_file':'brainlayer-queue', 'content':'synthetic'}) == 'RAW-ETAN-DIRECT'
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


def test_saved_result_apply_lives_outside_the_model_controller():
    from brainlayer.enrichment_replay import _apply_enrichment, _apply_enrichment_impl

    assert _apply_enrichment.__module__ == "brainlayer.enrichment_replay"
    assert _apply_enrichment_impl.__module__ == "brainlayer.enrichment_replay"


def test_historical_provenance_read_and_payload_helpers_are_local():
    from brainlayer import enrichment_replay

    for name in ("_get_chunk_readonly", "_previous_assistant_text", "_enrichment_update_payload"):
        assert getattr(enrichment_replay, name).__module__ == "brainlayer.enrichment_replay"


def test_historical_hash_and_class_helpers_are_local():
    from brainlayer import enrichment_replay

    for name in (
        "is_meta_research",
        "_is_duplicate_content",
        "_ensure_content_hash_column",
        "_backfill_content_hashes",
    ):
        assert getattr(enrichment_replay, name).__module__ == "brainlayer.enrichment_replay"


def test_controller_has_no_duplicate_historical_hash_helpers():
    import ast

    source = Path(__file__).resolve().parents[1] / "src/brainlayer/enrichment_controller.py"
    names = {node.name for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef)}
    assert names.isdisjoint(
        {"is_meta_research", "_is_duplicate_content", "_ensure_content_hash_column", "_backfill_content_hashes"}
    )
