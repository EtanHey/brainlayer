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
