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
