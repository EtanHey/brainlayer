"""Retained session readers live outside the retired model pipeline."""

import os
import subprocess
import sys
from pathlib import Path


def test_session_reconstruction_is_a_local_history_reader():
    from brainlayer.pipeline.session_history import reconstruct_session

    assert reconstruct_session.__module__ == "brainlayer.pipeline.session_history"


def test_session_history_import_does_not_load_a_model_transport(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = r"""
import builtins, pathlib, sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def local_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith(('requests', 'google.genai', 'brainlayer.pipeline.enrichment', 'brainlayer.enrichment_controller')):
        raise AssertionError(f'Historical reader imported transport: {name}')
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = local_import
from brainlayer.pipeline import session_history
assert session_history.reconstruct_session is not None
assert pathlib.Path(session_history.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())
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
