"""Importing the public evaluation API never starts optional numeric backends."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module,forbidden",
    [
        ("brainlayer.eval", {"scipy", "ranx", "numpy"}),
        ("brainlayer.pipeline.brain_graph", {"scipy", "sklearn", "numpy.testing"}),
        ("brainlayer.pipeline.style_embed", {"scipy", "sklearn", "numpy.testing", "sentence_transformers", "torch"}),
    ],
)
def test_cold_import_does_not_load_optional_numeric_backends(tmp_path, module, forbidden):
    source = Path(__file__).resolve().parents[1] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(source)!r}); "
        f"import importlib; importlib.import_module({module!r}); "
        f"assert not any(k == f or k.startswith(f + '.') for k in sys.modules for f in {forbidden!r})"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        env={**os.environ, "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
