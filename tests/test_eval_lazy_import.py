"""Importing the public evaluation API never starts optional numeric backends."""

import os
import subprocess
import sys
from pathlib import Path


def test_eval_cold_import_does_not_load_optional_numeric_backends(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(source)!r}); "
        "import brainlayer.eval; import brainlayer.eval.benchmark; "
        "assert not any(k.split('.')[0] in {'scipy', 'ranx', 'numpy'} for k in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        env={**os.environ, "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
