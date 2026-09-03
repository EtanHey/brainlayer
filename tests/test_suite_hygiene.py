"""The standing suite-hygiene rule, enforced instead of merely written down.

No test loads an embedding model, and no test opens the canonical BrainLayer DB. Both were broken
expensively: the pre-push `changed-only` fallback escalated to the full suite, which spawned
`scripts/reembed_bgem3.py --test` -- a 2.5 GB-RSS BGE-M3 load holding ~20 fds on the production DB,
the measured cause of a 14:22 UI stall on the M4.

The guards live in `tests/conftest.py` (arming) and at every model-load site in `src/brainlayer/`
and `scripts/` (refusing). These tests pin both halves, including the one that has to survive a
process boundary.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from brainlayer import paths as brain_paths
from brainlayer.embeddings import FORBID_MODEL_LOAD_ENV, guard_embedding_model_load

REPO_ROOT = Path(__file__).resolve().parents[1]
REEMBED_SCRIPT = REPO_ROOT / "scripts" / "reembed_bgem3.py"
# The real path, read before conftest's isolation fixture rewrites the module attribute -- the
# guard has to refuse THIS one, not the tmp path every test is pointed at.
PRODUCTION_DB_PATH = Path("~/.local/share/brainlayer/brainlayer.db").expanduser()


def test_unmarked_tests_are_forbidden_from_loading_an_embedding_model() -> None:
    assert os.environ.get(FORBID_MODEL_LOAD_ENV) == "1"

    with pytest.raises(RuntimeError, match=FORBID_MODEL_LOAD_ENV):
        guard_embedding_model_load("BAAI/bge-m3")


@pytest.mark.embedding_model
def test_a_marked_test_may_load_an_embedding_model() -> None:
    """The marker is the whole escape hatch, and it is the only one."""
    assert os.environ.get(FORBID_MODEL_LOAD_ENV) != "1"

    guard_embedding_model_load("BAAI/bge-m3")


def test_the_refusal_survives_a_process_boundary(tmp_path: Path) -> None:
    """A SPAWNED re-embedding script must refuse too — that is the shape that melted the M4.

    sys.modules cannot see a subprocess, so an in-process-only guard would have missed exactly the
    test that caused the incident.
    """
    program = textwrap.dedent(
        f"""
        import importlib.util

        spec = importlib.util.spec_from_file_location("reembed", {str(REEMBED_SCRIPT)!r})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            module.load_model()
        except RuntimeError as error:
            print("REFUSED:", error)
        else:
            raise SystemExit("the spawned script loaded a model anyway")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
        check=False,  # the refusal IS the assertion below; a raise here would hide it
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "REFUSED:" in result.stdout
    assert FORBID_MODEL_LOAD_ENV in result.stdout
    # It refused BEFORE paying for torch, which is the only refusal worth having here.
    assert "torch" not in result.stderr


def test_opening_the_canonical_db_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="canonical BrainLayer DB"):
        sqlite3.connect(str(PRODUCTION_DB_PATH))

    # Not a blanket ban on sqlite3: every test that uses a temp DB must keep working.
    scratch = tmp_path / "scratch.db"
    connection = sqlite3.connect(str(scratch))
    connection.execute("CREATE TABLE t (id INTEGER)")
    connection.close()
    assert scratch.is_file()


def test_the_isolated_canonical_path_is_not_the_production_one() -> None:
    """conftest redirects `_CANONICAL_DB_PATH`; this pins that the redirect actually happened."""
    # Read through the module, not by value: conftest rebinds the attribute, and a name imported
    # at module scope would still hold the production path and pass for the wrong reason.
    assert brain_paths._CANONICAL_DB_PATH != PRODUCTION_DB_PATH
    assert PRODUCTION_DB_PATH.parent not in brain_paths._CANONICAL_DB_PATH.parents
