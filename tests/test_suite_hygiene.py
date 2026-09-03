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

import ast
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import apsw
import pytest

from brainlayer import paths as brain_paths
from brainlayer.embeddings import FORBID_MODEL_LOAD_ENV, guard_embedding_model_load
from tests.conftest import EMBEDDING_MODEL_MODULES, hygiene_exemptions

REPO_ROOT = Path(__file__).resolve().parents[1]
REEMBED_SCRIPT = REPO_ROOT / "scripts" / "reembed_bgem3.py"
# The real path, read before conftest's isolation fixture rewrites the module attribute -- the
# guard has to refuse THIS one, not the tmp path every test is pointed at.
PRODUCTION_DB_PATH = Path("~/.local/share/brainlayer/brainlayer.db").expanduser()
# Bound at MODULE IMPORT, which is exactly the shape a fixture-scoped patch cannot reach.
_CONNECT_ALIAS = sqlite3.connect


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


def test_no_test_module_binds_an_embedding_model_class_directly() -> None:
    """The guard's contract, made impossible to violate rather than merely documented.

    `tests/conftest.py` patches `SentenceTransformer` **on the module**. A test that did
    `from sentence_transformers import SentenceTransformer` at import time would bind the real
    class before any fixture runs, and calling that alias reaches neither the module patch nor
    `BRAINLAYER_FORBID_EMBEDDING_MODEL` -- that env check lives at BrainLayer's load sites, not
    inside a third-party constructor (CodeRabbit, #755).

    Wrapping the import machinery to close that would cost every run an import hook for a shape no
    test uses. So the contract is narrowed and enforced instead: in `tests/`, reach an embedding
    model through `brainlayer.embeddings`, or through a module attribute -- never a direct alias.
    """
    offenders = []
    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in EMBEDDING_MODEL_MODULES:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} -> from {node.module} import ...")

    assert offenders == [], (
        "these test modules bind an embedding-model class directly, which the suite-hygiene guard "
        f"cannot intercept: {offenders}"
    )


class _StubNode:
    """The only thing `hygiene_exemptions` reads off a pytest item."""

    def __init__(self, *marks: str) -> None:
        self._marks = set(marks)

    def get_closest_marker(self, name: str) -> object | None:
        return object() if name in self._marks else None


def test_the_two_guards_have_separate_escapes() -> None:
    """`embedding_model` must NOT lift the canonical-DB guard.

    One bit used to control both, so a test marked only `embedding_model` got the DB guard lifted
    as well — a hole at exactly *model + production DB together*, which is the incident these
    guards exist to prevent. `scripts/reembed_bgem3.py` makes that concrete: `main()` opens its
    `--db` (defaulting to the CANONICAL path) at line 277, BEFORE `load_model()` at line 285.
    """
    assert hygiene_exemptions(_StubNode()) == (False, False)
    assert hygiene_exemptions(_StubNode("embedding_model")) == (True, False)
    assert hygiene_exemptions(_StubNode("integration")) == (False, True)
    assert hygiene_exemptions(_StubNode("live")) == (False, True)
    # A test that genuinely needs both has to say both.
    assert hygiene_exemptions(_StubNode("embedding_model", "live")) == (True, True)


@pytest.mark.embedding_model
def test_a_model_marked_test_still_cannot_open_the_canonical_db() -> None:
    """End-to-end proof of the split, from inside a test that really carries the model marker."""
    assert os.environ.get(FORBID_MODEL_LOAD_ENV) != "1"

    with pytest.raises(RuntimeError, match="canonical BrainLayer DB"):
        sqlite3.connect(str(PRODUCTION_DB_PATH))
    with pytest.raises(RuntimeError, match="canonical BrainLayer DB"):
        sqlite3.connect(f"file:{PRODUCTION_DB_PATH}?mode=ro", uri=True)


def test_the_canonical_db_is_refused_through_a_file_uri(tmp_path: Path) -> None:
    """`file:` URIs are how BrainLayer actually reads the DB, so the guard must parse them.

    `backup_daily`, `maintenance`, `kg_judge`, `t3_provenance` and friends all open
    `file:{db_path}?mode=ro`. A guard that fed that string to `Path` would resolve a nonexistent
    relative path, match nothing, and let every real reader through while looking like it worked.
    """
    for uri in (
        f"file:{PRODUCTION_DB_PATH}?mode=ro",
        f"{PRODUCTION_DB_PATH.as_uri()}?mode=ro&immutable=1",
        f"file://localhost{PRODUCTION_DB_PATH}",
    ):
        with pytest.raises(RuntimeError, match="canonical BrainLayer DB"):
            sqlite3.connect(uri, uri=True)

    # Still not a blanket ban: a temp-path URI, and the in-memory forms, must keep working.
    scratch = tmp_path / "scratch.db"
    sqlite3.connect(f"{scratch.as_uri()}?mode=rwc", uri=True).close()
    sqlite3.connect("file::memory:?cache=shared", uri=True).close()
    assert scratch.is_file()


def test_an_alias_bound_before_the_fixture_ran_is_still_guarded() -> None:
    """The guards are installed in `pytest_configure`, so import-time aliases cannot dodge them.

    `from sqlite3 import connect` at a test module's top binds whatever `sqlite3.connect` is at
    IMPORT time. A fixture-scoped patch would be invisible to that alias (CodeRabbit, #755).
    """
    assert getattr(sqlite3.connect, "_brainlayer_hygiene_guard", False) is True
    assert getattr(apsw.Connection, "_brainlayer_hygiene_guard", False) is True

    with pytest.raises(RuntimeError, match="canonical BrainLayer DB"):
        _CONNECT_ALIAS(str(PRODUCTION_DB_PATH))

    with pytest.raises(RuntimeError, match="canonical BrainLayer DB"):
        apsw.Connection(str(PRODUCTION_DB_PATH), flags=apsw.SQLITE_OPEN_READONLY)


def test_the_isolated_canonical_path_is_not_the_production_one() -> None:
    """conftest redirects `_CANONICAL_DB_PATH`; this pins that the redirect actually happened."""
    # Read through the module, not by value: conftest rebinds the attribute, and a name imported
    # at module scope would still hold the production path and pass for the wrong reason.
    assert brain_paths._CANONICAL_DB_PATH != PRODUCTION_DB_PATH
    assert PRODUCTION_DB_PATH.parent not in brain_paths._CANONICAL_DB_PATH.parents
