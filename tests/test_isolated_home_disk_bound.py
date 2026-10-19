"""Every unmarked test runs in its own isolated HOME; that HOME must stay small.

On 2026-10-19 one full-suite basetemp reached 42 GiB and all `pytest-of-*` trees 62 GiB, and the
nightly backup then failed on capacity at 03:22. The cause was the vite-plus `node` shim: it keys
its runtime store off HOME, so every `node --check` spawned inside a fresh isolated home copied a
~117 MiB Node runtime into `.local/share/vite-plus/js_runtime` -- hundreds of times per run, ~6 s
each. pytest's default retention then kept the last three of those trees.

The fixes live in `tests/conftest.py` (a real `node` first on PATH, a per-test HOME size bound,
passing homes removed) and `pyproject.toml` (`tmp_path_retention_policy = "failed"`). These tests
pin all three.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from tests.conftest import ISOLATED_HOME_BYTE_BOUND, isolated_home_overage

REPO_ROOT = Path(__file__).resolve().parents[1]

# pytest truncates a test name to 30 characters for its tmp_path, and a passing test's number is
# reused once its tree is removed -- so the two removal tests below must differ in that prefix.
_previous_home: list[Path] = []


def test_node_inside_an_isolated_home_does_not_bootstrap_a_runtime_into_it() -> None:
    if shutil.which("node") is None:
        pytest.skip("no node on PATH")
    home = Path(os.environ["HOME"])
    checked = subprocess.run(["node", "-e", "0"], capture_output=True, text=True, timeout=60)
    assert checked.returncode == 0, checked.stderr

    runtimes = [path for path in home.rglob("js_runtime") if path.is_dir()]
    assert runtimes == [], f"node bootstrapped a runtime into the isolated HOME: {runtimes}"
    assert isolated_home_overage(home) is None


def test_the_isolated_home_bound_flags_a_runtime_sized_tree(tmp_path: Path) -> None:
    home = tmp_path / "home"
    runtime = home / ".local" / "share" / "vite-plus" / "js_runtime" / "node" / "24.22.0" / "bin" / "node"
    runtime.parent.mkdir(parents=True)
    with runtime.open("wb") as stream:
        stream.truncate(117 * 1024 * 1024)  # sparse: the bound reads st_size, so no disk is spent

    overage = isolated_home_overage(home)

    assert overage is not None
    assert "js_runtime/node/24.22.0/bin/node" in overage
    assert isolated_home_overage(tmp_path / "missing") is None


def test_the_isolated_home_bound_is_below_one_node_runtime() -> None:
    assert ISOLATED_HOME_BYTE_BOUND < 117 * 1024 * 1024


def test_pytest_keeps_temp_trees_only_for_failed_tests() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["pytest"]["ini_options"]["tmp_path_retention_policy"] == "failed"


def test_record_a_passing_home_for_the_removal_check() -> None:
    _previous_home.append(Path(os.environ["HOME"]))
    assert _previous_home[-1].is_dir()


def test_the_recorded_passing_home_was_removed(request: pytest.FixtureRequest) -> None:
    if not _previous_home:
        pytest.skip("runs only after the recording test above")
    if request.config.getini("tmp_path_retention_policy") != "failed":
        pytest.skip("removal follows the failed-only retention policy")
    assert not _previous_home[-1].exists()
