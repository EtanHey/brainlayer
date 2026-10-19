"""Every unmarked test runs in its own isolated HOME; that HOME must stay small.

On 2026-10-19 one full-suite basetemp reached 42 GiB and all `pytest-of-*` trees 62 GiB, and the
nightly backup then failed on capacity at 03:22. The cause was the vite-plus `node` shim: it keys
its runtime store off HOME, so every `node --check` spawned inside a fresh isolated home copied a
~117 MiB Node runtime into `.local/share/vite-plus/js_runtime` -- hundreds of times per run, ~6 s
each. pytest's default retention then kept the last three of those trees.

The fixes live in `tests/conftest.py` (a real `node` first on PATH, a per-test HOME size bound,
passing homes removed) and `pyproject.toml` (`tmp_path_retention_policy = "failed"`). These tests
pin all three, and the HOME lifecycle through a real inner pytest run (review r1 of #1412): a HOME
is removed only when no phase failed, failure evidence is never removed (over the bound or not),
a kept HOME never collides with pytest reusing a numbered tmp dir, and `embedding_model` tests may
keep a model cache past the bound.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import tomllib
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

from tests.conftest import ISOLATED_HOME_BYTE_BOUND, isolated_home_overage

REPO_ROOT = Path(__file__).resolve().parents[1]


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


_PROBE = textwrap.dedent(
    """
    import json
    import os
    from pathlib import Path

    import pytest

    LEDGER = Path(__file__).with_name("homes.jsonl")
    BIG = 33 * 1024 * 1024


    def remember(kind, big_file=None):
        home = Path(os.environ["HOME"])
        with LEDGER.open("a") as stream:
            stream.write(json.dumps({"kind": kind, "home": str(home)}) + "\\n")
        (home / "evidence.txt").write_text(kind)
        if big_file is not None:
            path = home / big_file
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                stream.truncate(BIG)
        return home


    def test_failed_call():
        remember("failed-call")
        pytest.fail("deliberate call failure")


    def test_failed_large_call():
        remember("failed-large-call", "large-file")
        pytest.fail("deliberate call failure")


    def test_passing_large_home():
        remember("passing-large", "large-file")


    @pytest.fixture
    def broken_teardown():
        remember("teardown-failure")
        yield
        pytest.fail("deliberate fixture teardown failure")


    def test_teardown_failure(broken_teardown):
        pass


    @pytest.fixture
    def skipped_setup():
        remember("setup-skip")
        pytest.skip("deliberate fixture setup skip")


    def test_prefix_collision_after_setup_skip_first(skipped_setup):
        pass


    def test_prefix_collision_after_setup_skip_second():
        remember("setup-skip-second")


    @pytest.fixture
    def broken_setup():
        remember("setup-error")
        raise RuntimeError("deliberate fixture setup error")


    def test_prefix_collision_after_setup_error_first(broken_setup):
        pass


    def test_prefix_collision_after_setup_error_second():
        remember("setup-error-second")


    @pytest.fixture
    def broken_for_a(request):
        remember(f"param-{request.param}")
        if request.param == "a":
            raise RuntimeError("deliberate parametrized setup error")


    @pytest.mark.parametrize("broken_for_a", ["a", "b"], indirect=True)
    def test_parametrized_prefix_collision_after_error(broken_for_a):
        pass


    def test_passing_home():
        remember("passing")


    @pytest.mark.embedding_model
    def test_declared_model_keeps_a_legitimate_cache():
        remember("model-cache", ".cache/huggingface/hub/model.safetensors")
    """
)


@pytest.fixture(scope="module")
def inner_run(tmp_path_factory):
    """One real pytest session over `_PROBE`, loading this suite's conftest as a plugin."""
    probe_dir = tmp_path_factory.mktemp("home-lifecycle-probe")
    (probe_dir / "test_probe.py").write_text(_PROBE)
    (probe_dir / "pytest.ini").write_text(
        "[pytest]\ntmp_path_retention_policy = failed\nmarkers =\n    embedding_model: probe\n"
    )
    junit = probe_dir / "junit.xml"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO_ROOT), str(REPO_ROOT / "src"), env.get("PYTHONPATH"))))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.conftest",
            "-p",
            "no:cacheprovider",
            "-q",
            "--basetemp",
            str(probe_dir / "basetemp"),
            "--junitxml",
            str(junit),
            "test_probe.py",
        ],
        cwd=probe_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    ledger = probe_dir / "homes.jsonl"
    assert ledger.is_file() and junit.is_file(), completed.stdout + completed.stderr
    homes = {}
    for line in ledger.read_text().splitlines():
        entry = json.loads(line)
        homes[entry["kind"]] = Path(entry["home"])
    outcomes = {}
    for case in ElementTree.parse(junit).iter("testcase"):
        outcomes[case.get("name")] = {child.tag: child.get("message", "") for child in case}
    return completed, homes, outcomes


def _output(completed) -> str:
    return completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "kind", ["failed-call", "failed-large-call", "passing-large", "teardown-failure", "setup-error"]
)
def test_a_home_is_kept_with_its_evidence_when_any_phase_failed(inner_run, kind):
    completed, homes, _outcomes = inner_run
    assert (homes[kind] / "evidence.txt").read_text() == kind, _output(completed)


def test_an_over_bound_home_is_reported_and_kept(inner_run):
    completed, homes, outcomes = inner_run
    assert "error" in outcomes["test_passing_large_home"], _output(completed)
    assert "over the 32 MiB bound" in _output(completed)
    assert (homes["passing-large"] / "large-file").exists()


@pytest.mark.parametrize(
    "name",
    [
        "test_prefix_collision_after_setup_skip_second",
        "test_prefix_collision_after_setup_error_second",
        "test_parametrized_prefix_collision_after_error[b]",
    ],
)
def test_a_kept_or_skipped_home_never_collides_with_a_reused_tmp_dir(inner_run, name):
    completed, _homes, outcomes = inner_run
    assert outcomes[name] == {}, _output(completed)
    assert "FileExistsError" not in _output(completed)


@pytest.mark.parametrize("kind", ["passing", "setup-skip", "setup-skip-second", "setup-error-second", "param-b"])
def test_a_home_is_removed_when_no_phase_failed(inner_run, kind):
    completed, homes, _outcomes = inner_run
    assert not homes[kind].exists(), _output(completed)


def test_an_embedding_model_test_may_keep_a_model_cache_past_the_bound(inner_run):
    completed, homes, outcomes = inner_run
    assert outcomes["test_declared_model_keeps_a_legitimate_cache"] == {}, _output(completed)
    assert not homes["model-cache"].exists()
