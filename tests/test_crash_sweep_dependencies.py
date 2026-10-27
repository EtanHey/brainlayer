"""Derive PR coverage from sweep imports instead of maintaining a module list."""

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_gate():
    spec = importlib.util.spec_from_file_location("gate_dependencies", ROOT / "scripts/ci/crash_sweep_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


def test_every_marked_sweep_and_transitive_dependency_requires_sweeps():
    gate = load_gate()
    reached = gate.sweep_dependencies()
    marked = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("test_*.py")
        if any(
            isinstance(node, ast.Attribute) and node.attr == "crash_sweep"
            for node in ast.walk(ast.parse(path.read_text()))
        )
    }
    assert marked and marked <= reached
    # Regression anchors identified by the owner's independent import walk.
    assert {
        "src/brainlayer/__init__.py",
        "src/brainlayer/component_jobs.py",
        "src/brainlayer/config.py",
        "src/brainlayer/consolidation_plan.py",
        "src/brainlayer/drive_credentials.py",
        "src/brainlayer/hook_python.py",
        "src/brainlayer/job_alerts.py",
        "src/brainlayer/jobs.py",
        "src/brainlayer/paths.py",
        "src/brainlayer/pause.py",
        "src/brainlayer/retired_services.py",
        "tests/test_consolidation_plan.py",
        "tests/test_supervisor_jobs.py",
        "pyproject.toml",
        "tests/conftest.py",
        "scripts/run_tests.sh",
        "scripts/check_stray_apps.py",
        "src/brainlayer/vector_store.py",  # Imported by the recovery-oracle collection conftest.
        "tests/benchmarks/conftest.py",
        "tests/recovery_oracle/conftest.py",
    } <= reached
    # This loop checks self-consistency; the pinned anchors and synthetic fixture guard discovery.
    for path in reached:
        assert gate.required("pull_request", [path], []), path


def test_added_sweep_imports_are_followed_without_updating_a_path_list(tmp_path):
    gate = load_gate()
    sources = {
        "tests/test_new_sweep.py": "import pytest\nfrom tests.helper import helper\n@pytest.mark.crash_sweep\ndef test_cut(): pass\n",
        "tests/helper.py": 'import importlib\nfrom scripts.helper import value\nhelper = importlib.import_module("brainlayer.first")\n',
        "scripts/helper.py": "from top_helper import value\nfrom other_package import sibling\nfrom plugin import exported\nfrom custom_root_module import custom\n",
        "top_helper.py": "value = 1\n",
        "src/other_package/__init__.py": "from . import sibling\n",
        "src/other_package/sibling.py": "",
        ".plugins/plugin.py": "exported = True\n",
        "custom/custom_root_module.py": "custom = True\n",
        "src/brainlayer/__init__.py": "from . import sibling\n",
        "src/brainlayer/sibling.py": "",
        "src/brainlayer/first.py": 'from .second import value\nPROBE = "from brainlayer.probe import run"\nARGV = ["-m", "brainlayer.argv_only"]\n',
        "src/brainlayer/second.py": "from .pkg import child\nvalue = 1\n",
        "src/brainlayer/pkg/__init__.py": "",
        "src/brainlayer/pkg/child.py": "import brainlayer.first\n",
        "src/brainlayer/probe.py": "run = None\n",
        "src/brainlayer/argv_only.py": "value = 1\n",
        "src/brainlayer/unrelated.py": "",
    }
    for name, content in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["src", ".plugins"]\n')
    (tmp_path / "tests/conftest.py").write_text('import sys\nsys.path.append("custom")\n')
    reached = gate.sweep_dependencies(tmp_path)
    assert set(sources) - {"src/brainlayer/unrelated.py"} <= reached
    assert "src/brainlayer/unrelated.py" not in reached
    for name in set(sources) - {"src/brainlayer/unrelated.py"}:
        assert gate.required("pull_request", [name], [], root=tmp_path), name


def test_withdrawn_unrelated_modules_skip_the_migration_sweeps():
    gate = load_gate()
    for name in ("maintenance_calendar", "maintenance", "watcher", "watcher_bridge", "backup_daily"):
        path = f"src/brainlayer/{name}.py"
        assert path not in gate.sweep_dependencies()
        assert not gate.required("pull_request", [path], [])
    assert not gate.required("pull_request", ["src/brainlayer/pipeline/cluster_sampling.py"], [])


@pytest.mark.parametrize("module", ["scripts.missing_helper", "third_party_that_does_not_exist_anywhere"])
def test_unresolved_import_fails_instead_of_skipping_sweeps(tmp_path, module):
    gate = load_gate()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/__init__.py").write_text("")
    (tmp_path / "tests/test_cut.py").write_text(
        f"import pytest\nimport {module}\n@pytest.mark.crash_sweep\ndef test_cut(): pass\n"
    )
    with pytest.raises(RuntimeError, match="Unresolved import"):
        gate.sweep_dependencies(tmp_path)


def test_declared_optional_dependency_does_not_require_loading_its_model(tmp_path, monkeypatch):
    gate = load_gate()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_cut.py").write_text(
        "import pytest\nimport tree_sitter_languages\n@pytest.mark.crash_sweep\ndef test_cut(): pass\n"
    )
    (tmp_path / "pyproject.toml").write_text('[project.optional-dependencies]\nast = ["tree-sitter-languages>=1"]\n')
    original = gate.external_module
    monkeypatch.setattr(
        gate, "external_module", lambda name: False if name == "tree_sitter_languages" else original(name)
    )
    assert "tests/test_cut.py" in gate.sweep_dependencies(tmp_path)


def test_external_resolution_does_not_execute_package_initializers(tmp_path, monkeypatch):
    gate = load_gate()
    package = tmp_path / "external_fixture_package"
    package.mkdir()
    (package / "__init__.py").write_text("raise AssertionError('external package was executed')\n")
    (package / "child.py").write_text("value = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert gate.external_module("external_fixture_package.child")


@pytest.mark.parametrize("conftest", ["conftest.py", "tests/conftest.py", "tests/nested/conftest.py"])
def test_collection_conftest_imports_require_sweeps_transitively(tmp_path, conftest):
    gate = load_gate()
    sources = {
        "tests/nested/test_cut.py": "import pytest\n@pytest.mark.crash_sweep\ndef test_cut(): pass\n",
        conftest: "from scripts.helper2 import value\n",
        "scripts/helper2.py": "from scripts.helper3 import value\n",
        "scripts/helper3.py": "value = 1\n",
        "outside/conftest.py": "import nonexistent_outside_collection\n",
    }
    for name, text in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    reached = gate.sweep_dependencies(tmp_path)
    assert {conftest, "scripts/helper2.py", "scripts/helper3.py"} <= reached
    assert gate.required("pull_request", ["scripts/helper2.py"], [], root=tmp_path)
    assert gate.required("pull_request", ["scripts/helper3.py"], [], root=tmp_path)
    assert "outside/conftest.py" not in reached
