"""Derive PR coverage from sweep imports instead of maintaining a module list."""

import ast
import importlib.util
from pathlib import Path

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
    } <= reached
    for path in reached:
        assert gate.required("pull_request", [path], []), path


def test_added_sweep_imports_are_followed_without_updating_a_path_list(tmp_path):
    gate = load_gate()
    sources = {
        "tests/test_new_sweep.py": "import pytest\nfrom tests.helper import helper\n@pytest.mark.crash_sweep\ndef test_cut(): pass\n",
        "tests/helper.py": 'import importlib\nhelper = importlib.import_module("brainlayer.first")\n',
        "src/brainlayer/__init__.py": "from . import sibling\n",
        "src/brainlayer/sibling.py": "",
        "src/brainlayer/first.py": 'from .second import value\nPROBE = "from brainlayer.probe import run"\n',
        "src/brainlayer/second.py": "from .pkg import child\nvalue = 1\n",
        "src/brainlayer/pkg/__init__.py": "",
        "src/brainlayer/pkg/child.py": "import brainlayer.first\n",
        "src/brainlayer/probe.py": "run = None\n",
        "src/brainlayer/unrelated.py": "",
    }
    for name, content in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    reached = gate.sweep_dependencies(tmp_path)
    assert set(sources) - {"src/brainlayer/unrelated.py"} <= reached
    assert "src/brainlayer/unrelated.py" not in reached
    for name in set(sources) - {"src/brainlayer/unrelated.py"}:
        assert gate.required("pull_request", [name], [], root=tmp_path), name


def test_withdrawn_unrelated_modules_skip_the_migration_sweeps():
    gate = load_gate()
    for name in ("maintenance_calendar", "maintenance", "watcher", "watcher_bridge", "vector_store", "backup_daily"):
        path = f"src/brainlayer/{name}.py"
        assert path not in gate.sweep_dependencies()
        assert not gate.required("pull_request", [path], [])
    assert not gate.required("pull_request", ["src/brainlayer/pipeline/extract.py"], [])
