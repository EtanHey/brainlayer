"""Tests for the root CLI version probe."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import brainlayer.cli as cli
from brainlayer import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_package_version_without_opening_database(monkeypatch, flag: str) -> None:
    def fail_if_database_opens(*args, **kwargs):
        raise AssertionError("the version probe must not open SQLite")

    monkeypatch.setattr(cli.sqlite3, "connect", fail_if_database_opens)

    result = CliRunner().invoke(cli.app, [flag])

    assert result.exit_code == 0, result.stdout
    assert result.stdout == f"brainlayer {__version__}\n"


def test_version_flag_does_not_initialize_runtime_dependencies(tmp_path: Path) -> None:
    database_path = tmp_path / "must-not-exist.db"
    script = """
import sys

from typer.testing import CliRunner

from brainlayer.cli import app

result = CliRunner().invoke(app, ["--version"])
assert result.exit_code == 0, result.exception
heavy_roots = {"apsw", "numpy", "sentence_transformers", "torch", "transformers"}
loaded = sorted(name for name in sys.modules if name.partition(".")[0] in heavy_roots)
assert not loaded, loaded
print(result.stdout, end="")
"""
    env = os.environ.copy()
    env["BRAINLAYER_DB"] = str(database_path)
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (src_path, env.get("PYTHONPATH"))))

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"brainlayer {__version__}\n"
    assert not database_path.exists()


def test_root_help_keeps_app_description_and_lists_version_option() -> None:
    result = CliRunner().invoke(cli.app, ["--help"])

    assert result.exit_code == 0, result.stdout
    assert "זיכרון - Local knowledge pipeline" in result.stdout
    assert "--version" in result.stdout
    assert "-V" in result.stdout
