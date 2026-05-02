"""Regression tests for dev-only dependencies used by the pre-push harness."""

import tomllib
from pathlib import Path


def test_google_genai_listed_in_dev_extra() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text())
    dev_dependencies = payload["project"]["optional-dependencies"]["dev"]

    assert any(dep.startswith("google-genai") for dep in dev_dependencies)
