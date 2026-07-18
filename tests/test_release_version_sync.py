"""Release metadata version consistency checks."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import yaml

from brainlayer import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_stay_in_sync() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_version = pyproject["project"]["version"]

    server_manifest = json.loads((REPO_ROOT / "server.json").read_text(encoding="utf-8"))

    assert __version__ == package_version
    assert server_manifest["version"] == package_version
    assert server_manifest["packages"][0]["version"] == package_version


def test_tag_publishers_fail_closed_on_metadata_mismatch() -> None:
    for workflow_name in ("publish.yml", "brainbar-release.yml"):
        workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8"))
        steps = next(iter(workflow["jobs"].values()))["steps"]
        gate = next(
            (step for step in steps if step.get("name") == "Verify release tag matches package version"),
            None,
        )

        assert gate is not None, f"{workflow_name} must reject a tag that disagrees with pyproject.toml"
        run = gate.get("run", "")
        assert "GITHUB_REF_NAME" in run
        assert "pyproject.toml" in run
        assert "exit 1" in run
