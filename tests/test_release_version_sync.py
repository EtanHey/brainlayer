"""Release metadata version consistency checks."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from brainlayer import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_stay_in_sync() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_version = pyproject["project"]["version"]

    server_manifest = json.loads((REPO_ROOT / "server.json").read_text(encoding="utf-8"))

    assert __version__ == package_version
    assert server_manifest["version"] == package_version
    assert server_manifest["packages"][0]["version"] == package_version
