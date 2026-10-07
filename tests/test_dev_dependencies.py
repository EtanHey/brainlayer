"""Retired SDKs are absent while local imports and Drive dependencies remain."""

import ast
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
RETIRED = {"google-genai", "google-generativeai", "groq"}
SDK_MODULES = ("google.genai", "google.generativeai", "groq")


def _dependency_name(value):
    return canonicalize_name(Requirement(value).name)


def test_retired_sdks_are_not_runtime_or_optional_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    dependencies = project["dependencies"] + [
        dependency for group in project["optional-dependencies"].values() for dependency in group
    ]
    assert not RETIRED.intersection(map(_dependency_name, dependencies))
    assert "cloud" not in project["optional-dependencies"]


def test_drive_backup_dependencies_survive_sdk_retirement():
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert {"google-api-python-client", "google-auth", "google-auth-oauthlib", "requests"}.issubset(
        map(_dependency_name, dependencies)
    )


def test_no_direct_retired_sdk_import_across_shipped_code_and_tests():
    imports = []
    for directory in ("src", "scripts", "hooks", "tests"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [f"{node.module}.{alias.name}" for alias in node.names]
                else:
                    continue
                imports.extend(
                    f"{path.relative_to(ROOT)}:{node.lineno}: {name}"
                    for name in names
                    if any(name == sdk or name.startswith(sdk + ".") for sdk in SDK_MODULES)
                )
    assert not imports, "\n".join(imports)


def test_installed_import_gate_does_not_request_retired_cloud_extra():
    script = (ROOT / "scripts/installed_import_gate.sh").read_text()
    assert 'pip install "${wheel}[cloud]"' not in script
    assert 'pip install "$wheel"' in script
