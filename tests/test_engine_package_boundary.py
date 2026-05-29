from __future__ import annotations

import tomllib
from pathlib import Path

import tests.conftest as test_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_pure_engine_package_boundary() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    wheel_config = payload["tool"]["hatch"]["build"]["targets"]["wheel"]
    sdist_config = payload["tool"]["hatch"]["build"]["targets"]["sdist"]

    assert payload["project"]["name"] == "brainlayer"
    assert payload["project"]["scripts"] == {"brainlayer-mcp": "brainlayer.mcp:serve"}
    assert wheel_config["packages"] == ["src/brainlayer"]
    assert "src/brainlayer/cli" in wheel_config["exclude"]
    assert "src/brainlayer/cli_new.py" in wheel_config["exclude"]
    assert "src/brainlayer/dashboard" in wheel_config["exclude"]
    assert "src/brainlayer/cli/**" in sdist_config["exclude"]
    assert "src/brainlayer/cli_new.py" in sdist_config["exclude"]
    assert "src/brainlayer/dashboard/**" in sdist_config["exclude"]


def test_engine_suite_selection_is_explicit_and_excludes_root_surfaces() -> None:
    assert test_config.ENGINE_TEST_MARK == "engine"
    assert "tests/test_vector_store.py" in test_config.ENGINE_TEST_FILES
    assert "tests/test_search_quality.py" in test_config.ENGINE_TEST_FILES
    assert "tests/test_hybrid_helper_contract.py" in test_config.ENGINE_TEST_FILES

    assert "tests/test_cli_direct_sqlite.py" not in test_config.ENGINE_TEST_FILES
    assert "tests/test_cli_enrich.py" not in test_config.ENGINE_TEST_FILES
    assert "tests/test_dashboard.py" not in test_config.ENGINE_TEST_FILES
    assert "tests/test_brainbar_build_app_guards.py" not in test_config.ENGINE_TEST_FILES
    assert "tests/test_launchd_hygiene.py" not in test_config.ENGINE_TEST_FILES
