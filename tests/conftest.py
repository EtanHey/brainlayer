"""Shared test fixtures for BrainLayer tests."""

import os
import uuid
from pathlib import Path

import pytest

ENGINE_TEST_MARK = "engine"
ENGINE_TEST_EXCLUDED_FILES = {
    "tests/test_agent_profiles.py",
    "tests/test_behavioral_pr_loop.py",
    "tests/test_brainbar_build_app_guards.py",
    "tests/test_cli_direct_sqlite.py",
    "tests/test_cli_enrich.py",
    "tests/test_dashboard.py",
    "tests/test_dev_dependencies.py",
    "tests/test_enrich_defaults.py",
    "tests/test_git_learning.py",
    "tests/test_launchd_hygiene.py",
    "tests/test_newsyslog_config.py",
    "tests/test_run_tests_script.py",
    "tests/test_wizard.py",
}
ENGINE_TEST_EXCLUDED_DIR_PARTS = {
    "tests/mock_mcp",
}
ENGINE_TEST_FILES = frozenset(
    rel_path
    for path in (Path(__file__).resolve().parents[1] / "tests").rglob("test_*.py")
    if (rel_path := path.relative_to(Path(__file__).resolve().parents[1]).as_posix()) not in ENGINE_TEST_EXCLUDED_FILES
    and not any(rel_path.startswith(prefix) for prefix in ENGINE_TEST_EXCLUDED_DIR_PARTS)
)


def pytest_configure(config):
    """Register custom pytest marks."""
    config.addinivalue_line(
        "markers",
        "engine: pure-library engine tests (excludes CLI, dashboard, BrainBar, launchd, and root orchestration surfaces)",
    )
    config.addinivalue_line(
        "markers",
        "live: mark test as requiring a live production DB (skipped in CI if DB absent)",
    )


def pytest_collection_modifyitems(config, items):
    """Make the pure-library engine suite runnable as `pytest -m engine`."""
    for item in items:
        rel_path = Path(str(item.fspath)).resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
        if rel_path in ENGINE_TEST_FILES:
            item.add_marker(pytest.mark.engine)


@pytest.fixture
def eval_project() -> str:
    """Return a unique project name for each eval test case.

    Prevents cross-case data contamination when eval tests seed brain_store
    chunks. Each test invocation gets its own project namespace.
    """
    return f"eval-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def disable_live_gemini_for_unit_tests(monkeypatch, request):
    """Keep unit tests from making live Gemini calls through local shell env."""
    if request.node.get_closest_marker("live"):
        return

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)


@pytest.fixture
def test_user() -> str:
    """Username for path-based tests.

    Defaults to 'janedev' (safe for CI/commits).
    Set BRAINLAYER_TEST_USER to your real username for local filesystem tests.
    """
    return os.environ.get("BRAINLAYER_TEST_USER", "janedev")
