"""Shared test fixtures for BrainLayer tests."""

import os
import sys
import uuid
from pathlib import Path

import pytest

_PROTECTED_TEST_HOME = Path.home().resolve()

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


@pytest.fixture(autouse=True)
def isolate_backup_daily_log(monkeypatch, tmp_path):
    """Keep backup_daily tests and subprocesses from appending to the production heartbeat log."""
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PATH", str(tmp_path / "pytest-backup-daily.log"))
    monkeypatch.setenv("BRAINLAYER_BACKUP_LOG_PROVENANCE", "pytest")


@pytest.fixture(autouse=True)
def isolate_brainlayer_runtime_paths(monkeypatch, tmp_path, request):
    """Keep unit-test runtime resolvers out of production BrainLayer paths."""
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("live"):
        monkeypatch.setenv("BRAINLAYER_TEST_PATH_PROVENANCE", "live")
        return

    isolated_home = tmp_path.parent / f".{tmp_path.name}-brainlayer-home"
    isolated_home.mkdir()
    runtime_root = isolated_home / ".brainlayer"
    queue_dir = runtime_root / "queue"
    data_dir = isolated_home / ".local" / "share" / "brainlayer"

    monkeypatch.setenv("HOME", str(isolated_home))
    monkeypatch.setenv("BRAINLAYER_DB", str(data_dir / "brainlayer.db"))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    monkeypatch.setenv("BRAINLAYER_DRAIN_LOG_PATH", str(runtime_root / "logs" / "drain.log"))
    monkeypatch.setenv("BRAINLAYER_DRAIN_HEALTH_PATH", str(data_dir / "drain-health.json"))
    monkeypatch.setenv("BRAINLAYER_TEST_PATH_PROVENANCE", "pytest")
    monkeypatch.setenv("BRAINLAYER_TEST_PROTECTED_HOME", str(_PROTECTED_TEST_HOME))

    # paths.py caches these at import time, potentially before this fixture replaces HOME.
    from brainlayer import paths as brain_paths

    isolated_db = data_dir / "brainlayer.db"
    monkeypatch.setattr(brain_paths, "_CANONICAL_DB_PATH", isolated_db)
    monkeypatch.setattr(brain_paths, "DEFAULT_DB_PATH", isolated_db)

    protected_roots = (
        (_PROTECTED_TEST_HOME / ".brainlayer", runtime_root),
        (_PROTECTED_TEST_HOME / ".local" / "share" / "brainlayer", data_dir),
    )
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith("brainlayer"):
            continue
        for attribute, value in list(vars(module).items()):
            if not isinstance(value, Path):
                continue
            resolved = value.expanduser().resolve(strict=False)
            for protected_root, isolated_root in protected_roots:
                if resolved == protected_root or protected_root in resolved.parents:
                    monkeypatch.setattr(module, attribute, isolated_root / resolved.relative_to(protected_root))
                    break


@pytest.fixture(autouse=True)
def isolate_writer_telemetry(monkeypatch, tmp_path):
    """Keep writer tests from touching the live telemetry log or heartbeat directory."""
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_PATH", str(tmp_path / "writer-telemetry.jsonl"))
    monkeypatch.setenv("BRAINLAYER_WRITER_HEARTBEAT_DIR", str(tmp_path / "writer-heartbeats"))


@pytest.fixture
def test_user() -> str:
    """Username for path-based tests.

    Defaults to 'janedev' (safe for CI/commits).
    Set BRAINLAYER_TEST_USER to your real username for local filesystem tests.
    """
    return os.environ.get("BRAINLAYER_TEST_USER", "janedev")
