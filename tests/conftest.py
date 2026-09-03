"""Shared test fixtures for BrainLayer tests."""

import os
import sqlite3
import sys
import uuid
from pathlib import Path

import pytest

_PROTECTED_TEST_HOME = Path.home().resolve()

# Deterministic CLI output for assertions: several tests compare CLI messages as plain
# strings, and ANSI color escapes mid-sentence broke 3 test_installable_build assertions
# in color-enabled environments (2026-08-02; NO_COLOR=1 -> 3 passed, verified). Pin the
# test env here — the single place every pytest path (pre-push gate, CI, dev shell)
# flows through — so pass/fail cannot depend on the caller's terminal.
os.environ["NO_COLOR"] = "1"
os.environ.pop("FORCE_COLOR", None)
os.environ.setdefault("TERM", "dumb")

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


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register externally injectable Wave 5 benchmark implementations."""
    parser.addoption(
        "--wave5-ledger-factory",
        metavar="MODULE:ATTRIBUTE",
        help="Run the Wave 5 ledger contract against an external factory.",
    )
    parser.addoption(
        "--wave5-candidate-producer",
        metavar="MODULE:ATTRIBUTE",
        help="Run the Wave 5 correction benchmark against an external producer.",
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


# --------------------------------------------------------------------------------------------
# Suite hygiene: no test loads an embedding model, and no test opens the canonical DB.
#
# Both rules exist because both were broken, expensively. The pre-push full-suite fallback spawned
# `scripts/reembed_bgem3.py --test` -- a 2.5 GB-RSS embedding model holding ~20 fds on the
# production DB -- which is the measured cause of a 14:22 UI stall on the M4. `paths.py` already
# guards path RESOLUTION; these guard the two things that actually cost: the model LOAD and the
# connection OPEN, including the ones that reach them without going through `paths.py` at all.
#
# The single escape is an explicit marker. `embedding_model` says a test deliberately loads a real
# model; `scripts/run_tests.sh` deselects it, so it runs only where a run declares it can afford
# one (CI, which warms the HF cache on purpose). `integration` and `live` already mean "this test
# is an opt-in against real state" and keep that meaning here.
# --------------------------------------------------------------------------------------------

EMBEDDING_MODEL_MARK = "embedding_model"
EMBEDDING_MODEL_MODULES = ("sentence_transformers", "FlagEmbedding")
# Set for every unmarked test and inherited by subprocesses, which is the point: a test that SPAWNS
# a re-embedding script loads a model just as surely as one that imports it, and sys.modules cannot
# see that happen. Every load site in brainlayer/ and scripts/ checks this before constructing a
# model, so `--help` and syntax probes on the same scripts stay free.
FORBID_MODEL_LOAD_ENV = "BRAINLAYER_FORBID_EMBEDDING_MODEL"
_PROTECTED_BRAINLAYER_ROOTS = (
    _PROTECTED_TEST_HOME / ".brainlayer",
    _PROTECTED_TEST_HOME / ".local" / "share" / "brainlayer",
)
_HYGIENE_EXEMPT_MARKS = (EMBEDDING_MODEL_MARK, "integration", "live")


def _is_protected_runtime_path(candidate: object) -> bool:
    """Whether *candidate* names a file inside the real user's BrainLayer runtime state."""
    if isinstance(candidate, int) or candidate is None:
        return False
    text = str(candidate)
    if not text or text.startswith(":") or text.startswith("file::memory:"):
        return False
    try:
        resolved = Path(text).expanduser().resolve(strict=False)
    except (OSError, ValueError, RuntimeError):
        return False
    return any(resolved == root or root in resolved.parents for root in _PROTECTED_BRAINLAYER_ROOTS)


@pytest.fixture(autouse=True)
def forbid_embedding_models_and_canonical_db(monkeypatch, request):
    """Fail a test that loads an embedding model or opens the canonical BrainLayer DB."""
    if any(request.node.get_closest_marker(mark) for mark in _HYGIENE_EXEMPT_MARKS):
        return

    def _refuse_model(*_args, **_kwargs):
        raise RuntimeError(
            "suite hygiene: this test loaded a real embedding model. Mark it "
            f"`@pytest.mark.{EMBEDDING_MODEL_MARK}` if that is deliberate."
        )

    # The primary guard, because it is the only one that survives a process boundary and the only
    # one placed where the 2.5 GB is actually spent. Blocking the IMPORT instead would be wrong:
    # `pipeline/semantic_style.py` calls `find_spec("sentence_transformers")` at module load and
    # `pipeline/style_embed.py` imports it under a try/except, neither of which loads a model.
    monkeypatch.setenv(FORBID_MODEL_LOAD_ENV, "1")

    # Second net, for a model class reached without going through a brainlayer load site. Patched
    # only where the package is ALREADY imported (tests/test_semantic_style.py imports it at
    # collection time and then mocks the model) -- importing it here to patch it would be the very
    # cost this fixture exists to avoid.
    for name in EMBEDDING_MODEL_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        for attribute in ("SentenceTransformer", "BGEM3FlagModel", "FlagModel"):
            if hasattr(module, attribute):
                monkeypatch.setattr(module, attribute, _refuse_model)

    def _guard_open(database: object, opener: str) -> None:
        if _is_protected_runtime_path(database):
            raise RuntimeError(
                f"suite hygiene: this test opened the canonical BrainLayer DB via {opener} "
                f"({database}). Tests never touch ~/.local/share/brainlayer; mark the test "
                "`integration` or `live` if it is a deliberate production-DB check."
            )

    sqlite_connect = sqlite3.connect

    def guarded_sqlite_connect(database, *args, **kwargs):
        _guard_open(database, "sqlite3.connect")
        return sqlite_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", guarded_sqlite_connect)

    # apsw is patched only when it is already imported: forcing the import here would pay for a C
    # extension in every test that has nothing to do with storage. brainlayer.vector_store imports
    # it at module scope, so any test that can open the DB through apsw has already loaded it.
    apsw = sys.modules.get("apsw")
    if apsw is not None:
        apsw_connection = apsw.Connection

        class GuardedConnection(apsw_connection):
            def __init__(self, filename, *args, **kwargs):
                _guard_open(filename, "apsw.Connection")
                super().__init__(filename, *args, **kwargs)

        monkeypatch.setattr(apsw, "Connection", GuardedConnection)


@pytest.fixture(autouse=True)
def isolate_writer_telemetry(monkeypatch, tmp_path):
    """Keep writer tests from touching the live telemetry log or heartbeat directory."""
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_PATH", str(tmp_path / "writer-telemetry.jsonl"))
    monkeypatch.setenv("BRAINLAYER_WRITER_HEARTBEAT_DIR", str(tmp_path / "writer-heartbeats"))


@pytest.fixture(autouse=True)
def isolate_t3_runtime_state(monkeypatch, tmp_path):
    """Prevent unit provenance tests from reading the developer's live T3 app DB."""
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(tmp_path / "missing-t3-state.sqlite"))


@pytest.fixture
def test_user() -> str:
    """Username for path-based tests.

    Defaults to 'janedev' (safe for CI/commits).
    Set BRAINLAYER_TEST_USER to your real username for local filesystem tests.
    """
    return os.environ.get("BRAINLAYER_TEST_USER", "janedev")
