"""Shared test fixtures for BrainLayer tests."""

import os

# Git exports repository-routing variables to hooks. They override even explicit `git -C` calls,
# so a pytest fixture that creates a disposable repository can otherwise write into the checkout
# whose pre-push hook launched the suite. Scrub the whole family before test modules, fixtures, or
# subprocesses can observe it.
for _git_env_key in tuple(os.environ):
    if _git_env_key.startswith("GIT_"):
        os.environ.pop(_git_env_key, None)

import functools
import importlib
import sqlite3
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

_PROTECTED_TEST_HOME = Path.home().resolve()
_OPERATOR_BRAINBAR_SOCKET_PATH = os.environ.get("BRAINBAR_SOCKET_PATH")
_OPERATOR_MCP_SOCKET = os.environ.get("BRAINLAYER_MCP_SOCKET")

# Some test modules import brainlayer.config during collection. Import it first while HOME points
# at an empty directory so its production entrypoint behavior cannot load the developer's env file
# into the pytest process or its subprocesses. A deliberately forced early plugin import (`-p
# brainlayer.config`) happens before conftest discovery; in that shape, discard every assignment
# named by the user env file before reloading the module under the same isolated HOME.
_brainlayer_config = sys.modules.get("brainlayer.config")
if _brainlayer_config is not None:
    try:
        _user_env_lines = (
            (_PROTECTED_TEST_HOME / ".config" / "brainlayer" / "brainlayer.env")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    except (OSError, UnicodeDecodeError):
        _user_env_lines = []
    for _line in _user_env_lines:
        _assignment = _brainlayer_config._split_env_assignment(_line)
        if _assignment is not None:
            os.environ.pop(_assignment[0], None)

_original_home = os.environ.get("HOME")
with tempfile.TemporaryDirectory(prefix="brainlayer-pytest-import-home-") as _import_home:
    os.environ["HOME"] = _import_home
    if _brainlayer_config is None:
        importlib.import_module("brainlayer.config")
    else:
        importlib.reload(_brainlayer_config)
if _original_home is None:
    os.environ.pop("HOME", None)
else:
    os.environ["HOME"] = _original_home

import pytest


def _git_probe(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _shared_git_snapshot(common_dir: Path) -> dict[str, str]:
    commands = {
        "core.bare": ("config", "--local", "--get", "core.bare"),
        "user.*": ("config", "--local", "--null", "--get-regexp", "^user\\."),
        "origin/main": ("rev-parse", "--verify", "--quiet", "refs/remotes/origin/main"),
    }
    snapshot: dict[str, str] = {}
    for label, command in commands.items():
        result = _git_probe(f"--git-dir={common_dir}", *command)
        if result.returncode not in (0, 1):
            pytest.fail(
                f"shared Git guard could not read {label}: {result.stderr.strip() or f'exit {result.returncode}'}"
            )
        snapshot[label] = "<absent>" if label == "origin/main" and result.returncode == 1 else result.stdout
    return snapshot


@pytest.fixture(scope="session", autouse=True)
def guard_shared_git_state():
    """Fail the session if any test rewrites checkout-global Git state."""
    repo_root = Path(__file__).resolve().parents[1]
    common_result = _git_probe("rev-parse", "--git-common-dir", cwd=repo_root)
    if common_result.returncode != 0:
        yield
        return

    common_dir = Path(common_result.stdout.strip())
    if not common_dir.is_absolute():
        common_dir = (repo_root / common_dir).resolve()
    bare_result = _git_probe("rev-parse", "--is-bare-repository", cwd=repo_root)
    if bare_result.returncode != 0:
        pytest.fail(f"shared Git guard could not classify checkout: {bare_result.stderr.strip()}")
    if bare_result.stdout.strip() != "false":
        pytest.fail(
            "shared Git guard requires a non-bare checkout at session start; "
            f"got {bare_result.stdout.strip()!r} for {repo_root}"
        )

    initial = _shared_git_snapshot(common_dir)
    yield
    final = _shared_git_snapshot(common_dir)
    changed = [label for label in initial if initial[label] != final[label]]
    if changed:
        pytest.fail("shared Git state changed during pytest: " + ", ".join(changed))


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
    """Register custom pytest marks, and arm the DB-open guards before anything is collected."""
    _install_db_open_guards()
    helper = Path(__file__).resolve().parent / "socket_hygiene" / "sitecustomize.py"
    if helper.is_file():
        spec = importlib.util.spec_from_file_location("_brainlayer_test_socket_hygiene", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    os.environ["BRAINLAYER_FORBID_BRAINBAR_SOCKET"] = "1"
    missing_socket = f"/tmp/brainlayer-pytest-{uuid.uuid4().hex}.sock"
    os.environ["BRAINBAR_SOCKET_PATH"] = missing_socket
    os.environ["BRAINLAYER_MCP_SOCKET"] = missing_socket
    if helper.is_file():
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (str(helper.parent), os.environ.get("PYTHONPATH"))))
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
def isolate_fallback_gits_root(monkeypatch, request):
    """Keep the brain_store fallback queue's root out of the developer's real `~/Gits`.

    `BRAINLAYER_FALLBACK_GITS_ROOT` is an ABSOLUTE path, so unlike everything resolved from
    `Path.home()` it walks straight past the HOME remapping below. With it exported, a test that
    reaches the ratchet's debt row or the drain's startup sweep would inventory -- and `queue_entry`
    could write markers into -- production `docs.local` files, and exit-0 `main()` assertions would
    become host-dependent. Scrubbed for every unmarked test; `integration`/`live` keep whatever the
    operator set, same boundary as the fixture below.
    """
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("live"):
        return
    monkeypatch.delenv("BRAINLAYER_FALLBACK_GITS_ROOT", raising=False)


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
def forbid_live_brainbar_socket(monkeypatch, request):
    """Redirect BrainBar resolution and refuse direct live connects, including child Pythons."""
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("live"):
        monkeypatch.delenv("BRAINLAYER_FORBID_BRAINBAR_SOCKET", raising=False)
        for key, original in (
            ("BRAINBAR_SOCKET_PATH", _OPERATOR_BRAINBAR_SOCKET_PATH),
            ("BRAINLAYER_MCP_SOCKET", _OPERATOR_MCP_SOCKET),
        ):
            if original is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, original)
        return
    missing_socket = f"/tmp/brainlayer-pytest-{uuid.uuid4().hex}.sock"
    monkeypatch.setenv("BRAINLAYER_FORBID_BRAINBAR_SOCKET", "1")
    monkeypatch.setenv("BRAINBAR_SOCKET_PATH", missing_socket)
    monkeypatch.setenv("BRAINLAYER_MCP_SOCKET", missing_socket)


# --------------------------------------------------------------------------------------------
# Suite hygiene: no test loads an embedding model, and no test opens the canonical DB.
#
# Both rules exist because both were broken, expensively. The pre-push full-suite fallback spawned
# `scripts/reembed_bgem3.py --test` -- a 2.5 GB-RSS embedding model holding ~20 fds on the
# production DB -- which is the measured cause of a 14:22 UI stall on the M4. `paths.py` already
# guards path RESOLUTION; these guard the two things that actually cost: the model LOAD and the
# connection OPEN, including the ones that reach them without going through `paths.py` at all.
#
# Each guard has its OWN escape. `embedding_model` lifts only the model guard -- `run_tests.sh`
# deselects it, so it runs where a run declares it can afford a model (CI, which warms the HF cache
# on purpose). `integration`/`live` lift only the DB guard, keeping the meaning they already had.
# A test that needs both declares both.
# --------------------------------------------------------------------------------------------

EMBEDDING_MODEL_MARK = "embedding_model"
EMBEDDING_MODEL_MODULES = ("sentence_transformers", "FlagEmbedding")
# Set for every unmarked test and inherited by subprocesses, which is the point: a test that SPAWNS
# a re-embedding script loads a model just as surely as one that imports it, and sys.modules cannot
# see that happen. Every load site in brainlayer/ and scripts/ checks this before constructing a
# model, so `--help` and syntax probes on the same scripts stay free.
FORBID_MODEL_LOAD_ENV = "BRAINLAYER_FORBID_EMBEDDING_MODEL"
# Armed for EVERY test with no exemption marker, because unlike a model load or a DB open there is
# no legitimate reason for a test to put a popup on the developer's screen. Checked at each
# notification site rather than by patching, so it survives the CLI subprocesses the suite spawns.
FORBID_DESKTOP_NOTIFICATION_ENV = "BRAINLAYER_FORBID_DESKTOP_NOTIFICATION"
_PROTECTED_BRAINLAYER_ROOTS = (
    _PROTECTED_TEST_HOME / ".brainlayer",
    _PROTECTED_TEST_HOME / ".local" / "share" / "brainlayer",
)
# TWO switches, deliberately not one. `embedding_model` says "this test loads a real model"; it says
# nothing about the production DB. Collapsing both into one bit put the hole exactly at
# *model + canonical DB together*, which is the incident these guards exist to prevent: a test
# marked only `embedding_model` got the DB guard lifted too, and `scripts/reembed_bgem3.py` opens
# its `--db` (defaulting to the CANONICAL path, `_get_default_db()`) at main():277 -- BEFORE
# `load_model()` at :285. A test that genuinely needs both must declare both.
DB_GUARD_EXEMPT_MARKS = ("integration", "live")


def hygiene_exemptions(node) -> tuple[bool, bool]:
    """`(model guard lifted, DB guard lifted)` for *node*, from its markers."""
    model_exempt = node.get_closest_marker(EMBEDDING_MODEL_MARK) is not None
    db_exempt = any(node.get_closest_marker(mark) for mark in DB_GUARD_EXEMPT_MARKS)
    return model_exempt, db_exempt


class _DbGuardState:
    """The session-level switch the module-scope DB guards read.

    An attribute rather than a module global: the guards are installed once before collection and
    live for the whole session, so the per-test exemption has to reach them through some shared
    piece of state -- and rebinding a class attribute says where that state lives without a
    `global` statement scattered through the fixture (DeepSource, #755).
    """

    suspended = False


def _sqlite_target_path(target: object) -> Path | None:
    """The filesystem path a sqlite3/apsw target names, or None when it names no local file.

    `file:` URIs are parsed rather than fed to `Path`. BrainLayer's read paths open the DB as
    `file:{db_path}?mode=ro` almost everywhere (`backup_daily`, `maintenance`, `kg_judge`,
    `t3_provenance`, …), and `Path("file:///Users/…/brainlayer.db?mode=ro")` resolves to a
    nonexistent relative path — so a string guard would have missed every real reader in the
    codebase while looking like it worked (CodeRabbit, #755).
    """
    if isinstance(target, int) or target is None:
        return None
    text = os.fsdecode(target) if isinstance(target, (bytes, os.PathLike)) else str(target)
    if not text or text.startswith(":"):
        return None
    if text.startswith("file:"):
        parsed = urllib.parse.urlparse(text)
        if parsed.netloc not in ("", "localhost"):
            return None
        text = urllib.request.url2pathname(parsed.path)
        if not text or text.startswith(":"):
            return None
    try:
        return Path(text).expanduser().resolve(strict=False)
    except (OSError, ValueError, RuntimeError):
        return None


def _is_protected_runtime_path(candidate: object) -> bool:
    """Whether *candidate* names a file inside the real user's BrainLayer runtime state."""
    resolved = _sqlite_target_path(candidate)
    if resolved is None:
        return False
    return any(resolved == root or root in resolved.parents for root in _PROTECTED_BRAINLAYER_ROOTS)


def _refuse_protected_db(target: object, opener: str) -> None:
    if _DbGuardState.suspended or not _is_protected_runtime_path(target):
        return
    raise RuntimeError(
        f"suite hygiene: this test opened the canonical BrainLayer DB via {opener} ({target}). "
        "Tests never touch ~/.local/share/brainlayer; mark the test `integration` or `live` if it "
        "is a deliberate production-DB check."
    )


def _install_db_open_guards() -> None:
    """Wrap the DB entry points ONCE, from `pytest_configure` — before any test module is imported.

    Before collection, because a module that binds `from sqlite3 import connect` (or
    `from apsw import Connection`) at import time captures the original callable and would never
    see a fixture-scoped patch (CodeRabbit, #755). The wrappers read `_DbGuardState.suspended`, so
    marker-based exemptions still work per test.
    """
    if not getattr(sqlite3.connect, "_brainlayer_hygiene_guard", False):
        sqlite_connect = sqlite3.connect

        @functools.wraps(sqlite_connect)
        def guarded_sqlite_connect(database, *args, **kwargs):
            _refuse_protected_db(database, "sqlite3.connect")
            return sqlite_connect(database, *args, **kwargs)

        guarded_sqlite_connect._brainlayer_hygiene_guard = True
        sqlite3.connect = guarded_sqlite_connect

    try:
        import apsw
    except ImportError:  # apsw is a hard dependency, but a guard must never be the thing that fails
        return
    if getattr(apsw.Connection, "_brainlayer_hygiene_guard", False):
        return

    class GuardedConnection(apsw.Connection):
        _brainlayer_hygiene_guard = True

        def __init__(self, filename, *args, **kwargs):
            _refuse_protected_db(filename, "apsw.Connection")
            super().__init__(filename, *args, **kwargs)

    apsw.Connection = GuardedConnection


@pytest.fixture(autouse=True)
def forbid_embedding_models_and_canonical_db(monkeypatch, request):
    """Fail a test that loads an embedding model or opens the canonical BrainLayer DB."""
    model_exempt, db_exempt = hygiene_exemptions(request.node)
    previous = _DbGuardState.suspended
    _DbGuardState.suspended = db_exempt
    try:
        # No marker lifts this one: a test has no business notifying a human.
        monkeypatch.setenv(FORBID_DESKTOP_NOTIFICATION_ENV, "1")
        if not model_exempt:
            _arm_embedding_model_refusal(monkeypatch)
        yield
    finally:
        _DbGuardState.suspended = previous


def _arm_embedding_model_refusal(monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
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
                monkeypatch.setattr(module, attribute, refuse)


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
