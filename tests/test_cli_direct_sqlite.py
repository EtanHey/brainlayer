from __future__ import annotations

import importlib
import importlib.util
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cli_stats_p95_budget_seconds() -> float:
    return 0.750 if os.environ.get("CI") else 0.200


def test_cli_stats_p95_budget_is_strict_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)

    assert _cli_stats_p95_budget_seconds() == 0.200


def test_cli_stats_p95_budget_allows_shared_ci_runner_variance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "true")

    assert _cli_stats_p95_budget_seconds() == 0.750


def _daemon_pids(cli_pgid: int) -> list[str]:
    result = subprocess.run(
        ["/bin/ps", "-ww", "-axo", "pid=,pgid=,command="],
        capture_output=True,
        text=True,
        check=True,
    )
    owned = []
    for line in result.stdout.splitlines():
        pid, pgid, command = line.strip().split(None, 2)
        if int(pgid) == cli_pgid and int(pid) != cli_pgid and re.search(r"brainlayer[.-]daemon", command):
            owned.append(pid)
    return owned


def _run_cli(argv: list[str], *, env: dict[str, str], cwd: Path = REPO_ROOT, timeout: float = 10):
    # A fresh session isolates this CLI from production and other test processes.
    with subprocess.Popen(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    ) as cli:
        seen: set[str] = set()
        deadline = time.monotonic() + timeout
        timed_out = False
        try:
            while True:
                seen.update(_daemon_pids(cli.pid))
                try:
                    stdout, stderr = cli.communicate(timeout=0.05)
                    seen.update(_daemon_pids(cli.pid))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        timed_out = True
                        raise
        finally:
            if cli.poll() is None or timed_out or seen:
                # This group was created by this Popen, never selected by a name pattern.
                try:
                    os.killpg(cli.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                cli.communicate(timeout=2)
        assert not seen, f"CLI spawned daemon descendants: {sorted(seen)}"
        return subprocess.CompletedProcess(argv, cli.returncode, stdout, stderr)


def _seed_empty_db(db_path: Path) -> None:
    from brainlayer.vector_store import VectorStore

    store = VectorStore(db_path)
    store.close()


def test_cli_search_does_not_spawn_or_contact_daemon_process(tmp_path: Path) -> None:
    db_path = tmp_path / "brainlayer.db"
    _seed_empty_db(db_path)

    env = os.environ.copy()
    env["BRAINLAYER_DB"] = str(db_path)
    env["PATH"] = str(tmp_path)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    result = _run_cli(
        [sys.executable, "-c", "from brainlayer.cli import app; app()", "search", "foo", "--text", "--num", "1"],
        cwd=REPO_ROOT,
        env=env,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "daemon" not in (result.stderr + result.stdout).lower()


def test_cli_stats_missing_db_fails_closed_without_bootstrap(tmp_path: Path) -> None:
    db_path = tmp_path / "brainlayer.db"

    env = os.environ.copy()
    env["BRAINLAYER_DB"] = str(db_path)
    env["PATH"] = str(tmp_path)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    result = _run_cli(
        [sys.executable, "-c", "from brainlayer.cli import app; app()", "stats"],
        cwd=REPO_ROOT,
        env=env,
    )

    assert result.returncode == 1
    assert "migrate-store <copy.db>" in result.stdout
    assert not db_path.exists()


def test_cli_stats_empty_existing_db_fails_closed_without_bootstrap(tmp_path: Path) -> None:
    db_path = tmp_path / "brainlayer.db"
    db_path.touch()

    env = os.environ.copy()
    env["BRAINLAYER_DB"] = str(db_path)
    env["PATH"] = str(tmp_path)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    result = _run_cli(
        [sys.executable, "-c", "from brainlayer.cli import app; app()", "stats"],
        cwd=REPO_ROOT,
        env=env,
    )

    assert result.returncode == 1
    assert "migrate-store <copy.db>" in result.stdout
    assert db_path.stat().st_size == 0


@pytest.mark.parametrize(
    "check",
    [
        test_cli_search_does_not_spawn_or_contact_daemon_process,
        test_cli_stats_missing_db_fails_closed_without_bootstrap,
        test_cli_stats_empty_existing_db_fails_closed_without_bootstrap,
    ],
)
def test_cli_checks_ignore_an_outside_tree_daemon(tmp_path, check):
    dummy = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "com.brainlayer.daemon"], start_new_session=True
    )
    try:
        check(tmp_path)
    finally:
        dummy.terminate()
        dummy.wait(timeout=5)


def test_cli_check_detects_and_cleans_its_exited_parent_daemon(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    pid_file = tmp_path / "child.pid"
    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)', 'padding' * 30, 'com.brainlayer.daemon'], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "open(sys.argv[1], 'w').write(str(child.pid))\n"
        "time.sleep(.3)\n"
    )
    try:
        with pytest.raises(AssertionError, match="CLI spawned daemon"):
            _run_cli([sys.executable, "-c", code, str(pid_file)], env=os.environ.copy())
        child_pid = int(pid_file.read_text())
        state = subprocess.run(["/bin/ps", "-p", str(child_pid), "-o", "stat="], capture_output=True, text=True)
        assert not state.stdout.strip() or state.stdout.strip().startswith("Z")
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_cli_timeout_does_not_wait_on_descendants_inheriting_output():
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)', 'com.brainlayer.daemon'])\n"
        "time.sleep(10)\n"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        _run_cli([sys.executable, "-c", code], env=os.environ.copy(), timeout=0.1)
    assert time.monotonic() - started < 2


def test_cli_search_opens_vectorstore_readonly_directly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import brainlayer.cli_new as cli_new
    from brainlayer.cli import app

    db_path = tmp_path / "brainlayer.db"
    _seed_empty_db(db_path)
    calls: list[Path] = []

    class SpyStore:
        def __init__(self, path: Path):
            calls.append(Path(path))

        def hybrid_search(self, **kwargs):
            return {"ids": [["chunk-1"]], "documents": [["foo result"]], "metadatas": [[{}]], "distances": [[0.1]]}

        def close(self) -> None:
            pass

    class SpyModel:
        def embed_query(self, query: str) -> list[float]:
            return [0.0] * 8

    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setattr(cli_new, "ReadonlyStore", SpyStore)
    monkeypatch.setattr(cli_new, "get_embedding_model", lambda: SpyModel())

    result = CliRunner().invoke(app, ["search", "foo", "--num", "1"])

    assert result.exit_code == 0, result.output
    assert calls == [db_path]


def test_cli_search_agent_flag_threads_to_hybrid_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import brainlayer.cli_new as cli_new
    from brainlayer.cli import app

    db_path = tmp_path / "brainlayer.db"
    _seed_empty_db(db_path)
    hybrid_calls: list[dict] = []

    class SpyStore:
        def __init__(self, _path: Path):
            pass

        def hybrid_search(self, **kwargs):
            hybrid_calls.append(kwargs)
            return {"ids": [["chunk-1"]], "documents": [["foo result"]], "metadatas": [[{}]], "distances": [[0.1]]}

        def close(self) -> None:
            pass

    class SpyModel:
        def embed_query(self, query: str) -> list[float]:
            return [0.0] * 8

    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setattr(cli_new, "ReadonlyStore", SpyStore)
    monkeypatch.setattr(cli_new, "get_embedding_model", lambda: SpyModel())

    result = CliRunner().invoke(app, ["search", "--agent", "codex-test-agent", "foo", "--num", "1"])

    assert result.exit_code == 0, result.output
    assert hybrid_calls[0]["agent_id"] == "codex-test-agent"


def test_cli_stats_p95_under_200ms_with_direct_readonly_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import brainlayer.cli_new as cli_new
    from brainlayer.cli import app

    db_path = tmp_path / "brainlayer.db"
    _seed_empty_db(db_path)
    calls: list[Path] = []

    class FastStatsStore:
        def __init__(self, path: Path):
            calls.append(Path(path))

        def get_stats(self):
            return {"total_chunks": 0, "projects": [], "content_types": []}

        def close(self) -> None:
            pass

    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setattr(cli_new, "ReadonlyStore", FastStatsStore)

    runner = CliRunner()
    warmup = runner.invoke(app, ["stats"])
    assert warmup.exit_code == 0, warmup.output
    calls.clear()

    durations: list[float] = []
    for _ in range(10):
        start = time.perf_counter()
        result = runner.invoke(app, ["stats"])
        durations.append(time.perf_counter() - start)
        assert result.exit_code == 0, result.output

    p95 = sorted(durations)[math.ceil(len(durations) * 0.95) - 1]
    assert p95 < _cli_stats_p95_budget_seconds()
    assert calls == [db_path] * 10


def test_daemon_and_client_modules_are_gone() -> None:
    import brainlayer

    assert importlib.util.find_spec("brainlayer.daemon") is None
    assert importlib.util.find_spec("brainlayer.client") is None

    with pytest.raises(ImportError):
        importlib.import_module("brainlayer.daemon")
    with pytest.raises(ImportError):
        importlib.import_module("brainlayer.client")

    assert not hasattr(brainlayer, "daemon")
    assert not hasattr(brainlayer, "client")


def test_brainbar_swift_daemon_is_independent_of_python_daemon() -> None:
    assert not (REPO_ROOT / "src/brainlayer/daemon.py").exists()
    assert not (REPO_ROOT / "src/brainlayer/client.py").exists()

    package_text = (REPO_ROOT / "brain-bar/Package.swift").read_text()
    assert 'executable(name: "BrainBarDaemon"' in package_text

    required_surfaces = [
        REPO_ROOT / "brain-bar/Sources/BrainBarDaemon/BrainBarDaemonMain.swift",
        REPO_ROOT / "brain-bar/Sources/BrainBar/BrainBarWindowRootView.swift",
        REPO_ROOT / "brain-bar/Sources/BrainBar/BrainBarDashboardPanelController.swift",
        REPO_ROOT / "brain-bar/Tests/BrainBarTests/BrainBarWindowStateTests.swift",
        REPO_ROOT / "brain-bar/Tests/BrainBarTests/KnowledgeGraphTests.swift",
    ]
    for surface in required_surfaces:
        assert surface.exists(), surface

    swift_roots = [REPO_ROOT / "brain-bar/Sources", REPO_ROOT / "brain-bar/Tests"]
    swift_text = "\n".join(path.read_text() for root in swift_roots for path in root.rglob("*.swift") if path.is_file())
    forbidden_refs = ["brainlayer.daemon", "brainlayer.client", "brainlayer-daemon", "daemon.py", "client.py"]
    assert not any(ref in swift_text for ref in forbidden_refs)
