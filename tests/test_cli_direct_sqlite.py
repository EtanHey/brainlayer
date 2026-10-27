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
from typing import NamedTuple

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


class _Process(NamedTuple):
    ppid: int
    pgid: int
    started: float
    command: str


def _process_snapshot() -> dict[int, _Process]:
    result = subprocess.run(
        ["/bin/ps", "-ww", "-axo", "pid=,ppid=,pgid=,lstart=,command="],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    processes = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 8)
        started = time.mktime(time.strptime(" ".join(fields[3:8]), "%a %b %d %H:%M:%S %Y"))
        processes[int(fields[0])] = _Process(
            int(fields[1]), int(fields[2]), started, fields[8] if len(fields) > 8 else ""
        )
    return processes


def _daemon_pids(cli_pid, processes, before, cli_started, descendants):
    """Detect group/tree daemons and new init orphans, without host-global attribution.

    A double-forked child adopted by a non-init subreaper is not detected;
    ps exposes no subreaper flag. Never whitelist all ancestors as reapers.
    """
    owned = {cli_pid} | {pid for pid, started in descendants if pid in processes and processes[pid].started == started}
    while True:
        children = {pid for pid, process in processes.items() if process.ppid in owned}
        if children <= owned:
            break
        owned |= children
    descendants.update((pid, processes[pid].started) for pid in owned if pid in processes)
    return {
        (pid, process.started)
        for pid, process in processes.items()
        if pid != cli_pid
        and re.search(r"brainlayer[.-]daemon", process.command)
        and (
            process.pgid == cli_pid
            or pid in owned
            or (process.ppid == 1 and (pid, process.started) not in before and process.started >= cli_started)
        )
    }


def _run_cli(argv: list[str], *, env: dict[str, str], cwd: Path = REPO_ROOT, timeout: float = 10):
    # A fresh session isolates this CLI from production and other test processes.
    before = {(pid, process.started) for pid, process in _process_snapshot().items()}
    # Portable ps lstart has one-second resolution. The baseline excludes existing
    # orphans; a new orphan is a conservative signal, not proof of ancestry.
    cli_started = math.floor(time.time())
    with subprocess.Popen(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    ) as cli:
        seen: set[tuple[int, float]] = set()
        descendants: set[tuple[int, float]] = set()
        deadline = time.monotonic() + timeout
        timed_out = False
        try:
            while True:
                processes = _process_snapshot()
                if cli.pid in processes:
                    cli_started = processes[cli.pid].started
                seen.update(_daemon_pids(cli.pid, processes, before, cli_started, descendants))
                try:
                    stdout, stderr = cli.communicate(timeout=0.05)
                    seen.update(_daemon_pids(cli.pid, _process_snapshot(), before, cli_started, descendants))
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
                # New-orphan timing is insufficient custody for a signal. Clean
                # only the created group or descendants observed through ppid.
                current = _process_snapshot()
                for pid, started in seen:
                    if (
                        pid in current
                        and current[pid].started == started
                        and (current[pid].pgid == cli.pid or (pid, started) in descendants)
                    ):
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                cli.communicate(timeout=2)
        # Processes that finish between polls remain unobservable, as in the old check.
        assert not seen, f"CLI spawned daemon descendants or new daemon orphans: {sorted(pid for pid, _ in seen)}"
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


def test_cli_timeout_does_not_wait_on_descendants_inheriting_output(tmp_path):
    pid_file = tmp_path / "timeout-child.pid"
    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)', 'com.brainlayer.daemon'])\n"
        "open(sys.argv[1], 'w').write(str(child.pid))\n"
        "time.sleep(10)\n"
    )
    started = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_cli([sys.executable, "-c", code, str(pid_file)], env=os.environ.copy(), timeout=0.5)
        assert time.monotonic() - started < 2
        state = subprocess.run(["/bin/ps", "-p", pid_file.read_text(), "-o", "stat="], capture_output=True, text=True)
        assert not state.stdout.strip() or state.stdout.strip().startswith("Z")
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("shape", ["setsid-alive", "setsid-exited", "double-fork"])
def test_cli_check_detects_and_cleans_escaped_daemons(tmp_path, shape):
    pid_file = tmp_path / "escaped.pid"
    child_code = "import os, sys, time\n"
    if shape == "double-fork":
        child_code += "if os.fork(): sys.exit(0)\nos.setsid()\nif os.fork(): os._exit(0)\n"
    child_code += (
        "open(sys.argv[1], 'w').write(str(os.getpid()))\ntime.sleep("
        + ("1" if shape == "setsid-alive" else "5")
        + ")\n"
    )
    code = (
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1], 'padding' * 30, 'com.brainlayer.daemon'], "
        "start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "deadline = time.monotonic() + 2\n"
        "while not pathlib.Path(sys.argv[1]).exists() and time.monotonic() < deadline: time.sleep(.01)\n"
    )
    if shape == "setsid-alive":
        # Reap before CLI exit: only the ppid walk can observe this live daemon.
        code += "child.wait()\n"
    elif shape != "setsid-exited":
        code += "time.sleep(.3)\n"
    try:
        with pytest.raises(AssertionError, match="CLI spawned daemon"):
            _run_cli([sys.executable, "-c", code, str(pid_file), child_code], env=os.environ.copy())
        child_pid = int(pid_file.read_text())
        # The PID file proves this fixture's custody even when a double fork
        # escapes between polls. The detector must not signal unknown orphans.
        try:
            os.kill(child_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            state = subprocess.run(["/bin/ps", "-p", str(child_pid), "-o", "stat="], capture_output=True, text=True)
            if not state.stdout.strip() or state.stdout.strip().startswith("Z"):
                break
            time.sleep(0.01)
        state = subprocess.run(["/bin/ps", "-p", str(child_pid), "-o", "stat="], capture_output=True, text=True)
        assert not state.stdout.strip() or state.stdout.strip().startswith("Z")
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_cli_check_ignores_a_preexisting_orphan(tmp_path):
    pid_file = tmp_path / "outside-orphan.pid"
    code = (
        "import os, sys, time\n"
        "if os.fork(): sys.exit(0)\nos.setsid()\nif os.fork():\n time.sleep(.3)\n os._exit(0)\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\ntime.sleep(5)\n"
    )
    identity = None
    try:
        subprocess.run(
            [sys.executable, "-c", code, str(pid_file), "com.brainlayer.daemon"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        deadline = time.monotonic() + 2
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        pid = int(pid_file.read_text())
        while True:
            process = _process_snapshot()[pid]
            if process.ppid == 1 or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        identity = (pid, process.started)
        assert process.ppid == 1
        result = _run_cli([sys.executable, "-c", "pass"], env=os.environ.copy())
        assert result.returncode == 0
        assert _process_snapshot()[pid].started == process.started
    finally:
        if identity is not None:
            pid, started = identity
            current = _process_snapshot()
            if pid in current and current[pid].started == started:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


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
    # The supervisor's launchd label is not an import of the removed Python daemon.
    swift_text = swift_text.replace("com.brainlayer.daemon", "")
    assert not any(ref in swift_text for ref in forbidden_refs)


@pytest.mark.parametrize("module", ["brainlayer.daemon", "brainlayer.client"])
def test_swift_daemon_independence_guard_rejects_real_module_reference(monkeypatch, module) -> None:
    original_read = Path.read_text
    target = REPO_ROOT / "brain-bar/Sources/BrainBarDaemon/BrainBarDaemonMain.swift"

    def injected_read(path, *args, **kwargs):
        text = original_read(path, *args, **kwargs)
        return text + f'\nlet removedPythonModule = "{module}"\n' if path == target else text

    monkeypatch.setattr(Path, "read_text", injected_read)
    with pytest.raises(AssertionError):
        test_brainbar_swift_daemon_is_independent_of_python_daemon()
