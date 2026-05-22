from __future__ import annotations

import importlib
import importlib.util
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[1]


def _daemon_pids() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-f", r"brainlayer[.-]daemon|brainlayer\.daemon"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _seed_empty_db(db_path: Path) -> None:
    from brainlayer.vector_store import VectorStore

    store = VectorStore(db_path)
    store.close()


def test_cli_search_does_not_spawn_or_contact_daemon_process(tmp_path: Path) -> None:
    db_path = tmp_path / "brainlayer.db"
    _seed_empty_db(db_path)

    assert _daemon_pids() == []

    env = os.environ.copy()
    env["BRAINLAYER_DB"] = str(db_path)
    env["PATH"] = str(tmp_path)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        [sys.executable, "-c", "from brainlayer.cli import app; app()", "search", "foo", "--text", "--num", "1"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "daemon" not in (result.stderr + result.stdout).lower()
    assert _daemon_pids() == []


def test_cli_search_opens_vectorstore_readonly_directly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import brainlayer.cli_new as cli_new
    from brainlayer.cli import app

    db_path = tmp_path / "brainlayer.db"
    calls: list[tuple[Path, bool]] = []

    class SpyStore:
        def __init__(self, path: Path, readonly: bool = False):
            calls.append((Path(path), readonly))

        def hybrid_search(self, **kwargs):
            return {"ids": [["chunk-1"]], "documents": [["foo result"]], "metadatas": [[{}]], "distances": [[0.1]]}

        def close(self) -> None:
            pass

    class SpyModel:
        def embed_query(self, query: str) -> list[float]:
            return [0.0] * 8

    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setattr(cli_new, "VectorStore", SpyStore)
    monkeypatch.setattr(cli_new, "get_embedding_model", lambda: SpyModel())

    result = CliRunner().invoke(app, ["search", "foo", "--num", "1"])

    assert result.exit_code == 0, result.output
    assert calls == [(db_path, True)]


def test_cli_stats_p95_under_200ms_with_direct_readonly_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import brainlayer.cli_new as cli_new
    from brainlayer.cli import app

    db_path = tmp_path / "brainlayer.db"
    calls: list[tuple[Path, bool]] = []

    class FastStatsStore:
        def __init__(self, path: Path, readonly: bool = False):
            calls.append((Path(path), readonly))

        def get_stats(self):
            return {"total_chunks": 0, "projects": [], "content_types": []}

        def close(self) -> None:
            pass

    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setattr(cli_new, "VectorStore", FastStatsStore)

    runner = CliRunner()
    durations: list[float] = []
    for _ in range(10):
        start = time.perf_counter()
        result = runner.invoke(app, ["stats"])
        durations.append(time.perf_counter() - start)
        assert result.exit_code == 0, result.output

    p95 = sorted(durations)[math.ceil(len(durations) * 0.95) - 1]
    assert p95 < 0.200
    assert calls == [(db_path, True)] * 10


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
        REPO_ROOT / "brain-bar/Sources/BrainBar/InjectionFeedView.swift",
        REPO_ROOT / "brain-bar/Sources/BrainBar/BrainBarDashboardPanelController.swift",
        REPO_ROOT / "brain-bar/Tests/BrainBarTests/InjectionPresentationTests.swift",
        REPO_ROOT / "brain-bar/Tests/BrainBarTests/KnowledgeGraphTests.swift",
    ]
    for surface in required_surfaces:
        assert surface.exists(), surface

    swift_roots = [REPO_ROOT / "brain-bar/Sources", REPO_ROOT / "brain-bar/Tests"]
    swift_text = "\n".join(path.read_text() for root in swift_roots for path in root.rglob("*.swift") if path.is_file())
    forbidden_refs = ["brainlayer.daemon", "brainlayer.client", "brainlayer-daemon", "daemon.py", "client.py"]
    assert not any(ref in swift_text for ref in forbidden_refs)
