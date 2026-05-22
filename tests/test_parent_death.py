from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parent_death_watcher_noops_without_kqueue(monkeypatch):
    from brainlayer import parent_death

    monkeypatch.setattr(parent_death.select, "kqueue", None, raising=False)
    assert parent_death.install_parent_death_watcher() is False


def test_long_running_python_sidecar_entrypoints_install_parent_death_watcher():
    entrypoints = {
        "src/brainlayer/mcp/__init__.py",
        "src/brainlayer/drain.py",
        "src/brainlayer/pipeline/enrichment.py",
        "src/brainlayer/brainbar_hybrid_helper.py",
        "src/brainlayer/cli/__init__.py",
        "brain-bar/Scripts/brainbar_stdio_adapter.py",
    }

    for relative in entrypoints:
        text = (REPO_ROOT / relative).read_text()
        assert "install_parent_death_watcher" in text, relative


@pytest.mark.skipif(not hasattr(select, "kqueue"), reason="kqueue is only available on BSD/macOS")
def test_parent_death_watcher_exits_when_parent_dies(tmp_path):
    ready_path = tmp_path / "ready"
    parent_script = tmp_path / "parent.py"
    child_script = tmp_path / "child.py"

    child_script.write_text(
        textwrap.dedent(
            """
            import os
            import sys
            import time

            from brainlayer.parent_death import install_parent_death_watcher

            ready_path = sys.argv[1]
            installed = install_parent_death_watcher()
            if not installed:
                raise SystemExit("watcher did not install")
            with open(ready_path, "w", encoding="utf-8") as fh:
                fh.write(str(os.getpid()))
            while True:
                time.sleep(0.1)
            """
        )
    )
    parent_script.write_text(
        textwrap.dedent(
            """
            import os
            import subprocess
            import sys
            import time

            child = subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])
            deadline = time.monotonic() + 5
            while not os.path.exists(sys.argv[2]):
                if time.monotonic() > deadline:
                    child.kill()
                    raise SystemExit("child did not become ready")
                time.sleep(0.01)
            print(child.pid, flush=True)
            """
        )
    )

    env = os.environ.copy()
    src_path = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    parent = subprocess.Popen(
        [sys.executable, str(parent_script), str(child_script), str(ready_path)],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert parent.stdout is not None
    child_pid = int(parent.stdout.readline().strip())
    assert parent.wait(timeout=5) == 0

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        os.kill(child_pid, signal.SIGKILL)
        pytest.fail("child did not exit after parent death")
