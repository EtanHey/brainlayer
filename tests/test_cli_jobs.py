import json

from typer.testing import CliRunner

from brainlayer.cli import app


def test_jobs_component_cli_returns_json_and_exit_on_error(monkeypatch):
    monkeypatch.setattr("brainlayer.component_jobs.component_action", lambda *a, **k: {"source": "supervisor"})
    result = CliRunner().invoke(app, ["jobs", "component", "watch", "status"])
    assert result.exit_code == 0 and json.loads(result.stdout)["source"] == "supervisor"

    def fail(*a, **k):
        raise OSError("control unavailable")

    monkeypatch.setattr("brainlayer.component_jobs.component_action", fail)
    result = CliRunner().invoke(app, ["jobs", "component", "watch", "restart"])
    assert result.exit_code == 1 and "control unavailable" in result.stdout


def test_daemon_cli_requires_manifest_and_delegates(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr("brainlayer.supervisor_jobs.run_daemon", lambda *a: called.append(a))
    result = CliRunner().invoke(app, ["daemon", "--manifest", str(tmp_path / "jobs.json")])
    assert result.exit_code == 0 and called[0][0] == tmp_path / "jobs.json"
    assert CliRunner().invoke(app, ["daemon"]).exit_code != 0


def test_maintenance_cli_forwards_paths_date_and_signal_installation(monkeypatch, tmp_path):
    import sys
    from datetime import date
    from types import SimpleNamespace

    captured = {}

    def steps(python, cli, lib_dir, scheduled_date):
        captured.update(python=python, cli=cli, lib_dir=lib_dir, date=scheduled_date)
        return "plan"

    async def run(plan, directory, logs, **kwargs):
        captured.update(plan=plan, directory=directory, logs=logs, **kwargs)
        return 1

    monkeypatch.setitem(
        sys.modules, "brainlayer.maintenance_calendar", SimpleNamespace(calendar_steps=steps, run_calendar=run)
    )
    result = CliRunner().invoke(
        app,
        [
            "maintenance-calendar",
            "--lib-dir",
            str(tmp_path),
            "--date",
            "2026-10-11",
            "--state-dir",
            str(tmp_path / "state"),
            "--logs-dir",
            str(tmp_path / "logs"),
        ],
    )
    assert result.exit_code == 1
    assert captured["date"] == date(2026, 10, 11) and captured["lib_dir"] == tmp_path
    assert captured["plan"] == "plan" and captured["install_signals"] is True
    assert captured["directory"] == tmp_path / "state" and captured["logs"] == tmp_path / "logs"
    assert captured["python"] == sys.executable
    captured.clear()
    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_SOCKET", str(tmp_path / "state" / "control.sock"))
    result = CliRunner().invoke(app, ["maintenance-calendar", "--state-dir", str(tmp_path / "state")])
    assert result.exit_code == 2 and not captured


def test_real_cli_reads_private_daemon_status(monkeypatch):
    import os
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="bl-cli-", dir="/tmp") as directory:
        root = Path(directory)
        monkeypatch.setenv("HOME", directory)
        path, endpoint = root / "jobs.json", root / "control.sock"
        path.write_text(
            json.dumps(
                {
                    "com.brainlayer.watch": {
                        "enabled": True,
                        "plist": {
                            "Label": "com.brainlayer.watch",
                            "KeepAlive": True,
                            "ProgramArguments": [sys.executable, "-c", "import time;time.sleep(60)"],
                        },
                    }
                }
            )
        )
        path.chmod(0o600)
        daemon = subprocess.Popen(
            [sys.executable, "-m", "brainlayer", "daemon", "--manifest", str(path), "--socket", str(endpoint)]
        )
        try:
            deadline, result = time.monotonic() + 10, None
            while time.monotonic() < deadline and daemon.poll() is None:
                if endpoint.exists():
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "brainlayer",
                            "jobs",
                            "component",
                            "watch",
                            "status",
                            "--socket",
                            str(endpoint),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0 and json.loads(result.stdout)["runtime"]["pid"]:
                        break
                time.sleep(0.02)
            assert result is not None and result.returncode == 0, result
            state = json.loads(result.stdout)
            assert state["source"] == "supervisor" and state["runtime"]["configured"] is True
            pid = state["runtime"]["pid"]
        finally:
            daemon.terminate()
            daemon.wait(timeout=20)
        assert daemon.returncode == 0 and not endpoint.exists()
        import pytest

        with pytest.raises(ProcessLookupError):
            os.killpg(pid, 0)
