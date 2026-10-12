"""CLI parity: runtime evidence never substitutes configured state."""

import json
import plistlib
import subprocess

import pytest

from brainlayer.component_jobs import ALIASES, component_action


def setup(tmp_path, name="watch"):
    label = next(k for k, v in ALIASES.items() if v == name)
    agents = tmp_path / "agents"
    agents.mkdir()
    plist = dict(
        Label=label,
        KeepAlive=True,
        ProgramArguments=["/private/env-run", "/keg/brainlayer", "watch"],
        EnvironmentVariables={"BRAINLAYER_ENV_FILE": "/private/brainlayer.env", "API_KEY": "private-value"},
        StandardOutPath="/private/watch.out.log",
        StandardErrorPath="/private/watch.err.log",
    )
    (agents / f"{label}.plist").write_bytes(plistlib.dumps(plist))
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps({label: {"enabled": False, "plist": plist}}))
    manifest.chmod(0o600)
    return agents, manifest


def test_legacy_presence_retains_status_env_and_logs(tmp_path):
    agents, manifest = setup(tmp_path)
    calls = []

    def run(args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "state = running\npid = 42\n", "")

    result = component_action(
        "watch", "status", agents=agents, manifest=manifest, path=tmp_path / "missing", runner=run
    )
    assert result["source"] == "launchd" and result["runtime"]["pid"] == 42
    assert calls[0][:2] == ["launchctl", "print"]
    assert component_action("watch", "env", agents=agents, manifest=manifest)["env_file"] == "/private/brainlayer.env"
    assert component_action("watch", "logs", agents=agents, manifest=manifest)["stderr"] == "/private/watch.err.log"
    assert "private-value" not in json.dumps(result)


def test_archived_component_uses_supervisor_runtime_and_private_manifest(tmp_path, monkeypatch):
    agents, manifest = setup(tmp_path)
    (agents / "com.brainlayer.watch.plist").unlink()
    state = dict(configured=False, paused=False, pid=77, starts=3, exit_timeout=30)

    class Client:
        def __init__(self, *args):
            pass

        def request(self):
            return {"watch": state}

    monkeypatch.setattr("brainlayer.component_jobs.SupervisorCommands", Client)
    result = component_action("watch", "status", agents=agents, manifest=manifest, path=tmp_path / "socket")
    assert result["source"] == "supervisor" and result["runtime"] == state
    assert result["runtime"]["configured"] is False and result["runtime"]["pid"] == 77
    assert component_action("watch", "env", agents=agents, manifest=manifest)["env_file"] == "/private/brainlayer.env"
    assert component_action("watch", "logs", agents=agents, manifest=manifest)["stdout"] == "/private/watch.out.log"


def test_missing_managed_socket_is_an_error_not_legacy_revival(tmp_path):
    agents, manifest = setup(tmp_path)
    (agents / "com.brainlayer.watch.plist").unlink()
    with pytest.raises(RuntimeError, match="supervisor control failed"):
        component_action("watch", "restart", agents=agents, manifest=manifest, path=tmp_path / "missing")


@pytest.mark.parametrize("name", ["enrich", "../watch", "unknown", "brainbar"])
def test_unknown_or_retired_components_never_run_commands(tmp_path, name):
    with pytest.raises(ValueError):
        component_action(name, "status", agents=tmp_path)


@pytest.mark.parametrize("disabled", [True, None])
def test_legacy_restart_refuses_disabled_or_unmeasurable(tmp_path, monkeypatch, disabled):
    agents, manifest = setup(tmp_path)
    calls = []
    monkeypatch.setattr("brainlayer.component_jobs.is_launchd_label_disabled", lambda *a, **k: disabled)
    with pytest.raises(RuntimeError, match="disabled state"):
        component_action("watch", "restart", agents=agents, manifest=manifest, runner=lambda args: calls.append(args))
    assert not calls


@pytest.mark.parametrize("name", ["watch", "hotlane", "fleet-watchdog"])
def test_legacy_configured_off_restart_never_kickstarts(tmp_path, monkeypatch, name):
    agents, manifest = setup(tmp_path, name)
    monkeypatch.setattr("brainlayer.component_jobs.is_launchd_label_disabled", lambda *a, **k: False)
    monkeypatch.setattr("brainlayer.component_jobs.pause_sentinel_state", lambda *a, **k: (None, False, ""))
    expected = "BRAINLAYER_LAUNCHD_" + name.replace("-", "_").upper() + "_ENABLED"
    monkeypatch.setattr(
        "brainlayer.component_jobs.configured_brainlayer_env_value", lambda key, *a: "0" if key == expected else None
    )
    calls = []
    with pytest.raises(RuntimeError, match="configured OFF"):
        component_action(name, "restart", agents=agents, manifest=manifest, runner=lambda args: calls.append(args))
    assert not calls
