"""Real relay runs against disposable files: warning, confirmation, and wake recovery."""

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from tests import test_health_relay as core
from tests.test_health_relay import SCRIPT, _alerts, _blocked, _status

base_env = core.env


@pytest.fixture
def env(base_env):
    base_env["raw"]["first_run_grace_seconds"] = 100000
    base_env["config"].write_text(json.dumps(base_env["raw"]))
    return base_env


@pytest.fixture
def relay():
    spec = importlib.util.spec_from_file_location("relay_wake", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fault(env, issue, active=True):
    if issue == "watcher-stale":
        stamp = "2000-01-01T00:00:00+00:00" if active else datetime.now(timezone.utc).isoformat()
        (env["data"] / "watcher-health.json").write_text(json.dumps({"updated_at": stamp}))
    else:
        _alerts(env, {"mcp-transport": "dropped request"} if active else {})


@pytest.mark.parametrize("issue", ["job-alert:mcp-transport", "watcher-stale"])
def test_single_transient_warns_then_recovers(relay, env, issue):
    _fault(env, issue)
    assert relay.run(env["config"], clock=lambda: 1000) == 0
    assert _blocked(env) == []
    assert _status(env)["warnings"] == {"m4-max": [issue]}
    _fault(env, issue, active=False)
    assert relay.run(env["config"], clock=lambda: 1300) == 0
    assert _blocked(env) == []
    assert _status(env)["warnings"] == {}


@pytest.mark.parametrize("issue", ["job-alert:mcp-transport", "watcher-stale"])
def test_persistent_fault_blocks_on_next_reading(relay, env, issue):
    _fault(env, issue)
    assert relay.run(env["config"], clock=lambda: 1000) == 0
    assert _blocked(env) == []
    assert relay.run(env["config"], clock=lambda: 1300) == 0
    assert len(_blocked(env)) == 1
    assert issue in _blocked(env)[0]


@pytest.mark.parametrize("issue", ["job-alert:mcp-transport", "watcher-stale"])
def test_sleep_gap_requires_new_confirmation(relay, env, issue):
    _fault(env, issue)
    assert relay.run(env["config"], clock=lambda: 1000) == 0
    assert relay.run(env["config"], clock=lambda: 10000) == 0
    assert _blocked(env) == []
    assert relay.run(env["config"], clock=lambda: 10300) == 0
    assert len(_blocked(env)) == 1


def test_unreachable_count_resets_after_sleep(relay, env, monkeypatch):
    monkeypatch.setattr(relay, "_snapshot", lambda *args: None)
    for tick in (1000, 1300, 10000):
        assert relay.run(env["config"], clock=lambda: tick) == 0
    assert _blocked(env) == []
    for tick in (10300, 10600):
        assert relay.run(env["config"], clock=lambda: tick) == 0
    assert len(_blocked(env)) == 1
    assert "host-unreachable" in _blocked(env)[0]


def test_real_cli_reconfirms_after_a_sleep_gap(env):
    _fault(env, "job-alert:mcp-transport")
    command = [sys.executable, "-I", str(SCRIPT), "--config", str(env["config"])]
    assert subprocess.run(command, capture_output=True).returncode == 0
    state_path = env["tmp"] / "state/state.json"
    state = json.loads(state_path.read_text())
    state["hosts"]["m4-max"]["last_observed"] -= 10000
    state_path.write_text(json.dumps(state))
    assert subprocess.run(command, capture_output=True).returncode == 0
    assert _blocked(env) == []
    assert _status(env)["warnings"] == {"m4-max": ["job-alert:mcp-transport"]}
    assert subprocess.run(command, capture_output=True).returncode == 0
    assert len(_blocked(env)) == 1
