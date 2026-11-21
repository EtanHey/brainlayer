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


def test_persistent_remote_fault_survives_alternating_unreachable_reads(relay, env, monkeypatch):
    _fault(env, "watcher-stale")
    snapshot = relay._snapshot
    readings = iter([True, False, True, False, True, False, True])
    monkeypatch.setattr(relay, "_snapshot", lambda *args: snapshot(*args) if next(readings) else None)
    counts = []
    for index in range(7):
        assert relay.run(env["config"], clock=lambda: 1000 + index * 100) == 0
        counts.append(len(_blocked(env)))
    assert counts == [0, 0, 1, 1, 1, 1, 1]  # Confirmed on the second measured reading.
    assert "watcher-stale" in _blocked(env)[0]


def test_alternating_measured_fault_confirms_without_duplicate_episodes(relay, env):
    counts = []
    for index in range(12):
        _fault(env, "job-alert:mcp-transport", active=index % 2 == 0)
        assert relay.run(env["config"], clock=lambda: 1000 + index * 100) == 0
        counts.append(len(_blocked(env)))
    assert counts == [0, 0] + [1] * 10
    _fault(env, "job-alert:mcp-transport", active=False)
    for tick in (2200, 2300, 2400):
        assert relay.run(env["config"], clock=lambda: tick) == 0
    state = json.loads((env["tmp"] / "state/state.json").read_text())
    assert state["hosts"]["m4-max"]["episodes"]["job-alert:mcp-transport"]["open"] is False
    _fault(env, "job-alert:mcp-transport")
    for tick in (2500, 2600):
        assert relay.run(env["config"], clock=lambda: tick) == 0
    assert len(_blocked(env)) == 2  # A genuinely new episode after measured recovery.


def test_real_confirmation_defers_overdue_escalation_after_wake(relay, env):
    escalation = env["tmp"] / "escalation.md"
    escalation.write_text("# escalation\n")
    env["raw"]["escalation"] = {"inbox": str(escalation), "after_seconds": 1800}
    env["config"].write_text(json.dumps(env["raw"]))
    _fault(env, "job-alert:mcp-transport")
    for tick in (1000, 1300, 10000):
        assert relay.run(env["config"], clock=lambda: tick) == 0
        assert "BLOCKED" not in escalation.read_text()
    assert _status(env)["warnings"] == {"m4-max": ["job-alert:mcp-transport"]}
    assert relay.run(env["config"], clock=lambda: 10300) == 0
    assert escalation.read_text().count("@orc BLOCKED:") == 1
