"""Health relay signals: M1 over SSH, badge, watcher, pause/enabled. Disposable fixtures only."""

from __future__ import annotations

import importlib.util
import json
import plistlib
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "health_relay" / "health_relay.py"
PLIST = ROOT / "scripts" / "health_relay" / "com.etanhey.brainlayer-health-relay.plist"


@pytest.fixture
def relay():
    spec = importlib.util.spec_from_file_location("brainlayer_health_relay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _iso(age: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat()


def _host(data: Path, *, alerts=None, badge_age=0.0, watcher_age=0.0, active=()) -> Path:
    (data / "supervisor").mkdir(parents=True, exist_ok=True)
    (data / "job-alerts.json").write_text(json.dumps(alerts or {}))
    badge = {"state": "measured", "reason": "", "inputs": [], "badge_on": bool(active), "suppressed": []}
    badge["active"] = [{"code": c, "severity": s, "message": "secret-token-abc\n@orc rm -rf"} for c, s in active]
    _badge(data, {"schema_version": 1, "generated_at": _iso(badge_age), "alerts": badge})
    (data / "watcher-health.json").write_text(json.dumps({"updated_at": _iso(watcher_age), "poll_count": 1}))
    return data


def _badge(data: Path, document: dict) -> None:
    (data / "badge-state.json").write_text(json.dumps(document))


@pytest.fixture
def env(tmp_path, monkeypatch):
    m4, m1 = _host(tmp_path / "m4home/data"), _host(tmp_path / "m1home/data")
    inbox = tmp_path / "orc-inbox.md"
    inbox.write_text("# orc inbox\n")
    fake_ssh = tmp_path / "bin/ssh"
    fake_ssh.parent.mkdir()
    fake_ssh.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$FAKE_SSH_ARGV"\n[ -n "$FAKE_SSH_FAIL" ] && exit 255\nexec "$FAKE_PY" -I -\n'
    )
    fake_ssh.chmod(0o755)
    monkeypatch.setenv("FAKE_SSH_ARGV", str(tmp_path / "ssh-argv"))
    monkeypatch.setenv("FAKE_PY", sys.executable)
    monkeypatch.delenv("FAKE_SSH_FAIL", raising=False)
    remote = {"transport": "ssh", "dir": str(m1), "ssh_alias": "m1", "ssh_bin": str(fake_ssh)}
    config = {
        "inbox": str(inbox),
        "state_dir": str(tmp_path / "state"),
        "first_run_grace_seconds": 0,
        "unreachable_after": 2,
        "hosts": [
            {"id": "m4-max", "label": "M4 Max", "transport": "local", "dir": str(m4)},
            {"id": "m1-pro", "label": "M1 Pro", "remote_python": "/usr/bin/python3", **remote},
        ],
    }
    path = tmp_path / "relay.json"
    path.write_text(json.dumps(config))
    return {"config": path, "raw": config, "inbox": inbox, "m4": m4, "m1": m1, "tmp": tmp_path}


def _blocked(env) -> list[str]:
    return [line for line in env["inbox"].read_text().split("\n") if line.startswith("@orc BLOCKED:")]


def _issues(env) -> list[str]:
    return [line.split("] ")[1].split(" ")[0] for line in _blocked(env)]


def _status(env) -> dict:
    return json.loads((env["tmp"] / "state/status.json").read_text())


def _alerts(data: Path, alerts) -> None:
    (data / "job-alerts.json").write_text(json.dumps(alerts) if isinstance(alerts, dict) else alerts)


def test_both_hosts_reach_the_central_inbox_with_distinct_identity(relay, env):
    _alerts(env["m4"], {"backup-daily": "secret-token-abc"})
    _alerts(env["m1"], {"backup-daily": "secret-token-abc"})
    assert relay.run(env["config"]) == 0
    lines = _blocked(env)
    assert len(lines) == 2 and any("[BrainLayer M4 Max] job-alert:backup-daily" in line for line in lines)
    assert any("[BrainLayer M1 Pro] job-alert:backup-daily" in ln and "m1-pro:job-alerts.json" in ln for ln in lines)
    assert "secret-token-abc" not in env["inbox"].read_text()
    assert list((env["tmp"] / "m1home").rglob("*inbox*")) == []
    argv = (env["tmp"] / "ssh-argv").read_text().splitlines()
    for option in ("BatchMode=yes", "StrictHostKeyChecking=yes", "UpdateHostKeys=no"):
        assert option in argv
    assert argv[-4:] == ["m1", "/usr/bin/python3", "-I", "-"]
    assert _status(env)["pending"] == 0 and _status(env)["hosts"] == {"m4-max": "reached", "m1-pro": "reached"}


def test_unreadable_source_cannot_clear_while_other_sources_are_measured(relay, env):
    _alerts(env["m4"], {"backup-daily": "x"})
    relay.run(env["config"])
    _alerts(env["m4"], "{not json")
    assert relay.run(env["config"]) == 0
    assert "source-corrupt:job-alerts" in _issues(env)
    _alerts(env["m4"], {"backup-daily": "x"})
    relay.run(env["config"])
    assert _issues(env).count("job-alert:backup-daily") == 1


def test_ssh_unreachable_is_visible_and_clears_nothing(relay, env, monkeypatch):
    _alerts(env["m1"], {"backup-daily": "x"})
    relay.run(env["config"])
    monkeypatch.setenv("FAKE_SSH_FAIL", "1")
    _alerts(env["m1"], {})
    relay.run(env["config"])
    assert "host-unreachable" not in _issues(env) and _status(env)["hosts"]["m1-pro"] == "unreachable"
    relay.run(env["config"])
    assert any("[BrainLayer M1 Pro] host-unreachable" in line and "m1-pro:transport" in line for line in _blocked(env))
    monkeypatch.delenv("FAKE_SSH_FAIL")
    _alerts(env["m1"], {"backup-daily": "x"})
    relay.run(env["config"])
    assert _issues(env).count("job-alert:backup-daily") == 1


def test_dead_supervisor_is_observed_and_intentional_disable_is_honored(relay, env):
    _host(env["m4"], badge_age=7200, watcher_age=7200)
    assert relay.run(env["config"]) == 0
    assert set(_issues(env)) == {"health-stale", "watcher-stale"}
    jobs = {"com.brainlayer.health-check": {"enabled": False}, "com.brainlayer.watch": {"enabled": True}}
    (env["m1"] / "supervisor/jobs.json").write_text(json.dumps(jobs))
    (env["m1"] / "pause.sentinel").write_text(json.dumps({"labels": ["com.brainlayer.watch"]}))
    _host(env["m1"], badge_age=7200, watcher_age=7200)
    relay.run(env["config"])
    assert not any("M1 Pro" in line for line in _blocked(env))


def test_only_critical_badge_issues_escalate_and_first_run_grace_holds(relay, env):
    active = [("brain_search_canary_failed", "critical"), ("missing_embeddings", "warning")]
    _host(env["m4"], active=[*active, ("job_alert_backup-daily", "critical")])
    (env["m1"] / "badge-state.json").unlink()
    env["raw"]["first_run_grace_seconds"] = 3600
    env["config"].write_text(json.dumps(env["raw"]))
    relay.run(env["config"])
    assert _issues(env) == ["badge:brain-search-canary-failed"]


def test_stale_badge_cannot_clear_a_badge_incident(relay, env):
    _host(env["m4"], active=[("brain_search_canary_failed", "critical")])
    relay.run(env["config"])
    _host(env["m4"], badge_age=7200)
    relay.run(env["config"])
    _host(env["m4"], active=[("brain_search_canary_failed", "critical")])
    relay.run(env["config"])
    issues = _issues(env)
    assert issues.count("badge:brain-search-canary-failed") == 1 and issues.count("health-stale") == 1


def test_pending_first_run_badge_cannot_clear_a_badge_incident(relay, env):
    _host(env["m4"], active=[("brain_search_canary_failed", "critical")])
    relay.run(env["config"])
    pending = {"state": "pending_first_run", "reason": "", "inputs": [], "expected_first_run_by": _iso(-600)}
    _badge(env["m4"], {"schema_version": 1, "generated_at": _iso(), "alerts": pending})
    relay.run(env["config"])
    _host(env["m4"], active=[("brain_search_canary_failed", "critical")])
    relay.run(env["config"])
    assert _issues(env) == ["badge:brain-search-canary-failed"]


def test_disabled_health_check_is_honored_during_its_first_run_window(relay, env):
    pending = {"state": "pending_first_run", "reason": "", "inputs": [], "expected_first_run_by": _iso(600)}
    for data in (env["m4"], env["m1"]):
        _badge(data, {"schema_version": 1, "generated_at": _iso(1800), "alerts": pending})
    (env["m1"] / "supervisor/jobs.json").write_text(json.dumps({"com.brainlayer.health-check": {"enabled": False}}))
    assert relay.run(env["config"]) == 0
    assert [line.split("] ")[0] for line in _blocked(env)] == ["@orc BLOCKED: [BrainLayer M4 Max"]
    assert _issues(env) == ["health-stale"]


@pytest.mark.parametrize(
    "field,value",
    [("ssh_alias", "-oProxyCommand=x"), ("remote_python", "/usr/bin/python3; touch /tmp/x"), ("transport", "rsh")],
)
def test_unsafe_remote_config_is_refused(relay, env, field, value):
    env["raw"]["hosts"][1][field] = value
    env["config"].write_text(json.dumps(env["raw"]))
    assert relay.run(env["config"]) == 2
    assert not (env["tmp"] / "ssh-argv").exists()


def test_relay_is_scheduled_outside_the_supervisor():
    from brainlayer import consolidation_plan, supervisor_adapters

    plist = plistlib.loads(PLIST.read_bytes())
    label = plist["Label"]
    assert not label.startswith("com.brainlayer.") and label not in supervisor_adapters.ALIASES
    assert label not in consolidation_plan.REVIVERS | consolidation_plan.UNMANAGED
    assert plist["ProgramArguments"] == ["__PYTHON__", "-I", "__SCRIPT__", "--config", "__CONFIG__"]
    assert plist["StartInterval"] == 300 and "KeepAlive" not in plist


def _state(env):
    return json.loads((env["tmp"] / "state/state.json").read_text())


def _shell_ssh(env):
    """Model OpenSSH: join remote argv and let the remote login shell parse it."""
    shim = Path(env["raw"]["hosts"][1]["ssh_bin"])
    shim.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$FAKE_SSH_ARGV"\n'
        'cd "$(dirname "$FAKE_SSH_ARGV")" || exit 1\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "--" ]; then shift; shift; break; fi\n'
        '  shift\ndone\nexec /bin/sh -c "$*"\n'
    )


@pytest.mark.parametrize(
    "active",
    [
        {},
        "",
        [None],
        [{"code": 7, "severity": "critical"}],
        [{"code": "canary_failed", "severity": None}],
        [{"code": "canary_failed", "severity": "unknown"}],
        [{"code": "canary_failed"}],
        [{"code": "new_critical", "severity": "critical"}, {}],
    ],
)
def test_malformed_badge_preserves_open_episode(relay, env, active):
    _host(env["m4"], active=[("canary_failed", "critical")])
    assert relay.run(env["config"]) == 0
    path = env["m4"] / "badge-state.json"
    badge = json.loads(path.read_text())
    badge["alerts"]["active"] = active
    _badge(env["m4"], badge)
    assert relay.run(env["config"]) == 0
    assert "source-corrupt:badge" in _issues(env)
    episodes = _state(env)["hosts"]["m4-max"]["episodes"]
    assert episodes["badge:canary-failed"]["open"] is True
    assert "badge:new-critical" not in episodes  # validate the whole list before evaluating it
    _host(env["m4"], active=[("canary_failed", "critical")])
    assert relay.run(env["config"]) == 0
    assert _issues(env).count("badge:canary-failed") == 1
    assert _status(env)["pending"] == 0


@pytest.mark.parametrize(
    "pause",
    [
        {"labels": {"com.brainlayer.health-check": False, "com.brainlayer.watch": False}},
        {"labels": ["com.brainlayer.health-check", "com.brainlayer.watch"], "expires_at": "garbled"},
        {"labels": ["com.brainlayer.health-check", "com.brainlayer.watch"], "expires_at": 7},
    ],
)
def test_malformed_pause_cannot_suppress_staleness(relay, env, pause):
    _host(env["m4"], badge_age=7200, watcher_age=7200)
    (env["m4"] / "pause.sentinel").write_text(json.dumps(pause))
    assert relay.run(env["config"]) == 0
    assert set(_issues(env)) == {"source-corrupt:pause", "health-stale", "watcher-stale"}
    assert relay.run(env["config"]) == 0
    assert len(_blocked(env)) == 3 and _status(env)["pending"] == 0


@pytest.mark.parametrize("field", ["remote_python", "ssh_alias"])
@pytest.mark.parametrize("control", ["\n", "\r", "\t", "\x00", "\x1f", "\x7f", "\x85"])
def test_remote_config_rejects_controls_before_shell(relay, env, monkeypatch, field, control):
    _shell_ssh(env)
    marker = env["tmp"] / "sitecustomize-ran"
    (env["tmp"] / "sitecustomize.py").write_text(f"open({str(marker)!r}, 'w').write('executed')\n")
    monkeypatch.setenv("PYTHONPATH", str(env["tmp"]))
    env["raw"]["hosts"][1][field] += control
    env["config"].write_text(json.dumps(env["raw"]))
    assert relay.run(env["config"]) == 2
    assert not marker.exists() and not (env["tmp"] / "ssh-argv").exists()
    assert not (env["tmp"] / "state").exists()


def test_remote_shell_keeps_isolation_and_data_inert(relay, env, monkeypatch):
    _shell_ssh(env)
    marker = env["tmp"] / "sitecustomize-ran"
    (env["tmp"] / "sitecustomize.py").write_text(f"open({str(marker)!r}, 'w').write('executed')\n")
    monkeypatch.setenv("PYTHONPATH", str(env["tmp"]))
    hostile = _host(env["tmp"] / "quote' and $(touch INJECTED); data", alerts={"backup-daily": "x"})
    env["raw"]["hosts"][1]["dir"] = str(hostile)
    env["config"].write_text(json.dumps(env["raw"]))
    for _ in range(2):
        assert relay.run(env["config"]) == 0
    assert _issues(env) == ["job-alert:backup-daily"] and _status(env)["pending"] == 0
    assert not marker.exists() and not (env["tmp"] / "INJECTED").exists()
    assert (env["tmp"] / "ssh-argv").read_text().splitlines()[-3:] == ["/usr/bin/python3", "-I", "-"]


def test_remote_shell_quotes_each_argument(relay, env):
    _shell_ssh(env)
    cfg = relay.load_config(env["config"])
    # Exercise quoting independently of the stricter public config gate.
    executable = env["tmp"] / "python'; touch INJECTED; #"
    executable.write_text('#!/bin/sh\nexec /usr/bin/python3 "$@"\n')
    executable.chmod(0o755)
    cfg["hosts"][1]["remote_python"] = str(executable)
    snap = relay._snapshot(cfg["hosts"][1], cfg)
    assert snap is not None and snap["sources"]["badge"]["s"] == "ok"
    assert not (env["tmp"] / "INJECTED").exists()


@pytest.mark.parametrize("code", ["canary_failed\n", 7])
def test_invalid_badge_code_is_one_deliverable_event(relay, env, code):
    _host(env["m4"], active=[(code, "critical")])
    for _ in range(3):
        assert relay.run(env["config"]) == 0
        assert _status(env)["pending"] == 0
    assert _issues(env) == ["source-corrupt:badge"]
    assert _state(env)["outbox"] == []


@pytest.mark.parametrize("reply", [b"[" * 1100 + b"0" + b"]" * 1100, b"{garbled", b"\xff", b"[]"])
@pytest.mark.parametrize("python", [None, "/usr/bin/python3"], ids=["in-process", "python39-isolated"])
def test_malformed_remote_reply_commits_local_events_and_status(relay, env, reply, python):
    _alerts(env["m4"], {"backup-daily": "x"})
    response = env["tmp"] / "reply.bin"
    response.write_bytes(reply)
    shim = Path(env["raw"]["hosts"][1]["ssh_bin"])
    shim.write_text(f"#!/bin/sh\ncat '{response}'\n")

    def run():
        return (
            relay.run(env["config"])
            if python is None
            else subprocess.run(
                [python, "-I", str(SCRIPT), "--config", str(env["config"])], capture_output=True, timeout=5
            ).returncode
        )

    for _ in range(2):
        assert run() == 0
        assert _status(env)["pending"] == 0
        assert _status(env)["hosts"] == {"m4-max": "reached", "m1-pro": "unreachable"}
    assert set(_issues(env)) == {"job-alert:backup-daily", "host-unreachable"}
    assert _state(env)["hosts"]["m1-pro"]["unreach"] == 2
    assert run() == 0  # B's transport episode survives A's state validator
    assert len(_blocked(env)) == 2


@pytest.mark.parametrize(
    "source", ["badge-state.json", "watcher-health.json", "supervisor/jobs.json", "pause.sentinel"]
)
@pytest.mark.parametrize("text", ["[" * 1100 + "0" + "]" * 1100, "{garbled"])
@pytest.mark.parametrize("python", [None, "/usr/bin/python3"], ids=["in-process", "python39-isolated"])
def test_new_producer_decode_failures_are_corrupt_and_do_not_abort(relay, env, source, text, python):
    _alerts(env["m4"], {"backup-daily": "x"})
    (env["m4"] / source).write_text(text)
    code = (
        relay.run(env["config"])
        if python is None
        else subprocess.run(
            [python, "-I", str(SCRIPT), "--config", str(env["config"])], capture_output=True, timeout=5
        ).returncode
    )
    assert code == 0
    name = next(name for name, path in relay.SOURCES.items() if path == source)
    assert {"job-alert:backup-daily", f"source-corrupt:{name}"} <= set(_issues(env))
    assert _status(env)["pending"] == 0


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_ssh_capture_has_byte_limits_and_reaps_overflow(relay, env, monkeypatch, stream):
    # A small limit makes a real pipe overflow cheap; the shim has no descendants.
    monkeypatch.setattr(relay, "MAX_OUTPUT_BYTES", 1024)
    pid_path = env["tmp"] / "ssh.pid"
    monkeypatch.setenv("FAKE_SSH_PID", str(pid_path))
    shim = Path(env["raw"]["hosts"][1]["ssh_bin"])
    shim.write_text(
        f"#!{sys.executable}\nimport os, sys, time\n"
        "open(os.environ['FAKE_SSH_PID'], 'w').write(str(os.getpid()))\n"
        f"os.write({1 if stream == 'stdout' else 2}, b'x' * 4096)\n"
        "time.sleep(5)\n"
    )
    env["raw"]["ssh_timeout_seconds"] = 3
    env["config"].write_text(json.dumps(env["raw"]))
    _alerts(env["m4"], {"backup-daily": "x"})
    started = time.monotonic()
    assert relay.run(env["config"]) == 0
    assert time.monotonic() - started < 1.5  # overflow, not the 3-second timeout, ends capture
    with pytest.raises(ProcessLookupError):
        relay.os.kill(int(pid_path.read_text()), 0)
    assert _status(env)["hosts"]["m1-pro"] == "unreachable"
    assert _issues(env) == ["job-alert:backup-daily"]


def test_signal_issues_reload_under_v3_without_duplicate_delivery(relay, env, monkeypatch):
    _host(env["m4"], badge_age=7200, watcher_age=7200)
    monkeypatch.setenv("FAKE_SSH_FAIL", "1")
    for _ in range(3):
        assert relay.run(env["config"]) == 0
    state = _state(env)
    assert state["version"] == 3
    assert set(_issues(env)) == {"health-stale", "watcher-stale", "host-unreachable"}
    assert len(_blocked(env)) == 3 and _status(env)["pending"] == 0
    loaded = relay._load_state(env["tmp"] / "state/state.json", str(env["inbox"]), False)
    assert loaded == state


def test_remote_source_alias_checks_never_resolve_or_stat_remote_paths(relay, env, monkeypatch):
    remote = "/remote-only/unsearchable/brainlayer"
    env["raw"]["hosts"][1]["dir"] = remote
    env["config"].write_text(json.dumps(env["raw"]))
    real_resolve, real_samefile = Path.resolve, relay.os.path.samefile

    def resolve(path, *args, **kwargs):
        assert not str(path).startswith(remote), "remote source resolved locally"
        return real_resolve(path, *args, **kwargs)

    def samefile(left, right):
        assert not str(right).startswith(remote), "remote source statted locally"
        return real_samefile(left, right)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(relay.os.path, "samefile", samefile)
    assert relay.load_config(env["config"])["hosts"][1]["paths"]["badge"].startswith(remote)
    # Identical spelling on two machines does not make an alias.
    env["raw"]["hosts"][1]["sources"] = {"job-alerts": str(env["inbox"])}
    env["config"].write_text(json.dumps(env["raw"]))
    assert relay.load_config(env["config"])["hosts"][1]["paths"]["job-alerts"] == str(env["inbox"])


def test_unreachable_event_uses_persisted_epoch_and_new_epoch_after_state_loss(relay, env, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_FAIL", "1")
    for _ in range(2):
        assert relay.run(env["config"]) == 0
    first = _state(env)
    expected = f"bl-m1-pro-{first['epoch']}-1-host-unreachable"
    assert _blocked(env)[0].endswith(f"event={expected}")
    assert relay._marker(expected) in env["inbox"].read_text().splitlines()
    (env["tmp"] / "state/state.json").unlink()
    assert relay.run(env["config"]) == 1 and _status(env)["error"] == "state-missing"
    assert relay.run(env["config"], accept_new_epoch=True) == 0
    assert relay.run(env["config"]) == 0
    second = _state(env)
    assert second["epoch"] != first["epoch"]
    assert _blocked(env)[1].endswith(f"event=bl-m1-pro-{second['epoch']}-1-host-unreachable")


@pytest.mark.parametrize("read_state", ["ok", "unreadable", "oversize", "corrupt"])
def test_remote_seen_records_every_non_absent_observation(relay, env, monkeypatch, read_state):
    cfg = relay.load_config(env["config"])
    snap = {"now": time.time(), "sources": {name: {"s": read_state, "text": "{}"} for name in relay.SOURCES}}
    monkeypatch.setattr(relay, "_snapshot", lambda host, config: snap)
    host_state = {}
    relay._observe(cfg["hosts"][1], cfg, host_state, time.time(), "ab" * 16)
    assert set(host_state["seen"]) == set(relay.SOURCES)
    # An error followed by absence must not erase its episode or make job-alerts optional again.
    snap["sources"] = {name: {"s": "absent"} for name in relay.SOURCES}
    relay._observe(cfg["hosts"][1], cfg, host_state, time.time(), "ab" * 16)
    assert host_state["episodes"]["source-missing:job-alerts"]["open"] is True
    if read_state != "ok":
        assert host_state["episodes"][f"source-{read_state}:job-alerts"]["open"] is True


@pytest.mark.parametrize("host_id,issue", [("m4-max", "health-stale"), ("m1-pro", "host-unreachable")])
def test_stale_state_scan_uses_each_hosts_epoch_and_sequence(relay, env, monkeypatch, host_id, issue):
    assert relay.run(env["config"]) == 0
    state_path = env["tmp"] / "state/state.json"
    older = state_path.read_bytes()
    if host_id == "m4-max":
        _host(env["m4"], badge_age=7200)
    else:
        monkeypatch.setenv("FAKE_SSH_FAIL", "1")
    for _ in range(2):
        assert relay.run(env["config"]) == 0
    state = _state(env)
    seq = state["hosts"][host_id]["episodes"][issue]["seq"]
    assert f"bl-{host_id}-{state['epoch']}-{seq}-{issue}" in env["inbox"].read_text()
    inbox_before = env["inbox"].read_bytes()
    state_path.write_bytes(older)
    assert relay.run(env["config"], accept_new_epoch=True) == 1
    assert _status(env)["error"] == "state-stale"
    assert state_path.read_bytes() == older and env["inbox"].read_bytes() == inbox_before


def test_ssh_exited_with_stdout_complete_does_not_wait_for_stderr_eof(relay, env, monkeypatch):
    import os
    import threading

    # A thread holds the write end independently of the exited shim, like ControlPersist.
    read_fd, write_fd = os.pipe()
    real_popen = subprocess.Popen
    stopped = threading.Event()
    holder = threading.Thread(target=lambda: (stopped.wait(3), os.close(write_fd)))
    holder.start()

    def popen(*args, **kwargs):
        kwargs["stderr"] = write_fd
        proc = real_popen(*args, **kwargs)
        proc.stderr = os.fdopen(read_fd, "rb")
        return proc

    monkeypatch.setattr(relay.subprocess, "Popen", popen)
    try:
        started = time.monotonic()
        snap = relay._snapshot(relay.load_config(env["config"])["hosts"][1], {"ssh_timeout_seconds": 1})
        assert snap is not None and snap["sources"]["badge"]["s"] == "ok"
        assert time.monotonic() - started < 0.75
    finally:
        stopped.set()
        holder.join()


def test_ssh_timeout_kills_and_reaps_the_shim(relay, env, monkeypatch):
    pid_path = env["tmp"] / "timeout-shim.pid"
    shim = Path(env["raw"]["hosts"][1]["ssh_bin"])
    shim.write_text(
        f"#!{sys.executable}\nimport os, time\nopen({str(pid_path)!r}, 'w').write(str(os.getpid()))\ntime.sleep(30)\n"
    )
    host = relay.load_config(env["config"])["hosts"][1]
    started = time.monotonic()
    assert relay._snapshot(host, {"ssh_timeout_seconds": 0.5}) is None
    assert time.monotonic() - started < 2
    with pytest.raises(ProcessLookupError):
        relay.os.kill(int(pid_path.read_text()), 0)


def test_first_run_grace_only_holds_badge_and_watcher_absence(relay, env):
    env["raw"]["first_run_grace_seconds"] = 3600
    env["config"].write_text(json.dumps(env["raw"]))
    assert relay.run(env["config"]) == 0
    started = _state(env)["hosts"]["m4-max"]["first_seen"]
    for name in ("job-alerts.json", "badge-state.json", "watcher-health.json"):
        (env["m4"] / name).unlink()
    assert relay.run(env["config"]) == 0
    assert _issues(env) == ["source-missing:job-alerts"]
    assert relay.run(env["config"], clock=lambda: started + 3601) == 0
    assert set(_issues(env)) == {"source-missing:job-alerts", "source-missing:badge", "source-missing:watcher"}
