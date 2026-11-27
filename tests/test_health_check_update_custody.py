"""Updater custody fences health-check revival without consuming heal retries."""

import fcntl
import json
import subprocess

import pytest

import brainlayer.health_check as health
from tests.test_stability_health_check import _clock_at, _make_db, _ps_output_stub


@pytest.mark.parametrize("lock_mode", [fcntl.LOCK_EX, fcntl.LOCK_SH])
def test_failed_probe_heal_respects_update_custody(tmp_path, monkeypatch, caplog, lock_mode):
    root = tmp_path / "update"
    root.mkdir()
    monkeypatch.setenv("BRAINBAR_UPDATE_ROOT", str(root))
    monkeypatch.setattr(health, "_brainbar_answers_ping", lambda *_: False)
    db = tmp_path / "brain.db"
    _make_db(db, total=3, vector_rows=3)
    config = health.HealthCheckConfig(
        db_path=db, state_path=tmp_path / "state", heal=True, heal_min_consecutive_failures=1
    )
    commands = []

    def runner(args):
        commands.append(args)
        output = "pid = 123" if args[:2] == ["launchctl", "print"] else "S"
        if args[0] == "sysctl":
            output = "1"
        if args[:2] == ["launchctl", "kickstart"] and args[-1].endswith(config.brainbar_daemon_label):
            # Custody must cover the command, not just a preceding lock check.
            with (root / "update-custody.lock").open("a") as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return subprocess.CompletedProcess(args, 0, output, "")

    def failed_probe(*_):
        raise ConnectionError("updater stopped BrainBar")

    def run(minute):
        return health.run_health_check(
            config,
            ps_output_fn=_ps_output_stub(""),
            socket_request_fn=failed_probe,
            command_runner=runner,
            now_fn=_clock_at(minute),
        )

    def revivals():
        return [
            args
            for args in commands
            if args[:2] in (["launchctl", "kickstart"], ["launchctl", "bootstrap"])
            and "com.brainlayer.brainbar" in args[-1]
        ]

    key = health._heal_key(config.brainbar_daemon_label, "brain_search_canary_failed")
    with (root / "update-custody.lock").open("a") as updater:
        fcntl.flock(updater, lock_mode | fcntl.LOCK_NB)
        with caplog.at_level("INFO", logger=health.__name__):
            result = run(25)
        if lock_mode == fcntl.LOCK_EX:
            assert not revivals()
            assert any(
                args[:2] == ["launchctl", "kickstart"] and args[-1].endswith(config.hotlane_label) for args in commands
            )
            assert key not in json.loads(config.state_path.read_text())["heal_failures"]
            assert not any("heal_escalation" in action for action in result.actions)
            assert caplog.text.count("BrainBar heal deferred: update custody held") == 1
        else:
            assert len(revivals()) == 1
    commands.clear()
    run(30)
    assert len(revivals()) == 1


@pytest.mark.parametrize("label", ["com.brainlayer.brainbar", "com.brainlayer.brainbar-daemon"])
def test_bootstrap_heal_defers_without_advancing_breaker(tmp_path, monkeypatch, label):
    monkeypatch.setenv("BRAINBAR_UPDATE_ROOT", str(tmp_path))
    config = health.HealthCheckConfig(heal=True, heal_min_consecutive_failures=1)
    code = "watch_unloaded"
    key = health._heal_key(label, code)
    commands = []
    loaded = False

    def runner(args):
        nonlocal loaded
        commands.append(args)
        if args[:2] == ["launchctl", "print"] and not loaded:
            return subprocess.CompletedProcess(args, 113, "", "Could not find service")
        if args[:2] == ["launchctl", "bootstrap"]:
            with (tmp_path / "update-custody.lock").open("a") as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            loaded = True
        return subprocess.CompletedProcess(args, 0, "", "")

    def run():
        result = health.HealthCheckResult(
            checked_at="2026-11-27T12:00:00+00:00",
            ok=False,
            issues=[health.HealthIssue(code, "critical", "absent")],
        )
        counters, tripped = health._apply_heals(
            result=result,
            issue_labels={code: (label, tmp_path / f"{label}.plist")},
            previous_failures={key: 3},
            previous_tripped=set(),
            config=config,
            command_runner=runner,
        )
        return result, counters, tripped

    with (tmp_path / "update-custody.lock").open("a") as updater:
        fcntl.flock(updater, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, counters, tripped = run()
        assert not commands
        assert counters == {key: 3}
        assert not tripped and not result.actions
    result, counters, tripped = run()
    assert len([args for args in commands if args[:2] == ["launchctl", "bootstrap"]]) == 1
    assert result.actions == [f"bootstrap:{label}"]
    assert not counters and not tripped
