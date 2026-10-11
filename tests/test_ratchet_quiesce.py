"""Unit coverage only: this file never executes this Mac's launchctl."""

import subprocess

import pytest

from scripts import ratchet_quiesce as replay


def test_local_execution_refuses_before_launchctl(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match="GitHub macOS"):
        replay.require_runner()
    assert not calls


@pytest.mark.parametrize("namespace", ["com.brainlayer.watch", "../x", "", "com.brainlayer.ratchettest.x/subject"])
def test_namespace_cannot_address_real_jobs(namespace):
    with pytest.raises(ValueError):
        replay.labels(namespace)


def test_cleanup_uses_only_owned_labels_and_removes_plists(tmp_path):
    mapping = replay.labels("com.brainlayer.ratchettest.123-1-head")
    agents = tmp_path / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    for label in mapping.values():
        (agents / f"{label}.plist").write_text("fixture")
    calls = []

    def fake(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 113 if args[0] == "print" else 0, "", "")

    replay.cleanup(tmp_path, mapping, command=fake)
    assert not list(agents.glob("*.plist"))
    assert len(calls) == 8
    assert all(any(label in " ".join(call) for label in mapping.values()) for call in calls)


def test_missing_gui_is_failure(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_OS", "macOS")
    monkeypatch.setattr(replay.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(replay.shutil, "which", lambda _: "/bin/launchctl")
    monkeypatch.setattr(replay, "launchctl", lambda *args: subprocess.CompletedProcess(args, 1, "", "no GUI"))
    with pytest.raises(RuntimeError, match="gui domain"):
        replay.require_runner()


def test_cleanup_cannot_pass_while_a_job_remains_loaded(tmp_path):
    mapping = replay.labels("com.brainlayer.ratchettest.123-1-head")
    with pytest.raises(RuntimeError, match="unloaded"):
        replay.cleanup(tmp_path, mapping, command=lambda *args: subprocess.CompletedProcess(args, 0, "loaded", ""))


def test_loaded_subject_without_new_watchdog_log_is_not_held(tmp_path, monkeypatch):
    from itertools import count
    from types import SimpleNamespace

    import brainlayer
    from brainlayer import launchd_primitive

    source = tmp_path / "source"
    script = source / "scripts/launchd/fleet-watchdog.sh"
    script.parent.mkdir(parents=True)
    script.write_text("# fake watchdog; no launchctl execution\n")
    home = tmp_path / "fixture-home"
    log = home / "Library/Logs/brainlayer/fleet-watchdog.log"
    phase = {"value": "initial"}

    def loaded(service):
        if service == "fleet-watchdog":
            return phase["value"] != "held"
        if phase["value"] == "control-down":
            phase["value"] = "control-revived"
            return False
        return True  # subject unexpectedly remains loaded during the hold

    def fake_launchctl(*args, **kwargs):
        if args[0] == "bootout":
            phase["value"] = "control-down"
            log.parent.mkdir(parents=True)
            log.write_text("re-bootstrapped positive control\n")
        return subprocess.CompletedProcess(args, 0, "", "")

    def resume(*args):
        phase["value"] = "resumed"
        return []

    maintenance = SimpleNamespace(
        __file__=str(source / "src/brainlayer/maintenance.py"),
        PAUSE_SENTINEL_PATH=home / "data/pause.sentinel",
        _launchd_label=lambda service: service,
        _service_is_loaded=loaded,
        _quiesce_services=lambda *args: phase.update(value="held"),
        _resume_services=resume,
    )
    monkeypatch.setattr(brainlayer, "maintenance", maintenance, raising=False)
    monkeypatch.setattr(launchd_primitive, "is_launchd_label_disabled", lambda _: phase["value"] == "held")
    monkeypatch.setattr(replay, "launchctl", fake_launchctl)
    monkeypatch.setattr(replay.sys, "path", list(replay.sys.path))
    monkeypatch.setattr(replay.time, "monotonic", count().__next__)
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    report = replay.replay(source, home, replay.labels("com.brainlayer.ratchettest.fake"))
    assert report["positive_control"] and report["resumed"]
    assert not report["revived"]  # no new watchdog log beyond the control
    assert not report["held"]


def test_row_binds_both_replays_to_their_shas_and_requires_every_postcondition(tmp_path):
    import json

    from scripts.ci_ratchet_table import GREEN, RED, row_quiesce

    report = tmp_path / "report.json"
    assert row_quiesce(report, None, "a" * 40).status == RED
    report.write_text("{}")
    assert row_quiesce(report, None, "a" * 40).status == RED
    sample = dict(positive_control=True, resumed=True, marker_removed=True, interval_seconds=3, window_seconds=8)
    sample["resume_errors"] = []
    data = {
        "bug": dict(sample, sha="e59cf87142c88db044dd701cf5ee993267c8d090", status="RED", revived=True),
        "head": dict(sample, sha="a" * 40, status="GREEN", revived=False, held=True),
    }
    report.write_text(json.dumps(data))
    assert row_quiesce(report, None, "a" * 40).status == GREEN
    assert row_quiesce(report, None, "b" * 40).status == RED
    for side, key, value in [
        ("bug", "revived", False),
        ("head", "held", False),
        ("head", "marker_removed", False),
        ("head", "window_seconds", 5),
        ("head", "resume_errors", ["bootstrap failed"]),
    ]:
        changed = json.loads(json.dumps(data))
        changed[side][key] = value
        report.write_text(json.dumps(changed))
        assert row_quiesce(report, None, "a" * 40).status == RED
