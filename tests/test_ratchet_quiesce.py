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
