from __future__ import annotations

import importlib.util
import json
import os
import plistlib
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "launchd" / "throughput-watchdog.py"
PLIST_PATH = REPO_ROOT / "scripts" / "launchd" / "com.brainlayer.throughput-watchdog.plist"


def _load_module():
    spec = importlib.util.spec_from_file_location("brainlayer_throughput_watchdog", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(module, tmp_path: Path, *, stall_threshold: int = 3, cooldown_seconds: int = 600):
    return module.Config(
        db_path=tmp_path / "brainlayer.db",
        registry_path=tmp_path / "offsets.json",
        state_path=tmp_path / "throughput-watchdog-state.json",
        source_roots=(tmp_path / "sessions",),
        watch_label="com.example.brainlayer.watch",
        watch_plist_path=tmp_path / "com.example.brainlayer.watch.plist",
        stall_threshold=stall_threshold,
        cooldown_seconds=cooldown_seconds,
    )


def _progress(module, chunk_rowid: int, liveness_rowid: int = 0):
    return module.WatcherProgress(
        chunk_rowid=chunk_rowid,
        liveness_rowid=liveness_rowid,
    )


def _successful_recovery_runner(events: list[str] | None = None):
    print_count = 0
    kickstarted = False

    def command_runner(args: list[str]):
        nonlocal print_count, kickstarted
        if events is not None:
            events.append("command:" + " ".join(args))
        if args[:2] == ["launchctl", "print"]:
            print_count += 1
            if print_count <= 2:
                stdout = "state = running\npid = 4321\n"
            elif kickstarted:
                stdout = "state = running\npid = 9876\n"
            else:
                stdout = "state = exited\n"
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        if args[:3] == ["launchctl", "kickstart", "-k"]:
            kickstarted = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return command_runner


def test_first_observation_establishes_a_baseline_without_restart(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(module, tmp_path)
    commands: list[list[str]] = []

    result = module.run_once(
        config,
        now_epoch=1_000,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: module.SourceEvidence(2, 120, 3, 999.0),
        command_runner=lambda args: commands.append(args),
    )

    assert result.action == "baseline"
    assert result.stalled_ticks == 0
    assert commands == []


def test_process_alive_zero_throughput_with_pending_bytes_kickstarts_after_threshold(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(module, tmp_path, stall_threshold=3)
    command_events: list[str] = []

    command_runner = _successful_recovery_runner(command_events)

    def alert_fn(_config, _result):
        command_events.append("alert")

    evidence = module.SourceEvidence(2, 120, 3, 999.0)
    baseline = module.run_once(
        config,
        now_epoch=1_000,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
        command_runner=command_runner,
        alert_fn=alert_fn,
    )
    first = module.run_once(
        config,
        now_epoch=1_060,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
        command_runner=command_runner,
        alert_fn=alert_fn,
    )
    second = module.run_once(
        config,
        now_epoch=1_120,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
        command_runner=command_runner,
        alert_fn=alert_fn,
    )
    third = module.run_once(
        config,
        now_epoch=1_180,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
        command_runner=command_runner,
        alert_fn=alert_fn,
    )

    assert baseline.action == "baseline"
    assert [first.stalled_ticks, second.stalled_ticks] == [1, 2]
    assert third.action == "kickstart:com.example.brainlayer.watch"
    assert third.stalled_ticks == 0
    assert command_events[0] == "alert"
    assert command_events[1:] == [
        f"command:launchctl print gui/{os.getuid()}/com.example.brainlayer.watch",
        f"command:launchctl print gui/{os.getuid()}/com.example.brainlayer.watch",
        "command:/bin/kill -9 4321",
        f"command:launchctl print gui/{os.getuid()}/com.example.brainlayer.watch",
        f"command:launchctl kickstart -k gui/{os.getuid()}/com.example.brainlayer.watch",
        f"command:launchctl print gui/{os.getuid()}/com.example.brainlayer.watch",
    ]


def test_recovery_does_not_sigkill_a_pid_that_changed_during_validation(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(module, tmp_path)
    commands: list[list[str]] = []
    outputs = iter(
        [
            "state = running\npid = 4321\n",
            "state = running\npid = 9876\n",
        ]
    )

    def command_runner(args: list[str]):
        commands.append(args)
        stdout = next(outputs) if args[:2] == ["launchctl", "print"] else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    action = module._restart_watch(config, command_runner)

    assert action == "respawn:com.example.brainlayer.watch"
    assert commands == [
        ["launchctl", "print", f"gui/{os.getuid()}/com.example.brainlayer.watch"],
        ["launchctl", "print", f"gui/{os.getuid()}/com.example.brainlayer.watch"],
    ]


def test_recovery_does_not_kickstart_while_sigkilled_pid_still_owns_the_job(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    config = _config(module, tmp_path)
    commands: list[list[str]] = []

    def command_runner(args: list[str]):
        commands.append(args)
        stdout = "state = running\npid = 4321\n" if args[:2] == ["launchctl", "print"] else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(module, "SIGKILL_EXIT_TIMEOUT_SECONDS", 0.0)
    try:
        module._restart_watch(config, command_runner, sleep_fn=lambda _seconds: None)
    except RuntimeError as exc:
        assert "survived SIGKILL" in str(exc)
    else:
        raise AssertionError("recovery must fail while the lock-holding pid survives SIGKILL")

    assert ["/bin/kill", "-9", "4321"] in commands
    assert not any(command[:3] == ["launchctl", "kickstart", "-k"] for command in commands)


def test_recovery_does_not_treat_failed_status_poll_as_pid_exit(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    config = _config(module, tmp_path)
    commands: list[list[str]] = []
    print_count = 0
    kickstarted = False

    def command_runner(args: list[str]):
        nonlocal print_count, kickstarted
        commands.append(args)
        if args[:2] == ["launchctl", "print"]:
            print_count += 1
            if print_count <= 2:
                return SimpleNamespace(returncode=0, stdout="state = running\npid = 4321\n", stderr="")
            if print_count == 3:
                return SimpleNamespace(returncode=1, stdout="", stderr="launchctl temporarily unavailable")
            if kickstarted:
                return SimpleNamespace(returncode=0, stdout="state = running\npid = 9876\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="state = exited\n", stderr="")
        if args[:3] == ["launchctl", "kickstart", "-k"]:
            kickstarted = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "SIGKILL_EXIT_TIMEOUT_SECONDS", 0.0)
    try:
        module._restart_watch(config, command_runner, sleep_fn=lambda _seconds: None)
    except RuntimeError as exc:
        assert "could not confirm watcher pid 4321 exited" in str(exc)
    else:
        raise AssertionError("a failed launchctl poll must not authorize a second writer")

    assert ["/bin/kill", "-9", "4321"] in commands
    assert not any(command[:3] == ["launchctl", "kickstart", "-k"] for command in commands)


def test_default_runner_allows_bounded_time_for_uninterruptible_watcher_kickstart(monkeypatch) -> None:
    module = _load_module()
    observed_timeouts: list[int] = []

    def fake_run(args, **kwargs):
        observed_timeouts.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._default_command_runner(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/com.brainlayer.watch"])
    module._default_command_runner(["launchctl", "print", f"gui/{os.getuid()}/com.brainlayer.watch"])

    assert observed_timeouts == [45, 15]


def test_chunk_progress_or_no_pending_input_resets_stall_counter(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(module, tmp_path, stall_threshold=2)
    pending = module.SourceEvidence(1, 20, 1, 999.0)
    idle = module.SourceEvidence(0, 0, 1, 999.0)

    module.run_once(
        config,
        now_epoch=1_000,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: pending,
    )
    stalled = module.run_once(
        config,
        now_epoch=1_060,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: pending,
    )
    progressed = module.run_once(
        config,
        now_epoch=1_120,
        progress_reader=lambda _path: _progress(module, 41),
        source_probe=lambda _config, _now: pending,
    )
    idle_result = module.run_once(
        config,
        now_epoch=1_180,
        progress_reader=lambda _path: _progress(module, 41),
        source_probe=lambda _config, _now: idle,
    )

    assert stalled.stalled_ticks == 1
    assert progressed.action == "progress"
    assert progressed.watcher_highwater_delta == 1
    assert progressed.stalled_ticks == 0
    assert idle_result.action == "idle"
    assert idle_result.stalled_ticks == 0


def test_restart_failure_is_visible_and_not_recorded_as_recovered(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(module, tmp_path, stall_threshold=1)
    evidence = module.SourceEvidence(1, 20, 1, 999.0)
    module.run_once(
        config,
        now_epoch=1_000,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
    )

    def command_runner(args: list[str]):
        if args[:3] == ["launchctl", "kickstart", "-k"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="kickstart refused")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = module.run_once(
        config,
        now_epoch=1_060,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
        command_runner=command_runner,
        alert_fn=lambda _config, _result: None,
    )

    assert result.action == "recovery_failed"
    assert "kickstart refused" in result.error
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert state["last_restart_epoch"] == 1_060
    assert state["last_recovery_error"] == result.error

    commands: list[list[str]] = []
    cooldown = module.run_once(
        config,
        now_epoch=1_120,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
        command_runner=lambda args: commands.append(args),
        alert_fn=lambda _config, _result: None,
    )
    assert cooldown.action == "cooldown"
    assert not any(command[:3] == ["launchctl", "kickstart", "-k"] for command in commands)


def test_alert_failure_does_not_block_watcher_recovery(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(module, tmp_path, stall_threshold=1)
    evidence = module.SourceEvidence(1, 20, 1, 999.0)
    module.run_once(
        config,
        now_epoch=1_000,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
    )
    command_events: list[str] = []

    def broken_alert(_config, _result):
        raise OSError("notification path unavailable")

    result = module.run_once(
        config,
        now_epoch=1_060,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
        command_runner=_successful_recovery_runner(command_events),
        alert_fn=broken_alert,
    )

    assert result.action == "kickstart:com.example.brainlayer.watch"
    assert result.alert_error == "notification path unavailable"
    assert any(event.startswith("command:launchctl kickstart -k") for event in command_events)


def test_source_probe_treats_recent_unregistered_input_as_pending(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(module, tmp_path)
    source_root = config.source_roots[0]
    source_root.mkdir(parents=True)
    pending = source_root / "pending.jsonl"
    caught_up = source_root / "caught-up.jsonl"
    untracked = source_root / "untracked.jsonl"
    stale = source_root / "stale.jsonl"
    for source_file in (pending, caught_up, untracked, stale):
        source_file.write_text("1234567890\n", encoding="utf-8")
    os.utime(stale, (100, 100))
    config.registry_path.write_text(
        json.dumps(
            {
                str(pending): {"offset": 3},
                str(caught_up): {"offset": caught_up.stat().st_size},
                str(stale): {"offset": 0},
            }
        ),
        encoding="utf-8",
    )

    result = module.collect_source_evidence(config, now_epoch=1_000)

    assert result.recent_files == 3
    assert result.pending_files == 2
    assert result.pending_bytes == pending.stat().st_size - 3 + untracked.stat().st_size
    assert result.untracked_recent_files == 1


def test_source_probe_ignores_unwatched_cursor_project_jsonl(tmp_path: Path) -> None:
    module = _load_module()
    cursor_projects = tmp_path / ".cursor" / "projects"
    watched = cursor_projects / "repo" / "agent-transcripts" / "agent" / "session.jsonl"
    unwatched = cursor_projects / "repo" / "tool-state.jsonl"
    watched.parent.mkdir(parents=True)
    unwatched.parent.mkdir(parents=True, exist_ok=True)
    watched.write_text("watched\n", encoding="utf-8")
    unwatched.write_text("not watcher input\n", encoding="utf-8")
    config = replace(_config(module, tmp_path), source_roots=(cursor_projects,))
    config.registry_path.write_text("{}", encoding="utf-8")

    result = module.collect_source_evidence(config, now_epoch=1_000)

    assert result.recent_files == 1
    assert result.pending_files == 1
    assert result.pending_bytes == watched.stat().st_size
    assert result.untracked_recent_files == 1


def test_source_probe_treats_truncated_tracked_file_as_pending_rewind(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(module, tmp_path)
    config.source_roots[0].mkdir(parents=True)
    rewound = config.source_roots[0] / "rewound.jsonl"
    rewound.write_text("new\n", encoding="utf-8")
    config.registry_path.write_text(
        json.dumps({str(rewound): {"offset": rewound.stat().st_size + 100}}),
        encoding="utf-8",
    )

    result = module.collect_source_evidence(config, now_epoch=1_000)

    assert result.pending_files == 1
    assert result.pending_bytes == rewound.stat().st_size


def test_config_defaults_registry_next_to_selected_database(tmp_path: Path) -> None:
    module = _load_module()
    custom_db = tmp_path / "sandbox" / "brainlayer.db"

    args = module._build_parser().parse_args(["--db", str(custom_db)])
    config = module._config_from_args(args)

    assert config.db_path == custom_db
    assert config.registry_path == custom_db.parent / "offsets.json"


def test_source_probe_fails_visibly_when_find_cannot_complete(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    config = _config(module, tmp_path)
    config.source_roots[0].mkdir(parents=True)
    config.registry_path.write_text("{}", encoding="utf-8")

    def fail_scan(*_args, **_kwargs):
        raise RuntimeError("recent-source scan failed: permission denied")

    monkeypatch.setattr(module, "_bounded_nul_paths", fail_scan)

    try:
        module.collect_source_evidence(config, now_epoch=1_000)
    except RuntimeError as exc:
        assert "permission denied" in str(exc)
    else:
        raise AssertionError("an incomplete source scan must not report idle")


def test_bounded_find_streams_and_terminates_before_buffering_unbounded_paths(tmp_path: Path) -> None:
    module = _load_module()
    producer = tmp_path / "produce_paths.py"
    producer.write_text(
        "import sys, time\nsys.stdout.buffer.write(b'a\\0b\\0c\\0')\nsys.stdout.buffer.flush()\ntime.sleep(30)\n",
        encoding="utf-8",
    )

    try:
        module._bounded_nul_paths(
            [sys.executable, str(producer)],
            max_paths=2,
            timeout_seconds=5,
        )
    except RuntimeError as exc:
        assert "exceeded 2" in str(exc)
    else:
        raise AssertionError("the producer must be terminated as soon as the path bound is exceeded")


def test_bounded_find_preserves_nonzero_exit_and_timeout_failures(tmp_path: Path) -> None:
    module = _load_module()
    failed = tmp_path / "failed_scan.py"
    failed.write_text(
        "import sys\nsys.stderr.write('permission denied')\nraise SystemExit(7)\n",
        encoding="utf-8",
    )
    stalled = tmp_path / "stalled_scan.py"
    stalled.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")

    try:
        module._bounded_nul_paths(
            [sys.executable, str(failed)],
            max_paths=2,
            timeout_seconds=5,
        )
    except RuntimeError as exc:
        assert "permission denied" in str(exc)
    else:
        raise AssertionError("a nonzero find exit must fail the probe")

    try:
        module._bounded_nul_paths(
            [sys.executable, str(stalled)],
            max_paths=2,
            timeout_seconds=0.1,
        )
    except RuntimeError as exc:
        assert "exceeded 0.1s" in str(exc)
    else:
        raise AssertionError("a stalled find process must be terminated at the scan timeout")


def test_source_probe_fails_closed_when_a_recent_file_cannot_be_statted(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    config = _config(module, tmp_path)
    source_root = config.source_roots[0]
    source_root.mkdir(parents=True)
    unreadable = source_root / "unreadable.jsonl"
    unreadable.write_text("{}\n", encoding="utf-8")
    config.registry_path.write_text("{}", encoding="utf-8")
    original_stat = Path.stat

    def fail_selected_stat(path: Path, *args, **kwargs):
        if path == unreadable:
            raise PermissionError("stat denied")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_selected_stat)

    try:
        module.collect_source_evidence(config, now_epoch=int(original_stat(unreadable).st_mtime) + 1)
    except RuntimeError as exc:
        assert "stat denied" in str(exc)
    else:
        raise AssertionError("partial source evidence must never authorize recovery")


def test_progress_tracks_realtime_chunks_and_durable_watcher_liveness(tmp_path: Path) -> None:
    module = _load_module()
    db_path = tmp_path / "brainlayer.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, source TEXT)")
        connection.execute(
            "CREATE TABLE watcher_liveness_events (id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_id TEXT, ingested_at INTEGER)"
        )
        connection.executemany(
            "INSERT INTO chunks(id, source) VALUES (?, ?)",
            [("w1", "realtime_watcher"), ("m1", "mcp"), ("w2", "realtime_watcher")],
        )
        connection.executemany(
            "INSERT INTO watcher_liveness_events(chunk_id, ingested_at) VALUES (?, ?)",
            [("w1", 100), ("manual-canonical", 101)],
        )

    assert module.read_watcher_progress(db_path) == _progress(module, 3, 2)


def test_liveness_progress_prevents_false_restart_for_dedupe_only_ingest(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(module, tmp_path, stall_threshold=1)
    evidence = module.SourceEvidence(1, 20, 1, 999.0)
    commands: list[list[str]] = []
    module.run_once(
        config,
        now_epoch=1_000,
        progress_reader=lambda _path: _progress(module, 40, 10),
        source_probe=lambda _config, _now: evidence,
    )

    result = module.run_once(
        config,
        now_epoch=1_060,
        progress_reader=lambda _path: _progress(module, 40, 11),
        source_probe=lambda _config, _now: evidence,
        command_runner=lambda args: commands.append(args),
    )

    assert result.action == "progress"
    assert result.watcher_highwater_delta == 0
    assert result.watcher_liveness_highwater_delta == 1
    assert commands == []


def test_recovery_requires_kickstarted_job_to_reach_running_state(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(module, tmp_path, stall_threshold=1)
    evidence = module.SourceEvidence(1, 20, 1, 999.0)
    module.run_once(
        config,
        now_epoch=1_000,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
    )

    def command_runner(args: list[str]):
        stdout = "state = exited\nlast exit code = 1\n" if args[:2] == ["launchctl", "print"] else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    result = module.run_once(
        config,
        now_epoch=1_060,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
        command_runner=command_runner,
        alert_fn=lambda _config, _result: None,
    )

    assert result.action == "recovery_failed"
    assert "did not reach running state" in result.error


def test_recovery_attempt_is_persisted_before_external_kickstart(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    config = _config(module, tmp_path, stall_threshold=1)
    evidence = module.SourceEvidence(1, 20, 1, 999.0)
    module.run_once(
        config,
        now_epoch=1_000,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
    )
    events: list[str] = []
    original_write = module._atomic_write_json

    def recording_write(path: Path, payload: dict) -> None:
        events.append(f"write:{payload.get('last_action')}")
        original_write(path, payload)

    command_runner = _successful_recovery_runner(events)

    monkeypatch.setattr(module, "_atomic_write_json", recording_write)
    module.run_once(
        config,
        now_epoch=1_060,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
        command_runner=command_runner,
        alert_fn=lambda _config, _result: None,
    )

    attempt_index = events.index("write:recovery_attempt")
    kickstart_index = next(
        index for index, event in enumerate(events) if event.startswith("command:launchctl kickstart")
    )
    assert attempt_index < kickstart_index


def test_recovery_attempt_survives_a_post_kickstart_state_write_failure(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    config = _config(module, tmp_path, stall_threshold=1)
    evidence = module.SourceEvidence(1, 20, 1, 999.0)
    module.run_once(
        config,
        now_epoch=1_000,
        progress_reader=lambda _path: _progress(module, 40),
        source_probe=lambda _config, _now: evidence,
    )
    original_write = module._atomic_write_json
    writes = 0

    def fail_final_write(path: Path, payload: dict) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("disk unavailable after kickstart")
        original_write(path, payload)

    command_runner = _successful_recovery_runner()

    monkeypatch.setattr(module, "_atomic_write_json", fail_final_write)
    try:
        module.run_once(
            config,
            now_epoch=1_060,
            progress_reader=lambda _path: _progress(module, 40),
            source_probe=lambda _config, _now: evidence,
            command_runner=command_runner,
            alert_fn=lambda _config, _result: None,
        )
    except OSError as exc:
        assert "disk unavailable" in str(exc)
    else:
        raise AssertionError("the final state-write failure must remain visible")

    durable_state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert durable_state["last_action"] == "recovery_attempt"
    assert durable_state["last_restart_epoch"] == 1_060


def test_launchagent_runs_every_minute_and_invokes_installed_script() -> None:
    plist = plistlib.loads(PLIST_PATH.read_bytes())

    assert plist["Label"] == "com.brainlayer.throughput-watchdog"
    assert plist["StartInterval"] == 60
    assert plist["RunAtLoad"] is True
    assert plist["ProgramArguments"] == [
        "__BRAINLAYER_ENV_RUN__",
        "__PYTHON_BIN__",
        "__THROUGHPUT_WATCHDOG_SCRIPT__",
        "--json",
    ]
    assert plist["EnvironmentVariables"]["HOME"] == "__HOME__"
    assert plist["EnvironmentVariables"]["BRAINLAYER_ENV_FILE"] == "__BRAINLAYER_ENV_FILE__"
    assert plist["EnvironmentVariables"]["BRAINLAYER_LAUNCHD_SERVICE"] == "watch"
    assert plist["SoftResourceLimits"]["NumberOfFiles"] >= 4096


def test_alert_framing_pages_once_per_episode_not_per_kickstart(tmp_path: Path) -> None:
    """Alert-framing: a chronic wedge sawtooth pages on the FIRST wedge of an
    episode only, re-paging for a NEW episode after a sustained healthy run."""
    module = _load_module()
    config = _config(module, tmp_path, stall_threshold=3, cooldown_seconds=0)
    alerts: list[int] = []

    process = {"pid": 4321, "running": True}

    def command_runner(args: list[str]):
        stdout = ""
        if args[:2] == ["launchctl", "print"]:
            if process["running"]:
                stdout = f"state = running\npid = {process['pid']}\n"
            else:
                stdout = "state = exited\n"
        elif args[:2] == ["/bin/kill", "-9"] and args[2:3] == [str(process["pid"])]:
            process["running"] = False
        elif args[:3] == ["launchctl", "kickstart", "-k"]:
            process["pid"] += 1
            process["running"] = True
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    clock = {"t": 1_000}

    def tick(rowid: int, *, pending: bool) -> object:
        clock["t"] += 60
        evidence = module.SourceEvidence(2, 120, 3, 999.0) if pending else module.SourceEvidence(0, 0, 0, 999.0)
        res = module.run_once(
            config,
            now_epoch=clock["t"],
            progress_reader=lambda _p, r=rowid: _progress(module, r, r),
            source_probe=lambda _c, _n, e=evidence: e,
            command_runner=command_runner,
            alert_fn=lambda _c, _r: alerts.append(clock["t"]),
        )
        return res

    tick(40, pending=True)  # baseline
    tick(40, pending=True)  # stall 1
    tick(40, pending=True)  # stall 2
    k1 = tick(40, pending=True)  # stall 3 -> kickstart + FIRST alert
    assert k1.action.startswith("kickstart:")
    assert len(alerts) == 1  # episode 1 paged once

    tick(50, pending=True)  # brief recovery (1 healthy tick, < reset)
    tick(50, pending=True)  # stall 1
    tick(50, pending=True)  # stall 2
    k2 = tick(50, pending=True)  # stall 3 -> kickstart, SUPPRESSED (same episode)
    assert k2.action.startswith("kickstart:")
    assert len(alerts) == 1  # sawtooth did NOT re-page

    tick(60, pending=True)  # sustained recovery: healthy 1
    tick(70, pending=True)  # healthy 2
    tick(80, pending=True)  # healthy 3 -> episode latch clears
    tick(80, pending=True)  # stall 1
    tick(80, pending=True)  # stall 2
    k3 = tick(80, pending=True)  # stall 3 -> kickstart + NEW-episode alert
    assert k3.action.startswith("kickstart:")
    assert len(alerts) == 2  # new episode paged again


def test_failed_alert_does_not_latch_episode_and_retries(tmp_path: Path) -> None:
    """A transient notify failure on the first wedge must NOT silence the
    episode — page-once must never become page-zero; the next wedge retries."""
    module = _load_module()
    config = _config(module, tmp_path, stall_threshold=3, cooldown_seconds=0)
    calls = {"n": 0}

    def command_runner(args):
        stdout = "state = running\npid = 4321\n" if args[:2] == ["launchctl", "print"] else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    def flaky_alert(_config, _result):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("notify endpoint down")

    clock = {"t": 1_000}

    def tick():
        clock["t"] += 60
        return module.run_once(
            config,
            now_epoch=clock["t"],
            progress_reader=lambda _p: _progress(module, 40, 40),
            source_probe=lambda _c, _n: module.SourceEvidence(2, 120, 3, 999.0),
            command_runner=command_runner,
            alert_fn=flaky_alert,
        )

    tick()  # baseline (establishes highwater; not a stall)
    tick()  # stall 1
    tick()  # stall 2
    tick()  # stall 3 -> kickstart, alert RAISES (call 1), NOT latched
    assert calls["n"] == 1
    tick()  # stall 1
    tick()  # stall 2
    tick()  # stall 3 -> kickstart, alert RETRIES in same episode (call 2)
    assert calls["n"] == 2  # retried, not silenced by the earlier failure
