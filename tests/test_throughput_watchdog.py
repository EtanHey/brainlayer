from __future__ import annotations

import importlib.util
import json
import os
import plistlib
import sqlite3
import sys
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


def test_first_observation_establishes_a_baseline_without_restart(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(module, tmp_path)
    commands: list[list[str]] = []

    result = module.run_once(
        config,
        now_epoch=1_000,
        highwater_reader=lambda _path: 40,
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

    def command_runner(args: list[str]):
        command_events.append("command:" + " ".join(args))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def alert_fn(_config, _result):
        command_events.append("alert")

    evidence = module.SourceEvidence(2, 120, 3, 999.0)
    baseline = module.run_once(
        config,
        now_epoch=1_000,
        highwater_reader=lambda _path: 40,
        source_probe=lambda _config, _now: evidence,
        command_runner=command_runner,
        alert_fn=alert_fn,
    )
    first = module.run_once(
        config,
        now_epoch=1_060,
        highwater_reader=lambda _path: 40,
        source_probe=lambda _config, _now: evidence,
        command_runner=command_runner,
        alert_fn=alert_fn,
    )
    second = module.run_once(
        config,
        now_epoch=1_120,
        highwater_reader=lambda _path: 40,
        source_probe=lambda _config, _now: evidence,
        command_runner=command_runner,
        alert_fn=alert_fn,
    )
    third = module.run_once(
        config,
        now_epoch=1_180,
        highwater_reader=lambda _path: 40,
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
        f"command:launchctl kickstart -k gui/{os.getuid()}/com.example.brainlayer.watch",
        f"command:launchctl print gui/{os.getuid()}/com.example.brainlayer.watch",
    ]


def test_chunk_progress_or_no_pending_input_resets_stall_counter(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(module, tmp_path, stall_threshold=2)
    pending = module.SourceEvidence(1, 20, 1, 999.0)
    idle = module.SourceEvidence(0, 0, 1, 999.0)

    module.run_once(
        config,
        now_epoch=1_000,
        highwater_reader=lambda _path: 40,
        source_probe=lambda _config, _now: pending,
    )
    stalled = module.run_once(
        config,
        now_epoch=1_060,
        highwater_reader=lambda _path: 40,
        source_probe=lambda _config, _now: pending,
    )
    progressed = module.run_once(
        config,
        now_epoch=1_120,
        highwater_reader=lambda _path: 41,
        source_probe=lambda _config, _now: pending,
    )
    idle_result = module.run_once(
        config,
        now_epoch=1_180,
        highwater_reader=lambda _path: 41,
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
        highwater_reader=lambda _path: 40,
        source_probe=lambda _config, _now: evidence,
    )

    def command_runner(args: list[str]):
        if args[:3] == ["launchctl", "kickstart", "-k"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="kickstart refused")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = module.run_once(
        config,
        now_epoch=1_060,
        highwater_reader=lambda _path: 40,
        source_probe=lambda _config, _now: evidence,
        command_runner=command_runner,
        alert_fn=lambda _config, _result: None,
    )

    assert result.action == "recovery_failed"
    assert "kickstart refused" in result.error
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert "last_restart_epoch" not in state
    assert state["last_recovery_error"] == result.error


def test_alert_failure_does_not_block_watcher_recovery(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(module, tmp_path, stall_threshold=1)
    evidence = module.SourceEvidence(1, 20, 1, 999.0)
    module.run_once(
        config,
        now_epoch=1_000,
        highwater_reader=lambda _path: 40,
        source_probe=lambda _config, _now: evidence,
    )
    commands: list[list[str]] = []

    def broken_alert(_config, _result):
        raise OSError("notification path unavailable")

    result = module.run_once(
        config,
        now_epoch=1_060,
        highwater_reader=lambda _path: 40,
        source_probe=lambda _config, _now: evidence,
        command_runner=lambda args: commands.append(args),
        alert_fn=broken_alert,
    )

    assert result.action == "kickstart:com.example.brainlayer.watch"
    assert result.alert_error == "notification path unavailable"
    assert any(command[:3] == ["launchctl", "kickstart", "-k"] for command in commands)


def test_source_probe_only_treats_recent_registry_tracked_lag_as_pending(tmp_path: Path) -> None:
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
    assert result.pending_files == 1
    assert result.pending_bytes == pending.stat().st_size - 3
    assert result.untracked_recent_files == 1


def test_highwater_is_scoped_to_realtime_watcher_without_a_count_scan(tmp_path: Path) -> None:
    module = _load_module()
    db_path = tmp_path / "brainlayer.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, source TEXT)")
        connection.executemany(
            "INSERT INTO chunks(id, source) VALUES (?, ?)",
            [("w1", "realtime_watcher"), ("m1", "mcp"), ("w2", "realtime_watcher")],
        )

    assert module.read_watcher_highwater(db_path) == 3


def test_launchagent_runs_every_minute_and_invokes_installed_script() -> None:
    plist = plistlib.loads(PLIST_PATH.read_bytes())

    assert plist["Label"] == "com.brainlayer.throughput-watchdog"
    assert plist["StartInterval"] == 60
    assert plist["RunAtLoad"] is True
    assert plist["ProgramArguments"] == ["__PYTHON_BIN__", "__THROUGHPUT_WATCHDOG_SCRIPT__", "--json"]
    assert plist["EnvironmentVariables"]["HOME"] == "__HOME__"
    assert plist["SoftResourceLimits"]["NumberOfFiles"] >= 4096
