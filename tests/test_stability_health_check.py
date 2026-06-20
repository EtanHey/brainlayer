from __future__ import annotations

import json
import plistlib
import sqlite3
import inspect
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import brainlayer.health_check as health_check
from brainlayer.health_check import HealthCheckConfig, run_health_check

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_db(path: Path, *, total: int, vector_rows: int) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                content TEXT,
                archived_at TEXT,
                superseded_by TEXT,
                aggregated_into TEXT,
                archived INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            );
            CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
            """
        )
        for index in range(total):
            conn.execute(
                "INSERT INTO chunks (id, content) VALUES (?, ?)",
                (f"chunk-{index}", f"content {index}"),
            )
        for index in range(vector_rows):
            conn.execute("INSERT INTO chunk_vectors_rowids (id) VALUES (?)", (f"chunk-{index}",))
        conn.commit()
    finally:
        conn.close()


def _ok_canary(_socket_path: Path, _query: str, _timeout_seconds: float) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": '## Search results for "agentopology" - 1 of 1 shown\n\n### 1. relevant result',
                }
            ]
        },
    }


def test_backlog_batch_zero_alarms_but_waits_until_repeated_failure_to_kickstart_hotlane(tmp_path, capsys):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    _make_db(db_path, total=4, vector_rows=3)
    commands: list[list[str]] = []

    config = HealthCheckConfig(db_path=db_path, state_path=state_path, heal=True)
    first_result = run_health_check(
        config,
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 0 --enrich-limit 25\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=lambda args: commands.append(args),
        now_fn=lambda: datetime(2026, 6, 19, 4, 25, tzinfo=UTC),
    )

    assert first_result.ok is False
    assert "hotlane_backlog_disabled" in [issue.code for issue in first_result.issues]
    assert first_result.backlog_batch == 0
    assert not any(command[:3] == ["launchctl", "kickstart", "-k"] for command in commands)
    assert "kickstart" not in capsys.readouterr().err

    second_result = run_health_check(
        HealthCheckConfig(db_path=db_path, state_path=state_path, heal=True),
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 0 --enrich-limit 25\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=lambda args: commands.append(args),
        now_fn=lambda: datetime(2026, 6, 19, 4, 30, tzinfo=UTC),
    )

    assert second_result.ok is False
    assert "hotlane_backlog_disabled" in [issue.code for issue in second_result.issues]
    assert second_result.backlog_batch == 0
    kickstarts = [command for command in commands if command[:3] == ["launchctl", "kickstart", "-k"]]
    assert len(kickstarts) == 1
    assert "com.brainlayer.hotlane-brainbar" in " ".join(kickstarts[0])
    stderr = capsys.readouterr().err
    assert "heal action" in stderr
    assert "label=com.brainlayer.hotlane-brainbar" in stderr
    assert "issue=hotlane_backlog_disabled" in stderr
    assert "consecutive_failures=2" in stderr


def test_any_zero_backlog_batch_alarms_when_multiple_hotlanes_are_running(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    _make_db(db_path, total=4, vector_rows=3)
    commands: list[list[str]] = []

    result = run_health_check(
        HealthCheckConfig(db_path=db_path, state_path=state_path, heal=True, heal_min_consecutive_failures=1),
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 128\n"
            "456 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 0\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=lambda args: commands.append(args),
        now_fn=lambda: datetime(2026, 6, 19, 4, 30, tzinfo=UTC),
    )

    assert result.ok is False
    assert "hotlane_backlog_disabled" in [issue.code for issue in result.issues]
    assert result.backlog_batch == 0
    kickstarts = [command for command in commands if command[:3] == ["launchctl", "kickstart", "-k"]]
    assert len(kickstarts) == 1
    assert "com.brainlayer.hotlane-brainbar" in " ".join(kickstarts[0])


def test_missing_embeddings_not_draining_after_two_ticks(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    _make_db(db_path, total=5, vector_rows=3)
    state_path.write_text(
        json.dumps(
            {
                "missing_vectors": 2,
                "stalled_ticks": 1,
                "ts": "2026-06-19T04:25:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    result = run_health_check(
        HealthCheckConfig(db_path=db_path, state_path=state_path, max_stalled_ticks=2),
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py "
            "--interval 1 --backlog-batch 128 --enrich-limit 25\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=lambda _args: None,
        now_fn=lambda: datetime(2026, 6, 19, 4, 30, tzinfo=UTC),
    )

    assert result.ok is False
    assert "missing_embeddings_not_draining" in [issue.code for issue in result.issues]
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["missing_vectors"] == 2
    assert saved["stalled_ticks"] == 2


def test_missing_embeddings_climb_resets_stall_counter(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    _make_db(db_path, total=5, vector_rows=2)
    state_path.write_text(
        json.dumps(
            {
                "missing_vectors": 2,
                "stalled_ticks": 1,
                "ts": "2026-06-19T04:25:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    result = run_health_check(
        HealthCheckConfig(db_path=db_path, state_path=state_path, max_stalled_ticks=2),
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py "
            "--interval 1 --backlog-batch 128 --enrich-limit 25\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=lambda _args: None,
        now_fn=lambda: datetime(2026, 6, 19, 4, 30, tzinfo=UTC),
    )

    issue_codes = [issue.code for issue in result.issues]
    assert "missing_embeddings_climbing" in issue_codes
    assert "missing_embeddings_not_draining" not in issue_codes
    assert result.stalled_ticks == 0
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["missing_vectors"] == 3
    assert saved["stalled_ticks"] == 0


def test_heal_state_write_preserves_missing_vector_history_when_count_fails(tmp_path):
    state_path = tmp_path / "health-state.json"
    state_path.write_text(
        json.dumps(
            {
                "missing_vectors": 7,
                "stalled_ticks": 1,
                "ts": "2026-06-19T04:25:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    result = run_health_check(
        HealthCheckConfig(
            db_path=tmp_path / "missing" / "brainlayer.db",
            state_path=state_path,
            heal=True,
        ),
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 0 --enrich-limit 25\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=lambda _args: None,
        now_fn=lambda: datetime(2026, 6, 19, 4, 30, tzinfo=UTC),
    )

    assert result.ok is False
    assert "missing_embeddings_count_failed" in [issue.code for issue in result.issues]
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["missing_vectors"] == 7
    assert saved["stalled_ticks"] == 1
    assert saved["heal_failures"]["com.brainlayer.hotlane-brainbar:hotlane_backlog_disabled"] == 1


def test_brainbar_canary_error_waits_until_repeated_failure_to_kickstart_brainbar(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    _make_db(db_path, total=3, vector_rows=3)
    commands: list[list[str]] = []

    def failed_canary(_socket_path: Path, _query: str, _timeout_seconds: float) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": "Error: Database not available"}],
                "isError": True,
            },
        }

    config = HealthCheckConfig(db_path=db_path, state_path=state_path, heal=True)
    first_result = run_health_check(
        config,
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py "
            "--interval 1 --backlog-batch 128 --enrich-limit 25\n"
        ),
        socket_request_fn=failed_canary,
        command_runner=lambda args: commands.append(args),
        now_fn=lambda: datetime(2026, 6, 19, 4, 25, tzinfo=UTC),
    )

    assert first_result.ok is False
    assert "brain_search_canary_failed" in [issue.code for issue in first_result.issues]
    assert not any(command[:3] == ["launchctl", "kickstart", "-k"] for command in commands)

    second_result = run_health_check(
        config,
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py "
            "--interval 1 --backlog-batch 128 --enrich-limit 25\n"
        ),
        socket_request_fn=failed_canary,
        command_runner=lambda args: commands.append(args),
        now_fn=lambda: datetime(2026, 6, 19, 4, 30, tzinfo=UTC),
    )

    assert second_result.ok is False
    assert "brain_search_canary_failed" in [issue.code for issue in second_result.issues]
    assert any("com.brainlayer.brainbar-daemon" in " ".join(command) for command in commands)


def test_heal_min_consecutive_failures_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_HEAL_MIN_CONSECUTIVE_FAILURES", "3")

    assert HealthCheckConfig().heal_min_consecutive_failures == 3


def test_health_check_launchagent_runs_every_five_minutes_and_heals():
    plist_path = REPO_ROOT / "scripts/launchd/com.brainlayer.health-check.plist"
    plist = plistlib.loads(plist_path.read_bytes())

    assert plist["Label"] == "com.brainlayer.health-check"
    assert plist["StartInterval"] == 300
    assert plist["RunAtLoad"] is True
    assert plist["ProgramArguments"][:3] == [
        "__BRAINLAYER_ENV_RUN__",
        "__BRAINLAYER_BIN__",
        "health-check",
    ]
    assert "--heal" in plist["ProgramArguments"]
    assert "KeepAlive" not in plist


def test_health_check_bootstraps_absent_default_launchd_labels_instead_of_kickstart_only(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    _make_db(db_path, total=1, vector_rows=1)
    commands: list[list[str]] = []

    def command_runner(args: list[str]):
        commands.append(args)
        if args[:2] == ["launchctl", "print"]:
            return SimpleNamespace(returncode=113, stdout="", stderr="Could not find service")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=state_path,
            heal=True,
            heal_min_consecutive_failures=1,
        ),
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py "
            "--interval 1 --backlog-batch 4 --enrich-limit 5\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=command_runner,
        now_fn=lambda: datetime(2026, 6, 20, 10, 0, tzinfo=UTC),
    )

    issue_codes = [issue.code for issue in result.issues]
    assert {"watch_unloaded", "drain_unloaded", "health_check_unloaded"} <= set(issue_codes)
    assert ["launchctl", "enable", f"gui/{__import__('os').getuid()}/com.brainlayer.watch"] in commands
    assert ["launchctl", "enable", f"gui/{__import__('os').getuid()}/com.brainlayer.drain"] in commands
    assert ["launchctl", "enable", f"gui/{__import__('os').getuid()}/com.brainlayer.health-check"] in commands
    assert [
        "launchctl",
        "bootstrap",
        f"gui/{__import__('os').getuid()}",
        str(Path("~/Library/LaunchAgents/com.brainlayer.watch.plist").expanduser()),
    ] in commands
    assert not any(command[:3] == ["launchctl", "kickstart", "-k"] for command in commands)


def test_run_health_check_references_mode_d_detector_helpers():
    source = inspect.getsource(health_check.run_health_check)

    for helper_name in ("_pause_sentinel_state", "_source_recent", "_queue_stats", "_path_age_seconds"):
        assert helper_name in source


def test_health_check_reports_watcher_stalled_drain_no_progress_and_queue_backed_up(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    offsets_path = tmp_path / "offsets.json"
    watcher_health_path = tmp_path / "watcher-health.json"
    drain_health_path = tmp_path / "drain-health.json"
    queue_dir = tmp_path / "queue"
    source_dir = tmp_path / "source"
    queue_dir.mkdir()
    source_dir.mkdir()
    _make_db(db_path, total=1, vector_rows=1)

    now = datetime(2026, 6, 20, 10, 0, tzinfo=UTC)
    offsets_path.write_text("{}", encoding="utf-8")
    watcher_health_path.write_text(json.dumps({"poll_count": 5}), encoding="utf-8")
    drain_health_path.write_text(json.dumps({"drained_total": 10}), encoding="utf-8")
    queue_file = queue_dir / "watcher-test.jsonl"
    queue_file.write_text("{}\n", encoding="utf-8")
    source_file = source_dir / "session.jsonl"
    source_file.write_text("{}\n", encoding="utf-8")
    old_mtime = (now - timedelta(seconds=1000)).timestamp()
    recent_mtime = (now - timedelta(seconds=30)).timestamp()
    os.utime(offsets_path, (old_mtime, old_mtime))
    os.utime(watcher_health_path, (old_mtime, old_mtime))
    os.utime(drain_health_path, (old_mtime, old_mtime))
    os.utime(queue_file, (old_mtime, old_mtime))
    os.utime(source_file, (recent_mtime, recent_mtime))
    state_path.write_text(
        json.dumps({"watcher_poll_count": 5, "drain_drained_total": 10}),
        encoding="utf-8",
    )

    result = run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=state_path,
            offsets_path=offsets_path,
            watcher_health_path=watcher_health_path,
            drain_health_path=drain_health_path,
            queue_dir=queue_dir,
            source_jsonl_globs=[str(source_dir / "*.jsonl")],
            max_offsets_age_seconds=300,
            queue_auto_heal_count=1,
            queue_page_count=1,
        ),
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 4\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=lambda _args: SimpleNamespace(returncode=0, stdout="", stderr=""),
        now_fn=lambda: now,
    )

    issue_codes = [issue.code for issue in result.issues]
    assert "watcher_stalled" in issue_codes
    assert "drain_no_progress" in issue_codes
    assert "queue_backed_up" in issue_codes


def test_success_tick_clears_heal_breaker_state(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    _make_db(db_path, total=1, vector_rows=1)
    state_path.write_text(
        json.dumps(
            {
                "heal_failures": {"com.brainlayer.watch:watch_unloaded": 3},
                "heal_tripped": ["com.brainlayer.watch:watch_unloaded"],
                "missing_vectors": 0,
            }
        ),
        encoding="utf-8",
    )

    result = run_health_check(
        HealthCheckConfig(db_path=db_path, state_path=state_path),
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 4\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=lambda _args: SimpleNamespace(returncode=0, stdout="", stderr=""),
        now_fn=lambda: datetime(2026, 6, 20, 10, 0, tzinfo=UTC),
    )

    assert result.ok is True
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["heal_failures"] == {}
    assert saved["heal_tripped"] == []


def test_drain_launchagent_is_long_lived_keepalive_daemon():
    plist_path = REPO_ROOT / "scripts/launchd/com.brainlayer.drain.plist"
    plist = plistlib.loads(plist_path.read_bytes())

    assert plist["Label"] == "com.brainlayer.drain"
    assert plist["KeepAlive"] is True
    assert plist["RunAtLoad"] is True
    assert plist["ThrottleInterval"] == 10
    assert "--once" not in plist["ProgramArguments"]
    assert "WatchPaths" not in plist
    assert "QueueDirectories" not in plist
