from __future__ import annotations

import hashlib
import inspect
import json
import os
import plistlib
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import brainlayer.health_check as health_check
from brainlayer.health_check import HealthCheckConfig, run_health_check
from brainlayer.vector_store import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_default_live_paths(tmp_path_factory, monkeypatch):
    """Isolate HealthCheckConfig defaults that point at live developer state.

    Tests that build HealthCheckConfig without overriding every path otherwise
    read the developer's real ~/.brainlayer/queue and
    ~/.local/share/brainlayer/pause.sentinel. A populated live queue can make
    them spuriously report `queue_backed_up`, and a live pause sentinel suppresses
    `watcher_stalled`. We redirect the default readers to isolated paths, while
    tests that pass explicit paths are honored unchanged.
    """
    empty_queue = tmp_path_factory.mktemp("hc-queue")
    live_queue_default = Path("~/.brainlayer/queue").expanduser()
    real_queue_stats = health_check._queue_stats

    def _isolated_queue_stats(queue_dir, now):
        if queue_dir.expanduser() == live_queue_default:
            queue_dir = empty_queue
        return real_queue_stats(queue_dir, now)

    monkeypatch.setattr(health_check, "_queue_stats", _isolated_queue_stats)

    absent_pending_stores = tmp_path_factory.mktemp("hc-pending-stores") / "pending-stores.jsonl"
    live_pending_stores_default = Path("~/.local/share/brainlayer/pending-stores.jsonl").expanduser()
    real_pending_stores_count = health_check._pending_stores_count

    def _isolated_pending_stores_count(path):
        if path.expanduser() == live_pending_stores_default:
            path = absent_pending_stores
        return real_pending_stores_count(path)

    monkeypatch.setattr(health_check, "_pending_stores_count", _isolated_pending_stores_count)

    absent_sentinel = tmp_path_factory.mktemp("hc-sentinel") / "pause.sentinel"
    live_sentinel_default = Path("~/.local/share/brainlayer/pause.sentinel").expanduser()
    real_pause_state = health_check._pause_sentinel_state

    def _isolated_pause_state(config, now):
        if config.pause_sentinel_path.expanduser() == live_sentinel_default:
            from dataclasses import replace

            config = replace(config, pause_sentinel_path=absent_sentinel)
        return real_pause_state(config, now)

    monkeypatch.setattr(health_check, "_pause_sentinel_state", _isolated_pause_state)
    yield


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
                status TEXT DEFAULT 'active',
                enriched_at TEXT,
                enrich_status TEXT,
                char_count INTEGER
            );
            CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY, chunk_id INTEGER);
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


def _expected_writer_pidfile(pidfile_dir: Path, db_path: Path) -> Path:
    resolved_path = db_path.resolve()
    path_hash = hashlib.sha256(str(resolved_path).encode("utf-8")).hexdigest()[:16]
    return pidfile_dir / f"brainlayer-writer-{path_hash}-{resolved_path.name}.pid"


def _write_writer_pidfile(pidfile_dir: Path, db_path: Path, *, pid: int, start_time: str | None = None) -> Path:
    pidfile_dir.mkdir(parents=True, exist_ok=True)
    lines = [str(pid)]
    if start_time is not None:
        lines.append(f"start_time={start_time}")
    lines.append(f"db_path={db_path.resolve()}")
    pidfile = _expected_writer_pidfile(pidfile_dir, db_path)
    pidfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pidfile


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


def test_health_check_consumes_alerting_t3_health_snapshot(tmp_path):
    health_path = tmp_path / "t3-health.json"
    payload = {
        "alerting": True,
        "alert_reasons": ["schema_drift"],
        "failures": [{"code": "t3_schema_drift", "error": "missing text"}],
    }
    health_path.write_text(json.dumps(payload), encoding="utf-8")
    db_path = tmp_path / "brainlayer.db"
    _make_db(db_path, total=1, vector_rows=1)

    result = health_check.run_health_check(
        health_check.HealthCheckConfig(
            db_path=db_path,
            state_path=tmp_path / "state.json",
            t3_health_path=health_path,
            watcher_health_path=tmp_path / "watcher-health.json",
            drain_health_path=tmp_path / "drain-health.json",
            source_jsonl_globs=[],
            queue_dir=tmp_path / "queue",
            pending_stores_path=tmp_path / "pending-stores.jsonl",
        ),
        ps_output_fn=lambda: "",
        socket_request_fn=_ok_canary,
        command_runner=lambda _args: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    assert result.t3_health == payload
    issue = next(issue for issue in result.issues if issue.code == "t3_ingest_unhealthy")
    assert issue.severity == "critical"
    assert "schema_drift" in issue.message


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


def test_uninterruptible_launchd_process_backs_off_instead_of_kickstarting(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    _make_db(db_path, total=1, vector_rows=1)
    commands: list[list[str]] = []
    events: list[dict] = []
    monkeypatch.setattr(health_check, "_emit_heal_event", events.append)

    def command_runner(args: list[str]):
        commands.append(args)
        if args[:2] == ["launchctl", "print"] and "hotlane-brainbar" in args[-1]:
            return SimpleNamespace(returncode=0, stdout="pid = 123\n", stderr="")
        if args[:3] == ["ps", "-o", "state="]:
            return SimpleNamespace(returncode=0, stdout="UN\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=state_path,
            heal=True,
            heal_min_consecutive_failures=1,
            heal_circuit_breaker_limit=3,
        ),
        ps_output_fn=lambda: "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 0\n",
        socket_request_fn=_ok_canary,
        command_runner=command_runner,
        now_fn=lambda: datetime(2026, 7, 13, 9, 5, tzinfo=UTC),
    )

    assert not [command for command in commands if command[:3] == ["launchctl", "kickstart", "-k"]]
    assert "heal_backoff:com.brainlayer.hotlane-brainbar:hotlane_backlog_disabled:pid=123:state=UN" in result.actions
    assert any(
        event.get("_type") == "heal_backoff" and event.get("pid") == 123 and event.get("process_state") == "UN"
        for event in events
    )


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


def test_missing_embedding_query_scans_covering_id_indexes_before_chunk_payloads(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    _make_db(db_path, total=5, vector_rows=3)

    with sqlite3.connect(db_path) as conn:
        plan = conn.execute("EXPLAIN QUERY PLAN " + health_check.MISSING_EMBEDDINGS_SQL).fetchall()

    details = [str(row[-1]) for row in plan]
    assert any("chunks USING COVERING INDEX" in detail for detail in details)
    assert any("chunk_vectors_rowids USING COVERING INDEX" in detail for detail in details)
    assert "SCAN c" not in details


def test_interrupted_missing_embedding_count_writes_slow_state_and_returns_early(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    _make_db(db_path, total=1, vector_rows=1)
    canary_called = False

    def interrupted_count(_db_path, **_kwargs):
        raise sqlite3.OperationalError("interrupted")

    def canary(*_args):
        nonlocal canary_called
        canary_called = True
        return _ok_canary(*_args)

    monkeypatch.setattr(health_check, "count_missing_embeddings", interrupted_count)

    result = run_health_check(
        HealthCheckConfig(db_path=db_path, state_path=state_path, max_duration_seconds=45),
        ps_output_fn=lambda: "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 4\n",
        socket_request_fn=canary,
        command_runner=lambda _args: SimpleNamespace(returncode=0, stdout="", stderr=""),
        now_fn=lambda: datetime(2026, 7, 13, 9, 0, tzinfo=UTC),
    )

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert result.slow_check is True
    assert result.slow_check_stage == "missing_embeddings"
    assert saved["slow_check"] is True
    assert saved["slow_check_stage"] == "missing_embeddings"
    assert saved["ts"] == "2026-07-13T09:00:00+00:00"
    assert canary_called is False


def test_missing_embedding_count_interrupts_at_internal_deadline(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    _make_db(db_path, total=1, vector_rows=0)
    callbacks = []

    class DeadlineConnection:
        def execute(self, sql):
            if sql == "PRAGMA query_only = ON":
                return self
            raise sqlite3.OperationalError("interrupted")

        def set_progress_handler(self, callback, _instructions):
            callbacks.append(callback)

        def close(self):
            return None

    monkeypatch.setattr(health_check.sqlite3, "connect", lambda *_args, **_kwargs: DeadlineConnection())

    with pytest.raises(RuntimeError, match="deadline"):
        health_check.count_missing_embeddings(
            db_path,
            deadline_at=1.0,
            monotonic_fn=lambda: 2.0,
        )

    assert len(callbacks) == 1
    assert callbacks[0]() == 1


def test_source_recent_returns_at_deadline_when_glob_iteration_stalls(tmp_path, monkeypatch):
    import glob
    import threading
    import time

    release_glob = threading.Event()

    def stalled_glob(*_args, **_kwargs):
        release_glob.wait()
        yield str(tmp_path / "event.jsonl")

    monkeypatch.setattr(glob, "iglob", stalled_glob)
    outcome: dict[str, object] = {}

    def call_source_recent() -> None:
        try:
            health_check._source_recent(
                HealthCheckConfig(source_jsonl_globs=(str(tmp_path / "**" / "*.jsonl"),)),
                datetime(2026, 7, 13, 9, 0, tzinfo=UTC),
                60,
                deadline_at=time.monotonic() + 0.05,
            )
        except Exception as exc:
            outcome["error"] = exc

    caller = threading.Thread(target=call_source_recent)
    caller.start()
    caller.join(timeout=0.25)
    returned_before_release = not caller.is_alive()
    release_glob.set()
    caller.join(timeout=1)

    assert returned_before_release is True
    assert isinstance(outcome.get("error"), health_check.HealthCheckDeadlineExceeded)


def test_ps_snapshot_failure_does_not_heal_a_running_daemon_as_dead(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    _make_db(db_path, total=1, vector_rows=1)
    commands: list[list[str]] = []

    def command_runner(args):
        commands.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=state_path,
            heal=True,
            heal_min_consecutive_failures=1,
        ),
        ps_output_fn=lambda: None,
        socket_request_fn=_ok_canary,
        command_runner=command_runner,
        now_fn=lambda: datetime(2026, 7, 13, 9, 0, tzinfo=UTC),
    )

    issue_codes = {issue.code for issue in result.issues}
    assert "process_snapshot_failed" in issue_codes
    assert "hotlane_dead" not in issue_codes
    assert not any(
        args[:3] == ["launchctl", "kickstart", "-k"] and args[-1].endswith("/com.brainlayer.hotlane")
        for args in commands
    )


def test_default_ps_output_marks_timeout_as_unavailable(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise health_check.subprocess.TimeoutExpired(cmd=["ps"], timeout=5)

    monkeypatch.setattr(health_check.subprocess, "run", timeout)

    assert health_check._default_ps_output() is None


def test_lock_holder_wedge_flags_live_holder_when_drain_is_starved(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    drain_health_path = tmp_path / "drain-health.json"
    queue_dir = tmp_path / "queue"
    pidfile_dir = tmp_path / "pidfiles"
    queue_dir.mkdir()
    _make_db(db_path, total=1, vector_rows=1)
    drain_health_path.write_text(json.dumps({"drained_total": 10}), encoding="utf-8")
    (queue_dir / "blocked.jsonl").write_text("{}\n", encoding="utf-8")
    holder_pid = os.getpid()
    _write_writer_pidfile(pidfile_dir, db_path, pid=holder_pid, start_time="holder-start")
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    monkeypatch.setattr(VectorStore, "_pid_start_time", staticmethod(lambda _pid: "holder-start"))
    state_path.write_text(
        json.dumps(
            {
                "drain_drained_total": 10,
                "lock_holder_pid": holder_pid,
                "lock_holder_held_ticks": 1,
            }
        ),
        encoding="utf-8",
    )

    result = run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=state_path,
            drain_health_path=drain_health_path,
            queue_dir=queue_dir,
            max_stalled_ticks=2,
        ),
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 4\n"
            f"{holder_pid} /opt/homebrew/bin/brainlayer index\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=lambda _args: SimpleNamespace(returncode=0, stdout="", stderr=""),
        now_fn=lambda: datetime(2026, 7, 6, 9, 0, tzinfo=UTC),
    )

    payload = result.to_dict()
    assert payload["lock_holder"] == {
        "pid": holder_pid,
        "command": "/opt/homebrew/bin/brainlayer index",
        "db_path": str(db_path.resolve()),
        "held_ticks": 2,
    }
    issue_codes = [issue.code for issue in result.issues]
    assert "drain_no_progress" in issue_codes
    assert "lock_holder_wedge" in issue_codes
    assert any(str(holder_pid) in issue.message and "brainlayer index" in issue.message for issue in result.issues)


def _run_frozen_drain_liveness_scenario(
    tmp_path,
    monkeypatch,
    *,
    heartbeat_age: timedelta,
    pending_store_count: int,
    quota_blocked_enrichment: bool = False,
    heal: bool = False,
    command_runner=None,
):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    drain_health_path = tmp_path / "drain-health.json"
    queue_dir = tmp_path / "queue"
    pending_stores_path = tmp_path / "pending-stores.jsonl"
    pidfile_dir = tmp_path / "pidfiles"
    queue_dir.mkdir()
    _make_db(db_path, total=1, vector_rows=1)
    if quota_blocked_enrichment:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE chunks SET content = ?, char_count = ?",
                ("x" * 60, 60),
            )
        (db_path.parent / "enrich-daily-cost.json").write_text(
            json.dumps({"date": "2026-07-10", "spent_usd": 5.0}),
            encoding="utf-8",
        )
        monkeypatch.setenv("BRAINLAYER_ENRICH_DAILY_USD_CAP", "5.0")
    now = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    drain_health_path.write_text(
        json.dumps(
            {
                "drained_total": 10,
                "drain_cycles": 4,
                "updated_at": (now - heartbeat_age).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    pending_stores_path.write_text('{"content":"pending"}\n' * pending_store_count, encoding="utf-8")
    holder_pid = os.getpid()
    _write_writer_pidfile(pidfile_dir, db_path, pid=holder_pid, start_time="holder-start")
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    monkeypatch.setattr(VectorStore, "_pid_start_time", staticmethod(lambda _pid: "holder-start"))
    state_path.write_text(
        json.dumps(
            {
                "drain_drained_total": 10,
                "lock_holder_pid": holder_pid,
                "lock_holder_held_ticks": 1,
            }
        ),
        encoding="utf-8",
    )

    if command_runner is None:

        def command_runner(_args):
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=state_path,
            drain_health_path=drain_health_path,
            queue_dir=queue_dir,
            pending_stores_path=pending_stores_path,
            heal=heal,
            heal_min_consecutive_failures=1,
            max_stalled_ticks=2,
        ),
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 4\n"
            f"{holder_pid} /opt/homebrew/bin/brainlayer index\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=command_runner,
        now_fn=lambda: now,
    )
    return result, holder_pid


def test_frozen_drain_with_pending_stores_heals_live_index_lock_holder(tmp_path, monkeypatch):
    commands: list[list[str]] = []

    def command_runner(args: list[str]):
        commands.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result, _holder_pid = _run_frozen_drain_liveness_scenario(
        tmp_path,
        monkeypatch,
        heartbeat_age=timedelta(minutes=10),
        pending_store_count=2,
        heal=True,
        command_runner=command_runner,
    )

    issue_codes = [issue.code for issue in result.issues]
    assert "drain_liveness_stalled" in issue_codes
    assert "lock_holder_wedge" in issue_codes
    assert "kickstart:com.brainlayer.index" in result.actions
    kickstarts = [command for command in commands if command[:3] == ["launchctl", "kickstart", "-k"]]
    assert len(kickstarts) == 1
    assert "com.brainlayer.index" in " ".join(kickstarts[0])
    assert "com.brainlayer.drain" not in " ".join(kickstarts[0])


def test_frozen_drain_without_backlog_does_not_wedge_lock_holder(tmp_path, monkeypatch):
    result, _holder_pid = _run_frozen_drain_liveness_scenario(
        tmp_path,
        monkeypatch,
        heartbeat_age=timedelta(minutes=10),
        pending_store_count=0,
    )

    issue_codes = [issue.code for issue in result.issues]
    assert "drain_liveness_stalled" not in issue_codes
    assert "lock_holder_wedge" not in issue_codes


def test_frozen_drain_quota_blocker_does_not_wedge_lock_holder(tmp_path, monkeypatch):
    result, _holder_pid = _run_frozen_drain_liveness_scenario(
        tmp_path,
        monkeypatch,
        heartbeat_age=timedelta(minutes=10),
        pending_store_count=0,
        quota_blocked_enrichment=True,
    )

    issue_codes = [issue.code for issue in result.issues]
    assert "drain_liveness_quota_blocked" in issue_codes
    assert "drain_liveness_stalled" not in issue_codes
    assert "lock_holder_wedge" not in issue_codes


def test_fresh_drain_heartbeat_does_not_wedge_lock_holder(tmp_path, monkeypatch):
    result, _holder_pid = _run_frozen_drain_liveness_scenario(
        tmp_path,
        monkeypatch,
        heartbeat_age=timedelta(seconds=30),
        pending_store_count=2,
    )

    issue_codes = [issue.code for issue in result.issues]
    assert "drain_liveness_stalled" not in issue_codes
    assert "lock_holder_wedge" not in issue_codes


def test_pending_store_backlog_read_failure_is_reported(tmp_path, monkeypatch):
    def fail_pending_store_read(_path):
        raise PermissionError("pending stores unreadable")

    monkeypatch.setattr(health_check, "_pending_stores_count", fail_pending_store_read)

    result, _holder_pid = _run_frozen_drain_liveness_scenario(
        tmp_path,
        monkeypatch,
        heartbeat_age=timedelta(seconds=30),
        pending_store_count=1,
    )

    assert "pending_stores_count_failed" in [issue.code for issue in result.issues]


def test_enrichment_backlog_query_failure_is_reported(tmp_path, monkeypatch):
    def fail_enrichment_backlog(_path):
        raise sqlite3.OperationalError("enrichment backlog unavailable")

    monkeypatch.setattr(health_check, "_enrichment_backlog", fail_enrichment_backlog)

    result, _holder_pid = _run_frozen_drain_liveness_scenario(
        tmp_path,
        monkeypatch,
        heartbeat_age=timedelta(seconds=30),
        pending_store_count=0,
    )

    assert "enrichment_backlog_count_failed" in [issue.code for issue in result.issues]


@pytest.mark.parametrize(
    ("pid", "recorded_start_time", "current_start_time"),
    [
        (999999, "old-process", "old-process"),
        (os.getpid(), "old-process", "current-process"),
    ],
)
def test_lock_holder_stale_pidfile_is_ignored_without_false_wedge(
    tmp_path, monkeypatch, pid, recorded_start_time, current_start_time
):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    drain_health_path = tmp_path / "drain-health.json"
    queue_dir = tmp_path / "queue"
    pidfile_dir = tmp_path / "pidfiles"
    queue_dir.mkdir()
    _make_db(db_path, total=1, vector_rows=1)
    drain_health_path.write_text(json.dumps({"drained_total": 10}), encoding="utf-8")
    (queue_dir / "blocked.jsonl").write_text("{}\n", encoding="utf-8")
    _write_writer_pidfile(pidfile_dir, db_path, pid=pid, start_time=recorded_start_time)
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    monkeypatch.setattr(VectorStore, "_pid_start_time", staticmethod(lambda _pid: current_start_time))
    state_path.write_text(
        json.dumps(
            {
                "drain_drained_total": 10,
                "lock_holder_pid": pid,
                "lock_holder_held_ticks": 10,
            }
        ),
        encoding="utf-8",
    )

    result = run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=state_path,
            drain_health_path=drain_health_path,
            queue_dir=queue_dir,
            max_stalled_ticks=2,
        ),
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 4\n"
            f"{pid} /opt/homebrew/bin/brainlayer index\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=lambda _args: SimpleNamespace(returncode=0, stdout="", stderr=""),
        now_fn=lambda: datetime(2026, 7, 6, 9, 5, tzinfo=UTC),
    )

    assert result.to_dict()["lock_holder"] is None
    assert "lock_holder_wedge" not in [issue.code for issue in result.issues]
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["lock_holder_held_ticks"] == 0


def test_lock_holder_wedge_heal_targets_known_holder_label_and_respects_circuit_breaker(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    drain_health_path = tmp_path / "drain-health.json"
    queue_dir = tmp_path / "queue"
    pidfile_dir = tmp_path / "pidfiles"
    queue_dir.mkdir()
    _make_db(db_path, total=1, vector_rows=1)
    drain_health_path.write_text(json.dumps({"drained_total": 10}), encoding="utf-8")
    (queue_dir / "blocked.jsonl").write_text("{}\n", encoding="utf-8")
    holder_pid = os.getpid()
    _write_writer_pidfile(pidfile_dir, db_path, pid=holder_pid, start_time="holder-start")
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    monkeypatch.setattr(VectorStore, "_pid_start_time", staticmethod(lambda _pid: "holder-start"))
    notifications: list[tuple[str, str]] = []
    events: list[dict] = []
    monkeypatch.setattr(
        health_check, "_push_notification", lambda title, message: notifications.append((title, message))
    )
    monkeypatch.setattr(health_check, "_emit_heal_event", events.append)

    def ps_output() -> str:
        return (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 4\n"
            f"{holder_pid} /opt/homebrew/bin/brainlayer index\n"
        )

    commands: list[list[str]] = []

    def command_runner(args: list[str]):
        commands.append(args)
        if args[:2] == ["launchctl", "print"] and "com.brainlayer.index" in args[2]:
            return SimpleNamespace(returncode=0, stdout=f"pid = {holder_pid}\n", stderr="")
        if args[:2] == ["launchctl", "print"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    state_path.write_text(
        json.dumps(
            {
                "drain_drained_total": 10,
                "lock_holder_pid": holder_pid,
                "lock_holder_held_ticks": 1,
                "heal_failures": {"com.brainlayer.index:lock_holder_wedge": 1},
            }
        ),
        encoding="utf-8",
    )

    result = run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=state_path,
            drain_health_path=drain_health_path,
            queue_dir=queue_dir,
            heal=True,
            max_stalled_ticks=2,
            heal_min_consecutive_failures=2,
            heal_circuit_breaker_limit=3,
        ),
        ps_output_fn=ps_output,
        socket_request_fn=_ok_canary,
        command_runner=command_runner,
        now_fn=lambda: datetime(2026, 7, 6, 9, 10, tzinfo=UTC),
    )

    kickstarts = [command for command in commands if command[:3] == ["launchctl", "kickstart", "-k"]]
    assert len(kickstarts) == 1
    assert "com.brainlayer.index" in " ".join(kickstarts[0])
    assert "com.brainlayer.drain" not in " ".join(kickstarts[0])
    assert "kickstart:com.brainlayer.index" in result.actions
    assert any(str(holder_pid) in message and "brainlayer index" in message for _title, message in notifications)
    assert any(
        event.get("_type") == "heal" and event.get("lock_holder", {}).get("pid") == holder_pid for event in events
    )

    commands.clear()
    state_path.write_text(
        json.dumps(
            {
                "drain_drained_total": 10,
                "lock_holder_pid": holder_pid,
                "lock_holder_held_ticks": 1,
                "heal_failures": {"com.brainlayer.index:lock_holder_wedge": 3},
                "heal_tripped": ["com.brainlayer.index:lock_holder_wedge"],
            }
        ),
        encoding="utf-8",
    )

    run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=state_path,
            drain_health_path=drain_health_path,
            queue_dir=queue_dir,
            heal=True,
            max_stalled_ticks=2,
            heal_min_consecutive_failures=2,
            heal_circuit_breaker_limit=3,
        ),
        ps_output_fn=ps_output,
        socket_request_fn=_ok_canary,
        command_runner=command_runner,
        now_fn=lambda: datetime(2026, 7, 6, 9, 15, tzinfo=UTC),
    )

    assert not [command for command in commands if command[:3] == ["launchctl", "kickstart", "-k"]]


def test_lock_holder_without_starved_drain_reports_holder_but_no_wedge(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    drain_health_path = tmp_path / "drain-health.json"
    queue_dir = tmp_path / "queue"
    pidfile_dir = tmp_path / "pidfiles"
    queue_dir.mkdir()
    _make_db(db_path, total=1, vector_rows=1)
    drain_health_path.write_text(json.dumps({"drained_total": 11}), encoding="utf-8")
    holder_pid = os.getpid()
    _write_writer_pidfile(pidfile_dir, db_path, pid=holder_pid, start_time="holder-start")
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(pidfile_dir))
    monkeypatch.setattr(VectorStore, "_pid_start_time", staticmethod(lambda _pid: "holder-start"))
    state_path.write_text(json.dumps({"drain_drained_total": 10}), encoding="utf-8")

    result = run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=state_path,
            drain_health_path=drain_health_path,
            queue_dir=queue_dir,
            max_stalled_ticks=2,
        ),
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 4\n"
            f"{holder_pid} /opt/homebrew/bin/brainlayer index\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=lambda _args: SimpleNamespace(returncode=0, stdout="", stderr=""),
        now_fn=lambda: datetime(2026, 7, 6, 9, 20, tzinfo=UTC),
    )

    assert result.ok is True
    payload = result.to_dict()
    assert payload["lock_holder"]["pid"] == holder_pid
    assert payload["lock_holder"]["held_ticks"] == 0
    assert "lock_holder_wedge" not in [issue.code for issue in result.issues]


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
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 4 --enrich-limit 5\n"
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


def test_health_check_bootstraps_absent_enrichment_and_clears_tripped_after_success(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    state_path = tmp_path / "health-state.json"
    _make_db(db_path, total=1, vector_rows=1)
    state_path.write_text(
        json.dumps(
            {
                "heal_failures": {"com.brainlayer.enrichment:enrichment_unloaded": 3},
                "heal_tripped": ["com.brainlayer.enrichment:enrichment_unloaded"],
            }
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    enrichment_bootstrapped = False

    def command_runner(args: list[str]):
        nonlocal enrichment_bootstrapped
        commands.append(args)
        if args[:2] == ["launchctl", "print"] and "com.brainlayer.enrichment" in args[2]:
            return (
                SimpleNamespace(returncode=0, stdout="", stderr="")
                if enrichment_bootstrapped
                else SimpleNamespace(returncode=113, stdout="", stderr="Could not find service")
            )
        if args[:2] == ["launchctl", "bootstrap"] and str(args[-1]).endswith("com.brainlayer.enrichment.plist"):
            enrichment_bootstrapped = True
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:2] == ["launchctl", "print"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = run_health_check(
        HealthCheckConfig(
            db_path=db_path,
            state_path=state_path,
            heal=True,
            heal_min_consecutive_failures=1,
        ),
        ps_output_fn=lambda: (
            "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 4 --enrich-limit 5\n"
        ),
        socket_request_fn=_ok_canary,
        command_runner=command_runner,
        now_fn=lambda: datetime(2026, 6, 21, 10, 0, tzinfo=UTC),
    )

    assert "enrichment_unloaded" not in [issue.code for issue in result.issues]
    assert [
        "launchctl",
        "bootstrap",
        f"gui/{os.getuid()}",
        str(Path("~/Library/LaunchAgents/com.brainlayer.enrichment.plist").expanduser()),
    ] in commands
    assert "bootstrap:com.brainlayer.enrichment" in result.actions
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert "com.brainlayer.enrichment:enrichment_unloaded" not in saved["heal_tripped"]


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
        ps_output_fn=lambda: "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 4\n",
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
        ps_output_fn=lambda: "123 /usr/bin/python scripts/hotlane_brainbar_daemon.py --interval 1 --backlog-batch 4\n",
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
