import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import brainlayer.cli as cli
from brainlayer.cli import app


def _create_status_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            archived_at TEXT,
            superseded_by TEXT,
            aggregated_into TEXT
        );
        CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
        INSERT INTO chunks (id, content) VALUES ('missing-vector', 'needs vector');
        """
    )
    conn.commit()
    conn.close()


def _create_vectorized_status_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            archived_at TEXT,
            superseded_by TEXT,
            aggregated_into TEXT
        );
        CREATE TABLE chunk_vectors_rowids (id TEXT PRIMARY KEY);
        INSERT INTO chunks (id, content) VALUES ('vectorized', 'has vector');
        INSERT INTO chunk_vectors_rowids (id) VALUES ('vectorized');
        """
    )
    conn.commit()
    conn.close()


@dataclass
class _FakeDoctorResult:
    ok: bool
    roundtrip_latency_seconds: float | None = 0.0123
    exit_code: int = 0
    issues: list[SimpleNamespace] | None = None

    def to_dict(self):
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "roundtrip_latency_seconds": self.roundtrip_latency_seconds,
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "message": issue.message,
                    "details": getattr(issue, "details", {}),
                }
                for issue in (self.issues or [])
            ],
        }


def _fake_doctor_config():
    return SimpleNamespace(
        db_path="brainlayer.db",
        queue_dir="queue",
        watcher_health_path="watcher-health.json",
        drain_health_path="drain-health.json",
        roundtrip_timeout_seconds=2.5,
        roundtrip_probe_enabled=False,
        queue_movement_sample_seconds=0.75,
    )


def test_status_json_surfaces_external_coverage_and_backlog_truth(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    fallback_root = tmp_path / "gits"
    watcher_health = tmp_path / "watcher-health.json"
    drain_health = tmp_path / "drain-health.json"
    pending_stores = tmp_path / "pending-stores.jsonl"
    _create_status_db(db_path)
    queue_dir.mkdir()
    (queue_dir / "watcher-1.jsonl").write_text("{}\n")
    (queue_dir / "enrichment-1.jsonl").write_text("{}\n")
    pending_stores.write_text('{"content":"pending"}\n')
    watcher_health.write_text(
        json.dumps(
            {
                "providers": ["claude", "codex", "cursor", "gemini"],
                "alerting": False,
                "db_probe_failed": False,
                "updated_at": (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat(),
            }
        )
    )
    drain_health.write_text(json.dumps({"drained_total": 10, "drain_cycles": 2}))

    result = CliRunner().invoke(
        app,
        [
            "status",
            "--json",
            "--db",
            str(db_path),
            "--queue-dir",
            str(queue_dir),
            "--drain-health-path",
            str(drain_health),
            "--watcher-health-path",
            str(watcher_health),
            "--fallback-gits-root",
            str(fallback_root),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["operational_green"] is False
    assert payload["queue_depth"] == 2
    assert payload["queue_depth_by_source"] == {"enrichment": 1, "watcher": 1}
    assert payload["pending_store_lines"] == 1
    assert payload["fallback_replay"]["green"] is True
    assert payload["unembedded_chunks"] == 1
    assert payload["watcher_missing_providers"] == ["cursor-agent-transcripts"]
    assert payload["watcher_health"]["db_probe_failed"] is False
    assert payload["watcher_freshness"]["fresh"] is False
    assert payload["watcher_freshness"]["status"] == "stale"
    assert payload["vector_roundtrip"]["checked"] is False


def test_status_queue_depth_by_source_reads_hyphenated_jsonl_source(tmp_path):
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    (queue_dir / "fallback-replay-123.jsonl").write_text('{"source":"fallback-replay"}\n', encoding="utf-8")
    (queue_dir / "cursor-agent-transcripts-123.jsonl").write_text(
        '{"source":"cursor-agent-transcripts"}\n',
        encoding="utf-8",
    )

    assert cli._status_queue_depth_by_source(queue_dir) == {
        "cursor-agent-transcripts": 1,
        "fallback-replay": 1,
    }


def test_status_queue_depth_by_source_falls_back_for_invalid_utf8_jsonl(tmp_path):
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    (queue_dir / "fallback-replay-123.jsonl").write_bytes(b"\xff\xfe\n")

    assert cli._status_queue_depth_by_source(queue_dir) == {"fallback-replay": 1}


def test_status_unembedded_counter_uses_progress_timeout(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    db_path.write_bytes(b"")
    progress_calls = []

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def set_progress_handler(self, handler, n):
            progress_calls.append((handler, n))

        def execute(self, sql):
            if "PRAGMA" in sql:
                return self
            assert progress_calls
            raise sqlite3.OperationalError("interrupted")

    monkeypatch.setattr(cli.sqlite3, "connect", lambda *_args, **_kwargs: FakeConnection())

    assert cli._status_count_unembedded(db_path) is None
    assert progress_calls[0][1] == 1000
    assert progress_calls[-1] == (None, 0)


def test_status_check_doctor_can_go_green_with_fresh_draining_queue_tail(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    fallback_root = tmp_path / "gits"
    watcher_health = tmp_path / "watcher-health.json"
    drain_health = tmp_path / "drain-health.json"
    _create_vectorized_status_db(db_path)
    queue_dir.mkdir()
    (queue_dir / "watcher-1.jsonl").write_text("{}\n")
    now = datetime.now(timezone.utc)
    watcher_health.write_text(
        json.dumps(
            {
                "providers": ["claude", "codex", "cursor", "cursor-agent-transcripts", "gemini"],
                "alerting": False,
                "db_probe_failed": False,
                "updated_at": now.isoformat(),
            }
        )
    )
    drain_health.write_text(json.dumps({"drained_total": 10, "drain_cycles": 2, "updated_at": now.isoformat()}))

    def healthy_status_doctor(config):
        assert config.roundtrip_probe_enabled is False
        return _FakeDoctorResult(ok=True, roundtrip_latency_seconds=None)

    monkeypatch.setattr(cli, "_run_status_doctor_cli", healthy_status_doctor)

    result = CliRunner().invoke(
        app,
        [
            "status",
            "--json",
            "--check-doctor",
            "--db",
            str(db_path),
            "--queue-dir",
            str(queue_dir),
            "--drain-health-path",
            str(drain_health),
            "--watcher-health-path",
            str(watcher_health),
            "--fallback-gits-root",
            str(fallback_root),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["operational_green"] is True
    assert payload["queue_depth"] == 1
    assert payload["queue_health"]["green"] is True
    assert payload["queue_health"]["status"] == "draining"
    assert payload["vector_roundtrip"]["checked"] is False
    assert payload["vector_roundtrip"]["status"] == "skipped_by_status_doctor"


def test_status_json_surfaces_docs_local_fallback_replay_debt(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    fallback_root = tmp_path / "gits"
    watcher_health = tmp_path / "watcher-health.json"
    drain_health = tmp_path / "drain-health.json"
    _create_vectorized_status_db(db_path)
    queue_dir.mkdir()
    repo = fallback_root / "cmuxlayer"
    decision = repo / "docs.local" / "decisions" / "2026-06-27-correction-cmuxlayer-launcher-rollout.md"
    decision.parent.mkdir(parents=True)
    decision.write_text(
        "---\n"
        "intended_brain_store: true\n"
        "project: cmuxlayer\n"
        "importance: 8\n"
        "chunk_id:\n"
        "---\n"
        "cmuxlayer launcher rollout correction should be replayed.\n",
        encoding="utf-8",
    )
    now = datetime.now(timezone.utc)
    watcher_health.write_text(
        json.dumps(
            {
                "providers": ["claude", "codex", "cursor", "cursor-agent-transcripts", "gemini"],
                "alerting": False,
                "db_probe_failed": False,
                "updated_at": now.isoformat(),
            }
        )
    )
    drain_health.write_text(json.dumps({"drained_total": 10, "drain_cycles": 2, "updated_at": now.isoformat()}))
    monkeypatch.setattr(cli, "_run_status_doctor_cli", lambda _config: _FakeDoctorResult(ok=True))

    result = CliRunner().invoke(
        app,
        [
            "status",
            "--json",
            "--check-doctor",
            "--db",
            str(db_path),
            "--queue-dir",
            str(queue_dir),
            "--drain-health-path",
            str(drain_health),
            "--watcher-health-path",
            str(watcher_health),
            "--fallback-gits-root",
            str(fallback_root),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["operational_green"] is False
    assert payload["fallback_replay"]["green"] is False
    assert payload["fallback_replay"]["status"] == "debt"
    assert payload["fallback_replay"]["pending_count"] == 1
    assert payload["fallback_replay"]["legacy_count"] == 0
    assert payload["fallback_replay"]["pending_sample"] == [str(decision)]


def test_status_check_doctor_keeps_probe_failures_red(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    fallback_root = tmp_path / "gits"
    watcher_health = tmp_path / "watcher-health.json"
    drain_health = tmp_path / "drain-health.json"
    _create_vectorized_status_db(db_path)
    queue_dir.mkdir()
    now = datetime.now(timezone.utc)
    watcher_health.write_text(
        json.dumps(
            {
                "providers": ["claude", "codex", "cursor", "cursor-agent-transcripts", "gemini"],
                "alerting": False,
                "db_probe_failed": False,
                "updated_at": now.isoformat(),
            }
        )
    )
    drain_health.write_text(json.dumps({"drained_total": 10, "drain_cycles": 2, "updated_at": now.isoformat()}))
    issue = SimpleNamespace(
        code="roundtrip_vector_probe_failed",
        severity="fatal",
        message="probe failed",
        details={"reason": "probe_not_vector_retrievable"},
    )
    monkeypatch.setattr(
        cli,
        "_run_status_doctor_cli",
        lambda _config: _FakeDoctorResult(ok=False, exit_code=1, issues=[issue], roundtrip_latency_seconds=2.0),
    )

    result = CliRunner().invoke(
        app,
        [
            "status",
            "--json",
            "--check-doctor",
            "--db",
            str(db_path),
            "--queue-dir",
            str(queue_dir),
            "--drain-health-path",
            str(drain_health),
            "--watcher-health-path",
            str(watcher_health),
            "--fallback-gits-root",
            str(fallback_root),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["operational_green"] is False
    assert payload["doctor_gate"]["ok"] is False
    assert payload["vector_roundtrip"]["checked"] is True
    assert payload["vector_roundtrip"]["status"] == "failed"
    assert payload["vector_roundtrip"]["issue"]["details"]["reason"] == "probe_not_vector_retrievable"


def test_status_check_doctor_timeout_returns_structured_verdict(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    fallback_root = tmp_path / "gits"
    watcher_health = tmp_path / "watcher-health.json"
    drain_health = tmp_path / "drain-health.json"
    _create_vectorized_status_db(db_path)
    queue_dir.mkdir()
    now = datetime.now(timezone.utc)
    watcher_health.write_text(
        json.dumps(
            {
                "providers": ["claude", "codex", "cursor", "cursor-agent-transcripts", "gemini"],
                "alerting": False,
                "db_probe_failed": False,
                "updated_at": now.isoformat(),
            }
        )
    )
    drain_health.write_text(json.dumps({"drained_total": 10, "drain_cycles": 2, "updated_at": now.isoformat()}))
    captured_configs = []

    def timed_out_doctor(config):
        captured_configs.append(config)
        raise TimeoutError("doctor timed out after 0.01s")

    monkeypatch.setattr(cli, "_run_status_doctor_cli", timed_out_doctor, raising=False)

    result = CliRunner().invoke(
        app,
        [
            "status",
            "--json",
            "--check-doctor",
            "--db",
            str(db_path),
            "--queue-dir",
            str(queue_dir),
            "--drain-health-path",
            str(drain_health),
            "--watcher-health-path",
            str(watcher_health),
            "--fallback-gits-root",
            str(fallback_root),
        ],
    )

    assert result.exit_code == 0
    assert captured_configs
    assert captured_configs[0].roundtrip_probe_enabled is False
    assert captured_configs[0].queue_movement_sample_seconds > 0
    payload = json.loads(result.output)
    assert payload["operational_green"] is False
    assert payload["doctor_gate"]["checked"] is True
    assert payload["doctor_gate"]["ok"] is False
    assert payload["doctor_gate"]["issues"][0]["code"] == "status_doctor_timeout"
    assert payload["vector_roundtrip"]["checked"] is False
    assert payload["vector_roundtrip"]["status"] == "not_checked_status_doctor_error"
    assert payload["vector_roundtrip"]["blocked_by"]["code"] == "status_doctor_timeout"


def test_status_doctor_timeout_env_rejects_non_finite_values(monkeypatch):
    for raw_value in ("inf", "+inf", "nan"):
        monkeypatch.setenv("BRAINLAYER_STATUS_DOCTOR_TIMEOUT_SECONDS", raw_value)
        assert cli._status_doctor_timeout_seconds() == cli.DEFAULT_STATUS_DOCTOR_TIMEOUT_SECONDS


def test_status_doctor_gate_marks_roundtrip_skipped_without_latency():
    doctor_gate, vector_roundtrip = cli._status_doctor_gate(
        _FakeDoctorResult(ok=True, roundtrip_latency_seconds=None),
        None,
    )

    assert doctor_gate["ok"] is True
    assert vector_roundtrip["checked"] is False
    assert vector_roundtrip["status"] == "skipped_by_status_doctor"
    assert "brainlayer doctor --json" in vector_roundtrip["green_gate"]


def test_status_doctor_runner_execs_child_without_multiprocessing_spawn(tmp_path, monkeypatch):
    from brainlayer.doctor import DoctorConfig

    config = DoctorConfig(
        db_path=tmp_path / "brainlayer.db",
        queue_dir=tmp_path / "queue",
        watcher_health_path=tmp_path / "watcher-health.json",
        drain_health_path=tmp_path / "drain-health.json",
        roundtrip_timeout_seconds=2.5,
        roundtrip_probe_enabled=False,
        queue_movement_sample_seconds=0.75,
    )
    calls = {}

    def forbidden_context():
        raise AssertionError("status doctor must not use multiprocessing spawn from the CLI path")

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "exit_code": 0,
                    "roundtrip_latency_seconds": None,
                    "issues": [],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(cli, "_multiprocessing_context", forbidden_context, raising=False)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli._run_status_doctor_cli(config, timeout_seconds=0.01)

    assert result.ok is True
    assert calls["command"][:2] == [sys.executable, "-c"]
    assert "brainlayer.doctor" in calls["command"][2]
    assert calls["kwargs"]["timeout"] == 0.01
    env = calls["kwargs"]["env"]
    assert env["BRAINLAYER_STATUS_DOCTOR_DB"] == str(config.db_path)
    assert env["BRAINLAYER_STATUS_DOCTOR_QUEUE_DIR"] == str(config.queue_dir)
    assert env["BRAINLAYER_STATUS_DOCTOR_ROUNDTRIP_PROBE_ENABLED"] == "0"


def test_status_doctor_runner_maps_child_timeout(monkeypatch):
    def fake_run(command, **kwargs):
        raise cli.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    with pytest.raises(cli.StatusDoctorTimeoutError, match="doctor timed out after 0.01s"):
        cli._run_status_doctor_cli(_fake_doctor_config(), timeout_seconds=0.01)


def test_status_doctor_runner_reports_malformed_child_verdict(monkeypatch):
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=2, stdout="not-json\n", stderr="traceback")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="doctor returned a malformed verdict.*exit code 2.*traceback"):
        cli._run_status_doctor_cli(_fake_doctor_config(), timeout_seconds=0.01)


def test_status_check_doctor_accepts_visible_moving_queue_backlog(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    fallback_root = tmp_path / "gits"
    watcher_health = tmp_path / "watcher-health.json"
    drain_health = tmp_path / "drain-health.json"
    _create_vectorized_status_db(db_path)
    queue_dir.mkdir()
    for index in range(30):
        (queue_dir / f"enrichment-{index}.jsonl").write_text("{}\n")
    now = datetime.now(timezone.utc)
    watcher_health.write_text(
        json.dumps(
            {
                "providers": ["claude", "codex", "cursor", "cursor-agent-transcripts", "gemini"],
                "alerting": False,
                "db_probe_failed": False,
                "updated_at": now.isoformat(),
            }
        )
    )
    drain_health.write_text(json.dumps({"drained_total": 10, "drain_cycles": 2, "updated_at": now.isoformat()}))
    warning = SimpleNamespace(
        code="queue_backed_up_but_moving",
        severity="warning",
        message="queue is moving",
        details={"queue_count": 30, "drain_total": 11},
    )
    monkeypatch.setattr(cli, "_run_status_doctor_cli", lambda _config: _FakeDoctorResult(ok=True, issues=[warning]))

    result = CliRunner().invoke(
        app,
        [
            "status",
            "--json",
            "--check-doctor",
            "--db",
            str(db_path),
            "--queue-dir",
            str(queue_dir),
            "--drain-health-path",
            str(drain_health),
            "--watcher-health-path",
            str(watcher_health),
            "--fallback-gits-root",
            str(fallback_root),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["operational_green"] is True
    assert payload["queue_depth"] == 30
    assert payload["queue_health"]["green"] is True
    assert payload["queue_health"]["status"] == "moving_backlog"
    assert payload["queue_health"]["warning"] is True
    assert payload["queue_health"]["doctor_issue"]["code"] == "queue_backed_up_but_moving"
