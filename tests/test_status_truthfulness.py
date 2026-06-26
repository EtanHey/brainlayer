import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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


def test_status_json_surfaces_external_coverage_and_backlog_truth(tmp_path):
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
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
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["operational_green"] is False
    assert payload["queue_depth"] == 2
    assert payload["queue_depth_by_source"] == {"enrichment": 1, "watcher": 1}
    assert payload["pending_store_lines"] == 1
    assert payload["unembedded_chunks"] == 1
    assert payload["watcher_missing_providers"] == ["cursor-agent-transcripts"]
    assert payload["watcher_health"]["db_probe_failed"] is False
    assert payload["watcher_freshness"]["fresh"] is False
    assert payload["watcher_freshness"]["status"] == "stale"
    assert payload["vector_roundtrip"]["checked"] is False


def test_status_check_doctor_can_go_green_with_fresh_draining_queue_tail(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
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
    monkeypatch.setattr(cli, "_run_doctor_cli", lambda _config: _FakeDoctorResult(ok=True))

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
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["operational_green"] is True
    assert payload["queue_depth"] == 1
    assert payload["queue_health"]["green"] is True
    assert payload["queue_health"]["status"] == "draining"
    assert payload["vector_roundtrip"]["checked"] is True
    assert payload["vector_roundtrip"]["status"] == "passed"


def test_status_check_doctor_keeps_probe_failures_red(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
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
        "_run_doctor_cli",
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
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["operational_green"] is False
    assert payload["doctor_gate"]["ok"] is False
    assert payload["vector_roundtrip"]["checked"] is True
    assert payload["vector_roundtrip"]["status"] == "failed"
    assert payload["vector_roundtrip"]["issue"]["details"]["reason"] == "probe_not_vector_retrievable"


def test_status_check_doctor_accepts_visible_moving_queue_backlog(tmp_path, monkeypatch):
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
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
    monkeypatch.setattr(cli, "_run_doctor_cli", lambda _config: _FakeDoctorResult(ok=True, issues=[warning]))

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
