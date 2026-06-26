import json
import sqlite3
from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

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
