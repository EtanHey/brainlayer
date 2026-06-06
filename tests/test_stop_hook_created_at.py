"""Stop hook must stamp created_at on realtime chunk inserts (the #455 fleet-wide NULL source)."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / "hooks" / "brainbar-stop-index.py"


def _mk_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE chunks (
            id TEXT PRIMARY KEY, content TEXT, metadata TEXT, source_file TEXT,
            source TEXT, project TEXT, content_type TEXT, char_count INTEGER,
            conversation_id TEXT, importance REAL, created_at TEXT)"""
    )
    conn.commit()
    conn.close()


def test_stop_hook_stamps_created_at(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _mk_db(db)
    payload = {
        "session_id": "deadbeef-0000-0000-0000-000000000000",
        "last_assistant_message": "x" * 64,
        "cwd": str(tmp_path),
        "transcript_path": "t.jsonl",
    }
    env = {"BRAINLAYER_DB": str(db), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, str(HOOK)], input=json.dumps(payload), text=True,
        capture_output=True, env=env, timeout=15,
    )
    assert r.returncode == 0, r.stderr
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT created_at FROM chunks").fetchone()
    conn.close()
    assert row is not None, "no chunk inserted"
    assert row[0], "created_at is NULL — the regression this test guards"
    assert "T" in row[0] and row[0].endswith(("Z", "+00:00")), f"non-canonical stamp: {row[0]}"
