"""Audit: Claude cloud JSONL ingest contract (synthetic fixtures, scratch DB only).

Drives the real production boundary ``create_flush_callback`` (direct writer and
arbitrated queue + ``burn_drain_once``) and reads back stored rows. Assertions
pin OBSERVED behavior on main d6b82f04; they are not a statement of intent.
See docs/audits/2026-11-05-claude-cloud-jsonl-ingest-contract.md.
"""

import json

import pytest

from brainlayer.vector_store import VectorStore
from brainlayer.watcher_bridge import create_flush_callback

TEXT = (
    "Synthetic audit body: the deploy checklist was verified against the scratch database "
    "and every migration step completed without touching production state at all."
)
CLOUD_SESSION_ID = "cse_01SYNTHETICcloudSESSION0001"


def _entry(source_file, *, text=TEXT, role="assistant", session_id=CLOUD_SESSION_ID, cwd="/home/user/brainlayer", **extra):
    e = {
        "type": role,
        "sessionId": session_id,
        "cwd": cwd,
        "uuid": "00000000-0000-4000-8000-000000000001",
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
        "timestamp": "2026-11-05T07:00:00Z",
        "_source_file": str(source_file),
    }
    e.update(extra)
    return e


def _rows(db_path):
    store = VectorStore(db_path)
    try:
        cur = store.conn.cursor()
        cols = [r[1] for r in cur.execute("PRAGMA table_info(chunks)")]
        want = [c for c in ("id", "source_file", "project", "conversation_id", "content_type", "metadata", "seen_count", "content") if c in cols]
        return [dict(zip(want, r)) for r in cur.execute(f"SELECT {', '.join(want)} FROM chunks ORDER BY rowid")]
    finally:
        store.close()


@pytest.fixture
def env(tmp_path, monkeypatch):
    db_path = tmp_path / "scratch.db"
    VectorStore(db_path).close()
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    projects = tmp_path / ".claude" / "projects" / "-home-user-Gits-repoA"
    projects.mkdir(parents=True)
    return db_path, projects, tmp_path


def _write(path, entries):
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")


def test_probe_dump(env):
    db_path, projects, tmp_path = env
    flush = create_flush_callback(db_path, arbitrated=False)
    files = []
    for i in range(3):
        f = projects / f"cse_01SYNTH{i:04d}.jsonl"
        e = _entry(f, session_id=f"cse_01SYNTH{i:04d}", cwd="/home/user/brainlayer")
        _write(f, [e])
        files.append((f, e))
    flush([e for _, e in files])
    u = projects / "3679128a-f371-445f-82ba-b3946e2f20b6.jsonl"
    eu = _entry(u, session_id="3679128a-f371-445f-82ba-b3946e2f20b6")
    _write(u, [eu])
    flush([eu])
    tool = _entry(projects / "cse_01SYNTH0000.jsonl", role="user", text=None)
    tool["message"] = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_SYNTH1", "content": "OK"}]}
    tool["toolUseResult"] = {"stdout": "OK"}
    flush([tool])
    for r in _rows(db_path):
        print(json.dumps(r, default=str))
