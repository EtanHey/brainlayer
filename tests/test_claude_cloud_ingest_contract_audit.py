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


def _entry(
    source_file, *, text=TEXT, role="assistant", session_id=CLOUD_SESSION_ID, cwd="/home/user/brainlayer", **extra
):
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
        want = [
            c
            for c in (
                "id",
                "source_file",
                "project",
                "conversation_id",
                "content_type",
                "metadata",
                "seen_count",
                "content",
            )
            if c in cols
        ]
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


def _proj(tmp_path, name):
    d = tmp_path / ".claude" / "projects" / f"-home-user-Gits-{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _flush_file(flush, f, entry):
    _write(f, [entry])
    flush([entry])


@pytest.mark.parametrize("arbitrated", [False, True])
def test_contract(env, arbitrated):
    db_path, projA, tmp_path = env
    projB = _proj(tmp_path, "repoB")
    flush = create_flush_callback(db_path, arbitrated=arbitrated)

    # (1)+(2): three cse_-prefixed files, identical text, repo-encoded path vs cwd.
    for i in range(3):
        f = projA / f"cse_01SYNTH{i:04d}.jsonl"
        _flush_file(flush, f, _entry(f, session_id=f"cse_01SYNTH{i:04d}", cwd="/home/user/brainlayer"))
    # (2) cross-project: a cse_ file in repoB with the same text -> same rt-cse_01SY-<hash> id.
    fb = projB / "cse_01SYNTHB000.jsonl"
    _flush_file(flush, fb, _entry(fb, session_id="cse_01SYNTHB000"))
    # contrast: UUID-named files with identical OTHER text in repoA and repoB.
    other = TEXT.replace("deploy checklist", "rollback checklist")
    for proj, uid in ((projA, "3679128a-f371-445f-82ba-b3946e2f20b6"), (projB, "9c1d2e3f-0000-4000-8000-00000000abcd")):
        f = proj / f"{uid}.jsonl"
        _flush_file(flush, f, _entry(f, text=other, session_id=uid))
    # (4) tool_result "OK" in the first cse session.
    t = _entry(projA / "cse_01SYNTH0000.jsonl", role="user")
    t["message"] = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_SYNTH1", "content": "OK"}],
    }
    t["toolUseResult"] = {"stdout": "OK"}
    flush([t])

    if arbitrated:
        from brainlayer.drain import burn_drain_once

        burn_drain_once(
            db_path=db_path,
            queue_dir=tmp_path / "queue",
            log_path=tmp_path / "drain.log",
            embed_fn=lambda _t: [0.0] * 1024,
        )

    rows = _rows(db_path)
    print("\n".join(json.dumps(r, default=str) for r in rows))
    cse = [r for r in rows if r["id"].startswith("rt-cse_01SY-")]
    uuid_rows = [r for r in rows if "rollback" in r["content"]]

    # (1) repo-encoded Claude project dir wins over entry cwd.
    assert cse and cse[0]["project"] == "repoA"
    # (2) four cse files (3 repoA + 1 repoB) collapse into ONE row; only first session's
    #     source_file/conversation_id/project survive, the rest is a seen_count bump.
    assert len(cse) == 1
    assert cse[0]["seen_count"] == 4
    assert cse[0]["source_file"].endswith("repoA/cse_01SYNTH0000.jsonl")
    assert cse[0]["conversation_id"] == "cse_01SYNTH0000"
    assert not any(r["project"] == "repoB" and "deploy checklist" in r["content"] for r in rows)
    # UUID-named files in different projects stay separate (dedupe is project-scoped).
    assert sorted(r["project"] for r in uuid_rows) == ["repoA", "repoB"]
    # (3) cloud sessionId preserved in conversation_id + metadata.session_id;
    #     the per-entry internal uuid is not stored as an identifier.
    meta = json.loads(cse[0]["metadata"])
    assert meta["session_id"] == "cse_01SYNTH0000"
    assert "00000000-0000-4000-8000-000000000001" not in json.dumps(rows)
    assert cse[0]["content_type"] == "assistant_text"
    # (4) short tool_result "OK" produces no row.
    assert not any(r["content"].strip() == "OK" for r in rows)
