import json
import sqlite3
from pathlib import Path

from scripts.retag_t3_app_provenance import retag_t3_app_chunks, rollback_t3_app_chunks


def _create_t3_state_db(path: Path, session_id: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE provider_session_runtime (
                thread_id TEXT PRIMARY KEY,
                provider_name TEXT NOT NULL,
                resume_cursor_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO provider_session_runtime VALUES (?, ?, ?)",
            ("t3-thread", "codex", json.dumps({"threadId": session_id})),
        )


def _create_brain_db(path: Path, app_session_id: str, plain_session_id: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, source_file TEXT, provenance_class TEXT)")
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?)",
            [
                ("app-1", f"/home/etan/.codex/sessions/rollout-{app_session_id}.jsonl", "codex-session"),
                ("app-2", f"/home/etan/.codex/sessions/rollout-{app_session_id}.jsonl", "RAW-ETAN-DIRECT"),
                ("plain", f"/home/etan/.codex/sessions/rollout-{plain_session_id}.jsonl", "codex-session"),
            ],
        )


def test_retag_only_explicitly_t3_linked_sessions_and_write_rollback_artifact(tmp_path: Path) -> None:
    app_session_id = "11111111-1111-4111-8111-111111111111"
    plain_session_id = "22222222-2222-4222-8222-222222222222"
    state_db = tmp_path / "state.sqlite"
    brain_db = tmp_path / "brainlayer.db"
    rollback_artifact = tmp_path / "rollback.jsonl"
    _create_t3_state_db(state_db, app_session_id)
    _create_brain_db(brain_db, app_session_id, plain_session_id)

    report = retag_t3_app_chunks(
        db_path=brain_db,
        state_db=state_db,
        apply=True,
        rollback_artifact=rollback_artifact,
        batch_size=1,
    )

    assert report == {"linked_sessions": 1, "candidate_chunks": 1, "retagged_chunks": 1}
    assert [json.loads(line) for line in rollback_artifact.read_text().splitlines()] == [
        {"id": "app-1", "provenance_class": "codex-session"},
    ]
    with sqlite3.connect(brain_db) as conn:
        assert dict(conn.execute("SELECT id, provenance_class FROM chunks")) == {
            "app-1": "t3-app-session",
            "app-2": "RAW-ETAN-DIRECT",
            "plain": "codex-session",
        }

    assert rollback_t3_app_chunks(db_path=brain_db, rollback_artifact=rollback_artifact, batch_size=1) == {
        "restored_chunks": 1
    }
    with sqlite3.connect(brain_db) as conn:
        assert dict(conn.execute("SELECT id, provenance_class FROM chunks")) == {
            "app-1": "codex-session",
            "app-2": "RAW-ETAN-DIRECT",
            "plain": "codex-session",
        }
