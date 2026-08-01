import json
import sqlite3
from pathlib import Path

import pytest

import brainlayer.watcher_bridge as watcher_bridge
from brainlayer.agent_provenance import classify_provenance
from brainlayer.alarm import BrainLayerAlarm
from brainlayer.watcher_bridge import create_flush_callback


def _state_db(tmp_path: Path, runtime_rows: list[tuple[str, str, str]]) -> Path:
    path = tmp_path / "state.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE provider_session_runtime (
                thread_id TEXT PRIMARY KEY,
                provider_name TEXT NOT NULL,
                resume_cursor_json TEXT NOT NULL,
                runtime_payload_json TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO provider_session_runtime
                (thread_id, provider_name, resume_cursor_json, runtime_payload_json)
            VALUES (?, ?, ?, ?)
            """,
            [
                (thread_id, provider_name, resume_cursor_json, "{}")
                for thread_id, provider_name, resume_cursor_json in runtime_rows
            ],
        )
    return path


def _codex_source(tmp_path: Path, session_id: str) -> Path:
    return tmp_path / "home" / ".codex" / "sessions" / "2026" / "08" / f"rollout-2026-08-01-{session_id}.jsonl"


def test_plain_codex_working_on_t3layer_is_not_tagged_as_t3_app(tmp_path: Path) -> None:
    state_db = _state_db(
        tmp_path,
        [("another-t3-thread", "codex", json.dumps({"threadId": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}))],
    )

    decision = classify_provenance(
        str(_codex_source(tmp_path, "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")),
        t3_state_db=state_db,
    )

    assert (decision.provenance_tag, decision.search_policy) == ("codex-session", "KEEP")


def test_t3_app_initiated_codex_session_is_tagged_distinctly(tmp_path: Path) -> None:
    session_id = "019fb03f-0650-7f53-850b-921246951edc"
    state_db = _state_db(
        tmp_path,
        [("5ca576df-592c-406f-a8f1-7db9c56d36c9", "codex", json.dumps({"threadId": session_id}))],
    )

    decision = classify_provenance(str(_codex_source(tmp_path, session_id)), t3_state_db=state_db)

    assert (decision.provenance_tag, decision.search_policy) == ("t3-app-session", "KEEP")


def test_t3_app_linkage_outranks_recon_content_signature(tmp_path: Path) -> None:
    session_id = "019fb03f-0650-7f53-850b-921246951edc"
    state_db = _state_db(
        tmp_path,
        [("5ca576df-592c-406f-a8f1-7db9c56d36c9", "codex", json.dumps({"threadId": session_id}))],
    )

    decision = classify_provenance(
        str(_codex_source(tmp_path, session_id)),
        content="Task for brain-worker: mine the transcript.",
        t3_state_db=state_db,
    )

    assert decision.provenance_tag == "t3-app-session"


def test_new_t3_app_codex_ingestion_gets_distinct_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    state_db = _state_db(
        tmp_path,
        [("5ca576df-592c-406f-a8f1-7db9c56d36c9", "codex", json.dumps({"threadId": session_id}))],
    )
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(state_db))
    source_file = _codex_source(tmp_path, session_id)
    flush = create_flush_callback(db_path=tmp_path / "brainlayer.db", arbitrated=False)

    result = flush(
        [
            {
                "type": "user",
                "message": {
                    "content": [{"type": "text", "text": "T3 app sessions must retain their explicit origin."}]
                },
                "timestamp": "2026-08-01T12:00:00Z",
                "_source_file": str(source_file),
                "_line_end_offset": 100,
            }
        ]
    )

    assert result.inserted == 1
    with sqlite3.connect(tmp_path / "brainlayer.db") as conn:
        assert conn.execute("SELECT provenance_class FROM chunks").fetchone()[0] == "t3-app-session"


def test_watcher_loads_t3_session_ids_once_per_flush(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    state_db = _state_db(
        tmp_path,
        [("5ca576df-592c-406f-a8f1-7db9c56d36c9", "codex", json.dumps({"threadId": session_id}))],
    )
    calls: list[Path] = []

    def load_once(path: Path) -> set[str]:
        calls.append(path)
        return {session_id}

    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(state_db))
    monkeypatch.setattr(watcher_bridge, "t3_app_codex_session_ids", load_once)
    source_file = _codex_source(tmp_path, session_id)
    flush = create_flush_callback(db_path=tmp_path / "brainlayer.db", arbitrated=False)

    result = flush(
        [
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "First T3 watcher message."}]},
                "timestamp": "2026-08-01T12:00:00Z",
                "_source_file": str(source_file),
                "_line_end_offset": 100,
            },
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "Second T3 watcher message."}]},
                "timestamp": "2026-08-01T12:00:01Z",
                "_source_file": str(source_file),
                "_line_end_offset": 200,
            },
        ]
    )

    assert result.inserted == 2
    assert calls == [state_db]


def test_canonical_systems_codex_session_uses_runtime_cursor_linkage(tmp_path: Path) -> None:
    session_id = "019fb03f-0650-7f53-850b-921246951edc"
    state_db = _state_db(
        tmp_path,
        [("5ca576df-592c-406f-a8f1-7db9c56d36c9", "codex", json.dumps({"threadId": session_id}))],
    )

    decision = classify_provenance(str(_codex_source(tmp_path, session_id)), t3_state_db=state_db)

    assert decision.provenance_tag == "t3-app-session"
    assert "runtime cursor" in decision.reason


def test_missing_runtime_schema_raises_loud_alarm(tmp_path: Path) -> None:
    state_db = tmp_path / "state.sqlite"
    with sqlite3.connect(state_db) as conn:
        conn.execute("CREATE TABLE projection_thread_sessions (provider_session_id TEXT)")

    with pytest.raises(BrainLayerAlarm, match="t3_runtime_schema_drift"):
        classify_provenance(
            str(_codex_source(tmp_path, "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")),
            t3_state_db=state_db,
        )
