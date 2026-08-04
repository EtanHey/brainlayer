import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import brainlayer.watcher_bridge as watcher_bridge
from brainlayer.agent_provenance import classify_provenance
from brainlayer.alarm import BrainLayerAlarm
from brainlayer.watcher import JSONLWatcher, WatchRoot
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


def test_partially_installed_t3_does_not_stop_plain_claude_ingestion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_db = tmp_path / "state.sqlite"
    with sqlite3.connect(state_db) as conn:
        conn.execute("CREATE TABLE unrelated_bootstrap_state (id INTEGER PRIMARY KEY)")

    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(state_db))
    source_file = (
        tmp_path / "home" / ".claude" / "projects" / "-Users-etanheyman-Gits-brainlayer" / "plain-session.jsonl"
    )
    flush = create_flush_callback(db_path=tmp_path / "brainlayer.db", arbitrated=False)

    result = flush(
        [
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "Keep ordinary Claude ingestion available."}]},
                "timestamp": "2026-08-01T12:00:00Z",
                "_source_file": str(source_file),
                "_line_end_offset": 100,
            }
        ]
    )

    assert result.inserted == 1
    with sqlite3.connect(tmp_path / "brainlayer.db") as conn:
        assert conn.execute("SELECT provenance_class FROM chunks").fetchone()[0] == "direct-session"


def test_t3_linkage_alarm_defers_recon_codex_but_confirms_plain_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    state_db = tmp_path / "state.sqlite"
    queue_dir = tmp_path / "queue"
    with closing(sqlite3.connect(state_db)) as conn:
        conn.execute("CREATE TABLE unrelated_bootstrap_state (id INTEGER PRIMARY KEY)")
        conn.commit()

    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(state_db))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    codex_source = _codex_source(tmp_path, session_id)
    claude_source = (
        tmp_path / "home" / ".claude" / "projects" / "-Users-etanheyman-Gits-brainlayer" / "plain-session.jsonl"
    )
    codex_entry = {
        "type": "user",
        "message": {"content": [{"type": "text", "text": "Task for brain-worker: mine the transcript."}]},
        "timestamp": "2026-08-01T12:00:01Z",
        "_source_file": str(codex_source),
        "_line_end_offset": 555,
    }
    flush = create_flush_callback(arbitrated=True)

    def queued_events() -> list[dict[str, object]]:
        return [
            json.loads(line)
            for queue_file in queue_dir.glob("*.jsonl")
            for line in queue_file.read_text(encoding="utf-8").splitlines()
        ]

    degraded_result = flush(
        [
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "Keep ordinary Claude ingestion available."}]},
                "timestamp": "2026-08-01T12:00:00Z",
                "_source_file": str(claude_source),
                "_line_end_offset": 100,
            },
            codex_entry,
        ]
    )

    assert degraded_result.inserted == 1
    assert dict(degraded_result) == {str(claude_source): 100}
    assert [(event["source_file"], event["provenance_class"]) for event in queued_events()] == [
        (str(claude_source), "direct-session")
    ]

    with closing(sqlite3.connect(state_db)) as conn:
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
        conn.execute(
            """
            INSERT INTO provider_session_runtime
                (thread_id, provider_name, resume_cursor_json, runtime_payload_json)
            VALUES (?, ?, ?, ?)
            """,
            ("t3-thread", "codex", json.dumps({"threadId": session_id}), "{}"),
        )
        conn.commit()

    recovered_result = flush([codex_entry])

    assert recovered_result.inserted == 1
    assert dict(recovered_result) == {str(codex_source): 555}
    assert {event["source_file"]: event["provenance_class"] for event in queued_events()} == {
        str(claude_source): "direct-session",
        str(codex_source): "t3-app-session",
    }


def test_watcher_retries_deferred_t3_codex_without_advancing_across_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    state_db = tmp_path / "state.sqlite"
    queue_dir = tmp_path / "queue"
    registry_path = tmp_path / "offsets.json"
    with closing(sqlite3.connect(state_db)) as conn:
        conn.execute("CREATE TABLE unrelated_bootstrap_state (id INTEGER PRIMARY KEY)")
        conn.commit()

    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(state_db))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))

    codex_source = _codex_source(tmp_path, session_id)
    codex_source.parent.mkdir(parents=True)

    def sized_codex_record(text: str, byte_length: int) -> bytes:
        entry = {
            "role": "user",
            "content": text,
            "timestamp": "2026-08-01T12:00:01Z",
        }
        encoded = (json.dumps(entry) + "\n").encode()
        padding = byte_length - len(encoded)
        assert padding >= 0
        entry["content"] += "x" * padding
        encoded = (json.dumps(entry) + "\n").encode()
        assert len(encoded) == byte_length
        return encoded

    first_codex_record = sized_codex_record("Task for brain-worker: mine the transcript. ", 555)
    later_codex_record = sized_codex_record("Later T3 record must wait behind the deferred gap. ", 192)
    codex_source.write_bytes(first_codex_record)

    claude_source = (
        tmp_path / "home" / ".claude" / "projects" / "-Users-etanheyman-Gits-brainlayer" / "plain-session.jsonl"
    )
    claude_source.parent.mkdir(parents=True)
    claude_record = (
        json.dumps(
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "Keep ordinary Claude ingestion available."}]},
                "timestamp": "2026-08-01T12:00:00Z",
            }
        )
        + "\n"
    ).encode()
    claude_source.write_bytes(claude_record)

    watcher = JSONLWatcher(
        watch_roots=[
            WatchRoot("claude", claude_source.parents[1]),
            WatchRoot("codex", codex_source.parents[3]),
        ],
        registry_path=registry_path,
        on_flush=create_flush_callback(arbitrated=True),
        batch_size=2,
        flush_interval_ms=0,
    )

    def queued_events() -> list[dict[str, object]]:
        return [
            json.loads(line)
            for queue_file in queue_dir.glob("*.jsonl")
            for line in queue_file.read_text(encoding="utf-8").splitlines()
        ]

    assert watcher.poll_once() == 2
    assert [(event["source_file"], event["provenance_class"]) for event in queued_events()] == [
        (str(claude_source), "direct-session")
    ]
    assert watcher.registry.get(str(claude_source))[0] == len(claude_record)
    assert watcher.registry.get(str(codex_source))[0] == 0
    assert watcher._tailers[str(codex_source)].offset == 555
    assert [entry["_line_end_offset"] for entry in watcher.indexer._buffer] == [555]

    with codex_source.open("ab") as handle:
        handle.write(later_codex_record)

    assert watcher.poll_once() == 1
    assert watcher.registry.get(str(codex_source))[0] == 0
    assert watcher._tailers[str(codex_source)].offset == 747
    assert [entry["_line_end_offset"] for entry in watcher.indexer._buffer] == [555, 747]

    with closing(sqlite3.connect(state_db)) as conn:
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
        conn.execute(
            """
            INSERT INTO provider_session_runtime
                (thread_id, provider_name, resume_cursor_json, runtime_payload_json)
            VALUES (?, ?, ?, ?)
            """,
            ("t3-thread", "codex", json.dumps({"threadId": session_id}), "{}"),
        )
        conn.commit()

    assert watcher.poll_once() == 0
    codex_events = sorted(
        (event for event in queued_events() if event["source_file"] == str(codex_source)),
        key=lambda event: int(event["source_end_offset"]),
    )
    assert [(event["source_end_offset"], event["provenance_class"]) for event in codex_events] == [
        (555, "t3-app-session"),
        (747, "t3-app-session"),
    ]
    assert watcher.registry.get(str(codex_source))[0] == 747
    assert watcher.indexer._buffer == []


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
