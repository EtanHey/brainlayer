"""Antigravity (agy), the current Gemini agent, reaches the index under the source name "gemini".

agy exports each conversation's steps as JSONL at
`~/.gemini/antigravity-cli/brain/<conversation>/.system_generated/logs/transcript_full.jsonl`
(`transcript.jsonl` beside it holds the same steps with long fields truncated). Fixtures are synthetic,
in the shape measured on 500 real transcripts (2026-11-13): step_index, source, type
(USER_INPUT | PLANNER_RESPONSE | GENERIC | ERROR_MESSAGE), status, created_at, content.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import ExitStack, closing
from pathlib import Path

import apsw
import pytest

from brainlayer.agent_provenance import classify_provenance
from brainlayer.watcher import JSONLWatcher, default_watch_roots
from brainlayer.watcher_bridge import create_flush_callback

CONVERSATION = "0115a286-c627-4c2b-9dd2-d9550bfe78b4"


@pytest.fixture
def resources():
    with ExitStack() as stack:
        yield stack


@pytest.fixture(autouse=True)
def _scratch_env(tmp_path, monkeypatch, resources):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "db" / "brainlayer.db"))
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(tmp_path / "no-t3-state.db"))
    monkeypatch.setenv("BRAINLAYER_WATCHER_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    monkeypatch.delenv("AXIOM_TOKEN", raising=False)


def _text(marker: str) -> str:
    return f"agy turn {marker}: please explain why the nightly backup verifies the staged archive digest first."


def _step(index: int, kind: str, marker: str) -> dict:
    step = {"step_index": index, "source": "MODEL", "type": kind, "status": "DONE"}
    step["created_at"] = "2026-11-13T09:00:00Z"
    if kind == "USER_INPUT":
        step["source"] = "USER_EXPLICIT"
        step["content"] = (
            f"<USER_REQUEST>\n{_text(marker)}\n</USER_REQUEST>\n"
            "<ADDITIONAL_METADATA>\nmetaonly-xyz local time\n</ADDITIONAL_METADATA>\n"
            "<USER_SETTINGS_CHANGE>\nsettingsonly-xyz\n</USER_SETTINGS_CHANGE>"
        )
    else:
        step["content"] = _text(marker)
    return step


def _transcript(tmp_path: Path, conversation: str = CONVERSATION, name: str = "transcript_full.jsonl") -> Path:
    brain = tmp_path / "home" / ".gemini" / "antigravity-cli" / "brain"
    return brain / conversation / ".system_generated" / "logs" / name


def _summarise(path: Path, agent: str = "") -> None:
    """agy's own summary row for the transcript's conversation (`agent_name` is `agy --agent <name>`)."""
    store = path.parents[4] / "conversation_summaries.db"
    store.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(store)) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conversation_summaries (conversation_id TEXT PRIMARY KEY, agent_name TEXT)"
        )
        conn.execute("INSERT OR REPLACE INTO conversation_summaries VALUES (?, ?)", (path.parents[2].name, agent))


def _write(path: Path, steps: list[dict], mode: str = "w", agent: str | None = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as handle:
        handle.write("".join(json.dumps(step) + "\n" for step in steps))
    if agent is not None:
        _summarise(path, agent)


def _watcher(tmp_path: Path, db: Path, stack: ExitStack) -> JSONLWatcher:
    callback = create_flush_callback(db_path=db, arbitrated=False)
    stack.callback(callback.close)
    watcher = JSONLWatcher(
        watch_roots=default_watch_roots(tmp_path / "home"),
        registry_path=tmp_path / "offsets.json",
        on_flush=callback,
        db_path=db,
        batch_size=1,
        flush_interval_ms=0,
        registry_flush_interval_s=0,
    )
    stack.callback(watcher.stop)
    return watcher


def _poll(watcher: JSONLWatcher) -> None:
    watcher.poll_once()
    watcher.indexer.flush()


def _rows(db: Path, needle: str) -> list[tuple]:
    # Read through apsw, the writer's own SQLite: a second SQLite library in the same process (the
    # stdlib sqlite3 module) can read a stale WAL view of writes another library just committed.
    if not db.exists():
        return []
    conn = apsw.Connection(str(db))
    try:
        return list(
            conn.cursor().execute(
                "SELECT conversation_id, content, metadata FROM chunks WHERE content LIKE ?", (f"%{needle}%",)
            )
        )
    finally:
        conn.close()


def test_an_agy_conversation_is_indexed_as_gemini_with_its_own_conversation_id(tmp_path, resources):
    db = tmp_path / "db" / "brainlayer.db"
    path = _transcript(tmp_path)
    _write(path, [_step(0, "USER_INPUT", "u1"), _step(1, "PLANNER_RESPONSE", "a1")])

    watcher = _watcher(tmp_path, db, resources)
    _poll(watcher)

    assert watcher.provider_for_file(str(path)) == "gemini"
    for marker in ("u1", "a1"):
        rows = _rows(db, f"turn {marker}:")
        assert len(rows) == 1, marker
        conversation_id, _content, metadata = rows[0]
        meta = json.loads(metadata)
        assert conversation_id == CONVERSATION
        assert (meta["provenance_tag"], meta["source_class"]) == ("gemini-session", "cli-agent")
    # Only the user's request is indexed, not the metadata and settings agy wraps around it.
    assert _rows(db, "metaonly-xyz") == [] and _rows(db, "settingsonly-xyz") == []


def test_tool_and_error_steps_and_the_truncated_twin_are_not_indexed(tmp_path, resources):
    db = tmp_path / "db" / "brainlayer.db"
    steps = [_step(0, "USER_INPUT", "u1"), _step(1, "GENERIC", "g1"), _step(2, "ERROR_MESSAGE", "e1")]
    _write(_transcript(tmp_path), steps)
    _write(_transcript(tmp_path, name="transcript.jsonl"), steps)

    _poll(_watcher(tmp_path, db, resources))

    assert _rows(db, "turn g1:") == [] and _rows(db, "turn e1:") == []
    assert len(_rows(db, "turn u1:")) == 1


def test_an_appended_step_is_indexed_without_duplicating_earlier_ones(tmp_path, resources):
    db = tmp_path / "db" / "brainlayer.db"
    path = _transcript(tmp_path)
    _write(path, [_step(0, "USER_INPUT", "u1"), _step(1, "PLANNER_RESPONSE", "a1")])
    watcher = _watcher(tmp_path, db, resources)
    _poll(watcher)

    _write(path, [_step(2, "USER_INPUT", "u2"), _step(3, "PLANNER_RESPONSE", "a2")], mode="a")
    _poll(watcher)
    _poll(_watcher(tmp_path, db, resources))  # a restarted watcher resumes from its checkpoint

    for marker in ("u1", "a1", "u2", "a2"):
        assert len(_rows(db, f"turn {marker}:")) == 1, marker


def test_every_agy_conversation_keeps_its_own_conversation_id(tmp_path, resources):
    """Every agy file is named transcript_full.jsonl; the conversation dir, not the stem, names the session."""
    db = tmp_path / "db" / "brainlayer.db"
    other = "e1ea897a-32ba-42e2-b7af-e04ebe76a961"
    _write(_transcript(tmp_path), [_step(0, "USER_INPUT", "c1")])
    _write(_transcript(tmp_path, other), [_step(0, "USER_INPUT", "c2")])

    _poll(_watcher(tmp_path, db, resources))

    assert [row[0] for row in _rows(db, "turn c1:")] == [CONVERSATION]
    assert [row[0] for row in _rows(db, "turn c2:")] == [other]


def test_a_malformed_step_is_quarantined_and_counted_and_later_steps_still_land(tmp_path, resources):
    db = tmp_path / "db" / "brainlayer.db"
    path = _transcript(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_step(0, "USER_INPUT", "u1")) + "\n" + '{"step_index": 1, "type": \n')
    _summarise(path)
    _write(path, [_step(2, "PLANNER_RESPONSE", "a2")], mode="a")

    watcher = _watcher(tmp_path, db, resources)
    _poll(watcher)
    _poll(watcher)

    assert watcher._quarantined_record_count_total == 1
    assert len(_rows(db, "turn u1:")) == 1 and len(_rows(db, "turn a2:")) == 1


def test_agy_transcripts_are_tagged_gemini_session(tmp_path, resources):
    path = _transcript(tmp_path)
    _write(path, [], agent="")
    decision = classify_provenance(str(path))
    assert (decision.provenance_tag, decision.source_class) == ("gemini-session", "cli-agent")


def _class_of(db: Path, marker: str) -> tuple[str, str, str]:
    (row,) = _rows(db, f"turn {marker}:")
    meta = json.loads(row[2])
    return meta["provenance_tag"], meta["source_class"], meta["provenance_effective_visibility"]


def test_each_agy_class_lands_by_agys_own_agent_attribution(tmp_path, resources):
    db = tmp_path / "db" / "brainlayer.db"
    interactive, gatherer, reader = (
        "11111111-0000-4000-8000-000000000001",
        "22222222-0000-4000-8000-000000000002",
        "33333333-0000-4000-8000-000000000003",
    )
    _write(_transcript(tmp_path, interactive), [_step(0, "PLANNER_RESPONSE", "own1")], agent="")
    _write(_transcript(tmp_path, gatherer), [_step(0, "PLANNER_RESPONSE", "gath1")], agent="gatherer")
    _write(_transcript(tmp_path, reader), [_step(0, "PLANNER_RESPONSE", "read1")], agent="brain-worker")

    _poll(_watcher(tmp_path, db, resources))

    assert _class_of(db, "own1") == ("gemini-session", "cli-agent", "default")
    assert _class_of(db, "gath1") == ("gemini-subagent", "subagent", "default")
    assert _rows(db, "turn read1:") == []  # a memory reader is kept on disk but never indexed


def test_a_conversation_agy_has_not_summarised_yet_waits_for_its_attribution(tmp_path, resources):
    db = tmp_path / "db" / "brainlayer.db"
    path = _transcript(tmp_path)
    _write(path, [_step(0, "PLANNER_RESPONSE", "wait1")], agent=None)
    _summarise(_transcript(tmp_path, "44444444-0000-4000-8000-000000000004"))  # the store exists, this row does not

    watcher = _watcher(tmp_path, db, resources)
    _poll(watcher)
    assert _rows(db, "turn wait1:") == []
    assert watcher.registry.get(str(path))[0] == 0  # nothing confirmed while it waits

    _summarise(path, "gatherer")
    _poll(watcher)
    assert _class_of(db, "wait1") == ("gemini-subagent", "subagent", "default")


# --- r3: fail closed for memory readers (Codex r1 on #1786) ---------------------------------------


def _age(path: Path, seconds: float = 3600) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


def test_an_unreadable_attribution_db_holds_the_conversation_and_says_so(tmp_path, caplog, resources):
    db = tmp_path / "db" / "brainlayer.db"
    path = _transcript(tmp_path)
    _write(path, [_step(0, "PLANNER_RESPONSE", "unr1")], agent=None)
    (path.parents[4] / "conversation_summaries.db").write_bytes(b"not a sqlite database at all" * 64)
    _age(path)

    watcher = _watcher(tmp_path, db, resources)
    with caplog.at_level(logging.WARNING):
        _poll(watcher)
        _poll(watcher)

    assert _rows(db, "turn unr1:") == []
    assert watcher.registry.get(str(path))[0] == 0
    assert "agy attribution unreadable" in caplog.text


def test_a_failed_attribution_read_is_retried_not_cached(tmp_path, monkeypatch, resources):
    db = tmp_path / "db" / "brainlayer.db"
    path = _transcript(tmp_path)
    _write(path, [_step(0, "PLANNER_RESPONSE", "lock1")], agent="brain-worker")
    _age(path)
    real_connect, failures = sqlite3.connect, []

    def locked_once(target, *args, **kwargs):
        if "conversation_summaries.db" in str(target) and not failures:
            failures.append(target)
            raise sqlite3.OperationalError("database is locked")
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", locked_once)
    watcher = _watcher(tmp_path, db, resources)
    _poll(watcher)
    _poll(watcher)

    assert failures  # the lock was hit, and the next read saw the memory-reader row
    assert _rows(db, "turn lock1:") == []


def test_the_hold_back_counts_from_first_sight_not_the_latest_write_and_survives_restart(tmp_path, monkeypatch, resources):
    import brainlayer.agy_attribution as agy_attribution

    db = tmp_path / "db" / "brainlayer.db"
    path = _transcript(tmp_path)
    _write(path, [_step(0, "PLANNER_RESPONSE", "dl1")], agent=None)
    _summarise(_transcript(tmp_path, "44444444-0000-4000-8000-000000000004"))  # readable store, no row for ours
    clock = [1_000_000.0]
    monkeypatch.setattr(agy_attribution, "_now", lambda: clock[0])

    _poll(_watcher(tmp_path, db, resources))
    assert _rows(db, "turn dl1:") == []

    clock[0] += 300
    _write(path, [_step(1, "PLANNER_RESPONSE", "dl2")], mode="a", agent=None)  # an append must not reset it
    _poll(_watcher(tmp_path, db, resources))  # restarted watcher and flush callback
    assert _rows(db, "turn dl1:") == []

    clock[0] += 301
    _write(path, [_step(2, "PLANNER_RESPONSE", "dl3")], mode="a", agent=None)
    _poll(_watcher(tmp_path, db, resources))
    for marker in ("dl1", "dl2", "dl3"):
        assert _class_of(db, marker) == ("gemini-session", "cli-agent", "default"), marker


def test_a_late_memory_reader_row_stops_further_indexing_and_is_recorded(tmp_path, monkeypatch, resources):
    import brainlayer.agy_attribution as agy_attribution

    db = tmp_path / "db" / "brainlayer.db"
    path = _transcript(tmp_path)
    _write(path, [_step(0, "PLANNER_RESPONSE", "late1")], agent=None)
    _summarise(_transcript(tmp_path, "44444444-0000-4000-8000-000000000004"))
    clock = [1_000_000.0]
    monkeypatch.setattr(agy_attribution, "_now", lambda: clock[0])
    watcher = _watcher(tmp_path, db, resources)
    _poll(watcher)
    clock[0] += 601
    _poll(watcher)
    assert len(_rows(db, "turn late1:")) == 1  # indexed after the deadline, as no agent

    _summarise(path, "brain-worker")
    _write(path, [_step(1, "PLANNER_RESPONSE", "late2")], mode="a", agent=None)
    _poll(watcher)

    assert _rows(db, "turn late2:") == []
    state = json.loads((db.parent / "agy-attribution.json").read_text())
    assert state["late_memory_readers"][CONVERSATION]["agent"] == "brain-worker"


def test_a_literal_closing_tag_inside_the_request_survives(tmp_path, resources):
    db = tmp_path / "db" / "brainlayer.db"
    step = _step(0, "USER_INPUT", "tag1")
    step["content"] = (
        "<USER_REQUEST>\nExplain what </USER_REQUEST> means and keep tailmarker-tag1 after it\n</USER_REQUEST>\n"
        "<ADDITIONAL_METADATA>\nmetaonly-xyz\n</ADDITIONAL_METADATA>"
    )
    _write(_transcript(tmp_path), [step])

    _poll(_watcher(tmp_path, db, resources))

    (row,) = _rows(db, "tailmarker-tag1")
    assert "</USER_REQUEST> means" in row[1] and "metaonly-xyz" not in row[1]


def test_the_legacy_gemini_sessions_root_is_not_watched(tmp_path, resources):
    db = tmp_path / "db" / "brainlayer.db"
    legacy = tmp_path / "home" / ".gemini" / "sessions" / "legacy.jsonl"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps({"role": "user", "content": _text("leg1"), "timestamp": "2026-11-13T09:00:00Z"}) + "\n"
    )

    watcher = _watcher(tmp_path, db, resources)
    _poll(watcher)

    assert _rows(db, "turn leg1:") == []
    assert [r.resolved_path for r in default_watch_roots(tmp_path / "home") if r.provider == "gemini"] == [
        tmp_path / "home" / ".gemini" / "antigravity-cli" / "brain"
    ]


# --- r4 (lead check of Codex r2 on #1786) ---------------------------------------------------------


def test_the_deadline_counts_from_an_unreadable_first_sighting(tmp_path, monkeypatch, resources):
    import brainlayer.agy_attribution as agy_attribution

    db = tmp_path / "db" / "brainlayer.db"
    path = _transcript(tmp_path)
    _write(path, [_step(0, "PLANNER_RESPONSE", "first1")], agent=None)
    store = path.parents[4] / "conversation_summaries.db"
    store.write_bytes(b"not a sqlite database at all" * 64)
    clock = [1_000_000.0]
    monkeypatch.setattr(agy_attribution, "_now", lambda: clock[0])
    watcher = _watcher(tmp_path, db, resources)
    _poll(watcher)
    assert _rows(db, "turn first1:") == []

    store.unlink()  # agy's summaries DB is readable again 20 minutes later, still without this row
    _summarise(_transcript(tmp_path, "44444444-0000-4000-8000-000000000004"))
    clock[0] += 1200
    _poll(watcher)

    assert _class_of(db, "first1") == ("gemini-session", "cli-agent", "default")


def test_a_literal_closing_tag_inside_a_later_metadata_block_is_not_the_boundary(tmp_path, resources):
    db = tmp_path / "db" / "brainlayer.db"
    step = _step(0, "USER_INPUT", "meta1")
    step["content"] = (
        "<USER_REQUEST>\nkeep reqmarker-meta1 only\n</USER_REQUEST>\n"
        "<ADDITIONAL_METADATA>\nmetaonly-xyz quoting </USER_REQUEST> here\n</ADDITIONAL_METADATA>\n"
        "<USER_SETTINGS_CHANGE>\nsettingsonly-xyz\n</USER_SETTINGS_CHANGE>"
    )
    _write(_transcript(tmp_path), [step])

    _poll(_watcher(tmp_path, db, resources))

    (row,) = _rows(db, "reqmarker-meta1")
    assert "metaonly-xyz" not in row[1] and "</USER_REQUEST>" not in row[1]


def test_attribution_state_drops_conversations_whose_transcripts_are_gone(tmp_path, monkeypatch, resources):
    import brainlayer.agy_attribution as agy_attribution

    monkeypatch.setattr(agy_attribution, "_now", lambda: 1_000_000.0)
    _summarise(_transcript(tmp_path, "44444444-0000-4000-8000-000000000004"))  # readable, no rows for these
    gate = agy_attribution.AgyAttributionGate(tmp_path / "db" / "agy-attribution.json")
    paths = [_transcript(tmp_path, f"{i:08d}-0000-4000-8000-000000000000") for i in range(1000)]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        assert gate.resolve(path) == (False, None)
    assert len(gate.state["first_seen"]) == 1000

    for path in paths:
        path.unlink()
    (gate.save if hasattr(gate, "save") else gate._save)()

    state = json.loads((tmp_path / "db" / "agy-attribution.json").read_text())
    assert state == {"fallback_indexed": {}, "first_seen": {}, "late_memory_readers": {}}


def test_attribution_state_ages_out_old_entries(tmp_path, monkeypatch, resources):
    import brainlayer.agy_attribution as agy_attribution

    clock = [1_000_000.0]
    monkeypatch.setattr(agy_attribution, "_now", lambda: clock[0])
    _summarise(_transcript(tmp_path, "44444444-0000-4000-8000-000000000004"))
    gate = agy_attribution.AgyAttributionGate(tmp_path / "db" / "agy-attribution.json")
    path = _transcript(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    gate.resolve(path)
    clock[0] += 601
    assert gate.resolve(path) == (True, "")

    clock[0] += 31 * 24 * 3600
    gate.save()
    assert gate.state == {"fallback_indexed": {}, "first_seen": {}, "late_memory_readers": {}}


def test_agy_scenario_leaves_no_file_descriptors_open(tmp_path):
    before = len(os.listdir('/dev/fd'))
    with ExitStack() as resources:
        test_an_agy_conversation_is_indexed_as_gemini_with_its_own_conversation_id(tmp_path, resources)
    assert len(os.listdir('/dev/fd')) <= before
