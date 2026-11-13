"""Antigravity (agy), the current Gemini agent, reaches the index under the source name "gemini".

agy exports each conversation's steps as JSONL at
`~/.gemini/antigravity-cli/brain/<conversation>/.system_generated/logs/transcript_full.jsonl`
(`transcript.jsonl` beside it holds the same steps with long fields truncated). Fixtures are synthetic,
in the shape measured on 500 real transcripts (2026-11-13): step_index, source, type
(USER_INPUT | PLANNER_RESPONSE | GENERIC | ERROR_MESSAGE), status, created_at, content.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from brainlayer.agent_provenance import classify_provenance
from brainlayer.watcher import JSONLWatcher, default_watch_roots
from brainlayer.watcher_bridge import create_flush_callback

CONVERSATION = "0115a286-c627-4c2b-9dd2-d9550bfe78b4"


@pytest.fixture(autouse=True)
def _scratch_env(tmp_path, monkeypatch):
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
    with sqlite3.connect(store) as conn:
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


def _watcher(tmp_path: Path, db: Path) -> JSONLWatcher:
    return JSONLWatcher(
        watch_roots=default_watch_roots(tmp_path / "home"),
        registry_path=tmp_path / "offsets.json",
        on_flush=create_flush_callback(db_path=db, arbitrated=False),
        db_path=db,
        batch_size=1,
        flush_interval_ms=0,
        registry_flush_interval_s=0,
    )


def _poll(watcher: JSONLWatcher) -> None:
    watcher.poll_once()
    watcher.indexer.flush()


def _rows(db: Path, needle: str) -> list[tuple]:
    if not db.exists():
        return []
    with sqlite3.connect(db) as conn:
        return conn.execute(
            "SELECT conversation_id, content, metadata FROM chunks WHERE content LIKE ?", (f"%{needle}%",)
        ).fetchall()


def test_an_agy_conversation_is_indexed_as_gemini_with_its_own_conversation_id(tmp_path):
    db = tmp_path / "db" / "brainlayer.db"
    path = _transcript(tmp_path)
    _write(path, [_step(0, "USER_INPUT", "u1"), _step(1, "PLANNER_RESPONSE", "a1")])

    watcher = _watcher(tmp_path, db)
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


def test_tool_and_error_steps_and_the_truncated_twin_are_not_indexed(tmp_path):
    db = tmp_path / "db" / "brainlayer.db"
    steps = [_step(0, "USER_INPUT", "u1"), _step(1, "GENERIC", "g1"), _step(2, "ERROR_MESSAGE", "e1")]
    _write(_transcript(tmp_path), steps)
    _write(_transcript(tmp_path, name="transcript.jsonl"), steps)

    _poll(_watcher(tmp_path, db))

    assert _rows(db, "turn g1:") == [] and _rows(db, "turn e1:") == []
    assert len(_rows(db, "turn u1:")) == 1


def test_an_appended_step_is_indexed_without_duplicating_earlier_ones(tmp_path):
    db = tmp_path / "db" / "brainlayer.db"
    path = _transcript(tmp_path)
    _write(path, [_step(0, "USER_INPUT", "u1"), _step(1, "PLANNER_RESPONSE", "a1")])
    watcher = _watcher(tmp_path, db)
    _poll(watcher)

    _write(path, [_step(2, "USER_INPUT", "u2"), _step(3, "PLANNER_RESPONSE", "a2")], mode="a")
    _poll(watcher)
    _poll(_watcher(tmp_path, db))  # a restarted watcher resumes from its checkpoint

    for marker in ("u1", "a1", "u2", "a2"):
        assert len(_rows(db, f"turn {marker}:")) == 1, marker


def test_every_agy_conversation_keeps_its_own_conversation_id(tmp_path):
    """Every agy file is named transcript_full.jsonl; the conversation dir, not the stem, names the session."""
    db = tmp_path / "db" / "brainlayer.db"
    other = "e1ea897a-32ba-42e2-b7af-e04ebe76a961"
    _write(_transcript(tmp_path), [_step(0, "USER_INPUT", "c1")])
    _write(_transcript(tmp_path, other), [_step(0, "USER_INPUT", "c2")])

    _poll(_watcher(tmp_path, db))

    assert [row[0] for row in _rows(db, "turn c1:")] == [CONVERSATION]
    assert [row[0] for row in _rows(db, "turn c2:")] == [other]


def test_a_malformed_step_is_quarantined_and_counted_and_later_steps_still_land(tmp_path):
    db = tmp_path / "db" / "brainlayer.db"
    path = _transcript(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_step(0, "USER_INPUT", "u1")) + "\n" + '{"step_index": 1, "type": \n')
    _summarise(path)
    _write(path, [_step(2, "PLANNER_RESPONSE", "a2")], mode="a")

    watcher = _watcher(tmp_path, db)
    _poll(watcher)
    _poll(watcher)

    assert watcher._quarantined_record_count_total == 1
    assert len(_rows(db, "turn u1:")) == 1 and len(_rows(db, "turn a2:")) == 1


def test_agy_transcripts_are_tagged_gemini_session(tmp_path):
    decision = classify_provenance(str(_transcript(tmp_path)))
    assert (decision.provenance_tag, decision.source_class) == ("gemini-session", "cli-agent")


def _class_of(db: Path, marker: str) -> tuple[str, str, str]:
    (row,) = _rows(db, f"turn {marker}:")
    meta = json.loads(row[2])
    return meta["provenance_tag"], meta["source_class"], meta["provenance_effective_visibility"]


def test_each_agy_class_lands_by_agys_own_agent_attribution(tmp_path):
    db = tmp_path / "db" / "brainlayer.db"
    interactive, gatherer, reader = (
        "11111111-0000-4000-8000-000000000001",
        "22222222-0000-4000-8000-000000000002",
        "33333333-0000-4000-8000-000000000003",
    )
    _write(_transcript(tmp_path, interactive), [_step(0, "PLANNER_RESPONSE", "own1")], agent="")
    _write(_transcript(tmp_path, gatherer), [_step(0, "PLANNER_RESPONSE", "gath1")], agent="gatherer")
    _write(_transcript(tmp_path, reader), [_step(0, "PLANNER_RESPONSE", "read1")], agent="brain-worker")

    _poll(_watcher(tmp_path, db))

    assert _class_of(db, "own1") == ("gemini-session", "cli-agent", "default")
    assert _class_of(db, "gath1") == ("gemini-subagent", "subagent", "default")
    assert _rows(db, "turn read1:") == []  # a memory reader is kept on disk but never indexed


def test_a_conversation_agy_has_not_summarised_yet_waits_for_its_attribution(tmp_path):
    db = tmp_path / "db" / "brainlayer.db"
    path = _transcript(tmp_path)
    _write(path, [_step(0, "PLANNER_RESPONSE", "wait1")], agent=None)
    _summarise(_transcript(tmp_path, "44444444-0000-4000-8000-000000000004"))  # the store exists, this row does not

    watcher = _watcher(tmp_path, db)
    _poll(watcher)
    assert _rows(db, "turn wait1:") == []
    assert watcher.registry.get(str(path))[0] == 0  # nothing confirmed while it waits

    _summarise(path, "gatherer")
    _poll(watcher)
    assert _class_of(db, "wait1") == ("gemini-subagent", "subagent", "default")
