"""The three watcher/T3 orderings, on real private SQLite stores."""

import json
import os
import subprocess
import types
from datetime import datetime, timezone

import pytest

from brainlayer.ingest import t3
from tests.test_t3_thread_tables import _brainlayer_db, _session_chunk, _t3_fixture


@pytest.fixture
def lane(tmp_path, monkeypatch):
    module = t3
    if os.environ.get("T3_BASELINE"):
        module = types.ModuleType("brainlayer.ingest.t3_baseline")
        module.__package__ = "brainlayer.ingest"
        # Read baseline code without switching or editing the live root checkout.
        source = subprocess.check_output(["git", "show", "5119839e:src/brainlayer/ingest/t3.py"])
        import sys

        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(source, "baseline-t3.py", "exec"), module.__dict__)
    source = _t3_fixture(tmp_path / "t3.sqlite")
    db = _brainlayer_db(tmp_path / "brain.db")
    import sqlite3

    text = "Please preserve this specific turn and its full session ownership evidence."
    stamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(source) as conn:
        conn.execute("DELETE FROM orchestration_v2_projection_messages")
        conn.execute(
            "INSERT INTO orchestration_v2_projection_messages VALUES ('m', 'th-claude', 'user', ?, ?, 0, ?)",
            (stamp, json.dumps({"text": text}), stamp),
        )
    bookmark = tmp_path / "bookmark.json"

    def run():
        return module.ingest_t3(
            source, db_path=db, health_path=tmp_path / "health.json", bookmark_path=bookmark, projection_version=2
        )

    def watcher():
        with sqlite3.connect(db) as conn:
            _session_chunk(conn, "watcher", "sess-claude", stamp)

    return run, watcher, db, source, bookmark


def test_watcher_first_writes_zero(lane):
    run, watcher, *_ = lane
    watcher()
    assert run().chunks_indexed == 0


def test_watcher_late_holds_advances_then_retries_without_an_event(lane):
    run, watcher, _, _, bookmark = lane
    assert run().chunks_indexed == 0
    saved = json.loads(bookmark.read_text())
    state = next(iter(saved["sources"].values()))
    assert state["last_sequence"] == 1
    assert state["held"] == ["th-claude"]
    assert "pending_chunks" not in saved
    watcher()
    assert run().chunks_indexed == 0
    assert next(iter(json.loads(bookmark.read_text())["sources"].values()))["held"] == []


def test_watcher_never_falls_back_after_grace(lane, monkeypatch):
    from brainlayer.ingest import t3_ownership

    run, _, db, _, _ = lane
    assert run().chunks_indexed == 0
    real_owner = t3_ownership.message_owner
    monkeypatch.setattr(t3_ownership, "message_owner", lambda m, w: real_owner(m, w, now=10**12))
    assert run().chunks_indexed == 1
    import sqlite3

    with sqlite3.connect(db) as conn:
        assert (
            json.loads(conn.execute("SELECT metadata FROM chunks WHERE source='t3'").fetchone()[0])["t3_owner_reason"]
            == "session_late"
        )


def test_absent_side_tables_preserve_main(lane):
    import sqlite3

    run, watcher, db, *_ = lane
    watcher()
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE t3_thread_sessions")
        conn.execute("DROP TABLE t3_threads")
    assert run().chunks_indexed == 1


def test_short_user_is_kept_even_below_watermark(lane):
    import sqlite3

    run, watcher, _, source, _ = lane
    watcher()
    with sqlite3.connect(source) as conn:
        conn.execute(
            "UPDATE orchestration_v2_projection_messages SET payload_json=?", (json.dumps({"text": "yes do it"}),)
        )
    assert run().chunks_indexed == 1


def test_unlinked_session_is_owned_by_t3(lane):
    import sqlite3

    run, _, _, source, _ = lane
    with sqlite3.connect(source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_provider_threads SET payload_json='{}'")
    assert run().chunks_indexed == 1


def test_audit_detects_missing_text_and_recovers_after_fallback(lane, monkeypatch):
    import sqlite3

    from brainlayer.ingest.t3_audit import audit_t3
    from brainlayer.job_alerts import active_alerts

    run, _, db, source, bookmark = lane
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(bookmark.parent / "alerts.json"))
    with sqlite3.connect(source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_messages SET created_at='2026-11-20T12:00:00Z'")
    health = bookmark.parent / "health.json"
    before = audit_t3(source, db_path=db, health_path=health, projection_version=2)
    assert before["uncovered"] == 1
    assert "t3-audit" in active_alerts(bookmark.parent / "alerts.json")
    assert run().chunks_indexed == 1
    after = audit_t3(source, db_path=db, health_path=health, projection_version=2)
    assert after["uncovered"] == 0
    assert "t3-audit" not in active_alerts(bookmark.parent / "alerts.json")


@pytest.mark.parametrize(
    "role,text",
    [
        ("user", "yes"),
        ("user", "Please implement the complete behavior with evidence."),
        ("assistant", "Done."),
        ("assistant", "Here is a detailed implementation explanation. " * 6),
    ],
)
def test_admission_matches_watcher(role, text):
    from brainlayer.ingest.t3_ownership import session_admits
    from brainlayer.pipeline.chunk import chunk_content
    from brainlayer.watcher_bridge import classify_content, should_skip_chunk_content, should_skip_entry

    entry = {"type": role, "message": {"role": role, "content": [{"type": "text", "text": text}]}}
    classified = None if should_skip_entry(entry) else classify_content(entry)
    expected = classified is not None and any(
        not should_skip_chunk_content(c.content) for c in chunk_content(classified)
    )
    assert session_admits(t3.T3Message("m", "t", role, text, "2026-11-21T00:00:00Z")) == expected
