"""The three watcher/T3 orderings, on real private SQLite stores."""

import json
import os
import subprocess
import types
from datetime import datetime, timezone

import pytest

from brainlayer.ingest import t3
from tests.test_t3_thread_tables import _brainlayer_db, _session_chunk, _t3_fixture


def _clean_git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


@pytest.fixture
def lane(tmp_path, monkeypatch):
    module = t3
    if os.environ.get("T3_BASELINE"):
        module = types.ModuleType("brainlayer.ingest.t3_baseline")
        module.__package__ = "brainlayer.ingest"
        # Read baseline code without switching or editing the live root checkout.
        source = subprocess.check_output(["git", "show", "5119839e:src/brainlayer/ingest/t3.py"], env=_clean_git_env())
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
            conn.execute("UPDATE chunks SET content=? WHERE id='watcher'", (text,))
        conn.close()

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
    assert state["held_threads"] == ["th-claude"]
    assert "pending_chunks" not in saved
    watcher()
    assert run().chunks_indexed == 0
    assert next(iter(json.loads(bookmark.read_text())["sources"].values()))["held_threads"] == []


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


def test_deferred_run_keeps_held_threads_alongside_per_source_start(lane, monkeypatch):
    import sqlite3

    run, watcher, db, source, bookmark = lane
    watcher()
    run()
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM chunks WHERE id='watcher'")
    conn.close()
    with sqlite3.connect(source) as conn:
        conn.execute(
            "INSERT INTO orchestration_v2_projection_messages VALUES ('budget', 'th-none', 'user', ?, ?, 0, ?)",
            (
                "2026-11-19T00:00:00Z",
                json.dumps({"text": "An older T3-only turn awaiting its bounded write budget."}),
                "2026-11-19T00:00:00Z",
            ),
        )
        for thread in ["th-claude", "th-none"]:
            conn.execute(
                "INSERT INTO orchestration_events(event_type,stream_id,occurred_at,payload_json) VALUES ('message.updated',?,'2026-11-20T00:00:00Z','{}')",
                (thread,),
            )
    conn.close()
    monkeypatch.setattr(t3, "DEFAULT_T3_MAX_CHANGED_CHUNKS", 0)
    assert run().chunks_deferred == 1
    saved = json.loads(bookmark.read_text())
    state = next(iter(saved["sources"].values()))
    assert state["held_threads"] == ["th-claude"]
    assert state["last_sequence"] == 1
    assert state["occurred_at"] == "2026-11-19T00:00:00Z"
    assert state["pending_to"] == 3
    assert saved["pending_chunks"] == ["t3:th-none:budget:0"]
    monkeypatch.setattr(t3, "DEFAULT_T3_MAX_CHANGED_CHUNKS", 1)
    assert run().chunks_indexed == 1
    state = next(iter(json.loads(bookmark.read_text())["sources"].values()))
    assert state["held_threads"] == ["th-claude"]
    assert state["last_sequence"] == 3
    watcher()
    assert run().chunks_indexed == 0
    assert next(iter(json.loads(bookmark.read_text())["sources"].values()))["held_threads"] == []


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


def test_user_message_the_session_never_stored_but_a_later_reply_exists(lane):
    """A later session row is not evidence for an older queued steering message."""
    import sqlite3
    from datetime import timedelta

    run, _, db, source, _ = lane
    later = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with sqlite3.connect(source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_messages SET created_at=?", (old,))
    with sqlite3.connect(db) as conn:
        _session_chunk(conn, "reply-after-queued-message", "sess-claude", later)
    assert run().chunks_indexed == 1
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM chunks WHERE source='t3' AND json_extract(metadata,'$.message_id')='m'"
            ).fetchone()[0]
            == 1
        )


def test_last_seen_bump_from_a_merge_moves_the_watermark(lane):
    """A last_seen bump from another session must not cover this absent message."""
    import sqlite3
    from datetime import timedelta

    run, _, db, source, _ = lane
    later = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with sqlite3.connect(source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_messages SET created_at=?", (old,))
    with sqlite3.connect(db) as conn:
        _session_chunk(
            conn, "old-row-bumped-by-other-session", "sess-claude", "2026-11-19T00:00:00Z", last_seen_at=later
        )
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


@pytest.mark.parametrize("stored", ["split", "drift"])
def test_audit_accepts_stored_coverage_across_chunking_or_t3_revision(lane, monkeypatch, stored):
    import sqlite3

    from brainlayer.ingest.t3_audit import audit_t3

    _, _, db, source, bookmark = lane
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(bookmark.parent / "alerts.json"))
    with sqlite3.connect(source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_messages SET created_at='2026-11-20T12:00:00Z'")
        text = json.loads(conn.execute("SELECT payload_json FROM orchestration_v2_projection_messages").fetchone()[0])[
            "text"
        ]
    with sqlite3.connect(db) as conn:
        if stored == "split":
            # Stored in a different chunk order/shape from today's T3 chunker.
            for key, content in [("part-z", text[:33]), ("part-a", text[33:])]:
                _session_chunk(conn, key, "sess-claude", "2026-11-20T12:00:00Z")
                conn.execute("UPDATE chunks SET content=? WHERE id=?", (content, key))
        else:
            _session_chunk(conn, "older-t3-id", "th-claude", "2026-11-20T12:00:00Z", source="t3")
            conn.execute(
                "UPDATE chunks SET content=?, metadata=? WHERE id='older-t3-id'",
                ("The previous revision of this stored message.", json.dumps({"message_id": "m"})),
            )
    result = audit_t3(source, db_path=db, health_path=bookmark.parent / "health.json", projection_version=2)
    assert result["uncovered"] == 0


def test_audit_partial_session_text_still_alarms(lane, monkeypatch):
    import sqlite3

    from brainlayer.ingest.t3_audit import audit_t3
    from brainlayer.job_alerts import active_alerts

    _, _, db, source, bookmark = lane
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(bookmark.parent / "alerts.json"))
    with sqlite3.connect(source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_messages SET created_at='2026-11-20T12:00:00Z'")
    with sqlite3.connect(db) as conn:
        _session_chunk(conn, "partial", "sess-claude", "2026-11-20T12:00:00Z")
        conn.execute("UPDATE chunks SET content='Please preserve this specific turn' WHERE id='partial'")
    result = audit_t3(source, db_path=db, health_path=bookmark.parent / "health.json", projection_version=2)
    assert result["uncovered"] == 1
    assert "t3-audit" in active_alerts(bookmark.parent / "alerts.json")


def test_audit_reports_known_index_policy_exclusion_without_alert(lane, monkeypatch):
    import sqlite3

    from brainlayer.ingest.t3_audit import audit_t3
    from brainlayer.job_alerts import active_alerts

    run, _, db, source, bookmark = lane
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(bookmark.parent / "alerts.json"))
    text = "You are Codex.\n# Base context\n" + "Synthetic agent instruction scaffold. " * 8
    with sqlite3.connect(source) as conn:
        conn.execute(
            "UPDATE orchestration_v2_projection_messages SET created_at='2026-11-20T12:00:00Z', payload_json=?",
            (json.dumps({"text": text}),),
        )
    assert run().chunks_indexed == 0  # Real T3 index admission refuses the scaffold.
    result = audit_t3(source, db_path=db, health_path=bookmark.parent / "health.json", projection_version=2)
    assert result["uncovered_new"] == 0
    assert result["uncovered_known"] == 1
    assert result["known_skip_reasons"] == {"system_prompt_guard": 1}
    assert "t3-audit" not in active_alerts(bookmark.parent / "alerts.json")
    from typer.testing import CliRunner

    from brainlayer.cli import app

    cli = CliRunner().invoke(
        app,
        ["t3-audit", "--state-db", str(source), "--db", str(db), "--health-path", str(bookmark.parent / "health.json")],
    )
    assert cli.exit_code == 0, cli.output
    assert "uncovered_new = 0" in cli.output
    assert "uncovered_known = 1" in cli.output


@pytest.mark.parametrize("layer", ["t3", "session", "codex-session"])
@pytest.mark.parametrize("lifecycle", ["archived_at", "superseded_by", "aggregated_into"])
def test_audit_counts_stored_then_lifecycle_managed_text(lane, monkeypatch, layer, lifecycle):
    import sqlite3

    from brainlayer.ingest.t3_audit import audit_t3

    _, _, db, source, bookmark = lane
    monkeypatch.setenv("BRAINLAYER_JOB_ALERT_PATH", str(bookmark.parent / "alerts.json"))
    with sqlite3.connect(source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_messages SET created_at='2026-11-20T12:00:00Z'")
        text = json.loads(conn.execute("SELECT payload_json FROM orchestration_v2_projection_messages").fetchone()[0])[
            "text"
        ]
        if layer == "codex-session":
            native = "01a1aaaa-0000-7000-8000-000000000099"
            conn.execute(
                "UPDATE orchestration_v2_projection_provider_threads SET driver='codex', provider='codex', payload_json=? WHERE provider_thread_id='pt-claude'",
                (json.dumps({"nativeThreadRef": {"nativeId": native}}),),
            )
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM chunks WHERE conversation_id='sess-claude'")
        conversation = "th-claude" if layer == "t3" else "sess-claude"
        if layer == "codex-session":
            conversation = f"rollout-2026-11-20T12-00-00-{native}"
        _session_chunk(
            conn, "managed", conversation, "2026-11-20T12:00:00Z", source="t3" if layer == "t3" else "realtime_watcher"
        )
        value = "2026-11-20T13:00:00Z" if lifecycle == "archived_at" else "deliberate-replacement"
        conn.execute(
            f"UPDATE chunks SET content=?, metadata=?, {lifecycle}=? WHERE id='managed'",
            (text, json.dumps({"message_id": "m"}), value),
        )
    result = audit_t3(source, db_path=db, health_path=bookmark.parent / "health.json", projection_version=2)
    assert result["uncovered"] == 0
