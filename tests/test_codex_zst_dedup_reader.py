"""Exact-content resolution uses the writer's fixture rows and never writes them."""

import sqlite3

import pytest

from brainlayer.drain import _apply_watcher, _ensure_watcher_liveness_schema
from brainlayer.ingest import codex_manifest as reader
from brainlayer.vector_store import VectorStore


@pytest.mark.parametrize("project,kind", [("repo", "user_message"), (None, None)])
def test_other_source_is_resolved_without_merging(tmp_path, project, kind):
    path = tmp_path / "fixture.db"
    event = dict(
        chunk_id="kept",
        content="  Exact shared content for a distinct rollout.  ",
        source_file="old.jsonl",
        project=project,
        content_type=kind,
    )
    with VectorStore(path) as store:
        _ensure_watcher_liveness_schema(store.conn)
        _apply_watcher(store.conn, event)
    candidate = dict(event, chunk_id="proposed", source_file="new.jsonl")
    # On the old layer, exercise its actual id-only decision (no missing-symbol failure).
    resolve = getattr(reader, "_resolved_candidate", lambda db, row: reader._already_committed(db, row["chunk_id"]))
    before = path.read_bytes()
    with reader._committed(path) as db:
        assert resolve(db, candidate) == ("kept", "old.jsonl", "exact_hash")
        assert not resolve(db, dict(candidate, project="other"))
        assert not resolve(db, dict(candidate, content_type="ai_code"))
    assert path.read_bytes() == before
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT seen_count FROM chunks WHERE id='kept'").fetchone() == (1,)
        db.execute("UPDATE chunks SET archived_at='2026-11-08'")
    with reader._committed(path) as db:
        assert not resolve(db, candidate)


def test_archived_own_id_requires_writer_reactivation(tmp_path):
    path = tmp_path / "fixture.db"
    event = dict(
        chunk_id="own",
        content="Stored archive reactivation control",
        source_file="same.jsonl",
        project="repo",
        content_type="user_message",
        source_end_offset=12,
    )
    with VectorStore(path) as store:
        _ensure_watcher_liveness_schema(store.conn)
        _apply_watcher(store.conn, event)
        store.conn.execute("UPDATE chunks SET archived_at='2026-11-08'")
    with reader._committed(path) as db:
        assert reader._resolved_candidate(db, event) is None
        assert reader._resolved_candidate(db, dict(event, source_file="other.jsonl"))
    with sqlite3.connect(path) as db:
        db.execute("UPDATE chunks SET archived_at=NULL, superseded_by='later'")
    with reader._committed(path) as db:
        assert reader._resolved_candidate(db, event)


def test_writer_alias_resolves_rewritten_canonical(tmp_path):
    path = tmp_path / "fixture.db"
    one = dict(
        chunk_id="canonical",
        content="Remember the alpha rollout plan for staging today",
        source_file="old.jsonl",
        project="repo",
        content_type="user_message",
    )
    two = dict(
        one,
        chunk_id="candidate",
        content="remember THE alpha rollout plan, for staging today!!",
        source_file="new.jsonl",
    )
    with VectorStore(path) as store:
        _ensure_watcher_liveness_schema(store.conn)
        _apply_watcher(store.conn, one)
        _apply_watcher(store.conn, two)
    with reader._committed(path) as db:
        assert reader._resolved_candidate(db, two) == ("canonical", "old.jsonl", "writer_alias")


def test_archived_own_with_live_duplicate_resolves_before_reactivation(tmp_path):
    path = tmp_path / "fixture.db"
    event = dict(
        chunk_id="own",
        content="The quarterly capacity review moved to the second Tuesday each month",
        source_file="same.jsonl",
        project="repo",
        content_type="user_message",
        source_end_offset=12,
    )
    with VectorStore(path) as store:
        _ensure_watcher_liveness_schema(store.conn)
        _apply_watcher(store.conn, event)
        store.conn.execute("UPDATE chunks SET archived_at='2026-11-08'")
        _apply_watcher(store.conn, dict(event, chunk_id="live", source_file="elsewhere.jsonl"))
        for _ in range(3):
            with reader._committed(path) as db:
                match = reader._resolved_candidate(db, event)
            assert match is not None and match[:2] == ("live", "elsewhere.jsonl")
        assert store.conn.execute("SELECT seen_count FROM chunks WHERE id='live'").fetchone() == (1,)
