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


@pytest.mark.parametrize("shortcut", ["exact_hash", "writer_alias", "own_id"])
@pytest.mark.parametrize(
    "existing_class,incoming_class",
    [
        ("agent-thinking", "cli-agent"),
        ("cli-agent", "agent-thinking"),
        ("brain-worker", "agent-thinking"),
    ],
)
def test_manifest_queues_cross_visibility_candidates_for_the_writer(
    tmp_path, monkeypatch, shortcut, existing_class, incoming_class
):
    from types import SimpleNamespace

    from brainlayer import codex_zst, queue_io

    path = tmp_path / "fixture.db"
    existing = dict(
        chunk_id="canonical",
        content="Manifest visibility recovery must preserve this repeated assistant reply separately.",
        source_file="old.jsonl",
        project="repo",
        content_type="assistant_text",
        source_class=existing_class,
    )
    candidate = dict(existing, chunk_id="incoming", source_file="new.jsonl", source_class=incoming_class)
    if shortcut == "own_id":
        candidate["chunk_id"] = existing["chunk_id"]
    with VectorStore(path) as store:
        _ensure_watcher_liveness_schema(store.conn)
        _apply_watcher(store.conn, existing)
        if shortcut == "writer_alias":
            store.conn.execute(
                "INSERT INTO chunk_id_alias(old_chunk_id, canonical_chunk_id, deprecated_at) VALUES(?, ?, '2026-11-20')",
                (candidate["chunk_id"], existing["chunk_id"]),
            )
    queued = []
    monkeypatch.setattr(codex_zst, "queue_has_capacity", lambda: True)
    monkeypatch.setattr(queue_io, "enqueue_watcher_chunk", lambda **row: queued.append(row))
    receipt = dict(
        filtered=0,
        would_insert=0,
        duplicate_content=0,
        all_duplicate_matches=[],
        duplicate_matches=[],
        queued=0,
        queued_chunk_ids=[],
    )
    preview = []

    def plan(entries):
        preview.append(candidate)
        return SimpleNamespace(skipped=0, deferred_entries=[])

    with reader._committed(path) as db:
        reader._plan_window(plan, [], preview, receipt, db, False)
    assert queued == [candidate]
    assert receipt["queued"] == 1
    assert receipt["duplicate_content"] == 0
    # Real drain writer, not a planner mock: visible same-id input promotes the old thinking row.
    with VectorStore(path) as store:
        _apply_watcher(store.conn, queued[0])
        stored_class = store.conn.execute(
            "SELECT source_class FROM chunks WHERE id=?", (candidate["chunk_id"],)
        ).fetchone()[0]
        expected = existing_class if shortcut == "own_id" and incoming_class == "agent-thinking" else incoming_class
        assert stored_class == expected
