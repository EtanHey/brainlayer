"""Session writers must never choose a T3 mirror as their canonical row."""

import pytest

from brainlayer.dedupe import find_duplicate, normalized_exact_hash
from brainlayer.drain import _insert_or_merge_chunk
from brainlayer.vector_store import VectorStore
from brainlayer.watcher_bridge import _archive_reconciliation_stored, create_flush_callback

TEXT = "Session ownership preserves this exact original text for retrieval and conversation provenance."
STAMP = "2026-11-17T12:00:00Z"


def chunk(chunk_id, source="realtime_watcher", **extra):
    return {
        "id": chunk_id,
        "content": TEXT,
        "created_at": STAMP,
        "source": source,
        "source_file": "session.jsonl",
        "project": "alpha",
        "content_type": "assistant_text",
        "metadata": {},
        **extra,
    }


def seed(store, chunk_id="t3:mirror", source="t3", **extra):
    store.upsert_chunks([chunk(chunk_id, source, allow_duplicate=True, **extra)], [None])


@pytest.mark.parametrize("route", ["watcher", "archive", "drain", "index"])
@pytest.mark.parametrize("order", ["t3-first", "session-first"])
def test_session_row_survives_both_orders(tmp_path, monkeypatch, route, order):
    db = tmp_path / "test.db"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    with VectorStore(db) as store:
        if order == "t3-first":
            seed(store)
        if route in {"watcher", "archive"}:
            entry = {
                "type": "assistant",
                "message": {"role": "assistant", "content": TEXT},
                "timestamp": STAMP,
                "_source_file": str(tmp_path / "projects/alpha/session.jsonl"),
                "_archive_reconciliation": route == "archive",
            }
            assert create_flush_callback(db, arbitrated=False)([entry]) == 1
        elif route == "drain":
            assert _insert_or_merge_chunk(store.conn, chunk("session")) == "session"
        else:
            store.upsert_chunks([chunk("session", "claude_code")], [None])
        if order == "session-first":
            seed(store)
        assert store.conn.execute("SELECT count(*) FROM chunks WHERE source != 't3'").fetchone()[0] == 1
        assert store.conn.execute("SELECT count(*) FROM chunk_id_alias").fetchone()[0] == 0
        assert store.conn.execute("SELECT seen_count FROM chunks WHERE id='t3:mirror'").fetchone()[0] == 1


@pytest.mark.parametrize("source", ["t3", "claude_code", None])
def test_archive_reread_requires_a_session_twin(tmp_path, source):
    db = tmp_path / "test.db"
    with VectorStore(db) as store:
        seed(store, source=source)
        stored = _archive_reconciliation_stored(db, "new-session", TEXT, STAMP, "alpha", "assistant_text")
        assert stored is (source != "t3")
        if not stored:
            assert _insert_or_merge_chunk(store.conn, chunk("new-session")) == "new-session"
            assert store.conn.execute("SELECT source FROM chunks WHERE id='new-session'").fetchone()[0] != "t3"


@pytest.mark.parametrize("source", ["claude_code", None])
@pytest.mark.parametrize("route", ["drain", "index"])
def test_ordinary_session_dedupe_is_preserved(tmp_path, source, route):
    with VectorStore(tmp_path / "test.db") as store:
        seed(store)
        seed(store, "older-session", source)
        if route == "drain":
            assert _insert_or_merge_chunk(store.conn, chunk("new-session")) == "older-session"
        else:
            store.upsert_chunks([chunk("new-session")], [None])
        assert store.conn.execute(
            "SELECT canonical_chunk_id FROM chunk_id_alias WHERE old_chunk_id='new-session'"
        ).fetchone() == ("older-session",)


@pytest.mark.parametrize("near", [False, True])
def test_source_exclusion_applies_to_exact_and_simhash(tmp_path, near):
    with VectorStore(tmp_path / "test.db") as store:
        original = " ".join(f"token{index}" for index in range(100))
        seed(store, content=original)
        content = original.replace("token0", "changed0") if near else original
        args = dict(
            chunk_id="session", content=content, created_at=STAMP, project="alpha", content_type="assistant_text"
        )
        # First establish that this fixture really hits the respective matching path.
        hit, _ = find_duplicate(store.conn, **args)
        assert hit is not None
        assert hit.mechanism == ("simhash" if near else "sha256")
        hit, _ = find_duplicate(store.conn, **args, exclude_sources=("t3",))
        assert hit is None


@pytest.mark.parametrize("route", ["watcher", "archive", "drain", "index"])
@pytest.mark.parametrize("other", ["none", "twin", "older-unrelated"])
def test_returning_t3_alias_resolves_to_its_own_session_row(tmp_path, monkeypatch, route, other):
    """The active own row wins at read time while every original alias stays intact."""
    from brainlayer.dedupe import write_alias

    db = tmp_path / "test.db"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    incoming_id = "rt-12345678-" + normalized_exact_hash(TEXT)[:16]
    with VectorStore(db) as store:
        seed(store, content_type="user_message", metadata={"t3_provider_session_id": "original-session"})
        if other != "none":
            seed(
                store,
                other,
                "realtime_watcher",
                created_at="2026-10-01T12:00:01Z",
                conversation_id="original-session" if other == "twin" else "someone-else",
                content_type="user_message",
            )
        write_alias(store.conn, old_chunk_id=incoming_id, canonical_chunk_id="t3:mirror")
        preimage = store.conn.execute("SELECT * FROM chunk_id_alias WHERE old_chunk_id=?", (incoming_id,)).fetchone()
        write_alias(store.conn, old_chunk_id="unrelated", canonical_chunk_id="t3:mirror")
        unrelated = store.conn.execute("SELECT * FROM chunk_id_alias WHERE old_chunk_id='unrelated'").fetchone()

        def replay():
            if route in {"watcher", "archive"}:
                create_flush_callback(db, arbitrated=False)(
                    [
                        {
                            "type": "user",
                            "message": {"role": "user", "content": TEXT},
                            "timestamp": STAMP,
                            "_archive_reconciliation": route == "archive",
                            "_source_file": str(tmp_path / "projects/alpha/12345678-1111-2222-3333-444455556666.jsonl"),
                        }
                    ]
                )
            elif route == "drain":
                _insert_or_merge_chunk(store.conn, chunk(incoming_id, content_type="user_message"))
            else:
                store.upsert_chunks([chunk(incoming_id, content_type="user_message")], [None])

        replay()
        assert store.resolve_chunk_id(incoming_id) == incoming_id
        assert store.get_chunk(incoming_id)["id"] == incoming_id
        assert store.get_context(incoming_id)["target"]["id"] == incoming_id
        assert (
            store.conn.execute("SELECT * FROM chunk_id_alias WHERE old_chunk_id=?", (incoming_id,)).fetchone()
            == preimage
        )
        assert store.conn.execute("SELECT 1 FROM sqlite_master WHERE name='session_alias_preimage'").fetchone() is None
        replay()
        assert store.resolve_chunk_id(incoming_id) == incoming_id
        assert store.get_chunk(incoming_id)["id"] == incoming_id
        assert store.get_context(incoming_id)["target"]["id"] == incoming_id
        assert store.conn.execute("SELECT * FROM chunk_id_alias WHERE old_chunk_id='unrelated'").fetchone() == unrelated
        assert store.conn.execute("SELECT seen_count FROM chunks WHERE id='t3:mirror'").fetchone() == (1,)


@pytest.mark.parametrize("mode", ["exact", "alias", "reactivate", "own-shadow"])
@pytest.mark.parametrize("canonical_source", ["t3", "realtime_watcher"])
def test_codex_manifest_never_treats_a_t3_twin_as_stored(tmp_path, mode, canonical_source):
    import sqlite3

    from brainlayer.dedupe import write_alias
    from brainlayer.ingest.codex_manifest import _resolved_candidate

    db = tmp_path / "manifest.db"
    row = {**chunk("codex-session"), "chunk_id": "codex-session", "source_end_offset": 100}
    with VectorStore(db) as store:
        seed(store, source=canonical_source)
        if mode in {"alias", "own-shadow"}:
            write_alias(store.conn, old_chunk_id=row["chunk_id"], canonical_chunk_id="t3:mirror")
        if mode == "own-shadow":
            seed(store, row["chunk_id"], "realtime_watcher")
        if mode == "reactivate":
            seed(store, row["chunk_id"], "realtime_watcher")
            store.conn.execute("UPDATE chunks SET archived_at='2026-11-18' WHERE id=?", (row["chunk_id"],))
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as conn:
        match = _resolved_candidate(conn, row)
    if mode == "own-shadow":
        assert match[0] == row["chunk_id"]
    elif canonical_source == "t3":
        assert match is None
    else:
        assert match[0] == "t3:mirror"


def test_failed_index_insert_keeps_alias_unchanged(tmp_path, monkeypatch):
    from brainlayer import vector_store
    from brainlayer.dedupe import write_alias

    with VectorStore(tmp_path / "rollback.db") as store:
        seed(store)
        write_alias(store.conn, old_chunk_id="returning", canonical_chunk_id="t3:mirror")
        original = store.conn.execute("SELECT * FROM chunk_id_alias WHERE old_chunk_id='returning'").fetchone()

        def fail_prepare(*args, **kwargs):
            raise RuntimeError("fail before insertion")

        monkeypatch.setattr(vector_store, "prepare_canonical_insert", fail_prepare)
        with pytest.raises(RuntimeError, match="fail before insertion"):
            store.upsert_chunks([chunk("returning")], [None])
        assert store.conn.execute("SELECT * FROM chunk_id_alias WHERE old_chunk_id='returning'").fetchone() == original
        assert store.conn.execute("SELECT 1 FROM chunks WHERE id='returning'").fetchone() is None
        assert store.conn.execute("SELECT 1 FROM sqlite_master WHERE name='session_alias_preimage'").fetchone() is None


@pytest.mark.parametrize("route", ["watcher", "drain", "index"])
def test_unusable_non_t3_alias_cannot_shadow_materialized_session_id(tmp_path, monkeypatch, route):
    from brainlayer.dedupe import write_alias

    db = tmp_path / "non-t3-shadow.db"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    incoming_id = "rt-12345678-" + normalized_exact_hash(TEXT)[:16]
    with VectorStore(db) as store:
        seed(store, "archived-session", "realtime_watcher")
        store.conn.execute("UPDATE chunks SET archived_at='2026-11-18' WHERE id='archived-session'")
        write_alias(store.conn, old_chunk_id=incoming_id, canonical_chunk_id="archived-session")
        if route == "watcher":
            create_flush_callback(db, arbitrated=False)(
                [
                    {
                        "type": "assistant",
                        "message": {"role": "assistant", "content": TEXT},
                        "timestamp": STAMP,
                        "_source_file": str(tmp_path / "projects/alpha/12345678-1111-2222-3333-444455556666.jsonl"),
                    }
                ]
            )
        elif route == "drain":
            _insert_or_merge_chunk(store.conn, chunk(incoming_id))
        else:
            store.upsert_chunks([chunk(incoming_id)], [None])
        assert store.resolve_chunk_id(incoming_id) == incoming_id
        assert store.get_chunk(incoming_id)["id"] == incoming_id


@pytest.mark.parametrize("route", ["watcher", "drain", "index"])
def test_archived_dedupe_loser_still_resolves_to_survivor(tmp_path, monkeypatch, route):
    from brainlayer.dedupe import merge_duplicate_chunk

    db = tmp_path / "survivor.db"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    incoming_id = "rt-12345678-" + normalized_exact_hash(TEXT)[:16]
    with VectorStore(db) as store:
        seed(store, "canon-session", "realtime_watcher", created_at="2026-10-01T12:00:00Z")
        seed(store, incoming_id, "realtime_watcher")
        merge_duplicate_chunk(
            store.conn,
            canonical_id="canon-session",
            duplicate_id=incoming_id,
            incoming=chunk(incoming_id),
            mechanism="sha256",
            hamming_distance_value=0,
            archive_existing_duplicate=True,
        )
        alias = store.conn.execute("SELECT * FROM chunk_id_alias WHERE old_chunk_id=?", (incoming_id,)).fetchone()
        seen = store.conn.execute("SELECT seen_count FROM chunks WHERE id='canon-session'").fetchone()[0]
        if route == "watcher":
            create_flush_callback(db, arbitrated=False)(
                [
                    {
                        "type": "assistant",
                        "message": {"role": "assistant", "content": TEXT},
                        "timestamp": STAMP,
                        "_source_file": str(tmp_path / "projects/alpha/12345678-1111-2222-3333-444455556666.jsonl"),
                    }
                ]
            )
        elif route == "drain":
            _insert_or_merge_chunk(store.conn, chunk(incoming_id))
        else:
            store.upsert_chunks([chunk(incoming_id)], [None])
        assert store.resolve_chunk_id(incoming_id) == "canon-session"
        assert store.get_chunk(incoming_id)["id"] == "canon-session"
        assert store.get_context(incoming_id)["target"]["id"] == "canon-session"
        assert (
            store.conn.execute("SELECT * FROM chunk_id_alias WHERE old_chunk_id=?", (incoming_id,)).fetchone() == alias
        )
        assert store.conn.execute("SELECT seen_count FROM chunks WHERE id='canon-session'").fetchone()[0] > seen


@pytest.mark.parametrize("retired", [None, "archived_at", "superseded_by", "aggregated_into", "invalid_at"])
def test_read_precedence_depends_on_own_lifecycle(tmp_path, retired):
    from brainlayer.dedupe import write_alias

    with VectorStore(tmp_path / "precedence.db") as store:
        seed(store, "survivor", "realtime_watcher")
        seed(store, "own", "realtime_watcher")
        if retired:
            store.conn.execute(f"UPDATE chunks SET {retired}='retired' WHERE id='own'")
        write_alias(store.conn, old_chunk_id="own", canonical_chunk_id="survivor")
        expected = "own" if retired is None else "survivor"
        assert store.resolve_chunk_id("own") == expected
        assert store.get_chunk("own")["id"] == expected
        assert store.get_context("own")["target"]["id"] == expected
