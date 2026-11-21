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


@pytest.mark.parametrize("route", ["watcher", "drain", "index"])
def test_replayed_alias_moves_from_t3_to_its_session_twin(tmp_path, monkeypatch, route):
    """A later session's existing alias follows ordinary dedupe to the mirror's twin."""
    from brainlayer.dedupe import write_alias

    db = tmp_path / "test.db"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    incoming_id = "rt-12345678-" + normalized_exact_hash(TEXT)[:16]
    with VectorStore(db) as store:
        seed(store, metadata={"t3_provider_session_id": "original-session"})
        seed(
            store,
            "original-session-twin",
            "realtime_watcher",
            created_at="2026-11-17T12:00:01Z",
            conversation_id="original-session",
        )
        write_alias(store.conn, old_chunk_id=incoming_id, canonical_chunk_id="t3:mirror")
        write_alias(store.conn, old_chunk_id="unrelated", canonical_chunk_id="t3:mirror")
        unrelated = store.conn.execute("SELECT * FROM chunk_id_alias WHERE old_chunk_id='unrelated'").fetchone()
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
        assert store.conn.execute(
            "SELECT canonical_chunk_id FROM chunk_id_alias WHERE old_chunk_id=?", (incoming_id,)
        ).fetchone() == ("original-session-twin",)
        assert store.conn.execute("SELECT * FROM chunk_id_alias WHERE old_chunk_id='unrelated'").fetchone() == unrelated
        assert store.conn.execute("SELECT seen_count FROM chunks WHERE id='t3:mirror'").fetchone() == (1,)
