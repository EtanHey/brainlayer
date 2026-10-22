"""Independent #1450 repro: real store/FTS/vec paths in a private temp universe."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import apsw
import pytest

from brainlayer._helpers import serialize_f32
from brainlayer.store import embed_hot_chunk, embed_pending_chunks, store_memory
from brainlayer.vector_store import VectorStore

BASE = (
    "river forest meadow mountain island desert ocean garden canyon marsh prairie valley "
    "willow cedar maple fern moss amber violet copper silver orange indigo scarlet quartz "
    "granite basalt marble limestone shale pebble branch leaf root bloom seed fruit rain "
    "wind snow frost cloud sunlight moonlight dawn dusk summer winter spring autumn "
) * 4
CHANGED = BASE + "canarymerge"


@pytest.fixture
def private_store(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    assert root.is_relative_to(Path(__file__).resolve().parents[1] / ".build.noindex")
    paths = {
        "BRAINLAYER_DB": "private.db",
        "BRAINBAR_DB_PATH": "private.db",
        "BRAINBAR_SOCKET_PATH": "private.sock",
        "BRAINLAYER_MCP_SOCKET": "private.sock",
        "BRAINLAYER_QUEUE_DIR": "queue",
        "BRAINLAYER_DRAIN_HEALTH_PATH": "drain-health.json",
        "BRAINLAYER_DRAIN_LOG_PATH": "drain.log",
        "BRAINLAYER_DRAIN_ERROR_LOG_PATH": "drain.err.log",
        "BRAINLAYER_WRITER_TELEMETRY_PATH": "writer.jsonl",
        "BRAINLAYER_WRITER_HEARTBEAT_DIR": "heartbeats",
        "BRAINLAYER_WRITER_PIDFILE_DIR": "pidfiles",
        "BRAINLAYER_TRANSPORT_HEALTH_PATH": "transport.json",
        "BRAINLAYER_OBSERVABILITY_PATH": "observability.json",
    }
    for key, suffix in paths.items():
        path = (root / suffix).resolve()
        if not path.is_relative_to(root) or ".local/share/brainlayer" in str(path):
            raise ValueError(f"outside disposable universe: {path}")
        monkeypatch.setenv(key, str(path))
    monkeypatch.setenv("BRAINLAYER_FORBID_BRAINBAR_SOCKET", "1")
    monkeypatch.setenv("BRAINLAYER_FORBID_EMBEDDING_MODEL", "1")
    with VectorStore(root / "private.db") as store:
        assert store._binary_index_available
        yield store


def encode(text):
    return ([0.0, 1.0] if "canarymerge" in text else [1.0, 0.0]) + [0.0] * 1022


def put(store, content=BASE, chunk_id="canonical", embed_fn=None, **kwargs):
    return store_memory(
        store,
        embed_fn,
        content,
        "note",
        project="merge-vectors",
        chunk_id=chunk_id,
        created_at="2026-10-22T00:00:00Z",
        **kwargs,
    )


def vectors(store):
    return [
        list(store.conn.execute(f"SELECT chunk_id, embedding FROM {table}"))
        for table in ("chunk_vectors", "chunk_vectors_binary")
    ]


@pytest.mark.parametrize("same_id", [True, False])
def test_changed_merge_invalidates_both_vectors_and_recovers(private_store, same_id):
    store = private_store
    put(store, embed_fn=encode)
    assert all(vectors(store))
    result = put(store, CHANGED, chunk_id="canonical" if same_id else "incoming")
    assert result["outcome"] == "merged"
    assert result["id"] == "canonical"
    assert vectors(store) == [[], []], "MERGED retained stale float/binary vectors"
    assert list(store.conn.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'canarymerge'")) == [
        ("canonical",)
    ]

    def unavailable(_text):
        raise RuntimeError("encoder unavailable")

    assert not embed_hot_chunk(store, unavailable, "canonical")
    assert vectors(store) == [[], []]
    assert embed_hot_chunk(store, encode, "canonical")
    content = store.get_chunk("canonical")["content"]
    result = store.search(query_embedding=encode(content), n_results=1)
    assert result["ids"][0] == ["canonical"]
    assert result["distances"][0] == [0.0]
    assert all(len(rows) == 1 for rows in vectors(store))
    assert store.conn.execute("SELECT count(*) FROM chunks").fetchone() == (1,)


@pytest.mark.parametrize("same_id", [True, False])
def test_unchanged_duplicate_preserves_vectors(private_store, same_id):
    store = private_store
    put(store, embed_fn=encode)
    before = vectors(store)
    assert put(store, chunk_id="canonical" if same_id else "incoming")["outcome"] == "duplicate"
    assert vectors(store) == before


def test_merge_rollback_restores_content_and_vectors(private_store):
    store = private_store
    put(store, embed_fn=encode)
    before = vectors(store)
    with pytest.raises(ValueError, match="Unknown entity_id"):
        put(store, CHANGED, entity_id="missing-entity")
    assert store.get_chunk("canonical")["content"] == BASE.strip()
    assert vectors(store) == before
    assert not store.conn.in_transaction


def test_postcommit_encoder_failure_leaves_merge_repairable(private_store):
    store = private_store
    put(store, embed_fn=encode)

    def failed_canonical(text):
        if text.startswith("[Merged duplicate originals]"):
            raise RuntimeError("canonical encode failed")
        return encode(text)

    with pytest.raises(RuntimeError, match="canonical encode failed"):
        put(store, CHANGED, embed_fn=failed_canonical)
    assert "canarymerge" in store.get_chunk("canonical")["content"]
    assert vectors(store) == [[], []], "failed re-embedding retained stale vectors"
    assert embed_hot_chunk(store, encode, "canonical")


def run_drain(store, content, burn=False, embed_fn=encode):
    from brainlayer.drain import burn_drain_once, drain_once
    from brainlayer.queue_io import enqueue_store

    root = store.db_path.parent
    enqueue_store(
        content=content,
        chunk_id="incoming",
        project="merge-vectors",
        memory_type="note",
        created_at="2026-10-22T00:00:00Z",
        queue_dir=root / "queue",
    )
    worker = burn_drain_once if burn else drain_once
    return worker(
        db_path=store.db_path,
        queue_dir=root / "queue",
        log_path=root / "drain.log",
        pause_sentinel_path=root / "pause",
        embed_fn=embed_fn,
    )


@pytest.mark.parametrize("burn", [False, True])
def test_changed_drain_merge_never_publishes_incoming_vector(private_store, monkeypatch, burn):
    store = private_store
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "1")
    put(store, embed_fn=encode)
    run_drain(store, CHANGED, burn=burn)
    assert "[Merged duplicate originals]" in store.get_chunk("canonical")["content"]
    assert vectors(store) == [[], []], "drain published incoming vector for merged canonical content"
    assert embed_hot_chunk(store, encode, "canonical")


@pytest.mark.parametrize("burn", [False, True])
def test_unchanged_drain_merge_keeps_existing_vector(private_store, monkeypatch, burn):
    store = private_store
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "1")
    put(store, embed_fn=encode)
    before = vectors(store)
    run_drain(store, BASE.strip(), burn=burn, embed_fn=lambda _text: [0.0, 1.0] + [0.0] * 1022)
    assert vectors(store) == before, "unchanged drain merge overwrote the canonical vector"


def index_incoming(store, content, embedding):
    return store.upsert_chunks(
        [
            {
                "id": "incoming",
                "content": content,
                "metadata": {},
                "source_file": str(store.db_path.parent / "input.jsonl"),
                "project": "merge-vectors",
                "content_type": "note",
                "value_type": "high",
                "char_count": len(content),
                "created_at": "2026-10-22T00:00:00Z",
            }
        ],
        [embedding],
    )


def test_changed_index_merge_defers_canonical_reembedding(private_store):
    store = private_store
    put(store, embed_fn=encode)
    index_incoming(store, CHANGED, encode(CHANGED))
    assert "[Merged duplicate originals]" in store.get_chunk("canonical")["content"]
    assert vectors(store) == [[], []], "index blended vectors instead of deferring canonical re-embedding"
    assert embed_hot_chunk(store, encode, "canonical")


def test_unchanged_index_merge_keeps_existing_vector(private_store):
    store = private_store
    put(store, embed_fn=encode)
    before = vectors(store)
    index_incoming(store, BASE.strip(), [0.0, 1.0] + [0.0] * 1022)
    assert vectors(store) == before


SECOND_CHANGE = BASE + "secondmerge"
REPAIR_MODES = ["hot", "pending", "batch", "batch-fallback", "sync"]


def body_encode(text):
    """Different float AND binary vectors for originals, M1 and M2."""
    if text.startswith("[Merged duplicate originals]"):
        stage = 3 if "secondmerge" in text else 2
    else:
        stage = 1 if "canarymerge" in text else 0
    return [1.0 if index == stage else -1.0 for index in range(4)] + [0.0] * 1020


def assert_current_vectors(store, *, allow_missing=False):
    rows = vectors(store)
    if allow_missing and rows == [[], []]:
        return
    content = store.get_chunk("canonical")["content"]
    expected_float = serialize_f32(body_encode(content))
    expected_binary = store.conn.execute("SELECT vec_quantize_binary(?)", (expected_float,)).fetchone()[0]
    assert rows == [[("canonical", expected_float)], [("canonical", expected_binary)]]


@pytest.mark.parametrize("replay", ["same-id", "cross-id", "index"])
def test_original_replay_cannot_fill_merged_canonical_with_original_vector(private_store, replay):
    store = private_store
    # Cross-ID detection rejects differing numeric token sets. Include the
    # merged list's numbering in both originals so the real detector admits it.
    base = BASE if replay == "same-id" else BASE + "1 2"
    changed = CHANGED if replay == "same-id" else CHANGED + " 1 2"
    put(store, base, embed_fn=body_encode)
    put(store, changed)
    merged = store.get_chunk("canonical")["content"]
    assert merged.startswith("[Merged duplicate originals]")
    assert vectors(store) == [[], []]

    if replay != "same-id":
        from brainlayer.dedupe import find_duplicate

        hit, _ = find_duplicate(
            store.conn,
            chunk_id="incoming",
            content=changed,
            created_at="2026-10-22T00:00:00Z",
            project="merge-vectors",
            content_type="note",
        )
        assert hit is not None and hit.canonical_chunk_id == "canonical"
    if replay == "index":
        index_incoming(store, changed, body_encode(changed))
    else:
        result = put(
            store, changed, chunk_id="canonical" if replay == "same-id" else "incoming", embed_fn=body_encode
        )
        assert result["id"] == "canonical"
        if replay == "same-id":
            assert result["outcome"] == "duplicate"
    assert store.conn.execute("SELECT count(*) FROM chunks").fetchone() == (1,)
    assert store.get_chunk("canonical")["content"] == merged
    assert_current_vectors(store, allow_missing=True)
    if vectors(store) == [[], []]:
        assert embed_hot_chunk(store, body_encode, "canonical")
    assert_current_vectors(store)


def run_repair(store, mode, encoder):
    if mode == "sync":
        return put(store, CHANGED, embed_fn=encoder)
    if mode == "hot":
        return embed_hot_chunk(store, encoder, "canonical")

    def batch(texts):
        if mode == "batch-fallback":
            raise RuntimeError("deliberate batch encoder failure")
        return [encoder(text) for text in texts]

    return embed_pending_chunks(store, encoder, embed_batch_fn=batch if mode.startswith("batch") else None)


@pytest.mark.parametrize("mode", REPAIR_MODES)
@pytest.mark.parametrize("interleave", ["merge", "repaired", "archived_at", "superseded_by", "aggregated_into"])
def test_repair_rechecks_snapshot_and_eligibility_after_encoding(private_store, mode, interleave):
    store = private_store
    put(store, embed_fn=body_encode)
    if mode != "sync":
        put(store, CHANGED)
    entered, release = Event(), Event()
    encoded = []

    def paused_encoder(text):
        if text.startswith("[Merged duplicate originals]"):
            assert not store.conn.in_transaction, "encoding acquired the writer lock"
            encoded.append(text)
            entered.set()
            assert release.wait(10), "encoder barrier was not released"
        return body_encode(text)

    # Open the second connection before the encoder pauses, so schema setup is
    # never confused with the writer whose commit must beat vector publication.
    with VectorStore(store.db_path) as writer, ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(run_repair, store, mode, paused_encoder)
        try:
            assert entered.wait(10), "repair did not reach the M1 encoder barrier"
            m1 = writer.get_chunk("canonical")["content"]
            assert encoded == [m1]
            assert m1.startswith("[Merged duplicate originals]")
            if interleave in {"merge", "repaired"}:
                put(writer, SECOND_CHANGE)
                m2 = writer.get_chunk("canonical")["content"]
                assert m2 != m1 and "secondmerge" in m2
                assert body_encode(m2) != body_encode(m1) != body_encode(CHANGED)
                if interleave == "repaired":
                    assert embed_hot_chunk(writer, body_encode, "canonical")
            else:
                writer.conn.execute(
                    f"UPDATE chunks SET {interleave} = ? WHERE id = ?",
                    ("2026-10-23T00:00:00Z" if interleave == "archived_at" else "successor", "canonical"),
                )
            committed_vectors = vectors(writer)
        finally:
            release.set()
        result = pending.result(timeout=10)

    assert vectors(store) == committed_vectors, "stale repair published or overwrote a vector"
    assert not store.conn.in_transaction
    if mode != "sync":
        assert result == 0, "skipped publication must not count as repaired"
    if interleave == "merge":
        assert embed_hot_chunk(store, body_encode, "canonical")
        assert_current_vectors(store)
        assert list(store.conn.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'secondmerge'")) == [
            ("canonical",)
        ]
    elif interleave == "repaired":
        assert_current_vectors(store)
    else:
        assert not embed_hot_chunk(store, body_encode, "canonical")


@pytest.mark.parametrize("mode", REPAIR_MODES)
def test_repair_preserves_concurrent_vector_for_same_body(private_store, mode):
    store = private_store
    put(store, embed_fn=body_encode)
    if mode != "sync":
        put(store, CHANGED)
    entered, release = Event(), Event()

    def paused_encoder(text):
        if text.startswith("[Merged duplicate originals]"):
            assert not store.conn.in_transaction
            entered.set()
            assert release.wait(10)
            # Distinguish this repair from the concurrent valid encoder even
            # though both encoded the same body (e.g. model version changed).
            return [-value for value in body_encode(text)]
        return body_encode(text)

    with VectorStore(store.db_path) as writer, ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(run_repair, store, mode, paused_encoder)
        try:
            assert entered.wait(10)
            assert embed_hot_chunk(writer, body_encode, "canonical")
            committed_vectors = vectors(writer)
        finally:
            release.set()
        result = pending.result(timeout=10)
    assert vectors(store) == committed_vectors
    assert_current_vectors(store)
    if mode != "sync":
        assert result == 0


@pytest.mark.parametrize("mode", REPAIR_MODES)
def test_failed_repair_rolls_back_both_vectors(private_store, monkeypatch, mode):
    store = private_store
    put(store, embed_fn=body_encode)
    if mode != "sync":
        put(store, CHANGED)
    original = store._upsert_chunk_vector

    def failed_write(cursor, chunk_id, embedding):
        original(cursor, chunk_id, embedding)
        if store.conn.execute("SELECT content FROM chunks WHERE id = ?", (chunk_id,)).fetchone()[0].startswith(
            "[Merged duplicate originals]"
        ):
            raise RuntimeError("vector publication failure")

    monkeypatch.setattr(store, "_upsert_chunk_vector", failed_write)
    if mode == "sync":
        with pytest.raises(RuntimeError, match="vector publication failure"):
            run_repair(store, mode, body_encode)
    else:
        assert run_repair(store, mode, body_encode) == 0
    assert vectors(store) == [[], []], "failed repair leaked a committed vector"
    assert not store.conn.in_transaction


@pytest.mark.parametrize("mode", REPAIR_MODES)
def test_repair_publishes_vector_pair_under_exclusive_transaction(private_store, monkeypatch, mode):
    store = private_store
    put(store, embed_fn=body_encode)
    if mode != "sync":
        put(store, CHANGED)
    original = store._upsert_chunk_vector
    publication_locks = []

    with VectorStore(store.db_path) as contender:
        contender.conn.setbusytimeout(0)

        def checked_write(cursor, chunk_id, embedding):
            publication_locks.append(store.conn.in_transaction)
            try:
                contender.conn.execute("BEGIN IMMEDIATE")
            except apsw.BusyError:
                pass
            else:
                contender.conn.execute("ROLLBACK")
                raise AssertionError("another writer entered during float/binary publication")
            original(cursor, chunk_id, embedding)

        def unlocked_encoder(text):
            assert not store.conn.in_transaction
            return body_encode(text)

        monkeypatch.setattr(store, "_upsert_chunk_vector", checked_write)
        result = run_repair(store, mode, unlocked_encoder)
    assert publication_locks == [True]
    if mode != "sync":
        assert result == 1
    assert_current_vectors(store)
