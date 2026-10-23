"""Independent #1450 repro: real store/FTS/vec paths in a private temp universe."""

from pathlib import Path

import pytest

from brainlayer.store import embed_hot_chunk, store_memory
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
