"""t3-ingest writes rows without vectors and never loads the embedding model.

The hotlane's backlog lane embeds every vectorless row, so these tests run the
real hotlane loop (with the suite's fake encoder) against the rows T3 wrote.
"""

import importlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_ingest_t3 import _create_v2_fixture

USER_ID = "t3:v2-a:v2-user:0"
ASSISTANT_ID = "t3:v2-b:v2-assistant:0"
SRC = Path(__file__).resolve().parents[1] / "src"


def _child_env():
    # A child must import THIS checkout first (an editable install can point at another
    # one), and keep the suite's own PYTHONPATH entries, such as the socket-guard
    # sitecustomize directory that tests/conftest.py adds (red team a F3).
    inherited = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join([str(SRC), inherited]) if inherited else str(SRC)
    return {**os.environ, "PYTHONPATH": pythonpath, "BRAINLAYER_FORBID_EMBEDDING_MODEL": "1"}


def _vector_ids(db_path):
    from brainlayer.vector_store import VectorStore

    with VectorStore(db_path, readonly=True) as store:
        floats = {row[0] for row in store.conn.execute("SELECT chunk_id FROM chunk_vectors")}
        binary = {row[0] for row in store.conn.execute("SELECT chunk_id FROM chunk_vectors_binary")}
    return floats, binary


def _hotlane_pass(db_path, tmp_path, monkeypatch):
    """One real hotlane loop iteration with a fake encoder (never a real model)."""
    import numpy as np

    from brainlayer.embeddings import EmbeddingModel

    # Never let the test host's load average make the backlog lane yield.
    monkeypatch.setenv("BRAINLAYER_HOTLANE_BACKLOG_MAX_LOAD", "100000")
    importlib.invalidate_caches()
    sys.modules.pop("scripts.hotlane_brainbar_daemon", None)
    hotlane = importlib.import_module("scripts.hotlane_brainbar_daemon")
    encoded = []

    class FakeEncoder:
        def encode(self, texts, **_kwargs):
            encoded.extend(texts)
            return np.ones((len(texts), 1024), dtype=np.float32)

    model = EmbeddingModel()
    model._model = FakeEncoder()
    hotlane.run(
        db_path=db_path,
        interval=1,
        recent_limit=5,
        backlog_interval=7,
        backlog_batch=16,
        enrich_interval=0,
        enrich_limit=0,
        enrich_since_hours=0,
        model_factory=lambda **_kwargs: model,
        max_cycles=1,
        sleep_fn=lambda _seconds: None,
        queue_dir=tmp_path / "queue",
    )
    return encoded


def _search_ids(db_path, text):
    from brainlayer.vector_store import VectorStore

    with VectorStore(db_path, readonly=True) as store:
        # A real query vector: the semantic arm can only see rows that have vectors.
        results = store.hybrid_search(query_embedding=[1.0] * 1024, query_text=text, n_results=10)
    return results["ids"][0]


def test_t3_run_writes_vectorless_rows_and_never_imports_the_model(tmp_path):
    """Fails on main: t3-ingest called embed_chunks, which refuses under the guard."""
    from brainlayer.vector_store import VectorStore

    state = _create_v2_fixture(tmp_path / "statev2.sqlite")
    destination = tmp_path / "destination.db"
    health = tmp_path / "health.json"
    VectorStore(destination).close()
    probe = (
        "import json, sys\n"
        "from brainlayer.ingest.t3 import ingest_t3\n"
        "result = ingest_t3(sys.argv[1], db_path=sys.argv[2], health_path=sys.argv[3], projection_version=2)\n"
        "print(json.dumps({'indexed': result.chunks_indexed, 'embed_ms': result.embed_ms,\n"
        "  'embed_deferred_to': result.embed_deferred_to,\n"
        "  'loaded': sorted(m for m in ('torch', 'sentence_transformers') if m in sys.modules)}))\n"
    )
    # A fresh process: an earlier test may have warmed an in-process model cache.
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(state), str(destination), str(health)],
        capture_output=True,
        text=True,
        timeout=60,
        env=_child_env(),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report == {"indexed": 2, "embed_ms": 0.0, "embed_deferred_to": "hotlane", "loaded": []}
    with VectorStore(destination, readonly=True) as store:
        assert {row[0] for row in store.conn.execute("SELECT id FROM chunks")} == {USER_ID, ASSISTANT_ID}
    assert _vector_ids(destination) == (set(), set())
    snapshot = json.loads(health.read_text())
    assert snapshot["alerting"] is False and snapshot["embed_ms"] == 0
    assert snapshot["embed_deferred_to"] == "hotlane" and snapshot["chunks_indexed"] == 2


def test_t3_cli_entrypoint_succeeds_under_the_model_guard(tmp_path):
    """The production command, as launchd runs it, with model loading forbidden."""
    from brainlayer.vector_store import VectorStore

    state = _create_v2_fixture(tmp_path / "state.sqlite")
    destination = tmp_path / "destination.db"
    health = tmp_path / "health.json"
    VectorStore(destination).close()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "brainlayer",
            "ingest-t3",
            "--state-db",
            str(state),
            "--projection-version",
            "2",
            "--db",
            str(destination),
            "--health-path",
            str(health),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=_child_env(),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "chunks_indexed=2" in completed.stdout
    assert json.loads(health.read_text())["alerting"] is False
    assert _vector_ids(destination) == (set(), set())


def test_new_t3_rows_are_found_by_fts_then_embedded_by_one_hotlane_pass(tmp_path, monkeypatch):
    from brainlayer.ingest.t3 import ingest_t3
    from brainlayer.vector_store import VectorStore

    state = _create_v2_fixture(tmp_path / "statev2.sqlite")
    destination = tmp_path / "destination.db"
    VectorStore(destination).close()
    assert ingest_t3(state, db_path=destination, health_path=None, projection_version=2).chunks_indexed == 2

    # Right after the run, before any vector exists, keyword search returns the row.
    assert _vector_ids(destination) == (set(), set())
    assert ASSISTANT_ID in _search_ids(destination, "settled")

    encoded = _hotlane_pass(destination, tmp_path, monkeypatch)
    assert sorted(encoded) == ["settled V2 response", "u"]
    assert _vector_ids(destination) == ({USER_ID, ASSISTANT_ID}, {USER_ID, ASSISTANT_ID})


def test_edited_t3_message_replaces_content_and_metadata_and_drops_the_stale_vector(tmp_path, monkeypatch):
    """Fails on main: the first ingest loads the model (refused under the suite guard)."""
    from brainlayer.ingest.t3 import ingest_t3
    from brainlayer.vector_store import VectorStore

    state = _create_v2_fixture(tmp_path / "statev2.sqlite")
    destination = tmp_path / "destination.db"
    VectorStore(destination).close()
    ingest_t3(state, db_path=destination, health_path=None, projection_version=2)
    _hotlane_pass(destination, tmp_path, monkeypatch)
    assert _vector_ids(destination) == ({USER_ID, ASSISTANT_ID}, {USER_ID, ASSISTANT_ID})

    with sqlite3.connect(state) as conn:
        conn.execute(
            "UPDATE orchestration_v2_projection_messages SET payload_json=? WHERE message_id='v2-user'",
            (json.dumps({"text": "edited prompt about zebras"}),),
        )
        conn.execute("UPDATE orchestration_v2_projection_threads SET title='V2 A renamed' WHERE thread_id='v2-a'")
    edited = ingest_t3(state, db_path=destination, health_path=None, projection_version=2)
    assert edited.chunks_indexed == 1 and edited.embed_ms == 0

    with VectorStore(destination, readonly=True) as store:
        rows = list(store.conn.execute("SELECT id, content, metadata, source, seen_count FROM chunks ORDER BY id"))
        audits = store.conn.execute("SELECT COUNT(*) FROM dedupe_audit").fetchone()[0]
    by_id = {row[0]: row for row in rows}
    assert set(by_id) == {USER_ID, ASSISTANT_ID}, "the edit replaces the same id, never a new row"
    assert by_id[USER_ID][1] == "edited prompt about zebras"
    assert "[Merged duplicate originals]" not in by_id[USER_ID][1]
    assert json.loads(by_id[USER_ID][2])["t3_title"] == "V2 A renamed"
    assert by_id[USER_ID][3] == "t3" and by_id[USER_ID][4] == 1 and audits == 0
    # The edited row's old vector is gone; the untouched row keeps its vector.
    assert _vector_ids(destination) == ({ASSISTANT_ID}, {ASSISTANT_ID})
    assert USER_ID in _search_ids(destination, "zebras")

    assert _hotlane_pass(destination, tmp_path, monkeypatch) == ["edited prompt about zebras"]
    assert _vector_ids(destination) == ({USER_ID, ASSISTANT_ID}, {USER_ID, ASSISTANT_ID})


def test_change_check_and_write_share_one_writer_hold(tmp_path, monkeypatch):
    """Race guard for the new order (main re-checked after releasing the writer to embed).

    With no embed step, the change check and the write run under one writer hold, so
    another indexer process cannot land these IDs in between. The probes sit in the
    WRITE: the store that writes must be the very store that ran the check, never
    closed in between, and another process must be refused at that moment. A split
    hold (release after the check, reopen for the write) fails here (red team b F1).
    """
    from brainlayer import index_new
    from brainlayer.ingest import t3
    from brainlayer.vector_store import VectorStore

    state = _create_v2_fixture(tmp_path / "statev2.sqlite")
    destination = tmp_path / "destination.db"
    VectorStore(destination).close()
    real_changed = t3._changed_chunks
    real_index = index_new.index_chunks_to_sqlite
    checked_stores = []
    closed_stores = []
    other_writer = []
    probe = (
        "import sys\n"
        "from pathlib import Path\n"
        "from brainlayer.runtime_store import open_writer_store\n"
        "from brainlayer.vector_store import WriterInUseError\n"
        "try:\n"
        "    with open_writer_store(Path(sys.argv[1])):\n"
        "        print('admitted')\n"
        "except WriterInUseError:\n"
        "    print('refused')\n"
    )

    def tracked_check(chunks, store):
        checked_stores.append(store)
        real_close = store.close

        def close(*args, **kwargs):
            closed_stores.append(store)
            return real_close(*args, **kwargs)

        store.close = close
        return real_changed(chunks, store)

    def write_while_another_process_tries_to_write(chunks, *args, store=None, **kwargs):
        assert checked_stores and store is checked_stores[-1], "the write must reuse the check's writer hold"
        assert store not in closed_stores, "the writer was released between the check and the write"
        completed = subprocess.run(
            [sys.executable, "-c", probe, str(destination)],
            capture_output=True,
            text=True,
            timeout=60,
            env=_child_env(),
        )
        other_writer.append(completed.stdout.strip() or completed.stderr)
        return real_index(chunks, *args, store=store, **kwargs)

    monkeypatch.setattr(t3, "_changed_chunks", tracked_check)
    monkeypatch.setattr(index_new, "index_chunks_to_sqlite", write_while_another_process_tries_to_write)
    result = t3.ingest_t3(state, db_path=destination, health_path=None, projection_version=2)
    assert other_writer == ["refused"]
    assert result.chunks_indexed == 2
    with VectorStore(destination, readonly=True) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
        assert store.conn.execute("SELECT COUNT(*) FROM dedupe_audit").fetchone()[0] == 0


def test_defer_embedding_refuses_precomputed_vectors():
    from brainlayer import index_new
    from brainlayer.embeddings import EmbeddedChunk
    from brainlayer.pipeline.chunk import Chunk
    from brainlayer.pipeline.classify import ContentType, ContentValue

    chunk = Chunk("text", ContentType.USER_MESSAGE, ContentValue.HIGH, {"chunk_id": "t3:x:y:0"}, 4)
    with pytest.raises(ValueError, match="mutually exclusive"):
        index_new.index_chunks_to_sqlite(
            [chunk],
            "source.sqlite",
            store=object(),
            precomputed_embeddings=[EmbeddedChunk(chunk, [0.1])],
            defer_embedding=True,
        )
