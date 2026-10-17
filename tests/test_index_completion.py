"""Real embedder/adapter/writer regressions using only synthetic vectors and DBs."""

import json

import pytest
from typer.testing import CliRunner

import brainlayer.cli as cli
import brainlayer.index_new as adapter
from brainlayer.embeddings import EmbeddedChunk, EmbeddingModel
from brainlayer.pipeline.chunk import Chunk
from brainlayer.pipeline.classify import ContentType, ContentValue
from brainlayer.vector_store import VectorStore


@pytest.fixture
def calendar_fixture(tmp_path, monkeypatch):
    source = tmp_path / "projects"
    source.mkdir()
    (source / "a.jsonl").write_text("{}\n")
    db = tmp_path / "fixture.db"
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(tmp_path / "pidfiles"))
    with VectorStore(db):
        pass
    chunks = [
        Chunk(
            f"Synthetic correction input number {i}",
            ContentType.USER_MESSAGE,
            ContentValue.HIGH,
            {"allow_duplicate": True},
            36,
        )
        for i in range(66)
    ]
    monkeypatch.setattr(cli, "get_db_path", lambda: db)
    monkeypatch.setattr(cli, "_calendar_index_budget_s", lambda: 20)
    clock = [100.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("BRAINLAYER_LAUNCHD_SERVICE", "index")
    monkeypatch.setattr("brainlayer.pipeline.extract.parse_jsonl", lambda p: [{}])
    monkeypatch.setattr("brainlayer.pipeline.classify.classify_content", lambda x: x)
    monkeypatch.setattr("brainlayer.pipeline.chunk.chunk_content", lambda x: chunks)
    return source, db, chunks, clock


def readback(db, chunks):
    from brainlayer.runtime_store import ReadonlyStore

    with ReadonlyStore(db) as store:
        rows = list(store.conn.execute("SELECT id, position, content FROM chunks ORDER BY position"))
        assert store.conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    source = db.parent / "projects" / "a.jsonl"
    assert rows == [(f"{source}:{i}", i, c.content) for i, c in enumerate(chunks)]
    print(f"READBACK rows={len(rows)} ids=source:0..{len(rows) - 1} positions=contiguous integrity=ok")


class Vector:
    def tolist(self):
        return [0.1] * 1024


@pytest.mark.parametrize("failure", ["first", "second", "all", "short"])
def test_failed_embedding_is_error_and_retry_preserves_inputs(calendar_fixture, monkeypatch, failure):
    source, db, chunks, _ = calendar_fixture

    class Encoder:
        healthy = False
        calls = 0

        def encode(self, texts, **kw):
            self.calls += 1
            if not self.healthy:
                if failure == "short":
                    return [Vector() for _ in texts[:-1]]
                if failure == "all" or self.calls == (1 if failure == "first" else 2):
                    raise RuntimeError("synthetic transient encoder failure")
            return [Vector() for _ in texts]

    encoder = Encoder()
    model = EmbeddingModel()
    model._model = encoder
    monkeypatch.setattr(adapter, "embed_chunks", model.embed_chunks)
    first = CliRunner().invoke(cli.app, ["index", str(source)])
    assert first.exit_code == 1, first.output
    assert "INDEX_CHECKPOINTED" not in first.output
    checkpoint = json.loads(next(db.parent.glob("index-checkpoint-*.json")).read_text())
    assert checkpoint["files"] == {}
    readback(db, [])
    encoder.healthy = True
    before = encoder.calls
    second = CliRunner().invoke(cli.app, ["index", str(source)])
    assert second.exit_code == 0, second.output
    assert encoder.calls > before
    readback(db, chunks)
    third = CliRunner().invoke(cli.app, ["index", str(source)])
    assert third.exit_code == 0, third.output
    readback(db, chunks)


@pytest.mark.parametrize("variant", ["short", "reordered", "duplicate"])
def test_adapter_rejects_incomplete_or_misaligned_precomputed(calendar_fixture, variant):
    source, db, chunks, _ = calendar_fixture
    admitted = chunks[:3]
    indices = {"short": [1, 2], "reordered": [2, 1, 0], "duplicate": [0, 0, 2]}[variant]
    embedded = [EmbeddedChunk(admitted[i], [0.1] * 1024) for i in indices]
    with pytest.raises(ValueError, match="admitted"):
        adapter.index_chunks_to_sqlite(admitted, str(source / "a.jsonl"), db_path=db, precomputed_embeddings=embedded)
    readback(db, [])
