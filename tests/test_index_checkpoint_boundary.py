"""Prove adapter completion at real SQLite transaction boundaries."""

import json
from dataclasses import replace

from typer.testing import CliRunner

import brainlayer.cli as cli
import brainlayer.index_new as adapter
import brainlayer.runtime_store as runtime
from brainlayer.embeddings import EmbeddedChunk
from tests.test_index_completion import calendar_fixture as calendar_fixture
from tests.test_index_completion import readback


def checkpoint(db):
    return json.loads(next(db.parent.glob("index-checkpoint-*.json")).read_text())["files"]


def synthetic_vectors(chunks, **kw):
    return [EmbeddedChunk(c, [0.1] * 1024) for c in chunks]


def cap_after_commit(monkeypatch, clock):
    real_open = runtime.open_writer_store

    def traced(path, *, on_connection=None):
        def attach(conn):
            if on_connection:
                on_connection(conn)

            def profile(sql, elapsed):
                if sql == "COMMIT":
                    clock[0] = 121.0

            conn.setprofile(profile)

        return real_open(path, on_connection=attach)

    monkeypatch.setattr(runtime, "open_writer_store", traced)


def test_real_postcommit_cap_advances_proven_batches(calendar_fixture, monkeypatch):
    source, db, chunks, clock = calendar_fixture
    chunks.extend(replace(chunks[0], content=f"Synthetic extended input {i}") for i in range(66, 130))
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    cap_after_commit(monkeypatch, clock)
    for count in (64, 128, 130):
        clock[0] = 100.0
        result = CliRunner().invoke(cli.app, ["index", str(source)])
        assert result.exit_code == 0 and "INDEX_CHECKPOINTED" in result.output, result.output
        assert next(iter(checkpoint(db).values()))["next_chunk"] == count
        readback(db, chunks[:count])


def test_incomplete_transaction_batch_replays_conservatively(calendar_fixture, monkeypatch):
    source, db, chunks, clock = calendar_fixture
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    with monkeypatch.context() as partial:
        partial.setenv("BRAINLAYER_INDEX_TXN_BATCH", "16")
        cap_after_commit(partial, clock)
        result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 0, result.output
    assert checkpoint(db) == {}
    readback(db, chunks[:16])
    clock[0] = 100.0
    assert CliRunner().invoke(cli.app, ["index", str(source)]).exit_code == 0
    readback(db, chunks)


def test_intentional_store_admission_is_not_embedding_loss(calendar_fixture, monkeypatch):
    source, db, chunks, clock = calendar_fixture
    chunks[:] = chunks[:3]
    # Recursive output is intentionally omitted by the real store admission layer.
    chunks[1] = replace(
        chunks[1],
        content="MCP BrainLayer Memory: Invalid JSON-RPC message: "
        '{"jsonrpc":"2.0","id":24,"result":{"content":[{"type":"text","text":"recursive output"}]}}',
    )
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    cap_after_commit(monkeypatch, clock)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 0, result.output
    assert next(iter(checkpoint(db).values()))["next_chunk"] == 3
    with runtime.ReadonlyStore(db) as store:
        assert list(store.conn.execute("SELECT position FROM chunks ORDER BY position")) == [(0,), (2,)]
        assert store.conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_checkpoint_io_after_real_commit_remains_failure(calendar_fixture, monkeypatch):
    source, db, chunks, clock = calendar_fixture
    from brainlayer.drive_credentials import _atomic_write

    writes = []

    def fail(path, payload):
        writes.append(payload)
        if len(writes) == 2:
            raise OSError("synthetic checkpoint ENOSPC after COMMIT")
        return _atomic_write(path, payload)

    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    with monkeypatch.context() as fault:
        cap_after_commit(fault, clock)
        fault.setattr("brainlayer.drive_credentials._atomic_write", fail)
        result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 1 and "synthetic checkpoint ENOSPC" in result.output, result.output
    assert "INDEX_CHECKPOINTED" not in result.output and checkpoint(db) == {}
    readback(db, chunks[:64])
    clock[0] = 100.0
    assert CliRunner().invoke(cli.app, ["index", str(source)]).exit_code == 0
    readback(db, chunks)
