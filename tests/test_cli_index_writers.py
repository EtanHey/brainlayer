"""Calendar index uses existing bounded SQLite coordination and defers busy writers."""

import json
from contextlib import contextmanager

import apsw
import pytest
from typer.testing import CliRunner

import brainlayer.cli as cli
import brainlayer.runtime_store as runtime
import brainlayer.vector_store as storage
from tests.test_cli_index_calendar import prepare


@pytest.mark.parametrize("error", [apsw.BusyError("database is locked"), storage.WriterInUseError("resident writer")])
@pytest.mark.parametrize("calendar", [True, False])
def test_open_contention_yields_only_for_calendar(tmp_path, monkeypatch, error, calendar):
    source, _, _ = prepare(tmp_path, monkeypatch)
    if not calendar:
        monkeypatch.delenv("BRAINLAYER_LAUNCHD_SERVICE")

    def busy(*a, **kw):
        raise error

    monkeypatch.setattr(runtime, "open_writer_store", busy)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == (75 if calendar else 1), result.output
    if calendar:
        assert "INDEX_DEFERRED resident_writer_busy" in result.output


def test_calendar_opens_with_existing_bounded_busy_context(tmp_path, monkeypatch):
    source, _, _ = prepare(tmp_path, monkeypatch)
    captured = []
    original = runtime.open_writer_store

    @contextmanager
    def opening(*a, **kw):
        captured.append((storage._write_busy_timeout_ms(), storage._write_busy_deadline()))
        with original(*a, **kw) as store:
            yield store

    monkeypatch.setattr(runtime, "open_writer_store", opening)
    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", lambda *a, **kw: 1)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 0, result.output
    # The short 7s pass reserves half for finalization; busy_timeout stays 1s.
    assert captured == [(1000, 103.5)]
    assert storage._write_busy_deadline() is None


def test_zero_progress_busy_deadline_is_still_a_runtime_failure(tmp_path, monkeypatch):
    source, clock, alarms = prepare(tmp_path, monkeypatch)

    def expired_open(*args, **kwargs):
        clock[0] = 108.0
        # The existing busy budget raises BusyError once the pass deadline expires.
        storage._write_busy_timeout_ms()

    monkeypatch.setattr(runtime, "open_writer_store", expired_open)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 1, result.output
    assert [alarm.code for alarm in alarms] == ["INDEX_RUNTIME_EXCEEDED"]
    assert alarms[0].context["committed_chunks"] == 0
    assert "INDEX_DEFERRED" not in result.output
    assert storage._write_busy_deadline() is None


def test_write_contention_preserves_completed_checkpoint_for_next_pass(tmp_path, monkeypatch):
    source, _, _ = prepare(tmp_path, monkeypatch, chunks=130)
    seen = []

    def index(chunks, **kw):
        start = kw.get("start_position", 0)
        seen.append(start)
        if len(seen) == 2:
            raise apsw.BusyError("database is locked")
        return len(chunks)

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", index)
    first = CliRunner().invoke(cli.app, ["index", str(source)])
    assert first.exit_code == 75, first.output
    assert "committed_chunks=64" in first.output
    checkpoint = json.loads(next(tmp_path.glob("index-checkpoint-*.json")).read_text())
    assert next(iter(checkpoint["files"].values()))["next_chunk"] == 64
    second = CliRunner().invoke(cli.app, ["index", str(source)])
    assert second.exit_code == 0, second.output
    assert seen == [0, 64, 64, 128]


def test_real_sqlite_write_lock_defers_and_then_indexes(tmp_path, monkeypatch):
    real_open = runtime.open_writer_store
    source, _, _ = prepare(tmp_path, monkeypatch)
    db = tmp_path / "brainlayer.db"
    with storage.VectorStore(db):
        pass
    monkeypatch.setattr(runtime, "open_writer_store", real_open)
    # Keep this real lock reproduction fast, retaining upsert's real busy retries.
    monkeypatch.setattr(storage, "_write_busy_timeout_ms", lambda: 5)
    monkeypatch.setattr(runtime, "_write_busy_timeout_ms", lambda: 5)
    from brainlayer.embeddings import EmbeddedChunk
    from brainlayer.pipeline.chunk import Chunk
    from brainlayer.pipeline.classify import ContentType, ContentValue

    chunk = Chunk("Synthetic resident contention fixture", ContentType.USER_MESSAGE, ContentValue.HIGH, {}, 36)
    monkeypatch.setattr("brainlayer.pipeline.chunk.chunk_content", lambda entry: [chunk])
    monkeypatch.setattr(
        "brainlayer.index_new.embed_chunks",
        lambda chunks, **kw: [EmbeddedChunk(chunk=c, embedding=[0.1] * 1024) for c in chunks],
    )
    owner = apsw.Connection(str(db))
    try:
        owner.execute("BEGIN IMMEDIATE")
        first = CliRunner().invoke(cli.app, ["index", str(source)])
        assert first.exit_code == 75, first.output
        assert "INDEX_DEFERRED resident_writer_busy" in first.output
        assert owner.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        owner.execute("COMMIT")
        second = CliRunner().invoke(cli.app, ["index", str(source)])
        assert second.exit_code == 0, second.output
        assert owner.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
        assert owner.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        owner.close()
