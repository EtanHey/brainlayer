"""Calendar passes resume durable work and leave the maintenance window free."""

from datetime import datetime
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import brainlayer.cli as cli
from brainlayer.vector_store import IndexDeadlineExceeded


def prepare(tmp_path, monkeypatch, *, chunks=1):
    source = tmp_path / "projects"
    source.mkdir()
    (source / "a.jsonl").write_text("{}\n")
    db = tmp_path / "brainlayer.db"
    db.touch()
    monkeypatch.setattr(cli, "get_db_path", lambda: db)
    monkeypatch.setenv("BRAINLAYER_LAUNCHD_SERVICE", "index")
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", "7")
    clock = [100.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr("brainlayer.pipeline.extract.parse_jsonl", lambda path: [{}])
    monkeypatch.setattr("brainlayer.pipeline.classify.classify_content", lambda entry: entry)
    monkeypatch.setattr(
        "brainlayer.pipeline.chunk.chunk_content",
        lambda entry: [SimpleNamespace(content=f"fixture {i}", metadata={}) for i in range(chunks)],
    )

    class Store:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", lambda *a, **kw: Store())
    alarms = []
    monkeypatch.setattr("brainlayer.alarm.emit_alarm", lambda alarm: alarms.append(alarm))
    return source, clock, alarms


def test_calendar_cap_resumes_completed_file(tmp_path, monkeypatch):
    source, clock, alarms = prepare(tmp_path, monkeypatch)
    (source / "b.jsonl").write_text("{}\n")
    seen = []

    def index(chunks, **kwargs):
        seen.append(kwargs["source_file"])
        clock[0] = 108.0
        return len(chunks)

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", index)
    first = CliRunner().invoke(cli.app, ["index", str(source)])
    assert first.exit_code == 0, first.output
    assert "INDEX_CHECKPOINTED" in first.output
    assert "committed_chunks=1" in first.output
    assert not alarms
    clock[0] = 100.0
    second = CliRunner().invoke(cli.app, ["index", str(source)])
    assert second.exit_code == 0, second.output
    assert [p.rsplit("/", 1)[-1] for p in seen] == ["a.jsonl", "b.jsonl"]


def test_calendar_cap_resumes_inside_large_file(tmp_path, monkeypatch):
    source, clock, _ = prepare(tmp_path, monkeypatch, chunks=130)
    batches = []

    def index(chunks, **kwargs):
        batches.append((len(chunks), kwargs.get("start_position", 0)))
        clock[0] = 108.0
        return len(chunks)

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", index)
    for _ in range(3):
        clock[0] = 100.0
        result = CliRunner().invoke(cli.app, ["index", str(source)])
        assert result.exit_code == 0, result.output
    assert batches == [(64, 0), (64, 64), (2, 128)]


@pytest.mark.parametrize("failure", [IndexDeadlineExceeded(0), RuntimeError("disk failure")])
def test_calendar_zero_progress_and_real_errors_still_fail(tmp_path, monkeypatch, failure):
    source, _, _ = prepare(tmp_path, monkeypatch)

    def index(*args, **kwargs):
        raise failure

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", index)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 1
    assert "INDEX_CHECKPOINTED" not in result.output


def test_calendar_real_error_after_progress_is_failure(tmp_path, monkeypatch):
    source, clock, _ = prepare(tmp_path, monkeypatch)
    (source / "b.jsonl").write_text("{}\n")

    def index(chunks, **kwargs):
        if kwargs["source_file"].endswith("b.jsonl"):
            clock[0] = 108.0
            raise RuntimeError("disk failure")
        return len(chunks)

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", index)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 1
    assert "disk failure" in result.output
    assert "INDEX_CHECKPOINTED" not in result.output


@pytest.mark.parametrize("force", [False, True])
def test_checkpoint_replays_changed_or_forced_file(tmp_path, monkeypatch, force):
    source, _, _ = prepare(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", lambda chunks, **kw: seen.append(kw) or 1)
    assert CliRunner().invoke(cli.app, ["index", str(source)]).exit_code == 0
    if not force:
        (source / "a.jsonl").write_text("{}\n{}\n")
    args = ["index", str(source)] + (["--force"] if force else [])
    assert CliRunner().invoke(cli.app, args).exit_code == 0
    assert len(seen) == 2


@pytest.mark.parametrize(
    "now,expected",
    [
        (datetime(2026, 10, 18, 3, 15), 1800),
        (datetime(2026, 10, 18, 3, 50), 540),
        (datetime(2026, 10, 18, 4, 0), 0),
        (datetime(2026, 10, 18, 5, 59), 0),
        (datetime(2026, 10, 18, 6, 0), 1800),
    ],
)
def test_calendar_budget_cannot_use_maintenance_window(monkeypatch, now, expected):
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", "14400")
    assert cli._calendar_index_budget_s(now) == expected


def test_calendar_window_defers_before_opening_store(tmp_path, monkeypatch):
    source, _, _ = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_calendar_index_budget_s", lambda: 0)

    def forbidden(*a, **kw):
        pytest.fail("window deferral must not open writer")

    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", forbidden)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 75
    assert "INDEX_DEFERRED maintenance_window" in result.output


def test_partial_batch_replays_without_skipping_inputs(tmp_path, monkeypatch):
    source, clock, _ = prepare(tmp_path, monkeypatch, chunks=66)
    calls = []

    def index(chunks, **kwargs):
        calls.append((len(chunks), kwargs["start_position"]))
        if len(calls) == 2:
            raise IndexDeadlineExceeded(1)
        return len(chunks)

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", index)
    for _ in range(2):
        clock[0] = 100.0
        result = CliRunner().invoke(cli.app, ["index", str(source)])
        assert result.exit_code == 0, result.output
    assert calls == [(64, 0), (2, 64), (2, 64)]


def test_checkpoint_write_failure_is_not_clean_yield(tmp_path, monkeypatch):
    source, clock, _ = prepare(tmp_path, monkeypatch)

    def index(chunks, **kwargs):
        clock[0] = 108.0
        return len(chunks)

    def fail(*a, **kw):
        raise OSError("checkpoint disk full")

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", index)
    monkeypatch.setattr("brainlayer.drive_credentials._atomic_write", fail)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 1
    assert "checkpoint disk full" in result.output
    assert "INDEX_CHECKPOINTED" not in result.output


def test_adapter_preserves_ids_and_positions_across_calendar_batches(tmp_path, monkeypatch):
    import brainlayer.index_new as adapter
    from brainlayer.embeddings import EmbeddedChunk
    from brainlayer.pipeline.chunk import Chunk
    from brainlayer.pipeline.classify import ContentType, ContentValue
    from brainlayer.vector_store import VectorStore

    db = tmp_path / "fixture.db"
    source = tmp_path / "fixture.jsonl"
    source.write_text("{}\n")
    chunks = [
        Chunk(
            f"Synthetic calendar identity fixture number {i}",
            ContentType.USER_MESSAGE,
            ContentValue.HIGH,
            {"allow_duplicate": True},
            44,
        )
        for i in range(3)
    ]
    monkeypatch.setattr(
        adapter, "embed_chunks", lambda chunks, **kw: [EmbeddedChunk(chunk=c, embedding=[0.1] * 1024) for c in chunks]
    )
    with VectorStore(db) as store:
        for start in range(3):
            assert (
                adapter.index_chunks_to_sqlite(
                    chunks[start : start + 1], source_file=str(source), store=store, start_position=start
                )
                == 1
            )
        assert list(store.conn.execute("SELECT id, position FROM chunks ORDER BY position")) == [
            (f"{source}:{i}", i) for i in range(3)
        ]
        assert store.conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
