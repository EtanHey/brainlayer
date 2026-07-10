import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import brainlayer.cli as cli
from brainlayer.alarm import BrainLayerAlarm
from brainlayer.cli import app
from brainlayer.vector_store import IndexDeadlineExceeded


def _prepare_index_source(tmp_path, monkeypatch):
    source = tmp_path / "projects"
    project = source / "watchdog-test"
    project.mkdir(parents=True)
    (project / "session.jsonl").write_text("{}\n")

    monkeypatch.setattr("brainlayer.pipeline.extract.parse_jsonl", lambda _path: [{}])
    monkeypatch.setattr("brainlayer.pipeline.classify.classify_content", lambda entry: entry)
    monkeypatch.setattr("brainlayer.pipeline.chunk.chunk_content", lambda _entry: [object()])
    monkeypatch.setattr(cli.time, "monotonic", lambda: 100.0)
    return source


class _FakeRuntimeStore:
    def __init__(self):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True


def test_index_deadline_exits_nonzero_with_loud_alarm(tmp_path, monkeypatch):
    source = _prepare_index_source(tmp_path, monkeypatch)
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", "7")
    captured: dict[str, object] = {}
    alarms: list[BrainLayerAlarm] = []
    runtime_store = _FakeRuntimeStore()

    def fake_index(
        _chunks,
        *,
        source_file,
        project,
        on_progress,
        deadline_monotonic=None,
        store=None,
    ):
        captured["deadline"] = deadline_monotonic
        captured["store"] = store
        raise IndexDeadlineExceeded(processed_count=2)

    def fake_emit_alarm(alarm):
        alarms.append(alarm)
        print(alarm.human_message(), file=sys.stderr)
        return True

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", fake_index)
    monkeypatch.setattr("brainlayer.alarm.emit_alarm", fake_emit_alarm)
    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", lambda _path: runtime_store)

    result = CliRunner().invoke(app, ["index", str(source)])

    assert result.exit_code == 1
    assert captured["deadline"] == 107.0
    assert captured["store"] is runtime_store
    assert runtime_store.closed is True
    assert "BRAINLAYER_ALARM INDEX_RUNTIME_EXCEEDED" in result.stderr
    assert len(alarms) == 1
    assert alarms[0].context["max_runtime_s"] == 7.0
    assert alarms[0].context["committed_chunks"] == 2


def test_fast_index_under_deadline_completes_without_alarm(tmp_path, monkeypatch):
    source = _prepare_index_source(tmp_path, monkeypatch)
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", "7")
    captured: dict[str, object] = {}
    alarms: list[BrainLayerAlarm] = []
    runtime_store = _FakeRuntimeStore()

    def fake_index(
        _chunks,
        *,
        source_file,
        project,
        on_progress,
        deadline_monotonic=None,
        store=None,
    ):
        captured["deadline"] = deadline_monotonic
        captured["store"] = store
        return 3

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", fake_index)
    monkeypatch.setattr("brainlayer.alarm.emit_alarm", lambda alarm: alarms.append(alarm) or True)
    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", lambda _path: runtime_store)

    result = CliRunner().invoke(app, ["index", str(source)])

    assert result.exit_code == 0
    assert captured["deadline"] == 107.0
    assert captured["store"] is runtime_store
    assert alarms == []
    assert "Indexed 3 chunks" in result.stdout


def test_index_deadline_trips_after_file_that_produces_no_writes(tmp_path, monkeypatch):
    source = _prepare_index_source(tmp_path, monkeypatch)
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", "7")
    monotonic_values = iter([100.0, 100.0, 108.0])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(monotonic_values, 108.0))
    alarms: list[BrainLayerAlarm] = []
    runtime_store = _FakeRuntimeStore()

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("brainlayer.alarm.emit_alarm", lambda alarm: alarms.append(alarm) or True)
    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", lambda _path: runtime_store)

    result = CliRunner().invoke(app, ["index", str(source)])

    assert result.exit_code == 1
    assert len(alarms) == 1
    assert alarms[0].code == "INDEX_RUNTIME_EXCEEDED"
    assert alarms[0].context["committed_chunks"] == 0


def test_index_deadline_stops_before_parsing_next_entry(tmp_path, monkeypatch):
    source = _prepare_index_source(tmp_path, monkeypatch)
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", "7")
    monotonic_values = iter([100.0, 100.0, 108.0])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(monotonic_values, 108.0))
    index_called = False
    runtime_store = _FakeRuntimeStore()

    def fake_index(*_args, **_kwargs):
        nonlocal index_called
        index_called = True
        return 0

    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", fake_index)
    monkeypatch.setattr("brainlayer.alarm.emit_alarm", lambda _alarm: True)
    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", lambda _path: runtime_store)

    result = CliRunner().invoke(app, ["index", str(source)])

    assert result.exit_code == 1
    assert index_called is False


def test_index_adapter_stops_after_embedding_batch_deadline(tmp_path, monkeypatch):
    import brainlayer.index_new as index_new

    source_file = tmp_path / "session.jsonl"
    source_file.write_text('{"timestamp":"2026-07-10T00:00:00Z"}\n')
    chunk = SimpleNamespace(
        metadata={},
        content="Embedding deadline test content",
        content_type=SimpleNamespace(value="note"),
        value=SimpleNamespace(value="medium"),
        char_count=31,
    )

    def fake_embed(chunks, on_progress=None):
        assert on_progress is not None
        on_progress(len(chunks), len(chunks))
        return [SimpleNamespace(chunk=chunk, embedding=[0.1])]

    monkeypatch.setattr(index_new, "embed_chunks", fake_embed)
    monkeypatch.setattr(cli.time, "monotonic", lambda: 43.0)

    with pytest.raises(IndexDeadlineExceeded):
        index_new.index_chunks_to_sqlite(
            [chunk],
            source_file=str(source_file),
            db_path=tmp_path / "brainlayer.db",
            deadline_monotonic=42.0,
        )


def test_index_adapter_forwards_deadline_to_runtime_store(tmp_path, monkeypatch):
    import brainlayer.index_new as index_new

    source_file = tmp_path / "session.jsonl"
    source_file.write_text('{"timestamp":"2026-07-10T00:00:00Z"}\n')
    chunk = SimpleNamespace(
        metadata={},
        content="Deadline forwarding test content",
        content_type=SimpleNamespace(value="note"),
        value=SimpleNamespace(value="medium"),
        char_count=32,
    )
    captured: dict[str, object] = {}

    class FakeRuntimeStore:
        def __init__(self, db_path):
            captured["db_path"] = db_path

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def upsert_chunks(self, chunks, embeddings, *, deadline_monotonic=None):
            captured["deadline"] = deadline_monotonic
            return len(chunks)

    monkeypatch.setattr(
        index_new, "embed_chunks", lambda chunks, on_progress=None: [SimpleNamespace(chunk=chunk, embedding=[0.1])]
    )
    monkeypatch.setattr(index_new, "open_writer_store", FakeRuntimeStore)

    result = index_new.index_chunks_to_sqlite(
        [chunk],
        source_file=str(source_file),
        db_path=tmp_path / "brainlayer.db",
        deadline_monotonic=42.0,
    )

    assert result == 1
    assert captured["deadline"] == 42.0


def test_index_reuses_one_runtime_store_across_source_files(tmp_path, monkeypatch):
    source = _prepare_index_source(tmp_path, monkeypatch)
    project = source / "watchdog-test"
    (project / "session-two.jsonl").write_text("{}\n")
    opened: list[_FakeRuntimeStore] = []
    seen_stores: list[object] = []

    def fake_open(_path):
        store = _FakeRuntimeStore()
        opened.append(store)
        return store

    def fake_index(_chunks, **kwargs):
        seen_stores.append(kwargs["store"])
        return 1

    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", fake_open)
    monkeypatch.setattr("brainlayer.index_new.index_chunks_to_sqlite", fake_index)
    monkeypatch.setattr("brainlayer.alarm.emit_alarm", lambda _alarm: True)

    result = CliRunner().invoke(app, ["index", str(source)])

    assert result.exit_code == 0, result.output
    assert len(opened) == 1
    assert seen_stores == [opened[0], opened[0]]
    assert opened[0].closed is True
