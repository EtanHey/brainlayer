from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_hotlane_module():
    importlib.invalidate_caches()
    sys.modules.pop("scripts.hotlane_brainbar_daemon", None)
    return importlib.import_module("scripts.hotlane_brainbar_daemon")


def _raise_if_called(message: str):
    def inner(**_kwargs):
        raise AssertionError(message)

    return inner


def test_hotlane_cycle_runs_enrichment_through_same_writer_store():
    hotlane = _load_hotlane_module()
    writer_store = object()
    calls = []

    result = hotlane.run_cycle(
        store=writer_store,
        embed_fn=lambda _text: [0.0],
        recent_limit=5,
        backlog_batch=0,
        enrich_limit=25,
        enrich_since_hours=8760,
        candidate_chunk_ids_fn=lambda store, *, limit: calls.append(("candidates", store, limit)) or [],
        hot_embed_fn=_raise_if_called("no hot candidates"),
        pending_embed_fn=_raise_if_called("backlog disabled"),
        enrich_fn=lambda store, **kwargs: (
            calls.append(("enrich", store, kwargs)) or SimpleNamespace(attempted=2, enriched=1, skipped=0, failed=0)
        ),
    )

    assert result.embedded == 0
    assert result.enrich_attempted == 2
    assert result.enriched == 1
    assert calls == [
        ("candidates", writer_store, 5),
        ("enrich", writer_store, {"limit": 25, "since_hours": 8760}),
    ]


def test_hotlane_cycle_can_disable_enrichment():
    hotlane = _load_hotlane_module()

    result = hotlane.run_cycle(
        store=object(),
        embed_fn=lambda _text: [0.0],
        recent_limit=5,
        backlog_batch=0,
        enrich_limit=0,
        enrich_since_hours=8760,
        candidate_chunk_ids_fn=lambda _store, *, limit: [],
        hot_embed_fn=lambda **_kwargs: False,
        pending_embed_fn=lambda **_kwargs: 0,
        enrich_fn=_raise_if_called("enrichment disabled"),
    )

    assert result.enrich_attempted == 0
    assert result.enriched == 0


def test_hotlane_default_backlog_batch_drains_pending_embeddings():
    hotlane = _load_hotlane_module()

    assert hotlane.DEFAULT_BACKLOG_BATCH == 4


def test_hotlane_run_threads_model_batch_embedder_to_backlog_cycle():
    hotlane = _load_hotlane_module()
    received_batch_fns = []

    class FakeStore:
        def close(self):
            pass

    class FakeModel:
        def embed_query(self, _text):
            return [0.0]

        def embed_texts(self, texts):
            return [[0.0] * 1024 for _text in texts]

    def fake_cycle(**kwargs):
        received_batch_fns.append(kwargs.get("embed_batch_fn"))
        return hotlane.CycleResult()

    hotlane.run(
        db_path=Path("/tmp/unused.db"),
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=0,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=FakeModel,
        cycle_fn=fake_cycle,
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([0.0, 100.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=1,
    )

    assert len(received_batch_fns) == 1
    assert received_batch_fns[0].__func__ is FakeModel.embed_texts


def test_hotlane_run_uses_document_embeddings_for_stored_chunks():
    hotlane = _load_hotlane_module()
    received_embed_fns = []

    class FakeStore:
        def close(self):
            pass

    class FakeModel:
        def embed_query(self, _text):
            return [1.0]

        def embed_texts(self, texts):
            return [[2.0] for _text in texts]

    def fake_cycle(**kwargs):
        received_embed_fns.append(kwargs["embed_fn"])
        return hotlane.CycleResult()

    hotlane.run(
        db_path=Path("/tmp/unused.db"),
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=0,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=FakeModel,
        cycle_fn=fake_cycle,
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([0.0, 100.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=1,
    )

    assert len(received_embed_fns) == 1
    assert received_embed_fns[0]("stored chunk text") == [2.0]


def test_hotlane_run_schedules_backlog_on_first_cycle():
    hotlane = _load_hotlane_module()
    scheduled_backlog_batches = []

    class FakeStore:
        def close(self):
            pass

    def fake_cycle(**kwargs):
        scheduled_backlog_batches.append(kwargs["backlog_batch"])
        return hotlane.CycleResult()

    hotlane.run(
        db_path=Path("/tmp/unused.db"),
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=0,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=fake_cycle,
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([100.0, 100.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=1,
    )

    assert scheduled_backlog_batches == [hotlane.DEFAULT_BACKLOG_BATCH]


def test_open_store_readonly_accepts_one_argument_factory():
    hotlane = _load_hotlane_module()
    opened_paths = []
    store = object()

    def one_argument_factory(path):
        opened_paths.append(path)
        return store

    assert hotlane._open_store(one_argument_factory, Path("/tmp/brainlayer.db"), readonly=True) is store
    assert opened_paths == [Path("/tmp/brainlayer.db")]


def test_open_store_readonly_does_not_swallow_constructor_type_errors():
    hotlane = _load_hotlane_module()

    class BrokenStore:
        def __init__(self, _path, *, readonly=False):
            raise TypeError("constructor bug unrelated to readonly")

    with pytest.raises(TypeError, match="constructor bug unrelated to readonly"):
        hotlane._open_store(BrokenStore, Path("/tmp/brainlayer.db"), readonly=True)


def test_split_cycle_bootstraps_missing_db_before_readonly_open(tmp_path):
    hotlane = _load_hotlane_module()
    db_path = tmp_path / "missing-brainlayer.db"
    opened_modes = []

    class FakeStore:
        def __init__(self, path, *, readonly=False):
            opened_modes.append(readonly)
            if readonly and not path.exists():
                raise RuntimeError("readonly sqlite open cannot create the database")
            if not readonly:
                path.touch()

        def close(self):
            pass

    result = hotlane._run_split_cycle(
        db_path=db_path,
        vector_store_cls=FakeStore,
        embed_fn=lambda _text: [0.0],
        recent_limit=1,
        backlog_batch=0,
        enrich_limit=0,
        enrich_since_hours=8760,
        candidate_rows_fn=lambda _store, *, limit: [],
        pending_rows_fn=lambda _store, *, limit: [],
    )

    assert result.embedded == 0
    assert opened_modes == [False, True]


def test_write_embedded_vectors_skips_when_content_changed_after_snapshot():
    hotlane = _load_hotlane_module()
    events = []

    class FakeCursor:
        def execute(self, sql, params=()):
            events.append(("sql", sql.strip().splitlines()[0], params))
            if sql.strip().startswith("SELECT 1"):
                assert params == ("chunk-1", "old content")
                return SimpleNamespace(fetchone=lambda: None)
            return []

    class FakeConn:
        def cursor(self):
            return FakeCursor()

    class FakeStore:
        db_path = Path("/tmp/brainlayer.db")
        conn = FakeConn()

        def _upsert_chunk_vector(self, _cursor, chunk_id, embedding):
            events.append(("upsert", chunk_id, embedding))

    count = hotlane._write_embedded_vectors(
        FakeStore(),
        [hotlane.EmbeddedVector("chunk-1", "old content", [0.5])],
    )

    assert count == 0
    assert ("upsert", "chunk-1", [0.5]) not in events


def test_split_cycle_embeds_all_hot_candidates_before_writer_revalidation(tmp_path):
    hotlane = _load_hotlane_module()
    db_path = tmp_path / "brainlayer.db"
    db_path.touch()
    vectors_seen = []

    class FakeStore:
        def close(self):
            pass

    def write_vectors_fn(_store, vectors):
        vectors_seen.extend(vectors)
        return sum(1 for vector in vectors if vector.chunk_id == "hot-fresh")

    result = hotlane._run_split_cycle(
        db_path=db_path,
        vector_store_cls=lambda _path, readonly=False: FakeStore(),
        embed_fn=lambda text: [float(len(text))],
        recent_limit=2,
        backlog_batch=0,
        enrich_limit=0,
        enrich_since_hours=8760,
        candidate_rows_fn=lambda _store, *, limit: [
            hotlane.EmbedCandidate("hot-stale", "stale content"),
            hotlane.EmbedCandidate("hot-fresh", "fresh content"),
        ][:limit],
        pending_rows_fn=lambda _store, *, limit: [],
        write_vectors_fn=write_vectors_fn,
    )

    assert [vector.chunk_id for vector in vectors_seen] == ["hot-stale", "hot-fresh"]
    assert result.embedded == 1


def test_hotlane_run_advances_enrich_timer_before_failed_cycle():
    hotlane = _load_hotlane_module()
    scheduled_enrich_limits = []

    class FakeStore:
        def close(self):
            pass

    def fake_cycle(**kwargs):
        scheduled_enrich_limits.append(kwargs["enrich_limit"])
        if len(scheduled_enrich_limits) == 1:
            raise RuntimeError("gemini transient failure")
        return hotlane.CycleResult()

    hotlane.run(
        db_path=Path("/tmp/unused.db"),
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=0,
        enrich_interval=10.0,
        enrich_limit=25,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=fake_cycle,
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([0.0, 100.0, 101.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=2,
    )

    assert scheduled_enrich_limits == [25, 0]


def test_hotlane_run_disables_enrichment_after_daily_cap():
    hotlane = _load_hotlane_module()
    scheduled_enrich_limits = []

    class FakeStore:
        def close(self):
            pass

    def fake_cycle(**kwargs):
        scheduled_enrich_limits.append(kwargs["enrich_limit"])
        if len(scheduled_enrich_limits) == 1:
            return hotlane.CycleResult(enrich_attempted=1, enrich_failed=1, enrich_daily_cap_reached=True)
        return hotlane.CycleResult()

    hotlane.run(
        db_path=Path("/tmp/unused.db"),
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=0,
        enrich_interval=10.0,
        enrich_limit=25,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=fake_cycle,
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([0.0, 100.0, 111.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=2,
    )

    assert scheduled_enrich_limits == [25, 0]


def test_hotlane_run_opens_and_closes_writer_store_each_cycle(tmp_path):
    hotlane = _load_hotlane_module()
    events = []

    class FakeStore:
        def __init__(self, path):
            events.append(("open", path))

        def close(self):
            events.append(("close", None))

    def fake_cycle(**_kwargs):
        return hotlane.CycleResult()

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=0,
        enrich_interval=10.0,
        enrich_limit=0,
        enrich_since_hours=8760,
        vector_store_cls=FakeStore,
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=fake_cycle,
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([0.0, 100.0, 101.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=2,
    )

    assert [event[0] for event in events] == ["open", "close", "open", "close"]


def test_hotlane_run_yields_all_writer_work_during_enrichment_queue_backlog(tmp_path):
    hotlane = _load_hotlane_module()
    opened = []
    cycle_calls = []
    sleeps = []

    class FakeStore:
        def __init__(self, path):
            opened.append(path)

        def close(self):
            pass

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=hotlane.DEFAULT_HOTLANE_ENRICH_LIMIT,
        enrich_since_hours=8760,
        vector_store_cls=FakeStore,
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=lambda **kwargs: cycle_calls.append(kwargs) or hotlane.CycleResult(),
        time_fn=iter([0.0, 100.0]).__next__,
        sleep_fn=sleeps.append,
        max_cycles=1,
        queue_depth_fn=lambda _queue_dir: 3,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
    )

    assert opened == []
    assert cycle_calls == []
    assert sleeps == [0.25]


def test_hotlane_run_yields_all_writer_work_during_high_priority_queue_backlog(tmp_path):
    hotlane = _load_hotlane_module()
    opened = []
    cycle_calls = []
    sleeps = []

    class FakeStore:
        def __init__(self, path):
            opened.append(path)

        def close(self):
            pass

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=hotlane.DEFAULT_HOTLANE_ENRICH_LIMIT,
        enrich_since_hours=8760,
        vector_store_cls=FakeStore,
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=lambda **kwargs: cycle_calls.append(kwargs) or hotlane.CycleResult(),
        time_fn=iter([0.0, 100.0]).__next__,
        sleep_fn=sleeps.append,
        max_cycles=1,
        queue_depth_fn=lambda _queue_dir: 3,
        high_priority_queue_depth_fn=lambda _queue_dir: 1,
    )

    assert opened == []
    assert cycle_calls == []
    assert sleeps == [0.25]


def test_hotlane_run_keeps_hot_embedding_during_queue_backlog(tmp_path, monkeypatch):
    hotlane = _load_hotlane_module()
    split_calls = []
    sleeps = []

    def fake_split_cycle(**kwargs):
        split_calls.append(kwargs)
        return hotlane.CycleResult(embedded=1)

    monkeypatch.setattr(hotlane, "_run_split_cycle", fake_split_cycle)

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=hotlane.DEFAULT_BACKLOG_BATCH,
        enrich_interval=10.0,
        enrich_limit=hotlane.DEFAULT_HOTLANE_ENRICH_LIMIT,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path, readonly=False: None,
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        time_fn=iter([0.0, 100.0]).__next__,
        sleep_fn=sleeps.append,
        max_cycles=1,
        queue_depth_fn=lambda _queue_dir: 3,
        high_priority_queue_depth_fn=lambda _queue_dir: 1,
    )

    assert len(split_calls) == 1
    assert split_calls[0]["recent_limit"] == 5
    assert split_calls[0]["backlog_batch"] == 0
    assert split_calls[0]["enrich_limit"] == 0
    assert sleeps == [0.25]


def test_hotlane_run_caps_backlog_batch_at_priority_gate_limit(tmp_path):
    hotlane = _load_hotlane_module()
    scheduled_backlog_batches = []

    class FakeStore:
        def close(self):
            pass

    hotlane.run(
        db_path=tmp_path / "brainlayer.db",
        interval=0.25,
        recent_limit=5,
        backlog_interval=10.0,
        backlog_batch=128,
        enrich_interval=10.0,
        enrich_limit=0,
        enrich_since_hours=8760,
        vector_store_cls=lambda _path: FakeStore(),
        model_factory=lambda: SimpleNamespace(embed_query=lambda _text: [0.0]),
        cycle_fn=lambda **kwargs: scheduled_backlog_batches.append(kwargs["backlog_batch"]) or hotlane.CycleResult(),
        queue_depth_fn=lambda _queue_dir: 0,
        high_priority_queue_depth_fn=lambda _queue_dir: 0,
        time_fn=iter([100.0, 100.0]).__next__,
        sleep_fn=lambda _seconds: None,
        max_cycles=1,
    )

    assert scheduled_backlog_batches == [16]


def test_hotlane_split_cycle_embeds_before_opening_writer(tmp_path):
    hotlane = _load_hotlane_module()
    events = []
    db_path = tmp_path / "brainlayer.db"
    db_path.touch()

    class FakeCursor:
        def __init__(self, readonly):
            self.readonly = readonly

        def execute(self, sql, params=()):
            if self.readonly:
                if "c.source_file = 'brainbar-store'" in sql:
                    return [("hot-1", "hot content")]
                return [("pending-1", "pending content")]
            events.append(("sql", sql.strip().splitlines()[0]))
            if sql.strip().startswith("SELECT 1"):
                return SimpleNamespace(fetchone=lambda: (1,))
            return []

    class FakeConn:
        def __init__(self, readonly):
            self.readonly = readonly

        def cursor(self):
            return FakeCursor(self.readonly)

    class FakeStore:
        def __init__(self, path, readonly=False):
            self.db_path = path
            self.conn = FakeConn(readonly)
            events.append(("open", readonly))

        def _upsert_chunk_vector(self, _cursor, chunk_id, embedding):
            events.append(("upsert", chunk_id, embedding))

        def close(self):
            events.append(("close", None))

    result = hotlane._run_split_cycle(
        db_path=db_path,
        vector_store_cls=FakeStore,
        embed_fn=lambda text: events.append(("embed", text)) or [1.0],
        embed_batch_fn=lambda texts: events.append(("embed_batch", tuple(texts))) or [[2.0] for _ in texts],
        recent_limit=5,
        backlog_batch=4,
        enrich_limit=0,
        enrich_since_hours=8760,
    )

    assert result.embedded == 2
    assert events.index(("embed", "hot content")) < events.index(("open", False))
    assert events.index(("embed", "pending content")) < events.index(("open", False))
    assert events.count(("open", False)) == 1
    assert ("upsert", "hot-1", [1.0]) in events
    assert ("upsert", "pending-1", [1.0]) in events


def test_hotlane_split_cycle_falls_through_recent_candidates_after_embed_failure(tmp_path):
    hotlane = _load_hotlane_module()
    events = []

    class FakeCursor:
        def __init__(self, readonly):
            self.readonly = readonly

        def execute(self, sql, params=()):
            if self.readonly:
                if "c.source_file = 'brainbar-store'" in sql:
                    return [("hot-bad", "bad content"), ("hot-good", "good content")]
                return []
            events.append(("sql", sql.strip().splitlines()[0]))
            if sql.strip().startswith("SELECT 1"):
                return SimpleNamespace(fetchone=lambda: (1,))
            return []

    class FakeConn:
        def __init__(self, readonly):
            self.readonly = readonly

        def cursor(self):
            return FakeCursor(self.readonly)

    class FakeStore:
        def __init__(self, path, readonly=False):
            self.db_path = path
            self.conn = FakeConn(readonly)
            events.append(("open", readonly))

        def _upsert_chunk_vector(self, _cursor, chunk_id, embedding):
            events.append(("upsert", chunk_id, embedding))

        def close(self):
            events.append(("close", None))

    def embed_fn(text):
        events.append(("embed", text))
        if text == "bad content":
            raise RuntimeError("transient embed failure")
        return [3.0]

    result = hotlane._run_split_cycle(
        db_path=tmp_path / "brainlayer.db",
        vector_store_cls=FakeStore,
        embed_fn=embed_fn,
        recent_limit=5,
        backlog_batch=0,
        enrich_limit=0,
        enrich_since_hours=8760,
    )

    assert result.embedded == 1
    assert ("embed", "bad content") in events
    assert ("embed", "good content") in events
    assert ("upsert", "hot-good", [3.0]) in events
    assert ("upsert", "hot-bad", [3.0]) not in events


def test_hotlane_split_cycle_does_not_open_writer_when_no_embedding_or_enrichment_work(tmp_path):
    hotlane = _load_hotlane_module()
    events = []
    db_path = tmp_path / "brainlayer.db"
    db_path.touch()

    class FakeCursor:
        def execute(self, _sql, _params=()):
            return []

    class FakeConn:
        def cursor(self):
            return FakeCursor()

    class FakeStore:
        def __init__(self, _path, readonly=False):
            events.append(("open", readonly))
            self.conn = FakeConn()

        def close(self):
            events.append(("close", None))

    result = hotlane._run_split_cycle(
        db_path=db_path,
        vector_store_cls=FakeStore,
        embed_fn=lambda _text: [1.0],
        recent_limit=5,
        backlog_batch=4,
        enrich_limit=0,
        enrich_since_hours=8760,
    )

    assert result == hotlane.CycleResult()
    assert events == [("open", True), ("close", None)]
