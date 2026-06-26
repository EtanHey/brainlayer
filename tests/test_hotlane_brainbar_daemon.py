from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


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


def test_hotlane_run_yields_writer_to_enrichment_queue_backlog(tmp_path):
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


def test_hotlane_run_yields_writer_to_high_priority_queue_backlog(tmp_path):
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
