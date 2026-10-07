import logging
import os
import time
from pathlib import Path

import pytest


def _result(*, attempted: int, enriched: int = 0, skipped: int = 0, failed: int = 0, errors=None):
    from brainlayer.enrichment_controller import EnrichmentResult

    return EnrichmentResult(
        mode="realtime",
        attempted=attempted,
        enriched=enriched,
        skipped=skipped,
        failed=failed,
        errors=list(errors or []),
    )


@pytest.fixture(autouse=True)
def _isolate_live_queue(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))


def test_supervisor_propagates_writer_in_use_from_vector_store(tmp_path):
    from brainlayer import enrichment_controller as controller

    class WriterInUseError(RuntimeError):
        pass

    class FakeVectorStore:
        def __init__(self, db_path: Path):
            raise WriterInUseError("another writer is using brainlayer.db (pid 123)")

    with pytest.raises(WriterInUseError, match="another writer is using"):
        controller.run_enrich_supervisor(
            tmp_path / "brainlayer.db",
            vector_store_cls=FakeVectorStore,
            enrich_fn=lambda store, **kwargs: _result(attempted=0),
            sleep_fn=lambda seconds: None,
        )


def test_supervisor_propagates_pidfile_ref_mismatch_from_vector_store(tmp_path):
    from brainlayer import enrichment_controller as controller

    class WriterInUseError(RuntimeError):
        pass

    class FakeVectorStore:
        def __init__(self, db_path: Path):
            raise WriterInUseError("pidfile ref mismatch for brainlayer.db; refusing to clear active refs")

    with pytest.raises(WriterInUseError, match="pidfile ref mismatch"):
        controller.run_enrich_supervisor(
            tmp_path / "brainlayer.db",
            vector_store_cls=FakeVectorStore,
            enrich_fn=lambda store, **kwargs: _result(attempted=0),
            sleep_fn=lambda seconds: None,
        )


def test_supervisor_sleeps_when_queue_empty_and_polls_again(tmp_path):
    from brainlayer import enrichment_controller as controller

    class FakeVectorStore:
        def __init__(self, db_path: Path):
            pass

        def close(self):
            pass

    sleeps = []
    calls = 0

    def empty_then_work(store, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _result(attempted=0)
        return _result(attempted=1, enriched=1)

    result = controller.run_enrich_supervisor(
        tmp_path / "brainlayer.db",
        max_cycles=2,
        vector_store_cls=FakeVectorStore,
        enrich_fn=empty_then_work,
        sleep_fn=sleeps.append,
    )

    assert calls == 2
    assert sleeps == [30.0]
    assert result.cycles == 2
    assert result.enriched == 1


def test_supervisor_runs_unbounded_idle_backlog_before_sleep(tmp_path, monkeypatch):
    from brainlayer import enrichment_controller as controller

    monkeypatch.setenv("BRAINLAYER_ENRICH_IDLE_BACKLOG", "1")

    class FakeVectorStore:
        def __init__(self, db_path: Path):
            pass

        def close(self):
            pass

    calls = []

    def empty_window_then_backlog(store, **kwargs):
        calls.append(kwargs)
        if kwargs["since_hours"] is None:
            return _result(attempted=1, enriched=1)
        return _result(attempted=0)

    result = controller.run_enrich_supervisor(
        tmp_path / "brainlayer.db",
        limit=5,
        since_hours=24,
        max_cycles=1,
        vector_store_cls=FakeVectorStore,
        enrich_fn=empty_window_then_backlog,
        idle_backlog_ready_fn=lambda: True,
        sleep_fn=lambda seconds: (_ for _ in ()).throw(AssertionError("should not sleep before idle backlog")),
    )

    assert [call["since_hours"] for call in calls] == [24, None]
    assert result.cycles == 1
    assert result.attempted == 1
    assert result.enriched == 1


def test_supervisor_idle_backlog_error_does_not_exit(tmp_path, monkeypatch):
    from brainlayer import enrichment_controller as controller

    monkeypatch.setenv("BRAINLAYER_ENRICH_IDLE_BACKLOG", "1")

    class FakeVectorStore:
        def __init__(self, db_path: Path):
            pass

        def close(self):
            pass

    calls = []

    def idle_backlog_fails_once(store, **kwargs):
        calls.append(kwargs["since_hours"])
        if kwargs["since_hours"] is None and calls.count(None) == 1:
            raise RuntimeError("database is locked")
        return _result(attempted=0)

    sleeps = []
    result = controller.run_enrich_supervisor(
        tmp_path / "brainlayer.db",
        limit=5,
        since_hours=24,
        max_cycles=2,
        vector_store_cls=FakeVectorStore,
        enrich_fn=idle_backlog_fails_once,
        idle_backlog_ready_fn=lambda: True,
        sleep_fn=sleeps.append,
    )

    assert calls == [24, None, 24, None]
    assert result.cycles == 2
    assert result.failed_cycles == 1
    assert result.errors == ["supervisor-idle-backlog: database is locked"]
    assert sleeps == [30.0]


def test_idle_backlog_requires_watcher_offsets_and_health_to_be_idle(tmp_path, monkeypatch):
    from brainlayer import enrichment_controller as controller

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    offsets_path = tmp_path / "offsets.json"
    watcher_health_path = tmp_path / "watcher-health.json"
    offsets_path.write_text("{}", encoding="utf-8")
    watcher_health_path.write_text('{"poll_count": 7}', encoding="utf-8")
    monkeypatch.setenv("BRAINLAYER_ENRICH_IDLE_BACKLOG", "1")
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    monkeypatch.setenv("BRAINLAYER_WATCHER_OFFSETS_PATH", str(offsets_path))
    monkeypatch.setenv("BRAINLAYER_WATCHER_HEALTH_PATH", str(watcher_health_path))
    monkeypatch.setenv("BRAINLAYER_ENRICH_WATCHER_IDLE_SECONDS", "60")

    assert controller._idle_backlog_ready() is False

    old_mtime = time.time() - 120
    os.utime(offsets_path, (old_mtime, old_mtime))
    os.utime(watcher_health_path, (old_mtime, old_mtime))

    assert controller._idle_backlog_ready() is True


def test_supervisor_skips_idle_backlog_when_watcher_is_active(tmp_path, monkeypatch):
    from brainlayer import enrichment_controller as controller

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    offsets_path = tmp_path / "offsets.json"
    watcher_health_path = tmp_path / "watcher-health.json"
    offsets_path.write_text("{}", encoding="utf-8")
    watcher_health_path.write_text('{"poll_count": 7}', encoding="utf-8")
    monkeypatch.setenv("BRAINLAYER_ENRICH_IDLE_BACKLOG", "1")
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    monkeypatch.setenv("BRAINLAYER_WATCHER_OFFSETS_PATH", str(offsets_path))
    monkeypatch.setenv("BRAINLAYER_WATCHER_HEALTH_PATH", str(watcher_health_path))
    monkeypatch.setenv("BRAINLAYER_ENRICH_WATCHER_IDLE_SECONDS", "60")

    class FakeVectorStore:
        def __init__(self, db_path: Path):
            pass

        def close(self):
            pass

    calls = []

    def empty_recent_then_backlog(store, **kwargs):
        calls.append(kwargs["since_hours"])
        if kwargs["since_hours"] is None:
            return _result(attempted=1, enriched=1)
        return _result(attempted=0)

    result = controller.run_enrich_supervisor(
        tmp_path / "brainlayer.db",
        limit=5,
        since_hours=24,
        max_cycles=1,
        vector_store_cls=FakeVectorStore,
        enrich_fn=empty_recent_then_backlog,
        sleep_fn=lambda seconds: None,
    )

    assert calls == [24]
    assert result.cycles == 1
    assert result.attempted == 0


def test_supervisor_does_not_log_per_cycle_store_initialization_at_info(tmp_path, caplog):
    from brainlayer import enrichment_controller as controller

    class FakeVectorStore:
        def __init__(self, db_path: Path):
            pass

        def close(self):
            pass

    with caplog.at_level(logging.INFO, logger="brainlayer.enrichment_controller"):
        controller.run_enrich_supervisor(
            tmp_path / "brainlayer.db",
            max_cycles=3,
            vector_store_cls=FakeVectorStore,
            enrich_fn=lambda store, **kwargs: _result(attempted=1, enriched=1),
            sleep_fn=lambda seconds: None,
        )

    init_lines = [record for record in caplog.records if "initialized for enrich supervisor pass" in record.message]
    assert init_lines == []
