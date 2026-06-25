import logging
import os
import plistlib
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


def test_supervisor_releases_vector_store_between_cycles(tmp_path):
    from brainlayer import enrichment_controller as controller

    init_paths = []
    enrich_calls = []
    closed = []

    class FakeVectorStore:
        def __init__(self, db_path: Path):
            self.index = len(init_paths)
            init_paths.append(db_path)
            self.closed = False

        def close(self):
            self.closed = True
            closed.append(self.index)

    def fake_enrich(store, **kwargs):
        enrich_calls.append((store, kwargs))
        return _result(attempted=1, enriched=1)

    result = controller.run_enrich_supervisor(
        tmp_path / "brainlayer.db",
        limit=7,
        since_hours=12,
        max_cycles=3,
        vector_store_cls=FakeVectorStore,
        enrich_fn=fake_enrich,
        sleep_fn=lambda seconds: None,
    )

    assert init_paths == [tmp_path / "brainlayer.db"] * 3
    assert len({id(call[0]) for call in enrich_calls}) == 3
    assert len(enrich_calls) == 3
    assert all(call[1]["limit"] == 7 and call[1]["since_hours"] == 12 for call in enrich_calls)
    assert closed == [0, 1, 2]
    assert result.cycles == 3
    assert result.enriched == 3


def test_supervisor_uses_readonly_store_when_enrichment_queue_writes_enabled(tmp_path, monkeypatch):
    from brainlayer import enrichment_controller as controller

    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_QUEUE_WRITES", "1")
    init_args = []

    class FakeVectorStore:
        def __init__(self, db_path: Path, readonly: bool = False):
            init_args.append((db_path, readonly))

        def close(self):
            pass

    result = controller.run_enrich_supervisor(
        tmp_path / "brainlayer.db",
        max_cycles=1,
        vector_store_cls=FakeVectorStore,
        enrich_fn=lambda store, **kwargs: _result(attempted=0),
        sleep_fn=lambda seconds: None,
    )

    assert init_args == [(tmp_path / "brainlayer.db", True)]
    assert result.cycles == 1


def test_enrich_realtime_skips_schema_writer_when_enrichment_queue_writes_enabled(monkeypatch):
    from brainlayer import enrichment_controller as controller

    monkeypatch.setenv("BRAINLAYER_ENRICHMENT_QUEUE_WRITES", "1")
    monkeypatch.setattr(
        controller,
        "_ensure_enrichment_columns",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not open schema writer")),
    )
    monkeypatch.setattr(controller, "_enqueue_enrichment_write_batch", lambda items: None)
    monkeypatch.setattr(controller, "_get_gemini_client", lambda: object())

    class FakeStore:
        def get_enrichment_candidates(self, **kwargs):  # noqa: ARG002
            return [{"id": "c1", "content": "brain_search('recursive MCP output')"}]

    result = controller.enrich_realtime(FakeStore(), limit=1, rate_per_second=0)

    assert result.skipped == 1
    assert result.failed == 0


def test_supervisor_logs_gemini_error_and_continues(tmp_path, caplog):
    from brainlayer import enrichment_controller as controller

    class FakeVectorStore:
        def __init__(self, db_path: Path):
            self.db_path = db_path

        def close(self):
            pass

    calls = 0

    def flaky_enrich(store, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("gemini 503")
        return _result(attempted=1, enriched=1)

    with caplog.at_level(logging.ERROR, logger="brainlayer.enrichment_controller"):
        result = controller.run_enrich_supervisor(
            tmp_path / "brainlayer.db",
            max_cycles=2,
            vector_store_cls=FakeVectorStore,
            enrich_fn=flaky_enrich,
            sleep_fn=lambda seconds: None,
        )

    assert calls == 2
    assert result.enriched == 1
    assert result.failed == 0
    assert result.failed_cycles == 1
    assert result.errors == ["supervisor: gemini 503"]
    assert "Enrich supervisor cycle failed; continuing" in caplog.text


def test_supervisor_backs_off_between_persistent_cycle_errors(tmp_path):
    from brainlayer import enrichment_controller as controller

    class FakeVectorStore:
        def __init__(self, db_path: Path):
            self.db_path = db_path

        def close(self):
            pass

    sleeps = []

    def failing_enrich(store, **kwargs):
        raise RuntimeError("gemini still down")

    result = controller.run_enrich_supervisor(
        tmp_path / "brainlayer.db",
        max_cycles=3,
        vector_store_cls=FakeVectorStore,
        enrich_fn=failing_enrich,
        sleep_fn=sleeps.append,
    )

    assert result.cycles == 3
    assert result.failed == 0
    assert result.failed_cycles == 3
    assert sleeps == [30.0, 30.0]


def test_supervisor_backs_off_after_transient_store_open_error(tmp_path):
    from brainlayer import enrichment_controller as controller

    init_attempts = 0
    sleeps = []

    class FakeVectorStore:
        def __init__(self, db_path: Path):
            nonlocal init_attempts
            init_attempts += 1
            if init_attempts == 2:
                raise RuntimeError("sqlite transient open busy")

        def close(self):
            pass

    result = controller.run_enrich_supervisor(
        tmp_path / "brainlayer.db",
        max_cycles=3,
        vector_store_cls=FakeVectorStore,
        enrich_fn=lambda store, **kwargs: _result(attempted=1, enriched=1),
        sleep_fn=sleeps.append,
    )

    assert init_attempts == 3
    assert result.cycles == 3
    assert result.enriched == 2
    assert result.failed_cycles == 1
    assert result.errors == ["supervisor-open: sqlite transient open busy"]
    assert sleeps == [30.0]


def test_supervisor_graceful_shutdown_closes_store_after_inflight_cycle(tmp_path):
    from brainlayer import enrichment_controller as controller

    events = []
    stop_event = controller.threading.Event()

    class FakeVectorStore:
        def __init__(self, db_path: Path):
            events.append("init")

        def close(self):
            events.append("close")

    def stopping_enrich(store, **kwargs):
        events.append("enrich-start")
        stop_event.set()
        events.append("enrich-finish")
        return _result(attempted=1, enriched=1)

    result = controller.run_enrich_supervisor(
        tmp_path / "brainlayer.db",
        stop_event=stop_event,
        vector_store_cls=FakeVectorStore,
        enrich_fn=stopping_enrich,
        sleep_fn=lambda seconds: None,
    )

    assert events == ["init", "enrich-start", "enrich-finish", "close"]
    assert result.exit_code == 0
    assert result.cycles == 1


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


def test_enrichment_launchagent_runs_supervisor_not_shell_respawn_loop():
    plist_path = Path("scripts/launchd/com.brainlayer.enrichment.plist")
    plist = plistlib.loads(plist_path.read_bytes())

    args = plist["ProgramArguments"]

    assert args == ["__BRAINLAYER_ENV_RUN__", "__BRAINLAYER_BIN__", "enrich", "--mode", "realtime", "--supervisor"]
    assert "--limit" not in args
    assert "--since-hours" not in args
    assert "while true" not in " ".join(args)
    assert plist["KeepAlive"] is True
    assert plist["ProcessType"] == "Background"
    assert plist["EnvironmentVariables"]["BRAINLAYER_ENRICHMENT_QUEUE_WRITES"] == "1"
    assert plist["EnvironmentVariables"]["BRAINLAYER_ENRICH_IDLE_BACKLOG"] == "1"
