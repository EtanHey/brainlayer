import logging
import plistlib
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


def test_supervisor_reuses_one_vector_store_across_cycles(tmp_path):
    from brainlayer import enrichment_controller as controller

    init_paths = []
    enrich_calls = []

    class FakeVectorStore:
        def __init__(self, db_path: Path):
            init_paths.append(db_path)
            self.closed = False

        def close(self):
            self.closed = True

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

    assert init_paths == [tmp_path / "brainlayer.db"]
    assert len({id(call[0]) for call in enrich_calls}) == 1
    assert len(enrich_calls) == 3
    assert all(call[1]["limit"] == 7 and call[1]["since_hours"] == 12 for call in enrich_calls)
    assert result.cycles == 3
    assert result.enriched == 3


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
    assert result.failed == 1
    assert result.errors == ["supervisor: gemini 503"]
    assert "Enrich supervisor cycle failed; continuing" in caplog.text


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


def test_supervisor_logs_single_vectorstore_initialized_line(tmp_path, caplog):
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

    init_lines = [
        record for record in caplog.records if "VectorStore initialized for enrich supervisor" in record.message
    ]
    assert len(init_lines) == 1


def test_enrichment_launchagent_runs_supervisor_not_shell_respawn_loop():
    plist_path = Path("scripts/launchd/com.brainlayer.enrichment.plist")
    plist = plistlib.loads(plist_path.read_bytes())

    args = plist["ProgramArguments"]

    assert args == ["__BRAINLAYER_BIN__", "enrich", "--mode", "realtime", "--supervisor"]
    assert "--limit" not in args
    assert "--since-hours" not in args
    assert "while true" not in " ".join(args)
    assert plist["KeepAlive"] is True
    assert plist["ProcessType"] == "Background"
