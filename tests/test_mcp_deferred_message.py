"""The DEFERRED store receipt must read as SUCCESS so agents stop re-storing (Etan, 2026-08-09)."""

import os
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from brainlayer.mcp._format import format_store_result


@pytest.fixture
def owned_legacy_replay(monkeypatch, tmp_path, isolate_brainlayer_runtime_paths):
    """Own synthetic replay work; register every blocking collaborator's gate."""
    from brainlayer.mcp import store_handler

    @contextmanager
    def own():
        assert store_handler._pending_replay_active is False
        db_path = tmp_path / "owned-replay" / "brainlayer.db"
        db_path.parent.mkdir()
        path = db_path.parent / "pending-stores.jsonl"
        threads, errors, delays, resolved_paths = [], [], [], []
        completed = {}
        cleanup_gates = []
        sleeping, release = threading.Event(), threading.Event()
        real_path = store_handler._get_pending_store_path

        def checked_path():
            assert os.environ["BRAINLAYER_TEST_PATH_PROVENANCE"] == "pytest"
            assert os.environ["BRAINLAYER_DB"] == str(db_path)
            resolved = real_path()
            assert resolved == path
            resolved_paths.append(resolved)
            return resolved

        def controlled_sleep(delay):
            delays.append(delay)
            assert len(delays) <= 4, "unexpected unbounded replay in a synthetic test"
            sleeping.set()
            assert release.wait(timeout=2), "test did not release its owned replay"

        def owned_thread(*, target, daemon):
            finished = threading.Event()

            def run():
                try:
                    target()
                except BaseException as exc:
                    errors.append(exc)  # Re-raised after joining, never suppressed.
                finally:
                    finished.set()  # Target exit and error recording precede completion.

            thread = threading.Thread(target=run, daemon=daemon)
            completed[thread] = finished
            threads.append(thread)
            return thread

        with monkeypatch.context() as scope:
            scope.setenv("BRAINLAYER_DB", str(db_path))
            scope.setattr(store_handler, "_get_pending_store_path", checked_path)
            scope.setattr(store_handler, "threading", SimpleNamespace(**{**vars(threading), "Thread": owned_thread}))
            scope.setattr(store_handler, "time", SimpleNamespace(**{**vars(time), "sleep": controlled_sleep}))
            failures = []
            try:
                yield SimpleNamespace(
                    path=path,
                    db_path=db_path,
                    threads=threads,
                    delays=delays,
                    sleeping=sleeping,
                    resolved_paths=resolved_paths,
                    cleanup_gates=cleanup_gates,
                )
            except BaseException as exc:
                failures.append(exc)
            finally:
                release.set()
                index = 0
                # A completed target has already registered any successor before returning.
                while index < len(threads):
                    thread = threads[index]
                    if thread.ident is None:  # Synchronous start failure: no wrapper ran.
                        index += 1
                        continue
                    finished = completed[thread]
                    try:
                        thread.join(timeout=2)
                    except BaseException as exc:
                        failures.append(exc)
                    if not finished.is_set():
                        failures.append(AssertionError("owned replay exceeded its join bound"))
                        for gate in cleanup_gates:
                            gate.set()
                        # The bound reports failure; it never permits path restoration.
                        # Collaborators here are synthetic and cooperatively releasable.
                        while not finished.is_set():
                            try:
                                finished.wait()
                            except BaseException as exc:
                                failures.append(exc)
                    try:
                        thread.join()  # Bookkeeping only; target completion is already witnessed.
                    except BaseException as exc:
                        failures.append(exc)
                    index += 1
                failures.extend(errors)
                if store_handler._pending_replay_active:
                    failures.append(AssertionError("owned replay remained active after joins"))
                if len(failures) == 1:
                    raise failures[0]
                if failures:
                    raise BaseExceptionGroup("owned replay cleanup failures", failures)

    return own


@pytest.mark.parametrize("pending", [False, True])
def test_real_replay_finishes_before_private_paths_restore(owned_legacy_replay, monkeypatch, pending):
    from brainlayer import runtime_store
    from brainlayer.mcp import store_handler

    original_db = os.environ["BRAINLAYER_DB"]
    store = Mock()
    open_writer = Mock(return_value=store)
    monkeypatch.setattr(runtime_store, "open_writer_store", open_writer)
    flushed = []

    with owned_legacy_replay() as replay:
        assert store_handler.time.monotonic is time.monotonic
        assert store_handler.threading.Event is threading.Event
        replay.path.write_text("synthetic pending item\n" if pending else "")

        def flush(actual_store, embed_fn):
            assert actual_store is store
            assert embed_fn is None
            flushed.append(store_handler._get_pending_store_path())
            replay.path.write_text("")

        monkeypatch.setattr(store_handler, "_flush_pending_stores", flush)
        store_handler._schedule_pending_store_replay()
        assert replay.sleeping.wait(timeout=2)
        assert replay.threads[0].is_alive()
        assert store_handler._pending_replay_active is True
        store_handler._schedule_pending_store_replay()
        assert len(replay.threads) == 1  # Active replays coalesce.
        open_writer.assert_not_called()

    assert os.environ["BRAINLAYER_DB"] == original_db
    assert replay.resolved_paths
    assert replay.delays == ([2.0, 4.0] if pending else [2.0])
    assert flushed == ([replay.path] if pending else [])
    if pending:
        open_writer.assert_called_once_with(replay.db_path)
        store.close.assert_called_once_with()
    else:
        open_writer.assert_not_called()
        store.close.assert_not_called()


def test_real_replay_joins_successor_after_last_empty_check(owned_legacy_replay, monkeypatch):
    from brainlayer import runtime_store
    from brainlayer.mcp import store_handler

    store = Mock()
    open_writer = Mock(return_value=store)
    monkeypatch.setattr(runtime_store, "open_writer_store", open_writer)
    real_has_data = store_handler._pending_store_has_data
    appended = False

    def append_after_empty_probe(path):
        nonlocal appended
        has_data = real_has_data(path)
        if not has_data and not appended:
            path.write_text("synthetic late append\n")
            appended = True
        return has_data

    monkeypatch.setattr(store_handler, "_pending_store_has_data", append_after_empty_probe)
    with owned_legacy_replay() as replay:
        replay.path.write_text("")
        monkeypatch.setattr(store_handler, "_flush_pending_stores", lambda *_: replay.path.write_text(""))
        store_handler._schedule_pending_store_replay()
        assert replay.sleeping.wait(timeout=2)

    assert len(replay.threads) == 2
    assert replay.delays == [2.0, 2.0, 4.0]
    assert replay.path.read_text() == ""
    open_writer.assert_called_once_with(replay.db_path)
    store.close.assert_called_once_with()


@pytest.mark.parametrize("interrupted", [False, True])
def test_failed_owner_joins_blocked_worker_before_paths_restore(owned_legacy_replay, monkeypatch, interrupted):
    from brainlayer import runtime_store
    from brainlayer.mcp import store_handler

    gate, closing = threading.Event(), threading.Event()
    target_completed = threading.Event()
    proof = {"interrupted": interrupted}
    store = Mock()
    monkeypatch.setattr(runtime_store, "open_writer_store", Mock(return_value=store))
    original_join = None
    thread = None
    failure = None
    try:
        with owned_legacy_replay() as replay:
            replay.path.write_text("synthetic pending item\n")
            replay.cleanup_gates.append(gate)
            monkeypatch.setattr(store_handler, "_flush_pending_stores", lambda *_: replay.path.write_text(""))

            def blocked_close():
                closing.set()
                assert gate.wait(timeout=2), "owner failed to release its controlled collaborator"
                proof["worker_saw_private_paths"] = os.environ["BRAINLAYER_DB"] == str(replay.db_path)
                proof["worker_saw_provenance"] = os.environ["BRAINLAYER_TEST_PATH_PROVENANCE"] == "pytest"
                raise LookupError("synthetic worker failure")

            store.close.side_effect = blocked_close
            owned_factory = store_handler.threading.Thread

            def observed_thread(*, target, daemon):
                def observed_target():
                    try:
                        target()
                    finally:
                        target_completed.set()

                return owned_factory(target=observed_target, daemon=daemon)

            monkeypatch.setattr(store_handler.threading, "Thread", observed_thread)
            store_handler._schedule_pending_store_replay()
            assert replay.sleeping.wait(timeout=2)
            thread = replay.threads[0]
            original_join = thread.join

            def forced_timed_join(timeout=None):
                if timeout is not None:
                    assert closing.wait(timeout=2)
                    assert thread.is_alive()
                    proof["forced_timeout_with_live_worker"] = True
                    if interrupted:
                        monkeypatch.setattr(thread, "is_alive", lambda: False)
                        raise KeyboardInterrupt("synthetic join interruption")
                    return  # Deterministic timeout injection; no elapsed-time race.
                original_join()

            monkeypatch.setattr(thread, "join", forced_timed_join)
            raise ValueError("synthetic owner failure")
    except BaseException as exc:
        failure = exc
        proof["target_completed_after_context"] = target_completed.is_set()
        proof["worker_alive_after_context"] = thread is not None and thread.is_alive()
    finally:
        # RED containment only: never let the intentionally broken helper leak a worker.
        gate.set()
        if original_join is not None:
            original_join()
        print("forced cleanup proof:", proof)

    assert proof["target_completed_after_context"] is True
    assert proof["worker_alive_after_context"] is False
    assert proof["worker_saw_private_paths"] is True
    assert proof["worker_saw_provenance"] is True
    assert isinstance(failure, BaseExceptionGroup)
    expected = [(ValueError, "synthetic owner failure")]
    if interrupted:
        expected.append((KeyboardInterrupt, "synthetic join interruption"))
    assert [(type(exc), str(exc)) for exc in failure.exceptions] == expected + [
        (AssertionError, "owned replay exceeded its join bound"),
        (LookupError, "synthetic worker failure"),
    ]


def test_owned_replay_startup_failure_does_not_wait(owned_legacy_replay, monkeypatch):
    from brainlayer.mcp import store_handler

    monkeypatch.setattr(store_handler, "_pending_replay_active", False)
    with pytest.raises(ExceptionGroup) as caught:
        with owned_legacy_replay() as replay:
            factory = store_handler.threading.Thread

            def never_started(**kwargs):
                thread = factory(**kwargs)
                thread.start = Mock(side_effect=RuntimeError("synthetic startup failure"))
                thread.join = Mock(side_effect=AssertionError("unstarted worker must not be joined"))
                return thread

            monkeypatch.setattr(store_handler.threading, "Thread", never_started)
            store_handler._schedule_pending_store_replay()
    assert [(type(exc), str(exc)) for exc in caught.value.exceptions] == [
        (RuntimeError, "synthetic startup failure"),
        (AssertionError, "owned replay remained active after joins"),
    ]
    assert replay.threads[0].ident is None
    replay.threads[0].join.assert_not_called()
    assert replay.resolved_paths == []


def test_deferred_store_receipt_reads_as_success_and_forbids_restore() -> None:
    msg = format_store_result("brainbar-abc123", queued=True)
    assert "STORED (deferred)" in msg
    assert "DB busy" in msg
    assert "brainbar-abc123" in msg
    assert "durably queued" in msg
    assert "the drain persists it automatically" in msg
    assert "Do NOT re-store" in msg
    assert "fallback copy" in msg
    assert "DEFERRED:" not in msg  # old failure-sounding prefix retired


def test_deferred_store_receipt_carries_non_busy_reason() -> None:
    msg = format_store_result("brainbar-abc123", queued=True, queued_reason="INTERACTIVE_PRIORITY")
    assert "STORED (deferred): INTERACTIVE_PRIORITY" in msg
    assert "DB busy" not in msg
    assert "Do NOT re-store" in msg


def test_non_queued_store_receipt_names_the_outcome_and_is_not_confusable_with_deferred() -> None:
    """A fresh write says STORED and says it is new (Etan, 2026-08-19).

    The old text was a bare "Stored -> <id>", which read identically to a
    suppressed duplicate and to a deferred write -- the ambiguity that had
    agents re-storing the same memory.
    """
    msg = format_store_result("brainbar-abc123")
    assert "STORED" in msg
    assert "brainbar-abc123" in msg
    assert "new" in msg.lower()
    assert "deferred" not in msg.lower()
    assert "DUPLICATE" not in msg


def test_deferred_receipt_for_legacy_queue_schedules_replay(monkeypatch, tmp_path) -> None:
    """A queued_for_replay receipt must arm the background replay (codex P1, PR #693)."""
    from brainlayer.mcp import store_handler

    calls = []
    monkeypatch.setattr(store_handler, "_schedule_pending_store_replay", lambda: calls.append(True))

    store_handler._deferred_store_receipt("c1", tmp_path / "pending-stores.jsonl")
    assert calls == [True]

    store_handler._deferred_store_receipt("c2", tmp_path / "mcp-123.jsonl")
    assert calls == [True]  # drain-daemon path must NOT arm the replay


def test_schema_mismatch_deferral_names_its_reason(tmp_path) -> None:
    """Schema-fingerprint deferrals must not claim DB busy (codex P2, PR #693)."""
    msg = format_store_result("brainbar-x", queued=True, queued_reason="SCHEMA_FINGERPRINT_MISMATCH")
    assert "STORED (deferred): SCHEMA_FINGERPRINT_MISMATCH" in msg
    assert "DB busy" not in msg


def test_startup_rearms_replay_when_legacy_queue_nonempty(monkeypatch, tmp_path) -> None:
    """A stranded pending-stores.jsonl must re-arm replay at process start (codex round 4)."""
    from brainlayer.mcp import store_handler

    stranded = tmp_path / "pending-stores.jsonl"
    stranded.write_text('{"chunk_id": "c1", "content": "x"}\n')
    calls = []
    monkeypatch.setattr(store_handler, "_get_pending_store_path", lambda: stranded)
    monkeypatch.setattr(store_handler, "_schedule_pending_store_replay", lambda: calls.append(True))

    assert store_handler.rearm_stranded_pending_stores() is True
    assert calls == [True]

    stranded.write_text("")
    assert store_handler.rearm_stranded_pending_stores() is False
    assert calls == [True]


def test_startup_replay_probe_tolerates_concurrent_queue_unlink(monkeypatch, tmp_path) -> None:
    """A queue removed between existence and size probes must not kill replay recovery."""
    from contextlib import nullcontext

    from brainlayer.mcp import store_handler

    class UnlinkedPendingStorePath:
        parent = tmp_path
        name = "pending-stores.jsonl"

        @staticmethod
        def exists() -> bool:
            return True

        @staticmethod
        def stat():
            raise FileNotFoundError("concurrent replay removed pending-stores.jsonl")

    calls = []
    monkeypatch.setattr(store_handler, "_get_pending_store_path", UnlinkedPendingStorePath)
    monkeypatch.setattr(store_handler, "_pending_store_file_lock", lambda _path: nullcontext())
    monkeypatch.setattr(store_handler, "_schedule_pending_store_replay", lambda: calls.append(True))

    assert store_handler.rearm_stranded_pending_stores() is False
    assert calls == []


def test_queue_store_never_trims_acknowledged_entries(monkeypatch, tmp_path) -> None:
    """Acknowledged durably-queued stores must never be silently dropped (codex round 5)."""
    import json as _json

    from brainlayer.mcp import store_handler

    path = tmp_path / "pending-stores.jsonl"
    monkeypatch.setattr(store_handler, "_get_pending_store_path", lambda: path)
    monkeypatch.setattr(store_handler, "_schedule_pending_store_replay", lambda: None)

    def _fail_enqueue(**kwargs):
        raise RuntimeError("unified queue unavailable")

    import brainlayer.queue_io as queue_io

    monkeypatch.setattr(queue_io, "enqueue_store", _fail_enqueue)

    for i in range(store_handler._QUEUE_MAX_SIZE + 5):
        store_handler._queue_store({"chunk_id": f"c{i}", "content": f"memory {i}"})

    lines = path.read_text().strip().splitlines()
    assert len(lines) == store_handler._QUEUE_MAX_SIZE + 5
    assert _json.loads(lines[0])["chunk_id"] == "c0"  # oldest acknowledged entry survives
