from __future__ import annotations

import fcntl
import multiprocessing
import os
import threading
import traceback
from pathlib import Path

import apsw
import pytest

from brainlayer import writer_telemetry

_CONTENT = "Disposable producer memory for telemetry contention regression"


def _hold_telemetry_lock(path, control):
    with Path(path).open("ab") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        control.send("locked")
        # Independent fuse if the test controller dies before its finally block.
        if not control.poll(30):
            raise TimeoutError("test controller did not release telemetry holder")
        assert control.recv() == "release"
    control.close()


def _run_producer(root, producer, control):
    from brainlayer.drain import drain_once
    from brainlayer.queue_io import enqueue_store
    from brainlayer.store import store_memory
    from brainlayer.vector_store import VectorStore

    root = Path(root)
    store = None
    try:
        # Finish schema setup before the holder acquires the lock: the assertion
        # below must exercise the real producer, rather than VectorStore startup.
        os.environ["BRAINLAYER_WRITER_TELEMETRY"] = "0"
        store = VectorStore(root / "brainlayer.db")
        if producer == "drain":
            store.close()
            store = None
            enqueue_store(content=_CONTENT, queue_dir=root / "queue")
        os.environ["BRAINLAYER_WRITER_TELEMETRY"] = "1"
        control.send("ready")
        if not control.poll(20):
            raise TimeoutError("test controller did not start producer")
        assert control.recv() == "run"

        if producer == "store":
            result = store_memory(
                store=store,
                embed_fn=None,
                content=_CONTENT,
                memory_type="note",
                retry_on_busy=False,
            )["id"]
            store.close()
            store = None
        else:
            result = drain_once(
                db_path=root / "brainlayer.db",
                queue_dir=root / "queue",
                batch_size=1,
                log_path=root / "drain.log",
                embed_fn=lambda _content: [0.25] * 1024,
            )
        control.send(("done", result))
    except BaseException:
        control.send(("error", traceback.format_exc()))
    finally:
        if store is not None:
            store.close()
        control.close()


def _reap(process):
    if process.pid is None:
        return
    process.join(1)
    if process.is_alive():
        process.terminate()
        process.join(3)
    if process.is_alive():
        process.kill()
        process.join(3)
    assert not process.is_alive(), "disposable telemetry test process survived teardown"


@pytest.mark.parametrize("producer", ["store", "drain"])
def test_real_producer_completes_before_telemetry_lock_is_released(tmp_path, monkeypatch, producer):
    log_path = tmp_path / "writer-telemetry.jsonl"
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY", "1")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_PATH", str(log_path))
    monkeypatch.setenv("BRAINLAYER_WRITER_HEARTBEAT_DIR", str(tmp_path / "heartbeats"))
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(tmp_path / "pidfiles"))
    monkeypatch.setenv("BRAINLAYER_DRAIN_POST_COMMIT_YIELD_MS", "0")
    context = multiprocessing.get_context("spawn")
    controller, worker_control = context.Pipe()
    holder_controller, holder_control = context.Pipe()
    worker = context.Process(target=_run_producer, args=(tmp_path, producer, worker_control))
    holder = context.Process(
        target=_hold_telemetry_lock,
        args=(log_path.with_name(f".{log_path.name}.lock"), holder_control),
    )
    try:
        worker.start()
        worker_control.close()
        assert controller.poll(15), "producer setup did not finish"
        assert controller.recv() == "ready"
        holder.start()
        holder_control.close()
        assert holder_controller.poll(10), "telemetry holder did not acquire lock"
        assert holder_controller.recv() == "locked"

        controller.send("run")
        assert controller.poll(5), f"real {producer} stalled while a separate process held the telemetry lock"
        status, result = controller.recv()
        assert status == "done", result
        assert holder.is_alive(), "holder must retain telemetry lock through producer completion and readback"
        conn = apsw.Connection(str(tmp_path / "brainlayer.db"), flags=apsw.SQLITE_OPEN_READONLY)
        try:
            rows = list(conn.execute("SELECT id, content FROM chunks"))
            assert len(rows) == 1
            assert rows[0][1] == _CONTENT
            assert conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'contention'").fetchone() == (
                1,
            )
        finally:
            conn.close()
        assert holder.is_alive(), "telemetry lock was released before committed-row readback"
        if producer == "store":
            assert result == rows[0][0]
        else:
            assert result == 1
            assert not list((tmp_path / "queue").glob("*.jsonl"))
        assert not log_path.exists(), "contended observations must be dropped"
        assert not list(tmp_path.glob("writer-telemetry-summary-*.json"))
        worker.join(3)
        assert worker.exitcode == 0
    finally:
        # On the red implementation, reap the stuck producer BEFORE releasing
        # the holder; a late commit after unlocking cannot satisfy this test.
        _reap(worker)
        if holder.is_alive():
            holder_controller.send("release")
        _reap(holder)
        for pipe in (controller, worker_control, holder_controller, holder_control):
            pipe.close()

    assert holder.exitcode == 0
    assert writer_telemetry.append_event({"event": "after-contention"}) is True


def test_append_event_drops_when_process_local_sink_lock_is_busy(tmp_path, monkeypatch):
    log_path = tmp_path / "writer-telemetry.jsonl"
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY", "1")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_PATH", str(log_path))
    entered = threading.Event()
    finished = threading.Event()
    results = []

    def append():
        entered.set()
        results.append(writer_telemetry.append_event({"event": "contended"}))
        finished.set()

    worker = threading.Thread(target=append, daemon=True)
    with writer_telemetry._SINK_LOCK:
        worker.start()
        entered_while_locked = entered.wait(3)
        completed_while_locked = finished.wait(3)
    worker.join(3)
    assert not worker.is_alive()
    assert entered_while_locked
    assert completed_while_locked, "append_event waited for the process-local telemetry sink lock"
    assert results == [False]
    assert not log_path.exists()
    assert writer_telemetry.append_event({"event": "after-contention"}) is True
