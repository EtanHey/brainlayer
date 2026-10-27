"""Private bounded diagnostic sink behavior; mapped by the module filename."""

import json
import os


def test_diagnostic_sink_is_capped_and_blocked_writer_cannot_block_emit(tmp_path, monkeypatch):
    import threading
    import time

    from brainlayer.supervisor_diagnostics import DiagnosticSink

    path = tmp_path / "events.jsonl"
    sink = DiagnosticSink.open(path, capacity=8192)
    try:
        for n in range(80):
            sink.emit(event="sample", value=n, padding="x" * 200)
            time.sleep(0.001)
    finally:
        sink.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert path.stat().st_size <= 8192 and rows[-1]["value"] == 79

    entered, release = threading.Event(), threading.Event()
    original_write = os.write

    def blocked(fd, data):
        entered.set()
        release.wait(3)
        return original_write(fd, data)

    sink = DiagnosticSink.open(tmp_path / "blocked.jsonl")
    monkeypatch.setattr("brainlayer.supervisor_diagnostics.os.write", blocked)
    try:
        sink.emit(event="first")
        assert entered.wait(1)
        began = time.monotonic()
        for _ in range(1000):
            sink.emit(event="guard-timeout")
        sink.close()
        assert time.monotonic() - began < 0.2
    finally:
        release.set()
        sink.writer.join(timeout=1)


def test_emit_serializes_only_in_writer_and_idle_wait_is_not_polling(tmp_path, monkeypatch):
    import threading

    from brainlayer import supervisor_diagnostics as module

    serialized, waits = [], []
    dumps, get = module.json.dumps, module.queue.Queue.get

    def serialize(*args, **kwargs):
        serialized.append(threading.current_thread().name)
        return dumps(*args, **kwargs)

    def receive(self, *args, **kwargs):
        waits.append(kwargs.get("timeout"))
        return get(self, *args, **kwargs)

    monkeypatch.setattr(module.json, "dumps", serialize)
    monkeypatch.setattr(module.queue.Queue, "get", receive)
    sink = module.DiagnosticSink.open(tmp_path / "events.jsonl")
    sink.emit(event="guard-timeout")
    sink.close()
    assert serialized == ["guard-diagnostics"]
    assert waits and all(wait is None or wait >= 0.5 for wait in waits)


def test_rollover_retains_latest_timeout_and_complete_tail(tmp_path):
    from brainlayer.supervisor_diagnostics import DiagnosticSink

    path = tmp_path / "events.jsonl"
    history = [dict(event="sample", padding="x" * 140, n=n) for n in range(8)]
    history.append(dict(event="guard-timeout", parent_pid=1234))
    path.write_text("".join(json.dumps(row) + "\n" for row in history))
    path.chmod(0o600)
    capacity = path.stat().st_size + 10
    sink = DiagnosticSink.open(path, capacity=capacity)
    sink.emit(event="guard-started", padding="y" * 300)
    sink.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert path.stat().st_size <= capacity
    assert history[-1] in rows and rows[-1]["event"] == "guard-started"
    assert len(rows) > 2


def test_refuses_unsafe_sink_paths(tmp_path):
    import pytest

    from brainlayer.supervisor_diagnostics import DiagnosticSink

    target = tmp_path / "target"
    target.write_text("")
    target.chmod(0o600)
    symlink, hardlink, fifo, readable = (tmp_path / name for name in ("symlink", "hardlink", "fifo", "readable"))
    symlink.symlink_to(target)
    os.link(target, hardlink)
    os.mkfifo(fifo, 0o600)
    readable.write_text("")
    readable.chmod(0o644)
    for path in (symlink, hardlink, fifo, readable):
        with pytest.raises(OSError):
            DiagnosticSink.open(path)
