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
