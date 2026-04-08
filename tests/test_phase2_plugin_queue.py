"""Phase 2 plugin queue and hook helper tests."""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "extensions" / "brainlayer-plugin"
QUEUE_FLUSHER_PATH = PLUGIN_ROOT / "scripts" / "queue-flusher.py"
HOOK_HELPER_PATH = PLUGIN_ROOT / "scripts" / "brainbar-hook.py"


def _load_module(path: Path, name: str):
    assert path.exists(), f"expected file to exist: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_append_queue_event_writes_one_json_line(tmp_path):
    flusher = _load_module(QUEUE_FLUSHER_PATH, "queue_flusher")
    queue_path = tmp_path / ".memory-queue.jsonl"

    flusher.append_queue_event(queue_path, {"hook_event": "PostToolUse", "tool_name": "Read"})

    lines = queue_path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["tool_name"] == "Read"


def test_append_queue_event_is_safe_under_concurrent_writers(tmp_path):
    flusher = _load_module(QUEUE_FLUSHER_PATH, "queue_flusher")
    queue_path = tmp_path / ".memory-queue.jsonl"

    threads = [
        threading.Thread(
            target=flusher.append_queue_event,
            args=(queue_path, {"hook_event": "PostToolUse", "index": index}),
        )
        for index in range(100)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows = [json.loads(line) for line in queue_path.read_text().splitlines()]
    assert len(rows) == 100
    assert {row["index"] for row in rows} == set(range(100))


def test_flush_queue_batches_and_clears_file_on_success(tmp_path):
    flusher = _load_module(QUEUE_FLUSHER_PATH, "queue_flusher")
    queue_path = tmp_path / ".memory-queue.jsonl"
    sent_batches: list[list[dict]] = []

    for index in range(3):
        flusher.append_queue_event(queue_path, {"hook_event": "PostToolUse", "index": index})

    flushed = flusher.flush_queue(queue_path, sent_batches.append, batch_size=2)

    assert flushed == 3
    assert sent_batches == [
        [
            {"hook_event": "PostToolUse", "index": 0},
            {"hook_event": "PostToolUse", "index": 1},
        ],
        [{"hook_event": "PostToolUse", "index": 2}],
    ]
    assert queue_path.read_text() == ""


def test_flush_queue_keeps_events_when_sender_fails(tmp_path):
    flusher = _load_module(QUEUE_FLUSHER_PATH, "queue_flusher")
    queue_path = tmp_path / ".memory-queue.jsonl"
    flusher.append_queue_event(queue_path, {"hook_event": "PostToolUse", "index": 1})

    def failing_sender(_batch):
        raise RuntimeError("socket unavailable")

    flushed = flusher.flush_queue(queue_path, failing_sender)

    assert flushed == 0
    rows = [json.loads(line) for line in queue_path.read_text().splitlines()]
    assert rows == [{"hook_event": "PostToolUse", "index": 1}]


def test_handle_session_start_returns_additional_context_json():
    hook_helper = _load_module(HOOK_HELPER_PATH, "brainbar_hook")

    started = time.perf_counter()
    output = hook_helper.handle_session_start(
        {"session_id": "sess-123", "cwd": "/tmp/project"},
        invoke_tool=lambda tool, arguments: {
            "tool": tool,
            "arguments": arguments,
            "summary": "Recent work: decay batch benchmark 2.47s",
        },
    )
    duration_ms = (time.perf_counter() - started) * 1000

    assert duration_ms < 200
    assert output["additionalContext"].startswith("Recent work:")


def test_build_queue_event_for_failure_marks_error_importance():
    hook_helper = _load_module(HOOK_HELPER_PATH, "brainbar_hook")

    event = hook_helper.build_queue_event(
        "PostToolUseFailure",
        {
            "session_id": "sess-123",
            "cwd": "/tmp/project",
            "tool_name": "Bash",
            "tool_input": {"command": "pytest"},
            "tool_response": {"error": "exit 1"},
        },
    )

    assert event["importance"] == 8
    assert "error" in event["tags"]
    assert event["tool_name"] == "Bash"


def test_handle_stop_flushes_queue_and_triggers_brain_sleep(tmp_path):
    hook_helper = _load_module(HOOK_HELPER_PATH, "brainbar_hook")
    queue_path = tmp_path / ".memory-queue.jsonl"
    queue_path.write_text('{"hook_event":"PostToolUse","tool_name":"Read"}\n')
    observed_calls: list[tuple[str, dict]] = []

    output = hook_helper.handle_stop(
        {"session_id": "sess-123", "cwd": "/tmp/project"},
        queue_path=queue_path,
        flush_queue_fn=lambda path: 1,
        invoke_tool=lambda tool, arguments: observed_calls.append((tool, arguments)) or {"ok": True},
    )

    assert observed_calls == [("brain_sleep", {"mode": "plugin-stop", "session_id": "sess-123", "cwd": "/tmp/project"})]
    assert "queued 1 memory events" in output["systemMessage"].lower()


def test_handle_stop_does_not_crash_when_brain_sleep_is_unavailable(tmp_path):
    hook_helper = _load_module(HOOK_HELPER_PATH, "brainbar_hook")
    queue_path = tmp_path / ".memory-queue.jsonl"
    queue_path.write_text('{"hook_event":"PostToolUse","tool_name":"Read"}\n')

    def missing_sleep(_tool, _arguments):
        raise RuntimeError("Unknown tool: brain_sleep")

    output = hook_helper.handle_stop(
        {"session_id": "sess-123", "cwd": "/tmp/project"},
        queue_path=queue_path,
        flush_queue_fn=lambda path: 1,
        invoke_tool=missing_sleep,
    )

    assert "queued 1 memory events" in output["systemMessage"].lower()
    assert "brain_sleep unavailable" in output["systemMessage"].lower()
