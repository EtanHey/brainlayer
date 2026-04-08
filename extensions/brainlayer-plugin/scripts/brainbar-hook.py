#!/usr/bin/env python3
"""Thin Claude Code hook adapter for BrainBar MCP."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

MCP_SOCKET_PATH = "/tmp/brainbar.sock"


def _plugin_root() -> Path:
    env_root = os.environ.get("CLAUDE_PLUGIN_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parents[1]


def _queue_path() -> Path:
    override = os.environ.get("BRAINLAYER_PLUGIN_QUEUE")
    if override:
        return Path(override)
    plugin_data = os.environ.get("CLAUDE_PLUGIN_DATA")
    base_dir = Path(plugin_data) if plugin_data else _plugin_root()
    return base_dir / ".memory-queue.jsonl"


def _queue_module():
    script_path = _plugin_root() / "scripts" / "queue-flusher.py"
    spec = importlib.util.spec_from_file_location("queue_flusher", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_message(stream, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    stream.write(body)
    stream.flush()


def _read_message(stream) -> dict[str, Any]:
    headers = b""
    while b"\r\n\r\n" not in headers:
        chunk = stream.read(1)
        if not chunk:
            raise RuntimeError("unexpected EOF while reading MCP headers")
        headers += chunk

    header_text = headers.decode("utf-8")
    content_length = None
    for line in header_text.split("\r\n"):
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())
            break

    if content_length is None:
        raise RuntimeError("missing Content-Length header")

    body = stream.read(content_length)
    if len(body) != content_length:
        raise RuntimeError("short MCP body read")
    return json.loads(body.decode("utf-8"))


def call_mcp_tool(tool_name: str, arguments: dict[str, Any], socket_path: str = MCP_SOCKET_PATH) -> dict[str, Any]:
    proc = subprocess.Popen(
        ["socat", "STDIO", f"UNIX-CONNECT:{socket_path}"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    try:
        _write_message(
            proc.stdin,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "brainlayer-plugin", "version": "0.1.0"},
                },
            },
        )
        _read_message(proc.stdout)

        _write_message(proc.stdin, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _write_message(
            proc.stdin,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )

        while True:
            message = _read_message(proc.stdout)
            if message.get("id") == 2:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})
    finally:
        proc.kill()
        proc.wait(timeout=1)


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if isinstance(result.get("summary"), str):
            return result["summary"]
        content = result.get("content")
        if isinstance(content, list):
            texts = [item.get("text", "") for item in content if isinstance(item, dict)]
            text = "\n".join(part for part in texts if part).strip()
            if text:
                return text
    return json.dumps(result, ensure_ascii=True)


def format_additional_context(text: str) -> dict[str, str]:
    return {"additionalContext": text.strip()}


def build_queue_event(hook_event: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    tool_name = payload.get("tool_name", "")
    if tool_name.startswith("mcp__brainlayer__"):
        return None

    is_failure = hook_event == "PostToolUseFailure"
    response = payload.get("tool_response") or payload.get("tool_result") or {}
    return {
        "hook_event": hook_event,
        "session_id": payload.get("session_id"),
        "cwd": payload.get("cwd"),
        "tool_name": tool_name,
        "tool_input": payload.get("tool_input", {}),
        "tool_response": response,
        "importance": 8 if is_failure else 5,
        "tags": ["claude-code", "tool-observation"] + (["error"] if is_failure else []),
        "captured_at": time.time(),
    }


def handle_session_start(
    payload: dict[str, Any],
    invoke_tool: Callable[[str, dict[str, Any]], Any] = call_mcp_tool,
    deadline_ms: int = 200,
) -> dict[str, str]:
    started = time.perf_counter()
    result = invoke_tool("brain_recall", {"mode": "context", "session_id": payload.get("session_id")})
    text = _result_text(result).strip()
    if not text:
        return {}
    elapsed_ms = (time.perf_counter() - started) * 1000
    if elapsed_ms > deadline_ms:
        return {}
    return format_additional_context(text)


def _send_queue_batch_to_brainbar(batch: list[dict[str, Any]]) -> None:
    for event in batch:
        arguments = {
            "content": json.dumps(
                {
                    "tool_name": event.get("tool_name"),
                    "tool_input": event.get("tool_input"),
                    "tool_response": event.get("tool_response"),
                },
                ensure_ascii=True,
            ),
            "tags": event.get("tags", []),
            "importance": event.get("importance", 5),
            "context": event.get("cwd"),
            "session_id": event.get("session_id"),
        }
        call_mcp_tool("brain_store", arguments)


def handle_tool_event(payload: dict[str, Any], hook_event: str, queue_path: Path | None = None) -> None:
    event = build_queue_event(hook_event, payload)
    if event is None:
        return
    queue_module = _queue_module()
    queue_module.append_queue_event(queue_path or _queue_path(), event)


def handle_stop(
    payload: dict[str, Any],
    queue_path: Path | None = None,
    flush_queue_fn: Callable[[Path], int] | None = None,
    invoke_tool: Callable[[str, dict[str, Any]], Any] = call_mcp_tool,
) -> dict[str, str]:
    resolved_queue = queue_path or _queue_path()
    if flush_queue_fn is None:
        queue_module = _queue_module()
        flushed = queue_module.flush_queue(resolved_queue, _send_queue_batch_to_brainbar)
    else:
        flushed = flush_queue_fn(resolved_queue)

    sleep_note = "triggered brain_sleep"
    try:
        invoke_tool(
            "brain_sleep",
            {
                "mode": "plugin-stop",
                "session_id": payload.get("session_id"),
                "cwd": payload.get("cwd"),
            },
        )
    except RuntimeError as exc:
        if "brain_sleep" not in str(exc):
            raise
        sleep_note = "brain_sleep unavailable"

    return {"systemMessage": f"BrainLayer queued {flushed} memory events and {sleep_note}."}


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    payload = json.loads(sys.stdin.read() or "{}")

    if action == "session-start":
        output = handle_session_start(payload)
        if output:
            print(json.dumps(output))
        return 0
    if action == "tool-observe":
        handle_tool_event(payload, "PostToolUse")
        return 0
    if action == "tool-error":
        handle_tool_event(payload, "PostToolUseFailure")
        return 0
    if action == "session-stop":
        output = handle_stop(payload)
        if output:
            print(json.dumps(output))
        return 0

    print(json.dumps({"systemMessage": f"Unknown brainbar-hook action: {action}"}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
