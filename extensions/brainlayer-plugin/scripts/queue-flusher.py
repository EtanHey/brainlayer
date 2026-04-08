#!/usr/bin/env python3
"""Append-only queue for plugin tool observations."""

from __future__ import annotations

import fcntl
import json
import sys
from pathlib import Path
from typing import Any, Callable


def append_queue_event(queue_path: Path, event: dict[str, Any]) -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_queue_events(queue_path: Path) -> list[dict[str, Any]]:
    if not queue_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with queue_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def flush_queue(queue_path: Path, sender: Callable[[list[dict[str, Any]]], None], batch_size: int = 50) -> int:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    if not queue_path.exists():
        queue_path.write_text("", encoding="utf-8")
        return 0

    with queue_path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        raw_lines = [line for line in handle.readlines() if line.strip()]
        events = [json.loads(line) for line in raw_lines]
        try:
            for start in range(0, len(events), batch_size):
                sender(events[start : start + batch_size])
        except Exception:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return 0

        handle.seek(0)
        handle.truncate(0)
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return len(events)


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: queue-flusher.py <append|flush> <queue_path>", file=sys.stderr)
        return 1

    command = sys.argv[1]
    queue_path = Path(sys.argv[2])

    if command == "append":
        append_queue_event(queue_path, json.loads(sys.stdin.read() or "{}"))
        return 0

    if command == "flush":
        from importlib import util

        hook_path = queue_path.resolve().parents[0]
        plugin_root = hook_path.parent if hook_path.name == "scripts" else None
        if plugin_root is None:
            plugin_root = Path(__file__).resolve().parents[1]
        spec = util.spec_from_file_location("brainbar_hook", plugin_root / "scripts" / "brainbar-hook.py")
        module = util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        flushed = flush_queue(queue_path, module._send_queue_batch_to_brainbar)
        print(flushed)
        return 0

    print(f"unknown command: {command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
