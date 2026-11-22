"""Admission for Claude Code's durable queued user input."""

from typing import Any


def queued_user_text(entry: dict[str, Any]) -> str | None:
    """Return enqueue text only; queue lifecycle records carry no user message."""
    if entry.get("type") != "queue-operation" or entry.get("operation") != "enqueue":
        return None
    content = entry.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
    else:
        return None
    return text if text.strip() else None
