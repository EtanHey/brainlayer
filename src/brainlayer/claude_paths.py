"""Helpers for Claude Code transcript path metadata."""

import re
from pathlib import Path

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def extract_claude_conversation_id(source_file: str) -> str | None:
    """Extract Claude Code's resumable conversation UUID from a JSONL source path."""
    p = Path(source_file)
    if UUID_RE.match(p.stem):
        return p.stem
    for parent in p.parents:
        if UUID_RE.match(parent.name):
            return parent.name
    return None
