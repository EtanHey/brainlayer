"""Stage 1: Extract system prompts and detect conversation continuations."""

import hashlib
from pathlib import Path
from typing import Iterator

import orjson


def hash_content(content: str) -> str:
    """SHA-256 hash for content-addressable storage."""
    return hashlib.sha256(content.encode()).hexdigest()


def parse_jsonl(file_path: Path) -> Iterator[dict]:
    """Parse a JSONL file, yielding each line as a dict."""
    with open(file_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError:
                continue  # Skip malformed lines


def extract_system_prompts(conversations_dir: Path) -> dict[str, str]:
    """
    Extract and deduplicate system prompts from all conversations.

    Returns: {hash: prompt_content} mapping for content-addressable storage.
    """
    prompts: dict[str, str] = {}

    for jsonl_file in conversations_dir.rglob("*.jsonl"):
        for entry in parse_jsonl(jsonl_file):
            # First user message often contains system prompt
            if entry.get("type") == "user":
                message = entry.get("message", {})
                content = message.get("content", "")

                # Heuristic: system prompts are typically >2000 chars
                # and contain CLAUDE.md or system instructions
                if len(content) > 2000 and ("CLAUDE.md" in content or "system" in content.lower()):
                    content_hash = hash_content(content)
                    if content_hash not in prompts:
                        prompts[content_hash] = content

    return prompts


def detect_continuations(conversations_dir: Path) -> list[list[Path]]:
    """
    Detect conversation continuations across multiple JSONL files.

    Uses:
    - Session ID matching
    - Temporal proximity (within 30 min)
    - Same project directory

    Returns: List of continuation chains (each chain is a list of file paths).
    """
    # TODO: Implement continuation detection
    # For now, treat each file as independent
    return [[f] for f in conversations_dir.rglob("*.jsonl")]
