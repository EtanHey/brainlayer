"""Session-scoped dedup coordination for BrainLayer hooks.

Both SessionStart and UserPromptSubmit hooks read/write a shared JSON file
at /tmp/brainlayer_session_{session_id}.json to avoid injecting the same
chunk twice.

File format:
{
  "schema_version": 1,
  "session_id": "abc123",
  "is_handoff_session": false,
  "injected_chunks": [
    {"chunk_id": "manual-abc123", "source_hook": "SessionStart", "brief": "..."}
  ],
  "injected_ids_set": ["manual-abc123"],
  "total_tokens_injected": 200
}

Design:
- Atomic writes via write-to-tmp + rename
- File lock (fcntl.flock) protects read-modify-write sequences
- Graceful degradation: if file is missing/corrupt, hooks fall back to no-dedup
- Auto-cleans on reboot (/tmp)
"""

import fcntl
import hashlib
import json
import os
import tempfile

_SCHEMA_VERSION = 1
_COORD_DIR = "/tmp"

# Multi-word patterns only — bare agent names like "coachclaude" would
# suppress search on any prompt that merely mentions the agent.
HANDOFF_KEYWORDS = frozenset(
    {
        "handoff",
        "session-handoff",
        "pick up where",
        "resume session",
        "handoff to coachclaude",
        "handoff to orcclaude",
        "handoff to brainclaude",
    }
)


def coord_path(session_id: str) -> str:
    """Return the coordination file path for a session."""
    safe_id = hashlib.sha256(session_id.encode()).hexdigest()[:16]
    return os.path.join(_COORD_DIR, f"brainlayer_session_{safe_id}.json")


def _lock_path(session_id: str) -> str:
    """Return the lock file path for a session."""
    safe_id = hashlib.sha256(session_id.encode()).hexdigest()[:16]
    return os.path.join(_COORD_DIR, f"brainlayer_session_{safe_id}.lock")


def read_coord(session_id: str) -> dict | None:
    """Read the coordination file. Returns None on missing/corrupt."""
    path = coord_path(session_id)
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if data.get("schema_version") != _SCHEMA_VERSION:
            return None
        return data
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def write_coord(session_id: str, data: dict) -> bool:
    """Atomically write the coordination file. Returns True on success."""
    path = coord_path(session_id)
    data["schema_version"] = _SCHEMA_VERSION
    data["session_id"] = session_id
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=_COORD_DIR, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.rename(tmp_path, path)
        return True
    except OSError:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return False


def register_chunks(
    session_id: str,
    chunk_ids: list[str],
    source_hook: str,
    briefs: list[str] | None = None,
    token_estimate: int = 0,
) -> dict:
    """Register injected chunks in the coordination file.

    Uses file locking to prevent concurrent read-modify-write races
    between SessionStart and UserPromptSubmit hooks.
    Returns the updated coordination data.
    """
    lock = _lock_path(session_id)
    lock_fd = None
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_RDWR)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        data = read_coord(session_id) or {
            "session_id": session_id,
            "is_handoff_session": False,
            "injected_chunks": [],
            "injected_ids_set": [],
            "total_tokens_injected": 0,
        }

        existing_ids = set(data.get("injected_ids_set", []))
        new_count = 0

        for i, cid in enumerate(chunk_ids):
            if cid in existing_ids:
                continue
            entry = {"chunk_id": cid, "source_hook": source_hook}
            if briefs and i < len(briefs):
                entry["brief"] = briefs[i][:100]
            data["injected_chunks"].append(entry)
            existing_ids.add(cid)
            new_count += 1

        data["injected_ids_set"] = sorted(existing_ids)
        # Only add token estimate proportional to new chunks
        if chunk_ids and new_count > 0:
            per_chunk = token_estimate / len(chunk_ids) if token_estimate else 0
            data["total_tokens_injected"] = data.get("total_tokens_injected", 0) + int(per_chunk * new_count)

        write_coord(session_id, data)
        return data
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


def get_injected_ids(session_id: str) -> set[str]:
    """Return the set of already-injected chunk IDs for this session."""
    data = read_coord(session_id)
    if not data:
        return set()
    return set(data.get("injected_ids_set", []))


def is_handoff_prompt(prompt: str) -> bool:
    """Detect handoff-related prompts that should skip auto-search."""
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in HANDOFF_KEYWORDS)


def mark_handoff_session(session_id: str) -> None:
    """Mark this session as a handoff session in the coordination file."""
    lock = _lock_path(session_id)
    lock_fd = None
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_RDWR)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        data = read_coord(session_id) or {
            "session_id": session_id,
            "is_handoff_session": False,
            "injected_chunks": [],
            "injected_ids_set": [],
            "total_tokens_injected": 0,
        }
        data["is_handoff_session"] = True
        write_coord(session_id, data)
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
