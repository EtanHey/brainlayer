"""Atomic replay state and bounded, read-only T3 event selection."""

import json
import os
import tempfile
from pathlib import Path


def load(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(state, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def identity(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "dev": stat.st_dev, "inode": stat.st_ino}


def window(conn, previous, source_identity):
    # First data read in the transaction: later commits belong to the next run.
    ceiling = conn.execute("SELECT coalesce(max(sequence), 0) FROM orchestration_events").fetchone()[0]
    metadata = dict(conn.execute("SELECT projection_name, last_sequence FROM orchestration_v2_projection_metadata"))
    if metadata.get("thread-projections", ceiling) < ceiling:
        raise RuntimeError("T3 projection is behind its event log; retain bookmark for replay")
    start = previous.get("last_sequence")
    valid = (
        previous.get("identity") == source_identity
        and type(start) is int
        and "event-log-pruned-through" in metadata
        and metadata.get("event-log-pruned-through", 0) <= start <= ceiling
    )
    info = {
        "mode": "incremental" if valid else "full",
        "events_read": 0,
        "bookmark_from": start,
        "bookmark_to": ceiling,
        "bookmark_held": None,
    }
    messages, threads = set(), set()
    if valid:
        rows = conn.execute(
            "SELECT event_type, stream_id, CASE WHEN event_type='message.updated' "
            "THEN json_extract(payload_json, '$.id') END FROM orchestration_events "
            "WHERE sequence > ? AND sequence <= ?",
            (start, ceiling),
        )
        for event_type, stream_id, message_id in rows:
            info["events_read"] += 1
            if event_type == "message.updated":
                if not isinstance(message_id, str):
                    raise RuntimeError("T3 message event has no message id; retain bookmark")
                messages.add(message_id)
            elif event_type.startswith(("thread.", "provider-thread.")):
                threads.add(stream_id)
    return info, messages, threads, {"identity": source_identity, "last_sequence": ceiling}
