"""Atomic replay state and bounded, read-only T3 event selection."""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
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
    head = conn.execute("SELECT coalesce(max(sequence), 0) FROM orchestration_events").fetchone()[0]
    metadata = dict(conn.execute("SELECT projection_name, last_sequence FROM orchestration_v2_projection_metadata"))
    ceiling = min(head, metadata.get("thread-projections", head))
    same_source = previous.get("identity") == source_identity
    pending = previous.get("pending_to") if same_source else None
    resume_pending = type(pending) is int and pending <= ceiling
    if resume_pending:
        ceiling = pending
    stamp = conn.execute(
        "SELECT occurred_at FROM orchestration_events WHERE sequence <= ? ORDER BY sequence DESC LIMIT 1",
        (ceiling,),
    ).fetchone()
    occurred_at = stamp[0] if stamp else previous.get("occurred_at")
    start = previous.get("last_sequence")
    valid = (
        same_source
        and type(start) is int
        and "event-log-pruned-through" in metadata
        and metadata.get("event-log-pruned-through", 0) <= start <= ceiling <= head
    )
    cutoff = None
    if (
        same_source
        and type(start) is int
        and start <= ceiling <= head
        and metadata.get("event-log-pruned-through", 0) > start
        and isinstance(previous.get("occurred_at"), str)
    ):
        try:
            stamp_time = datetime.fromisoformat(previous["occurred_at"].replace("Z", "+00:00"))
            if stamp_time.tzinfo is not None:
                cutoff = (stamp_time.astimezone(timezone.utc) - timedelta(minutes=10)).isoformat()
        except (ValueError, OverflowError):
            pass  # An unusable timestamp requires the full safety read.
    info = {
        "mode": "incremental" if valid else "bounded" if cutoff is not None else "full",
        "events_read": 0,
        "bookmark_from": start,
        "bookmark_to": ceiling,
        "bookmark_held": None,
    }
    messages, threads = set(), set()
    if cutoff is not None:
        messages.update(
            row[0]
            for row in conn.execute(
                "SELECT message_id FROM orchestration_v2_projection_messages "
                "WHERE julianday(updated_at) > julianday(?)",
                (cutoff,),
            )
        )
    elif valid:
        rows = conn.execute(
            "SELECT stream_id FROM orchestration_events WHERE sequence > ? AND sequence <= ?",
            (start, ceiling),
        )
        for (stream_id,) in rows:
            info["events_read"] += 1
            if isinstance(stream_id, str):
                threads.add(stream_id)
    return (
        info,
        messages,
        threads,
        {
            "identity": source_identity,
            "last_sequence": ceiling,
            "occurred_at": occurred_at,
            "resume_pending": resume_pending,
        },
    )
