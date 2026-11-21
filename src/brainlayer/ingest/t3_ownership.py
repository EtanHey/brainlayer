"""Conservative, recomputed session ownership for T3 text."""

from datetime import datetime, timezone

from ..t3_thread_tables import ELIGIBLE_DRIVERS, plan_t3_thread_rows

GRACE_SECONDS = 30 * 60
WATERMARK_MARGIN_SECONDS = 5


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (AttributeError, ValueError, OverflowError):
        return None


def session_admits(message):
    """Run the watcher's pure admission functions, including its thinking path.

    Exceptions are uncertainty: the caller must write rather than discard.
    """
    from ..pipeline.chunk import chunk_content
    from ..pipeline.classify import classify_content, classify_thinking
    from ..watcher_bridge import _strip_system_reminders, should_skip_chunk_content, should_skip_entry
    from .t3 import _is_thinking_message

    block = (
        {"type": "thinking", "thinking": message.text}
        if _is_thinking_message(message)
        else {"type": "text", "text": message.text}
    )
    entry = {"type": message.role, "message": {"role": message.role, "content": [block]}}
    thinking = classify_thinking(entry)
    classified = None if should_skip_entry(entry) else classify_content(entry)
    return any(
        not should_skip_chunk_content(_strip_system_reminders(chunk.content))
        for item in (classified, thinking)
        if item is not None
        for chunk in chunk_content(item)
    )


def ownership_facts(conn, entries):
    """Use this run's complete catalog and active session rows, never stored ownership."""
    entries = list(entries)
    plan = plan_t3_thread_rows(conn, entries)
    water = {row[0]: timestamp(row[11]) for row in plan["threads"]}
    facts = {}
    for entry in entries:
        native = [p for p in entry.provider_threads if p.native_session_id]
        if len(native) == 1 and native[0].driver in ELIGIBLE_DRIVERS:
            facts[entry.thread_id] = water.get(entry.thread_id)
    return facts


def message_owner(message, water, *, now=None):
    """Return covered, held, dropped, short_user, session_late, or uncertain."""
    try:
        admitted = session_admits(message)
    except Exception:
        return "uncertain"
    if message.role == "user" and len(message.text.strip()) < 20:
        return "short_user"
    if not admitted:
        return "dropped"
    created = timestamp(message.created_at)
    if created is None:
        return "uncertain"
    if water is not None and created <= water + WATERMARK_MARGIN_SECONDS:
        return "covered"
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    return "held" if now - created < GRACE_SECONDS else "session_late"
