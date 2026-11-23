"""Conservative, recomputed session ownership for T3 text."""

from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime, timezone

from ..dedupe import normalize_for_exact_hash
from ..t3_thread_tables import ELIGIBLE_DRIVERS, plan_t3_thread_rows

GRACE_SECONDS = 30 * 60
WATERMARK_MARGIN_SECONDS = 5


@dataclass(frozen=True)
class SessionCoverage:
    high_water: float | None
    texts: Collection[str]


def normalized_message_text(text):
    from ..pipeline.secret_scrub import scrub_for_storage
    from ..watcher_bridge import _strip_system_reminders

    return normalize_for_exact_hash(_strip_system_reminders(scrub_for_storage(text)[0]))


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
    """Use this run's catalog and stored session rows, never stored ownership."""
    from .t3_audit import TextCoverageIndex

    entries = list(entries)
    plan = plan_t3_thread_rows(conn, entries)
    water = {row[0]: timestamp(row[11]) for row in plan["threads"]}
    conversations = {row[0]: row[5] for row in plan["sessions"] if row[5]}
    evidence = {}
    facts = {}
    for entry in entries:
        native = [p for p in entry.provider_threads if p.native_session_id]
        if len(native) == 1 and native[0].driver in ELIGIBLE_DRIVERS:
            conversation = conversations.get(entry.thread_id)
            if conversation and conversation not in evidence:
                # A T3 row, alias into T3, or a row in another conversation is
                # not evidence that the session stored this message. Managed
                # session rows still prove storage and therefore remain eligible.
                evidence[conversation] = TextCoverageIndex(
                    normalize_for_exact_hash(row[0] or "")
                    for row in conn.execute(
                        "SELECT content FROM chunks WHERE conversation_id=? AND source != 't3'",
                        (conversation,),
                    )
                )
            facts[entry.thread_id] = SessionCoverage(water.get(entry.thread_id), evidence.get(conversation, ()))
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
    high_water = water.high_water if isinstance(water, SessionCoverage) else water
    if high_water is not None and created <= high_water + WATERMARK_MARGIN_SECONDS:
        from .t3_audit import _text_covered

        texts = water.texts if isinstance(water, SessionCoverage) else ()
        if _text_covered(normalized_message_text(message.text), texts):
            return "covered"
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    return "held" if now - created < GRACE_SECONDS else "session_late"
