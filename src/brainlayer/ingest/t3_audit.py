"""Slow read-only text coverage check for T3; run daily at background priority."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..dedupe import normalize_for_exact_hash
from ..job_alerts import alert_path, report
from ..system_prompt_guard import looks_like_system_prompt
from ..t3_thread_tables import ELIGIBLE_DRIVERS, _codex_conversations
from .t3 import T3Reader, _chunk_id, _message_chunks
from .t3_bookmark import save
from .t3_ownership import GRACE_SECONDS, normalized_message_text, session_admits, timestamp


def _text_covered(needle, stored_texts):
    """Cover normalized text across stored chunks, independent of split/order/overlap.

    A whole-message containment is enough. Otherwise each stored piece must be a
    substring of the message and the union must cover it without a content gap.
    Whitespace at a chunk boundary is immaterial; missing text is not.
    """
    if not needle:
        return False
    intervals = []
    for text in stored_texts:
        if needle in text:
            return True
        if not text:
            continue
        start = needle.find(text)
        while start >= 0:
            intervals.append((start, start + len(text)))
            start = needle.find(text, start + 1)
    covered = 0
    for start, end in sorted(intervals):
        if start > covered and needle[covered:start].strip():
            return False
        covered = max(covered, end)
        if covered == len(needle):
            return True
    return False


def audit_t3(state_db, *, db_path, health_path, projection_version=None):
    """Count old messages without stored coverage, separating known policy exclusions.

    Both databases stay read-only. Only the audit JSON and job alert are written.
    An unreadable input is an error, never uncovered=0.
    Lifecycle-managed rows still prove that text was stored. A gap rejected by
    T3's actual system-prompt guard is reported as known, never backfilled around
    that guard or hidden in an identity allowlist. All other gaps alarm as new.
    """
    from ..t3_sources import resolve_t3_sources

    output = Path(health_path).with_name("t3-audit.json")
    alert = alert_path(db_path=Path(db_path))
    result = {"updated_at": datetime.now(timezone.utc).isoformat(), "state": "error"}
    conn = None
    try:
        conn = sqlite3.connect(f"{Path(db_path).absolute().as_uri()}?mode=ro", uri=True, timeout=5)
        conn.execute("PRAGMA query_only=ON")
        checked = uncovered = known = 0
        codex = None
        now = datetime.now(timezone.utc).timestamp()
        seen = set()
        for source, version in resolve_t3_sources(state_db, projection_version):
            reader = T3Reader(source, health_path=None, projection_version=version)
            if version == 2:
                reader._catalog_request = "all"
            threads = reader.read_threads(report_success=False)
            conversations = {}
            # Resolve against stored history, not just active rows: a wholly
            # archived session still owns evidence that its text was ingested.
            for entry in getattr(reader, "_catalog", ()):
                for provider in entry.provider_threads:
                    native = provider.native_session_id
                    if not native or provider.driver not in ELIGIBLE_DRIVERS:
                        continue
                    if provider.driver == "codex":
                        if codex is None:
                            codex = _codex_conversations(conn)
                        conversation = codex.get(native)
                    else:
                        conversation = native
                    if conversation:
                        conversations.setdefault(entry.thread_id, set()).add(conversation)
            for thread in threads:
                ids = {thread.thread_id, *conversations.get(thread.thread_id, ())}
                rows = conn.execute(
                    "SELECT id, content, metadata, source, conversation_id FROM chunks WHERE conversation_id IN"
                    " (SELECT value FROM json_each(?))",
                    (json.dumps(sorted(ids)),),
                ).fetchall()
                own_rows = [row for row in rows if row[3] == "t3" and row[4] == thread.thread_id]
                own_ids = {row[0] for row in own_rows}
                own_messages = {
                    metadata["message_id"]
                    for row in own_rows
                    if (metadata := json.loads(row[2] or "{}")).get("message_id")
                }
                texts = None
                for message in thread.messages:
                    identity = (thread.thread_id, message.message_id)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    created = timestamp(message.created_at)
                    if message.role not in {"user", "assistant"} or created is None or now - created < GRACE_SECONDS:
                        continue
                    if not session_admits(message) and not (message.role == "user" and len(message.text.strip()) < 20):
                        continue
                    if not message.text.strip():
                        continue
                    checked += 1
                    # T3 edits/chunker drift and deliberate lifecycle management
                    # do not erase this thread's stored identity.
                    if _chunk_id(thread, message, 0) in own_ids or message.message_id in own_messages:
                        continue
                    if texts is None:
                        texts = {}
                        for row in rows:
                            texts.setdefault(row[4], []).append(normalize_for_exact_hash(row[1]))
                    needle = normalized_message_text(message.text)
                    if not any(_text_covered(needle, values) for values in texts.values()):
                        candidates = _message_chunks(thread, message)
                        if candidates and all(
                            c.metadata.get("is_system_prompt") or looks_like_system_prompt(c.content)
                            for c in candidates
                        ):
                            known += 1
                        else:
                            uncovered += 1
        result.update(
            state="measured",
            checked=checked,
            uncovered=uncovered,
            uncovered_new=uncovered,
            uncovered_known=known,
            known_skip_reasons={"system_prompt_guard": known} if known else {},
        )
        save(output, result)
        report("t3-audit", f"T3 text coverage: uncovered_new = {uncovered}" if uncovered else None, path=alert)
        return result
    except Exception:
        save(output, result)
        report("t3-audit", "T3 text coverage audit failed; check the job log", path=alert)
        raise
    finally:
        if conn is not None:
            conn.close()
