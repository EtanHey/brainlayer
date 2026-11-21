"""Slow read-only text coverage check for T3; run daily at background priority."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..dedupe import normalized_exact_hash
from ..job_alerts import alert_path, report
from ..t3_thread_tables import _active_clause, plan_t3_thread_rows
from .t3 import T3Reader, _message_chunks
from .t3_bookmark import save
from .t3_ownership import GRACE_SECONDS, session_admits, timestamp


def audit_t3(state_db, *, db_path, health_path, projection_version=None):
    """Count old admissible messages without a text twin in either owning layer.

    Both databases stay read-only. Only the audit JSON and job alert are written.
    An unreadable input is an error, never uncovered=0.
    """
    from ..t3_sources import resolve_t3_sources

    output = Path(health_path).with_name("t3-audit.json")
    alert = alert_path(db_path=Path(db_path))
    result = {"updated_at": datetime.now(timezone.utc).isoformat(), "state": "error"}
    conn = None
    try:
        conn = sqlite3.connect(f"{Path(db_path).absolute().as_uri()}?mode=ro", uri=True, timeout=5)
        conn.execute("PRAGMA query_only=ON")
        active = _active_clause(conn)
        checked = uncovered = 0
        now = datetime.now(timezone.utc).timestamp()
        seen = set()
        for source, version in resolve_t3_sources(state_db, projection_version):
            reader = T3Reader(source, health_path=None, projection_version=version)
            if version == 2:
                reader._catalog_request = "all"
            threads = reader.read_threads(report_success=False)
            plan = plan_t3_thread_rows(conn, getattr(reader, "_catalog", ()))
            conversations = {}
            for row in plan["sessions"]:
                if row[5]:
                    conversations.setdefault(row[0], set()).add(row[5])
            for thread in threads:
                ids = {thread.thread_id, *conversations.get(thread.thread_id, ())}
                hashes = {
                    normalized_exact_hash(content)
                    for (content,) in conn.execute(
                        f"SELECT content FROM chunks WHERE conversation_id IN (SELECT value FROM json_each(?)) AND {active}",
                        (json.dumps(sorted(ids)),),
                    )
                }
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
                    from ..index_new import _scrubbed_chunk

                    expected = [_scrubbed_chunk(c).content for c in _message_chunks(thread, message)]
                    if not expected:
                        continue
                    checked += 1
                    if not all(normalized_exact_hash(text) in hashes for text in expected):
                        uncovered += 1
        result.update(state="measured", checked=checked, uncovered=uncovered)
        save(output, result)
        report("t3-audit", f"T3 text coverage: uncovered = {uncovered}" if uncovered else None, path=alert)
        return result
    except Exception:
        save(output, result)
        report("t3-audit", "T3 text coverage audit failed; check the job log", path=alert)
        raise
    finally:
        if conn is not None:
            conn.close()
