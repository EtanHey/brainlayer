#!/usr/bin/env python3
"""Daily P0 audit-recursion counter for the post-PR287 longitudinal window."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from brainlayer.paths import get_db_path

SINCE = "2026-05-16T16:56:00+03:00"
DEFAULT_LOG_DIR = Path.home() / ".brainlayer-p0-counter"

P0_SQL = """
SELECT date(datetime(created_at)) AS day, COUNT(*) AS new_audit_chunks
FROM chunks
WHERE datetime(created_at) > datetime(?)
  AND (content GLOB '┌─brain_search:*'
       OR content GLOB '┌─brain_*:*'
       OR content GLOB '┌─ brain_search:*'
       OR content GLOB '┌─ brain_*:*'
       OR content GLOB '*"jsonrpc"*"2.0"*'
       OR content GLOB '*MCP BrainLayer Memory: Invalid JSON-RPC*')
GROUP BY day
ORDER BY day;
""".strip()


def _db_path() -> Path:
    return get_db_path().expanduser()


def _log_dir() -> Path:
    return Path(os.environ.get("BRAINLAYER_P0_COUNTER_DIR", DEFAULT_LOG_DIR)).expanduser()


def run_count(db_path: Path, since: str = SINCE) -> list[dict[str, object]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    try:
        conn.execute("PRAGMA query_only=true")
        conn.execute("PRAGMA busy_timeout=5000")
        return [
            {"day": day, "new_audit_chunks": int(count)} for day, count in conn.execute(P0_SQL, (since,)).fetchall()
        ]
    finally:
        conn.close()


def build_payload(db_path: Path, now: datetime | None = None) -> dict[str, object]:
    rows = run_count(db_path)
    total = sum(int(row["new_audit_chunks"]) for row in rows)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    since = datetime.fromisoformat(SINCE)
    elapsed_seconds = max(0.0, (now - since).total_seconds())
    elapsed_days = int(elapsed_seconds // 86400)
    verdict_ready = elapsed_days >= 7
    structural_fix = verdict_ready and total == 0
    return {
        "generated_at": now.isoformat(),
        "since": SINCE,
        "db_path": str(db_path),
        "sql": P0_SQL,
        "sql_params": [SINCE],
        "rows": rows,
        "total_new_audit_chunks": total,
        "elapsed_days": elapsed_days,
        "verdict_ready": verdict_ready,
        "structural_fix_p_lt_0_001": structural_fix,
        "brain_store_verdict_content": (
            f"P0 longitudinal counter verdict: 0 audit-recursion chunks across 7 days "
            f"since {SINCE}; structural fix p<0.001 per R51."
            if structural_fix
            else None
        ),
    }


def main() -> int:
    db_path = _db_path()
    if not db_path.exists():
        raise SystemExit(f"BrainLayer DB not found: {db_path}")

    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(db_path)
    output_path = log_dir / f"{datetime.now().date().isoformat()}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), **payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
