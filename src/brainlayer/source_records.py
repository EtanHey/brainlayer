from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


def ensure_chunk_source_records(conn: Any) -> None:
    conn.execute('''CREATE TABLE IF NOT EXISTS chunk_source_records (
        record_key TEXT PRIMARY KEY, canonical_chunk_id TEXT NOT NULL,
        source_file TEXT NOT NULL, record_start_offset INTEGER NOT NULL,
        record_end_offset INTEGER NOT NULL, raw_record_hash TEXT NOT NULL,
        session_id TEXT, observed_at TEXT NOT NULL,
        FOREIGN KEY(canonical_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE)''')
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_source_records_canonical ON chunk_source_records(canonical_chunk_id)")

def record_watcher_source(conn: Any, *, canonical_chunk_id: str, source_file: str, metadata: dict[str, Any]) -> None:
    start, end, digest = (metadata.get('_line_start_offset'), metadata.get('_line_end_offset'), metadata.get('_raw_record_hash'))
    if type(start) is not int or type(end) is not int or start < 0 or end <= start or not isinstance(digest, str) or len(digest) != 64:
        return
    try:
        int(digest, 16)
    except ValueError:
        return
    key = hashlib.sha256(json.dumps([canonical_chunk_id, source_file, start, end, digest], separators=(',', ':')).encode()).hexdigest()
    conn.execute("INSERT OR IGNORE INTO chunk_source_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (key, canonical_chunk_id, source_file, start, end, digest, metadata.get('session_id') if isinstance(metadata.get('session_id'), str) else None, datetime.now(timezone.utc).isoformat()))
