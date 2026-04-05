"""Mine frustration chunks into benchmark-ready query/qrels pairs."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brainlayer._helpers import _escape_fts5_query
from brainlayer.paths import get_db_path

FRUSTRATION_PATTERNS = [
    r"\byou should (?:know|remember)\b",
    r"\bwe (?:discussed|talked about|decided)\b",
    r"\bI (?:already|just) (?:told|said|mentioned)\b",
    r"\bwhy (?:don't|didn't) you (?:know|remember)\b",
    r"\b(?:wrong|incorrect|not right)\b",
    r"\b(?:again|already|before)\?\b",
    r"אתה צריך לדעת",
    r"כבר דיברנו על",
    r"למה אתה לא זוכר",
]

CASUAL_REPEAT_PATTERNS = [
    r"\bcan you .*again\??$",
    r"\bsend .*again\??$",
    r"\btry again\??$",
]

ENGLISH_EXPECTED_PATTERNS = [
    r"we (?:discussed|talked about|decided)\s+(?P<topic>.+?)(?:[.!?]|$)",
    r"you should (?:know|remember)\s+(?P<topic>.+?)(?:[.!?]|$)",
    r"i (?:already|just) (?:told|said|mentioned)\s+(?P<topic>.+?)(?:[.!?]|$)",
]

HEBREW_EXPECTED_PATTERNS = [
    r"כבר דיברנו על\s+(?P<topic>.+?)(?:[.!?]|$)",
    r"למה אתה לא זוכר\??\s*(?P<topic>.+?)(?:[.!?]|$)",
]


@dataclass(frozen=True)
class FrustrationPair:
    query_id: str
    query_text: str
    expected_result: str
    chunk_ids: list[str]


def detect_frustration(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    lowered = normalized.lower()
    if not normalized:
        return False
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in CASUAL_REPEAT_PATTERNS):
        return False
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in FRUSTRATION_PATTERNS)


def load_conversation_context(conn: sqlite3.Connection, chunk_id: str, window: int = 2) -> list[dict[str, Any]]:
    row = conn.execute(
        "SELECT conversation_id, position FROM chunks WHERE id = ?",
        (chunk_id,),
    ).fetchone()
    if row is None:
        return []

    conversation_id, position = row
    if conversation_id is None or position is None:
        return []
    rows = conn.execute(
        """
        SELECT id, content, conversation_id, position, sentiment_label, tags
        FROM chunks
        WHERE conversation_id = ?
          AND position BETWEEN ? AND ?
        ORDER BY position
        """,
        (conversation_id, position - window, position + window),
    ).fetchall()

    context: list[dict[str, Any]] = []
    for item in rows:
        context.append(
            {
                "id": item[0],
                "content": item[1],
                "conversation_id": item[2],
                "position": item[3],
                "sentiment_label": item[4],
                "tags": json.loads(item[5]) if item[5] else [],
            }
        )
    return context


def extract_query_from_context(context: list[dict[str, Any]], frustration_index: int) -> str:
    if not context:
        return ""

    for offset in range(frustration_index - 1, -1, -1):
        candidate = context[offset]["content"].strip()
        if candidate and not detect_frustration(candidate):
            return candidate
    return ""


def extract_expected_result(frustration_text: str, context: list[dict[str, Any]] | None = None) -> str:
    stripped = frustration_text.strip()

    for pattern in ENGLISH_EXPECTED_PATTERNS:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            return match.group("topic").strip(" .!?")

    for pattern in HEBREW_EXPECTED_PATTERNS:
        match = re.search(pattern, stripped)
        if match:
            return match.group("topic").strip(" .!?")

    clauses = [part.strip() for part in re.split(r"[.!?]+", stripped) if part.strip()]
    if len(clauses) > 1:
        return clauses[-1]
    if context:
        for entry in reversed(context):
            if entry["content"] != frustration_text and not detect_frustration(entry["content"]):
                return entry["content"].strip()
    return stripped


def find_matching_chunks(
    conn: sqlite3.Connection,
    expected_result: str,
    *,
    limit: int = 5,
    exclude_chunk_ids: set[str] | None = None,
) -> list[str]:
    exclude_chunk_ids = exclude_chunk_ids or set()
    terms = [term for term in re.findall(r"[\w\u0590-\u05FF-]+", expected_result.lower()) if len(term) >= 3]
    if not terms:
        return []

    existing_tables = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
    }

    if "chunks_fts" in existing_tables:
        escaped_query = _escape_fts5_query(" ".join(terms), match_mode="or")
        if escaped_query:
            rows = conn.execute(
                """
                SELECT f.chunk_id
                FROM chunks_fts f
                WHERE chunks_fts MATCH ?
                LIMIT ?
                """,
                (escaped_query, limit * 3),
            ).fetchall()
            chunk_ids = [chunk_id for (chunk_id,) in rows if chunk_id not in exclude_chunk_ids]
            if chunk_ids:
                return chunk_ids[:limit]

    like_clauses = " OR ".join("LOWER(content) LIKE ?" for _ in terms)
    params: list[Any] = [f"%{term}%" for term in terms]
    rows = conn.execute(
        f"""
        SELECT id, content
        FROM chunks
        WHERE {like_clauses}
        LIMIT ?
        """,
        (*params, max(limit * 10, 20)),
    ).fetchall()

    scored: list[tuple[int, str]] = []
    for chunk_id, content in rows:
        if chunk_id in exclude_chunk_ids:
            continue
        content_lower = content.lower()
        score = sum(term in content_lower for term in terms)
        if score > 0:
            scored.append((score, chunk_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk_id for _score, chunk_id in scored[:limit]]


def generate_qrels(pairs: list[FrustrationPair]) -> dict[str, dict[str, int]]:
    return {pair.query_id: {chunk_id: 3 for chunk_id in pair.chunk_ids} for pair in pairs}


def append_qrels(
    existing: dict[str, dict[str, int]], new_entries: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    merged = dict(existing)
    merged.update(new_entries)
    return merged


def write_qrels_file(path: str | Path, qrels: dict[str, dict[str, int]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(qrels, indent=2, sort_keys=True) + "\n")


def mine_frustration_pairs(conn: sqlite3.Connection) -> list[FrustrationPair]:
    rows = conn.execute(
        """
        SELECT id, content
        FROM chunks
        WHERE sentiment_label = 'frustration'
           OR COALESCE(tags, '') LIKE '%frustration%'
           OR COALESCE(tags, '') LIKE '%expectation-failure%'
           OR COALESCE(tags, '') LIKE '%user-correction%'
        ORDER BY id
        """
    ).fetchall()

    pairs: list[FrustrationPair] = []
    next_id = 1
    for chunk_id, content in rows:
        if not detect_frustration(content):
            continue

        context = load_conversation_context(conn, chunk_id)
        frustration_index = next((idx for idx, item in enumerate(context) if item["id"] == chunk_id), -1)
        if frustration_index < 0:
            continue

        query_text = extract_query_from_context(context, frustration_index)
        if not query_text:
            continue

        expected_result = extract_expected_result(content, context)
        matches = find_matching_chunks(conn, expected_result, exclude_chunk_ids={item["id"] for item in context})
        if not matches:
            continue

        pairs.append(
            FrustrationPair(
                query_id=f"frustration_{next_id:03d}",
                query_text=query_text,
                expected_result=expected_result,
                chunk_ids=matches,
            )
        )
        next_id += 1

    return pairs


def mine_frustration_qrels(
    db_path: str | Path | None = None,
) -> tuple[list[FrustrationPair], dict[str, dict[str, int]]]:
    resolved_db_path = Path(db_path) if db_path is not None else get_db_path()
    with sqlite3.connect(resolved_db_path) as conn:
        pairs = mine_frustration_pairs(conn)
    return pairs, generate_qrels(pairs)
