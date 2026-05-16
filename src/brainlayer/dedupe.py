"""Ingest-time duplicate detection and merge helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import apsw
import sqlite_vec

from .paths import get_db_path

logger = logging.getLogger(__name__)

SIMHASH_BITS = 64
SIMHASH_THRESHOLD = 3
SHINGLE_WIDTH = 4
MIN_SIMHASH_TOKENS = 8
ALIAS_GRACE_DAYS = 90
AUDIT_RETENTION_DAYS = 30

_ISO_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ][0-2]\d:[0-5]\d:[0-5]\d(?:\.\d+)?(?:Z|[+-][0-2]\d:?[0-5]\d)?\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


@dataclass(frozen=True)
class DedupeFields:
    content_hash: str
    simhash: str
    bands: tuple[str, str, str, str]


@dataclass(frozen=True)
class DuplicateHit:
    canonical_chunk_id: str
    mechanism: str
    hamming_distance: int


@dataclass(frozen=True)
class BackfillResult:
    scanned: int
    merged: int
    hashed: int


def normalize_for_dedupe(content: str) -> str:
    """Normalize text before exact hashing and SimHash tokenization."""
    without_timestamps = _ISO_TIMESTAMP_RE.sub(" ", content.lower())
    tokens = [token for token in _TOKEN_RE.findall(without_timestamps) if token not in _STOPWORDS]
    return " ".join(tokens)


def _numeric_tokens(content: str) -> set[str]:
    return {token for token in normalize_for_dedupe(content).split() if token.isdigit()}


def _numeric_tokens_conflict(left: str, right: str) -> bool:
    left_numbers = _numeric_tokens(left)
    right_numbers = _numeric_tokens(right)
    return bool(left_numbers or right_numbers) and left_numbers != right_numbers


def normalized_exact_hash(content: str) -> str:
    return hashlib.sha256(normalize_for_dedupe(content).encode("utf-8")).hexdigest()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _week_bucket(created_at: str | None) -> str | None:
    parsed = _parse_datetime(created_at)
    if parsed is None:
        return None
    year, week, _ = parsed.isocalendar()
    return f"{year:04d}-W{week:02d}"


def _stable_u64(feature: str) -> int:
    return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")


def _weighted_features(content: str, created_at: str | None = None) -> Iterable[tuple[str, float]]:
    tokens = normalize_for_dedupe(content).split()
    if not tokens:
        return []
    if len(tokens) < SHINGLE_WIDTH:
        features = [(f"tok:{token}", 1.0) for token in tokens]
    else:
        features = [
            (f"sh:{' '.join(tokens[index : index + SHINGLE_WIDTH])}", 1.0)
            for index in range(0, len(tokens) - SHINGLE_WIDTH + 1)
        ]
    bucket = _week_bucket(created_at)
    if bucket is not None:
        # Low relative to long chunks, but strong enough to separate short weekly milestones.
        features.extend((f"week:{bucket}:{bit}", 2.0) for bit in range(2))
    return features


def simhash64(content: str, *, created_at: str | None = None) -> int:
    weights = [0.0] * SIMHASH_BITS
    for feature, weight in _weighted_features(content, created_at):
        hashed = _stable_u64(feature)
        for bit in range(SIMHASH_BITS):
            if hashed & (1 << bit):
                weights[bit] += weight
            else:
                weights[bit] -= weight
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def _simhash_hex(value: int) -> str:
    return f"{value & ((1 << SIMHASH_BITS) - 1):016x}"


def _int_simhash(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    return int(text, 16)


def hamming_distance(left: int | str, right: int | str) -> int:
    left_int = _int_simhash(left)
    right_int = _int_simhash(right)
    if left_int is None or right_int is None:
        return SIMHASH_BITS
    return (left_int ^ right_int).bit_count()


def is_near_duplicate(
    left: str,
    right: str,
    *,
    created_at_a: str | None = None,
    created_at_b: str | None = None,
    threshold: int = SIMHASH_THRESHOLD,
) -> bool:
    if normalized_exact_hash(left) == normalized_exact_hash(right):
        return True
    if _numeric_tokens_conflict(left, right):
        return False
    return (
        hamming_distance(simhash64(left, created_at=created_at_a), simhash64(right, created_at=created_at_b))
        <= threshold
    )


def compute_dedupe_fields(content: str, created_at: str | None = None) -> DedupeFields:
    fingerprint = simhash64(content, created_at=created_at)
    simhash = _simhash_hex(fingerprint)
    return DedupeFields(
        content_hash=normalized_exact_hash(content),
        simhash=simhash,
        bands=(simhash[0:4], simhash[4:8], simhash[8:12], simhash[12:16]),
    )


def ensure_dedupe_schema(conn: Any) -> None:
    cursor = conn.cursor()
    cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunks)")}
    for col, typ in [
        ("half_life_days", "REAL DEFAULT 30.0"),
        ("archived", "INTEGER DEFAULT 0"),
        ("superseded_by", "TEXT"),
        ("archived_at", "TEXT"),
        ("seen_count", "INTEGER DEFAULT 1"),
        ("last_seen_at", "TEXT"),
        ("content_hash", "TEXT"),
        ("simhash", "TEXT"),
        ("simhash_band_0", "TEXT"),
        ("simhash_band_1", "TEXT"),
        ("simhash_band_2", "TEXT"),
        ("simhash_band_3", "TEXT"),
    ]:
        if col not in cols:
            cursor.execute(f"ALTER TABLE chunks ADD COLUMN {col} {typ}")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dedupe_audit (
            chunk_id_dropped TEXT NOT NULL,
            chunk_id_kept TEXT NOT NULL,
            mechanism TEXT NOT NULL,
            hamming_distance INTEGER,
            ts TEXT NOT NULL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dedupe_audit_ts ON dedupe_audit(ts)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dedupe_audit_kept ON dedupe_audit(chunk_id_kept)")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunk_id_alias (
            old_chunk_id TEXT PRIMARY KEY,
            canonical_chunk_id TEXT NOT NULL,
            deprecated_at TEXT NOT NULL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id_alias_canonical ON chunk_id_alias(canonical_chunk_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash)")
    for index in range(4):
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_chunks_simhash_band_{index} ON chunks(simhash_band_{index})")


def _active_clause() -> str:
    return "COALESCE(archived, 0) = 0 AND archived_at IS NULL AND superseded_by IS NULL"


def find_duplicate(
    conn: Any,
    *,
    chunk_id: str,
    content: str,
    created_at: str | None,
) -> tuple[DuplicateHit | None, DedupeFields]:
    fields = compute_dedupe_fields(content, created_at)
    cursor = conn.cursor()
    exact = cursor.execute(
        f"""
        SELECT id FROM chunks
        WHERE content_hash = ?
          AND id != ?
          AND {_active_clause()}
        ORDER BY created_at ASC, id ASC
        LIMIT 1
        """,
        (fields.content_hash, chunk_id),
    ).fetchone()
    if exact:
        return DuplicateHit(str(exact[0]), "sha256", 0), fields

    if len(normalize_for_dedupe(content).split()) < MIN_SIMHASH_TOKENS:
        return None, fields

    candidates = cursor.execute(
        f"""
        SELECT id, simhash, content FROM chunks
        WHERE id != ?
          AND {_active_clause()}
          AND simhash IS NOT NULL
          AND (
            simhash_band_0 = ? OR simhash_band_1 = ? OR simhash_band_2 = ? OR simhash_band_3 = ?
          )
        ORDER BY created_at ASC, id ASC
        LIMIT 100
        """,
        (chunk_id, *fields.bands),
    ).fetchall()
    best: DuplicateHit | None = None
    for candidate_id, candidate_simhash, candidate_content in candidates:
        if _numeric_tokens_conflict(content, str(candidate_content)):
            continue
        distance = hamming_distance(fields.simhash, candidate_simhash)
        if distance <= SIMHASH_THRESHOLD and (best is None or distance < best.hamming_distance):
            best = DuplicateHit(str(candidate_id), "simhash", distance)
    return best, fields


def _loads_tags(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item) for item in value if str(item)}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in value.split(",")]
        if isinstance(parsed, list):
            return {str(item) for item in parsed if str(item)}
    return set()


def _max_optional_number(left: Any, right: Any) -> Any:
    values = []
    for value in (left, right):
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def _latest_timestamp(*values: Any) -> str:
    parsed = [(candidate, _parse_datetime(str(candidate))) for candidate in values if candidate]
    parsed = [(raw, dt) for raw, dt in parsed if dt is not None]
    if not parsed:
        return datetime.now(timezone.utc).isoformat()
    return max(parsed, key=lambda item: item[1])[0]


def _merged_content(existing: str, incoming: str) -> str:
    if normalize_for_dedupe(existing) == normalize_for_dedupe(incoming):
        return existing
    marker = "[Merged duplicate originals]"
    originals: list[str] = []
    if existing.startswith(marker):
        numbered = existing[len(marker) :].strip()
        parts = re.split(r"(?:^|\n\n)\d+\. ", numbered)
        originals.extend(part.strip() for part in parts if part.strip())
    else:
        originals.append(existing.strip())
    if incoming.strip() not in originals:
        originals.append(incoming.strip())
    return marker + "".join(f"\n\n{index}. {item}" for index, item in enumerate(originals, start=1))


def record_dedupe_audit(
    conn: Any,
    *,
    dropped_id: str,
    kept_id: str,
    mechanism: str,
    hamming_distance_value: int | None,
    ts: str | None = None,
) -> None:
    now = ts or datetime.now(timezone.utc).isoformat()
    conn.cursor().execute(
        """
        INSERT INTO dedupe_audit(chunk_id_dropped, chunk_id_kept, mechanism, hamming_distance, ts)
        VALUES (?, ?, ?, ?, ?)
        """,
        (dropped_id, kept_id, mechanism, hamming_distance_value, now),
    )
    cutoff = (datetime.now(timezone.utc) - timedelta(days=AUDIT_RETENTION_DAYS)).isoformat()
    conn.cursor().execute("DELETE FROM dedupe_audit WHERE ts < ?", (cutoff,))


def write_alias(conn: Any, *, old_chunk_id: str, canonical_chunk_id: str, deprecated_at: str | None = None) -> None:
    if old_chunk_id == canonical_chunk_id:
        return
    conn.cursor().execute(
        """
        INSERT INTO chunk_id_alias(old_chunk_id, canonical_chunk_id, deprecated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(old_chunk_id) DO UPDATE SET
            canonical_chunk_id = excluded.canonical_chunk_id,
            deprecated_at = excluded.deprecated_at
        """,
        (old_chunk_id, canonical_chunk_id, deprecated_at or datetime.now(timezone.utc).isoformat()),
    )


def resolve_chunk_id(conn: Any, chunk_id: str) -> str:
    row = (
        conn.cursor()
        .execute(
            "SELECT canonical_chunk_id, deprecated_at FROM chunk_id_alias WHERE old_chunk_id = ?",
            (chunk_id,),
        )
        .fetchone()
    )
    if not row:
        return chunk_id
    deprecated_at = _parse_datetime(str(row[1]))
    if deprecated_at is None:
        return str(row[0])
    if deprecated_at < datetime.now(timezone.utc) - timedelta(days=ALIAS_GRACE_DAYS):
        return chunk_id
    return str(row[0])


def merge_duplicate_chunk(
    conn: Any,
    *,
    canonical_id: str,
    duplicate_id: str,
    incoming: dict[str, Any],
    mechanism: str,
    hamming_distance_value: int | None,
    archive_existing_duplicate: bool = False,
) -> bool:
    if canonical_id == duplicate_id:
        return False
    ensure_dedupe_schema(conn)
    cursor = conn.cursor()
    row = cursor.execute(
        """
        SELECT content, tags, importance, half_life_days, COALESCE(seen_count, 1), last_seen_at, created_at
        FROM chunks WHERE id = ?
        """,
        (canonical_id,),
    ).fetchone()
    if not row:
        return False
    (
        existing_content,
        existing_tags,
        existing_importance,
        existing_half_life,
        existing_seen,
        existing_last,
        existing_created,
    ) = row
    incoming_seen = int(incoming.get("seen_count") or 1)
    merged_tags = sorted(_loads_tags(existing_tags) | _loads_tags(incoming.get("tags")))
    merged_importance = _max_optional_number(existing_importance, incoming.get("importance"))
    merged_half_life = _max_optional_number(existing_half_life, incoming.get("half_life_days"))
    merged_content = _merged_content(str(existing_content), str(incoming.get("content") or ""))
    content_changed = merged_content != str(existing_content)
    fields = compute_dedupe_fields(merged_content, existing_created)
    last_seen_at = _latest_timestamp(
        existing_last, existing_created, incoming.get("last_seen_at"), incoming.get("created_at")
    )

    updates: dict[str, Any] = {
        "content": merged_content,
        "char_count": len(merged_content),
        "tags": json.dumps(merged_tags) if merged_tags else None,
        "seen_count": int(existing_seen or 1) + incoming_seen,
        "last_seen_at": last_seen_at,
        "content_hash": fields.content_hash,
        "simhash": fields.simhash,
        "simhash_band_0": fields.bands[0],
        "simhash_band_1": fields.bands[1],
        "simhash_band_2": fields.bands[2],
        "simhash_band_3": fields.bands[3],
    }
    if merged_importance is not None:
        updates["importance"] = merged_importance
    if merged_half_life is not None:
        updates["half_life_days"] = merged_half_life
    assignments = ", ".join(f"{column} = ?" for column in updates)
    cursor.execute("SAVEPOINT dedupe_merge")
    try:
        cursor.execute(f"UPDATE chunks SET {assignments} WHERE id = ?", [*updates.values(), canonical_id])

        write_alias(conn, old_chunk_id=duplicate_id, canonical_chunk_id=canonical_id)
        record_dedupe_audit(
            conn,
            dropped_id=duplicate_id,
            kept_id=canonical_id,
            mechanism=mechanism,
            hamming_distance_value=hamming_distance_value,
        )
        if archive_existing_duplicate:
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                """
                UPDATE chunks
                SET value_type = 'ARCHIVED',
                    archived = 1,
                    archived_at = COALESCE(archived_at, ?),
                    superseded_by = COALESCE(superseded_by, ?)
                WHERE id = ?
                """,
                (now, canonical_id, duplicate_id),
            )
            for table in ("chunk_vectors", "chunk_vectors_binary"):
                if cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (table,)
                ).fetchone():
                    cursor.execute(f"DELETE FROM {table} WHERE chunk_id = ?", (duplicate_id,))
    except Exception:
        cursor.execute("ROLLBACK TO dedupe_merge")
        cursor.execute("RELEASE dedupe_merge")
        raise
    else:
        cursor.execute("RELEASE dedupe_merge")
    return content_changed


def merge_existing_chunk_seen(conn: Any, *, chunk_id: str, incoming: dict[str, Any]) -> bool:
    """Merge a repost that generated the same chunk_id as the canonical row."""
    ensure_dedupe_schema(conn)
    row = (
        conn.cursor()
        .execute(
            """
        SELECT content, tags, importance, half_life_days, COALESCE(seen_count, 1), last_seen_at, created_at
        FROM chunks
        WHERE id = ?
        """,
            (chunk_id,),
        )
        .fetchone()
    )
    if not row:
        return False
    (
        existing_content,
        existing_tags,
        existing_importance,
        existing_half_life,
        existing_seen,
        existing_last,
        existing_created,
    ) = row
    if normalize_for_dedupe(str(existing_content)) != normalize_for_dedupe(str(incoming.get("content") or "")):
        return False

    merged_tags = sorted(_loads_tags(existing_tags) | _loads_tags(incoming.get("tags")))
    merged_importance = _max_optional_number(existing_importance, incoming.get("importance"))
    merged_half_life = _max_optional_number(existing_half_life, incoming.get("half_life_days"))
    last_seen_at = _latest_timestamp(
        existing_last, existing_created, incoming.get("last_seen_at"), incoming.get("created_at")
    )
    updates: dict[str, Any] = {
        "tags": json.dumps(merged_tags) if merged_tags else None,
        "seen_count": int(existing_seen or 1) + int(incoming.get("seen_count") or 1),
        "last_seen_at": last_seen_at,
    }
    if merged_importance is not None:
        updates["importance"] = merged_importance
    if merged_half_life is not None:
        updates["half_life_days"] = merged_half_life
    assignments = ", ".join(f"{column} = ?" for column in updates)
    cursor = conn.cursor()
    cursor.execute("SAVEPOINT dedupe_seen")
    try:
        cursor.execute(f"UPDATE chunks SET {assignments} WHERE id = ?", [*updates.values(), chunk_id])
        record_dedupe_audit(
            conn,
            dropped_id=chunk_id,
            kept_id=chunk_id,
            mechanism="sha256_same_id",
            hamming_distance_value=0,
        )
    except Exception:
        cursor.execute("ROLLBACK TO dedupe_seen")
        cursor.execute("RELEASE dedupe_seen")
        raise
    else:
        cursor.execute("RELEASE dedupe_seen")
    return True


def row_to_incoming(row: Any) -> dict[str, Any]:
    return {
        "id": row[0],
        "content": row[1],
        "created_at": row[2],
        "tags": row[3],
        "importance": row[4],
        "half_life_days": row[5],
        "seen_count": row[6],
        "last_seen_at": row[7],
    }


def backfill_dedupe_database(
    db_path: str | Path,
    *,
    batch_size: int = 1000,
    allow_live: bool = False,
) -> BackfillResult:
    path = Path(db_path).expanduser().resolve()
    if path == Path(get_db_path()).expanduser().resolve() and not allow_live:
        raise ValueError(
            "Refusing to dedupe-backfill the default live DB; run against a snapshot or pass allow_live=True"
        )

    conn = apsw.Connection(str(path))
    conn.setbusytimeout(10_000)
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    try:
        ensure_dedupe_schema(conn)
        cursor = conn.cursor()
        cursor.execute("PRAGMA wal_checkpoint(FULL)")
        total = cursor.execute(f"SELECT COUNT(*) FROM chunks WHERE {_active_clause()}").fetchone()[0]
        seen_hashes: dict[str, str] = {}
        band_index: dict[str, set[str]] = {}
        fingerprints: dict[str, str] = {}
        numeric_index: dict[str, set[str]] = {}
        merged = 0
        hashed = 0
        scanned = 0
        batch_number = 0
        last_created_at = ""
        last_id = ""
        while True:
            rows = cursor.execute(
                f"""
                SELECT id, content, created_at, tags, importance, half_life_days,
                       COALESCE(seen_count, 1), last_seen_at
                FROM chunks
                WHERE {_active_clause()}
                  AND (COALESCE(created_at, '') > ?
                       OR (COALESCE(created_at, '') = ? AND id > ?))
                ORDER BY COALESCE(created_at, '') ASC, id ASC
                LIMIT ?
                """,
                (last_created_at, last_created_at, last_id, batch_size),
            ).fetchall()
            if not rows:
                break
            batch_number += 1
            last_created_at = str(rows[-1][2] or "")
            last_id = str(rows[-1][0])
            for row in rows:
                scanned += 1
                incoming = row_to_incoming(row)
                chunk_id = str(incoming["id"])
                still_active = cursor.execute(
                    f"SELECT 1 FROM chunks WHERE id = ? AND {_active_clause()}",
                    (chunk_id,),
                ).fetchone()
                if not still_active:
                    continue
                fields = compute_dedupe_fields(str(incoming["content"]), incoming.get("created_at"))
                cursor.execute(
                    """
                    UPDATE chunks
                    SET content_hash = ?, simhash = ?,
                        simhash_band_0 = ?, simhash_band_1 = ?, simhash_band_2 = ?, simhash_band_3 = ?
                    WHERE id = ?
                    """,
                    (fields.content_hash, fields.simhash, *fields.bands, chunk_id),
                )
                hashed += 1
                hit: DuplicateHit | None = None
                if fields.content_hash in seen_hashes:
                    hit = DuplicateHit(seen_hashes[fields.content_hash], "sha256", 0)
                elif len(normalize_for_dedupe(str(incoming["content"])).split()) >= MIN_SIMHASH_TOKENS:
                    candidate_ids: set[str] = set()
                    for band in fields.bands:
                        candidate_ids.update(band_index.get(band, set()))
                    best_id = None
                    best_distance = SIMHASH_BITS
                    for candidate_id in candidate_ids:
                        if numeric_index.get(candidate_id, set()) != _numeric_tokens(str(incoming["content"])):
                            left_numbers = numeric_index.get(candidate_id, set())
                            right_numbers = _numeric_tokens(str(incoming["content"]))
                            if left_numbers or right_numbers:
                                continue
                        distance = hamming_distance(fields.simhash, fingerprints.get(candidate_id))
                        if distance <= SIMHASH_THRESHOLD and distance < best_distance:
                            best_id = candidate_id
                            best_distance = distance
                    if best_id is not None:
                        hit = DuplicateHit(best_id, "simhash", best_distance)
                if hit is not None:
                    merge_duplicate_chunk(
                        conn,
                        canonical_id=hit.canonical_chunk_id,
                        duplicate_id=chunk_id,
                        incoming=incoming,
                        mechanism=hit.mechanism,
                        hamming_distance_value=hit.hamming_distance,
                        archive_existing_duplicate=True,
                    )
                    merged += 1
                else:
                    seen_hashes[fields.content_hash] = chunk_id
                    fingerprints[chunk_id] = fields.simhash
                    numeric_index[chunk_id] = _numeric_tokens(str(incoming["content"]))
                    for band in fields.bands:
                        band_index.setdefault(band, set()).add(chunk_id)
            logger.info("dedupe backfill progress: scanned=%d/%d merged=%d", scanned, total, merged)
            if batch_number % 3 == 0:
                cursor.execute("PRAGMA wal_checkpoint(FULL)")
        cursor.execute("PRAGMA wal_checkpoint(FULL)")
        return BackfillResult(scanned=scanned, merged=merged, hashed=hashed)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill BrainLayer dedupe fields and merge duplicate chunks.")
    parser.add_argument("--db", required=True, help="SQLite snapshot path to process")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--allow-live", action="store_true", help="Allow processing the canonical live DB")
    args = parser.parse_args(argv)
    result = backfill_dedupe_database(args.db, batch_size=args.batch_size, allow_live=args.allow_live)
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
