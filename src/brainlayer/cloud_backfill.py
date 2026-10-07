#!/usr/bin/env python3
"""Local compatibility readers and replay for historical batch enrichment data.

Batch production and all cloud polling/download workflows are retired. Existing
checkpoint tables, saved result files, provenance and local import semantics stay
unchanged. The former executable entry point fails loudly without opening a DB.
"""

import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import apsw

from .enrichment_controller import _apply_enrichment
from .paths import get_db_path
from .pipeline.enrichment_results import (
    HIGH_VALUE_TYPES,
    parse_enrichment,
)
from .vector_store import VectorStore

_sleep = time.sleep

# ── Config ──────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = get_db_path()
EXPORT_DIR = DEFAULT_DB_PATH.parent / "backfill_data"
CHECKPOINT_TABLE = "enrichment_checkpoints"
CHECKPOINT_DB_PATH = DEFAULT_DB_PATH.with_name("enrichment_checkpoints.db")
CHECKPOINT_STABLE_STATUSES = ("submitted", "completed", "imported", "expired")
CHECKPOINT_WRITE_MAX_RETRIES = 6
CHECKPOINT_WRITE_BASE_DELAY = 0.25
# Historical provenance for saved outputs, not a model activation setting.
DEFAULT_BATCH_MODEL = "models/gemini-2.5-flash-lite"
REENRICHMENT_VERSION = "2.0"

CHECKPOINT_COLUMNS = (
    "batch_id",
    "backend",
    "model",
    "status",
    "chunk_count",
    "jsonl_path",
    "submitted_at",
    "completed_at",
    "error",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "import_mode",
)

IMPORT_MODE_AUTO = "auto"
IMPORT_MODE_DRAIN = "drain"
IMPORT_MODE_PREVIEW = "preview"


# ── DB helpers ──────────────────────────────────────────────────────────


def get_checkpoint_db_path(db_path: Path | str | None = None) -> Path:
    """Resolve the checkpoint sidecar path for a selected main DB."""
    if db_path is None:
        return CHECKPOINT_DB_PATH
    return Path(db_path).with_name("enrichment_checkpoints.db")


def _store_db_path(store: Any | None) -> Path | None:
    """Best-effort DB path lookup from a store-like object."""
    if store is None:
        return None
    db_path = getattr(store, "db_path", None)
    if db_path is None:
        return None
    return Path(db_path)


def _open_checkpoint_conn(db_path: Path | str | None = None) -> apsw.Connection:
    """Open the sidecar checkpoint DB without VectorStore startup hooks."""
    checkpoint_db_path = get_checkpoint_db_path(db_path)
    checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = apsw.Connection(str(checkpoint_db_path))
    conn.setbusytimeout(5000)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_checkpoint_table_in_conn(conn: apsw.Connection) -> None:
    """Create checkpoint table in the sidecar DB if it does not exist."""
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {CHECKPOINT_TABLE} (
            batch_id TEXT PRIMARY KEY,
            backend TEXT NOT NULL,
            model TEXT,
            status TEXT NOT NULL,
            chunk_count INTEGER,
            jsonl_path TEXT,
            submitted_at TEXT,
            completed_at TEXT,
            error TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0,
            import_mode TEXT DEFAULT 'auto'
        )
    """)
    existing_columns = {row[1] for row in cursor.execute(f"PRAGMA table_info({CHECKPOINT_TABLE})")}
    if "import_mode" not in existing_columns:
        cursor.execute(f"ALTER TABLE {CHECKPOINT_TABLE} ADD COLUMN import_mode TEXT DEFAULT 'auto'")


def _main_checkpoint_rows(store: Optional[VectorStore]) -> List[tuple]:
    """Read legacy checkpoint rows from the main DB for one-way migration."""
    if store is None:
        return []

    cursor = store.conn.cursor()
    table_exists = list(
        cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (CHECKPOINT_TABLE,),
        )
    )
    if not table_exists:
        return []

    available_columns = {row[1] for row in cursor.execute(f"PRAGMA table_info({CHECKPOINT_TABLE})")}
    selected_columns = [column for column in CHECKPOINT_COLUMNS if column in available_columns]
    if not selected_columns:
        return []

    rows = []
    for raw_row in cursor.execute(f"SELECT {', '.join(selected_columns)} FROM {CHECKPOINT_TABLE}"):
        values = dict(zip(selected_columns, raw_row))
        rows.append(
            tuple(
                values.get(column, IMPORT_MODE_AUTO if column == "import_mode" else None)
                for column in CHECKPOINT_COLUMNS
            )
        )
    return rows


def _migrate_checkpoints_from_main_db(store: Optional[VectorStore], conn: apsw.Connection) -> int:
    """Copy legacy checkpoint rows into the sidecar DB so state is preserved."""
    rows = _main_checkpoint_rows(store)
    if not rows:
        return 0

    placeholders = ", ".join("?" for _ in CHECKPOINT_COLUMNS)
    conn.cursor().executemany(
        f"""
        INSERT OR REPLACE INTO {CHECKPOINT_TABLE} ({", ".join(CHECKPOINT_COLUMNS)})
        VALUES ({placeholders})
        """,
        rows,
    )
    return len(rows)


def _get_recorded_jsonl_paths(statuses: tuple[str, ...], db_path: Path | str | None = None) -> set[str]:
    """Return JSONL paths already represented by stable checkpoint states."""
    checkpoint_db_path = get_checkpoint_db_path(db_path)
    if not checkpoint_db_path.exists():
        return set()

    conn = _open_checkpoint_conn(db_path)
    try:
        _ensure_checkpoint_table_in_conn(conn)
        placeholders = ", ".join("?" for _ in statuses)
        rows = list(
            conn.cursor().execute(
                f"""
                SELECT jsonl_path
                FROM {CHECKPOINT_TABLE}
                WHERE status IN ({placeholders})
                  AND jsonl_path IS NOT NULL
                """,
                statuses,
            )
        )
        return {row[0] for row in rows if row and row[0]}
    finally:
        conn.close()


def get_unsubmitted_export_files(export_dir: Optional[Path] = None, db_path: Path | str | None = None) -> List[Path]:
    """Reuse existing JSONL exports and skip files already checkpointed."""
    export_dir = export_dir or EXPORT_DIR
    existing_files = sorted(export_dir.glob("batch_*.jsonl"))
    if not existing_files:
        return []

    recorded_paths = _get_recorded_jsonl_paths(CHECKPOINT_STABLE_STATUSES, db_path=db_path)
    return [path for path in existing_files if str(path) not in recorded_paths]


def ensure_checkpoint_table(store: VectorStore) -> None:
    """Create the sidecar checkpoint DB and migrate legacy rows if present."""
    conn = _open_checkpoint_conn(_store_db_path(store))
    try:
        _ensure_checkpoint_table_in_conn(conn)
        _migrate_checkpoints_from_main_db(store, conn)
    finally:
        conn.close()


def save_checkpoint(store: VectorStore, batch_id: str, **kwargs) -> None:
    """Insert or update a checkpoint row in the sidecar checkpoint DB."""
    last_error: Exception | None = None
    checkpoint_db_path = _store_db_path(store)

    for attempt in range(CHECKPOINT_WRITE_MAX_RETRIES):
        conn = None
        try:
            conn = _open_checkpoint_conn(checkpoint_db_path)
            _ensure_checkpoint_table_in_conn(conn)
            cursor = conn.cursor()
            existing = list(
                cursor.execute(
                    f"SELECT batch_id FROM {CHECKPOINT_TABLE} WHERE batch_id = ?",
                    [batch_id],
                )
            )
            if existing:
                sets = ", ".join(f"{k} = ?" for k in kwargs)
                params = list(kwargs.values()) + [batch_id]
                cursor.execute(f"UPDATE {CHECKPOINT_TABLE} SET {sets} WHERE batch_id = ?", params)
            else:
                cols = ["batch_id"] + list(kwargs.keys())
                vals = [batch_id] + list(kwargs.values())
                placeholders = ", ".join("?" for _ in cols)
                cursor.execute(
                    f"INSERT INTO {CHECKPOINT_TABLE} ({', '.join(cols)}) VALUES ({placeholders})",
                    vals,
                )
            return
        except apsw.BusyError as exc:
            last_error = exc
            if attempt == CHECKPOINT_WRITE_MAX_RETRIES - 1:
                break
            _sleep(CHECKPOINT_WRITE_BASE_DELAY * (2**attempt))
        finally:
            if conn is not None:
                conn.close()

    if last_error is not None:
        raise last_error


def get_pending_jobs(store: VectorStore) -> List[Dict[str, Any]]:
    """Get jobs that need polling (submitted but not completed/failed)."""
    conn = _open_checkpoint_conn(_store_db_path(store))
    try:
        _ensure_checkpoint_table_in_conn(conn)
        rows = list(
            conn.cursor().execute(
                f"""
                SELECT batch_id, backend, model, status, jsonl_path, import_mode
                FROM {CHECKPOINT_TABLE}
                WHERE status = 'submitted'
                """
            )
        )
        return [
            {
                "batch_id": r[0],
                "backend": r[1],
                "model": r[2],
                "status": r[3],
                "jsonl_path": r[4],
                "import_mode": r[5] or IMPORT_MODE_AUTO,
            }
            for r in rows
        ]
    finally:
        conn.close()


class ReadOnlyBackfillStore:
    """Minimal read-only store for export-only runs when VectorStore open is blocked."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5)

    def _read_cursor(self):
        return self.conn.cursor()

    def get_enrichment_stats(self) -> Dict[str, Any]:
        cursor = self._read_cursor()
        total = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        enriched = cursor.execute("SELECT COUNT(*) FROM chunks WHERE enrich_status = 'success'").fetchone()[0]
        skipped = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE enrich_status IS NOT NULL AND enrich_status != 'success'"
        ).fetchone()[0]
        remaining = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE enriched_at IS NULL AND enrich_status IS NULL"
        ).fetchone()[0]
        enrichable = total - skipped
        by_intent = cursor.execute(
            """
            SELECT intent, COUNT(*) FROM chunks
            WHERE intent IS NOT NULL
            GROUP BY intent ORDER BY COUNT(*) DESC
            """
        ).fetchall()
        return {
            "total_chunks": total,
            "enrichable": enrichable,
            "enriched": enriched,
            "skipped": skipped,
            "remaining": remaining,
            "percent": round(enriched / enrichable * 100, 1) if enrichable > 0 else 0,
            "naive_percent": round((enriched + skipped) / total * 100, 1) if total > 0 else 0,
            "by_intent": {row[0]: row[1] for row in by_intent},
        }

    def close(self) -> None:
        self.conn.close()


def open_backfill_store(db_path: Path, *, allow_read_only_fallback: bool = False):
    """Open the main DB for backfill work, with read-only fallback for export-only runs."""
    try:
        return VectorStore(db_path)
    except apsw.BusyError:
        if not allow_read_only_fallback:
            raise
        print("Main DB open hit BusyError; falling back to read-only export mode.")
        return ReadOnlyBackfillStore(db_path)


def _chunk_columns(store: Any) -> set[str]:
    """Return the available chunk columns for schema-aware export/import queries."""
    return {row[1] for row in store.conn.cursor().execute("PRAGMA table_info(chunks)")}


def _build_reenrichment_filters(
    store: Any,
    *,
    content_types: Optional[List[str]] = None,
    min_char_count: int = 50,
    pending_only: bool = False,
) -> tuple[str, list[Any]]:
    """Build the shared WHERE clause for preview-summary re-enrichment candidates."""
    columns = _chunk_columns(store)
    where_parts = ["char_count >= ?"]
    params: list[Any] = [min_char_count]

    if content_types:
        type_placeholders = ", ".join("?" for _ in content_types)
        where_parts.append(f"content_type IN ({type_placeholders})")
        params.extend(content_types)

    if {"enriched_at", "summary"}.issubset(columns):
        # Candidate set = genuinely unenriched chunks OR legacy summarized chunks.
        where_parts.append("(enriched_at IS NULL OR summary IS NOT NULL)")
    elif "enriched_at" in columns:
        where_parts.append("enriched_at IS NULL")
    elif "summary" in columns:
        where_parts.append("summary IS NOT NULL")

    if pending_only and "summary_v2" in columns:
        where_parts.append("summary_v2 IS NULL")

    return " AND ".join(where_parts), params


def get_reenrichment_stats(
    store: Any,
    *,
    content_types: Optional[List[str]] = None,
    min_char_count: int = 50,
) -> Dict[str, Any]:
    """Return non-destructive summary_v2 preview progress for batch re-enrichment."""
    cursor = store.conn.cursor()
    columns = _chunk_columns(store)
    where, params = _build_reenrichment_filters(
        store,
        content_types=content_types or HIGH_VALUE_TYPES,
        min_char_count=min_char_count,
        pending_only=False,
    )
    total = cursor.execute(f"SELECT COUNT(*) FROM chunks WHERE {where}", params).fetchone()[0]

    if "summary_v2" in columns:
        previewed = cursor.execute(
            f"SELECT COUNT(*) FROM chunks WHERE {where} AND summary_v2 IS NOT NULL",
            params,
        ).fetchone()[0]
    else:
        previewed = 0

    remaining = total - previewed
    percent = round(previewed / total * 100, 1) if total > 0 else 0.0
    return {
        "eligible": total,
        "previewed": previewed,
        "remaining": remaining,
        "percent": percent,
    }


def get_backlog_drain_stats(store: Any, *, min_char_count: int = 50) -> Dict[str, Any]:
    """Return the live eligible backlog count used by realtime enrichment."""
    cursor = store.conn.cursor()
    remaining = cursor.execute(
        """
        SELECT COUNT(*)
        FROM chunks
        WHERE enriched_at IS NULL
          AND enrich_status IS NULL
          AND char_count >= ?
        """,
        (min_char_count,),
    ).fetchone()[0]
    return {"remaining": remaining}


# ── Import results ──────────────────────────────────────────────────────


def _clear_imported_preview(store: VectorStore, chunk_id: str) -> None:
    """Remove preview fields written by a failed remote import so the chunk can retry."""
    store.conn.cursor().execute(
        """
        UPDATE chunks
        SET summary_v2 = NULL,
            enrichment_version = NULL
        WHERE id = ?
        """,
        (chunk_id,),
    )


def _extract_batch_response_text(result: Dict[str, Any]) -> Optional[str]:
    """Extract generated text from a Gemini Batch JSONL result row."""
    try:
        response = result.get("response", result.get("output", {}))
        if isinstance(response, dict):
            candidates = response.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        elif isinstance(response, str):
            return response
    except (KeyError, IndexError, TypeError):
        return None
    return None


def _get_canonical_import_candidate(store: VectorStore, chunk_id: str) -> Optional[dict[str, Any]]:
    """Return the realtime-eligible chunk dict for a batch import, if still eligible."""
    candidates = store.get_enrichment_candidates(limit=1, chunk_ids=[chunk_id])
    if candidates and candidates[0]["id"] == chunk_id:
        return candidates[0]
    return None


def _commit_batch_enrichment(
    store: VectorStore,
    chunk: dict[str, Any],
    enrichment: dict[str, Any],
    *,
    model: str,
) -> None:
    """Commit batch output through the same canonical write path as realtime."""
    _apply_enrichment(
        store,
        chunk,
        enrichment,
        enrichment_model=model,
        enrichment_backend="gemini-batch",
    )


def import_results(
    store: VectorStore,
    results: List[Dict[str, Any]],
    batch_id: str,
    *,
    import_mode: str = IMPORT_MODE_AUTO,
    model: str = DEFAULT_BATCH_MODEL,
) -> Dict[str, int]:
    """Import batch results, committing eligible backlog rows canonically."""
    success = 0
    failed = 0
    skipped = 0

    cursor = store.conn.cursor()

    for result in results:
        chunk_id = result.get("key")
        if not chunk_id:
            failed += 1
            continue

        response_text = _extract_batch_response_text(result)
        if not response_text:
            failed += 1
            continue

        enrichment = parse_enrichment(response_text)
        if enrichment:
            canonical_chunk = None
            if import_mode in {IMPORT_MODE_AUTO, IMPORT_MODE_DRAIN}:
                canonical_chunk = _get_canonical_import_candidate(store, chunk_id)

            if canonical_chunk is not None:
                try:
                    _commit_batch_enrichment(store, canonical_chunk, enrichment, model=model)
                except Exception as exc:
                    print(f"  WARNING: canonical import failed for {chunk_id}: {exc}")
                    failed += 1
                    continue
                success += 1
                continue

            if import_mode == IMPORT_MODE_DRAIN:
                skipped += 1
                continue

            # Skip if preview summary already exists.
            existing = list(
                cursor.execute(
                    "SELECT summary_v2 FROM chunks WHERE id = ?",
                    [chunk_id],
                )
            )
            if existing and existing[0][0] is not None:
                skipped += 1
                continue

            try:
                store.update_reenrichment_preview(
                    chunk_id=chunk_id,
                    summary_v2=enrichment.get("summary"),
                    enrichment_version=REENRICHMENT_VERSION,
                )
            except Exception as exc:
                print(f"  WARNING: preview import failed for {chunk_id}: {exc}")
                _clear_imported_preview(store, chunk_id)
                failed += 1
                continue
            success += 1
        else:
            failed += 1

        if (success + failed + skipped) % 1000 == 0:
            print(f"  Progress: {success} ok, {failed} fail, {skipped} skip")

    print(f"  Import done: {success} ok, {failed} fail, {skipped} skip")

    # Update checkpoint
    save_checkpoint(
        store,
        batch_id=batch_id,
        status="imported",
        completed_at=datetime.now(timezone.utc).isoformat(),
    )

    return {"success": success, "failed": failed, "skipped": skipped}


def show_status(db_path: Path) -> None:
    """Show status of all checkpoint jobs."""
    store = VectorStore(db_path)
    ensure_checkpoint_table(store)

    try:
        conn = _open_checkpoint_conn(db_path)
        try:
            _ensure_checkpoint_table_in_conn(conn)
            rows = list(
                conn.cursor().execute(
                    f"""
                    SELECT batch_id, backend, model, status, chunk_count, submitted_at, completed_at, error
                    FROM {CHECKPOINT_TABLE}
                    ORDER BY submitted_at DESC
                    """
                )
            )
        finally:
            conn.close()

        if not rows:
            print("No batch jobs recorded.")
            return

        print(f"{'Status':<12} {'Chunks':<8} {'Backend':<10} {'Submitted':<22} {'Completed':<22} {'Error'}")
        print("-" * 100)
        for batch_id, backend, model, status, chunk_count, submitted, completed, error in rows:
            err_str = (error or "")[:30]
            print(
                f"{status:<12} {chunk_count or 0:<8} {backend:<10} {(submitted or '')[:19]:<22} {(completed or '')[:19]:<22} {err_str}"
            )

        stats = get_reenrichment_stats(store)
        print(
            f"\nPreview progress: {stats['previewed']}/{stats['eligible']} "
            f"({stats['percent']}%), remaining={stats['remaining']}"
        )

    finally:
        store.close()


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> int:
    """Fail stale batch commands before opening a database or creating a client."""
    print("ERROR: batch enrichment is retired; checkpoints and local result replay are preserved", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
