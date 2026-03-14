#!/usr/bin/env python3
"""Cloud backfill — enrich unenriched BrainLayer chunks via Gemini Batch API.

Exports unenriched chunks to JSONL, submits to Gemini Batch API, polls for
completion, and imports results back into the DB.

Usage:
    # Dry run — export JSONL only, don't submit
    python3 scripts/cloud_backfill.py --dry-run

    # Run 100-chunk validation sample
    python3 scripts/cloud_backfill.py --sample 100

    # Full backfill
    python3 scripts/cloud_backfill.py

    # Resume from checkpoint
    python3 scripts/cloud_backfill.py --resume

    # Show status of running/completed jobs
    python3 scripts/cloud_backfill.py --status
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import apsw

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.paths import get_db_path
from brainlayer.pipeline.batch_extraction import DEFAULT_SEED_ENTITIES
from brainlayer.pipeline.enrichment import (
    HIGH_VALUE_TYPES,
    build_external_prompt,
    build_prompt,
    parse_enrichment,
)
from brainlayer.pipeline.kg_extraction import extract_kg_from_chunk
from brainlayer.pipeline.sanitize import SanitizeConfig, Sanitizer
from brainlayer.vector_store import VectorStore

# ── Config ──────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = get_db_path()
EXPORT_DIR = Path(__file__).resolve().parent / "backfill_data"
CHECKPOINT_TABLE = "enrichment_checkpoints"
CHECKPOINT_DB_PATH = DEFAULT_DB_PATH.with_name("enrichment_checkpoints.db")
CHECKPOINT_STABLE_STATUSES = ("submitted", "completed", "imported", "expired")
CHECKPOINT_WRITE_MAX_RETRIES = 6
CHECKPOINT_WRITE_BASE_DELAY = 0.25

# Gemini Batch API limits (Tier 1)
# AIDEV-NOTE: Tier 1 has low enqueued-token quota for 2.5-flash batch.
# 12K chunks/job (~9M tokens) hits 429 RESOURCE_EXHAUSTED.
# 500 chunks/job (~350K tokens) stays safely under quota.
MAX_TOKENS_PER_JOB = 350_000  # Conservative: stay under Tier 1 enqueued limit
AVG_PROMPT_TOKENS = 700  # Estimated avg tokens per chunk prompt
CHUNKS_PER_JOB = MAX_TOKENS_PER_JOB // AVG_PROMPT_TOKENS  # ~500
BATCH_INPUT_COST_PER_MILLION = 0.075
BATCH_OUTPUT_COST_PER_MILLION = 0.30

# Supabase usage logging
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

# 10-field JSON schema for structured output
ENRICHMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "One sentence describing what this chunk is about"},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "3-7 lowercase hyphenated topic tags"},
        "importance": {"type": "integer", "description": "1-10 importance score"},
        "intent": {
            "type": "string",
            "enum": ["debugging", "designing", "configuring", "discussing", "deciding", "implementing", "reviewing"],
        },
        "primary_symbols": {"type": "array", "items": {"type": "string"}, "description": "Classes, functions, files mentioned"},
        "resolved_query": {"type": "string", "description": "Hypothetical question this chunk answers"},
        "epistemic_level": {"type": "string", "enum": ["hypothesis", "substantiated", "validated"]},
        "version_scope": {"type": "string", "description": "Version or system state discussed, or null"},
        "debt_impact": {"type": "string", "enum": ["introduction", "resolution", "none"]},
        "external_deps": {"type": "array", "items": {"type": "string"}, "description": "Libraries or external APIs used"},
    },
    "required": ["summary", "tags", "importance", "intent", "primary_symbols", "resolved_query", "epistemic_level", "debt_impact", "external_deps"],
}

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
)


def build_batch_request_line(chunk_id: str, prompt: str) -> Dict[str, Any]:
    """Build a Gemini Batch API request line with thinking disabled.

    thinkingBudget=0 is mandatory for this backfill job class. On February 18,
    2026, non-zero thinking tokens caused a high-cost incident on
    gemini-2.5-flash. Centralizing the payload keeps export behavior testable.
    """
    return {
        "key": chunk_id,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingBudget": 0},
            },
        },
    }


def estimate_batch_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Estimate Gemini 2.5 Flash Batch API cost using the 50% batch discount."""
    return (
        input_tokens * BATCH_INPUT_COST_PER_MILLION + output_tokens * BATCH_OUTPUT_COST_PER_MILLION
    ) / 1_000_000


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
            cost_usd REAL DEFAULT 0
        )
    """)


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

    return list(cursor.execute(f"SELECT {', '.join(CHECKPOINT_COLUMNS)} FROM {CHECKPOINT_TABLE}"))


def _migrate_checkpoints_from_main_db(store: Optional[VectorStore], conn: apsw.Connection) -> int:
    """Copy legacy checkpoint rows into the sidecar DB so state is preserved."""
    rows = _main_checkpoint_rows(store)
    if not rows:
        return 0

    placeholders = ", ".join("?" for _ in CHECKPOINT_COLUMNS)
    conn.cursor().executemany(
        f"""
        INSERT OR REPLACE INTO {CHECKPOINT_TABLE} ({', '.join(CHECKPOINT_COLUMNS)})
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
            time.sleep(CHECKPOINT_WRITE_BASE_DELAY * (2**attempt))
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
                SELECT batch_id, backend, model, status, jsonl_path
                FROM {CHECKPOINT_TABLE}
                WHERE status = 'submitted'
                """
            )
        )
        return [{"batch_id": r[0], "backend": r[1], "model": r[2], "status": r[3], "jsonl_path": r[4]} for r in rows]
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
        enriched = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE enriched_at IS NOT NULL AND enriched_at NOT LIKE 'skipped:%'"
        ).fetchone()[0]
        skipped = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE enriched_at LIKE 'skipped:%'"
        ).fetchone()[0]
        remaining = cursor.execute("SELECT COUNT(*) FROM chunks WHERE enriched_at IS NULL").fetchone()[0]
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


# ── Export ──────────────────────────────────────────────────────────────

def _init_sanitizer(store: VectorStore) -> Sanitizer:
    """Initialize the sanitizer with name dictionary from the DB.

    AIDEV-NOTE: This is called once at the start of any external export.
    The sanitizer is THE gate for external API calls — not optional.
    """
    sanitizer = Sanitizer.from_env()

    # Build name dictionary from WhatsApp senders in DB
    known_names = sanitizer.build_name_dictionary(store)
    if known_names:
        print(f"  Built name dictionary: {len(known_names)} names from WhatsApp contacts")
        # Rebuild sanitizer with the full name dictionary
        new_config = SanitizeConfig(
            owner_names=sanitizer.config.owner_names,
            owner_emails=sanitizer.config.owner_emails,
            owner_paths=sanitizer.config.owner_paths,
            known_names=frozenset(known_names) | sanitizer.config.known_names,
            strip_emails=sanitizer.config.strip_emails,
            strip_ips=sanitizer.config.strip_ips,
            strip_jwts=sanitizer.config.strip_jwts,
            strip_op_refs=sanitizer.config.strip_op_refs,
            strip_phone_numbers=sanitizer.config.strip_phone_numbers,
            use_spacy_ner=sanitizer.config.use_spacy_ner,
        )
        sanitizer = Sanitizer(new_config)

    # Load existing mapping for pseudonym consistency across runs
    mapping_path = EXPORT_DIR / "pii_mapping.json"
    sanitizer.load_mapping(mapping_path)

    return sanitizer


def export_unenriched_chunks(
    store: VectorStore,
    max_chunks: int = 0,
    content_types: Optional[List[str]] = None,
    min_char_count: int = 50,
    no_sanitize: bool = False,
) -> List[Path]:
    """Export unenriched chunks to JSONL files (one per batch job).

    Content is sanitized before export — PII is stripped via build_external_prompt().
    Use no_sanitize=True only for local testing with trusted backends.
    """
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    types = content_types or HIGH_VALUE_TYPES

    if no_sanitize:
        print("WARNING: PII sanitization DISABLED — use only for local testing!")
        sanitizer = None
    else:
        print("Initializing PII sanitizer...")
        sanitizer = _init_sanitizer(store)

    cursor = store.conn.cursor()

    # Count total unenriched (parameterized queries for safety)
    where_parts = ["enriched_at IS NULL", "char_count >= ?"]
    params: list = [min_char_count]
    if types:
        type_placeholders = ", ".join("?" for _ in types)
        where_parts.append(f"content_type IN ({type_placeholders})")
        params.extend(types)
    where = " AND ".join(where_parts)

    total = list(cursor.execute(f"SELECT COUNT(*) FROM chunks WHERE {where}", params))[0][0]
    print(f"Total unenriched chunks (eligible): {total}")

    if max_chunks > 0:
        total = min(total, max_chunks)
        print(f"Limiting to: {total}")

    # Fetch all eligible chunk IDs, content, and metadata for sanitization
    query = f"""
        SELECT id, content, project, content_type, source, sender
        FROM chunks
        WHERE {where}
        ORDER BY rowid
    """
    fetch_params = list(params)  # Copy params for the fetch query
    if max_chunks > 0:
        query += " LIMIT ?"
        fetch_params.append(max_chunks)

    rows = list(cursor.execute(query, fetch_params))
    print(f"Fetched {len(rows)} chunks for export")

    # Split into batch files
    jsonl_files = []
    batch_num = 0
    total_pii_found = 0

    for i in range(0, len(rows), CHUNKS_PER_JOB):
        batch = rows[i:i + CHUNKS_PER_JOB]
        batch_num += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = EXPORT_DIR / f"batch_{ts}_{batch_num:03d}.jsonl"

        with open(filename, "w") as f:
            for chunk_id, content, project, content_type, source, sender in batch:
                chunk_dict = {
                    "content": content,
                    "project": project,
                    "content_type": content_type,
                    "source": source,
                    "sender": sender,
                }

                if sanitizer is not None:
                    prompt, sanitize_result = build_external_prompt(
                        chunk_dict, sanitizer
                    )
                    if sanitize_result.pii_detected:
                        total_pii_found += 1
                else:
                    # No sanitization — use local prompt builder
                    prompt = build_prompt(chunk_dict)

                # Gemini Batch API format uses camelCase field names.
                request_line = build_batch_request_line(chunk_id, prompt)
                f.write(json.dumps(request_line) + "\n")

        jsonl_files.append(filename)
        print(f"  Wrote {filename.name} ({len(batch)} chunks)")

    # Save PII mapping for reversibility (local only, never uploaded)
    if sanitizer is not None:
        mapping_path = EXPORT_DIR / "pii_mapping.json"
        sanitizer.save_mapping(mapping_path)
        print(f"Mapping saved to: {mapping_path}")
    print(f"\nPII sanitization: {total_pii_found}/{len(rows)} chunks had PII stripped")
    print(f"Exported {len(rows)} chunks to {len(jsonl_files)} JSONL files")
    return jsonl_files


# ── Gemini Batch API (google.genai SDK) ────────────────────────────────

def _get_genai_client():
    """Get a google.genai Client configured with API key."""
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai not installed. Run: pip install google-genai")
        sys.exit(1)

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY not set")
        sys.exit(1)

    return genai.Client(api_key=api_key)


def submit_gemini_batch(
    jsonl_path: Path,
    model: str = "models/gemini-2.5-flash",
    store: Optional[VectorStore] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """Upload JSONL and submit a Gemini batch job. Returns batch job name or None on failure."""
    client = _get_genai_client()

    # Count chunks in file
    with open(jsonl_path) as f:
        chunk_count = sum(1 for _ in f)

    print(f"\nSubmitting batch: {jsonl_path.name} ({chunk_count} chunks)")

    # Upload file
    print("  Uploading JSONL to File API...")
    uploaded_file = client.files.upload(
        file=str(jsonl_path),
        config={"display_name": f"brainlayer-backfill-{jsonl_path.stem}", "mime_type": "application/json"},
    )
    print(f"  Uploaded: {uploaded_file.name}")

    # Create batch job with retry on 429
    for attempt in range(max_retries):
        try:
            print(f"  Creating batch job (model: {model})..." + (f" (retry {attempt})" if attempt else ""))
            batch_job = client.batches.create(
                model=model,
                src=uploaded_file.name,
                config={"display_name": f"brainlayer-enrichment-{jsonl_path.stem}"},
            )
            print(f"  Job created: {batch_job.name} (state: {batch_job.state})")

            # Save checkpoint
            if store:
                save_checkpoint(
                    store,
                    batch_id=batch_job.name,
                    backend="gemini",
                    model=model,
                    status="submitted",
                    chunk_count=chunk_count,
                    jsonl_path=str(jsonl_path),
                    submitted_at=datetime.now(timezone.utc).isoformat(),
                )

            return batch_job.name

        except Exception as e:
            err = str(e)
            if "429" in err and attempt < max_retries - 1:
                wait = 30 * (2 ** attempt)  # 30s, 60s, 120s
                print(f"  429 RESOURCE_EXHAUSTED — waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"  FAILED: {err[:120]}")
                if store:
                    save_checkpoint(
                        store,
                        batch_id=f"failed-{jsonl_path.stem}",
                        backend="gemini",
                        model=model,
                        status="failed",
                        chunk_count=chunk_count,
                        jsonl_path=str(jsonl_path),
                        error=err[:500],
                    )
                return None

    return None


def poll_gemini_batch(batch_name: str, timeout_hours: float = 25) -> Dict[str, Any]:
    """Poll a Gemini batch job until completion. Returns job state."""
    client = _get_genai_client()

    deadline = time.time() + (timeout_hours * 3600)
    poll_interval = 30  # Start with 30s, increase over time

    print(f"\nPolling job: {batch_name}")
    while time.time() < deadline:
        try:
            batch_job = client.batches.get(name=batch_name)
        except Exception as exc:
            err_str = str(exc)
            if "404" in err_str or "NOT_FOUND" in err_str:
                print(f"  Job not found (404) — may have expired: {batch_name}")
                return {"state": "failed", "error": "NOT_FOUND (expired or invalid)"}
            raise
        # state may be "JOB_STATE_SUCCEEDED" or "JobState.JOB_STATE_SUCCEEDED"
        state = str(batch_job.state)
        state_normalized = state.split(".")[-1]  # strip "JobState." prefix if present

        if state_normalized == "JOB_STATE_SUCCEEDED":
            print("  Job SUCCEEDED!")
            return {"state": "succeeded", "job": batch_job}
        elif state_normalized == "JOB_STATE_FAILED":
            error = getattr(batch_job, "error", "unknown error")
            print(f"  Job FAILED: {error}")
            return {"state": "failed", "error": str(error), "job": batch_job}
        elif state_normalized == "JOB_STATE_CANCELLED":
            print("  Job CANCELLED")
            return {"state": "failed", "error": "cancelled"}

        print(f"  State: {state} — waiting {poll_interval:.0f}s...")
        time.sleep(poll_interval)

        # Gradually increase poll interval (max 5 min)
        poll_interval = min(poll_interval * 1.2, 300)

    return {"state": "timeout", "error": "Exceeded timeout"}


def download_gemini_results(batch_job) -> List[Dict[str, Any]]:
    """Download and parse results from a completed Gemini batch job."""
    client = _get_genai_client()
    results = []

    # The dest field contains the output file reference
    dest = batch_job.dest
    if not dest:
        print("  WARNING: No dest on batch job")
        return results

    # dest is a BatchJobDestination with file_name
    file_name = None
    if hasattr(dest, "file_name"):
        file_name = dest.file_name
    elif isinstance(dest, str):
        file_name = dest
    elif isinstance(dest, dict):
        file_name = dest.get("file_name", dest.get("gcs_uri"))

    if not file_name:
        print(f"  WARNING: Could not extract file name from dest: {dest}")
        return results

    print(f"  Downloading results: {file_name}")

    try:
        # Download the result file (SDK uses file= not name=)
        content = client.files.download(file=file_name)
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        for line in content.strip().splitlines():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"  Error downloading results: {e}")

    print(f"  Downloaded {len(results)} results")
    return results


# ── Import results ──────────────────────────────────────────────────────

def _clear_imported_enrichment(store: VectorStore, chunk_id: str) -> None:
    """Remove enrichment written by a failed remote import so the chunk can retry."""
    store.conn.cursor().execute(
        """
        UPDATE chunks
        SET enriched_at = NULL,
            summary = NULL,
            tags = NULL,
            importance = NULL,
            intent = NULL,
            primary_symbols = NULL,
            resolved_query = NULL,
            epistemic_level = NULL,
            version_scope = NULL,
            debt_impact = NULL,
            external_deps = NULL
        WHERE id = ?
        """,
        (chunk_id,),
    )


def import_results(
    store: VectorStore,
    results: List[Dict[str, Any]],
    batch_id: str,
) -> Dict[str, int]:
    """Import batch results back to DB. Returns counts."""
    success = 0
    failed = 0
    skipped = 0

    cursor = store.conn.cursor()

    for result in results:
        chunk_id = result.get("key")
        if not chunk_id:
            failed += 1
            continue

        # Skip if already enriched (local enrichment may have caught it)
        existing = list(cursor.execute(
            "SELECT enriched_at FROM chunks WHERE id = ?", [chunk_id]
        ))
        if existing and existing[0][0] is not None:
            skipped += 1
            continue

        # Extract the generated text from Gemini response
        response_text = None
        try:
            response = result.get("response", result.get("output", {}))
            if isinstance(response, dict):
                candidates = response.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        response_text = parts[0].get("text", "")
            elif isinstance(response, str):
                response_text = response
        except (KeyError, IndexError, TypeError):
            pass

        if not response_text:
            failed += 1
            continue

        enrichment = parse_enrichment(response_text)
        if enrichment:
            store.update_enrichment(
                chunk_id=chunk_id,
                summary=enrichment.get("summary"),
                tags=enrichment.get("tags"),
                importance=enrichment.get("importance"),
                intent=enrichment.get("intent"),
                primary_symbols=enrichment.get("primary_symbols"),
                resolved_query=enrichment.get("resolved_query"),
                epistemic_level=enrichment.get("epistemic_level"),
                version_scope=enrichment.get("version_scope"),
                debt_impact=enrichment.get("debt_impact"),
                external_deps=enrichment.get("external_deps"),
            )
            try:
                extract_kg_from_chunk(
                    store=store,
                    chunk_id=chunk_id,
                    seed_entities=DEFAULT_SEED_ENTITIES,
                    use_llm=False,
                    use_gliner=False,
                )
            except Exception as exc:
                print(f"  WARNING: KG extraction failed for {chunk_id}: {exc}")
                _clear_imported_enrichment(store, chunk_id)
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


# ── Usage logging ───────────────────────────────────────────────────────

def log_batch_usage(batch_id: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
    """Log batch usage to Supabase. Best-effort."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return
    try:
        import requests
        requests.post(
            f"{SUPABASE_URL}/rest/v1/llm_usage",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json={
                "model": model,
                "source": "enrichment-batch",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "tier": "paid",
            },
            timeout=5,
        )
    except Exception:
        pass  # Never let logging failure block backfill


# ── Main workflows ──────────────────────────────────────────────────────

def run_full_backfill(
    db_path: Path,
    model: str = "models/gemini-2.5-flash",
    dry_run: bool = False,
    sample: int = 0,
    no_sanitize: bool = False,
    submit_only: bool = False,
) -> None:
    """Run the full backfill: export → submit → poll → import."""
    store = open_backfill_store(db_path, allow_read_only_fallback=(submit_only or dry_run))
    ensure_checkpoint_table(store)

    try:
        # Show current state
        stats = store.get_enrichment_stats()
        print(f"Current enrichment: {stats['enriched']}/{stats['total_chunks']} ({stats['percent']}%)")
        print(f"Remaining: {stats['remaining']}\n")

        # Step 1: Export or reuse existing batch files
        max_chunks = sample if sample > 0 else 0
        if submit_only and sample == 0:
            jsonl_files = get_unsubmitted_export_files(db_path=db_path)
            if jsonl_files:
                print(
                    f"Reusing {len(jsonl_files)} existing JSONL batch files not yet checkpointed "
                    f"from {EXPORT_DIR}"
                )
            else:
                jsonl_files = export_unenriched_chunks(store, max_chunks=max_chunks, no_sanitize=no_sanitize)
                if not jsonl_files:
                    if get_pending_jobs(store):
                        print("Existing batch jobs are already submitted. Run --resume to import them.")
                    elif list(EXPORT_DIR.glob("batch_*.jsonl")):
                        print("All existing JSONL batch files are already checkpointed. No new submissions needed.")
                    return
        else:
            jsonl_files = export_unenriched_chunks(store, max_chunks=max_chunks, no_sanitize=no_sanitize)

        if not jsonl_files:
            print("Nothing to export!")
            return

        if dry_run:
            print(f"\n[DRY RUN] Exported {len(jsonl_files)} files. Not submitting.")
            for f in jsonl_files:
                print(f"  {f}")
            return

        # Step 2: Submit all batch jobs
        batch_names = []
        for jsonl_path in jsonl_files:
            batch_name = submit_gemini_batch(jsonl_path, model=model, store=store)
            if batch_name is not None:
                batch_names.append(batch_name)
            else:
                print(f"  [WARN] Submission failed for {jsonl_path}, skipping")
            time.sleep(2)  # Brief pause between submissions

        if not batch_names:
            print("\nNo batch jobs submitted successfully.")
            return

        if submit_only:
            print(f"\nSubmitted {len(batch_names)} batch jobs. Run --resume later to import results.")
            for name in batch_names:
                print(f"  {name}")
            return

        print(f"\nSubmitted {len(batch_names)} batch jobs. Polling for results...")

        # Step 3: Poll all jobs
        total_imported = {"success": 0, "failed": 0, "skipped": 0}
        for batch_name in batch_names:
            result = poll_gemini_batch(batch_name)

            if result["state"] == "succeeded":
                # Download + import
                batch_results = download_gemini_results(result["job"])
                counts = import_results(store, batch_results, batch_name)
                for k in total_imported:
                    total_imported[k] += counts[k]

                # Log usage to Supabase (best-effort)
                job = result.get("job")
                if job and hasattr(job, "usage_metadata") and job.usage_metadata:
                    um = job.usage_metadata
                    in_tok = getattr(um, "prompt_token_count", 0) or 0
                    out_tok = getattr(um, "candidates_token_count", 0) or 0
                    cost = estimate_batch_cost_usd(in_tok, out_tok)
                    log_batch_usage(batch_name, model, in_tok, out_tok, cost)

                save_checkpoint(store, batch_id=batch_name, status="completed",
                                completed_at=datetime.now(timezone.utc).isoformat())
            else:
                save_checkpoint(store, batch_id=batch_name, status="failed",
                                error=result.get("error", "unknown"),
                                completed_at=datetime.now(timezone.utc).isoformat())
                print(f"  FAILED: {batch_name} — {result.get('error')}")

        # Final stats
        final_stats = store.get_enrichment_stats()
        print(f"\n{'='*60}")
        print("BACKFILL COMPLETE")
        print(f"  Imported: {total_imported['success']} ok, {total_imported['failed']} fail, {total_imported['skipped']} skip")
        print(f"  Enrichment: {final_stats['enriched']}/{final_stats['total_chunks']} ({final_stats['percent']}%)")
        print(f"{'='*60}")

    finally:
        store.close()


def resume_backfill(db_path: Path) -> None:
    """Resume polling/importing for any submitted but incomplete jobs."""
    store = VectorStore(db_path)
    ensure_checkpoint_table(store)

    try:
        pending = get_pending_jobs(store)
        if not pending:
            print("No pending jobs to resume.")
            return

        print(f"Found {len(pending)} pending jobs to resume")
        imported_count = 0
        failed_count = 0
        for i, job in enumerate(pending):
            print(f"\nResuming [{i+1}/{len(pending)}]: {job['batch_id']}")
            try:
                result = poll_gemini_batch(job["batch_id"])
            except Exception as exc:
                print(f"  ERROR polling job: {exc}")
                save_checkpoint(store, batch_id=job["batch_id"], status="failed",
                                error=str(exc)[:500],
                                completed_at=datetime.now(timezone.utc).isoformat())
                failed_count += 1
                continue

            if result["state"] == "succeeded":
                batch_results = download_gemini_results(result["job"])
                import_results(store, batch_results, job["batch_id"])
                imported_count += 1

                # Log usage to Supabase (best-effort)
                batch_job = result.get("job")
                if batch_job and hasattr(batch_job, "usage_metadata") and batch_job.usage_metadata:
                    um = batch_job.usage_metadata
                    in_tok = getattr(um, "prompt_token_count", 0) or 0
                    out_tok = getattr(um, "candidates_token_count", 0) or 0
                    cost = estimate_batch_cost_usd(in_tok, out_tok)
                    log_batch_usage(job["batch_id"], job.get("model", "gemini"), in_tok, out_tok, cost)

                save_checkpoint(store, batch_id=job["batch_id"], status="completed",
                                completed_at=datetime.now(timezone.utc).isoformat())
            else:
                save_checkpoint(store, batch_id=job["batch_id"], status="failed",
                                error=result.get("error", "unknown"),
                                completed_at=datetime.now(timezone.utc).isoformat())
                failed_count += 1

        final_stats = store.get_enrichment_stats()
        print(f"\nImported {imported_count} jobs, {failed_count} failed")
        print(f"Enrichment: {final_stats['enriched']}/{final_stats['total_chunks']} ({final_stats['percent']}%)")

    finally:
        store.close()


def show_status(db_path: Path) -> None:
    """Show status of all checkpoint jobs."""
    store = VectorStore(db_path)
    ensure_checkpoint_table(store)

    try:
        conn = _open_checkpoint_conn()
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
            print(f"{status:<12} {chunk_count or 0:<8} {backend:<10} {(submitted or '')[:19]:<22} {(completed or '')[:19]:<22} {err_str}")

        stats = store.get_enrichment_stats()
        print(f"\nOverall: {stats['enriched']}/{stats['total_chunks']} ({stats['percent']}%)")

    finally:
        store.close()


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud backfill for BrainLayer enrichment")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument("--model", type=str, default="models/gemini-2.5-flash",
                        help="Gemini model (default: gemini-2.5-flash; flash-lite does not support batch)")
    parser.add_argument("--dry-run", action="store_true", help="Export JSONL only, don't submit")
    parser.add_argument("--sample", type=int, default=0, help="Run N-chunk validation sample")
    parser.add_argument("--resume", action="store_true", help="Resume pending batch jobs")
    parser.add_argument("--status", action="store_true", help="Show batch job status")
    parser.add_argument("--no-sanitize", action="store_true",
                        help="Skip PII sanitization (local testing only — NEVER use for external APIs)")
    parser.add_argument("--submit-only", action="store_true",
                        help="Submit batch jobs and exit — don't poll. Use --resume later to import.")

    args = parser.parse_args()
    db = Path(args.db) if args.db else DEFAULT_DB_PATH

    if args.status:
        show_status(db)
    elif args.resume:
        resume_backfill(db)
    else:
        run_full_backfill(db, model=args.model, dry_run=args.dry_run, sample=args.sample,
                          no_sanitize=args.no_sanitize, submit_only=args.submit_only)
