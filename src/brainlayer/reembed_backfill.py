"""Batch re-embed active chunks that are missing semantic vectors."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from .embeddings import DEFAULT_MODEL
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

HEAVY_ML_PATTERNS = (
    "llama-server",
    "ollama",
    "whisper",
    "mlx",
    "brainlayer enrich",
    "enrichment",
)


class BatchEmbeddingModel(Protocol):
    def encode(self, texts: list[str], **kwargs: Any) -> Iterable[Any]:
        """Return one embedding per input text."""


@dataclass(frozen=True)
class PendingChunk:
    chunk_id: str
    content: str


@dataclass(frozen=True)
class BackfillResult:
    before_count: int
    after_count: int
    processed: int
    failed: int
    elapsed_seconds: float

    @property
    def chunks_per_second(self) -> float:
        return self.processed / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


def _chunk_columns(store: VectorStore) -> set[str]:
    return {row[1] for row in store.conn.cursor().execute("PRAGMA table_info(chunks)")}


def _active_lifecycle_clauses(store: VectorStore) -> list[str]:
    columns = _chunk_columns(store)
    clauses = []
    if "archived_at" in columns:
        clauses.append("c.archived_at IS NULL")
    if "superseded_by" in columns:
        clauses.append("c.superseded_by IS NULL")
    if "aggregated_into" in columns:
        clauses.append("c.aggregated_into IS NULL")
    return clauses


def _pending_where_sql(store: VectorStore, *, include_inactive: bool) -> str:
    clauses = [
        "v.chunk_id IS NULL",
        "c.content IS NOT NULL",
        "c.content != ''",
    ]
    if not include_inactive:
        clauses.extend(_active_lifecycle_clauses(store))
    return " AND ".join(clauses)


def count_unvectored_chunks(store: VectorStore, *, include_inactive: bool = False) -> int:
    """Count chunks with no float vector row.

    By default this matches normal semantic search eligibility and skips
    archived, superseded, and aggregated lifecycle rows.
    """
    where_sql = _pending_where_sql(store, include_inactive=include_inactive)
    return int(
        store.conn.cursor()
        .execute(
            f"""
            SELECT COUNT(*)
            FROM chunks c
            LEFT JOIN chunk_vectors v ON c.id = v.chunk_id
            WHERE {where_sql}
            """
        )
        .fetchone()[0]
    )


def fetch_unvectored_batch(
    store: VectorStore,
    *,
    batch_size: int,
    include_inactive: bool = False,
) -> list[PendingChunk]:
    """Fetch the next resumable batch of active chunks missing vectors."""
    where_sql = _pending_where_sql(store, include_inactive=include_inactive)
    rows = store.conn.cursor().execute(
        f"""
        SELECT c.id, c.content
        FROM chunks c
        LEFT JOIN chunk_vectors v ON c.id = v.chunk_id
        WHERE {where_sql}
        ORDER BY c.created_at ASC, c.id ASC
        LIMIT ?
        """,
        (batch_size,),
    )
    return [PendingChunk(chunk_id=str(chunk_id), content=str(content)) for chunk_id, content in rows]


def _to_float_list(embedding: Any) -> list[float]:
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    return [float(value) for value in embedding]


def write_chunk_embeddings(store: VectorStore, chunks: list[PendingChunk], embeddings: Iterable[Any]) -> int:
    """Write float and binary vectors for a batch, returning successful rows."""
    cursor = store.conn.cursor()
    written = 0
    for chunk, embedding in zip(chunks, embeddings):
        try:
            store._upsert_chunk_vector(cursor, chunk.chunk_id, _to_float_list(embedding))
            written += 1
        except Exception as exc:
            logger.warning("Failed to write embedding for %s: %s", chunk.chunk_id, exc)
    return written


def load_embedding_model(model_name: str = DEFAULT_MODEL):
    """Load the sentence-transformers model on MPS when available."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Loading %s on %s", model_name, device)
    return SentenceTransformer(model_name, device=device)


def find_heavy_ml_processes() -> list[str]:
    """Return running heavy-ML process command lines that should block this job."""
    try:
        result = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True)
    except Exception as exc:
        logger.warning("Unable to inspect process list for heavy-ML mutex: %s", exc)
        return []

    current_pid = str(os.getpid())
    matches = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid, _, command = stripped.partition(" ")
        if pid == current_pid:
            continue
        lower = command.lower()
        if any(pattern in lower for pattern in HEAVY_ML_PATTERNS):
            matches.append(stripped)
    return matches


def run_reembed_backfill(
    *,
    db_path: str | Path,
    model: BatchEmbeddingModel | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    limit: int | None = None,
    dry_run: bool = False,
    include_inactive: bool = False,
    progress_every: int = 10,
) -> BackfillResult:
    """Batch-embed all chunks missing vectors.

    The run is resumable because each batch is selected with a LEFT JOIN against
    `chunk_vectors`; reruns skip rows that already received vectors.
    """
    store = VectorStore(db_path)
    start = time.monotonic()
    processed = 0
    failed = 0
    before_count = count_unvectored_chunks(store, include_inactive=include_inactive)

    try:
        if dry_run or before_count == 0:
            elapsed = time.monotonic() - start
            return BackfillResult(
                before_count=before_count,
                after_count=before_count,
                processed=0,
                failed=0,
                elapsed_seconds=elapsed,
            )

        active_model = model or load_embedding_model(model_name)
        batch_number = 0
        while True:
            remaining_limit = None if limit is None else max(limit - processed, 0)
            if remaining_limit == 0:
                break
            effective_batch_size = batch_size if remaining_limit is None else min(batch_size, remaining_limit)
            chunks = fetch_unvectored_batch(
                store,
                batch_size=effective_batch_size,
                include_inactive=include_inactive,
            )
            if not chunks:
                break

            texts = [chunk.content for chunk in chunks]
            embeddings = active_model.encode(
                texts,
                batch_size=effective_batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            written = write_chunk_embeddings(store, chunks, embeddings)
            processed += written
            failed += len(chunks) - written
            batch_number += 1

            if batch_number % progress_every == 0:
                elapsed = time.monotonic() - start
                rate = processed / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "Reembed backfill progress: processed=%d failed=%d remaining=%d rate=%.1f chunks/s",
                    processed,
                    failed,
                    count_unvectored_chunks(store, include_inactive=include_inactive),
                    rate,
                )

        after_count = count_unvectored_chunks(store, include_inactive=include_inactive)
        elapsed = time.monotonic() - start
        return BackfillResult(
            before_count=before_count,
            after_count=after_count,
            processed=processed,
            failed=failed,
            elapsed_seconds=elapsed,
        )
    finally:
        store.close()
