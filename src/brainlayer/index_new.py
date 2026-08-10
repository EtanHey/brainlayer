"""New indexing pipeline using sqlite-vec and sentence-transformers."""

import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

from .agent_provenance import derive_source_class
from .claude_paths import extract_claude_conversation_id as _extract_claude_conversation_id
from .embeddings import embed_chunks
from .pipeline.chunk import Chunk
from .runtime_store import ReadonlyStore, open_writer_store
from .system_prompt_guard import looks_like_system_prompt
from .vector_store import IndexDeadlineExceeded, VectorStore

logger = logging.getLogger(__name__)

from .paths import get_db_path


def index_chunks_to_sqlite(
    chunks: List[Chunk],
    source_file: str,
    project: Optional[str] = None,
    db_path: Path | None = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    deadline_monotonic: float | None = None,
    store: VectorStore | None = None,
) -> int:
    """Index chunks to sqlite-vec database."""
    if not chunks:
        return 0

    filtered_chunks = [
        chunk
        for chunk in chunks
        if not chunk.metadata.get("is_system_prompt") and not looks_like_system_prompt(chunk.content)
    ]

    filtered_count = len(chunks) - len(filtered_chunks)
    if filtered_count:
        logger.info("Skipping %s system prompt chunks from %s", filtered_count, source_file)

    if not filtered_chunks:
        return 0

    # Generate embeddings
    def embedding_progress(completed: int, total: int) -> None:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise IndexDeadlineExceeded(processed_count=0)
        if on_progress is not None:
            on_progress(completed, total)

    embedded_chunks = embed_chunks(
        filtered_chunks,
        on_progress=embedding_progress if deadline_monotonic is not None else on_progress,
    )

    if not embedded_chunks:
        return 0

    # Try to get timestamp from a text source (first JSONL message). SQLite
    # adapters already put timestamps in chunk metadata; opening a live DB as
    # JSONL would wastefully read binary data and can block on a large file.
    created_at = None
    if Path(source_file).suffix.lower() not in {".sqlite", ".db"}:
        try:
            import json as _json

            with open(source_file) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line:
                        continue
                    _data = _json.loads(_line)
                    if "timestamp" in _data:
                        created_at = _data["timestamp"]
                        break
        except Exception as e:
            logger.debug("Could not extract timestamp from %s: %s", source_file, e)
    if not created_at:
        from datetime import datetime, timezone

        created_at = datetime.now(timezone.utc).isoformat()

    # Derive conversation_id: prefer session_id from chunk metadata,
    # fall back to the JSONL filename stem (which is the session UUID).
    file_stem = Path(source_file).stem
    claude_conversation_id = _extract_claude_conversation_id(source_file)

    # Prepare data for vector store
    chunk_data = []
    embeddings = []

    for i, ec in enumerate(embedded_chunks):
        chunk = ec.chunk
        metadata = dict(chunk.metadata)

        chunk_id = metadata.get("chunk_id") or f"{source_file}:{i}"
        conversation_id = metadata.get("conversation_id") or metadata.get("session_id") or file_stem
        if claude_conversation_id:
            metadata["claude_conversation_id"] = claude_conversation_id

        chunk_data.append(
            {
                "id": chunk_id,
                "content": chunk.content,
                "metadata": metadata,
                "source_file": source_file,
                "project": metadata.get("project") or project,
                "content_type": chunk.content_type.value,
                "value_type": chunk.value.value,
                "char_count": chunk.char_count,
                "created_at": metadata.get("created_at") or created_at,
                "conversation_id": conversation_id,
                "position": i,
                "sender": metadata.get("sender"),
                "source": metadata.get("source", "claude_code"),
                "provenance_class": metadata.get("provenance_class"),
                "source_class": metadata.get("source_class")
                or derive_source_class(
                    source_file,
                    provenance_class=metadata.get("provenance_class"),
                    source=metadata.get("source", "claude_code"),
                    content=chunk.content,
                ),
                "source_uri": metadata.get("source_uri"),
                "allow_duplicate": metadata.get("allow_duplicate", False),
            }
        )

        embeddings.append(ec.embedding)

    # A complete CLI index run injects one shared runtime store. Standalone
    # callers still get one bounded runtime open for this adapter invocation.
    if store is not None:
        return store.upsert_chunks(chunk_data, embeddings, deadline_monotonic=deadline_monotonic)
    with open_writer_store(db_path or get_db_path()) as opened_store:
        return opened_store.upsert_chunks(chunk_data, embeddings, deadline_monotonic=deadline_monotonic)


def get_stats(db_path: Path | None = None) -> dict:
    """Get database statistics."""
    with ReadonlyStore(db_path or get_db_path()) as store:
        return store.get_stats()
