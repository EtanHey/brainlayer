"""New indexing pipeline using sqlite-vec and sentence-transformers."""

import logging
from pathlib import Path
from typing import Callable, List, Optional

from .embeddings import embed_chunks
from .pipeline.chunk import Chunk
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

from .paths import DEFAULT_DB_PATH


def index_chunks_to_sqlite(
    chunks: List[Chunk],
    source_file: str,
    project: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Index chunks to sqlite-vec database."""
    if not chunks:
        return 0

    # Generate embeddings
    embedded_chunks = embed_chunks(chunks, on_progress=on_progress)

    if not embedded_chunks:
        return 0

    # Try to get timestamp from source file (first JSONL message)
    created_at = None
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
    except Exception:
        pass
    if not created_at:
        from datetime import datetime, timezone

        created_at = datetime.now(timezone.utc).isoformat()

    # Prepare data for vector store
    chunk_data = []
    embeddings = []

    for i, ec in enumerate(embedded_chunks):
        chunk = ec.chunk

        chunk_id = f"{source_file}:{i}"

        chunk_data.append(
            {
                "id": chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "source_file": source_file,
                "project": project,
                "content_type": chunk.content_type.value,
                "value_type": chunk.value.value,
                "char_count": chunk.char_count,
                "created_at": created_at,
            }
        )

        embeddings.append(ec.embedding)

    # Store in database
    with VectorStore(db_path) as store:
        return store.upsert_chunks(chunk_data, embeddings)


def get_stats(db_path: Path = DEFAULT_DB_PATH) -> dict:
    """Get database statistics."""
    with VectorStore(db_path) as store:
        return store.get_stats()
