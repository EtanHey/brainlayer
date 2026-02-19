"""New indexing pipeline using sqlite-vec and sentence-transformers."""

from pathlib import Path
from typing import List, Optional, Callable
import logging

from .vector_store import VectorStore
from .embeddings import embed_chunks
from .pipeline.chunk import Chunk

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "brainlayer" / "brainlayer.db"


def index_chunks_to_sqlite(
    chunks: List[Chunk],
    source_file: str,
    project: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
    on_progress: Optional[Callable[[int, int], None]] = None
) -> int:
    """Index chunks to sqlite-vec database."""
    if not chunks:
        return 0
    
    # Generate embeddings
    embedded_chunks = embed_chunks(chunks, on_progress=on_progress)
    
    if not embedded_chunks:
        return 0
    
    # Prepare data for vector store
    chunk_data = []
    embeddings = []
    
    for i, ec in enumerate(embedded_chunks):
        chunk = ec.chunk
        
        chunk_id = f"{source_file}:{i}"
        
        chunk_data.append({
            "id": chunk_id,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "source_file": source_file,
            "project": project,
            "content_type": chunk.content_type.value,
            "value_type": chunk.value.value,
            "char_count": chunk.char_count
        })
        
        embeddings.append(ec.embedding)
    
    # Store in database
    with VectorStore(db_path) as store:
        return store.upsert_chunks(chunk_data, embeddings)


def get_stats(db_path: Path = DEFAULT_DB_PATH) -> dict:
    """Get database statistics."""
    with VectorStore(db_path) as store:
        return store.get_stats()
