"""Stage 5: Index embeddings to ChromaDB."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from .embed import EmbeddedChunk


# Default storage location
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "zikaron" / "chromadb"


def get_db_path() -> Path:
    """Get the database storage path."""
    return DEFAULT_DB_PATH


def get_client(db_path: Path = DEFAULT_DB_PATH) -> chromadb.Client:
    """Get ChromaDB client with persistent storage."""
    db_path.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )


def get_or_create_collection(
    client: chromadb.Client,
    name: str = "conversations"
) -> chromadb.Collection:
    """Get or create the main collection."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )


# ChromaDB max batch size
CHROMADB_BATCH_SIZE = 5000


def index_to_chromadb(
    embedded_chunks: list[EmbeddedChunk],
    collection: chromadb.Collection,
    source_file: str,
    project: str | None = None
) -> int:
    """
    Index embedded chunks to ChromaDB.

    Args:
        embedded_chunks: Chunks with embeddings
        collection: ChromaDB collection
        source_file: Source JSONL file path
        project: Project name/path for filtering

    Returns:
        Number of chunks indexed
    """
    if not embedded_chunks:
        return 0

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, ec in enumerate(embedded_chunks):
        chunk = ec.chunk

        # Generate unique ID
        chunk_id = f"{source_file}:{i}"

        ids.append(chunk_id)
        embeddings.append(ec.embedding)
        documents.append(chunk.content)
        metadatas.append({
            "source_file": source_file,
            "project": project or "unknown",
            "content_type": chunk.content_type.value,
            "value": chunk.value.value,
            "char_count": chunk.char_count,
            **{k: str(v) for k, v in chunk.metadata.items()}  # Stringify metadata values
        })

    # Upsert in batches to respect ChromaDB limits
    for i in range(0, len(ids), CHROMADB_BATCH_SIZE):
        batch_end = min(i + CHROMADB_BATCH_SIZE, len(ids))
        collection.upsert(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )

    return len(ids)


def search(
    collection: chromadb.Collection,
    query_embedding: list[float] | None = None,
    n_results: int = 10,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
    hybrid: bool = False,
    query_text: str | None = None
) -> dict:
    """
    Search the collection using hybrid retrieval or traditional methods.

    Args:
        collection: ChromaDB collection
        query_embedding: Query embedding vector (optional, for semantic search)
        n_results: Number of results to return
        where: Metadata filter
        where_document: Document content filter (for text search with $contains)
        hybrid: Use hybrid BM25 + semantic search
        query_text: Original query text (required for hybrid search)

    Returns:
        Search results with documents, metadatas, and distances
    """
    if hybrid and query_text:
        # Use hybrid search
        from ..dashboard.search import HybridSearchEngine
        engine = HybridSearchEngine()
        return engine.search(collection, query_text, n_results, where)
    
    # Original search logic
    if query_embedding is None and where_document is None:
        raise ValueError("Either query_embedding or where_document must be provided")
    
    if query_embedding is not None:
        # Semantic search with embeddings
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )
    else:
        # Text-only search using where_document
        # ChromaDB requires query_texts for text search
        # Extract the search term from where_document if it's a $contains filter
        query_text = None
        if where_document and "$contains" in where_document:
            query_text = where_document["$contains"]
        
        if query_text:
            try:
                # Try query_texts first (requires ChromaDB with text search support)
                result = collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where,
                    where_document=where_document,
                    include=["documents", "metadatas", "distances"]
                )
                return result
            except Exception:
                # Fallback: use get() with where_document filter and filter in Python
                all_results = collection.get(
                    where=where,
                    where_document=where_document,
                    limit=min(n_results * 10, 10000),  # Get more to filter
                    include=["documents", "metadatas"]
                )
                
                # Filter documents that contain the query text
                filtered_docs = []
                filtered_metas = []
                query_lower = query_text.lower()
                
                for doc, meta in zip(all_results.get("documents", []), all_results.get("metadatas", [])):
                    if doc and query_lower in doc.lower():
                        filtered_docs.append(doc)
                        filtered_metas.append(meta)
                        if len(filtered_docs) >= n_results:
                            break
                
                # Return in same format as query()
                return {
                    "documents": [filtered_docs],
                    "metadatas": [filtered_metas],
                    "distances": [[None] * len(filtered_docs)]  # No distances for text search
                }
        else:
            # Fallback: use get() with where_document filter
            all_results = collection.get(
                where=where,
                where_document=where_document,
                limit=n_results,
                include=["documents", "metadatas"]
            )
            # Return in same format as query()
            return {
                "documents": [all_results.get("documents", [])],
                "metadatas": [all_results.get("metadatas", [])],
                "distances": [[None] * len(all_results.get("documents", []))]  # No distances
            }


def get_stats(collection: chromadb.Collection) -> dict:
    """Get collection statistics."""
    count = collection.count()

    # Sample some metadata to get project list
    if count > 0:
        sample = collection.peek(min(100, count))
        projects = set()
        content_types = set()

        for meta in sample.get("metadatas", []):
            if meta:
                projects.add(meta.get("project", "unknown"))
                content_types.add(meta.get("content_type", "unknown"))

        return {
            "total_chunks": count,
            "projects": list(projects),
            "content_types": list(content_types)
        }

    return {"total_chunks": 0, "projects": [], "content_types": []}
