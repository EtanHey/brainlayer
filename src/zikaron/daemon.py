"""FastAPI daemon service for fast zikaron queries."""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .vector_store import VectorStore
from .embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"
SOCKET_PATH = Path("/tmp/zikaron.sock")

# Global state
vector_store: Optional[VectorStore] = None
embedding_model = None


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    n_results: int = 10
    project_filter: Optional[str] = None
    content_type_filter: Optional[str] = None
    source_filter: Optional[str] = None
    use_semantic: bool = True
    hybrid: bool = True


class SearchResponse(BaseModel):
    """Search response model."""
    ids: List[Optional[str]] = []
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[Optional[float]]
    total_time_ms: float


class StatsResponse(BaseModel):
    """Stats response model."""
    total_chunks: int
    projects: List[str]
    content_types: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global vector_store, embedding_model
    
    # Startup
    logger.info("Starting zikaron daemon...")
    
    # Initialize vector store
    vector_store = VectorStore(DEFAULT_DB_PATH)
    logger.info(f"Loaded vector store: {vector_store.count()} chunks")
    
    # Pre-load embedding model
    embedding_model = get_embedding_model()
    logger.info(f"Loaded embedding model: {embedding_model.model_name}")
    
    # Warm up model with dummy query
    try:
        embedding_model.embed_query("test query")
        logger.info("Model warmed up successfully")
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down zikaron daemon...")
    if vector_store:
        vector_store.close()


app = FastAPI(
    title="Zikaron Daemon",
    description="Fast search daemon for zikaron knowledge base",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "chunks": vector_store.count() if vector_store else 0}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get collection statistics."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    stats = vector_store.get_stats()
    return StatsResponse(**stats)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search the knowledge base."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    import time
    start_time = time.time()
    
    try:
        if request.hybrid and request.use_semantic:
            # Hybrid search: semantic + FTS5 keyword via RRF
            query_embedding = embedding_model.embed_query(request.query)
            results = vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=request.query,
                n_results=request.n_results,
                project_filter=request.project_filter,
                content_type_filter=request.content_type_filter,
                source_filter=request.source_filter
            )
        elif request.use_semantic:
            # Semantic-only search
            query_embedding = embedding_model.embed_query(request.query)
            results = vector_store.search(
                query_embedding=query_embedding,
                n_results=request.n_results,
                project_filter=request.project_filter,
                content_type_filter=request.content_type_filter,
                source_filter=request.source_filter
            )
        else:
            # Text-only search
            results = vector_store.search(
                query_text=request.query,
                n_results=request.n_results,
                project_filter=request.project_filter,
                content_type_filter=request.content_type_filter,
                source_filter=request.source_filter
            )
        
        total_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            ids=results.get("ids", [[]])[0],
            documents=results["documents"][0],
            metadatas=results["metadatas"][0],
            distances=results["distances"][0],
            total_time_ms=total_time_ms
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/context/{chunk_id}")
async def get_context(chunk_id: str, before: int = 3, after: int = 3):
    """Get surrounding conversation context for a chunk."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        result = vector_store.get_context(chunk_id, before=before, after=after)
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Context lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def setup_signal_handlers():
    """Setup graceful shutdown signal handlers."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def main():
    """Main daemon entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Remove existing socket
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    
    # Run server
    config = uvicorn.Config(
        app,
        uds=str(SOCKET_PATH),
        log_level="info",
        access_log=False
    )
    
    server = uvicorn.Server(config)
    
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Daemon stopped by user")
    except Exception as e:
        logger.error(f"Daemon failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
