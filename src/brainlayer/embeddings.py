"""Fast embeddings using sentence-transformers with bge-large-en-v1.5."""

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from sentence_transformers import SentenceTransformer

from .pipeline.chunk import Chunk

logger = logging.getLogger(__name__)

# Use bge-large-en-v1.5 for high-quality embeddings (1024 dims, 63.5 MTEB score)
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024  # bge-large dimension
MAX_EMBEDDING_CHARS = 512  # context length
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding vector."""

    chunk: Chunk
    embedding: List[float]


class EmbeddingModel:
    """Sentence-transformers embedding model."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def _load_model(self) -> SentenceTransformer:
        """Load model on first use."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def embed_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int = 32,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[EmbeddedChunk]:
        """Generate embeddings for chunks."""
        if not chunks:
            return []

        model = self._load_model()
        results = []
        total = len(chunks)

        # Prepare texts with truncation
        texts = []
        for chunk in chunks:
            content = chunk.content
            if len(content) > MAX_EMBEDDING_CHARS:
                # Keep first part for context
                content = content[: MAX_EMBEDDING_CHARS - 50] + "..."
            texts.append(content)

        # Generate embeddings in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_chunks = chunks[i : i + batch_size]

            try:
                embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)

                for chunk, embedding in zip(batch_chunks, embeddings):
                    results.append(EmbeddedChunk(chunk=chunk, embedding=embedding.tolist()))

                if on_progress:
                    on_progress(len(results), total)

            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")
                continue

        return results

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query with BGE prefix."""
        model = self._load_model()

        # Truncate if too long
        if len(query) > MAX_EMBEDDING_CHARS:
            query = query[: MAX_EMBEDDING_CHARS - 3] + "..."

        # BGE models need query prefix for optimal retrieval
        prefixed_query = f"{BGE_QUERY_PREFIX}{query}"

        try:
            embedding = model.encode([prefixed_query], convert_to_numpy=True)[0]
            return embedding.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to embed query: {e}") from e


# Global model instance
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> EmbeddingModel:
    """Get global embedding model instance."""
    global _embedding_model
    if _embedding_model is None or _embedding_model.model_name != model_name:
        _embedding_model = EmbeddingModel(model_name)
    return _embedding_model


def embed_chunks(
    chunks: List[Chunk],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[EmbeddedChunk]:
    """Generate embeddings for chunks using global model."""
    model = get_embedding_model(model_name)
    return model.embed_chunks(chunks, batch_size, on_progress)


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> List[float]:
    """Generate embedding for search query using global model."""
    model = get_embedding_model(model_name)
    return model.embed_query(query)
