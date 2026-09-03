"""Fast embeddings using sentence-transformers with bge-large-en-v1.5."""

# `from __future__ import annotations` keeps every annotation in this module a
# lazy string, so a SentenceTransformer type hint never forces the (~3.7s) torch
# import at module-load time. torch + sentence_transformers are imported lazily
# inside _load_model() — this keeps `import brainlayer.embeddings` cheap, which
# is what lets the MCP server answer its initialize/tools/list handshake fast
# instead of stalling on the startup critical path (fix/mcp-lazy-connect).
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

from .pipeline.chunk import Chunk

if TYPE_CHECKING:  # for type-checkers only; never imported at runtime module-load
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Use bge-large-en-v1.5 for high-quality embeddings (1024 dims, 63.5 MTEB score)
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024  # bge-large dimension
# bge-large-en-v1.5 supports 512 tokens (~2000+ characters).
# sentence-transformers handles token-level truncation natively — no char truncation needed.
MAX_QUERY_CHARS = 2000  # generous cap for query strings only (avoids degenerate inputs)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Suite hygiene, enforced where the cost is actually paid. tests/conftest.py sets this for every
# test that is not marked `embedding_model`, and it is inherited by subprocesses, so a spawned
# script cannot quietly load a 2.5 GB model that the in-process guards would have refused.
FORBID_MODEL_LOAD_ENV = "BRAINLAYER_FORBID_EMBEDDING_MODEL"


def guard_embedding_model_load(model_name: str) -> None:
    """Raise when this process has been told it must not load an embedding model."""
    if os.environ.get(FORBID_MODEL_LOAD_ENV) == "1":
        raise RuntimeError(
            f"{FORBID_MODEL_LOAD_ENV}=1: refusing to load `{model_name}`. A test that really needs "
            "a real embedding model must be marked `@pytest.mark.embedding_model`."
        )


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding vector."""

    chunk: Chunk
    embedding: List[float]


class EmbeddingModel:
    """Sentence-transformers embedding model."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model: Optional[SentenceTransformer] = None

    def _load_model(self) -> SentenceTransformer:
        """Load model on first use.

        torch + sentence_transformers are imported here (not at module scope) so
        that merely importing this module — e.g. during MCP server startup
        validation — never pays the ~3.7s torch import cost.
        """
        if self._model is None:
            guard_embedding_model_load(self.model_name)

            from sentence_transformers import SentenceTransformer

            device = self.device
            if device is None:
                import torch

                device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info("Loading embedding model: %s device=%s", self.model_name, device)
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

        # Pass full content — sentence-transformers tokenizes and truncates at the
        # model's actual token limit (512 tokens ≈ 2000+ chars), so content beyond
        # 512 chars is now included in the embedding instead of being discarded.
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_chunks = chunks[i : i + batch_size]

            try:
                embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)

                for chunk, embedding in zip(batch_chunks, embeddings):
                    results.append(EmbeddedChunk(chunk=chunk, embedding=embedding.tolist()))

            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")
                continue

            if on_progress:
                on_progress(len(results), total)

        return results

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query with BGE prefix."""
        model = self._load_model()

        # Cap degenerate query inputs; model handles token truncation internally
        if len(query) > MAX_QUERY_CHARS:
            query = query[:MAX_QUERY_CHARS]

        # BGE models need query prefix for optimal retrieval
        prefixed_query = f"{BGE_QUERY_PREFIX}{query}"

        try:
            embedding = model.encode([prefixed_query], convert_to_numpy=True)[0]
            return embedding.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to embed query: {e}") from e

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Generate passage embeddings for stored chunk content in batches."""
        if not texts:
            return []

        model = self._load_model()
        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            raise RuntimeError(f"Failed to embed text batch: {e}") from e


# Global model instances, keyed by model and requested device.
_embedding_models: dict[tuple[str, str | None], EmbeddingModel] = {}


def get_embedding_model(model_name: str = DEFAULT_MODEL, *, device: str | None = None) -> EmbeddingModel:
    """Get global embedding model instance."""
    key = (model_name, device)
    if key not in _embedding_models:
        _embedding_models[key] = EmbeddingModel(model_name, device=device)
    return _embedding_models[key]


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
