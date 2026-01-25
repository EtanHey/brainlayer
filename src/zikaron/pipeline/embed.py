"""Stage 4: Generate embeddings using Ollama."""

from dataclasses import dataclass
from typing import Any

import ollama

from .chunk import Chunk


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding vector."""
    chunk: Chunk
    embedding: list[float]


# Default embedding model (local via Ollama)
DEFAULT_MODEL = "nomic-embed-text"


def embed_chunks(
    chunks: list[Chunk],
    model: str = DEFAULT_MODEL,
    batch_size: int = 32
) -> list[EmbeddedChunk]:
    """
    Generate embeddings for chunks using Ollama.

    IMPORTANT: nomic-embed-text requires prefix for optimal results:
    - "search_document: " for indexing
    - "search_query: " for querying

    Args:
        chunks: List of chunks to embed
        model: Ollama model name (default: nomic-embed-text)
        batch_size: Number of chunks to embed at once

    Returns:
        List of chunks with embeddings
    """
    results = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        for chunk in batch:
            # Add prefix for nomic-embed-text
            text = f"search_document: {chunk.content}"

            try:
                response = ollama.embeddings(model=model, prompt=text)
                embedding = response["embedding"]

                results.append(EmbeddedChunk(
                    chunk=chunk,
                    embedding=embedding
                ))
            except Exception as e:
                print(f"Warning: Failed to embed chunk: {e}")
                continue

    return results


def embed_query(query: str, model: str = DEFAULT_MODEL) -> list[float]:
    """
    Generate embedding for a search query.

    Uses "search_query: " prefix for nomic-embed-text.
    """
    text = f"search_query: {query}"

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def check_model_available(model: str = DEFAULT_MODEL) -> bool:
    """Check if the embedding model is available in Ollama."""
    try:
        models = ollama.list()
        model_names = [m["name"].split(":")[0] for m in models.get("models", [])]
        return model in model_names or f"{model}:latest" in [m["name"] for m in models.get("models", [])]
    except Exception:
        return False


def ensure_model(model: str = DEFAULT_MODEL) -> None:
    """Pull the embedding model if not available."""
    if not check_model_available(model):
        print(f"Pulling embedding model: {model}")
        ollama.pull(model)
