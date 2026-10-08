"""Embeddings for communication style analysis.

Uses StyleDistance (HuggingFace) — the best model for style-aware clustering.
General embeddings (Qwen3, bge-m3) cluster by topic, not style; StyleDistance
clusters by how you write (formality, emoji, punctuation, phrasing).
"""

from importlib.util import find_spec
from typing import Optional

from .unified_timeline import UnifiedMessage

HAS_SENTENCE_TRANSFORMERS = find_spec("sentence_transformers") is not None

# Best for style analysis: clusters by writing style, not content
STYLE_MODEL = "StyleDistance/mstyledistance"
MAX_EMBEDDING_CHARS = 8000


def _get_model() -> "SentenceTransformer":
    """Load mStyleDistance. Multilingual (Hebrew+English)."""
    from brainlayer.embeddings import guard_embedding_model_load

    guard_embedding_model_load(STYLE_MODEL)

    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(STYLE_MODEL)


def embed_message(text: str) -> list[float]:
    """Embed a single message for style clustering."""
    content = text[:MAX_EMBEDDING_CHARS] if len(text) > MAX_EMBEDDING_CHARS else text
    encoder = _get_model()
    emb = encoder.encode(content, convert_to_numpy=True)
    return emb.tolist()


def embed_messages(
    messages: list[UnifiedMessage],
    batch_size: int = 32,
    on_progress: Optional[callable] = None,
) -> list[tuple[UnifiedMessage, list[float]]]:
    """
    Embed messages for style-aware cluster sampling.

    Returns:
        List of (message, embedding) tuples.
    """
    total = len(messages)
    encoder = _get_model()
    texts = [m.text[:MAX_EMBEDDING_CHARS] if len(m.text) > MAX_EMBEDDING_CHARS else m.text for m in messages]
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    results = [(msg, emb.tolist()) for msg, emb in zip(messages, embeddings)]
    if on_progress:
        on_progress(total, total)
    return results


def ensure_model() -> None:
    """Ensure StyleDistance is available. Downloads on first use."""
    _get_model()
