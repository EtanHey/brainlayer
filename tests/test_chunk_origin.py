"""Ingest chunk_origin is not an enrichment-model field."""

from brainlayer.chunk_origin import (
    CHUNK_ORIGIN_UNKNOWN,
    VALID_CHUNK_ORIGINS,
    detect_chunk_origin,
)


def test_detect_chunk_origin_does_not_accept_model_names():
    for model_name in ("groq", "ollama", "mlx", "gemini-2.5-flash-lite"):
        assert model_name not in VALID_CHUNK_ORIGINS
        assert detect_chunk_origin("ordinary memory", model_name) == CHUNK_ORIGIN_UNKNOWN
