"""Local validation and metadata stamps for historical enrichment results."""

import os
from typing import Any

from ..tag_normalization import (
    enrichment_tag_mode,
    normalize_enrichment_tag_values,
    taxonomy_content_sha,
    taxonomy_git_sha,
)


def _detect_default_backend() -> str:
    """Auto-detect the best enrichment backend for this platform.

    arm64 Mac → mlx (native Apple Silicon, no Docker overhead)
    Everything else → ollama (universal, works everywhere)
    """
    import platform

    explicit = os.environ.get("BRAINLAYER_ENRICH_BACKEND")
    if explicit:
        return explicit

    if platform.machine() == "arm64" and platform.system() == "Darwin":
        return "mlx"
    return "ollama"


ENRICH_BACKEND = _detect_default_backend()


MODEL = os.environ.get("BRAINLAYER_ENRICH_MODEL", "glm-4.7-flash")


ENRICHMENT_PROMPT_VERSION = os.environ.get("BRAINLAYER_ENRICHMENT_PROMPT_VERSION", "r82-hybrid-taxonomy")


HIGH_VALUE_TYPES = ["ai_code", "stack_trace", "user_message", "assistant_text"]


VALID_INTENTS = [
    "debugging",
    "designing",
    "configuring",
    "discussing",
    "deciding",
    "implementing",
    "reviewing",
]


VALID_EPISTEMIC = ["hypothesis", "substantiated", "validated"]


VALID_DEBT_IMPACT = ["introduction", "resolution", "none"]


VALID_SENTIMENTS = ["frustration", "confusion", "positive", "satisfaction", "neutral"]


def normalize_enrichment_tags(tags: Any, *, limit: int = 10) -> list[str]:
    return normalize_enrichment_tag_values(tags, limit=limit)


def enrichment_version_metadata(*, model: str | None = None, backend: str | None = None) -> dict[str, str]:
    return {
        "prompt_version": ENRICHMENT_PROMPT_VERSION,
        "taxonomy_git_sha": taxonomy_git_sha(),
        "taxonomy_content_sha": taxonomy_content_sha(),
        "tag_mode": enrichment_tag_mode(),
        "model": model or os.environ.get("BRAINLAYER_ENRICHMENT_MODEL_STAMP", MODEL),
        "enriched_by": model or os.environ.get("BRAINLAYER_ENRICHMENT_MODEL_STAMP", MODEL),
        "backend": backend or os.environ.get("BRAINLAYER_ENRICHMENT_BACKEND_STAMP", ENRICH_BACKEND),
        "run_id": os.environ.get("BRAINLAYER_ENRICHMENT_RUN_ID", f"pid-{os.getpid()}"),
    }
