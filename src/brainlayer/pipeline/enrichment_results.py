"""Local validation and metadata stamps for historical enrichment results."""

import json
import logging
import math
import os
from typing import Any, Dict, Optional

from ..tag_normalization import (
    enrichment_tag_mode,
    normalize_enrichment_tag_values,
    taxonomy_content_sha,
    taxonomy_git_sha,
)
from .cloud_scrub import scrub_llm_output
from .entity_extraction import normalize_entity_type

logger = logging.getLogger(__name__)


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


def parse_enrichment(text: str) -> Optional[Dict[str, Any]]:
    """Parse GLM's JSON response into enrichment metadata."""
    if not text:
        return None
    try:
        # Find JSON in response
        match = None
        for start in range(len(text)):
            if text[start] == "{":
                for end in range(len(text) - 1, start, -1):
                    if text[end] == "}":
                        try:
                            match = json.loads(text[start : end + 1])
                            break
                        except json.JSONDecodeError:
                            continue
                if match:
                    break

        if not match:
            return None

        # The model may echo a secret from its prompt into any field. Scrub the
        # whole payload before normalizing; a scrub failure lands in the except
        # below and returns None, so nothing unscrubbed is ever persisted.
        match = scrub_llm_output(match)

        # Validate and normalize
        result: Dict[str, Any] = {}

        summary = match.get("summary", "")
        if isinstance(summary, str) and len(summary) > 5:
            result["summary"] = summary[:500]  # Cap at 500 chars

        result["tags"] = normalize_enrichment_tags(match.get("tags", []))
        result["enrichment_metadata"] = enrichment_version_metadata()

        importance = match.get("importance")
        if isinstance(importance, (int, float)):
            result["importance"] = max(1.0, min(10.0, float(importance)))

        intent = match.get("intent", "")
        if isinstance(intent, str) and intent.lower().strip() in VALID_INTENTS:
            result["intent"] = intent.lower().strip()

        # Extended fields (graceful — missing is OK)
        primary_symbols = match.get("primary_symbols", [])
        if isinstance(primary_symbols, list):
            cleaned = [str(s).strip() for s in primary_symbols if isinstance(s, str) and s.strip()][:20]
            if cleaned:
                result["primary_symbols"] = cleaned

        resolved_query = match.get("resolved_query", "")
        if isinstance(resolved_query, str) and len(resolved_query) > 10:
            result["resolved_query"] = resolved_query[:500]

        key_facts = match.get("key_facts", [])
        if isinstance(key_facts, list):
            cleaned = [str(f).strip() for f in key_facts if isinstance(f, str) and f.strip()][:30]
            if cleaned:
                result["key_facts"] = cleaned

        resolved_queries = match.get("resolved_queries", [])
        if isinstance(resolved_queries, list):
            cleaned = [str(q).strip() for q in resolved_queries if isinstance(q, str) and len(q.strip()) > 10][:3]
            if cleaned:
                result["resolved_queries"] = cleaned

        epistemic_level = match.get("epistemic_level", "")
        if isinstance(epistemic_level, str) and epistemic_level.lower().strip() in VALID_EPISTEMIC:
            result["epistemic_level"] = epistemic_level.lower().strip()

        version_scope = match.get("version_scope")
        if isinstance(version_scope, str) and version_scope.strip() and version_scope.lower() != "null":
            result["version_scope"] = version_scope.strip()[:200]

        debt_impact = match.get("debt_impact", "")
        if isinstance(debt_impact, str) and debt_impact.lower().strip() in VALID_DEBT_IMPACT:
            result["debt_impact"] = debt_impact.lower().strip()

        sentiment_label = match.get("sentiment_label", "")
        if isinstance(sentiment_label, str) and sentiment_label.lower().strip() in VALID_SENTIMENTS:
            result["sentiment_label"] = sentiment_label.lower().strip()

        sentiment_score = match.get("sentiment_score")
        if isinstance(sentiment_score, (int, float)) and not isinstance(sentiment_score, bool):
            try:
                normalized_score = float(sentiment_score)
            except (OverflowError, ValueError):
                normalized_score = None
            if normalized_score is not None and math.isfinite(normalized_score):
                result["sentiment_score"] = max(-1.0, min(1.0, normalized_score))

        sentiment_signals = match.get("sentiment_signals", [])
        if isinstance(sentiment_signals, list):
            cleaned = []
            seen = set()
            for signal in sentiment_signals:
                if not isinstance(signal, str):
                    continue
                normalized = signal.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                cleaned.append(normalized)
                if len(cleaned) == 10:
                    break
            if cleaned:
                result["sentiment_signals"] = cleaned

        external_deps = match.get("external_deps", [])
        if isinstance(external_deps, list):
            cleaned = [str(d).strip().lower() for d in external_deps if isinstance(d, str) and d.strip()][:15]
            if cleaned:
                result["external_deps"] = cleaned

        VALID_ENTITY_TYPES = {
            "person",
            "agent",
            "company",
            "project",
            "technology",
            "tool",
            "concept",
            "topic",
            "source",
        }
        entities = match.get("entities", [])
        if isinstance(entities, list):
            cleaned_entities = []
            for e in entities:
                if isinstance(e, dict):
                    name = e.get("name", "")
                    etype = e.get("type", "")
                    if isinstance(name, str) and name.strip() and isinstance(etype, str) and etype.strip():
                        relation = e.get("relation")
                        entity_type, entity_subtype = normalize_entity_type(
                            name,
                            etype,
                            e.get("entity_subtype") or e.get("subtype"),
                        )
                        if entity_type not in VALID_ENTITY_TYPES:
                            continue
                        entity = {"name": name.strip(), "type": entity_type}
                        if entity_subtype:
                            entity["entity_subtype"] = entity_subtype
                        if isinstance(relation, str) and relation.strip() and relation.lower() != "null":
                            entity["relation"] = relation.strip()
                        cleaned_entities.append(entity)
            if cleaned_entities:
                result["entities"] = cleaned_entities[:20]

        # Must have at least summary + tags to be valid
        if "summary" in result and "tags" in result:
            return result
        return None

    except Exception as e:
        logger.debug("Enrichment result validation failed: %s", e)
        return None
