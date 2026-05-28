"""Agent ranking profile helpers.

Profile JSON schema:

```json
{
  "boost_weights": {
    "fts_weight": 1.4,
    "vector_weight": 0.9,
    "importance_weight": 1.0,
    "recency_weight": 1.0,
    "recency_intent_weight": 1.0,
    "decay_weight": 1.0,
    "kg_weight": 1.0
  },
  "source_weights": {
    "precompact": 1.3
  }
}
```

All weights are optional positive numbers. Missing profile rows, missing keys,
and unknown agents fall back to weight `1.0`, preserving current ranking.
Short keys such as `"fts"` are accepted as aliases for `"fts_weight"`.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .chunk_origin import CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT

MAX_PROFILE_WEIGHT = 1000.0

DEFAULT_AGENT_PROFILES: dict[str, dict[str, Any]] = {
    "orcClaude": {
        "boost_weights": {
            "fts_weight": 3.0,
            "vector_weight": 0.5,
            "importance_weight": 1.0,
            "recency_weight": 1.0,
            "recency_intent_weight": 1.0,
            "decay_weight": 1.0,
            "kg_weight": 1.0,
        },
        "source_weights": {},
    },
    "coachClaude": {
        "boost_weights": {
            "fts_weight": 0.85,
            "vector_weight": 1.3,
            "importance_weight": 1.05,
            "recency_weight": 1.0,
            "recency_intent_weight": 1.0,
            "decay_weight": 1.0,
            "kg_weight": 1.0,
        },
        "source_weights": {},
    },
    "skillCreatorClaude": {
        "boost_weights": {
            "fts_weight": 1.0,
            "vector_weight": 1.0,
            "importance_weight": 1.0,
            "recency_weight": 1.0,
            "recency_intent_weight": 1.0,
            "decay_weight": 1.0,
            "kg_weight": 1.0,
        },
        "source_weights": {"precompact": 1.3},
    },
}


def load_profile_json(raw: str) -> dict[str, Any]:
    """Parse and validate an agent profile JSON object."""
    try:
        profile = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc
    return validate_agent_profile(profile)


def validate_agent_profile(profile: Any) -> dict[str, Any]:
    """Return a normalized profile or raise ValueError."""
    if not isinstance(profile, dict):
        raise ValueError("Agent profile must be a JSON object")

    normalized: dict[str, Any] = {}
    boost_weights = _validate_weight_map(profile.get("boost_weights", {}), "boost_weights")
    source_weights = _validate_weight_map(profile.get("source_weights", {}), "source_weights")
    normalized["boost_weights"] = boost_weights
    normalized["source_weights"] = source_weights
    return normalized


def boost_weight(profile: dict[str, Any] | None, feature: str) -> float:
    """Return the configured feature boost weight, defaulting to 1.0."""
    if not profile:
        return 1.0
    weights = profile.get("boost_weights") or {}
    value = weights.get(feature, weights.get(f"{feature}_weight", 1.0))
    return float(value)


def source_weight(profile: dict[str, Any] | None, source: str | None) -> float:
    """Return the configured source boost weight, defaulting to 1.0."""
    if not profile or not source:
        return 1.0
    weights = profile.get("source_weights") or {}
    weight = weights.get(source)
    if weight is None and source == CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT:
        weight = weights.get("precompact")
    return float(weight or 1.0)


def _validate_weight_map(value: Any, field_name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")

    normalized: dict[str, float] = {}
    for key, weight in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be non-empty strings")
        key_stripped = key.strip()
        if not key_stripped:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if key_stripped in normalized:
            raise ValueError(f"{field_name}.{key_stripped} is duplicated after trimming whitespace")
        if isinstance(weight, bool) or not isinstance(weight, int | float):
            raise ValueError(f"{field_name}.{key_stripped} must be a number")
        numeric = float(weight)
        if not math.isfinite(numeric) or numeric <= 0 or numeric > MAX_PROFILE_WEIGHT:
            raise ValueError(f"{field_name}.{key_stripped} must be positive and <= {MAX_PROFILE_WEIGHT:g}")
        normalized[key_stripped] = numeric
    return normalized
