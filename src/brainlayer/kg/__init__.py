"""BrainLayer Knowledge Graph — standardized KG spec (matches Convex kgSpec.ts)."""

import math
from typing import Optional

# ── Shared Constants (must match kgSpec.ts in 6PM) ──────────────────────────

ENTITY_TYPES = [
    "person",
    "constraint",
    "preference",
    "life_event",
    "meeting",
    "location",
    "organization",
]

RELATION_TYPES = [
    "has_constraint",
    "has_preference",
    "blocked_during",
    "attended",
    "organized_by",
    "knows",
    "works_at",
    "supersedes",
    "held_at",
]

# Half-life decay constants (lambda) for time-based relevance scoring
DECAY_CONSTANTS: dict[str, float] = {
    "constraint": 0.0019,  # ~365 day half-life
    "preference": 0.0077,  # ~90 day half-life
    "life_event": 0,  # date-bounded, no decay
    "casual": 0.0231,  # ~30 day half-life
    "meeting": 0.0046,  # ~150 day half-life
}


def effective_score(
    confidence: float,
    importance: float,
    age_days: float,
    entity_type: Optional[str] = None,
) -> float:
    """Calculate time-decayed effective score for a KG entity or relation.

    Score = confidence * importance * exp(-lambda * age_days)

    Args:
        confidence: How certain we are about this fact (0-1)
        importance: How important this fact is (0-1)
        age_days: Age in days since creation
        entity_type: Entity type for decay rate lookup (defaults to 'preference' rate)
    """
    lam = DECAY_CONSTANTS.get(entity_type or "", 0.0077)
    return confidence * importance * math.exp(-lam * age_days)
