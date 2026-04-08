import math
import time

MIN_DECAY_SCORE = 0.05
PINNED_SCORE = 1.0


def compute_decay_score(
    half_life_days: float,
    last_retrieved: float | None,
    created_at: float,
    retrieval_count: int,
    pinned: bool,
) -> float:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be > 0")
    if retrieval_count < 0:
        raise ValueError("retrieval_count must be >= 0")

    if pinned:
        return PINNED_SCORE

    reference = last_retrieved if last_retrieved is not None else created_at
    elapsed_days = max((time.time() - reference) / 86400.0, 0.0)
    effective_half_life = half_life_days * (1.0 + math.log1p(retrieval_count) * 0.3)
    decay = (1.0 + elapsed_days / (9.0 * effective_half_life)) ** -1.0
    return max(MIN_DECAY_SCORE, decay)
