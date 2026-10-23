"""Shared calendar index admission and cooperative-unwind allowance."""

from datetime import datetime

CALENDAR_INDEX_UNWIND_S = 30.0
CALENDAR_INDEX_FINALIZE_RESERVE_S = 30.0


def calendar_index_work_budget_s(budget_s: float) -> float:
    """Yield early inside the existing budget; outer custody remains budget + unwind.

    Short caller budgets retain half for work. The reserve covers close/fsync/publication,
    in addition to the existing outer allowance, without extending writer custody.
    """
    return budget_s - min(CALENDAR_INDEX_FINALIZE_RESERVE_S, budget_s / 2)


def calendar_index_budget_s(requested: float, now: datetime | None = None, *, reserve_s: float = 60) -> float:
    now = now or datetime.now().astimezone()
    if 4 <= now.hour < 6:
        return 0.0
    budget = min(requested, 1800.0)
    if now.hour < 4:
        cutoff = now.replace(hour=4, minute=0, second=0, microsecond=0)
        budget = min(budget, max(0.0, (cutoff - now).total_seconds() - reserve_s))
    return budget
