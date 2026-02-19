"""Time-based batching for longitudinal analysis.

Groups messages by time periods (half-year by default) for evolution tracking.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .unified_timeline import UnifiedMessage, UnifiedTimeline


@dataclass
class TimeBatch:
    """A batch of messages from a specific time period."""

    period: str  # e.g., "2024-H1", "2025-H2"
    start_date: datetime
    end_date: datetime
    messages: list[UnifiedMessage]

    @property
    def count(self) -> int:
        return len(self.messages)

    @property
    def hebrew_messages(self) -> list[UnifiedMessage]:
        return [m for m in self.messages if m.language == "hebrew"]

    @property
    def english_messages(self) -> list[UnifiedMessage]:
        return [m for m in self.messages if m.language == "english"]

    def get_relationship_context(self) -> str:
        """Get relationship mix for prompt enrichment (e.g. '~60% family, ~30% friends')."""
        tagged = [m for m in self.messages if m.relationship_tag]
        if not tagged:
            return ""
        from collections import Counter

        counts = Counter(m.relationship_tag for m in tagged)
        total = len(tagged)
        parts = [f"~{100 * c // total}% {tag}" for tag, c in counts.most_common(5)]
        contacts = list({m.contact_name for m in tagged if m.contact_name})[:8]
        return f"Relationship context: {', '.join(parts)}. Contacts: {', '.join(contacts)}."

    def get_stats(self) -> dict:
        """Get statistics for this batch."""
        if not self.messages:
            return {
                "period": self.period,
                "total": 0,
                "hebrew": 0,
                "english": 0,
                "avg_length": 0,
            }

        total_length = sum(len(m.text) for m in self.messages)

        return {
            "period": self.period,
            "total": self.count,
            "hebrew": len(self.hebrew_messages),
            "english": len(self.english_messages),
            "avg_length": total_length / self.count,
            "date_range": f"{self.start_date.date()} to {self.end_date.date()}",
        }


def get_period_key(timestamp: datetime, granularity: Literal["year", "half", "quarter", "month"] = "half") -> str:
    """
    Get the period key for a timestamp.

    Args:
        timestamp: The datetime to categorize
        granularity: How to group periods
            - "year": "2024", "2025"
            - "half": "2024-H1", "2024-H2"
            - "quarter": "2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"
            - "month": "2024-01", "2024-02"

    Returns:
        Period key string
    """
    year = timestamp.year
    month = timestamp.month

    if granularity == "year":
        return str(year)
    elif granularity == "half":
        half = "H1" if month <= 6 else "H2"
        return f"{year}-{half}"
    elif granularity == "quarter":
        quarter = (month - 1) // 3 + 1
        return f"{year}-Q{quarter}"
    elif granularity == "month":
        return f"{year}-{month:02d}"
    else:
        raise ValueError(f"Unknown granularity: {granularity}")


def get_period_dates(
    period: str, granularity: Literal["year", "half", "quarter", "month"] = "half"
) -> tuple[datetime, datetime]:
    """
    Get the start and end dates for a period.

    Returns:
        (start_date, end_date) tuple
    """
    if granularity == "year":
        year = int(period)
        return (datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59))
    elif granularity == "half":
        year, half = period.split("-")
        year = int(year)
        if half == "H1":
            return (datetime(year, 1, 1), datetime(year, 6, 30, 23, 59, 59))
        else:
            return (datetime(year, 7, 1), datetime(year, 12, 31, 23, 59, 59))
    elif granularity == "quarter":
        year, quarter = period.split("-Q")
        year = int(year)
        quarter = int(quarter)
        start_month = (quarter - 1) * 3 + 1
        end_month = quarter * 3
        if end_month == 12:
            end_day = 31
        elif end_month in [4, 6, 9, 11]:
            end_day = 30
        else:
            end_day = 31  # Simplified
        return (datetime(year, start_month, 1), datetime(year, end_month, end_day, 23, 59, 59))
    elif granularity == "month":
        year, month = period.split("-")
        year = int(year)
        month = int(month)
        # Last day of month (simplified)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
        return (datetime(year, month, 1), end_date)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")


from datetime import timedelta


def create_time_batches(
    timeline: UnifiedTimeline,
    granularity: Literal["year", "half", "quarter", "month"] = "half",
    min_messages: int = 10,
) -> list[TimeBatch]:
    """
    Create time batches from a unified timeline.

    Args:
        timeline: The unified timeline to batch
        granularity: How to group periods
        min_messages: Minimum messages required for a batch (otherwise merged)

    Returns:
        List of TimeBatch objects sorted by period
    """
    # Group messages by period
    period_messages: dict[str, list[UnifiedMessage]] = defaultdict(list)

    for msg in timeline.messages:
        period = get_period_key(msg.timestamp, granularity)
        period_messages[period].append(msg)

    # Create batches
    batches = []
    for period in sorted(period_messages.keys()):
        messages = period_messages[period]
        start_date, end_date = get_period_dates(period, granularity)

        batches.append(
            TimeBatch(
                period=period,
                start_date=start_date,
                end_date=end_date,
                messages=messages,
            )
        )

    # Optionally merge small batches (skip for now, can add later)
    # if min_messages > 0:
    #     batches = merge_small_batches(batches, min_messages)

    return batches


def get_period_weight(period: str, current_year: int = None) -> float:
    """
    Get weight for a period (more recent = higher weight).

    Args:
        period: Period string like "2024-H1"
        current_year: Current year for reference (default: now)

    Returns:
        Weight between 0.4 and 1.0
    """
    if current_year is None:
        current_year = datetime.now().year

    # Extract year from period
    try:
        year = int(period.split("-")[0])
    except (ValueError, IndexError):
        return 0.5  # Default weight

    years_ago = current_year - year

    if years_ago <= 0:
        return 1.0  # Current year
    elif years_ago == 1:
        return 0.8
    elif years_ago == 2:
        return 0.6
    else:
        return 0.4  # Older


def format_batches_summary(batches: list[TimeBatch]) -> str:
    """Format a summary of all batches."""
    lines = ["# Time Batches Summary\n"]

    total_messages = sum(b.count for b in batches)
    lines.append(f"Total batches: {len(batches)}")
    lines.append(f"Total messages: {total_messages:,}")
    lines.append("")
    lines.append("| Period | Messages | Hebrew | English | Avg Length |")
    lines.append("|--------|----------|--------|---------|------------|")

    for batch in batches:
        stats = batch.get_stats()
        lines.append(
            f"| {stats['period']} | {stats['total']:,} | "
            f"{stats['hebrew']:,} | {stats['english']:,} | "
            f"{stats['avg_length']:.0f} |"
        )

    return "\n".join(lines)
