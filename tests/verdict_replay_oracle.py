"""Evidence-only oracle, independent of production verdict code."""

from datetime import datetime

ORACLE_LIMITS_H = {"backup_local": (30, 54), "backup_offsite": (30, 78), "transcripts": (30, 54)}


def _oracle_state(events: list[dict], check: str, at: datetime) -> str:
    mine = [e for e in events if e["check"] == check]
    lo = [e for e in mine if datetime.fromisoformat(e["known_by"]) <= at]
    hi = [e for e in mine if datetime.fromisoformat(e["known_from"]) <= at]

    def state(known: list[dict]) -> str:
        if check in ("drive_auth", "integrity"):
            if not known:
                return "unknown"
            last = max(known, key=lambda e: e["known_from"])["value"]
            return {"ok": "green", "consent_needed": "red", "refresh_failed": "amber"}.get(last, "red")
        if not known:
            return "unknown"  # no history for this period, not evidence that no copy existed
        if check == "backup_local":
            latest = max(known, key=lambda e: e["known_from"])
            if latest["value_at"] is None:
                return "red"
            newest = datetime.fromisoformat(latest["value_at"])
        else:
            newest = max(datetime.fromisoformat(e["value_at"]) for e in known)
        age = (at - newest).total_seconds()
        amber, red = (hours * 3600 for hours in ORACLE_LIMITS_H[check])
        return "red" if age > red else "amber" if age > amber else "green"

    low, high = state(lo), state(hi)
    return low if low == high else "unknown"
