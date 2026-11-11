"""Outcome verdict behind the BrainBar dot: is memory at risk, or has a pipe stopped, right now?

Pure policy. Inputs are plain facts gathered for one health tick plus the previous tick's
persisted verdict state; the output is the dot, one entry per check, the notes, and the state
to persist. Nothing here reads files, sockets or the clock.

Ratified by Etan (2026-10-26): the dot lights "only when something horribly goes wrong"; a
skipped job while a verified backup exists is a note, never a dot. So:

- Job failures are history, not state. A job alert is a note; it matters only through the
  outcome it damages (an aging backup, a stopped pipe).
- "Couldn't look" is unknown, never red. Unknown keeps the last known state and turns amber
  after ``unknown_amber_ticks`` consecutive ticks. A known red holds until positive evidence.
- Readable receipts are authoritative for backup clocks, including absent or revoked copies.
  Only an unobserved receipt source falls back to the last verified timestamp; its age keeps
  running while the source is unreadable.
- Recovery needs real evidence. Backups go green on the first verified receipt; ingest, stores
  and search need ``recovery_good_ticks`` consecutive observed good ticks. A tick with
  no new evidence (an unobserved fact, or an age that only looks fresh because the monitor was not
  running) resets recovery and never improves a known amber or red.
- Latches: a failed store canary stays failed until a store canary succeeds; confirmed Drive
  access loss stays red until Drive access is observed ``ok``.
- Wake grace: after a monitoring gap, a backup that just crossed red gets one catch-up window,
  whose deadline is persisted and never renewed by a later gap. Heartbeat and stall clocks
  subtract confirmed suspension overlapping their own timestamp windows. Clock pairs establish
  total suspension within a tick interval, so only guaranteed overlap is credited. Without
  same-boot clock evidence, pipes retain the older bounded monitoring-gap grace.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime, timedelta
from typing import Any

GREEN, AMBER, RED, UNKNOWN = "green", "amber", "red", "unknown"
OUTCOME_CODE_PREFIX = "outcome_"  # Never "job_alert_": BrainBar 1.5.83 reconciles those against job-alerts.json.
HOURS = 3600.0
ENV_PREFIX = "BRAINLAYER_VERDICT_"


class _Unobserved:
    """The fact could not be observed this tick (distinct from None = observed, nothing on record)."""

    def __repr__(self) -> str:
        return "UNOBSERVED"


UNOBSERVED: Any = _Unobserved()


@dataclass(frozen=True)
class Thresholds:
    """Plan §3 table. Each field is overridable as ``BRAINLAYER_VERDICT_<FIELD>`` (seconds or count)."""

    backup_local_amber_s: float = 30 * HOURS
    backup_local_red_s: float = 54 * HOURS
    backup_offsite_amber_s: float = 30 * HOURS
    backup_offsite_red_s: float = 78 * HOURS
    transcripts_amber_s: float = 30 * HOURS
    transcripts_red_s: float = 54 * HOURS
    ingest_amber_s: float = 10 * 60
    ingest_red_s: float = 30 * 60
    stores_amber_s: float = 15 * 60
    stores_red_s: float = 45 * 60
    embedding_amber_s: float = 1 * HOURS
    embedding_red_s: float = 6 * HOURS
    canary_red_failures: int = 3
    unknown_amber_ticks: int = 3
    recovery_good_ticks: int = 2
    sleep_gap_s: float = 15 * 60
    wake_grace_s: float = 2 * HOURS
    disk_floor_multiplier: float = 2.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Thresholds:
        env = os.environ if env is None else env
        values: dict[str, Any] = {}
        for item in fields(cls):
            raw = env.get(ENV_PREFIX + item.name.upper(), "").strip()
            if raw:
                value = float(raw)
                if math.isnan(value) or value <= 0:
                    raise ValueError(f"{ENV_PREFIX}{item.name.upper()} must be positive")
                values[item.name] = int(value) if isinstance(item.default, int) else value
        return cls(**values)


@dataclass(frozen=True)
class Note:
    code: str
    message: str


@dataclass(frozen=True)
class Facts:
    """One tick's observations. Every field defaults to UNOBSERVED.

    Timestamp facts use None for "observed, nothing on record" (never verified, queue empty,
    not stalled). ``check_notes`` attaches context (e.g. "hotlane not running") to a check.
    """

    now: datetime
    awake_clock_s: float | None = None
    boot_session: str | None = None
    backup_local_verified_at: Any = UNOBSERVED
    backup_snapshot_local_removed: bool = False
    backup_offsite_verified_at: Any = UNOBSERVED
    transcripts_verified_at: Any = UNOBSERVED
    drive_auth: Any = UNOBSERVED  # "ok" | "refresh_failed" | "consent_needed"
    integrity: Any = UNOBSERVED  # "ok" | "corrupt" (confirmed results only)
    watcher_updated_at: Any = UNOBSERVED
    ingest_paused: bool = False
    oldest_pending_store_at: Any = UNOBSERVED
    store_canary_ok: Any = UNOBSERVED  # True only for a persisted, retrieved round trip
    search_canary_ok: Any = UNOBSERVED
    embedding_stalled_since: Any = UNOBSERVED
    embedding_activity: str | None = None
    embedding_progress_recent: bool = False
    disk_free_bytes: Any = UNOBSERVED
    disk_peak_bytes: Any = UNOBSERVED
    disk_floor_bytes: Any = UNOBSERVED
    blind_codes: tuple[str, ...] = ()
    notes: tuple[Note, ...] = ()
    check_notes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Check:
    name: str
    state: str
    age_s: float | None
    threshold_s: float | None
    since: str | None
    note: str  # bare sentence; consumers add CHECK_LABELS[name]

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("name", "state", "age_s", "threshold_s", "since", "note")}


@dataclass(frozen=True)
class Verdict:
    dot: str
    checks: tuple[Check, ...]
    notes: tuple[Note, ...]
    state: dict[str, Any]

    def check(self, name: str) -> Check:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)


CHECK_ORDER = (
    "backup_local",
    "backup_offsite",
    "transcripts",
    "drive_auth",
    "integrity",
    "ingest",
    "embedding",
    "stores",
    "search",
    "disk",
    "monitor",
)
CHECK_LABELS = {
    "backup_local": "Database snapshot (verified)",
    "backup_offsite": "Offsite (Drive) database backup",
    "transcripts": "Transcript backup",
    "drive_auth": "Drive authorization",
    "integrity": "Database integrity",
    "ingest": "Transcript ingest",
    "embedding": "Embedding",
    "stores": "Memory stores",
    "search": "Memory search",
    "disk": "Disk space for backups",
    "monitor": "Health monitor",
}
_HYSTERESIS = frozenset({"ingest", "stores", "search"})
_WAKE_GRACE = frozenset({"backup_local", "backup_offsite", "transcripts"})


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _aware(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return None


@dataclass
class _Raw:
    state: str
    note: str = ""
    age_s: float | None = None
    threshold_s: float | None = None
    clock: datetime | None = None  # the persisted positive timestamp, when the check has one
    amber_s: float | None = None
    keep: dict[str, Any] = field(default_factory=dict)  # extra persisted fields (counters, latches)
    evidence: bool = True  # False: nothing new was observed, so this tick cannot show recovery
    graced: bool = False  # the state is milder than the real age only because the monitor was not running


def _age_state(age: float, amber: float, red: float) -> str:
    return RED if age > red else AMBER if age > amber else GREEN


def _hours(seconds: float) -> str:
    return f"{seconds / HOURS:.1f} h"


def _verified_age(
    observed: Any,
    prior: Mapping[str, Any],
    now: datetime,
    amber: float,
    red: float,
) -> _Raw:
    """Readable receipts own the clock; persisted state is only an unreadable-source fallback."""
    clock = _parse(prior.get("clock")) if observed is UNOBSERVED else _aware(observed)
    if clock is None:
        if observed is UNOBSERVED:
            return _Raw(UNKNOWN, "receipts could not be read", threshold_s=red, evidence=False)
        return _Raw(RED, "no verified copy on record", threshold_s=red)
    age = max(0.0, (now - clock).total_seconds())
    state = _age_state(age, amber, red)
    note = f"newest verified copy {_hours(age)} old"
    if observed is UNOBSERVED:
        note += " (receipts unreadable this check; age runs from the last verified copy)"
    if state != GREEN:
        note += f" (amber after {_hours(amber)}, red after {_hours(red)})"
    return _Raw(state, note, age, red, clock, amber, evidence=observed is not UNOBSERVED)


def _stall_age(
    observed: Any,
    prior: Mapping[str, Any],
    now: datetime,
    resumed_at: datetime | None,
    amber: float,
    red: float,
    what: str,
    wake_grace_s: float,
) -> _Raw:
    """Age of an ongoing stall (oldest pending store, embed backlog not shrinking).

    None = observed, nothing pending. Unobserved keeps aging a known stall; an unknown with
    no known stall stays unknown. Time the monitor was not running is graced, but a graced age
    is never evidence of progress: only an observed clearance (None) or a newer stall start is.
    """
    if observed is UNOBSERVED:
        if prior.get("clock") is None:
            return _Raw(UNKNOWN, f"{what}: not observed this check", threshold_s=red, evidence=False)
        start = _parse(prior["clock"])
    else:
        start = _aware(observed)
    if start is None:
        return _Raw(GREEN, "", 0.0, red, None, evidence=observed is not UNOBSERVED)
    return _graced_age(start, prior, now, resumed_at, amber, red, what, observed is not UNOBSERVED, wake_grace_s)


def _graced_age(
    start: datetime,
    prior: Mapping[str, Any],
    now: datetime,
    resumed_at: datetime | None,
    amber: float,
    red: float,
    what: str,
    observed: bool,
    wake_grace_s: float,
) -> _Raw:
    """One persisted wake window per clock value; grace never proves recovery."""
    grace_at = deadline = None
    if prior.get("grace_clock") == _iso(start):
        grace_at = _parse(prior.get("grace_started_at"))
        deadline = _parse(prior.get("grace_until"))
    if deadline is None and resumed_at is not None and start < resumed_at:
        grace_at = resumed_at
        deadline = grace_at + timedelta(seconds=wake_grace_s)
    keep = {}
    if deadline is not None:
        # Keep expired windows too: another gap in this episode cannot grant a new one.
        keep = {"grace_clock": _iso(start), "grace_started_at": _iso(grace_at), "grace_until": _iso(deadline)}
    sleep_offset = 0.0
    for window in prior.get("suspensions", ()):
        left, right = _parse(window["from"]), _parse(window["to"])
        # The pair locates total suspension inside [left, right], not its exact edges.
        # Credit only overlap guaranteed to fall within this heartbeat/stall's own age.
        outside = max(0.0, (start - left).total_seconds()) + max(0.0, (right - now).total_seconds())
        sleep_offset += max(0.0, window["seconds"] - outside)
    true_age = max(0.0, (now - start).total_seconds() - sleep_offset)
    effective = max(start, grace_at) if grace_at is not None and deadline is not None and now <= deadline else start
    age = min(true_age, max(0.0, (now - effective).total_seconds()))
    state = _age_state(age, amber, red)
    graced = _RANK[state] < _RANK[_age_state(true_age, amber, red)]
    note = f"{what} {_hours(true_age) if true_age >= HOURS else f'{true_age / 60:.0f} min'}"
    if graced:
        note += " (the monitor was not running for part of it)"
    return _Raw(
        state,
        note if state != GREEN or graced else "",
        age,
        red,
        start,
        amber,
        keep=keep,
        evidence=observed and not graced,
        graced=graced,
    )


def _counter(observed: Any, prior: Mapping[str, Any], red_failures: int, latch: bool = False) -> _Raw:
    """Consecutive failures of a canary. With ``latch``, an unrun canary keeps its last failure."""
    fails = int(prior.get("fails") or 0)
    if observed is UNOBSERVED:
        if latch and fails:
            state = RED if fails >= red_failures else AMBER
            note = f"last {fails} canary run(s) failed; not run since (holds until a run succeeds)"
            return _Raw(state, note, keep={"fails": fails}, evidence=False)
        return _Raw(UNKNOWN, "canary not run this check", keep={"fails": fails}, evidence=False)
    fails = 0 if observed else fails + 1
    if fails == 0:
        return _Raw(GREEN, keep={"fails": 0})
    state = RED if fails >= red_failures else AMBER
    return _Raw(state, f"canary failed {fails}x in a row (red at {red_failures})", keep={"fails": fails})


def _drive_auth(drive: Any, prior: Mapping[str, Any]) -> _Raw:
    """Confirmed access loss latches: only an observed ``ok`` (verified Drive access) clears it."""
    lost = bool(prior.get("lost"))
    if drive is UNOBSERVED:
        return _Raw(UNKNOWN, "authorization not checked", keep={"lost": lost}, evidence=False)
    if drive == "ok":
        return _Raw(GREEN, keep={"lost": False})
    if drive == "consent_needed" or lost:
        note = "access lost; consent needed (Reconnect in BrainBar)"
        if drive != "consent_needed":
            note += f"; latest attempt: {drive}, which does not restore access"
        return _Raw(RED, note, keep={"lost": True})
    return _Raw(AMBER, "token refresh failed; retrying", keep={"lost": False})


def _raw_checks(
    facts: Facts,
    prior: Mapping[str, Mapping[str, Any]],
    t: Thresholds,
    resumed_at: datetime | None,
    suspensions: list[dict[str, Any]],
    confirmed_clock: bool,
) -> dict[str, _Raw]:
    now = facts.now

    def p(name):
        saved = dict(prior.get(name, {}))
        saved["suspensions"] = suspensions
        if confirmed_clock:
            for key in ("grace_clock", "grace_started_at", "grace_until"):
                saved.pop(key, None)
        return saved

    raw = {
        "backup_local": _verified_age(
            facts.backup_local_verified_at,
            p("backup_local"),
            now,
            t.backup_local_amber_s,
            t.backup_local_red_s,
        ),
        "backup_offsite": _verified_age(
            facts.backup_offsite_verified_at,
            p("backup_offsite"),
            now,
            t.backup_offsite_amber_s,
            t.backup_offsite_red_s,
        ),
        "transcripts": _verified_age(
            facts.transcripts_verified_at,
            p("transcripts"),
            now,
            t.transcripts_amber_s,
            t.transcripts_red_s,
        ),
    }
    raw["backup_local"].note = raw["backup_local"].note.replace("verified copy", "verified snapshot")
    if facts.backup_snapshot_local_removed and _aware(facts.backup_local_verified_at) is not None:
        raw["backup_local"].note += "; local copy unavailable after Drive verification"
    raw["drive_auth"] = _drive_auth(facts.drive_auth, p("drive_auth"))
    integrity = facts.integrity
    raw["integrity"] = (
        _Raw(UNKNOWN, "no confirmed integrity result on record", evidence=False)
        if integrity is UNOBSERVED
        else _Raw(RED, "corruption confirmed in the newest snapshot")
        if integrity == "corrupt"
        else _Raw(GREEN)
    )

    watcher = _aware(facts.watcher_updated_at)
    persisted = _parse(p("ingest").get("clock"))
    clock = max((x for x in (watcher, persisted) if x is not None), default=None)
    if clock is None:
        raw["ingest"] = _Raw(UNKNOWN, "watcher heartbeat not observed", threshold_s=t.ingest_red_s, evidence=False)
    else:
        r = _graced_age(
            clock,
            p("ingest"),
            now,
            resumed_at,
            t.ingest_amber_s,
            t.ingest_red_s,
            "heartbeat age is",
            True,
            t.wake_grace_s,
        )
        r.evidence = r.evidence and watcher is not None  # an observed heartbeat, fresh by its real age
        if watcher is None:
            r.note = f"watcher heartbeat not observed; {r.note}"
        if facts.ingest_paused and r.state == RED:
            r.state, r.note = AMBER, r.note + " (watcher paused by maintenance)"
        raw["ingest"] = r

    raw["embedding"] = _stall_age(
        facts.embedding_stalled_since,
        p("embedding"),
        now,
        resumed_at,
        t.embedding_amber_s,
        t.embedding_red_s,
        "backlog has not shrunk for",
        t.wake_grace_s,
    )
    if facts.embedding_activity:
        if facts.embedding_progress_recent:
            raw["embedding"] = _Raw(AMBER, facts.embedding_activity)
        else:
            raw["embedding"].note += f"; {facts.embedding_activity}"
    pending = _stall_age(
        facts.oldest_pending_store_at,
        p("stores"),
        now,
        resumed_at,
        t.stores_amber_s,
        t.stores_red_s,
        "oldest pending store has waited",
        t.wake_grace_s,
    )
    prior_stores = p("stores")
    canary = _counter(facts.store_canary_ok, prior_stores, t.canary_red_failures, latch=True)
    parts = [pending] + ([canary] if canary.state != UNKNOWN else [])
    known = [r for r in parts if r.state != UNKNOWN] or parts
    worst = max(known, key=lambda r: _RANK[r.state])
    # A canary failure is owed a successful round trip: an empty queue never repays it.
    owed = bool(prior_stores.get("canary_owed")) or canary.keep["fails"] > 0
    canary_ok = facts.store_canary_ok is True
    evidence = not pending.graced and (canary_ok if owed else (pending.evidence or canary_ok))
    raw["stores"] = _Raw(
        worst.state,
        "; ".join(r.note for r in known if r.note),
        pending.age_s,
        pending.threshold_s,
        pending.clock,  # the stall start persists even while the canary decides the state
        pending.amber_s if worst is pending else None,
        {**pending.keep, **canary.keep, "canary_owed": owed},
        evidence=evidence,
        graced=pending.graced,
    )
    raw["search"] = _counter(facts.search_canary_ok, p("search"), t.canary_red_failures)

    disk = (facts.disk_free_bytes, facts.disk_peak_bytes, facts.disk_floor_bytes)
    if any(value is UNOBSERVED or value is None for value in disk):
        raw["disk"] = _Raw(UNKNOWN, "free space versus the next backup's need not measured", evidence=False)
    else:
        free, peak, floor = (int(v) for v in disk)
        gib = 1024**3
        detail = f"{free / gib:.1f} GiB free; next backup needs {peak / gib:.1f} GiB"
        if free < peak:
            raw["disk"] = _Raw(RED, f"{detail}: the next backup cannot run")
        elif free < peak + t.disk_floor_multiplier * floor:
            raw["disk"] = _Raw(AMBER, f"{detail} plus a {t.disk_floor_multiplier:g}x {floor / gib:.1f} GiB floor")
        else:
            raw["disk"] = _Raw(GREEN)
    raw["monitor"] = (
        _Raw(UNKNOWN, "could not look: " + ", ".join(sorted(set(facts.blind_codes))), evidence=False)
        if facts.blind_codes
        else _Raw(GREEN)
    )
    for name, extra in facts.check_notes.items():
        if name in raw and extra:
            raw[name].note = f"{raw[name].note}; {extra}" if raw[name].note else extra
    return raw


_RANK = {GREEN: 0, UNKNOWN: 1, AMBER: 2, RED: 3}


def evaluate(facts: Facts, prior: Mapping[str, Any] | None = None, thresholds: Thresholds | None = None) -> Verdict:
    """One tick: facts + last tick's persisted state -> dot, checks, notes and the state to persist."""
    t = thresholds or Thresholds()
    now = _aware(facts.now)
    if now is None:
        raise ValueError("facts.now must be a datetime")
    facts = Facts(**{**{f.name: getattr(facts, f.name) for f in fields(Facts)}, "now": now})
    prior = prior if isinstance(prior, Mapping) else {}
    prior_checks = prior.get("checks") if isinstance(prior.get("checks"), Mapping) else {}
    ticked = _parse(prior.get("ticked_at"))
    resumed_at = _parse(prior.get("resumed_at"))
    confirmed_clock = False
    suspensions = [
        w
        for w in prior.get("suspensions", ())
        if _parse(w.get("to")) is not None and (now - _parse(w["to"])).total_seconds() < 86400
    ]
    previous_awake = prior.get("awake_clock_s")
    if (
        ticked is not None
        and facts.awake_clock_s is not None
        and previous_awake is not None
        and facts.boot_session is not None
        and facts.boot_session == prior.get("boot_session")
        and facts.awake_clock_s >= previous_awake
    ):
        confirmed_clock = True
        suspended = (now - ticked).total_seconds() - (facts.awake_clock_s - previous_awake)
        if suspended > 5:
            suspensions.append({"from": _iso(ticked), "to": _iso(now), "seconds": suspended})
            resumed_at = now
    elif ticked is not None and (now - ticked).total_seconds() > t.sleep_gap_s:
        resumed_at = now
    if resumed_at is not None and (now - resumed_at).total_seconds() > t.wake_grace_s:
        resumed_at = None
    raw = _raw_checks(facts, prior_checks, t, None if confirmed_clock else resumed_at, suspensions, confirmed_clock)

    checks: list[Check] = []
    state_checks: dict[str, Any] = {}
    for name in CHECK_ORDER:
        r = raw[name]
        before = prior_checks.get(name) if isinstance(prior_checks.get(name), Mapping) else {}
        prior_state = before.get("state") if before.get("state") in _RANK else None
        unknown_ticks = int(before.get("unknown_ticks") or 0)
        good_ticks = int(before.get("good_ticks") or 0)  # consecutive observed good ticks
        deadline = _parse(before.get("catch_up_until"))
        state, note = r.state, r.note
        if r.state != GREEN or not r.evidence or r.graced:
            good_ticks = 0
        if state == UNKNOWN:
            unknown_ticks += 1
            base = prior_state if prior_state not in (None, UNKNOWN) else UNKNOWN
            if base == RED:
                state, note = RED, f"{note}; last known red holds until positive evidence"
            elif unknown_ticks >= t.unknown_amber_ticks:
                state, note = AMBER, f"{note} ({unknown_ticks} checks in a row)"
            else:
                state = base
        else:
            unknown_ticks = 0
            if not r.evidence and prior_state in (AMBER, RED) and _RANK[state] < _RANK[prior_state]:
                # Nothing new was observed (or the age only looks fresh because the monitor was
                # not running): a known amber or red holds, and this is not a good tick.
                state, note = prior_state, f"{note}; {prior_state} holds until positive evidence".lstrip("; ")
            elif name in _HYSTERESIS and prior_state == RED and state == AMBER:
                state, note = RED, f"{note}; red holds until consecutive observed good checks"
            elif name in _HYSTERESIS and state == GREEN and prior_state in (AMBER, RED):
                good_ticks += 1
                if good_ticks < t.recovery_good_ticks:
                    state, note = prior_state, f"recovering ({good_ticks}/{t.recovery_good_ticks} good checks)"
                else:
                    good_ticks = 0
            elif state != GREEN or prior_state not in (AMBER, RED):
                good_ticks = 0
            if name in _WAKE_GRACE:
                same_copy = r.clock is not None and _iso(r.clock) == before.get("clock")
                if not same_copy or r.state in (GREEN, AMBER):
                    deadline = None  # loss/replacement of the copy ends its catch-up episode
                if state == RED and prior_state != RED and same_copy:
                    expired_at = r.clock + timedelta(seconds=r.threshold_s)
                    if (
                        deadline is None
                        and ticked is not None
                        and resumed_at is not None
                        and ticked <= expired_at < resumed_at
                    ):
                        # Only this verified clock crossing red during a gap earns grace.
                        # A revoked copy exposing an older clock never does; later gaps cannot renew it.
                        deadline = resumed_at + timedelta(seconds=t.wake_grace_s)
                    if deadline is not None and now <= deadline:
                        state = AMBER
                        note = f"{note}; Mac was not running, catch-up run has until {_iso(deadline)}"
        if r.clock is not None and r.amber_s is not None and r.threshold_s is not None and state == r.state:
            since_at = r.clock + timedelta(
                seconds=r.threshold_s if state == RED else r.amber_s if state == AMBER else 0
            )
            since = _iso(since_at)
        else:
            since = before.get("since") if state == prior_state and before.get("since") else _iso(now)
        checks.append(Check(name, state, r.age_s, r.threshold_s, since, note))
        state_checks[name] = {
            "state": state,
            "since": since,
            "unknown_ticks": unknown_ticks,
            "good_ticks": good_ticks,
            "clock": _iso(r.clock) if r.clock is not None else (before.get("clock") if r.state == UNKNOWN else None),
            **r.keep,
        }
        if deadline is not None:
            state_checks[name]["catch_up_until"] = _iso(deadline)
        if name == "stores" and state == GREEN:
            state_checks[name]["canary_owed"] = False  # recovery completed on a successful round trip
    worst = max((check.state for check in checks), key=lambda s: _RANK[s] if s != UNKNOWN else 0)
    dot = worst if worst in (RED, AMBER) else GREEN
    return Verdict(
        dot=dot,
        checks=tuple(checks),
        notes=tuple(facts.notes),
        state={
            "ticked_at": _iso(now),
            "suspensions": suspensions,
            "awake_clock_s": facts.awake_clock_s,
            "boot_session": facts.boot_session,
            "resumed_at": _iso(resumed_at),
            "checks": state_checks,
        },
    )


def badge_alerts(verdict: Verdict) -> list[dict[str, str]]:
    """The schema-1 ``alerts.active`` entries for BrainBar 1.5.83.

    Only a red check is ``critical``; every amber/unknown check and every note is ``warning``.
    That keeps the old reader's invariant ``badge_on == any(severity != "warning")`` with
    ``badge_on = dot == red``.
    """
    active = [
        {
            "code": OUTCOME_CODE_PREFIX + check.name,
            "severity": "critical" if check.state == RED else "warning",
            "message": f"{CHECK_LABELS[check.name]}: {check.note or check.state}",
        }
        for check in verdict.checks
        if check.state != GREEN
    ]
    consent = next((note for note in verdict.notes if note.code == "job_alert_drive-consent"), None)
    if consent:
        for item in active:
            if item["code"] == OUTCOME_CODE_PREFIX + "drive_auth":
                item["message"] = consent.message
    active.extend(
        {"code": note.code, "severity": "warning", "message": note.message}
        for note in verdict.notes
        if not (note is consent and any(item["code"] == OUTCOME_CODE_PREFIX + "drive_auth" for item in active))
    )
    return active


def notes_from_job_alerts(alerts: Mapping[str, str]) -> tuple[Note, ...]:
    """Job alerts are history: one warning note per job, code unchanged so Show log still finds it."""
    return tuple(Note(f"job_alert_{key}", reason) for key, reason in sorted(alerts.items()))


# --- Receipt readers (pure: they take already-parsed receipt dicts) -------------------------

_CONSENT_MARKERS = ("expired", "reconnect", "consent", "invalid_grant", "lacks refresh authorization")
_CORRUPTION_MARKERS = ("malformed", "corrupt", "not a database")


def _receipt_time(receipt: Mapping[str, Any]) -> datetime | None:
    return _parse(receipt.get("attempted_at"))


def drive_event(receipt: Mapping[str, Any]) -> str | None:
    """Current upload/auth outcomes only; historical reuse alone says nothing about Drive today."""
    text = " ".join(str(receipt.get(k) or "") for k in ("error_type", "error_code", "error")).lower()
    if "drivecredentialerror" in text or "drive access expired" in text or "drive token" in text:
        return "consent_needed" if any(marker in text for marker in _CONSENT_MARKERS) else "refresh_failed"
    if receipt.get("status") == "reused":
        return "ok" if receipt.get("drive_auth_probe") == "ok" else None
    if receipt.get("uploaded") is True and receipt.get("verified") is True:
        return "ok"
    return None


def integrity_result(receipt: Mapping[str, Any]) -> str | None:
    """Confirmed results only: "ok", "corrupt", or None when the check did not run or could not finish."""
    value = receipt.get("local_integrity")
    if not isinstance(value, str) or not value or value == "skipped":
        return None
    if value == "ok":
        return "ok"
    name, _, detail = value.partition(":")
    if name.strip().endswith("Error") and " " not in name.strip():
        return "corrupt" if any(marker in detail.lower() for marker in _CORRUPTION_MARKERS) else None
    return "corrupt"  # PRAGMA integrity_check/quick_check returned problem rows.


@dataclass(frozen=True)
class BackupFacts:
    local_verified_at: datetime | None
    offsite_verified_at: datetime | None
    drive: tuple[datetime, str] | None  # (receipt time, drive_event) of the newest Drive outcome
    integrity: Any
    snapshot_local_removed: bool = False

    def facts_kwargs(self) -> dict[str, Any]:
        """Authoritative backup observations for the policy, after a successful receipt read."""
        return {
            "backup_local_verified_at": self.local_verified_at,
            "backup_snapshot_local_removed": self.snapshot_local_removed,
            "backup_offsite_verified_at": self.offsite_verified_at,
            "integrity": self.integrity,
        }


def backup_facts(receipts: Iterable[Mapping[str, Any]], *, as_of: datetime | None = None) -> BackupFacts:
    """Newest verified snapshot (local or Drive) and Drive copies from ``backup-daily.log`` receipts, any trigger.

    A receipt counts once its invocation has started (``attempted_at <= as_of``). A Drive-verified
    upload also proves the local snapshot it was made from. A later receipt for the same snapshot
    that reports a verification error or a Drive md5 mismatch revokes both copies. Superseding
    invalidates the local copy while preserving a verified Drive copy.
    Local removal or invalidation drops the clock only without a verified Drive copy of that snapshot.
    Coverage is restored separately: a local-only success cannot restore a revoked Drive copy.
    Exact creation times from history survive filename-only receipts for the same artifact.
    """
    from .backup_daily import _snapshot_created_time

    local: dict[str, datetime] = {}
    offsite: dict[str, datetime] = {}
    created_by_snapshot: dict[str, datetime] = {}
    drive: tuple[datetime, str] | None = None
    integrity: tuple[datetime, str] | None = None
    for receipt in receipts:
        if receipt.get("backup_log_provenance") != "real":
            continue
        at = _receipt_time(receipt)
        if at is None or (as_of is not None and at > as_of):
            continue
        key = receipt.get("snapshot") if isinstance(receipt.get("snapshot"), str) else None
        created = _snapshot_created_time(receipt)
        if key:
            if receipt.get("snapshot_created_at") is None:
                created = created_by_snapshot.get(key, created)
            if created is not None:
                created_by_snapshot[key] = created
        is_revoked = bool(receipt.get("verification_error")) or receipt.get("drive_md5_match") is False
        if key and is_revoked:
            local.pop(key, None)
            offsite.pop(key, None)
        elif key and created is not None:
            if receipt.get("uploaded") is True and receipt.get("verified") is True:
                offsite[key] = local[key] = created
            elif receipt.get("local_verified") is True and receipt.get("local_integrity") == "ok":
                local[key] = created
        if key and (
            receipt.get("local_removed") is True
            or receipt.get("local_verified") is False
            or receipt.get("upload_state") in ("abandoned", "superseded")
        ):
            local.pop(key, None)
        if (event := drive_event(receipt)) is not None and (drive is None or at >= drive[0]):
            drive = (at, event)
        if (result := integrity_result(receipt)) is not None and (integrity is None or at >= integrity[0]):
            integrity = (at, result)
    snapshots = {**offsite, **local}
    newest = max(snapshots.values(), default=None)
    return BackupFacts(
        local_verified_at=newest,
        snapshot_local_removed=newest is not None and not any(created == newest for created in local.values()),
        offsite_verified_at=max(offsite.values(), default=None),
        drive=drive,
        integrity=integrity[1] if integrity else UNOBSERVED,
    )


def transcript_facts(
    receipts: Iterable[Mapping[str, Any]], *, as_of: datetime | None = None
) -> tuple[datetime | None, tuple[datetime, str] | None]:
    """Newest verified transcript archive attempt and the newest Drive event from ``jsonl-backup.log``."""
    from .health_check import _jsonl_backup_attempt_time

    newest: datetime | None = None
    drive: tuple[datetime, str] | None = None
    for receipt in receipts:
        at = _jsonl_backup_attempt_time(dict(receipt))
        if at is None or (as_of is not None and at > as_of):
            continue
        status, verified = receipt.get("status"), receipt.get("verified") is True
        if verified and (status in ("no-op", "created") or (status == "uploaded" and receipt.get("uploaded") is True)):
            newest = at if newest is None else max(newest, at)
        if (event := drive_event(receipt)) is not None and (drive is None or at >= drive[0]):
            drive = (at, event)
    return newest, drive
