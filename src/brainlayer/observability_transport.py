"""Fail-closed consumption of BrainBar's aggregate, bounded transport evidence."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .paths import get_db_path

if TYPE_CHECKING:
    from .observability_surface import InputRecorder

COUNTS = (
    "connections_accepted",
    "connections_closed",
    "disconnects_with_in_flight",
    "requests_received",
    "requests_answered",
    "requests_dropped",
    "accept_errors",
    "read_errors",
    "write_errors",
)
IDENTITY_REASON = "no stable client identity on the socket (fd reuse; proxy pid)"
WINDOW_SECONDS = 900
FUTURE_SKEW_SECONDS = 2  # Bound wall-clock skew/whole-second producer timestamps; never waive staleness.
MAXIMUM_BASELINE_AGE_SECONDS = 300  # Five grace periods, including KeepAlive relaunch.
DISCARD_REASONS = (
    "baseline from a previous boot; reconnect deficit not measured",
    "baseline expired; reconnect deficit not measured",
)


def _read_receipt(path: Path) -> tuple[bytes, os.stat_result]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:

        def validate() -> os.stat_result:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
            ):
                raise ValueError(
                    "transport receipt must be a single-link regular file owned by this uid with mode 0600"
                )
            return info

        info = validate()
        if info.st_size > 65_536:
            raise ValueError("receipt exceeds 64 KiB")
        data = bytearray()
        while len(data) <= 65_536:
            chunk = os.read(fd, 65_537 - len(data))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > 65_536:
            raise ValueError("receipt exceeds 64 KiB")
        return bytes(data), validate()
    finally:
        os.close(fd)


class TransportSection(dict[str, Any]):
    """Internal validated episode accompanies the section, outside its JSON fields."""

    overflow = False

    def __init__(
        self,
        value: dict,
        *,
        incident: dict | None = None,
        incident_unknown: bool = False,
        reconnect: dict | None = None,
    ):
        super().__init__(value)
        self.incident = incident
        self.incident_unknown = incident_unknown
        self.reconnect = reconnect


def _time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("producer timestamp has no timezone")
    return result.astimezone(UTC)


def _frame(data: dict, *, db: Path, now: datetime, historical: bool = False) -> dict:
    if data.get("state") != "measured" or data.get("reason") != "":
        raise ValueError("BrainBar transport evidence unavailable: " + str(data.get("reason") or "unknown state"))
    if Path(data["db_path"]).expanduser().resolve() != db:
        raise ValueError("receipt belongs to a different database")
    generated, started = _time(data["generated_at"]), _time(data["producer_started_at"])
    age = (now - generated).total_seconds()
    if started > generated or age < -FUTURE_SKEW_SECONDS or (not historical and age > 60):
        raise ValueError("transport receipt is stale, future, or has invalid producer timestamps")
    if (
        type(data.get("window_seconds")) is not int
        or data["window_seconds"] != WINDOW_SECONDS
        or type(data["observed_seconds"]) is not int
        or not 0 <= data["observed_seconds"] <= WINDOW_SECONDS
    ):
        raise ValueError("invalid transport observation window")
    if data["observed_seconds"] > (generated - started).total_seconds() + 1:
        raise ValueError("observation window exceeds producer lifetime")
    counts = data["counts"]
    if (
        not isinstance(counts, dict)
        or set(counts) != set(COUNTS)
        or any(type(v) is not int or v < 0 for v in counts.values())
    ):
        raise ValueError("invalid transport counts")
    if not historical and (type(data["in_flight_requests"]) is not int or data["in_flight_requests"] < 0):
        raise ValueError("invalid in-flight count")
    return {k: data[k] for k in ("generated_at", "producer_started_at", "window_seconds", "observed_seconds", "counts")}


def _reconnect_frame(frame: dict, *, parent: dict, db: Path, now: datetime) -> dict:
    generated, started = _time(parent["generated_at"]), _time(parent["producer_started_at"])
    if isinstance(frame, dict) and frame.get("state") == "unmeasurable" and frame.get("reason") in DISCARD_REASONS:
        return {"state": "unmeasurable", "reason": frame["reason"]}
    if not isinstance(frame, dict) or frame.get("state") != "measured" or frame.get("reason") != "":
        raise ValueError(
            "previous socket recovery evidence unreadable: "
            + str(frame.get("reason", "") if isinstance(frame, dict) else "")
        )
    if Path(frame["db_path"]).resolve() != db or not -FUTURE_SKEW_SECONDS <= (now - generated).total_seconds() <= 60:
        raise ValueError("foreign, stale, or future socket recovery evidence")
    severed = _time(frame["severed_at"])
    keys = (
        "minimum_connection_age_seconds",
        "grace_seconds",
        "observed_seconds",
        "severed_persistent",
        "recovered_persistent",
    )
    if any(type(frame.get(k)) is not int or frame[k] < 0 for k in keys):
        raise ValueError("invalid socket recovery counts")
    if (
        frame[keys[0]] != 10
        or frame[keys[1]] != 60
        or not 0 <= frame[keys[2]] <= 60
        or frame[keys[3]] == 0
        or frame[keys[4]] > frame[keys[3]]
        or started > generated
        or severed > generated
        or frame[keys[2]] > (generated - started).total_seconds() + 1
    ):
        raise ValueError("invalid socket recovery observation window")
    from .supervisor_guard import boot_time

    stamp = frame.get("boot_time")
    if not isinstance(stamp, str) or not stamp:
        raise ValueError("socket recovery boot identity unreadable")
    if stamp != boot_time():
        return {"state": "unmeasurable", "reason": DISCARD_REASONS[0]}
    if (now - severed).total_seconds() > MAXIMUM_BASELINE_AGE_SECONDS:
        return {"state": "unmeasurable", "reason": DISCARD_REASONS[1]}
    return {k: frame[k] for k in keys}


def build_transport_section(
    *, env: Mapping[str, str], now: datetime | None = None, record_input: InputRecorder
) -> dict[str, Any]:
    raw_db = Path(env["BRAINLAYER_DB"]).expanduser() if env.get("BRAINLAYER_DB") else get_db_path()
    db = raw_db.resolve()
    path = Path(
        env.get("BRAINLAYER_TRANSPORT_HEALTH_PATH", str(raw_db.with_name("transport-health.json")))
    ).expanduser()
    failures = (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError)
    try:
        raw, metadata = _read_receipt(path)
        now = datetime.now(UTC) if now is None else now  # Capture immediately after the secure FD snapshot.
        data = json.loads(raw)
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported transport receipt")
        if Path(data["db_path"]).expanduser().resolve() != db:
            raise ValueError("receipt belongs to a different database")
    except failures as exc:
        return TransportSection(
            {
                "state": "unmeasurable",
                "reason": f"transport evidence unavailable: {exc}",
                "inputs": [
                    record_input.file_snapshot(
                        path, status="missing" if isinstance(exc, FileNotFoundError) else "malformed"
                    )
                ],
            }
        )
    item = record_input.file_snapshot(
        path, status="read", metadata=metadata, data=raw, observed_at=now, future_skew_seconds=FUTURE_SKEW_SECONDS
    )
    reconnect = None

    def unknown(reason: str, incident: dict | None = None, uncertain: bool = False) -> TransportSection:
        return TransportSection(
            {"state": "unmeasurable", "reason": reason, "inputs": [item]},
            incident=incident,
            incident_unknown=uncertain,
            reconnect=reconnect,
        )

    if item["status"] != "read":
        return unknown("transport input status: " + item["status"])
    if data.get("state") == "unmeasurable" and data.get("reason") == "transport event buffer overflow":
        try:
            generated, started = _time(data["generated_at"]), _time(data["producer_started_at"])
            age = (now - generated).total_seconds()
            if started > generated or not -FUTURE_SKEW_SECONDS <= age <= 60:
                raise ValueError("invalid overflow producer timestamps")
            if type(data.get("window_seconds")) is not int or data["window_seconds"] != WINDOW_SECONDS:
                raise ValueError("invalid overflow observation window")
            # Independent categorical evidence, even when older history is unreadable.
            result = unknown(data["reason"])
            result.overflow = True
            return result
        except failures:
            pass  # Fall through to the ordinary unknown/history validation path.
    incident = None
    if data.get("incident") is not None:
        try:
            incident = _frame(data["incident"], db=db, now=now, historical=True)
            if not _failing(incident["counts"]):
                raise ValueError("historical episode does not cross a failure threshold")
        except failures as exc:
            return unknown(f"historical transport episode unmeasurable: {exc}", uncertain=True)
    try:
        if data.get("reconnect") is not None:
            reconnect = _reconnect_frame(data["reconnect"], parent=data, db=db, now=now)
    except failures as exc:
        return unknown(f"socket recovery evidence unmeasurable: {exc}", incident=incident)
    try:
        current = _frame(data, db=db, now=now)
    except failures as exc:
        return unknown(f"transport evidence unavailable: {exc}", incident=incident)
    return TransportSection(
        {
            "state": "measured",
            "reason": "",
            "inputs": [item],
            **current,
            "in_flight_requests": data["in_flight_requests"],
            "per_client_reconnects": {"state": "unmeasurable", "reason": IDENTITY_REASON, "inputs": [item]},
        },
        incident=incident,
        reconnect=reconnect,
    )


def _failing(counts: dict) -> bool:
    errors = sum(counts[k] for k in ("accept_errors", "read_errors", "write_errors"))
    churn = (
        counts["connections_accepted"] >= 50
        and counts["connections_closed"] >= 50
        and counts["requests_answered"] < counts["connections_closed"]
    )
    return counts["requests_dropped"] >= 3 or errors >= 5 or churn


def report_transport_alert(section: dict[str, Any], *, env: Mapping[str, str]) -> None:
    """Unknown or partial recovery must not clear a previously observed incident."""
    if getattr(section, "overflow", False):
        from .job_alerts import alert_path, report

        db = Path(env["BRAINLAYER_DB"]).expanduser().resolve() if env.get("BRAINLAYER_DB") else get_db_path()
        report(
            "mcp-transport",
            "BrainBar MCP transport: event buffer overflow; storm counts unmeasurable",
            path=alert_path(env=env, db_path=db),
        )
        return
    incident = getattr(section, "incident", None)
    reconnect = getattr(section, "reconnect", None)
    if reconnect and reconnect.get("state") == "unmeasurable":
        reconnect = None  # Intentional discard cannot suppress independent request evidence or clearing.
    if reconnect and reconnect["observed_seconds"] >= 60:
        deficit = reconnect["severed_persistent"] - reconnect["recovered_persistent"]
        # Allow two ordinary session departures; three sustained missing sockets
        # is an aggregate tripwire, never per-client attribution.
        if deficit >= 3:
            from .job_alerts import alert_path, report

            db = Path(env["BRAINLAYER_DB"]).expanduser().resolve() if env.get("BRAINLAYER_DB") else get_db_path()
            report(
                "mcp-transport",
                f"BrainBar MCP transport: {deficit} persistent connections did not recover after restart",
                path=alert_path(env=env, db_path=db),
            )
            return
    if getattr(section, "incident_unknown", False) or (incident is None and section["state"] != "measured"):
        return
    counts = incident["counts"] if incident else section["counts"]
    hard_errors = sum(counts[k] for k in ("accept_errors", "read_errors", "write_errors"))
    # R6's failed-call cluster peaked at 11 in five minutes. Three lost answers
    # is a conservative tripwire, not a claim that all client failures are visible here.
    # Churn is an operational tripwire: suppress normal short-lived answered clients.
    churn = (
        counts["connections_accepted"] >= 50
        and counts["connections_closed"] >= 50
        and counts["requests_answered"] < counts["connections_closed"]
    )
    reason = None
    if incident or counts["requests_dropped"] >= 3 or hard_errors >= 5 or churn:
        reason = f"BrainBar MCP transport: {counts['requests_dropped']} dropped requests, {hard_errors} I/O errors, {counts['connections_closed']} closed connections in {WINDOW_SECONDS}s"
    elif section["observed_seconds"] < WINDOW_SECONDS or counts["requests_dropped"] > 0 or hard_errors > 0:
        return
    if incident:
        reason = f"historical transport episode at {incident['generated_at']}: {reason}"
    if reason is None and reconnect and reconnect["observed_seconds"] < 60:
        return
    from .job_alerts import alert_path, report

    db = Path(env["BRAINLAYER_DB"]).expanduser().resolve() if env.get("BRAINLAYER_DB") else get_db_path()
    report("mcp-transport", reason, path=alert_path(env=env, db_path=db))
