"""Fail-closed consumption of BrainBar's aggregate, bounded transport evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .paths import get_db_path

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


def build_transport_section(
    *, env: Mapping[str, str], now: datetime, record_input: Callable[..., dict]
) -> dict[str, Any]:
    raw_db = Path(env["BRAINLAYER_DB"]).expanduser() if env.get("BRAINLAYER_DB") else get_db_path()
    db = raw_db.resolve()
    path = Path(
        env.get("BRAINLAYER_TRANSPORT_HEALTH_PATH", str(raw_db.with_name("transport-health.json")))
    ).expanduser()
    status = "read"
    try:
        if path.stat().st_size > 65_536:
            raise ValueError("receipt exceeds 64 KiB")
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported transport receipt")
        if data.get("state") != "measured" or data.get("reason") != "":
            raise ValueError("BrainBar transport evidence unavailable: " + str(data.get("reason") or "unknown state"))
        if Path(data["db_path"]).expanduser().resolve() != db:
            raise ValueError("receipt belongs to a different database")
        generated = datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))
        started = datetime.fromisoformat(data["producer_started_at"].replace("Z", "+00:00"))
        if generated.tzinfo is None or started.tzinfo is None or started > generated:
            raise ValueError("invalid producer timestamps")
        age = (now - generated.astimezone(UTC)).total_seconds()
        if not 0 <= age <= 60:
            raise ValueError("transport receipt is stale or from the future")
        if (
            data.get("window_seconds") != WINDOW_SECONDS
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
        if type(data["in_flight_requests"]) is not int or data["in_flight_requests"] < 0:
            raise ValueError("invalid in-flight count")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        status = "missing" if not path.exists() else "malformed"
        return {
            "state": "unmeasurable",
            "reason": f"transport evidence unavailable: {exc}",
            "inputs": [record_input(path, status=status)],
        }
    item = record_input(path, status=status)
    if item["status"] != "read":
        return {"state": "unmeasurable", "reason": "transport input status: " + item["status"], "inputs": [item]}
    return {
        "state": "measured",
        "reason": "",
        "inputs": [item],
        "generated_at": data["generated_at"],
        "producer_started_at": data["producer_started_at"],
        "window_seconds": WINDOW_SECONDS,
        "observed_seconds": data["observed_seconds"],
        "counts": counts,
        "in_flight_requests": data["in_flight_requests"],
        "per_client_reconnects": {"state": "unmeasurable", "reason": IDENTITY_REASON, "inputs": [item]},
    }


def report_transport_alert(section: dict[str, Any], *, env: Mapping[str, str]) -> None:
    """Unknown or partial recovery must not clear a previously observed incident."""
    if section["state"] != "measured":
        return
    counts = section["counts"]
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
    if counts["requests_dropped"] >= 3 or hard_errors >= 5 or churn:
        reason = f"BrainBar MCP transport: {counts['requests_dropped']} dropped requests, {hard_errors} I/O errors, {counts['connections_closed']} closed connections in {WINDOW_SECONDS}s"
    elif section["observed_seconds"] < WINDOW_SECONDS or counts["requests_dropped"] > 0 or hard_errors > 0:
        return
    from .job_alerts import alert_path, report

    db = Path(env["BRAINLAYER_DB"]).expanduser().resolve() if env.get("BRAINLAYER_DB") else get_db_path()
    report("mcp-transport", reason, path=alert_path(env=env, db_path=db))
