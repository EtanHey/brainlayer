"""Read-only installed-Mac hotlane CPU budget, using a full ten-minute ps window."""

import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROW = "hotlane backlog CPU"


def installed(command):
    return (
        any(prefix in command for prefix in ("/Cellar/brainlayer/", "/opt/brainlayer/"))
        and "hotlane_brainbar_daemon.py" in command
    )


def ps_sample(pid):
    output = subprocess.check_output(
        ["ps", "-p", str(pid), "-o", "lstart=", "-o", "time=", "-o", "command="], text=True, timeout=5
    ).strip()
    fields = output.split(None, 6)
    if len(fields) != 7 or not installed(fields[6]):
        raise ValueError("PID is not an installed Homebrew hotlane daemon")
    parts = fields[5].split(":")
    seconds = sum(float(value) * 60**index for index, value in enumerate(reversed(parts)))
    return (" ".join(fields[:5]), fields[6]), seconds


def backlog(db):
    with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True, timeout=1) as connection:
        return connection.execute(
            "SELECT count(*) FROM chunks c LEFT JOIN chunk_vectors_rowids r ON r.id=c.id "
            "WHERE r.id IS NULL AND c.content IS NOT NULL AND c.content != '' "
            "AND c.archived_at IS NULL AND c.superseded_by IS NULL AND c.aggregated_into IS NULL"
        ).fetchone()[0]


def measure(pid, db):
    if sys.platform != "darwin":
        return {"status": "unavailable", "reason": "requires an installed Mac; this host is not Darwin"}
    identity, cpu_before = ps_sample(pid)
    pending = [backlog(db)]
    started = time.monotonic()
    for _ in range(60):
        time.sleep(10)
        current, cpu_after = ps_sample(pid)
        if current != identity:
            raise ValueError("hotlane PID identity changed during measurement")
        pending.append(backlog(db))
    elapsed = time.monotonic() - started
    return dict(
        status="measured",
        platform="darwin",
        pid=pid,
        command=identity[1],
        measured_at=datetime.now(timezone.utc).isoformat(),
        window_seconds=elapsed,
        cpu_percent=100 * (cpu_after - cpu_before) / elapsed,
        backlog=pending,
    )


def read_report(path):
    if path is None:
        return "n/a", "n/a — no ten-minute installed-Mac hotlane ps measurement supplied"
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("CPU report must be an object")
        if payload.get("status") == "unavailable" and payload.get("reason"):
            return "n/a", "n/a — " + payload["reason"]
        cpu, window, pending = payload["cpu_percent"], payload["window_seconds"], payload["backlog"]
        budget = float(os.environ.get("BRAINLAYER_HOTLANE_CPU_BUDGET_PERCENT", "60"))
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(payload["measured_at"])).total_seconds()
        if (
            payload["status"] != "measured"
            or payload["platform"] != "darwin"
            or not installed(payload["command"])
            or not all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in (cpu, window, budget))
            or window < 600
            or not 0 <= age <= 3600
            or len(pending) != 61
            or not all(type(x) is int and x >= 0 for x in pending)
        ):
            raise ValueError("invalid, short, or stale installed-Mac CPU evidence")
        if not any(pending):
            return "n/a", "n/a — backlog was empty throughout the measurement"
        return ("RED" if cpu > budget else "GREEN"), (
            f"{cpu:.1f}% / {budget:g}% budget over {window:.1f}s; backlog {pending[0]} → {pending[-1]}"
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        return "RED", f"invalid hotlane CPU report: {error}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = measure(args.pid, args.db)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    status, value = read_report(args.out)
    print(value)
    return int(status == "RED")


if __name__ == "__main__":
    raise SystemExit(main())
