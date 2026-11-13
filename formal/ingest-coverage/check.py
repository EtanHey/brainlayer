#!/usr/bin/env python3
"""Run every IngestCoverage case through TLC and compare it with its expectation.

Exit 0: every case matched. 1: a case did not, a config is stale, or (--require-sync) a covered
source changed. 2: incomplete (no TLC, a crash, an unparseable result).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SPEC = "IngestCoverage"
BOUNDS = {
    "AmberT": 2,
    "RedT": 4,
    "UnknownAmber": 3,
    "Recovery": 2,
    "SleepGap": 3,
    "WakeGrace": 5,
    "AlertRed": 2,
    "Cap": 8,
}  # ordering-preserving: Amber < Red < WakeGrace, SleepGap = 3 ticks
HEARTBEAT = ("none", "dead", "stuck", "blind_stale", "blind_unreadable", "rewrite")
ALL = ("none", "dead", "stuck", "lag", "coverage", "dbfail", "blind_stale", "blind_unreadable", "rootgone", "rewrite")
THREE = ("IngestFailureLeavesGreen", "SleepIsNotFailure", "StallAcrossWakeStillCounts", "StallClockIsTruthful")


def case(
    name, kind, why, *, rule="real", mut="none", clock="confirmed", n=2, faults=HEARTBEAT, inv=THREE, expect="pass"
):
    return {
        "name": name,
        "kind": kind,
        "why": why,
        "rule": rule,
        "mut": mut,
        "clock": clock,
        "n": n,
        "faults": list(faults),
        "invariants": list(inv),
        "expect": expect,
    }


CASES = [
    case(
        "real-heartbeat-shapes",
        "guarantee",
        "Today's code: dead, stuck, blind (stale or unreadable) leave green within 2 ticks; a detected rewrite re-ingests",
    ),
    case(
        "real-unconfirmed-clock",
        "guarantee",
        "Without a boot-session clock, wake grace delays the bound to 7 ticks; a stall still never stays green",
        clock="unconfirmed",
        n=7,
        inv=("IngestFailureLeavesGreen", "StallAcrossWakeStillCounts", "StallClockIsTruthful"),
    ),
    case(
        "real-bound-tight",
        "bound",
        "N = 2 is tight: with N = 1 TLC finds a green tick",
        n=1,
        inv=("IngestFailureLeavesGreen",),
        expect="IngestFailureLeavesGreen",
    ),
    *[
        case(
            f"finding-real-{s}",
            "finding",
            f"FINDING: today's verdict never reads the {s} signal, so a {s} stall stays green forever",
            faults=("none", s),
            inv=("IngestFailureLeavesGreen",),
            expect="IngestFailureLeavesGreen",
        )
        for s in ("lag", "coverage", "dbfail", "rootgone")
    ],
    case(
        "finding-real-rewrite-lossy",
        "finding",
        "FINDING: a same-length edit past the 4 KiB head digest is invisible to the #732 fingerprint; "
        "the edited records are never ingested and the check stays green",
        faults=("none", "rewrite_lossy"),
        inv=("IngestFailureLeavesGreen",),
        expect="IngestFailureLeavesGreen",
    ),
    case(
        "finding-proposed-rewrite-lossy",
        "finding",
        "The proposed alert rule does NOT fix the lossy rewrite: the watcher never knows; it needs a stronger fingerprint",
        rule="proposed",
        faults=("none", "rewrite_lossy"),
        inv=("IngestFailureLeavesGreen",),
        expect="IngestFailureLeavesGreen",
    ),
    case(
        "finding-unconfirmed-short-sleep",
        "finding",
        "FINDING (minor): unconfirmed clock, a 10-15 min lid close, monitor reads before the first poll: amber",
        clock="unconfirmed",
        n=7,
        faults=("none",),
        inv=("SleepIsNotFailure",),
        expect="SleepIsNotFailure",
    ),
    case(
        "proposed-all-shapes",
        "guarantee",
        "The follow-up fix (not in code): verdict also reads the watcher's alerts and missing roots; "
        "all shapes but the lossy rewrite hold",
        rule="proposed",
        faults=ALL,
    ),
    case(
        "mut-stale-green",
        "mutation",
        "Treat a readable but stale health file as green",
        mut="stale_green",
        inv=("IngestFailureLeavesGreen",),
        expect="IngestFailureLeavesGreen",
    ),
    case(
        "mut-trust-alive",
        "mutation",
        "Ignore a frozen poll_count: trust a live watcher process",
        mut="trust_alive",
        faults=("none", "stuck"),
        inv=("IngestFailureLeavesGreen",),
        expect="IngestFailureLeavesGreen",
    ),
    case(
        "mut-no-coverage",
        "mutation",
        "Drop the coverage check from the proposed rule",
        rule="proposed",
        mut="no_coverage",
        faults=("none", "coverage"),
        inv=("IngestFailureLeavesGreen",),
        expect="IngestFailureLeavesGreen",
    ),
    case(
        "mut-sleep-resets",
        "mutation",
        "A wake forgives the heartbeat clock",
        mut="sleep_resets",
        faults=("none", "dead"),
        inv=("StallAcrossWakeStillCounts",),
        expect="StallAcrossWakeStillCounts",
    ),
    case(
        "mut-rewrite-progress",
        "mutation",
        "Count every rewrite as progress, lossy or not",
        mut="rewrite_progress",
        faults=("none", "rewrite_lossy"),
        inv=("StallClockIsTruthful",),
        expect="StallClockIsTruthful",
    ),
]


def tla_set(items) -> str:
    return "{" + ", ".join(f'"{x}"' for x in items) + "}"


def render_cfg(c: dict) -> str:
    consts = {
        **BOUNDS,
        "Rule": f'"{c["rule"]}"',
        "Mut": f'"{c["mut"]}"',
        "Clock": f'"{c["clock"]}"',
        "N": c["n"],
        "Faults": tla_set(c["faults"]),
    }
    lines = [
        "\\* generated by check.py --write-configs; edit CASES there",
        "SPECIFICATION Spec",
        "CONSTANTS",
        "  Script <- NoScript",
        *(f"  {k} = {v}" for k, v in consts.items()),
        "INVARIANTS TypeOK",
        *(f"INVARIANTS {i}" for i in c["invariants"]),
    ]
    return "\n".join(lines) + "\n"


def tlc_command(cfg: Path, metadir: Path) -> list[str] | None:
    workers = os.environ.get("TLC_WORKERS", "1")
    jar = os.environ.get("TLC_JAR") or str(Path.home() / ".local/share/tlaplus/tla2tools-1.7.4.jar")
    if not Path(jar).is_file() or not shutil.which("java"):
        return None
    tmp = metadir.parent / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return [
        "java",
        f"-Djava.io.tmpdir={tmp}",
        "-Xmx512m",
        "-cp",
        jar,
        "tlc2.TLC",
        "-workers",
        workers,
        "-nowarning",
        "-config",
        str(cfg),
        "-metadir",
        str(metadir),
        SPEC,
    ]


def run_tlc(cfg: Path, out: Path, name: str) -> tuple[str, str]:
    """('pass' | violated invariant | 'error', log text). Runs in ``out`` beside a copy of the spec."""
    shutil.copy(HERE / f"{SPEC}.tla", out / f"{SPEC}.tla")
    cmd = tlc_command(cfg, out / f"{name}-states")
    if cmd is None:
        return "error", "TLC not found: set TLC_JAR or install ~/.local/share/tlaplus/tla2tools-1.7.4.jar"
    proc = subprocess.run(cmd, cwd=out, capture_output=True, text=True, check=False)
    log = proc.stdout + proc.stderr
    if "Model checking completed. No error has been found." in log:
        return "pass", log
    hit = re.search(r"Invariant (\w+) is violated", log)
    return (hit.group(1) if hit else "error"), log


def trace_states(log: str) -> list[dict[str, str]]:
    """TLC's printed behaviour as one dict of raw values per state."""
    states, cur = [], None
    for line in log.splitlines():
        if re.match(r"^State \d+:", line):
            cur = {}
            states.append(cur)
        elif cur is not None and (m := re.match(r"^/\\ (\w+) = (.+)$", line.strip())):
            cur[m.group(1)] = m.group(2).strip()
        elif cur is not None and not line.strip():
            cur = None
    return states


def compact_trace(log: str) -> str:
    out, prev = [], {}
    for i, s in enumerate(trace_states(log), 1):
        changed = {k: v for k, v in s.items() if prev.get(k) != v}
        out.append(f"State {i}: " + "  ".join(f"{k}={v}" for k, v in sorted(changed.items())))
        prev = s
    return "\n".join(out) + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync_problems() -> list[str]:
    coverage = json.loads((HERE / "coverage.json").read_text())
    return [
        f"SYNC STALE: {src['path']} sha256 {sha256(ROOT / src['path'])[:12]}… != pinned {src['sha256'][:12]}…"
        for src in coverage["sources"]
        if sha256(ROOT / src["path"]) != src["sha256"]
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", help="run only cases whose name starts with this")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--write-configs", action="store_true")
    ap.add_argument("--require-sync", action="store_true", help="fail when a covered source changed")
    args = ap.parse_args(argv)
    cfg_dir = HERE / "configs"
    if args.list:
        for c in CASES:
            print(f"{c['name']:34} {c['kind']:9} expect={c['expect']:26} {c['why']}")
        return 0
    if args.write_configs:
        cfg_dir.mkdir(exist_ok=True)
        for old in cfg_dir.glob("*.cfg"):
            old.unlink()
        for c in CASES:
            (cfg_dir / f"{c['name']}.cfg").write_text(render_cfg(c))
        print(f"wrote {len(CASES)} configs")
        return 0
    stale = [
        c["name"]
        for c in CASES
        if not (cfg_dir / f"{c['name']}.cfg").is_file() or (cfg_dir / f"{c['name']}.cfg").read_text() != render_cfg(c)
    ]
    stale += [p.stem for p in cfg_dir.glob("*.cfg") if p.stem not in {c["name"] for c in CASES}]
    if stale:
        print(f"CONFIGS STALE (run --write-configs): {', '.join(stale)}")
        return 1
    sync = sync_problems()
    for line in sync:
        print(line)
    out = ROOT / ".build.noindex" / "ingest-coverage" / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{os.getpid()}"
    out.mkdir(parents=True, exist_ok=True)
    results, mismatches, incomplete = [], 0, 0
    for c in CASES:
        if args.case and not c["name"].startswith(args.case):
            continue
        got, log = run_tlc(cfg_dir / f"{c['name']}.cfg", out, c["name"])
        (out / f"{c['name']}.log").write_text(log)
        if got not in ("pass", "error"):
            (out / f"{c['name']}.trace.txt").write_text(compact_trace(log))
        states = (re.findall(r"([\d,]+) distinct states found", log) or [None])[-1]
        ok = got == c["expect"]
        incomplete += got == "error"
        mismatches += not ok and got != "error"
        results.append({**c, "got": got, "matched": ok, "distinct_states": states})
        print(f"{'OK  ' if ok else 'FAIL'} {c['name']:34} expect={c['expect']:26} got={got} ({states or '?'} states)")
    traces = None
    if not args.case:
        sys.path.insert(0, str(HERE))
        import replay  # noqa: PLC0415 - sibling module

        traces = replay.stale_traces(out / "replay")
        print("TRACES STALE (run replay.py --generate): " + ", ".join(traces) if traces else "replay traces current")
        mismatches += bool(traces)
    (out / "results.json").write_text(json.dumps({"results": results, "sync": sync, "stale_traces": traces}, indent=2))
    print(f"output: {out.relative_to(ROOT)}")
    if incomplete:
        return 2
    return 1 if mismatches or (args.require_sync and sync) else 0


if __name__ == "__main__":
    sys.exit(main())
