#!/usr/bin/env python3
"""Run the backup-verdict TLA+ cases and compare each result with its expectation.

Every case in CASES names a module, a config and the expected outcome: "pass", or the
invariant/property TLC must report violated. All-fixes cases must pass; each pre-fix case
re-enables one behaviour tonight's PRs removed and must reproduce its counterexample.

Exit 0: every case matched. Exit 1: at least one case did not (a fixed case failed, or a
pre-fix case stopped reproducing). Exit 2: the run could not complete (no TLC, a crash,
a parse error, a timeout). Stale source pins print a SYNC STALE warning; with
--require-sync they also exit 1.

Output goes to .build.noindex/backup-verdict/<UTC timestamp>-<pid>/: one .log per case,
the shortest trace as .trace.txt, and results.json.
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
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIGS = HERE / "configs"
LOAD_LIMIT = 40.0
LOAD_WAIT_S = 600
CASE_TIMEOUT_S = 900

# name, module, expected ("pass" or the violated invariant/property), PR, one-line meaning
CASES: list[dict[str, str]] = [
    # --- all fixes on: must pass -------------------------------------------------------
    {
        "name": "current-backup",
        "module": "BackupVerdict",
        "expect": "pass",
        "pr": "",
        "what": "receipts, grace and labels",
    },
    {
        "name": "current-vacuum",
        "module": "BackupVerdict",
        "expect": "pass",
        "pr": "",
        "what": "weekly VACUUM gate and last_weekly",
    },
    {"name": "current-pipes-drive", "module": "PipeVerdict", "expect": "pass", "pr": "", "what": "drive_auth latch"},
    {
        "name": "current-pipes-stores",
        "module": "PipeVerdict",
        "expect": "pass",
        "pr": "",
        "what": "stores canary latch and hysteresis",
    },
    {
        "name": "current-pipes-embedding",
        "module": "PipeVerdict",
        "expect": "pass",
        "pr": "",
        "what": "embedding stall clears only on progress",
    },
    {
        "name": "current-dot",
        "module": "DotTable",
        "expect": "pass",
        "pr": "",
        "what": "dot is the plan section 3d table; job alerts are notes",
    },
    {
        "name": "current-dot-liveness",
        "module": "DotTable",
        "expect": "pass",
        "pr": "",
        "what": "a clean run clears its job alert",
    },
    # --- pre-fix behaviours: must each produce a counterexample -------------------------
    {
        "name": "bug-1583-B1-canary-latch",
        "module": "PipeVerdict",
        "expect": "RedHoldsUntilEvidence",
        "pr": "#1583 R1 B1",
        "what": "an empty queue clears a failed store canary",
    },
    {
        "name": "bug-1583-B2-resume-grace",
        "module": "PipeVerdict",
        "prefix": "672c9f97",
        "fixed_here": "yes",
        "expect": "RedClearsToGreenOnlyOnEvidence",
        "pr": "#1583 R1 B2",
        "what": "a resume makes a dead watcher look fresh",
    },
    {
        "name": "bug-1583-B3-renewable-grace",
        "module": "BackupVerdict",
        "prefix": "672c9f97",
        "fixed_here": "yes",
        "expect": "NoFalseGreen",
        "pr": "#1583 R1 B3",
        "what": "every monitoring gap renews backup grace",
    },
    {
        "name": "bug-1583-B4-drive-latch",
        "module": "PipeVerdict",
        "expect": "RedHoldsUntilEvidence",
        "pr": "#1583 R1 B4",
        "what": "a failed refresh clears lost Drive access",
    },
    {
        "name": "bug-1583-B5-persisted-max",
        "module": "BackupVerdict",
        "prefix": "672c9f97",
        "fixed_here": "yes",
        "expect": "NoFalseGreen",
        "pr": "#1583 R1 B5",
        "what": "the persisted max keeps a revoked copy",
    },
    {
        "name": "bug-1589-B5a-grace-without-clock",
        "module": "BackupVerdict",
        "prefix": "7f7f6245",
        "fixed_here": "yes",
        "expect": "NoFalseGreen",
        "pr": "#1589 B5a",
        "what": "wake grace hides revocation of the only copy",
    },
    {
        "name": "bug-1589-B5b-shared-revocation",
        "module": "BackupVerdict",
        "prefix": "7f7f6245",
        "fixed_here": "yes",
        "expect": "NoFalseGreen",
        "pr": "#1589 B5b",
        "what": "a local-only re-verify restores a revoked Drive copy",
    },
    {
        "name": "bug-1589-B5c-legacy-identity",
        "module": "BackupVerdict",
        "prefix": "7f7f6245",
        "fixed_here": "yes",
        "expect": "NoFalseGreen",
        "pr": "#1589 B5c",
        "what": "a filename-only revocation misses legacy state",
    },
    {
        "name": "bug-1595-local-removed",
        "module": "BackupVerdict",
        "prefix": "d9d7bff1",
        "fixed_here": "yes",
        "expect": "TruthfulSnapshotLabel",
        "pr": "#1595",
        "what": "a removed local copy still reads as a local backup",
    },
    {
        "name": "bug-1595-recovery-reset",
        "module": "PipeVerdict",
        "expect": "RedHoldsUntilEvidence",
        "pr": "#1595",
        "what": "a failed tick mid-recovery keeps its good-tick count",
    },
    {
        "name": "bug-1596-abandoned",
        "module": "BackupVerdict",
        "prefix": "12b9ddd4",
        "fixed_here": "yes",
        "expect": "NoFalseGreen",
        "pr": "#1596",
        "what": "an abandoned snapshot still counts",
    },
    {
        "name": "bug-1596-superseded",
        "module": "BackupVerdict",
        "prefix": "12b9ddd4",
        "fixed_here": "yes",
        "expect": "NoFalseGreen",
        "pr": "#1596",
        "what": "a deleted superseded stall comes back as the clock",
    },
    {
        "name": "bug-1587-B4-backlog-ages-out",
        "module": "PipeVerdict",
        "expect": "RedHoldsUntilEvidence",
        "pr": "#1587 B4",
        "what": "an embedding stall clears when rows age out",
    },
    {
        "name": "bug-1587-B5-stale-canary-success",
        "module": "PipeVerdict",
        "expect": "RedHoldsUntilEvidence",
        "pr": "#1587 B5",
        "what": "a failed round trip leaves the older success readable",
    },
    {
        "name": "bug-1583-S1-alert-lights-dot",
        "module": "DotTable",
        "expect": "NoStuckAlert",
        "pr": "#1583 S1",
        "what": "a job alert turns the dot red with every check green",
    },
    {
        "name": "bug-s1a5-F1-red-to-amber",
        "module": "PipeVerdict",
        "prefix": "67d23f14",
        "fixed_here": "no",
        "expect": "RedHoldsUntilEvidence",
        "pr": "S1a-5 F1",
        "what": "one stale-but-advanced heartbeat steps ingest from red to amber",
    },
    {
        "name": "bug-s1a5-F2-unobserved-tick-keeps-count",
        "module": "PipeVerdict",
        "prefix": "67d23f14",
        "fixed_here": "no",
        "expect": "RedToGreenNeedsConsecutive",
        "pr": "S1a-5 F2",
        "what": "an unobserved tick between two good ticks does not reset the count",
    },
    {
        "name": "bug-s1a5-F3-graced-stale-keeps-count",
        "module": "PipeVerdict",
        "prefix": "67d23f14",
        "fixed_here": "no",
        "expect": "RedClearsToGreenOnlyOnEvidence",
        "pr": "S1a-5 F3",
        "what": "a stale heartbeat seen inside wake grace does not reset recovery",
    },
    {
        "name": "bug-renewable-grace",
        "module": "PipeVerdict",
        "prefix": "67d23f14",
        "fixed_here": "no",
        "expect": "NoFalseGreenStall",
        "pr": "S1a-6",
        "what": "embedding: green x5 while the true stall age is past red from tick 3 (reviewer trace)",
    },
    {
        "name": "bug-renewable-grace-ingest",
        "module": "PipeVerdict",
        "prefix": "67d23f14",
        "fixed_here": "no",
        "expect": "NoFalseGreenStall",
        "pr": "S1a-6",
        "what": "ingest: every gap renews the grace on a dead watcher",
    },
    {
        "name": "bug-renewable-grace-stores",
        "module": "PipeVerdict",
        "prefix": "67d23f14",
        "fixed_here": "no",
        "expect": "NoFalseGreenStall",
        "pr": "S1a-6",
        "what": "stores: every gap renews the grace on a stuck pending store",
    },
    # --- conformance: random model behaviours replayed through the real evaluate() ---------
    {
        "name": "conform-ingest-main",
        "module": "PipeVerdict",
        "expect": "conforms",
        "pr": "",
        "against": "67d23f14",
        "simulate": "3000,60,7",
        "what": "ingest: the model of main (renewable grace, no S1a-5) agrees with 67d23f14 on every tick",
    },
    {
        "name": "conform-embedding-main",
        "module": "PipeVerdict",
        "expect": "conforms",
        "pr": "",
        "against": "67d23f14",
        "simulate": "3000,60,7",
        "what": "embedding: the model of main (renewable grace, no S1a-5) agrees with 67d23f14 on every tick",
    },
    {
        "name": "conform-stores-main",
        "module": "PipeVerdict",
        "expect": "conforms",
        "pr": "",
        "against": "67d23f14",
        "simulate": "3000,60,7",
        "what": "stores: the model of main (renewable grace, no S1a-5) agrees with 67d23f14 on every tick",
    },
    {
        "name": "conform-ingest-s1a5",
        "module": "PipeVerdict",
        "expect": "conforms",
        "pr": "",
        "against": "4e07651b",
        "simulate": "3000,60,7",
        "what": "ingest: the S1a-5 model (hysteresis, renewable grace) agrees with 4e07651b on every tick (historical; the final tree is covered by conform-ingest-s1a6)",
    },
    {
        "name": "conform-ingest-s1a6",
        "module": "PipeVerdict",
        "expect": "diverges",
        "pr": "S1a-6",
        "fixed_here": "no",
        "simulate": "3000,60,7",
        "what": "ingest: all fixes (S1a-5 + S1a-6 one-window grace) vs this tree; conforms at 8eecb5ee",
    },
    {
        "name": "conform-embedding-s1a6",
        "module": "PipeVerdict",
        "expect": "diverges",
        "pr": "S1a-6",
        "fixed_here": "no",
        "simulate": "3000,60,7",
        "what": "embedding: all fixes (S1a-5 + S1a-6 one-window grace) vs this tree; conforms at 8eecb5ee",
    },
    {
        "name": "conform-stores-s1a6",
        "module": "PipeVerdict",
        "expect": "diverges",
        "pr": "S1a-6",
        "fixed_here": "no",
        "simulate": "3000,60,7",
        "what": "stores: all fixes (S1a-5 + S1a-6 one-window grace) vs this tree; conforms at 8eecb5ee",
    },
    # --- current-code findings (reported, not fixed here) ----------------------------------
    {
        "name": "finding-ingest-red-to-amber",
        "module": "PipeVerdict",
        "expect": "pass",
        "pr": "",
        "what": "one stale-but-advanced heartbeat steps ingest from red to amber",
    },
    {
        "name": "finding-ingest-graced-stale-keeps-count",
        "module": "PipeVerdict",
        "expect": "pass",
        "pr": "",
        "what": "a stale heartbeat seen inside wake grace does not reset recovery",
    },
    {
        "name": "finding-ingest-unobserved-tick-keeps-count",
        "module": "PipeVerdict",
        "expect": "pass",
        "pr": "",
        "what": "an unobserved tick between two good ticks does not reset the count",
    },
    # --- design limit and scratch breaks: prove the properties can fail ------------------
    {
        "name": "limit-unreadable-log",
        "module": "BackupVerdict",
        "expect": "NoFalseGreenStrict",
        "pr": "",
        "what": "receipts appended while the log is unreadable are invisible",
    },
    {
        "name": "scratch-vacuum-any-receipt",
        "module": "BackupVerdict",
        "expect": "VacuumNeedsFreshSnapshot",
        "pr": "",
        "what": "the gate trusts any old local_ok receipt",
    },
    {
        "name": "scratch-deferred-advances",
        "module": "BackupVerdict",
        "expect": "VacuumNeedsFreshSnapshot",
        "pr": "",
        "what": "a deferred weekly run advances last_weekly",
    },
    {
        "name": "scratch-clean-run-keeps-alert",
        "module": "DotTable",
        "expect": "AlertClears",
        "pr": "",
        "what": "a clean run does not clear its job alert",
    },
]

COVERED = (
    "src/brainlayer/health_verdict.py",
    "src/brainlayer/backup_daily.py",
    "src/brainlayer/maintenance.py",
    "src/brainlayer/maintenance_calendar.py",
    "src/brainlayer/job_alerts.py",
    "src/brainlayer/health_facts.py",
)

_VIOLATION = re.compile(r"Error: (?:Invariant|Action property) (\w+) is violated")
_TEMPORAL = re.compile(r"Error: Temporal properties were violated")
TEMPORAL = {"AlertClears"}
_STATE = re.compile(r"^State (\d+): <(\w+)")


def tlc_command() -> list[str]:
    """TLC as one executable path (a wrapper like ~/.local/bin/tlc), or java -cp $TLC_JAR."""
    jar = os.environ.get("TLC_JAR")
    if jar:
        return ["java", "-XX:+UseParallelGC", "-Xmx512m", "-cp", str(Path(jar).expanduser().resolve()), "tlc2.TLC"]
    raw = os.environ.get("TLC") or str(Path.home() / ".local/bin/tlc")
    return [str(Path(raw).expanduser().absolute())]


def wait_for_load() -> dict[str, float]:
    started = time.monotonic()
    while True:
        load = os.getloadavg()[0]
        waited = time.monotonic() - started
        if load <= LOAD_LIMIT:
            return {"load_1m": load, "waited_s": round(waited, 1)}
        if waited >= LOAD_WAIT_S:
            raise RuntimeError(f"host load {load:.1f} stayed above {LOAD_LIMIT} for {LOAD_WAIT_S}s; TLC not started")
        time.sleep(15)


def parse_trace(text: str) -> list[tuple[str, dict[str, str]]]:
    """TLC's error trace as (action, {variable: value}) per state."""
    states: list[tuple[str, dict[str, str]]] = []
    current: dict[str, str] | None = None
    name = None
    for line in text.splitlines():
        match = _STATE.match(line)
        if match:
            current = {}
            states.append((match.group(2), current))
            name = None
            continue
        if current is None:
            continue
        if not line.strip():
            current = None
            continue
        stripped = line.strip()
        if stripped.startswith("/\\ ") and " = " in stripped:
            name, _, value = stripped[3:].partition(" = ")
            current[name] = value.strip()
        elif name is not None:
            current[name] += " " + stripped
    return states


def render_trace(states: list[tuple[str, dict[str, str]]]) -> str:
    """Each step with only the variables it changed."""
    lines: list[str] = []
    previous: dict[str, str] = {}
    for index, (action, values) in enumerate(states, 1):
        changed = {k: v for k, v in values.items() if previous.get(k) != v}
        lines.append(f"{index}. {action}")
        lines.extend(f"     {k} = {' '.join(v.split())}" for k, v in sorted(changed.items()))
        previous = values
    return "\n".join(lines)


def outcome(text: str, returncode: int) -> str:
    """'pass', the violated name, 'temporal', or 'error'."""
    if match := _VIOLATION.search(text):
        return match.group(1)
    if _TEMPORAL.search(text):
        return "temporal"
    if returncode == 0 and "Model checking completed. No error has been found." in text:
        return "pass"
    return "error"


SRC_REF: str | None = None


def run_case(case: dict[str, str], out: Path) -> dict[str, object]:
    module = HERE / f"{case['module']}.tla"
    work = out / case["name"]
    work.mkdir(parents=True)
    shutil.copy(module, work / module.name)
    cfg = CONFIGS / f"{case['name']}.cfg"
    load = wait_for_load()
    command = [
        *tlc_command(),
        "-workers",
        os.environ.get("TLC_WORKERS", "1"),
        "-config",
        str(cfg),
        "-metadir",
        str(work / "states"),
        module.name,
    ]
    if case.get("simulate"):
        # -simulate num is per worker: one worker keeps the trace count and seed reproducible.
        traces, depth, seed = case["simulate"].split(",")
        command[command.index("-workers") + 1] = "1"
        command[-1:-1] = ["-simulate", f"file={work}/tr,num={traces}", "-depth", depth, "-seed", seed]
    started = time.monotonic()
    proc = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=CASE_TIMEOUT_S)
    seconds = round(time.monotonic() - started, 1)
    text = proc.stdout + proc.stderr
    (out / f"{case['name']}.log").write_text(text)
    got = outcome(text, proc.returncode)
    conformance = {}
    if case.get("simulate"):
        got, conformance = simulate_conformance(case, work, text, proc.returncode, out)
    if got == "temporal" and case["expect"] in TEMPORAL:
        got = case["expect"]  # TLC names no property; each temporal config checks exactly one
    trace = parse_trace(text)
    if trace:
        (out / f"{case['name']}.trace.txt").write_text(render_trace(trace) + "\n")
    replays = {}
    if case.get("prefix"):
        if not trace:
            # Name each ref so the report prints the failed replay instead of raising KeyError and
            # hiding every other case's result (Macroscope, #1601). Still fails closed.
            replays = {
                "prefix": {"ref": case["prefix"], "error": "no counterexample trace to replay"},
                "src": {"ref": "working tree", "error": "no counterexample trace to replay"},
            }
        elif case["module"] == "BackupVerdict":
            replays = {"prefix": replay(case, trace, case["prefix"], out), "src": replay(case, trace, SRC_REF, out)}
        else:
            states = [values for _, values in trace]
            replays = {
                "prefix": replay_pipe(case, [states], case["prefix"], out),
                "src": replay_pipe(case, [states], SRC_REF, out),
            }
    distinct = re.findall(r"(\d+) distinct states found", text.replace(",", ""))
    version = re.search(r"TLC2 Version (\S+)", text)
    return {
        **case,
        "got": got,
        "matched": got == case["expect"] and replay_matches(case, replays),
        "replay": replays,
        "conformance": conformance,
        "trace_steps": len(trace),
        "distinct_states": int(distinct[-1]) if distinct else None,
        "seconds": seconds,
        "tlc_version": version.group(1) if version else None,
        **load,
    }


def cfg_constants(name: str) -> dict[str, str]:
    lines = (CONFIGS / f"{name}.cfg").read_text().splitlines()
    return {k.strip(): v.strip() for k, _, v in (l.partition("=") for l in lines if l.startswith("  ") and "=" in l)}


def source_tree(ref: str | None, out: Path) -> Path:
    """health_verdict.py + backup_daily.py of a ref (None: this working tree), never imported as a package."""
    if ref is None:
        return REPO / "src" / "brainlayer"
    target = out / f"src-{ref}"
    if not target.exists():
        target.mkdir(parents=True)
        for name in ("health_verdict.py", "backup_daily.py"):
            blob = subprocess.check_output(["git", "show", f"{ref}:src/brainlayer/{name}"], cwd=REPO)
            (target / name).write_bytes(blob)
    return target


def replay(case: dict[str, str], states: list, ref: str | None, out: Path) -> dict[str, object]:
    """Run the counterexample through real backup_facts()/evaluate() at ref; did it reproduce?"""
    sys.path.insert(0, str(HERE))
    import replay as rp  # noqa: PLC0415 - sibling script

    constants = {k: int(v) if v.lstrip("-").isdigit() else v for k, v in cfg_constants(case["name"]).items()}
    payload = {"constants": constants, "ops": rp.ops_from_trace(states)}
    if not any(op["op"] == "tick" for op in payload["ops"]):
        return {"ref": ref or "working tree", "error": "counterexample has no health tick to replay"}
    proc = subprocess.run(
        [sys.executable, "-S", str(HERE / "replay.py"), "--run", str(source_tree(ref, out))],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
        cwd=out,
    )
    if proc.returncode != 0:
        return {"ref": ref or "working tree", "error": proc.stderr.strip().splitlines()[-1:]}
    ticks = json.loads(proc.stdout)
    if not ticks:
        return {"ref": ref or "working tree", "error": "replay returned no tick"}
    last = ticks[-1]
    if case["expect"] == "TruthfulSnapshotLabel":
        reproduces = (
            last["src"]["snap"] != "red"
            and last["snap_label"] == "Local database backup"
            and "local copy removed" not in last["snap_note"]
        )
    else:
        reproduces = all(last["src"][c] != "red" for c in ("snap", "off") if last["model"][c] != "red")
    conforms = all(t["src"] == t["model"] for t in ticks)
    return {"ref": ref or "working tree", "reproduces": reproduces, "conforms": conforms, "ticks": ticks}


def replay_pipe(case: dict[str, str], traces: list, ref: str | None, out: Path) -> dict[str, object]:
    """Feed each trace's observations to the real evaluate() at ref; compare the focused check per tick.

    Fails closed: no trace, a trace that parsed to no state, or a replay that returns a different
    number of runs is an error, never conformance.
    """
    sys.path.insert(0, str(HERE))
    import replay as rp  # noqa: PLC0415 - sibling script

    where = ref or "working tree"
    if not traces or any(len(states) < 2 for states in traces):
        return {"ref": where, "error": "no trace (or a trace with no step) to replay"}
    constants = {k: int(v) if v.lstrip("-").isdigit() else v for k, v in cfg_constants(case["name"]).items()}
    focus = constants["Focus"].strip('"')
    ops = [rp.pipe_ops(states, focus) for states in traces]
    payload = {"constants": constants, "focus": focus, "traces": ops}
    proc = subprocess.run(
        [sys.executable, "-S", str(HERE / "replay.py"), "--run-pipe", str(source_tree(ref, out))],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=600,
        cwd=out,
    )
    if proc.returncode != 0:
        return {"ref": where, "error": proc.stderr.strip().splitlines()[-1:]}
    runs = json.loads(proc.stdout)
    if len(runs) != len(traces):
        return {"ref": where, "error": f"replayed {len(runs)} of {len(traces)} traces"}
    ticks = sum(len(run) for run in runs)
    if ticks == 0:
        return {"ref": where, "error": "no health tick in any trace"}
    diverging = [run for run in runs if any(t["src"] != t["model"] for t in run)]
    last = runs[0][-1] if runs[0] else None
    return {
        "ref": where,
        "reproduces": last is not None and last["src"] == last["model"],
        "conforms": not diverging,
        "traces": len(runs),
        "ticks": ticks,
        "diverging": len(diverging),
        "first_divergence": diverging[0] if diverging else None,
    }


def simulate_conformance(case: dict[str, str], work: Path, text: str, code: int, out: Path) -> tuple[str, dict]:
    """Random behaviours of the model, replayed through the real code: do they agree on every tick?

    Fails closed: TLC must have succeeded, written at least one trace file, every file must parse,
    and at least one health tick must be compared; anything else is "error".
    """
    sys.path.insert(0, str(HERE))
    import replay as rp  # noqa: PLC0415 - sibling script

    if code != 0 or "Error:" in text:
        return "error", {"error": "TLC simulation did not succeed"}
    files = sorted(work.glob("tr_*"))
    if not files:
        return "error", {"error": "simulation wrote no trace file"}
    traces = [rp.parse_sim_file(f.read_text()) for f in files]
    if any(not states for states in traces):
        return "error", {"error": "a simulation trace file parsed to no state"}
    result = replay_pipe(case, traces, case.get("against") or SRC_REF, out)
    if "error" in result:
        return "error", result
    return ("conforms" if result["conforms"] else "diverges"), result


def replay_matches(case: dict[str, str], replays: dict) -> bool:
    """A case with a pre-fix ref needs both replays, without error.

    The pre-fix ref must reproduce the counterexample and agree with the model on every tick. This
    tree must reproduce it exactly when the fix is not here yet (fixed_here "no"), and then also
    agree on every tick; once fixed, it is expected to differ.
    """
    if not case.get("prefix"):
        return not replays
    prefix, src = replays.get("prefix") or {}, replays.get("src") or {}
    if "error" in prefix or "error" in src or "reproduces" not in prefix or "reproduces" not in src:
        return False
    if prefix["reproduces"] is not True or prefix.get("conforms") is not True:
        return False
    still_here = case.get("fixed_here") == "no"
    if src["reproduces"] != still_here:
        return False
    return src.get("conforms") is True if still_here else True


def sync_report() -> list[str]:
    pins = json.loads((HERE / "coverage.json").read_text())["sources"]
    stale = []
    for path in COVERED:
        source = REPO / path
        digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        if pins.get(path, {}).get("sha256") != digest:
            stale.append(path)
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", action="append", default=[], help="run only cases whose name starts with this")
    parser.add_argument("--require-sync", action="store_true", help="fail when a covered source changed since the pin")
    parser.add_argument("--src-ref", help="replay counterexamples against this git ref instead of the working tree")
    parser.add_argument("--list", action="store_true", help="print the case table and exit")
    args = parser.parse_args(argv)
    global SRC_REF
    SRC_REF = args.src_ref
    cases = [c for c in CASES if not args.case or any(c["name"].startswith(p) for p in args.case)]
    if args.list:
        for c in cases:
            print(f"{c['name']:40} {c['module']:14} expect {c['expect']:26} {c['pr']}")
        return 0
    if not cases:
        print("no case matches", file=sys.stderr)
        return 2
    missing = [c["name"] for c in cases if not (CONFIGS / f"{c['name']}.cfg").is_file()]
    if missing:
        print(f"missing configs: {', '.join(missing)}", file=sys.stderr)
        return 2
    stale = sync_report()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / ".build.noindex" / "backup-verdict" / f"{stamp}-{os.getpid()}"
    out.mkdir(parents=True)
    results = []
    for case in cases:
        result = run_case(case, out)
        results.append(result)
        mark = "ok  " if result["matched"] else "FAIL"
        label = f" ({case['pr']})" if case["pr"] else ""
        print(
            f"{mark} {case['name']}: expected {case['expect']}, got {result['got']}"
            f" [{result['distinct_states']} states, {result['seconds']}s]{label}",
            flush=True,
        )
        for where, r in result["replay"].items():
            if "error" in r:
                print(f"       replay at {r['ref']}: ERROR {r['error']}")
            else:
                verdict = "reproduces" if r["reproduces"] else "does not reproduce"
                fit = "" if r["conforms"] or where == "src" else " (tick states differ from the model)"
                note = " <- still in this source" if where == "src" and r["reproduces"] else ""
                if where == "src" and r["reproduces"] != (case.get("fixed_here") == "no"):
                    note += " (UNEXPECTED: update fixed_here once the fix's PR state changed)"
                print(f"       replay at {r['ref']}: {verdict}{fit}{note}")
    (out / "results.json").write_text(json.dumps({"stale_sources": stale, "results": results}, indent=2) + "\n")
    for path in stale:
        print(f"SYNC STALE: {path} changed since coverage.json's pin; re-confirm the model, then move the pin")
    print(f"results: {out}")
    if any(r["got"] == "error" for r in results):
        return 2
    if not all(r["matched"] for r in results):
        return 1
    return 1 if stale and args.require_sync else 0


def cli() -> int:
    try:
        return main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - any crash is an incomplete verification, never a pass
        print(f"Verification could not complete: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(cli())
