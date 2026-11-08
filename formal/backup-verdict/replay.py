#!/usr/bin/env python3
"""Replay a BackupVerdict counterexample against real health_verdict source.

The TLC trace becomes real backup-daily.log receipt dicts and real evaluate() ticks:
one model hour is one real hour, and the thresholds are set to the model's constants
(amber = red, so amber appears only from wake grace, as in the model). The real
backup_facts() and evaluate() from the chosen source run on them, so the result says
whether that code reproduces the counterexample.

The source never imports the BrainLayer package: health_verdict.py is loaded on its own,
and backup_daily is replaced by a stub that executes the real _snapshot_created_time and
_parse_snapshot_date bodies, extracted from that source's backup_daily.py. Nothing reads a
file, a database or a socket.

    python3 -S replay.py --run <dir with health_verdict.py and backup_daily.py> < ops.json
    python3 -S replay.py --run-pipe <same dir> < ops.json       (PipeVerdict stall-check ticks)
    python3 -S replay.py --run-weekly <same dir> < ops.json     (BackupVerdict weekly VACUUM gate)

--run-weekly is the one mode that touches files: it writes real snapshot files and a real
backup-daily.log under a fresh directory in the current working directory (check.py runs it
inside .build.noindex/), because the gate it replays, _recent_local_verified_backup, proves a
snapshot by hashing its bytes.
"""

from __future__ import annotations
import __future__

import ast
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import types
import uuid
from pathlib import Path

BASE = dt.datetime(2026, 11, 2, 3, tzinfo=dt.UTC)  # 03:00Z, so filename midnight != exact time
HOUR = 3600.0
REAL_STATES = {"green", "amber", "red", "unknown"}


def _at(hour: int) -> str:
    return (BASE + dt.timedelta(hours=hour)).isoformat()


def _key(snap: int) -> str:
    return f"/b/2026-11-0{snap}.db.zst"  # one dated artifact per snapshot id, dates in id order


def receipt(op: dict) -> dict:
    """The receipt backup_daily writes for one model receipt kind."""
    body = {"backup_log_provenance": "real", "snapshot": _key(op["snap"]), "attempted_at": _at(op["hour"])}
    if op["kind"] != "revoke" or op["exact"]:
        body["snapshot_created_at"] = _at(op["created"])
    kind = op["kind"]
    if kind == "local_ok":
        body |= {"local_verified": True, "local_integrity": "ok"}
    elif kind == "upload_ok":
        body |= {"uploaded": True, "verified": True, "local_verified": True, "local_integrity": "ok"}
    elif kind == "revoke":
        body |= {"drive_md5_match": False}
    elif kind == "removed":
        body |= {"local_removed": True}
    elif kind == "abandon":
        body |= {"upload_state": "abandoned", "local_verified": False, "error_code": "local_snapshot_invalid"}
    elif kind == "supersede":
        body |= {"upload_state": "superseded", "local_verified": False, "local_removed": False}
    return body


def _tla_seq(value: str) -> list[str]:
    return [part.strip().strip('"') for part in value.strip("<> ").split(",")]


def _tla_record(value: str) -> dict[str, str]:
    inner = value.strip().strip("[]")
    out = {}
    for part in inner.split(","):
        if "|->" in part:
            k, v = part.split("|->")
            out[k.strip()] = v.strip().strip('"')
    return out


def _acc_rev(acc: str) -> list[str]:
    """The per-snapshot ``rev`` sequence of the reader fold, e.g. ``<<"approx", "none">>``."""
    match = re.search(r"rev\s*\|->\s*<<([^>]*)>>", acc)
    if match is None:
        raise ValueError(f"trace acc has no rev sequence: {acc!r}")  # fail closed, never guess
    return [part.strip().strip('"') for part in match.group(1).split(",")]


def ops_from_trace(states: list[tuple[str, dict[str, str]]]) -> list[dict]:
    """Receipts appended and ticks taken, in trace order, from TLC's state sequence."""
    ops: list[dict] = []
    previous: dict[str, str] | None = None
    for action, values in states:
        if previous is not None and action == "World":
            before, after = _tla_seq(previous["lastLoc"]), _tla_seq(values["lastLoc"])
            created = [int(x) for x in _tla_seq(values["created"])]
            snap = next(i for i, (a, b) in enumerate(zip(before, after), 1) if a != b)
            # Only a revocation can be filename-only ("approx"), and it marks ITS snapshot's rev
            # entry. Read that entry, not "approx anywhere": an earlier snapshot's marker must not
            # make a later approximate revocation replay as exact (Macroscope, #1601).
            exact = _acc_rev(values.get("acc", ""))[snap - 1] != "approx"
            ops.append(
                {
                    "op": "receipt",
                    "hour": int(values["now"]),
                    "kind": after[snap - 1],
                    "snap": snap,
                    "created": created[snap - 1],
                    "exact": exact,
                    # The one receipt whose line the model marks unreadable (dmg turns TRUE).
                    "damaged": previous.get("dmg") == "FALSE" and values.get("dmg") == "TRUE",
                }
            )
        elif previous is not None and action == "WeeklyRun":
            ops.append({"op": "weekly", "hour": int(values["now"]), "model": _tla_record(values["vac"])["kind"]})
        elif previous is not None and action == "Tick":
            st = _tla_record(values["st"])
            ops.append(
                {
                    "op": "tick",
                    "hour": int(values["now"]),
                    "readable": values["lastReadable"] == "TRUE",
                    "model": {"snap": st["snap"], "off": st["off"]},
                }
            )
        previous = values
    return ops


def _stub_backup_daily(source: str) -> types.ModuleType:
    tree = ast.parse(source)
    wanted = {"_snapshot_created_time", "_parse_snapshot_date"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    consts = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id.endswith("_SUFFIX")
            or isinstance(t, ast.Name)
            and t.id == "DB_SNAPSHOT_SUFFIXES"
            for t in n.targets
        )
    ]
    module = types.ModuleType("brainlayer.backup_daily")
    module.__dict__.update({"dt": dt, "Path": Path, "Mapping": dict, "Any": object})
    code = ast.Module(body=consts + nodes, type_ignores=[])
    flags = __future__.annotations.compiler_flag
    exec(compile(code, "backup_daily.py(extracted)", "exec", flags=flags), module.__dict__)  # noqa: S102 - real source bodies
    missing = wanted - set(module.__dict__)
    if missing:
        raise RuntimeError(f"backup_daily.py lacks {sorted(missing)}")
    return module


def _load(src: Path) -> types.ModuleType:
    package = types.ModuleType("brainlayer")
    package.__path__ = []
    sys.modules["brainlayer"] = package
    sys.modules["brainlayer.backup_daily"] = _stub_backup_daily((src / "backup_daily.py").read_text())
    spec = importlib.util.spec_from_file_location("brainlayer.health_verdict", src / "health_verdict.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["brainlayer.health_verdict"] = module
    spec.loader.exec_module(module)
    return module


def run(src: Path, payload: dict) -> list[dict]:
    hv = _load(src)
    c = payload["constants"]
    t = hv.Thresholds(
        backup_local_amber_s=c["RedSnap"] * HOUR,
        backup_local_red_s=c["RedSnap"] * HOUR,
        backup_offsite_amber_s=c["RedOff"] * HOUR,
        backup_offsite_red_s=c["RedOff"] * HOUR,
        wake_grace_s=c["Grace"] * HOUR,
        sleep_gap_s=c["Gap"] * HOUR,
    )
    receipts: list[dict] = []
    prior = None
    out = []
    for op in payload["ops"]:
        if op["op"] == "receipt":
            if not op.get("damaged"):  # health_facts.read_receipts skips a line it cannot parse
                receipts.append(receipt(op))
            continue
        if op["op"] != "tick":
            continue
        now = BASE + dt.timedelta(hours=op["hour"])
        kwargs = {}
        if op["readable"]:
            facts = hv.backup_facts(receipts, as_of=now)
            if hasattr(facts, "facts_kwargs"):
                kwargs = facts.facts_kwargs()
            else:
                kwargs = {
                    "backup_local_verified_at": facts.local_verified_at,
                    "backup_offsite_verified_at": facts.offsite_verified_at,
                }
        if prior is not None and c.get("LegacyState") == "TRUE":
            for name in ("backup_local", "backup_offsite"):  # state written before clock_id existed
                prior["checks"].get(name, {}).pop("clock_id", None)
        verdict = hv.evaluate(hv.Facts(now=now, **kwargs), prior, t)
        prior = verdict.state
        snap, off = verdict.check("backup_local"), verdict.check("backup_offsite")
        out.append(
            {
                "hour": op["hour"],
                "model": op["model"],
                "src": {"snap": snap.state, "off": off.state},
                "snap_note": snap.note,
                "snap_label": hv.CHECK_LABELS["backup_local"],
            }
        )
    return out


# --- weekly VACUUM gate (BackupVerdict, Weekly = TRUE) ------------------------------------

GATE_FUNCTIONS = {
    "_recent_local_verified_backup",
    "_read_backup_receipts",
    "_scan_backup_receipts",
    "_local_snapshot_intact",
    "_upload_identity",
    "_snapshot_created_time",
    "_parse_snapshot_date",
    "local_backup_date",
}
GATE_CLASSES = {"BackupReceiptUnreadableError"}


def _gate(source: str) -> types.FunctionType:
    """The real _recent_local_verified_backup and the bodies it calls, extracted from one source."""
    tree = ast.parse(source)
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in GATE_FUNCTIONS
        or isinstance(n, ast.ClassDef)
        and n.name in GATE_CLASSES
        or isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and (t.id.endswith("_SUFFIX") or t.id in {"DB_SNAPSHOT_SUFFIXES", "_KNOWN_PROGRESS_NOISE"})
            for t in n.targets
        )
    ]
    found = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    missing = (GATE_FUNCTIONS | GATE_CLASSES) - found
    if missing:
        raise RuntimeError(f"backup_daily.py lacks {sorted(missing)}")  # fail closed, never stub a gate body
    namespace = {"dt": dt, "Path": Path, "Mapping": dict, "Any": object, "re": re, "json": json}
    namespace |= {"hashlib": hashlib, "os": os}
    code = ast.Module(body=nodes, type_ignores=[])
    flags = __future__.annotations.compiler_flag
    exec(compile(code, "backup_daily.py(extracted)", "exec", flags=flags), namespace)  # noqa: S102 - real source bodies
    return namespace["_recent_local_verified_backup"]


def run_weekly(src: Path, payload: dict) -> list[dict]:
    """Write the trace's receipts as a real backup-daily.log, one real file per snapshot, and ask
    the real gate at each weekly run whether a fresh verified local snapshot exists."""
    os.environ["TZ"] = "UTC"  # local_backup_date() compares local days; BASE..Horizon stays on one
    time.tzset()
    gate = _gate((src / "backup_daily.py").read_text())
    root = Path.cwd() / f"weekly-{uuid.uuid4().hex[:8]}"
    staging = root / "staging"
    staging.mkdir(parents=True)
    log, db = root / "backup-daily.log", root / "brainlayer.db"
    removes = {"removed", "abandon", "supersede"}  # the model's file[s] turns FALSE
    out = []
    for op in payload["ops"]:
        if op["op"] == "receipt":
            path = staging / Path(_key(op["snap"])).name
            if not path.exists() and op["kind"] == "local_ok":
                path.write_bytes(f"snapshot {op['snap']}\n".encode() * 64)
            proof = {}
            if path.is_file():  # the local proof every receipt about an existing artifact carries
                proof = {
                    "local_verified": True,
                    "local_integrity": "ok",
                    "local_integrity_pragma": "integrity_check",
                    "local_artifact": {
                        "path": str(path.resolve()),
                        "size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    },
                }
            body = {**proof, **receipt(op), "snapshot": str(path), "db": str(db)}
            line = json.dumps(body, sort_keys=True)
            # A damaged line: newline-terminated but not JSON, so not a torn final record either.
            with log.open("a", encoding="utf-8") as handle:
                handle.write((line[:-1] if op.get("damaged") else line) + "\n")
            if op["kind"] in removes and path.exists():
                path.unlink()
            continue
        if op["op"] != "weekly":
            continue
        now = BASE + dt.timedelta(hours=op["hour"])
        found = gate(log, db, now=now, max_age_hours=float(payload["constants"]["VacAge"])) if log.exists() else None
        out.append({"hour": op["hour"], "model": op["model"], "src": "fresh" if found else "none"})
    return out


# --- ingest (PipeVerdict, Focus = "ingest") ---------------------------------------------


def parse_sim_file(text: str) -> list[dict[str, str]]:
    """States of a TLC -simulate trace module (STATE_n == /\\ var = value ...)."""
    states: list[dict[str, str]] = []
    name = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("STATE_") and stripped.endswith("=="):
            states.append({})
            name = None
        elif states and stripped.startswith("/\\ ") and " = " in stripped:
            name, _, value = stripped[3:].partition(" = ")
            states[-1][name] = value.strip()
        elif states and name is not None and stripped and not stripped.startswith("="):
            states[-1][name] += " " + stripped
    return states


def _fact_at(hour_value: str) -> list:
    return ["at", int(hour_value)] if int(hour_value) >= 0 else ["none"]


DAEMON_ACTIVITY = {
    "progress": "Embedding progressing but incoming work is keeping the queue full: 1 queued, last batch 00:00 (local time)",
    "paused": "Embedding paused by machine load: 1 queued, last batch time unavailable",
}


def pipe_ops(states: list[dict[str, str]], focus: str) -> list[dict]:
    """One op per health tick (lastTick changes on every tick and nothing else), with the facts
    that tick observed for the focused check, as evaluate() receives them."""
    ops = []
    for before, after in zip(states, states[1:]):
        if after.get("lastTick") == before.get("lastTick"):
            continue
        obs = _tla_record(after["lastObs"])
        facts: dict[str, list] = {}
        if focus == "ingest":
            facts["watcher_updated_at"] = ["at", int(after["hb"])] if obs["hb"] == "seen" else ["unobs"]
        elif focus == "embedding":
            e = obs["emb"]
            stalled = after["backlog"] == "TRUE"
            facts["embedding_stalled_since"] = (
                ["unobs"]
                if e == "unobs"
                else ["none"]
                if e == "clear" or (e == "progress" and not stalled)
                else _fact_at(after["stallSince"])
            )
            # #1686: hotlane daemon evidence, as collect_facts hands it to evaluate(). Only
            # Daemon # "none" models emit these, so pre-#1686 refs never receive the fields.
            if e in DAEMON_ACTIVITY:
                facts["embedding_activity"] = ["str", DAEMON_ACTIVITY[e]]
                facts["embedding_progress_recent"] = ["bool", e == "progress"]
        elif focus == "stores":
            q, k = obs["queue"], obs["canary"]
            facts["oldest_pending_store_at"] = (
                ["unobs"] if q == "unobs" else ["none"] if q == "empty" else _fact_at(after["pendStart"])
            )
            facts["store_canary_ok"] = ["unobs"] if k == "unobs" else ["bool", k == "ok"]
        else:
            raise ValueError(f"no pipe replay for {focus}")
        ops.append({"hour": int(after["now"]), "facts": facts, "model": _tla_record(after["st"])[focus]})
    return ops


def ingest_ops(states: list[dict[str, str]]) -> list[dict]:
    return pipe_ops(states, "ingest")


def run_pipe(src: Path, payload: dict, hv: types.ModuleType | None = None) -> list[dict]:
    """The real evaluate() for one pipe check, fed the same observations tick by tick."""
    hv = hv or _load(src)
    c, focus = payload["constants"], payload["focus"]
    fields = {
        "ingest_amber_s": c["IngestAmber"] * HOUR,
        "ingest_red_s": c["IngestRed"] * HOUR,
        "embedding_amber_s": c["EmbedAmber"] * HOUR,
        "embedding_red_s": c["EmbedRed"] * HOUR,
        "stores_amber_s": c["StoreAmber"] * HOUR,
        "stores_red_s": c["StoreRed"] * HOUR,
        "canary_red_failures": c["CanaryRed"],
        "sleep_gap_s": c["Gap"] * HOUR,
        "wake_grace_s": c["Grace"] * HOUR,
        "recovery_good_ticks": c["GoodTicks"],
        "unknown_amber_ticks": c["UnknownAmber"],
    }
    t = hv.Thresholds(**fields)

    def value(fact: list):
        kind = fact[0]
        if kind == "unobs":
            return hv.UNOBSERVED
        if kind == "none":
            return None
        if kind in ("bool", "str"):
            return fact[1]
        return BASE + dt.timedelta(hours=fact[1])

    prior = None
    out = []
    for op in payload["ops"]:
        now = BASE + dt.timedelta(hours=op["hour"])
        facts = {name: value(fact) for name, fact in op["facts"].items()}
        verdict = hv.evaluate(hv.Facts(now=now, **facts), prior, t)
        prior = verdict.state
        out.append({"hour": op["hour"], "model": op["model"], "src": verdict.check(focus).state})
    return out


def run_ingest(src: Path, payload: dict, hv: types.ModuleType | None = None) -> list[dict]:
    return run_pipe(src, {"focus": "ingest", **payload}, hv)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--run", "--run-pipe", "--run-weekly"):
        sys.exit(__doc__)
    payload = json.load(sys.stdin)
    if sys.argv[1] == "--run":
        print(json.dumps(run(Path(sys.argv[2]), payload)))
    elif sys.argv[1] == "--run-weekly":
        print(json.dumps(run_weekly(Path(sys.argv[2]), payload)))
    else:  # one payload carries every trace, so one interpreter replays them all
        hv = _load(Path(sys.argv[2]))
        traces = payload["traces"]
        print(json.dumps([run_pipe(Path(sys.argv[2]), {**payload, "ops": ops}, hv) for ops in traces]))
