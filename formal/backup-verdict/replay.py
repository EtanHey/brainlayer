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
"""

from __future__ import annotations
import __future__

import ast
import datetime as dt
import importlib.util
import json
import sys
import types
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


def ops_from_trace(states: list[tuple[str, dict[str, str]]]) -> list[dict]:
    """Receipts appended and ticks taken, in trace order, from TLC's state sequence."""
    ops: list[dict] = []
    previous: dict[str, str] | None = None
    for action, values in states:
        if previous is not None and action == "World":
            before, after = _tla_seq(previous["lastLoc"]), _tla_seq(values["lastLoc"])
            created = [int(x) for x in _tla_seq(values["created"])]
            snap = next(i for i, (a, b) in enumerate(zip(before, after), 1) if a != b)
            exact = '"approx"' not in values.get("acc", "") or '"approx"' in previous.get("acc", "")
            ops.append(
                {
                    "op": "receipt",
                    "hour": int(values["now"]),
                    "kind": after[snap - 1],
                    "snap": snap,
                    "created": created[snap - 1],
                    "exact": exact,
                }
            )
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
            receipts.append(receipt(op))
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


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--run":
        sys.exit(__doc__)
    print(json.dumps(run(Path(sys.argv[2]), json.load(sys.stdin))))
