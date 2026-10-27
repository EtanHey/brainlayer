"""Bounded TLC checks. No product imports, processes, sockets, or production data."""

import argparse
import ast
import asyncio
import datetime as dt
import hashlib
import inspect
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CALENDAR = "src/brainlayer/maintenance_calendar.py"
GUARD = "src/brainlayer/supervisor_guard.py"
CURRENT = "a9c3f184821006dd29b40f2758bee1ba2df4afc5"
MAIN = "f01469cfdac1368d54a9f9d47bf743f5fc49ca8d"
HISTORIES = {
    "pre-1519": MAIN,
    "pre-1517": "54c580ca6849c2d32628f9ad38d924c54109f055",
    "1519-r1": "b8e476db50ed77045f53d0212922637f1df72384",
}


def source(ref, path=CALENDAR):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=REPO, text=True)


def extract(text, names, namespace):
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), CALENDAR, "exec"), namespace)
    return {n.name: n for n in nodes}


def facts(text):
    """Execute the actual pulse body once below and once above its deadline."""
    pulses = []
    stop = SimpleNamespace(is_set=lambda: False)

    async def sleep(_):
        stop.is_set = lambda: True

    ns = dict(
        asyncio=SimpleNamespace(sleep=sleep),
        time=SimpleNamespace(monotonic=lambda: 10),
        os=SimpleNamespace(write=lambda *_: pulses.append("H"), getpid=lambda: 42),
        start_time=lambda _: "birth",
        logging=logging,
    )
    extract(text, {"pulse"}, ns)
    guard = SimpleNamespace(poll=lambda: None, pid=43, stdin=SimpleNamespace(fileno=lambda: 99))
    signature = inspect.signature(ns["pulse"])

    def sample(started):
        pulses.clear()
        stop.is_set = lambda: False
        kwargs = {
            "operations": {1: SimpleNamespace(started=started, reported=False, label="probe")},
            "operation_timeout": 4,
            "diagnostics": SimpleNamespace(emit=lambda **_: None),
        }
        asyncio.run(ns["pulse"](guard, stop, 0.1, **{k: v for k, v in kwargs.items() if k in signature.parameters}))
        return len(pulses)

    healthy, overdue = sample(9), sample(0)
    if healthy != 1 or overdue not in (0, 1):
        raise RuntimeError("pulse probe contract changed; resync the model")
    tree = ast.parse(text)
    functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    offloop = "_blocking" in functions and any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_blocking"
        for n in ast.walk(functions["_execute_step"])
    )
    scan = functions.get("_recover_abandoned")
    skip_torn = scan is not None and any(
        "ValueError" in ast.unparse(n.type)
        for n in ast.walk(scan)
        if isinstance(n, ast.ExceptHandler) and n.type is not None
    )
    return dict(OffLoop=offloop, Budgeted=overdue == 0, DetectAbandoned=scan is not None, SkipTorn=skip_torn)


def journal_probe(text, output):
    """Actual record/scan bodies on private temporary files; no module imports."""
    ns = dict(
        json=json,
        time=time,
        dt=dt,
        math=math,
        os=os,
        LOG=logging.getLogger("calendar-model-probe"),
        NAMES=("index",),
        start_time=lambda _: None,
        KEYS={},
        _atomic_write=lambda path, state: path.write_text(json.dumps(state)),
        _abandonment_alerts=lambda event: {},
        _retire_incomplete=lambda *_: None,
    )
    extract(text, {"record", "_scan_record", "_recover_abandoned"}, ns)
    results = {}
    with tempfile.TemporaryDirectory(dir=output) as temp:
        for scenario in ("nightly", "recovery", "late", "two-pending"):
            kind = "recovery" if scenario == "recovery" else "nightly"
            directory = Path(temp) / scenario
            directory.mkdir()
            event = "runner" if kind == "nightly" else "transcript-recovery"
            ns["record"](
                directory,
                event=event,
                kind=kind,
                ts=1,
                scheduled_date="2026-10-27",
                steps=["index"],
                guard=424242,
                guard_start_time="old-birth",
            )
            with (directory / "runs.jsonl").open("ab") as handle:
                handle.write(b'{"event":')
            ns["record"](directory, event="spawn", step="index", pid=424243, start_time="old-birth")
            if scenario == "two-pending":
                ns["record"](
                    directory,
                    event="runner",
                    kind=kind,
                    ts=2,
                    scheduled_date="2026-10-27",
                    steps=["index"],
                    guard=424242,
                    guard_start_time="old-birth",
                )
            state = {"deferrals": {}}
            for _ in range(2):
                ns["_recover_abandoned"](
                    directory, state, [], today=dt.date(2026, 10, 29 if scenario == "late" else 27)
                )
            rows = []
            for line in (directory / "runs.jsonl").read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
            results[scenario] = {e: sum(r["event"] == e for r in rows) for e in ("spawn", "abandoned", "summary")}
            if scenario == "two-pending":
                results[scenario]["abandoned_identities"] = [
                    r["runner_started_at"] for r in rows if r["event"] == "abandoned"
                ]
    expected = {
        "nightly": {"spawn": 1, "abandoned": 1, "summary": 1},
        "recovery": {"spawn": 1, "abandoned": 0, "summary": 0},
        "late": {"spawn": 1, "abandoned": 0, "summary": 0},
        "two-pending": {"spawn": 1, "abandoned": 1, "summary": 1, "abandoned_identities": [2]},
    }
    if results != expected:
        raise RuntimeError(f"journal probe drift: {results}; resync before changing expectations")
    return results


def guard_probe(text):
    """Execute real guard deadline and recovery bodies with fake OS boundaries."""
    clock, signals, reaped = [0], [], []
    messages = [b"H\n", b"Q\n", b"H\n"]

    def select(_, __, ___, remaining):
        clock[0] += 1 if messages else remaining
        return ([0] if messages else [], [], [])

    ns = dict(
        time=SimpleNamespace(monotonic=lambda: clock[0]),
        os=SimpleNamespace(
            getppid=lambda: 42,
            getpid=lambda: 43,
            write=lambda *_: None,
            close=lambda _: None,
            read=lambda *_: messages.pop(0),
            kill=lambda pid, sig: signals.append((pid, sig, clock[0])),
        ),
        signal=SimpleNamespace(SIGKILL=9),
        select=SimpleNamespace(select=select),
        start_time=lambda _: "new-birth",
        emit=lambda *_, **__: None,
        reap_groups=lambda groups, *_, **__: reaped.append(sorted(groups)),
    )
    extract(text, {"run_guard"}, ns)
    ns["run_guard"](42, 2, 99, 0.1, shutdown_timeout=1)
    if signals != [(42, 9, 5)] or reaped != [[]]:
        raise RuntimeError(f"H/Q contract drift: {signals}, {reaped}")
    custody = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "Custody")
    recover = next(n for n in custody.body if isinstance(n, ast.FunctionDef) and n.name == "recover")
    ns.update(group_exists=lambda _: True)
    exec(compile(ast.Module(body=[recover], type_ignores=[]), GUARD, "exec"), ns)
    reaped.clear()
    owner = SimpleNamespace(groups={"44": {"pgid": 44, "start_time": "old-birth"}}, _save=lambda: None)
    ns["recover"](owner, 0.1)
    if reaped != [[]]:
        raise RuntimeError("PID reuse reached reap_groups")
    return dict(Q_deadline=5, repeated_H_extends_Q=False, reused_PID_reaped=False)


def case(name, module, constants, invariants, properties=(), violation=None):
    return dict(
        name=name, module=module, constants=constants, invariants=invariants, properties=properties, violation=violation
    )


def matrix(current, histories):
    g = dict(
        OffLoop=current["OffLoop"],
        Budgeted=current["Budgeted"],
        Fault="custody",
        Mode="healthy",
        Kill="none",
        OperationBudget=4,
        HeartbeatBudget=2,
        Horizon=40,
    )
    j = dict(
        DetectAbandoned=current["DetectAbandoned"],
        SkipTorn=current["SkipTorn"],
        BirthChecked=True,
        Kind="nightly",
        Crash=True,
        Torn=True,
        Reuse=True,
        Late=False,
        TwoPending=False,
        LostStart=False,
    )
    gi = ["TypeOK", "ExactlyOneChild", "NoFalseKill", "BoundedTermination"]
    ji = ["TypeOK", "AtMostOneTerminal", "PidReuseSafe", "Custody"]
    cases = [case("current-healthy", "CalendarGuard", g, gi, ["EventuallyReleased"])]
    for phase in ("runner", "custody", "finish"):
        cases.append(
            case(
                f"current-wedge-{phase}",
                "CalendarGuard",
                g | dict(Mode="wedge", Fault=phase),
                gi,
                ["EventuallyReleased"],
            )
        )
    cases += [
        case("current-loop-wedge", "CalendarGuard", g | dict(Mode="loop", Fault="custody"), gi, ["EventuallyReleased"]),
        case("current-q-timeout", "CalendarGuard", g | dict(Mode="q-stall"), gi, ["EventuallyReleased"]),
        case("current-parent-killed", "CalendarGuard", g | dict(Kill="parent"), gi, ["EventuallyReleased"]),
        case("current-journal", "CalendarJournal", j, ji, ["TornJournalLive"]),
        case("current-completed", "CalendarJournal", j | dict(Crash=False), ji, ["TornJournalLive"]),
    ]
    for phase in ("pre", "summary"):
        cases.append(
            case(
                f"finding-wedge-{phase}",
                "CalendarGuard",
                g | dict(Mode="wedge", Fault=phase),
                gi,
                violation="BoundedTermination",
            )
        )
    for name, changes in (
        ("recovery", dict(Kind="recovery")),
        ("late", dict(Late=True)),
        ("two-pending", dict(TwoPending=True)),
        ("lost-start", dict(LostStart=True)),
    ):
        cases.append(case(f"finding-{name}", "CalendarJournal", j | changes, ji, violation="Custody"))
    cases.append(
        case("registration-ack", "CalendarGuard", g, ["RegistrationAcknowledged"], violation="RegistrationAcknowledged")
    )
    cases.append(
        case(
            "finding-terminal-record-count",
            "CalendarJournal",
            j,
            ["TerminalRecordUnique"],
            violation="TerminalRecordUnique",
        )
    )
    cases.append(
        case(
            "finding-guard-killed",
            "CalendarGuard",
            g | dict(Kill="guard", Mode="wedge"),
            gi,
            violation="BoundedTermination",
        )
    )
    cases.append(
        case(
            "historical-sync",
            "CalendarGuard",
            g | dict(OffLoop=histories["pre-1519"]["OffLoop"]),
            ["NoFalseKill"],
            violation="NoFalseKill",
        )
    )
    cases.append(
        case(
            "historical-missing-terminal",
            "CalendarJournal",
            j | dict(DetectAbandoned=histories["pre-1517"]["DetectAbandoned"]),
            ji,
            violation="Custody",
        )
    )
    cases.append(
        case(
            "historical-worker-wedge",
            "CalendarGuard",
            g | dict(Budgeted=histories["1519-r1"]["Budgeted"], Mode="wedge"),
            gi,
            violation="BoundedTermination",
        )
    )
    cases.append(
        case(
            "historical-worker-liveness",
            "CalendarGuard",
            g | dict(Budgeted=histories["1519-r1"]["Budgeted"], Mode="wedge"),
            ["TypeOK", "ExactlyOneChild"],
            ["EventuallyReleased"],
            "EventuallyReleased",
        )
    )
    return cases


def run_tlc(entry, output):
    constants = "\n".join(
        f"  {k} = {str(v).upper() if isinstance(v, bool) else json.dumps(v)}" for k, v in entry["constants"].items()
    )
    cfg = output / f"{entry['name']}.cfg"
    cfg.write_text(
        "SPECIFICATION Spec\nCHECK_DEADLOCK FALSE\nCONSTANTS\n"
        + constants
        + "\nINVARIANTS "
        + " ".join(entry["invariants"])
        + "\n"
        + ("PROPERTIES " + " ".join(entry["properties"]) + "\n" if entry["properties"] else "")
    )
    command = [
        str(Path.home() / ".local/bin/tlc"),
        "-workers",
        "1",
        "-fp",
        "0",
        "-metadir",
        str(output / "states" / entry["name"]),
        "-config",
        str(cfg),
        entry["module"] + ".tla",
    ]
    began = time.monotonic()
    result = subprocess.run(
        command,
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=60,
        env=dict(os.environ, JAVA_TOOL_OPTIONS="-Xmx512m"),
    )
    log = result.stdout + result.stderr
    (output / f"{entry['name']}.log").write_text(log)
    count = re.search(r"([\d,]+) states generated, ([\d,]+) distinct states found", log)
    depth = re.search(r"The depth of the complete state graph search is (\d+)", log)
    violated = re.search(r"Invariant (\w+) is violated", log)
    expected = entry["violation"]
    temporal = expected in entry["properties"] and "Temporal properties were violated" in log
    progress = re.findall(r"Progress\((\d+)\)", log)
    row = dict(
        name=entry["name"],
        rc=result.returncode,
        violation=violated[1] if violated else expected if temporal else None,
        states=int(count[2].replace(",", "")) if count else None,
        depth=int(depth[1]) if depth else int(progress[-1]) if progress else None,
        wall_s=round(time.monotonic() - began, 3),
    )
    row["reproduced"] = (
        result.returncode == 0
        if expected is None
        else result.returncode == (13 if temporal else 12) and row["violation"] == expected
    )
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-known", action="store_true", help="accept documented findings; never calls them PASS")
    parser.add_argument("--scratch-break", action="store_true", help="mutate real pulse in a private scratch copy")
    args = parser.parse_args()
    output = REPO / ".build.noindex/calendar-guard" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output.mkdir(parents=True, exist_ok=False)
    text = source(CURRENT)
    coverage = json.loads((HERE / "coverage.json").read_text())
    for path, entry in coverage["sources"].items():
        actual = hashlib.sha256(source(entry["sha"], path).encode()).hexdigest()
        if actual != entry["sha256"]:
            raise RuntimeError(f"source fingerprint mismatch: {path}")
        accepted = {entry["sha256"], hashlib.sha256(source(MAIN, path).encode()).hexdigest()}
        if hashlib.sha256((REPO / path).read_bytes()).hexdigest() not in accepted:
            raise RuntimeError(f"checkout drift: {path}; review the model and update coverage.json")
    for module, actions in coverage["actions"].items():
        model = (HERE / f"{module}.tla").read_text()
        for action, sites in actions.items():
            if not re.search(rf"^{action}\s*==", model, re.MULTILINE):
                raise RuntimeError(f"missing modeled action: {module}.{action}")
            for site in sites:
                path, lines = site.split(":")
                if int(lines.split("-")[-1]) > len(source(coverage["sources"][path]["sha"], path).splitlines()):
                    raise RuntimeError(f"invalid source pointer: {site}")
    current = facts(text)
    histories = {name: facts(source(ref)) for name, ref in HISTORIES.items()}
    if current != dict(OffLoop=True, Budgeted=True, DetectAbandoned=True, SkipTorn=True):
        raise RuntimeError(f"source projection drift: {current}")
    probes = dict(
        current=current,
        histories=histories,
        journal=journal_probe(text, output),
        guard=guard_probe(source(MAIN, GUARD)),
    )
    entries = matrix(current, histories)
    if args.scratch_break:
        scratch = output / "scratch" / CALENDAR
        scratch.parent.mkdir(parents=True)
        broken = text.replace("if not overdue:\n", "if True:  # scratch: disable operation budget\n")
        if text.count("if not overdue:\n") != 1:
            raise RuntimeError("scratch break must replace exactly one guard")
        scratch.write_text(broken)
        shutil.copy2(HERE / "CalendarGuard.tla", output / "scratch/CalendarGuard.tla")
        projected = facts(scratch.read_text())
        probes["scratch"] = dict(
            facts=projected,
            original_sha256=hashlib.sha256(text.encode()).hexdigest(),
            scratch_sha256=hashlib.sha256(broken.encode()).hexdigest(),
        )
        if projected["Budgeted"]:
            raise RuntimeError("real pulse scratch break escaped the probe")
        base = entries[1]
        entries.append(
            case(
                "scratch-worker-wedge",
                "CalendarGuard",
                base["constants"] | dict(Budgeted=projected["Budgeted"]),
                base["invariants"],
                violation="BoundedTermination",
            )
        )
    (output / "probes.json").write_text(json.dumps(probes, indent=2) + "\n")
    print(f"Source: calendar={CURRENT} guard={MAIN}; TLC=1.7.4", flush=True)
    rows = [run_tlc(entry, output) for entry in entries]
    (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Evidence: {output}", flush=True)
    if not all(row["reproduced"] for row in rows):
        return 2
    print("Known counterexamples reproduced; current product is NOT_GREEN.")
    return 0 if args.expect_known else 1


if __name__ == "__main__":
    raise SystemExit(main())
