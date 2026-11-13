#!/usr/bin/env python3
"""Conformance replay: the model's per-tick ingest verdict against the REAL status path.

Two halves:
- ``--generate`` runs each scenario through TLC (``Script`` mode, production thresholds) and writes
  the model's behaviour to ``traces/<scenario>.json``. ``--check-traces`` fails when those files
  are stale (check.py runs it).
- ``run_real`` drives the real code with one trace: a real ``JSONLWatcher`` over the real
  ``default_watch_roots`` of a scratch home, the real ``create_flush_callback`` into a scratch DB,
  the real ``watcher_fact`` and ``evaluate``. Each wired agent writes small JSONL in its own
  format. Per tick it compares the ingest state, what the monitor saw, and whether the newest
  record reached the DB. ``tests/test_ingest_coverage_replay.py`` runs it for every agent.

Faults are made real where the OS allows it: a write-only transcript (lag), a renamed root, a
read-only health directory (stale), a file rewritten in place, records in a shape no parser takes
(coverage). Three are stand-ins at a boundary: a dead or stuck watcher is "no completed poll", DB
failure raises from the flush callback, and an unreadable health file is one failed read.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TRACES = HERE / "traces"
TICK_S = 300
AGENTS = ("claude", "codex", "cursor", "cursor-agent-transcripts", "gemini", "gemini-agy")
# Replay agents that share a watcher provider name: Antigravity (agy), the current Gemini agent, is
# provider "gemini" on its own root.
_AGENT_ROOT = {"gemini-agy": ("gemini", ".gemini/antigravity-cli/brain")}
# Production thresholds in ticks: amber 600 s, red 1800 s, sleep gap 900 s, wake grace 7200 s.
MODEL = {
    "AmberT": 2,
    "RedT": 6,
    "UnknownAmber": 3,
    "Recovery": 2,
    "SleepGap": 3,
    "WakeGrace": 24,
    "AlertRed": 2,
    "Cap": 40,
    "N": 2,
}
SHAPES = (
    "dead",
    "stuck",
    "lag",
    "coverage",
    "dbfail",
    "blind_stale",
    "blind_unreadable",
    "rootgone",
    "rewrite",
    "rewrite_lossy",
)
DROPPING = frozenset({"coverage", "rewrite_lossy"})  # faults whose records are consumed and never indexed
PAD_RECORDS = 64


def _awake(f="none", o="watcherFirst"):
    return {"s": "awake", "f": f, "w": True, "o": o}


def _sleep(n):
    return [{"s": "sleep", "f": "none", "w": False, "o": "watcherFirst"}] * n


def _fault(f, ticks=9, after="none"):
    return [_awake()] * 2 + [_awake(f)] * ticks + [_awake(after)] * 4


SCENARIOS: dict[str, dict[str, Any]] = {
    **{s: {"clock": "confirmed", "events": _fault(s)} for s in SHAPES},
    "sleep": {"clock": "confirmed", "events": [_awake()] * 2 + _sleep(6) + [_awake()] * 3},
    "sleep-monitor-first": {
        "clock": "confirmed",
        "events": [_awake()] * 2 + _sleep(6) + [_awake(o="monitorFirst")] + [_awake()] * 2,
    },
    "stall-across-wake": {
        "clock": "confirmed",
        "events": [_awake()] * 2
        + [_awake("dead")] * 2
        + _sleep(6)
        + [{**_awake("dead"), "o": "monitorFirst"}]
        + [_awake("dead")] * 7
        + [_awake()] * 4,
    },
    "unconfirmed-short-sleep": {
        "clock": "unconfirmed",
        "events": [_awake()] * 2 + _sleep(2) + [_awake(o="monitorFirst")] + [_awake()] * 3,
    },
    "unconfirmed-long-sleep": {
        "clock": "unconfirmed",
        "events": [_awake()] * 2 + _sleep(6) + [_awake(o="monitorFirst")] + [_awake()] * 3,
    },
}


# --- model side ---------------------------------------------------------------------------------


def _tla_value(event: dict[str, Any]) -> str:
    w = "TRUE" if event["w"] else "FALSE"
    return f'[s |-> "{event["s"]}", f |-> "{event["f"]}", w |-> {w}, o |-> "{event["o"]}"]'


def model_trace(name: str, out: Path) -> dict[str, Any]:
    """The model's behaviour for one scenario, from TLC."""
    sys.path.insert(0, str(HERE))
    import check  # noqa: PLC0415 - sibling module, path-loaded

    scenario = SCENARIOS[name]
    events = scenario["events"]
    work = out / name
    work.mkdir(parents=True, exist_ok=True)
    module = f"MC_{name.replace('-', '_')}"
    (work / f"{module}.tla").write_text(
        f"---- MODULE {module} ----\nEXTENDS IngestCoverage\nReplayScript == <<\n  "
        + ",\n  ".join(_tla_value(e) for e in events)
        + "\n>>\n====\n"
    )
    consts = {
        **MODEL,
        "Rule": '"real"',
        "Mut": '"none"',
        "Clock": f'"{scenario["clock"]}"',
        "Faults": check.tla_set(("none", *SHAPES)),
    }
    cfg = work / f"{module}.cfg"
    cfg.write_text(
        "SPECIFICATION Spec\nCONSTANTS\n  Script <- ReplayScript\n"
        + "".join(f"  {k} = {v}\n" for k, v in consts.items())
        + "INVARIANTS ScriptRunning\n"
    )
    (work / "IngestCoverage.tla").write_text((HERE / "IngestCoverage.tla").read_text())
    cmd = check.tlc_command(cfg, work / "states")
    if cmd is None:
        raise RuntimeError("TLC not found: set TLC_JAR")
    cmd[-1] = module
    proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, check=False)
    log = proc.stdout + proc.stderr
    (work / "tlc.log").write_text(log)
    if "Invariant ScriptRunning is violated" not in log:
        raise RuntimeError(f"{name}: TLC did not run the script to its end; see {work / 'tlc.log'}")
    states = check.trace_states(log)
    if len(states) != len(events) + 1:
        raise RuntimeError(f"{name}: expected {len(events) + 1} states, TLC printed {len(states)}")
    keep = ("v", "obs", "backlog", "stall", "hbA", "pA")
    ticks = [{k: json.loads(s[k].replace("TRUE", "true").replace("FALSE", "false")) for k in keep} for s in states[1:]]
    return {"scenario": name, "clock": scenario["clock"], "model": MODEL, "events": events, "ticks": ticks}


def generate(out: Path) -> dict[str, dict[str, Any]]:
    return {name: model_trace(name, out) for name in SCENARIOS}


def _dump(trace: dict[str, Any]) -> str:
    return json.dumps(trace, indent=1, sort_keys=True) + "\n"


def write_traces(out: Path) -> None:
    TRACES.mkdir(exist_ok=True)
    for name, trace in generate(out).items():
        (TRACES / f"{name}.json").write_text(_dump(trace))


def stale_traces(out: Path) -> list[str]:
    fresh = generate(out)
    stale = [
        n
        for n, t in fresh.items()
        if not (TRACES / f"{n}.json").is_file() or (TRACES / f"{n}.json").read_text() != _dump(t)
    ]
    return stale + [p.stem for p in TRACES.glob("*.json") if p.stem not in fresh]


def load_trace(name: str) -> dict[str, Any]:
    return json.loads((TRACES / f"{name}.json").read_text())


# --- real side ----------------------------------------------------------------------------------


class SimClock(datetime):
    """``brainlayer.watcher.datetime`` during a replay: ``now()`` is the simulated tick time."""

    current: datetime = datetime(2026, 11, 13, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):  # noqa: D102
        return cls.current if tz is None else cls.current.astimezone(tz)


_RELATIVE = {
    "claude": "-Users-example-proj/5b1e2c1a-0000-4000-8000-00000000r001.jsonl",
    "codex": "2026/11/13/rollout-2026-11-13T00-00-00-0199replay.jsonl",
    "cursor": "replay-session.jsonl",
    "cursor-agent-transcripts": "example-proj/agent-transcripts/replay-1.jsonl",
    "gemini": "replay-session.jsonl",
    "gemini-agy": "0115a286-0000-4000-8000-00000000a001/.system_generated/logs/transcript_full.jsonl",
}


def agent_record(provider: str, text: str, when: datetime, *, drift: bool = False) -> dict[str, Any]:
    """One user turn in the agent's own transcript format (what the watcher's parsers read)."""
    ts = when.isoformat().replace("+00:00", "Z")
    if drift:  # a shape no parser takes: format drift after an agent update
        return {"kind": "turn", "body": text, "at": ts}
    if provider == "claude":
        return {
            "type": "user",
            "message": {"role": "user", "content": text},
            "timestamp": ts,
            "sessionId": "5b1e2c1a-0000-4000-8000-00000000r001",
            "cwd": "/srv/example/proj",
        }
    if provider == "codex":
        return {
            "timestamp": ts,
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
        }
    if provider == "cursor-agent-transcripts":
        return {"role": "user", "message": {"content": [{"type": "text", "text": text}]}}
    if provider == "gemini-agy":
        return {
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "status": "DONE",
            "created_at": ts,
            "content": f"<USER_REQUEST>\n{text}\n</USER_REQUEST>",
        }
    return {"role": "user", "content": text, "timestamp": ts}  # cursor sessions, gemini sessions


def _header(provider: str, when: datetime) -> list[dict[str, Any]]:
    if provider != "codex":
        return []
    ts = when.isoformat().replace("+00:00", "Z")
    return [
        {
            "timestamp": ts,
            "type": "session_meta",
            "payload": {"id": "0199replay", "cwd": "/srv/example/proj", "model_provider": "openai"},
        }
    ]


def _text(provider: str, tick: int, variant: str = "") -> tuple[str, str]:
    marker = f"rpl{provider.replace('-', '')}t{tick}{variant}x"
    return marker, (
        f"Replay turn {tick} for {provider}: {marker} please check why the nightly backup "
        "verification compares the staged archive digest against the uploaded copy first."
    )


def _db_has(db: Path, marker: str) -> bool:
    if not db.exists():
        return False
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return (
            conn.execute("SELECT 1 FROM chunks WHERE content LIKE ? LIMIT 1", (f"%{marker}%",)).fetchone() is not None
        )
    finally:
        conn.close()


def run_real(provider: str, trace: dict[str, Any], work: Path) -> list[dict[str, Any]]:
    """Drive the real watcher + verdict through ``trace``; one observation per event."""
    import brainlayer.watcher as watcher_module
    from brainlayer.health_facts import watcher_fact
    from brainlayer.health_verdict import UNOBSERVED, Facts, Thresholds, evaluate
    from brainlayer.watcher import JSONLWatcher, default_watch_roots
    from brainlayer.watcher_bridge import create_flush_callback

    home, db, health = work / "home", work / "db" / "brainlayer.db", work / "health" / "watcher-health.json"
    for d in (home, db.parent, health.parent):
        d.mkdir(parents=True, exist_ok=True)
    root = _watch_root(provider, default_watch_roots(home), home)
    moved = root.with_name(root.name + "-moved")
    transcript = root / _RELATIVE[provider]
    transcript.parent.mkdir(parents=True, exist_ok=True)
    state = {"dbfail": False}
    real_flush = create_flush_callback(db_path=db, arbitrated=False)

    def on_flush(items):
        if state["dbfail"]:
            raise sqlite3.OperationalError("database is locked (replay: DB writes fail)")
        return real_flush(items)

    original_datetime = watcher_module.datetime
    watcher_module.datetime = SimClock
    try:
        start = datetime(2026, 11, 13, 9, 0, tzinfo=UTC)
        SimClock.current = start
        lines = [*_header(provider, start)]
        marker0, text0 = _text(provider, 0)
        lines.append(agent_record(provider, text0, start))
        if trace["scenario"] == "rewrite_lossy":  # long enough that a middle record is past the 4 KiB head digest
            lines += [agent_record(provider, _text(provider, 0, f"p{i:03d}")[1], start) for i in range(PAD_RECORDS)]
        transcript.write_text("".join(json.dumps(x) + "\n" for x in lines))
        if provider == "gemini-agy":  # agy summarises each conversation; the row carries its agent attribution
            store = root.parent / "conversation_summaries.db"
            with sqlite3.connect(store) as conn:
                conn.execute("CREATE TABLE conversation_summaries (conversation_id TEXT PRIMARY KEY, agent_name TEXT)")
                conn.execute("INSERT INTO conversation_summaries VALUES (?, '')", (transcript.parents[2].name,))
            conn.close()
        # Every marker the agent wrote in an indexable form. A marker written under a fault that drops
        # records (format drift, an edit the fingerprint cannot see) counts only while that fault lasts.
        pending: dict[str, str] = {marker0: "none"}
        watcher = JSONLWatcher(
            watch_roots=default_watch_roots(home),
            registry_path=work / "offsets.json",
            on_flush=on_flush,
            db_path=db,
            batch_size=1,
            flush_interval_ms=0,
            registry_flush_interval_s=0,
            health_path=health,
        )
        confirmed = trace["clock"] == "confirmed"
        wall, awake = start, 0.0

        def facts_now(updated):
            return Facts(
                now=wall,
                watcher_updated_at=updated,
                awake_clock_s=awake if confirmed else None,
                boot_session="replay-boot" if confirmed else None,
            )

        def read():
            try:
                value = watcher_fact(health)
            except Exception:
                return UNOBSERVED
            return value

        watcher.poll_once()  # Init: a heartbeat at t0 and one verdict tick on it (the model's InitSteady)
        verdict = evaluate(facts_now(read()), None, Thresholds())
        prior, current = verdict.state, verdict.check("ingest").state
        fault, asleep = "none", False

        def backlog() -> bool:
            for marker in [m for m, f in pending.items() if _db_has(db, m)]:
                del pending[marker]
            return any(f not in DROPPING or f == fault for f in pending.values())

        out = []
        for k, event in enumerate(trace["events"], 1):
            wall += timedelta(seconds=TICK_S)
            SimClock.current = wall
            if event["s"] == "sleep":
                asleep = True
                out.append({"v": current, "obs": "asleep", "backlog": backlog()})
                continue
            awake += TICK_S
            f = event["f"]
            if fault != f:  # leave the old fault, enter the new one
                if fault == "lag":
                    _chmod(watcher_path(transcript, root, moved), 0o644)
                if fault == "rootgone":
                    moved.rename(root)
                if fault == "blind_stale":
                    health.parent.chmod(0o755)
                if f == "lag":
                    _chmod(watcher_path(transcript, root, moved), 0o200)
                if f == "rootgone":
                    root.rename(moved)
                if f == "blind_stale":
                    health.parent.chmod(0o555)
                fault = f
            state["dbfail"] = f == "dbfail"
            path = watcher_path(transcript, root, moved)
            marker, text = _text(provider, k)
            if f == "rewrite":  # rewritten in place, longer than before, under the stored offset
                body = [*_header(provider, wall)]
                for i in range(len(lines) - len(body)):
                    variant, rewritten = _text(provider, k, f"r{i}")
                    body.append(agent_record(provider, rewritten, wall))
                    pending[variant] = "none"  # every rewritten record must be re-ingested
                body.append(agent_record(provider, text, wall))
                with open(path, "r+", encoding="utf-8") as handle:
                    handle.seek(0)
                    handle.write("".join(json.dumps(x) + "\n" for x in body))
                    handle.truncate()
                lines = body
                pending[marker] = "none"
            elif event["w"]:
                if f == "rewrite_lossy":  # same-length edit of a middle record, before the appended turn
                    edited, _ = _edit_middle_record(path, provider, lines, k, start)
                    pending[edited] = f
                record = agent_record(provider, text, wall, drift=f == "coverage")
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                lines.append(record)
                pending[marker] = "coverage" if f == "coverage" else "none"
            polls = f not in ("dead", "stuck")
            monitor_first = asleep and event["o"] == "monitorFirst"
            if polls and not monitor_first:
                watcher.poll_once()
            if f == "blind_unreadable":  # this tick's read fails; the file itself is left intact
                updated = UNOBSERVED
            else:
                updated = read()
            if polls and monitor_first:
                watcher.poll_once()
            asleep = False
            verdict = evaluate(facts_now(updated), prior, Thresholds())
            prior, current = verdict.state, verdict.check("ingest").state
            if updated is UNOBSERVED:
                seen = "unreadable"
            elif updated is None:
                seen = "missing"
            else:
                seen = "fresh" if updated == wall else "old"
            out.append({"v": current, "obs": seen, "backlog": backlog()})
        if fault == "lag":
            _chmod(watcher_path(transcript, root, moved), 0o644)
        if fault == "blind_stale":
            health.parent.chmod(0o755)
        return out
    finally:
        watcher_module.datetime = original_datetime


def _edit_middle_record(path: Path, provider: str, lines: list, tick: int, when: datetime) -> tuple[str, int]:
    """Rewrite one padding record in place with a same-length record carrying a new marker.

    The file's first 4 KiB and the bytes just before the watcher's offset are untouched, which is
    exactly what the #732 resume fingerprint compares.
    """
    index = len(_header(provider, when)) + 1 + PAD_RECORDS * 5 // 8 + tick % 8
    old = json.dumps(lines[index]) + "\n"
    marker, text = _text(provider, 0, f"L{tick:03d}")
    new_record = agent_record(provider, text, when)
    new = json.dumps(new_record) + "\n"
    if len(new.encode()) != len(old.encode()):
        raise AssertionError("lossy rewrite must keep the record length")
    offset = sum(len((json.dumps(x) + "\n").encode()) for x in lines[:index])
    if offset < 4096:
        raise AssertionError("lossy rewrite must sit past the 4 KiB head digest")
    with open(path, "r+b") as handle:
        handle.seek(offset)
        handle.write(new.encode())
    lines[index] = new_record
    return marker, offset


def _watch_root(agent: str, roots: list, home: Path) -> Path:
    """The real watch root an agent writes under (default_watch_roots, matched by provider and path)."""
    provider, subdir = _AGENT_ROOT.get(agent, (agent, None))
    matches = [r.resolved_path for r in roots if r.provider == provider]
    if subdir is not None:
        matches = [path for path in matches if path == home / subdir]
    else:
        matches = matches[:1]
    if len(matches) != 1:
        raise AssertionError(f"no single watch root for {agent}")
    return matches[0]


def watcher_path(transcript: Path, root: Path, moved: Path) -> Path:
    """Where the agent writes now: under the moved root while the root is gone."""
    return transcript if root.exists() else moved / transcript.relative_to(root)


def _chmod(path: Path, mode: int) -> None:
    path.chmod(mode | stat.S_IWUSR)


def mismatches(trace: dict[str, Any], real: list[dict[str, Any]]) -> list[str]:
    bad = []
    for k, (event, want, got) in enumerate(zip(trace["events"], trace["ticks"], real, strict=True), 1):
        model = {"v": want["v"], "obs": want["obs"], "backlog": want["backlog"]}
        if model != got:
            bad.append(f"tick {k} {event['s']}/{event['f']}: model {model} real {got}")
    return bad


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generate", action="store_true", help="write traces/ from TLC")
    ap.add_argument("--check-traces", action="store_true", help="fail when traces/ is stale")
    ap.add_argument("--out", type=Path, default=ROOT / ".build.noindex" / "ingest-coverage" / "replay")
    args = ap.parse_args(argv)
    if args.generate:
        write_traces(args.out)
        print(f"wrote {len(SCENARIOS)} traces to {TRACES.relative_to(ROOT)}")
        return 0
    if args.check_traces:
        stale = stale_traces(args.out)
        print("TRACES STALE (run replay.py --generate): " + ", ".join(stale) if stale else "traces current")
        return 1 if stale else 0
    sys.path.insert(0, str(ROOT / "src"))
    failed = 0
    for name in SCENARIOS:
        for provider in AGENTS:
            bad = mismatches(
                load_trace(name), run_real(provider, load_trace(name), args.out / "real" / name / provider)
            )
            failed += bool(bad)
            print(f"{'OK  ' if not bad else 'FAIL'} {name:26} {provider}" + "".join(f"\n     {b}" for b in bad))
    return 1 if failed else 0


if __name__ == "__main__":
    os.umask(0o022)
    sys.exit(main())
