#!/usr/bin/env python3
"""Independent BrainLayer health relay: per-host evidence -> primary inbox + optional escalation.

Runs from its own LaunchAgent on the inbox's Mac, outside the BrainLayer supervisor, so it still
observes when that supervisor is dead. Remote hosts are read over an existing SSH alias with a
fixed reader program on stdin. Stdlib only: it never imports BrainLayer, opens a database, reads
credentials, heals, or runs a producer. It reads job-alerts, the badge, watcher health, the
supervisor job manifest and the pause sentinel, and nothing else.

Delivery is at-least-once with whole-line dedup. Pending events are persisted before any append.
Every attempt appends one self-contained record in a single write, so no record depends on the
bytes before it: other inbox writers do not take `relay.lock`, and may append at any moment. An
event is acknowledged only when, after an inbox fsync in the same run, the inbox holds its exact
BLOCKED line and its marker as whole lines. Torn fragments are left in place and never counted.
"""

from __future__ import annotations

import argparse
import fcntl
import inspect
import json
import math
import os
import re
import selectors
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,31}$")
SAFE_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SAFE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]{1,255}$")  # joined into the remote shell's command line
SSH_OPTIONS = (
    "BatchMode=yes",
    "StrictHostKeyChecking=yes",
    "UpdateHostKeys=no",
    "ConnectTimeout=10",
    "ServerAliveInterval=5",
    "ServerAliveCountMax=2",
)
SOURCES = {
    "job-alerts": "job-alerts.json",
    "badge": "badge-state.json",
    "watcher": "watcher-health.json",
    "jobs": "supervisor/jobs.json",
    "pause": "pause.sentinel",
}
READ_STATES = ("ok", "absent", "unreadable", "oversize", "corrupt")
HEALTH_LABEL, WATCH_LABEL = "com.brainlayer.health-check", "com.brainlayer.watch"
MAX_SOURCE_BYTES, MAX_OUTPUT_BYTES, MAX_DELIVER_PER_RUN, MAX_CLOSED = 262144, 2097152, 32, 128
CLOCK_SKEW_SECONDS = 300
ESCALATION_ISSUES = (
    "host-unreachable",
    "health-stale",
    "watcher-stale",
    "job-alert:backup-daily",
    "job-alert:jsonl-backup",
    "badge:brain-search-canary-failed",
    "source-missing:badge",
)
DEFAULTS = {
    "first_run_grace_seconds": 900,
    "badge_max_age_seconds": 1200,
    "watcher_max_age_seconds": 600,
    "unreachable_after": 3,
    "ssh_timeout_seconds": 45,
}


class RelayError(Exception):
    """Carries a fixed classification only; never a raw reason, path or exception text."""


def _read_sources(paths, max_bytes):
    import os
    import stat
    import time

    out = {}
    for name, path in paths.items():
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            out[name] = {"s": "absent"}
            continue
        except OSError:
            out[name] = {"s": "unreadable"}
            continue
        try:
            with os.fdopen(fd, "rb") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    out[name] = {"s": "unreadable"}
                    continue
                data = handle.read(max_bytes + 1)
            out[name] = {"s": "oversize"} if len(data) > max_bytes else {"s": "ok", "text": data.decode("utf-8")}
        except UnicodeDecodeError:
            out[name] = {"s": "corrupt"}
        except OSError:
            out[name] = {"s": "unreadable"}
    return {"now": time.time(), "sources": out}


def _abs(value) -> str:
    if not isinstance(value, str) or not os.path.isabs(value) or re.search(r"[\x00-\x1f\x7f]", value):
        raise RelayError("config-invalid")
    return value


def _match(pattern, value) -> str:
    if not isinstance(value, str) or re.search(r"[\x00-\x1f\x7f-\x9f]", value) or not pattern.fullmatch(value):
        raise RelayError("config-invalid")
    return value


def load_config(path: Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = {key: raw.get(key, default) for key, default in DEFAULTS.items()}
    if any(type(value) is not int or not 0 <= value <= 2**31 - 1 for value in cfg.values()):
        raise RelayError("config-invalid")
    cfg.update(inbox=_abs(raw["inbox"]), state_dir=_abs(raw["state_dir"]), hosts=[])
    if "escalation" in raw:
        escalation = raw["escalation"]
        if not isinstance(escalation, dict) or "inbox" not in escalation:
            raise RelayError("config-invalid")
        after = escalation.get("after_seconds", 1800)
        patterns = escalation.get("issues", list(ESCALATION_ISSUES))
        if (
            type(after) is not int
            or not 0 <= after <= 2**31 - 1
            or not isinstance(patterns, list)
            or not all(isinstance(p, str) for p in patterns)
        ):
            raise RelayError("config-invalid")
        try:
            cfg["escalation"] = {
                "inbox": _abs(escalation["inbox"]),
                "after_seconds": after,
                "issues": [re.compile(p) for p in patterns],
            }
        except re.error:
            raise RelayError("config-invalid") from None
    inbox, state_dir = Path(cfg["inbox"]), Path(cfg["state_dir"])
    destinations = [state_dir / name for name in ("state.json", "status.json", "relay.lock")]
    destinations += [_atomic_temp(path) for path in destinations[:2]]
    for item in raw["hosts"]:
        host = {"id": _match(SAFE_ID, item["id"]), "label": _match(SAFE_LABEL, item["label"])}
        base, overrides = _abs(item["dir"]), item.get("sources", {})
        if not isinstance(overrides, dict) or set(overrides) - set(SOURCES):
            raise RelayError("config-invalid")
        host["paths"] = {name: _abs(overrides.get(name, os.path.join(base, rel))) for name, rel in SOURCES.items()}
        host["transport"] = item["transport"]
        if host["transport"] == "ssh":
            host["ssh_alias"] = _match(SAFE_ALIAS, item["ssh_alias"])
            host["ssh_bin"] = _abs(item.get("ssh_bin", "/usr/bin/ssh"))
            host["remote_python"] = _match(SAFE_REMOTE_PATH, item.get("remote_python", "/usr/bin/python3"))
        elif host["transport"] != "local":
            raise RelayError("config-invalid")
        cfg["hosts"].append(host)
    sources = [Path(path) for host in cfg["hosts"] if host["transport"] == "local" for path in host["paths"].values()]
    inboxes = [inbox] + ([Path(cfg["escalation"]["inbox"])] if "escalation" in cfg else [])
    for inbox in inboxes:
        checks = [(p, False) for p in destinations + [p for p in inboxes if p is not inbox]]
        checks += [(p, True) for p in sources]
        for destination, is_source in checks:
            try:
                resolved = destination.resolve()
            except (OSError, RuntimeError):
                if not is_source:
                    raise
                resolved = Path(os.path.abspath(destination))
            if inbox.resolve() == resolved:
                raise RelayError("config-invalid")
            try:
                same_file = os.path.samefile(inbox, destination)
            except FileNotFoundError:
                same_file = False
            except OSError:
                if not is_source:
                    raise
                same_file = False
            if same_file:
                raise RelayError("config-invalid")
    if not cfg["hosts"] or len({h["id"] for h in cfg["hosts"]}) != len(cfg["hosts"]):
        raise RelayError("config-invalid")
    return cfg


def _ssh_output(argv: list, program: bytes, timeout: int) -> bytes:
    """Bound each output pipe while writing stdin; kill and reap on timeout/overflow."""
    deadline = time.monotonic() + timeout
    with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        try:
            with selectors.DefaultSelector() as ready:
                for pipe, event in (
                    (proc.stdin, selectors.EVENT_WRITE),
                    (proc.stdout, selectors.EVENT_READ),
                    (proc.stderr, selectors.EVENT_READ),
                ):
                    os.set_blocking(pipe.fileno(), False)
                    ready.register(pipe, event)
                pending, output, sizes = memoryview(program), bytearray(), {proc.stdout: 0, proc.stderr: 0}
                while ready.get_map():
                    # ControlPersist may retain stderr after ssh exits and stdout is complete.
                    if proc.stdout.closed and proc.poll() is not None:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, timeout)
                    for key, _ in ready.select(min(remaining, 0.05)):
                        pipe = key.fileobj
                        if pipe is proc.stdin:
                            pending = pending[os.write(pipe.fileno(), pending[:65536]) :]
                            if pending:
                                continue
                        else:
                            data = os.read(pipe.fileno(), min(65536, MAX_OUTPUT_BYTES - sizes[pipe] + 1))
                            sizes[pipe] += len(data)
                            if sizes[pipe] > MAX_OUTPUT_BYTES:
                                raise ValueError("ssh-output-limit")
                            if pipe is proc.stdout:
                                output.extend(data)
                            if data:
                                continue
                        ready.unregister(pipe)
                        pipe.close()
                if proc.wait(timeout=max(0, deadline - time.monotonic())) != 0:
                    raise ValueError("ssh-failed")
                return bytes(output)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def _snapshot(host: dict, cfg: dict):
    if host["transport"] == "local":
        return _read_sources(host["paths"], MAX_SOURCE_BYTES)
    try:
        reader = inspect.getsource(_read_sources)
        program = f"{reader}\nimport json, sys\nsys.stdout.write(json.dumps(_read_sources({host['paths']!r}, {MAX_SOURCE_BYTES})))\n"
        options = [arg for option in SSH_OPTIONS for arg in ("-o", option)]
        remote_argv = [shlex.quote(arg) for arg in (host["remote_python"], "-I", "-")]
        argv = [host["ssh_bin"], "-T", *options, "--", host["ssh_alias"], *remote_argv]
        snap = json.loads(_ssh_output(argv, program.encode(), cfg["ssh_timeout_seconds"]))
        sources = snap["sources"]
        valid = isinstance(snap["now"], float) and set(sources) == set(SOURCES)
        valid = valid and all(
            r.get("s") in READ_STATES and (r["s"] != "ok" or isinstance(r.get("text"), str)) for r in sources.values()
        )
        return snap if valid else None
    except (OSError, subprocess.SubprocessError, ValueError, RecursionError, KeyError, TypeError, AttributeError):
        return None


def _when(value) -> float:
    if not isinstance(value, str):
        raise ValueError("timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp")
    return parsed.timestamp()


def evaluate(snap: dict, cfg: dict, seen: list, in_grace: bool):
    """Return (issues, measured sources, held episodes).

    A source is measured only by fresh, validated evidence, or by the healthy absence of an
    optional file never seen before. Stale, pending or erroring evidence can open issues but
    clears nothing. Codes present in either badge list hold existing episodes open; only
    critical, unsuppressed active codes open new badge episodes.
    """
    now, issues, measured, held, docs = snap["now"], set(), set(), set(), {}
    for name, read in snap["sources"].items():
        if read["s"] == "ok":
            try:
                docs[name] = json.loads(read["text"])
            except (ValueError, RecursionError):
                issues.add(f"source-corrupt:{name}")
        elif read["s"] != "absent":
            issues.add(f"source-{read['s']}:{name}")
    jobs, pause = docs.get("jobs", {}), docs.get("pause", {})
    for name, doc in (("jobs", jobs), ("pause", pause)):
        try:
            if not isinstance(doc, dict):
                raise ValueError("control")
            if name == "pause" and name in docs:
                if not isinstance(doc.get("labels"), list):
                    raise ValueError("pause labels")
                if doc.get("expires_at") is not None:
                    _when(doc["expires_at"])
        except (ValueError, OverflowError):
            issues.add(f"source-corrupt:{name}")
            docs.pop(name, None)
        else:
            if name in docs or (snap["sources"][name]["s"] == "absent" and name not in seen):
                measured.add(name)
    jobs, pause = docs.get("jobs", {}), docs.get("pause", {})

    def off(label: str) -> bool:
        record = jobs.get(label) if isinstance(jobs, dict) else None
        if isinstance(record, dict) and record.get("enabled") is False:
            return True
        if label not in [str(x) for x in pause.get("labels", [])]:
            return False
        try:
            return pause.get("expires_at") is None or _when(pause["expires_at"]) > now
        except ValueError:
            return False

    def fresh(stamp, max_age: int) -> bool:
        when = _when(stamp)
        if when > now + CLOCK_SKEW_SECONDS:
            raise ValueError("future timestamp")
        return now - when <= max_age

    def absent(name: str, label: str) -> None:
        if not (in_grace or off(label)):
            issues.add(f"source-missing:{name}")

    if isinstance(docs.get("job-alerts"), dict):
        issues.update(
            f"job-alert:{k}" if SAFE_ID.fullmatch(str(k)) else "job-alert:invalid-key" for k in docs["job-alerts"]
        )
        measured.add("job-alerts")
    elif "job-alerts" in docs:
        issues.add("source-corrupt:job-alerts")
    elif snap["sources"]["job-alerts"]["s"] == "absent":
        if "job-alerts" in seen:
            issues.add("source-missing:job-alerts")
        else:
            measured.add("job-alerts")
    try:
        if "badge" in docs:
            badge = docs["badge"]
            alerts, state = badge["alerts"], badge["alerts"]["state"]
            if badge["schema_version"] != 1 or state not in ("measured", "pending_first_run"):
                raise ValueError("badge")
            if state == "pending_first_run":
                if _when(alerts["expected_first_run_by"]) < now and not off(HEALTH_LABEL):
                    issues.add("health-stale")
            else:
                normalized = {}
                for group in ("active", "suppressed"):
                    if not isinstance(alerts[group], list):
                        raise ValueError("badge list")
                    normalized[group] = []
                    for item in alerts[group]:
                        if (
                            not isinstance(item, dict)
                            or not isinstance(item.get("code"), str)
                            or item.get("severity") not in ("critical", "warning")
                        ):
                            raise ValueError("badge entry")
                        code = item["code"].replace("_", "-")
                        if not SAFE_ID.fullmatch(code):
                            raise ValueError("badge code")
                        normalized[group].append((code, item["severity"]))
                muted = {f"badge:{code}" for code, _ in normalized["suppressed"]}
                held.update(muted | {f"badge:{code}" for code, _ in normalized["active"]})
                current = fresh(badge["generated_at"], cfg["badge_max_age_seconds"])
                if not (current or off(HEALTH_LABEL)):
                    issues.add("health-stale")
                for code, severity in normalized["active"]:
                    if severity == "critical" and not code.startswith("job-alert-") and f"badge:{code}" not in muted:
                        issues.add(f"badge:{code}")
                if current:
                    measured.add("badge")
        elif snap["sources"]["badge"]["s"] == "absent":
            absent("badge", HEALTH_LABEL)
    except (KeyError, TypeError, ValueError, AttributeError):
        issues.add("source-corrupt:badge")
    try:
        if "watcher" in docs:
            if fresh(docs["watcher"]["updated_at"], cfg["watcher_max_age_seconds"]):
                measured.add("watcher")
            elif not off(WATCH_LABEL):
                issues.add("watcher-stale")
        elif snap["sources"]["watcher"]["s"] == "absent":
            absent("watcher", WATCH_LABEL)
    except (KeyError, TypeError, ValueError, AttributeError):
        issues.add("source-corrupt:watcher")
    return issues, measured, held


def _source_of(issue: str):
    """The source whose measured healthy read may close this issue; None means transport."""
    kind, _, name = issue.partition(":")
    if kind.startswith("source-"):
        return name
    return {"job-alert": "job-alerts", "badge": "badge", "health-stale": "badge", "watcher-stale": "watcher"}.get(kind)


def _observe(host: dict, cfg: dict, hstate: dict, clock: float, epoch: str) -> list:
    """Advance one host's episodes; return newly opened events."""
    hstate.setdefault("first_seen", clock)
    snap = _snapshot(host, cfg)
    reached = snap is not None
    if reached:
        hstate["unreach"] = 0
        # A future first_seen (a clock that ran ahead) grants no grace rather than endless grace.
        in_grace = 0 <= clock - hstate["first_seen"] < cfg["first_run_grace_seconds"]
        issues, measured, held = evaluate(snap, cfg, hstate.setdefault("seen", []), in_grace)
        hstate["seen"] = sorted(set(hstate["seen"]) | {n for n, r in snap["sources"].items() if r["s"] != "absent"})
    else:
        hstate["unreach"] = hstate.get("unreach", 0) + 1
        issues = {"host-unreachable"} if hstate["unreach"] >= cfg["unreachable_after"] else set()
        measured, held = set(), set()
    episodes, events = hstate.setdefault("episodes", {}), []
    for issue in sorted(issues):
        episode = episodes.setdefault(issue, {"open": False})
        if not episode["open"]:
            hstate["seq"] = hstate.get("seq", 0) + 1
            event = _event(host, issue, epoch, hstate["seq"], clock)
            episode.update(
                open=True, seq=hstate["seq"], opened_at=clock, event_id=event["id"], acknowledged=False, escalated=False
            )
            episode.pop("migrated", None)
            events.append(event)
    for issue, episode in episodes.items():
        source = _source_of(issue)
        if episode["open"] and issue not in issues | held and (source in measured if source else reached):
            episode["open"] = False
    closed = sorted((e["seq"], i) for i, e in episodes.items() if not e["open"])
    for _, issue in closed[: max(0, len(closed) - MAX_CLOSED)]:
        del episodes[issue]
    return events


def _event(host: dict, issue: str, epoch: str, seq: int, clock: float, event_id=None) -> dict:
    event_id = event_id or f"bl-{host['id']}-{epoch}-{seq}-{issue.replace(':', '.')}"
    source = SOURCES.get(_source_of(issue), "transport")
    line = (
        f"@orc BLOCKED: [BrainLayer {host['label']}] {issue} on host={host['id']}; "
        f"source={host['id']}:{source}; event={event_id}"
    )
    stamp = datetime.fromtimestamp(clock, timezone.utc).isoformat(timespec="seconds")
    return {"id": event_id, "ts": stamp, "line": line}


class _Locked:
    def __init__(self, path: Path):
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)

    def __enter__(self):
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def _load_state(path: Path, inbox: str, accept_new_epoch: bool, clock=None) -> dict:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        try:
            text = _read_inbox(inbox)
        except RelayError:
            if not accept_new_epoch:
                raise RelayError("state-missing") from None
            text = ""
        whole, tail = _lines(text)
        if not accept_new_epoch and any(
            re.fullmatch(r"<!-- brainlayer-health:[a-z0-9][a-z0-9._-]* -->", line) for line in whole | {tail}
        ):
            raise RelayError("state-missing") from None
        return {
            "version": 4,
            "epoch": os.urandom(16).hex(),
            "hosts": {},
            "outbox": [],
            "escalation_outbox": [],
            "escalation_inbox": None,
        }
    except (OSError, ValueError, RecursionError):
        raise RelayError("state-corrupt") from None
    _validate_state(state)
    if state["version"] < 4:
        legacy = state["version"] == 2
        if legacy:
            state["epoch"] = os.urandom(16).hex()
        for host_id, host in state["hosts"].items():
            for issue, episode in host.get("episodes", {}).items():
                identity = host_id if legacy else f"{host_id}-{state['epoch']}"
                episode.update(
                    opened_at=time.time() if clock is None else clock,
                    event_id=f"bl-{identity}-{episode['seq']}-{issue.replace(':', '.')}",
                    acknowledged=False,
                    escalated=False,
                    migrated=True,
                )
        state.update(version=4, escalation_outbox=[], escalation_inbox=None)
    try:
        _verify_inbox(state, inbox)
    except RelayError as exc:
        if str(exc) == "state-stale":
            raise
        state.setdefault("verify_floor", {key: host.get("seq", 0) for key, host in state["hosts"].items()})
    return state


def _verify_inbox(state: dict, inbox: str) -> None:
    """Verify against the earliest unscanned sequence, before any outage observations."""
    text = _read_inbox(inbox)
    whole, tail = _lines(text)
    marker = re.compile(
        rf"<!-- brainlayer-health:bl-([a-z0-9][a-z0-9._-]{{0,63}})-{state['epoch']}-([0-9]+)-[a-z0-9._-]+ -->"
    )
    for line in whole | {tail}:
        match = marker.fullmatch(line)
        if match:
            floor = state.get("verify_floor")
            persisted = str(
                floor.get(match[1], 0) if floor is not None else state["hosts"].get(match[1], {}).get("seq", 0)
            )
            delivered = match[2].lstrip("0") or "0"
            if (len(delivered), delivered) > (len(persisted), persisted):
                raise RelayError("state-stale")
    state.pop("verify_floor", None)


def _validate_state(state) -> None:
    def require(valid):
        if not valid:
            raise RelayError("state-corrupt")

    def sequence(value):
        return type(value) is int and value >= 0

    def single_line(value):
        return isinstance(value, str) and bool(value) and not re.search(r"[\x00-\x1f\x7f]", value)

    require(isinstance(state, dict))
    require(type(state.get("version")) is int and state["version"] in (2, 3, 4))
    if state["version"] >= 3 or "epoch" in state:
        require(isinstance(state.get("epoch"), str) and re.fullmatch(r"[0-9a-f]{32}", state["epoch"]))
    if state["version"] == 4:
        require(isinstance(state.get("escalation_outbox"), list))
        require("escalation_inbox" in state)
        destination = state["escalation_inbox"]
        require(destination is None or (single_line(destination) and os.path.isabs(destination)))
        require(not state["escalation_outbox"] or destination is not None)
    require(isinstance(state.get("hosts"), dict) and isinstance(state.get("outbox"), list))
    for host_id, host in state["hosts"].items():
        require(isinstance(host_id, str) and SAFE_ID.fullmatch(host_id))
        require(isinstance(host, dict))
        require(sequence(host.get("seq", 0)))
        require(sequence(host.get("unreach", 0)))
        first_seen = host.get("first_seen", 0.0)
        # An int beyond float range (10**400 is valid JSON) would overflow the clock arithmetic.
        require(type(first_seen) in (int, float) and abs(first_seen) < 1e15 and math.isfinite(first_seen))
        seen, episodes = host.get("seen", []), host.get("episodes", {})
        require(isinstance(seen, list) and all(isinstance(name, str) and name in SOURCES for name in seen))
        require(isinstance(episodes, dict))
        for issue, episode in episodes.items():
            require(
                isinstance(issue, str)
                and (
                    issue in ("health-stale", "watcher-stale", "host-unreachable")
                    or re.fullmatch(r"[a-z0-9._-]+:[a-z0-9._-]+", issue)
                )
            )
            require(isinstance(episode, dict) and type(episode.get("open")) is bool)
            require(sequence(episode.get("seq")) and 0 < episode["seq"] <= host.get("seq", 0))
            if state["version"] == 4:
                opened = episode.get("opened_at")
                require(type(opened) in (int, float) and abs(opened) < 1e15 and math.isfinite(opened))
                require(type(episode.get("acknowledged")) is bool and type(episode.get("escalated")) is bool)
                require(type(episode.get("migrated", False)) is bool)
                require(single_line(episode.get("event_id")))
                require(re.fullmatch(r"[a-z0-9][a-z0-9._-]*", episode["event_id"]))
    if "verify_floor" in state:
        require(isinstance(state["verify_floor"], dict))
        for host_id, floor in state["verify_floor"].items():
            require(isinstance(host_id, str) and SAFE_ID.fullmatch(host_id))
            require(sequence(floor) and floor <= state["hosts"].get(host_id, {}).get("seq", 0))
    for event in state["outbox"] + (state["escalation_outbox"] if state["version"] == 4 else []):
        require(isinstance(event, dict))
        require(all(single_line(event.get(key)) for key in ("id", "ts", "line")))
        require(re.fullmatch(r"[a-z0-9][a-z0-9._-]*", event["id"]))


def _atomic_temp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _write_atomic(path: Path, document: dict) -> None:
    tmp = _atomic_temp(path)
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(document, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _save_state(path: Path, state: dict) -> None:
    try:
        _write_atomic(path, state)
    except OSError:
        raise RelayError("state-commit-failed") from None


def _marker(event_id: str) -> str:
    return f"<!-- brainlayer-health:{event_id} -->"


def _read_inbox(inbox: str) -> str:
    try:
        fd = os.open(inbox, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "r", encoding="utf-8", errors="replace", newline="") as handle:
            return handle.read()
    except FileNotFoundError:
        raise RelayError("inbox-unavailable") from None
    except OSError:
        raise RelayError("inbox-unreadable") from None


def _lines(text: str):
    """(set of newline-terminated lines, unterminated tail)."""
    *whole, tail = text.split("\n")
    return set(whole), tail


def _owed(lines, event: dict) -> str:
    """The self-contained record this event still needs, or "" once it is delivered.

    Delivered: the exact BLOCKED line is a whole terminated line and the marker is a whole line,
    wherever each sits. A BLOCKED line already complete (or the unterminated tail) owes only a
    marker record, whose leading newline terminates it. Anything else, a torn fragment or a marker
    without its line, owes the full record again. Every record starts and ends on a line boundary.
    """
    whole, tail = lines
    line, marker = event["line"], _marker(event["id"])
    if line in whole and (marker in whole or marker == tail):
        return ""
    if line in whole or line == tail:
        return f"\n{marker}\n"
    return f"\n### BrainLayer health relay — {event['ts']} — event={event['id']}\n{line}\n{marker}\n"


def _append(inbox: str, records: list) -> None:
    """One write per record, then fsync. A short write fails the run and is never continued."""
    try:
        fd = os.open(inbox, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    except OSError:
        raise RelayError("inbox-unavailable") from None
    try:
        for record in records:
            data = record.encode("utf-8")
            if os.write(fd, data) != len(data):
                raise RelayError("inbox-append-failed")
        os.fsync(fd)
    except OSError:
        raise RelayError("inbox-append-failed") from None
    finally:
        os.close(fd)


def _deliver(cfg: dict, state: dict, state_path: Path, outbox_key="outbox") -> None:
    """Deliver at most MAX_DELIVER_PER_RUN events; the rest stay pending for the next tick."""
    batch = state[outbox_key][:MAX_DELIVER_PER_RUN]
    lines = _lines(_read_inbox(cfg["inbox"]))
    if batch:
        _append(cfg["inbox"], [record for record in (_owed(lines, event) for event in batch) if record])
        lines = _lines(_read_inbox(cfg["inbox"]))
    done = {event["id"] for event in batch if not _owed(lines, event)}
    remaining = [event for event in state[outbox_key] if event["id"] not in done]
    _save_state(state_path, {**state, outbox_key: remaining})
    state[outbox_key] = remaining
    if len(done) < len(batch):
        raise RelayError("inbox-ack-missing")


def _escalate(cfg: dict, state: dict, path: Path, clock: float) -> None:
    """Lead ACKs and measured recovery cancel undelivered escalations, never primary events."""
    escalation = cfg["escalation"]
    if state["escalation_inbox"] not in (None, escalation["inbox"]):
        raise RelayError("destination-changed")
    state["escalation_inbox"] = escalation["inbox"]
    whole, tail = _lines(_read_inbox(cfg["inbox"]))
    lines = whole | {tail}
    retained = set()
    for host in cfg["hosts"]:
        for issue, episode in state["hosts"][host["id"]].get("episodes", {}).items():
            if episode.pop("migrated", False):
                # A v3 state may itself have migrated v2 episodes; resolve only after inbox recovery.
                legacy_id = f"bl-{host['id']}-{episode['seq']}-{issue.replace(':', '.')}"
                if _marker(episode["event_id"]) not in lines and _marker(legacy_id) in lines:
                    episode["event_id"] = legacy_id
            event_id = episode["event_id"]
            if f"ACK {event_id}" in lines:
                episode["acknowledged"] = True
            if not episode["open"] or episode["acknowledged"]:
                continue
            retained.add(event_id + ".esc")
            if (
                not episode["escalated"]
                and clock - episode["opened_at"] >= escalation["after_seconds"]
                and any(p.fullmatch(issue) for p in escalation["issues"])
            ):
                state["escalation_outbox"].append(
                    _event(host, issue, state["epoch"], episode["seq"], clock, event_id + ".esc")
                )
                episode["escalated"] = True
    state["escalation_outbox"] = [e for e in state["escalation_outbox"] if e["id"] in retained]
    _save_state(path, state)
    _deliver(escalation, state, path, "escalation_outbox")


def run(config_path: Path, *, clock=time.time, accept_new_epoch=False) -> int:
    try:
        cfg = load_config(config_path)
        state_dir = Path(cfg["state_dir"])
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, RelayError):
        return 2
    status, code, state = {"ts": clock(), "error": None, "pending": None, "hosts": {}}, 0, None
    try:
        lock = _Locked(state_dir / "relay.lock")
    except OSError:
        status["error"] = "lock-open-failed"
        try:
            _write_atomic(state_dir / "status.json", status)
        except OSError:
            pass
        return 1
    with lock:
        try:
            state = _load_state(state_dir / "state.json", cfg["inbox"], accept_new_epoch, clock())
            for host in cfg["hosts"]:
                hstate = state["hosts"].setdefault(host["id"], {})
                state["outbox"].extend(_observe(host, cfg, hstate, clock(), state["epoch"]))
                status["hosts"][host["id"]] = "unreachable" if hstate.get("unreach") else "reached"
            _save_state(state_dir / "state.json", state)
            if "verify_floor" in state:
                # Access may recover within this run; verify before any delivery can occur.
                _verify_inbox(state, cfg["inbox"])
                _save_state(state_dir / "state.json", state)
            _deliver(cfg, state, state_dir / "state.json")
            if "escalation" in cfg:
                try:
                    _escalate(cfg, state, state_dir / "state.json", clock())
                except RelayError as exc:
                    raise RelayError(f"escalation-{exc}") from None
        except RelayError as exc:
            status["error"] = str(exc)
            code = 2 if str(exc) == "state-corrupt" else 1
        status["pending"] = len(state["outbox"]) + len(state["escalation_outbox"]) if state else None
        try:
            _write_atomic(state_dir / "status.json", status)
        except OSError:
            code = code or 1
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--accept-new-epoch", action="store_true", help="Allow lost-state recovery with a new epoch")
    args = parser.parse_args(argv)
    return run(args.config, accept_new_epoch=args.accept_new_epoch)


if __name__ == "__main__":
    sys.exit(main())
