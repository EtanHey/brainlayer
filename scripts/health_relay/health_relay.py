#!/usr/bin/env python3
"""Independent BrainLayer health relay: per-host health evidence -> `@orc BLOCKED:` inbox lines.

Runs from its own LaunchAgent on the inbox's Mac, outside the BrainLayer supervisor. Stdlib only:
it never imports BrainLayer, opens a database, reads credentials, heals, or runs a producer.

Delivery is at-least-once with whole-line dedup. Pending events are persisted before any append.
Every attempt appends one self-contained record in a single write, so no record depends on the
bytes before it: other inbox writers do not take `relay.lock`, and may append at any moment. An
event is acknowledged only when, after an inbox fsync in the same run, the inbox holds its exact
BLOCKED line and its marker as whole lines. Torn fragments are left in place and never counted.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,31}$")
SOURCES = {"job-alerts": "job-alerts.json"}
MAX_SOURCE_BYTES, MAX_DELIVER_PER_RUN, MAX_CLOSED = 262144, 32, 128


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
    if not isinstance(value, str) or not pattern.match(value):
        raise RelayError("config-invalid")
    return value


def load_config(path: Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = {"inbox": _abs(raw["inbox"]), "state_dir": _abs(raw["state_dir"]), "hosts": []}
    for item in raw["hosts"]:
        host = {"id": _match(SAFE_ID, item["id"]), "label": _match(SAFE_LABEL, item["label"])}
        base, overrides = _abs(item["dir"]), item.get("sources", {})
        if not isinstance(overrides, dict) or set(overrides) - set(SOURCES):
            raise RelayError("config-invalid")
        host["paths"] = {name: _abs(overrides.get(name, os.path.join(base, rel))) for name, rel in SOURCES.items()}
        if item["transport"] != "local":
            raise RelayError("config-invalid")
        cfg["hosts"].append(host)
    if not cfg["hosts"] or len({h["id"] for h in cfg["hosts"]}) != len(cfg["hosts"]):
        raise RelayError("config-invalid")
    return cfg


def evaluate(snap: dict, seen: list):
    """Return (issues, measured sources).

    A source is measured only by a validated read, or by the healthy absence of an optional file
    never seen before. Erroring evidence opens issues but clears nothing.
    """
    issues, measured, docs = set(), set(), {}
    for name, read in snap["sources"].items():
        if read["s"] == "ok":
            try:
                docs[name] = json.loads(read["text"])
            except ValueError:
                issues.add(f"source-corrupt:{name}")
        elif read["s"] != "absent":
            issues.add(f"source-{read['s']}:{name}")
    if isinstance(docs.get("job-alerts"), dict):
        issues.update(
            f"job-alert:{k}" if SAFE_ID.match(str(k)) else "job-alert:invalid-key" for k in docs["job-alerts"]
        )
        measured.add("job-alerts")
    elif "job-alerts" in docs:
        issues.add("source-corrupt:job-alerts")
    elif snap["sources"]["job-alerts"]["s"] == "absent":
        if "job-alerts" in seen:
            issues.add("source-missing:job-alerts")
        else:
            measured.add("job-alerts")
    return issues, measured


def _source_of(issue: str):
    """The source whose measured healthy read may close this issue."""
    kind, _, name = issue.partition(":")
    return name if kind.startswith("source-") else {"job-alert": "job-alerts"}.get(kind)


def _observe(host: dict, hstate: dict, clock: float) -> list:
    """Advance one host's episodes; return newly opened events."""
    snap = _read_sources(host["paths"], MAX_SOURCE_BYTES)
    issues, measured = evaluate(snap, hstate.setdefault("seen", []))
    hstate["seen"] = sorted(set(hstate["seen"]) | {n for n, r in snap["sources"].items() if r["s"] == "ok"})
    episodes, events = hstate.setdefault("episodes", {}), []
    for issue in sorted(issues):
        episode = episodes.setdefault(issue, {"open": False})
        if not episode["open"]:
            hstate["seq"] = hstate.get("seq", 0) + 1
            episode.update(open=True, seq=hstate["seq"])
            event_id = f"bl-{host['id']}-{hstate['seq']}-{issue.replace(':', '.')}"
            source = SOURCES.get(_source_of(issue), "transport")
            line = (
                f"@orc BLOCKED: [BrainLayer {host['label']}] {issue} on host={host['id']}; "
                f"source={host['id']}:{source}; event={event_id}"
            )
            stamp = datetime.fromtimestamp(clock, timezone.utc).isoformat(timespec="seconds")
            events.append({"id": event_id, "ts": stamp, "line": line})
    for issue, episode in episodes.items():
        if episode["open"] and issue not in issues and _source_of(issue) in measured:
            episode["open"] = False
    closed = sorted((e["seq"], i) for i, e in episodes.items() if not e["open"])
    for _, issue in closed[: max(0, len(closed) - MAX_CLOSED)]:
        del episodes[issue]
    return events


class _Locked:
    def __init__(self, path: Path):
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)

    def __enter__(self):
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def _load_state(path: Path) -> dict:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": 2, "hosts": {}, "outbox": []}
    except (OSError, ValueError):
        raise RelayError("state-corrupt") from None
    if not (isinstance(state, dict) and isinstance(state.get("hosts"), dict) and isinstance(state.get("outbox"), list)):
        raise RelayError("state-corrupt")
    return state


def _write_atomic(path: Path, document: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
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


def _deliver(cfg: dict, state: dict, state_path: Path) -> None:
    """Deliver at most MAX_DELIVER_PER_RUN events; the rest stay pending for the next tick."""
    batch = state["outbox"][:MAX_DELIVER_PER_RUN]
    lines = _lines(_read_inbox(cfg["inbox"]))
    if batch:
        _append(cfg["inbox"], [record for record in (_owed(lines, event) for event in batch) if record])
        lines = _lines(_read_inbox(cfg["inbox"]))
    done = {event["id"] for event in batch if not _owed(lines, event)}
    remaining = [event for event in state["outbox"] if event["id"] not in done]
    _save_state(state_path, {**state, "outbox": remaining})
    state["outbox"] = remaining
    if len(done) < len(batch):
        raise RelayError("inbox-ack-missing")


def run(config_path: Path, *, clock=time.time) -> int:
    try:
        cfg = load_config(config_path)
        state_dir = Path(cfg["state_dir"])
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RelayError):
        return 2
    status, code, state = {"ts": clock(), "error": None, "pending": None}, 0, None
    with _Locked(state_dir / "relay.lock"):
        try:
            state = _load_state(state_dir / "state.json")
            for host in cfg["hosts"]:
                state["outbox"].extend(_observe(host, state["hosts"].setdefault(host["id"], {}), clock()))
            _save_state(state_dir / "state.json", state)
            _deliver(cfg, state, state_dir / "state.json")
        except RelayError as exc:
            status["error"] = str(exc)
            code = 2 if str(exc) == "state-corrupt" else 1
        status["pending"] = len(state["outbox"]) if state else None
        try:
            _write_atomic(state_dir / "status.json", status)
        except OSError:
            code = code or 1
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    return run(parser.parse_args(argv).config)


if __name__ == "__main__":
    sys.exit(main())
