#!/usr/bin/env python3
"""Recover a live-but-wedged watcher using disk and DB throughput evidence."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

DEFAULT_WATCH_LABEL = "com.brainlayer.watch"
DEFAULT_NOTIFY_ENDPOINT = "http://localhost:3847/notify"


@dataclass(frozen=True)
class Config:
    db_path: Path
    registry_path: Path
    state_path: Path
    source_roots: tuple[Path, ...]
    watch_label: str = DEFAULT_WATCH_LABEL
    watch_plist_path: Path = Path("~/Library/LaunchAgents/com.brainlayer.watch.plist").expanduser()
    stall_threshold: int = 3
    cooldown_seconds: int = 600
    recent_window_seconds: int = 600
    max_source_files: int = 100_000
    max_scan_seconds: float = 20.0
    dry_run: bool = False
    log_path: Path = Path("~/.local/share/brainlayer/logs/throughput-watchdog.log").expanduser()
    notify_endpoint: str = DEFAULT_NOTIFY_ENDPOINT


@dataclass(frozen=True)
class SourceEvidence:
    pending_files: int
    pending_bytes: int
    recent_files: int
    newest_mtime: float | None
    untracked_recent_files: int = 0
    scan_errors: int = 0


@dataclass
class WatchdogResult:
    checked_at_epoch: int
    watcher_highwater_rowid: int
    watcher_highwater_delta: int | None
    pending_files: int
    pending_bytes: int
    recent_files: int
    untracked_recent_files: int
    newest_source_mtime: float | None
    scan_errors: int
    stalled_ticks: int
    action: str
    error: str = ""
    alert_error: str = ""


CommandRunner = Callable[[list[str]], object]
HighwaterReader = Callable[[Path], int]
SourceProbe = Callable[[Config, int], SourceEvidence]
AlertFn = Callable[[Config, WatchdogResult], None]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _default_source_roots(home: Path) -> tuple[Path, ...]:
    return (
        home / ".claude" / "projects",
        home / ".codex" / "sessions",
        home / ".cursor" / "sessions",
        home / ".cursor" / "projects",
        home / ".gemini" / "sessions",
    )


def _read_json_object(path: Path) -> dict:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read trustworthy JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict) -> None:
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_watcher_highwater(db_path: Path) -> int:
    database = db_path.expanduser().resolve()
    if not database.is_file():
        raise RuntimeError(f"BrainLayer DB does not exist: {database}")
    uri = f"{database.as_uri()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=2.0) as connection:
            connection.execute("PRAGMA query_only = ON")
            row = connection.execute(
                "SELECT COALESCE(MAX(rowid), 0) FROM chunks WHERE source = 'realtime_watcher'"
            ).fetchone()
    except sqlite3.Error as exc:
        raise RuntimeError(f"watcher high-water query failed: {exc}") from exc
    if row is None:
        raise RuntimeError("watcher high-water query returned no row")
    return int(row[0])


def _registry_offset(entry: object, *, current_inode: int) -> int | None:
    if not isinstance(entry, dict):
        return None
    offset = entry.get("offset")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        return None
    registry_inode = entry.get("inode")
    if isinstance(registry_inode, int) and registry_inode > 0 and registry_inode != current_inode:
        return 0
    return offset


def collect_source_evidence(config: Config, now_epoch: int) -> SourceEvidence:
    registry = _read_json_object(config.registry_path)
    cutoff = now_epoch - config.recent_window_seconds
    started = time.monotonic()
    pending_files = 0
    pending_bytes = 0
    recent_files = 0
    untracked_recent_files = 0
    newest_mtime: float | None = None
    scan_errors = 0
    roots = [source_root.expanduser() for source_root in config.source_roots if source_root.expanduser().exists()]
    if not roots:
        return SourceEvidence(0, 0, 0, None)
    find_binary = shutil.which("find")
    if find_binary is None:
        raise RuntimeError("find is required for the bounded recent-source scan")
    recent_minutes = max(1, math.ceil(config.recent_window_seconds / 60))
    try:
        completed = subprocess.run(
            [
                find_binary,
                *(str(root) for root in roots),
                "-type",
                "f",
                "-name",
                "*.jsonl",
                "-mmin",
                f"-{recent_minutes}",
                "-print0",
            ],
            capture_output=True,
            timeout=config.max_scan_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"recent-source scan exceeded {config.max_scan_seconds:.1f}s") from exc
    if completed.returncode != 0:
        detail = os.fsdecode(completed.stderr).strip()
        raise RuntimeError(f"recent-source scan failed: {detail or completed.returncode}")
    recent_paths = [Path(os.fsdecode(raw)) for raw in completed.stdout.split(b"\0") if raw]
    if len(recent_paths) > config.max_source_files:
        raise RuntimeError(f"recent-source scan exceeded {config.max_source_files} JSONL files")

    for source_file in recent_paths:
        if time.monotonic() - started > config.max_scan_seconds:
            raise RuntimeError(f"source evidence exceeded {config.max_scan_seconds:.1f}s")
        try:
            source_stat = source_file.stat()
        except OSError:
            scan_errors += 1
            continue
        if source_stat.st_mtime < cutoff:
            continue
        recent_files += 1
        newest_mtime = source_stat.st_mtime if newest_mtime is None else max(newest_mtime, source_stat.st_mtime)
        offset = _registry_offset(registry.get(str(source_file)), current_inode=source_stat.st_ino)
        if offset is None:
            untracked_recent_files += 1
            continue
        if source_stat.st_size > offset:
            pending_files += 1
            pending_bytes += source_stat.st_size - offset

    return SourceEvidence(
        pending_files=pending_files,
        pending_bytes=pending_bytes,
        recent_files=recent_files,
        newest_mtime=newest_mtime,
        untracked_recent_files=untracked_recent_files,
        scan_errors=scan_errors,
    )


def _default_command_runner(args: list[str]):
    return subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)


def _command_result(command_runner: CommandRunner, args: list[str]) -> tuple[int, str]:
    try:
        completed = command_runner(args)
    except Exception as exc:
        return 1, str(exc)
    returncode = int(getattr(completed, "returncode", 0))
    stderr = str(getattr(completed, "stderr", "") or "").strip()
    stdout = str(getattr(completed, "stdout", "") or "").strip()
    return returncode, stderr or stdout


def _restart_watch(config: Config, command_runner: CommandRunner) -> str:
    domain = f"gui/{os.getuid()}"
    target = f"{domain}/{config.watch_label}"
    loaded_returncode, _loaded_output = _command_result(command_runner, ["launchctl", "print", target])
    if loaded_returncode != 0:
        plist_path = config.watch_plist_path.expanduser()
        if not plist_path.is_file():
            raise RuntimeError(f"watch label is absent and plist is missing: {plist_path}")
        bootstrap_returncode, bootstrap_output = _command_result(
            command_runner,
            ["launchctl", "bootstrap", domain, str(plist_path)],
        )
        if bootstrap_returncode != 0:
            raise RuntimeError(f"launchctl bootstrap failed: {bootstrap_output or bootstrap_returncode}")
    kickstart_returncode, kickstart_output = _command_result(
        command_runner,
        ["launchctl", "kickstart", "-k", target],
    )
    if kickstart_returncode != 0:
        raise RuntimeError(f"launchctl kickstart failed: {kickstart_output or kickstart_returncode}")
    verify_returncode, verify_output = _command_result(command_runner, ["launchctl", "print", target])
    if verify_returncode != 0:
        raise RuntimeError(f"launchctl print after kickstart failed: {verify_output or verify_returncode}")
    return f"kickstart:{config.watch_label}"


def _best_effort_alert(config: Config, result: WatchdogResult) -> None:
    config.log_path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with config.log_path.expanduser().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
    body = (
        "Watcher has registry-tracked JSONL bytes pending but realtime_watcher chunks are flat; "
        "automatic recovery is starting."
    )
    try:
        subprocess.run(
            [
                "/usr/bin/osascript",
                "-e",
                f'display notification "{body}" with title "BrainLayer throughput watchdog"',
            ],
            capture_output=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    request = urllib.request.Request(
        config.notify_endpoint,
        data=json.dumps({"title": "BrainLayer throughput watchdog", "body": body, "source": "alerts"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3):
            pass
    except Exception:
        pass


def run_once(
    config: Config,
    *,
    now_epoch: int | None = None,
    highwater_reader: HighwaterReader = read_watcher_highwater,
    source_probe: SourceProbe = collect_source_evidence,
    command_runner: CommandRunner = _default_command_runner,
    alert_fn: AlertFn = _best_effort_alert,
) -> WatchdogResult:
    checked_at = int(time.time()) if now_epoch is None else int(now_epoch)
    state = _read_json_object(config.state_path)
    current_highwater = highwater_reader(config.db_path)
    evidence = source_probe(config, checked_at)
    previous_highwater = state.get("watcher_highwater_rowid")
    previous_stalled = state.get("stalled_ticks", 0)
    if not isinstance(previous_stalled, int) or previous_stalled < 0:
        previous_stalled = 0
    delta = current_highwater - previous_highwater if isinstance(previous_highwater, int) else None

    if delta is None or delta < 0:
        action = "baseline"
        stalled_ticks = 0
    elif delta > 0:
        action = "progress"
        stalled_ticks = 0
    elif evidence.pending_files == 0:
        action = "idle"
        stalled_ticks = 0
    else:
        action = "stalled"
        stalled_ticks = previous_stalled + 1

    result = WatchdogResult(
        checked_at_epoch=checked_at,
        watcher_highwater_rowid=current_highwater,
        watcher_highwater_delta=delta,
        pending_files=evidence.pending_files,
        pending_bytes=evidence.pending_bytes,
        recent_files=evidence.recent_files,
        untracked_recent_files=evidence.untracked_recent_files,
        newest_source_mtime=evidence.newest_mtime,
        scan_errors=evidence.scan_errors,
        stalled_ticks=stalled_ticks,
        action=action,
    )

    last_restart_epoch = state.get("last_restart_epoch")
    cooldown_active = (
        isinstance(last_restart_epoch, int)
        and checked_at >= last_restart_epoch
        and checked_at - last_restart_epoch < config.cooldown_seconds
    )
    if stalled_ticks >= config.stall_threshold:
        if cooldown_active:
            result.action = "cooldown"
        elif config.dry_run:
            result.action = "would_kickstart"
        else:
            try:
                alert_fn(config, result)
            except Exception as exc:
                result.alert_error = str(exc)
                print(f"throughput-watchdog alert failed: {exc}", file=sys.stderr)
            try:
                result.action = _restart_watch(config, command_runner)
            except RuntimeError as exc:
                result.action = "recovery_failed"
                result.error = str(exc)
            else:
                result.stalled_ticks = 0

    next_state = dict(state)
    next_state.update(
        {
            "checked_at_epoch": checked_at,
            "watcher_highwater_rowid": current_highwater,
            "pending_files": evidence.pending_files,
            "pending_bytes": evidence.pending_bytes,
            "recent_files": evidence.recent_files,
            "untracked_recent_files": evidence.untracked_recent_files,
            "newest_source_mtime": evidence.newest_mtime,
            "scan_errors": evidence.scan_errors,
            "stalled_ticks": result.stalled_ticks,
            "last_action": result.action,
        }
    )
    if result.action.startswith("kickstart:"):
        next_state["last_restart_epoch"] = checked_at
        next_state["restart_count"] = int(state.get("restart_count", 0)) + 1
        next_state.pop("last_recovery_error", None)
    elif result.action == "recovery_failed":
        next_state["last_recovery_error"] = result.error
    if result.alert_error:
        next_state["last_alert_error"] = result.alert_error
    else:
        next_state.pop("last_alert_error", None)
    _atomic_write_json(config.state_path, next_state)
    return result


def _build_parser() -> argparse.ArgumentParser:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, default=Path(os.environ.get("BRAINLAYER_DB", home / ".local/share/brainlayer/brainlayer.db"))
    )
    parser.add_argument("--registry", type=Path, default=home / ".local/share/brainlayer/offsets.json")
    parser.add_argument("--state", type=Path, default=home / ".local/share/brainlayer/throughput-watchdog-state.json")
    parser.add_argument("--source-root", action="append", type=Path, dest="source_roots")
    parser.add_argument("--watch-label", default=DEFAULT_WATCH_LABEL)
    parser.add_argument("--watch-plist", type=Path, default=home / "Library/LaunchAgents/com.brainlayer.watch.plist")
    parser.add_argument("--stall-threshold", type=_positive_int, default=3)
    parser.add_argument("--cooldown-seconds", type=_positive_int, default=600)
    parser.add_argument("--recent-window-seconds", type=_positive_int, default=600)
    parser.add_argument("--max-source-files", type=_positive_int, default=100_000)
    parser.add_argument("--max-scan-seconds", type=float, default=20.0)
    parser.add_argument("--log", type=Path, default=home / ".local/share/brainlayer/logs/throughput-watchdog.log")
    parser.add_argument("--notify-endpoint", default=DEFAULT_NOTIFY_ENDPOINT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--now-epoch", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_scan_seconds <= 0:
        raise SystemExit("--max-scan-seconds must be positive")
    config = Config(
        db_path=args.db,
        registry_path=args.registry,
        state_path=args.state,
        source_roots=tuple(args.source_roots or _default_source_roots(Path.home())),
        watch_label=args.watch_label,
        watch_plist_path=args.watch_plist,
        stall_threshold=args.stall_threshold,
        cooldown_seconds=args.cooldown_seconds,
        recent_window_seconds=args.recent_window_seconds,
        max_source_files=args.max_source_files,
        max_scan_seconds=args.max_scan_seconds,
        dry_run=args.dry_run,
        log_path=args.log,
        notify_endpoint=args.notify_endpoint,
    )
    lock_path = config.state_path.expanduser().with_suffix(config.state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload = {"action": "already_running", "checked_at_epoch": int(time.time())}
            print(json.dumps(payload, sort_keys=True) if args.json else "throughput watchdog already running")
            return 0
        try:
            result = run_once(config, now_epoch=args.now_epoch)
        except Exception as exc:
            payload = {"action": "probe_failed", "error": str(exc), "checked_at_epoch": int(time.time())}
            print(json.dumps(payload, sort_keys=True) if args.json else f"throughput watchdog failed: {exc}")
            return 1
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    print(json.dumps(asdict(result), sort_keys=True) if args.json else f"{result.action}: {result}")
    return 1 if result.action == "recovery_failed" else 0


if __name__ == "__main__":
    sys.exit(main())
