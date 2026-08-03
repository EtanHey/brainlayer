"""Lightweight stability health-check for live BrainLayer services."""

from __future__ import annotations

import json
import os
import re
import shlex
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .drain_liveness import (
    DEFAULT_DRAIN_LIVENESS_STALE_SECONDS,
    ENRICH_DAILY_COST_COUNTER_FILENAME,
    STALLED_CODE,
    check_drain_liveness,
)
from .launchd_primitive import (
    LaunchdLabelDisabledError,
    LaunchdVerificationError,
    install_and_verify_launchagent,
    is_launchd_label_loaded,
    launchd_target,
)
from .paths import get_db_path
from .pause import DEFAULT_PAUSE_SENTINEL_PATH, pause_applies_to_label, pause_sentinel_state
from .watcher import default_watch_roots

DEFAULT_SOCKET_PATH = Path("/tmp/brainbar.sock")
DEFAULT_STATE_PATH = Path("~/.local/share/brainlayer/health-check-state.json").expanduser()
DEFAULT_CANARY_QUERY = "agentopology"
DEFAULT_HOTLANE_LABEL = "com.brainlayer.hotlane-brainbar"
DEFAULT_BRAINBAR_DAEMON_LABEL = "com.brainlayer.brainbar-daemon"
DEFAULT_WATCH_LABEL = "com.brainlayer.watch"
DEFAULT_DRAIN_LABEL = "com.brainlayer.drain"
DEFAULT_HEALTH_CHECK_LABEL = "com.brainlayer.health-check"
DEFAULT_ENRICHMENT_LABEL = "com.brainlayer.enrichment"
DEFAULT_INDEX_LABEL = "com.brainlayer.index"
DEFAULT_BACKLOG_BATCH = 4
DEFAULT_HEAL_MIN_CONSECUTIVE_FAILURES = 2
DEFAULT_HEAL_CIRCUIT_BREAKER_LIMIT = 3
DEFAULT_MAX_DURATION_SECONDS = 45.0
HEAL_MIN_CONSECUTIVE_FAILURES_ENV = "BRAINLAYER_HEAL_MIN_CONSECUTIVE_FAILURES"

MISSING_EMBEDDINGS_SQL = """
    SELECT COUNT(*)
    FROM chunks c
    JOIN (
        SELECT id FROM chunks
        EXCEPT
        SELECT id FROM chunk_vectors_rowids
    ) missing ON missing.id = c.id
    WHERE c.content IS NOT NULL
      AND c.content != ''
      AND c.archived_at IS NULL
      AND c.superseded_by IS NULL
      AND c.aggregated_into IS NULL
      AND COALESCE(c.archived, 0) = 0
      AND COALESCE(c.status, 'active') = 'active'
"""

ENRICHMENT_BACKLOG_SQL = """
    SELECT COUNT(*)
    FROM chunks
    WHERE enriched_at IS NULL
      AND enrich_status IS NULL
      AND COALESCE(char_count, length(content), 0) >= 50
      AND content IS NOT NULL
      AND content != ''
      AND archived_at IS NULL
      AND superseded_by IS NULL
      AND aggregated_into IS NULL
      AND COALESCE(archived, 0) = 0
      AND COALESCE(status, 'active') = 'active'
"""


class HealthCheckDeadlineExceeded(RuntimeError):
    """Raised when a read-only health query consumes the check's time budget."""


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return max(minimum, value)


@dataclass(frozen=True)
class HealthIssue:
    code: str
    severity: str
    message: str


@dataclass
class HealthCheckConfig:
    db_path: Path = field(default_factory=get_db_path)
    state_path: Path = field(default_factory=lambda: DEFAULT_STATE_PATH)
    socket_path: Path = DEFAULT_SOCKET_PATH
    canary_query: str = DEFAULT_CANARY_QUERY
    hotlane_label: str = DEFAULT_HOTLANE_LABEL
    brainbar_daemon_label: str = DEFAULT_BRAINBAR_DAEMON_LABEL
    watch_label: str = DEFAULT_WATCH_LABEL
    drain_label: str = DEFAULT_DRAIN_LABEL
    health_check_label: str = DEFAULT_HEALTH_CHECK_LABEL
    enrichment_label: str = DEFAULT_ENRICHMENT_LABEL
    index_label: str = DEFAULT_INDEX_LABEL
    watch_plist_path: Path = field(
        default_factory=lambda: Path("~/Library/LaunchAgents/com.brainlayer.watch.plist").expanduser()
    )
    drain_plist_path: Path = field(
        default_factory=lambda: Path("~/Library/LaunchAgents/com.brainlayer.drain.plist").expanduser()
    )
    health_check_plist_path: Path = field(
        default_factory=lambda: Path("~/Library/LaunchAgents/com.brainlayer.health-check.plist").expanduser()
    )
    enrichment_plist_path: Path = field(
        default_factory=lambda: Path("~/Library/LaunchAgents/com.brainlayer.enrichment.plist").expanduser()
    )
    offsets_path: Path = field(default_factory=lambda: Path("~/.local/share/brainlayer/offsets.json").expanduser())
    watcher_health_path: Path = field(
        default_factory=lambda: Path("~/.local/share/brainlayer/watcher-health.json").expanduser()
    )
    drain_health_path: Path = field(
        default_factory=lambda: Path("~/.local/share/brainlayer/drain-health.json").expanduser()
    )
    t3_health_path: Path = field(default_factory=lambda: Path("~/.local/share/brainlayer/t3-health.json").expanduser())
    queue_dir: Path = field(default_factory=lambda: Path("~/.brainlayer/queue").expanduser())
    pending_stores_path: Path = field(
        default_factory=lambda: Path("~/.local/share/brainlayer/pending-stores.jsonl").expanduser()
    )
    drain_liveness_stale_seconds: float = DEFAULT_DRAIN_LIVENESS_STALE_SECONDS
    source_jsonl_globs: list[str] = field(
        default_factory=lambda: [str(root.resolved_path / "**" / "*.jsonl") for root in default_watch_roots()]
    )
    pause_sentinel_path: Path = field(default_factory=lambda: DEFAULT_PAUSE_SENTINEL_PATH)
    max_offsets_age_seconds: int = 900
    queue_auto_heal_count: int = 25
    queue_page_count: int = 200
    queue_page_oldest_seconds: int = 4 * 60 * 60
    queue_page_bytes: int = 2 * 1024 * 1024 * 1024
    heal_circuit_breaker_limit: int = DEFAULT_HEAL_CIRCUIT_BREAKER_LIMIT
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS
    heal: bool = False
    socket_timeout_seconds: float = 5.0
    max_stalled_ticks: int = 2
    heal_min_consecutive_failures: int = field(
        default_factory=lambda: _env_int(
            HEAL_MIN_CONSECUTIVE_FAILURES_ENV,
            DEFAULT_HEAL_MIN_CONSECUTIVE_FAILURES,
        )
    )


@dataclass
class HotlaneProcess:
    pid: int
    command: str
    backlog_batch: int


@dataclass(frozen=True)
class LockHolder:
    pid: int
    command: str
    db_path: str
    held_ticks: int = 0


@dataclass
class HealthCheckResult:
    checked_at: str
    ok: bool
    issues: list[HealthIssue] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    hotlane_running: bool = False
    backlog_batch: int | None = None
    missing_vectors: int | None = None
    previous_missing_vectors: int | None = None
    stalled_ticks: int = 0
    lock_holder: LockHolder | None = None
    canary_ok: bool = False
    canary_result_count: int | None = None
    duration_seconds: float = 0.0
    slow_check: bool = False
    slow_check_stage: str | None = None
    t3_health: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CommandRunner = Callable[[list[str]], Any]
SocketRequestFn = Callable[[Path, str, float], dict[str, Any]]


def _default_ps_output() -> str | None:
    try:
        result = subprocess.run(
            ["ps", "axo", "pid=,command="],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
        return result.stdout if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _default_command_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, text=True, capture_output=True, check=False, timeout=5)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(args=args, returncode=127, stdout="", stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=args, returncode=124, stdout=exc.stdout or "", stderr="command timed out"
        )


def _command_returncode(result: Any) -> int:
    return int(getattr(result, "returncode", 0) or 0)


def _command_stdout(result: Any) -> str:
    return str(getattr(result, "stdout", "") or "")


def _launchd_target(label: str) -> str:
    return launchd_target(label)


def _launchd_label_loaded(label: str, command_runner: CommandRunner) -> bool | None:
    return is_launchd_label_loaded(label, command_runner=command_runner)


def _kickstart(label: str, command_runner: CommandRunner) -> str:
    target = _launchd_target(label)
    command_runner(["launchctl", "kickstart", "-k", target])
    return f"kickstart:{label}"


def _launchd_process_state(label: str, command_runner: CommandRunner) -> tuple[int, str] | None:
    launchd_result = command_runner(["launchctl", "print", _launchd_target(label)])
    if _command_returncode(launchd_result) != 0:
        return None
    match = re.search(r"\bpid\s*=\s*(\d+)\b", _command_stdout(launchd_result))
    if match is None:
        return None
    pid = int(match.group(1))
    ps_result = command_runner(["ps", "-o", "state=", "-p", str(pid)])
    if _command_returncode(ps_result) != 0:
        return None
    process_state = _command_stdout(ps_result).strip().splitlines()
    if not process_state:
        return None
    return pid, process_state[0].strip()


def _state_is_uninterruptible(process_state: str) -> bool:
    normalized = process_state.upper()
    return "U" in normalized or normalized.startswith("D")


def _bootstrap_if_absent(label: str, plist_path: Path, command_runner: CommandRunner) -> str:
    loaded = _launchd_label_loaded(label, command_runner)
    if loaded is True:
        return f"loaded:{label}"
    if loaded is None:
        return f"launchctl-unavailable:{label}"
    try:
        install_and_verify_launchagent(label, plist_path, command_runner=command_runner)
        return f"bootstrap:{label}"
    except LaunchdLabelDisabledError:
        return f"disabled:{label}"
    except LaunchdVerificationError:
        return f"bootstrap_failed:{label}"


def _emit_heal_event(event: dict[str, Any]) -> None:
    try:
        from .telemetry import emit

        emit("brainlayer-watcher", event)
    except Exception:
        pass


def _push_notification(title: str, message: str) -> None:
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message[:180]}" with title "{title[:80]}"',
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=2,
        )
    except Exception:
        pass


def _parse_backlog_batch(command: str) -> int:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    for index, part in enumerate(parts):
        if part == "--backlog-batch" and index + 1 < len(parts):
            try:
                return int(parts[index + 1])
            except ValueError:
                return 0
        if part.startswith("--backlog-batch="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return 0
    return DEFAULT_BACKLOG_BATCH


def parse_hotlane_processes(ps_output: str) -> list[HotlaneProcess]:
    processes: list[HotlaneProcess] = []
    for line in ps_output.splitlines():
        stripped = line.strip()
        if "hotlane_brainbar_daemon.py" not in stripped:
            continue
        if " rg " in stripped or "ripgrep" in stripped:
            continue
        match = re.match(r"(?P<pid>\d+)\s+(?P<command>.+)", stripped)
        if not match:
            continue
        command = match.group("command")
        processes.append(
            HotlaneProcess(
                pid=int(match.group("pid")),
                command=command,
                backlog_batch=_parse_backlog_batch(command),
            )
        )
    return processes


def _command_for_pid(ps_output: str, pid: int) -> str:
    for line in ps_output.splitlines():
        stripped = line.strip()
        match = re.match(r"(?P<pid>\d+)\s+(?P<command>.+)", stripped)
        if match and int(match.group("pid")) == pid:
            return match.group("command")
    return ""


def _read_writer_lock_holder(db_path: Path, ps_output: str) -> LockHolder | None:
    from .vector_store import VectorStore

    store = object.__new__(VectorStore)
    store.db_path = db_path.expanduser()
    try:
        pidfile = store._writer_pidfile_path()
    except (OSError, RuntimeError, ValueError):
        return None
    fd = VectorStore._open_writer_pidfile_readonly(pidfile)
    if fd is None:
        return None
    try:
        owner_pid, owner_start_time, owner_db_path = VectorStore._read_writer_pidfile_record_fd(fd)
    finally:
        os.close(fd)
    if owner_db_path is not None and not store._pidfile_db_path_matches(owner_db_path):
        return None
    if owner_pid is None or not store._pidfile_owner_matches(owner_pid, owner_start_time):
        return None
    try:
        resolved_db_path = (
            str(Path(owner_db_path).expanduser().resolve()) if owner_db_path else str(store.db_path.resolve())
        )
    except (OSError, RuntimeError, ValueError):
        resolved_db_path = str(store.db_path.expanduser())
    return LockHolder(
        pid=owner_pid,
        command=_command_for_pid(ps_output, owner_pid),
        db_path=resolved_db_path,
    )


def _text_has_sqlite_busy_signal(value: Any) -> bool:
    text = str(value).lower()
    return (
        "sqlite_busy" in text
        or "writerinuseerror" in text
        or "database is locked" in text
        or "database table is locked" in text
        or "database is busy" in text
    )


def _drain_health_has_sqlite_busy_signal(payload: dict[str, Any]) -> bool:
    for value in payload.values():
        if isinstance(value, dict):
            if _drain_health_has_sqlite_busy_signal(value):
                return True
        elif isinstance(value, list):
            if any(_text_has_sqlite_busy_signal(item) for item in value):
                return True
        elif _text_has_sqlite_busy_signal(value):
            return True
    return False


def _same_lock_holder_ticks(state: dict[str, Any], lock_holder: LockHolder | None, *, drain_starved: bool) -> int:
    if lock_holder is None or not drain_starved:
        return 0
    previous_pid = state.get("lock_holder_pid")
    previous_ticks = state.get("lock_holder_held_ticks")
    prior_ticks = previous_ticks if isinstance(previous_ticks, int) else 0
    return prior_ticks + 1 if previous_pid == lock_holder.pid else 1


def _known_lock_holder_labels(config: HealthCheckConfig) -> tuple[str, ...]:
    labels = [
        config.index_label,
        config.enrichment_label,
        config.watch_label,
        config.drain_label,
        config.hotlane_label,
    ]
    return tuple(dict.fromkeys(label for label in labels if label))


def _launchd_print_mentions_pid(stdout: str, pid: int) -> bool:
    return re.search(rf"\bpid\s*=\s*{pid}\b", stdout) is not None


def _command_implies_lock_holder_label(command: str, config: HealthCheckConfig) -> str | None:
    if not command:
        return None
    if "hotlane_brainbar_daemon.py" in command:
        return config.hotlane_label
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    commands = {
        "index": config.index_label,
        "enrich": config.enrichment_label,
        "watch": config.watch_label,
        "drain": config.drain_label,
    }
    for index, part in enumerate(parts[:-1]):
        if Path(part).name == "brainlayer":
            label = commands.get(parts[index + 1])
            if label:
                return label
    return None


def _known_lock_holder_label(
    holder: LockHolder,
    config: HealthCheckConfig,
    command_runner: CommandRunner,
) -> str | None:
    for label in _known_lock_holder_labels(config):
        result = command_runner(["launchctl", "print", _launchd_target(label)])
        if _command_returncode(result) != 0:
            continue
        if _launchd_print_mentions_pid(_command_stdout(result), holder.pid):
            return label
    return _command_implies_lock_holder_label(holder.command, config)


def _holder_message(holder: LockHolder) -> str:
    command = holder.command or "<unknown>"
    return f"holder pid={holder.pid} command={command} db_path={holder.db_path}"


def _read_only_count(
    db_path: Path,
    sql: str,
    *,
    stage: str,
    deadline_at: float | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> int:
    uri = f"file:{db_path.expanduser()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    try:
        conn.execute("PRAGMA query_only = ON")
        if deadline_at is not None:
            conn.set_progress_handler(lambda: int(monotonic_fn() >= deadline_at), 1_000)
        try:
            row = conn.execute(sql).fetchone()
        except sqlite3.OperationalError as exc:
            if deadline_at is not None and "interrupted" in str(exc).lower():
                raise HealthCheckDeadlineExceeded(f"health-check deadline exceeded during {stage}") from exc
            raise
        return int(row[0] if row else 0)
    finally:
        conn.close()


def count_missing_embeddings(
    db_path: Path,
    *,
    deadline_at: float | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> int:
    return _read_only_count(
        db_path,
        MISSING_EMBEDDINGS_SQL,
        stage="missing_embeddings",
        deadline_at=deadline_at,
        monotonic_fn=monotonic_fn,
    )


def _enrichment_backlog(
    db_path: Path,
    *,
    deadline_at: float | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> int:
    return _read_only_count(
        db_path,
        ENRICHMENT_BACKLOG_SQL,
        stage="enrichment_backlog",
        deadline_at=deadline_at,
        monotonic_fn=monotonic_fn,
    )


def _load_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, resolved)


def send_brainbar_search_canary(socket_path: Path, query: str, timeout_seconds: float) -> dict[str, Any]:
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "brain_search",
            "arguments": {"query": query, "num_results": 1},
        },
    }
    payload = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout_seconds)
        client.connect(str(socket_path.expanduser()))
        client.sendall(payload)
        data = b""
        while not data.endswith(b"\n"):
            chunk = client.recv(65_536)
            if not chunk:
                break
            data += chunk
    if not data:
        raise RuntimeError(f"BrainBar socket closed without response: {socket_path}")
    return json.loads(data.decode("utf-8"))


def _canary_text(response: dict[str, Any]) -> tuple[bool, str]:
    if response.get("error"):
        return False, str(response["error"])
    result = response.get("result") or {}
    content = result.get("content") or []
    text = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    if result.get("isError"):
        return False, text or "BrainBar returned isError=true"
    return True, text


def _canary_count(text: str) -> int | None:
    match = re.search(r"-\s*(\d+)\s+of\s+(\d+)\s+shown", text)
    if match:
        return int(match.group(1))
    match = re.search(r"Found\s+(\d+)\s+results?", text)
    if match:
        return int(match.group(1))
    return 1 if text.strip() else 0


def _previous_heal_failures(state: dict[str, Any]) -> dict[str, int]:
    raw_failures = state.get("heal_failures")
    if not isinstance(raw_failures, dict):
        return {}
    failures: dict[str, int] = {}
    for key, count in raw_failures.items():
        if not isinstance(key, str):
            continue
        if isinstance(count, dict):
            raw_count = count.get("count")
        else:
            raw_count = count
        if not isinstance(raw_count, int):
            continue
        failures[key] = max(0, raw_count)
    return failures


def _previous_tripped_heals(state: dict[str, Any]) -> set[str]:
    raw = state.get("heal_tripped")
    return {str(item) for item in raw} if isinstance(raw, list) else set()


def _heal_key(label: str, issue_code: str) -> str:
    return f"{label}:{issue_code}"


def _heal_notification_message(action: str, issue_code: str, details: dict[str, Any]) -> str:
    lock_holder = details.get("lock_holder")
    if isinstance(lock_holder, dict):
        pid = lock_holder.get("pid")
        command = lock_holder.get("command") or "<unknown>"
        return f"{action} for {issue_code}: holder pid={pid} command={command}"
    return f"{action} for {issue_code}"


def _apply_heals(
    *,
    result: HealthCheckResult,
    issue_labels: dict[str, tuple[str, Path]],
    issue_details: dict[str, dict[str, Any]] | None = None,
    previous_failures: dict[str, int],
    previous_tripped: set[str],
    config: HealthCheckConfig,
    command_runner: CommandRunner,
) -> tuple[dict[str, int], set[str]]:
    issue_details = issue_details or {}
    current_issue_codes = {issue.code for issue in result.issues}
    heal_failures: dict[str, int] = {}
    tripped = set(previous_tripped)
    for issue_code, (label, _plist_path) in issue_labels.items():
        if issue_code in current_issue_codes:
            key = _heal_key(label, issue_code)
            heal_failures[key] = previous_failures.get(key, 0) + 1
        else:
            tripped.discard(_heal_key(label, issue_code))
    if not config.heal:
        return heal_failures, tripped
    threshold = max(1, config.heal_min_consecutive_failures)
    breaker_limit = max(threshold, config.heal_circuit_breaker_limit)
    bootstrap_issue_codes = {
        "watch_unloaded",
        "drain_unloaded",
        "health_check_unloaded",
        "hotlane_unloaded",
        "enrichment_unloaded",
    }
    for issue_code, (label, plist_path) in issue_labels.items():
        key = _heal_key(label, issue_code)
        consecutive_failures = heal_failures.get(key, 0)
        details = issue_details.get(issue_code, {})
        if consecutive_failures >= breaker_limit:
            if issue_code in bootstrap_issue_codes:
                action = _bootstrap_if_absent(label, plist_path, command_runner)
                if action.startswith("bootstrap:"):
                    tripped.discard(key)
                    heal_failures.pop(key, None)
                    if action not in result.actions:
                        result.actions.append(action)
                    continue
            if key not in tripped:
                tripped.add(key)
                result.actions.append(f"heal_escalation:{label}:{issue_code}")
                _emit_heal_event(
                    {
                        "_type": "heal_escalation",
                        "label": label,
                        "issue_code": issue_code,
                        "consecutive_failures": consecutive_failures,
                        **details,
                    }
                )
                message = (
                    _heal_notification_message(label, issue_code, details)
                    if details
                    else f"{label} {issue_code} failed repeatedly"
                )
                _push_notification(
                    "BrainLayer heal escalation",
                    message,
                )
            continue
        if consecutive_failures >= threshold:
            if issue_code in bootstrap_issue_codes:
                action = _bootstrap_if_absent(label, plist_path, command_runner)
            else:
                process = _launchd_process_state(label, command_runner)
                if process is not None and _state_is_uninterruptible(process[1]):
                    pid, process_state = process
                    action = f"heal_backoff:{label}:{issue_code}:pid={pid}:state={process_state}"
                    if action not in result.actions:
                        result.actions.append(action)
                        _emit_heal_event(
                            {
                                "_type": "heal_backoff",
                                "label": label,
                                "issue_code": issue_code,
                                "consecutive_failures": consecutive_failures,
                                "pid": pid,
                                "process_state": process_state,
                                **details,
                            }
                        )
                    continue
                action = _kickstart(label, command_runner)
            if action.startswith("bootstrap:"):
                tripped.discard(key)
                heal_failures.pop(key, None)
            if action not in result.actions:
                print(
                    f"heal action label={label} issue={issue_code} "
                    f"consecutive_failures={consecutive_failures} action={action}",
                    file=sys.stderr,
                )
                result.actions.append(action)
                _emit_heal_event(
                    {
                        "_type": "heal",
                        "label": label,
                        "issue_code": issue_code,
                        "action": action,
                        "consecutive_failures": consecutive_failures,
                        **details,
                    }
                )
                _push_notification(
                    "BrainLayer heal action",
                    _heal_notification_message(action, issue_code, details),
                )
    return heal_failures, tripped


def _path_age_seconds(path: Path, now: datetime) -> float | None:
    try:
        return max(0.0, now.timestamp() - path.expanduser().stat().st_mtime)
    except OSError:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _t3_health_issue(payload: dict[str, Any]) -> HealthIssue | None:
    """Turn the T3 adapter's durable health snapshot into a check issue."""
    if not payload.get("alerting"):
        return None
    reasons = payload.get("alert_reasons")
    reason_text = ", ".join(str(reason) for reason in reasons if reason) if isinstance(reasons, list) else "unknown"
    failures = payload.get("failures")
    failure_text = ""
    if isinstance(failures, list) and failures:
        latest = failures[-1]
        if isinstance(latest, dict):
            failure_text = f" ({latest.get('message') or latest.get('code') or 'reader failure'})"
    return HealthIssue(
        "t3_ingest_unhealthy",
        "critical",
        f"T3 ingestion health alert: {reason_text}{failure_text}",
    )


def _pause_sentinel_state(config: HealthCheckConfig, now: datetime) -> tuple[dict[str, Any], bool, bool]:
    return pause_sentinel_state(config.pause_sentinel_path, now)


def _source_recent(
    config: HealthCheckConfig,
    now: datetime,
    window_seconds: int,
    *,
    deadline_at: float | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> bool:
    import glob

    def scan() -> bool:
        cutoff = now.timestamp() - window_seconds
        for pattern in config.source_jsonl_globs:
            for raw_path in glob.iglob(str(Path(pattern).expanduser()), recursive=True):
                try:
                    if Path(raw_path).stat().st_mtime >= cutoff:
                        return True
                except OSError:
                    continue
        return False

    if deadline_at is None:
        return scan()

    remaining = deadline_at - monotonic_fn()
    if remaining <= 0:
        raise HealthCheckDeadlineExceeded("health-check deadline exceeded during source_recent")

    completed = threading.Event()
    outcome: dict[str, Any] = {}

    def bounded_scan() -> None:
        try:
            outcome["value"] = scan()
        except Exception as exc:
            outcome["error"] = exc
        finally:
            completed.set()

    threading.Thread(target=bounded_scan, name="brainlayer-health-source-scan", daemon=True).start()
    if not completed.wait(timeout=remaining):
        raise HealthCheckDeadlineExceeded("health-check deadline exceeded during source_recent")
    if error := outcome.get("error"):
        raise error
    return bool(outcome.get("value", False))


def _queue_stats(queue_dir: Path, now: datetime) -> tuple[int, int, float | None]:
    count = 0
    total_bytes = 0
    oldest: float | None = None
    try:
        paths = list(queue_dir.expanduser().glob("*.jsonl"))
    except OSError:
        return 0, 0, None
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        count += 1
        total_bytes += stat.st_size
        age = max(0.0, now.timestamp() - stat.st_mtime)
        oldest = age if oldest is None else max(oldest, age)
    return count, total_bytes, oldest


def _pending_stores_count(path: Path) -> int:
    try:
        with path.expanduser().open(encoding="utf-8") as pending_stores:
            return sum(1 for line in pending_stores if line.strip())
    except FileNotFoundError:
        return 0


def _plist_for_label(config: HealthCheckConfig, label: str) -> Path:
    if label == config.watch_label:
        return config.watch_plist_path
    if label == config.drain_label:
        return config.drain_plist_path
    if label == config.health_check_label:
        return config.health_check_plist_path
    if label == config.enrichment_label:
        return config.enrichment_plist_path
    if label == config.index_label:
        return Path(f"~/Library/LaunchAgents/{label}.plist").expanduser()
    if label == config.hotlane_label:
        return Path(f"~/Library/LaunchAgents/{label}.plist").expanduser()
    if label == config.brainbar_daemon_label:
        return Path(f"~/Library/LaunchAgents/{label}.plist").expanduser()
    return Path(f"~/Library/LaunchAgents/{label}.plist").expanduser()


def run_health_check(
    config: HealthCheckConfig,
    *,
    ps_output_fn: Callable[[], str | None] = _default_ps_output,
    socket_request_fn: SocketRequestFn = send_brainbar_search_canary,
    command_runner: CommandRunner = _default_command_runner,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> HealthCheckResult:
    started_monotonic = monotonic_fn()
    deadline_at = started_monotonic + max(1.0, config.max_duration_seconds)
    now = now_fn()
    result = HealthCheckResult(checked_at=now.isoformat(), ok=True)
    state = _load_state(config.state_path)
    previous_heal_failures = _previous_heal_failures(state)
    previous_tripped = _previous_tripped_heals(state)
    heal_issue_labels: dict[str, tuple[str, Path]] = {}
    heal_issue_details: dict[str, dict[str, Any]] = {}

    def add_issue(code: str, severity: str, message: str) -> None:
        result.issues.append(HealthIssue(code, severity, message))
        _emit_heal_event(
            {
                "_type": "health_issue_detected",
                "issue_code": code,
                "severity": severity,
                "message": message[:500],
            }
        )

    def finish_slow(stage: str, message: str) -> HealthCheckResult:
        result.slow_check = True
        result.slow_check_stage = stage
        result.duration_seconds = max(0.0, monotonic_fn() - started_monotonic)
        add_issue("slow_check", "critical", message)
        state_payload: dict[str, Any] = dict(state)
        state_payload["ts"] = now.isoformat()
        state_payload["slow_check"] = True
        state_payload["slow_check_stage"] = stage
        state_payload["duration_seconds"] = result.duration_seconds
        _write_state(config.state_path, state_payload)
        result.ok = False
        return result

    def deadline_reached(stage: str) -> HealthCheckResult | None:
        if monotonic_fn() < deadline_at:
            return None
        return finish_slow(stage, f"health-check exceeded {config.max_duration_seconds:.0f}s during {stage}")

    pause_payload, pause_active, pause_stale = _pause_sentinel_state(config, now)

    t3_health = _load_json(config.t3_health_path)
    result.t3_health = t3_health or None
    if t3_issue := _t3_health_issue(t3_health):
        add_issue(t3_issue.code, t3_issue.severity, t3_issue.message)

    ps_output = ps_output_fn()
    lock_holder = None
    if ps_output is None:
        add_issue(
            "process_snapshot_failed",
            "critical",
            "could not read the process table; daemon liveness and writer ownership were not evaluated",
        )
    else:
        lock_holder = _read_writer_lock_holder(config.db_path, ps_output)
        result.lock_holder = lock_holder
        hotlane_processes = parse_hotlane_processes(ps_output)
        result.hotlane_running = bool(hotlane_processes)
        if not hotlane_processes:
            add_issue("hotlane_dead", "critical", "hotlane BrainBar embedding daemon is not running")
            heal_issue_labels["hotlane_dead"] = (
                config.hotlane_label,
                _plist_for_label(config, config.hotlane_label),
            )
        else:
            result.backlog_batch = min(process.backlog_batch for process in hotlane_processes)
            if any(process.backlog_batch <= 0 for process in hotlane_processes):
                add_issue("hotlane_backlog_disabled", "critical", "--backlog-batch is 0; embeddings will not drain")
                heal_issue_labels["hotlane_backlog_disabled"] = (
                    config.hotlane_label,
                    _plist_for_label(config, config.hotlane_label),
                )

    previous_missing = state.get("missing_vectors")
    result.previous_missing_vectors = int(previous_missing) if isinstance(previous_missing, int) else None
    try:
        result.missing_vectors = count_missing_embeddings(
            config.db_path,
            deadline_at=deadline_at,
            monotonic_fn=monotonic_fn,
        )
    except HealthCheckDeadlineExceeded as exc:
        return finish_slow("missing_embeddings", str(exc))
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            return finish_slow("missing_embeddings", "health-check deadline interrupted missing_embeddings")
        add_issue(
            "missing_embeddings_count_failed",
            "critical",
            f"could not count missing embeddings: {exc}",
        )
    except Exception as exc:
        add_issue(
            "missing_embeddings_count_failed",
            "critical",
            f"could not count missing embeddings: {exc}",
        )
    if slow_result := deadline_reached("missing_embeddings"):
        return slow_result

    stalled_ticks = 0
    if result.missing_vectors is not None:
        if result.previous_missing_vectors is not None:
            prior_stalled_ticks = int(state.get("stalled_ticks", 0) or 0)
            if result.missing_vectors > result.previous_missing_vectors:
                add_issue(
                    "missing_embeddings_climbing",
                    "warning",
                    f"missing embeddings increased {result.previous_missing_vectors} -> {result.missing_vectors}",
                )
                stalled_ticks = 0
                heal_issue_labels["missing_embeddings_climbing"] = (
                    config.hotlane_label,
                    _plist_for_label(config, config.hotlane_label),
                )
            elif result.missing_vectors == result.previous_missing_vectors and result.missing_vectors > 0:
                stalled_ticks = prior_stalled_ticks + 1
                if stalled_ticks >= config.max_stalled_ticks:
                    add_issue(
                        "missing_embeddings_not_draining",
                        "warning",
                        f"missing embeddings stayed at {result.missing_vectors} for {stalled_ticks} checks",
                    )
                    heal_issue_labels["missing_embeddings_not_draining"] = (
                        config.hotlane_label,
                        _plist_for_label(config, config.hotlane_label),
                    )
    result.stalled_ticks = stalled_ticks

    try:
        response = socket_request_fn(config.socket_path, config.canary_query, config.socket_timeout_seconds)
        canary_success, text = _canary_text(response)
        result.canary_result_count = _canary_count(text) if canary_success else 0
        result.canary_ok = canary_success and (result.canary_result_count or 0) > 0
        if not result.canary_ok:
            code = "brain_search_canary_failed" if not canary_success else "brain_search_canary_empty"
            add_issue(
                code,
                "critical",
                f"BrainBar brain_search canary returned no usable results: {text[:240]}",
            )
            heal_issue_labels[code] = (
                config.brainbar_daemon_label,
                _plist_for_label(config, config.brainbar_daemon_label),
            )
    except Exception as exc:
        result.canary_ok = False
        add_issue("brain_search_canary_failed", "critical", f"BrainBar brain_search canary failed: {exc}")
        heal_issue_labels["brain_search_canary_failed"] = (
            config.brainbar_daemon_label,
            _plist_for_label(config, config.brainbar_daemon_label),
        )
    if slow_result := deadline_reached("brain_search_canary"):
        return slow_result

    if config.heal:
        for label in (
            config.watch_label,
            config.drain_label,
            config.health_check_label,
            config.enrichment_label,
        ):
            if label and not (pause_active and pause_applies_to_label(pause_payload, label)):
                action = _bootstrap_if_absent(label, _plist_for_label(config, label), command_runner)
                if action.startswith(("bootstrap:", "bootstrap_failed:", "launchctl-unavailable:", "disabled:")):
                    result.actions.append(action)

    drain_loaded: bool | None = None
    for label, issue_code, message in (
        (config.watch_label, "watch_unloaded", "watch launchd label is not loaded"),
        (config.drain_label, "drain_unloaded", "drain launchd label is not loaded"),
        (config.health_check_label, "health_check_unloaded", "health-check launchd label is not loaded"),
        (config.enrichment_label, "enrichment_unloaded", "enrichment launchd label is not loaded"),
    ):
        if not label:
            continue
        loaded = _launchd_label_loaded(label, command_runner)
        if issue_code == "drain_unloaded":
            drain_loaded = loaded
        if loaded is False:
            add_issue(issue_code, "critical", message)
            heal_issue_labels[issue_code] = (label, _plist_for_label(config, label))
    if slow_result := deadline_reached("launchd_status"):
        return slow_result

    if pause_stale:
        add_issue(
            "pause_sentinel_stale", "critical", "pause sentinel is expired; launchd resume may have been forgotten"
        )
        _push_notification("BrainLayer pause expired", "pause.sentinel is stale")
        if config.heal:
            try:
                config.pause_sentinel_path.expanduser().unlink()
                result.actions.append("resume:stale-pause-sentinel")
            except OSError:
                result.actions.append("resume_failed:stale-pause-sentinel")

    queue_count, queue_bytes, queue_oldest_age = _queue_stats(config.queue_dir, now)
    try:
        pending_stores_count = _pending_stores_count(config.pending_stores_path)
    except OSError as exc:
        pending_stores_count = 0
        add_issue(
            "pending_stores_count_failed",
            "critical",
            f"could not count pending stores: {exc}",
        )
    if queue_count >= config.queue_auto_heal_count:
        severity = "critical" if queue_count >= config.queue_page_count else "warning"
        add_issue(
            "queue_backed_up",
            severity,
            f"durable queue backlog count={queue_count} bytes={queue_bytes} oldest_age={queue_oldest_age}",
        )
        heal_issue_labels["queue_backed_up"] = (config.drain_label, _plist_for_label(config, config.drain_label))
    if queue_count > 0 and (
        queue_count >= config.queue_page_count
        or queue_bytes >= config.queue_page_bytes
        or (queue_oldest_age is not None and queue_oldest_age >= config.queue_page_oldest_seconds)
    ):
        _push_notification("BrainLayer queue backlog", f"queue_count={queue_count} queue_bytes={queue_bytes}")
    if slow_result := deadline_reached("queue_stats"):
        return slow_result

    watcher_health = _load_json(config.watcher_health_path)
    watcher_poll_count = watcher_health.get("poll_count")
    previous_watcher_poll_count = state.get("watcher_poll_count")
    try:
        source_recent = _source_recent(
            config,
            now,
            config.max_offsets_age_seconds,
            deadline_at=deadline_at,
            monotonic_fn=monotonic_fn,
        )
    except HealthCheckDeadlineExceeded as exc:
        return finish_slow("source_recent", str(exc))
    offsets_age = _path_age_seconds(config.offsets_path, now)
    watcher_health_age = _path_age_seconds(config.watcher_health_path, now)
    if (
        not pause_active
        and source_recent
        and isinstance(watcher_poll_count, int)
        and watcher_poll_count == previous_watcher_poll_count
        and offsets_age is not None
        and offsets_age >= config.max_offsets_age_seconds
        and watcher_health_age is not None
        and watcher_health_age >= config.max_offsets_age_seconds
    ):
        add_issue(
            "watcher_stalled",
            "critical",
            f"watcher poll_count flat at {watcher_poll_count}; offsets_age={offsets_age:.0f}s",
        )
        heal_issue_labels["watcher_stalled"] = (config.watch_label, config.watch_plist_path)

    drain_health = _load_json(config.drain_health_path)
    drain_total = drain_health.get("drained_total")
    previous_drain_total = state.get("drain_drained_total")
    drain_starved = _drain_health_has_sqlite_busy_signal(drain_health)
    try:
        enrichment_backlog = _enrichment_backlog(
            config.db_path,
            deadline_at=deadline_at,
            monotonic_fn=monotonic_fn,
        )
    except HealthCheckDeadlineExceeded as exc:
        return finish_slow("enrichment_backlog", str(exc))
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            return finish_slow("enrichment_backlog", "health-check deadline interrupted enrichment_backlog")
        enrichment_backlog = 0
        add_issue(
            "enrichment_backlog_count_failed",
            "critical",
            f"could not count enrichment backlog: {exc}",
        )
    except Exception as exc:
        enrichment_backlog = 0
        add_issue(
            "enrichment_backlog_count_failed",
            "critical",
            f"could not count enrichment backlog: {exc}",
        )
    if slow_result := deadline_reached("enrichment_backlog"):
        return slow_result
    drain_liveness_issue = check_drain_liveness(
        drain_label=config.drain_label,
        drain_loaded=drain_loaded,
        queue_count=queue_count + pending_stores_count,
        enrichment_backlog=enrichment_backlog,
        drain_health=drain_health,
        now=now,
        stale_seconds=config.drain_liveness_stale_seconds,
        enrich_cost_counter_path=config.db_path.expanduser().parent / ENRICH_DAILY_COST_COUNTER_FILENAME,
    )
    if drain_liveness_issue is not None:
        severity = "critical" if drain_liveness_issue.code == STALLED_CODE else drain_liveness_issue.severity
        add_issue(drain_liveness_issue.code, severity, drain_liveness_issue.message)
        if drain_liveness_issue.code == STALLED_CODE:
            drain_starved = True
    if queue_count > 0 and isinstance(drain_total, int) and drain_total == previous_drain_total:
        add_issue(
            "drain_no_progress",
            "critical",
            f"drain drained_total flat at {drain_total} while queue_count={queue_count}",
        )
        heal_issue_labels["drain_no_progress"] = (config.drain_label, config.drain_plist_path)
        drain_starved = True

    lock_holder_held_ticks = _same_lock_holder_ticks(state, lock_holder, drain_starved=drain_starved)
    if lock_holder is not None:
        result.lock_holder = replace(lock_holder, held_ticks=lock_holder_held_ticks)
    lock_holder_wedge = (
        result.lock_holder is not None and drain_starved and lock_holder_held_ticks >= config.max_stalled_ticks
    )
    if lock_holder_wedge and result.lock_holder is not None:
        holder = result.lock_holder
        add_issue(
            "lock_holder_wedge",
            "critical",
            f"write-lock holder wedge detected after {holder.held_ticks} checks: {_holder_message(holder)}",
        )
        holder_details = {"lock_holder": asdict(holder)}
        heal_issue_details["lock_holder_wedge"] = holder_details
        for victim_issue_code in ("drain_no_progress", "queue_backed_up"):
            heal_issue_labels.pop(victim_issue_code, None)
        if config.heal:
            _emit_heal_event(
                {
                    "_type": "lock_holder_wedge",
                    **holder_details,
                }
            )
            _push_notification("BrainLayer lock-holder wedge", _holder_message(holder))
            holder_label = _known_lock_holder_label(holder, config, command_runner)
            if holder_label:
                heal_issue_labels["lock_holder_wedge"] = (
                    holder_label,
                    _plist_for_label(config, holder_label),
                )

    if pause_active:
        heal_issue_labels = {
            issue_code: label_and_path
            for issue_code, label_and_path in heal_issue_labels.items()
            if not pause_applies_to_label(pause_payload, label_and_path[0])
        }

    heal_failures, heal_tripped = _apply_heals(
        result=result,
        issue_labels=heal_issue_labels,
        issue_details=heal_issue_details,
        previous_failures=previous_heal_failures,
        previous_tripped=previous_tripped,
        config=config,
        command_runner=command_runner,
    )
    state_payload: dict[str, Any] = dict(state)
    state_payload["heal_failures"] = heal_failures
    state_payload["heal_tripped"] = sorted(heal_tripped) if result.issues else []
    state_payload["ts"] = now.isoformat()
    if result.missing_vectors is not None:
        state_payload["missing_vectors"] = result.missing_vectors
        state_payload["stalled_ticks"] = result.stalled_ticks
    if isinstance(watcher_poll_count, int):
        state_payload["watcher_poll_count"] = watcher_poll_count
    if isinstance(drain_total, int):
        state_payload["drain_drained_total"] = drain_total
    if result.lock_holder is not None:
        state_payload["lock_holder_pid"] = result.lock_holder.pid
        state_payload["lock_holder_db_path"] = result.lock_holder.db_path
        state_payload["lock_holder_held_ticks"] = result.lock_holder.held_ticks
    else:
        state_payload.pop("lock_holder_pid", None)
        state_payload.pop("lock_holder_db_path", None)
        state_payload["lock_holder_held_ticks"] = 0
    if pause_payload:
        state_payload["pause_sentinel"] = pause_payload
    result.duration_seconds = max(0.0, monotonic_fn() - started_monotonic)
    result.slow_check = result.duration_seconds >= config.max_duration_seconds
    state_payload["duration_seconds"] = result.duration_seconds
    state_payload["slow_check"] = result.slow_check
    if result.slow_check:
        result.slow_check_stage = "finalize"
        state_payload["slow_check_stage"] = result.slow_check_stage
        add_issue(
            "slow_check",
            "critical",
            f"health-check exceeded {config.max_duration_seconds:.0f}s before state write",
        )
    else:
        state_payload.pop("slow_check_stage", None)
    _write_state(config.state_path, state_payload)
    result.ok = not result.issues
    return result
