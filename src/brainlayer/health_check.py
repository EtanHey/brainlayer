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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .launchd_primitive import (
    LaunchdVerificationError,
    install_and_verify_launchagent,
    is_launchd_label_loaded,
    launchd_target,
)
from .paths import get_db_path
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
DEFAULT_BACKLOG_BATCH = 4
DEFAULT_HEAL_MIN_CONSECUTIVE_FAILURES = 2
DEFAULT_HEAL_CIRCUIT_BREAKER_LIMIT = 3
HEAL_MIN_CONSECUTIVE_FAILURES_ENV = "BRAINLAYER_HEAL_MIN_CONSECUTIVE_FAILURES"


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
    queue_dir: Path = field(default_factory=lambda: Path("~/.brainlayer/queue").expanduser())
    source_jsonl_globs: list[str] = field(
        default_factory=lambda: [str(root.resolved_path / "**" / "*.jsonl") for root in default_watch_roots()]
    )
    pause_sentinel_path: Path = field(
        default_factory=lambda: Path("~/.local/share/brainlayer/pause.sentinel").expanduser()
    )
    max_offsets_age_seconds: int = 900
    queue_auto_heal_count: int = 25
    queue_page_count: int = 200
    queue_page_oldest_seconds: int = 4 * 60 * 60
    queue_page_bytes: int = 2 * 1024 * 1024 * 1024
    heal_circuit_breaker_limit: int = DEFAULT_HEAL_CIRCUIT_BREAKER_LIMIT
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
    canary_ok: bool = False
    canary_result_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CommandRunner = Callable[[list[str]], Any]
SocketRequestFn = Callable[[Path, str, float], dict[str, Any]]


def _default_ps_output() -> str:
    result = subprocess.run(["ps", "axo", "pid=,command="], text=True, capture_output=True, check=False)
    return result.stdout


def _default_command_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(args=args, returncode=127, stdout="", stderr=str(exc))


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


def _bootstrap_if_absent(label: str, plist_path: Path, command_runner: CommandRunner) -> str:
    loaded = _launchd_label_loaded(label, command_runner)
    if loaded is True:
        return f"loaded:{label}"
    if loaded is None:
        return f"launchctl-unavailable:{label}"
    try:
        install_and_verify_launchagent(label, plist_path, command_runner=command_runner)
        return f"bootstrap:{label}"
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


def count_missing_embeddings(db_path: Path) -> int:
    uri = f"file:{db_path.expanduser()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    try:
        conn.execute("PRAGMA query_only = ON")
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM chunks c
            LEFT JOIN chunk_vectors_rowids r ON r.id = c.id
            WHERE r.id IS NULL
              AND c.content IS NOT NULL
              AND c.content != ''
              AND c.archived_at IS NULL
              AND c.superseded_by IS NULL
              AND c.aggregated_into IS NULL
              AND COALESCE(c.archived, 0) = 0
              AND COALESCE(c.status, 'active') = 'active'
            """
        ).fetchone()
        return int(row[0] if row else 0)
    finally:
        conn.close()


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


def _apply_heals(
    *,
    result: HealthCheckResult,
    issue_labels: dict[str, tuple[str, Path]],
    previous_failures: dict[str, int],
    previous_tripped: set[str],
    config: HealthCheckConfig,
    command_runner: CommandRunner,
) -> tuple[dict[str, int], set[str]]:
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
                    }
                )
                _push_notification("BrainLayer heal escalation", f"{label} {issue_code} failed repeatedly")
            continue
        if consecutive_failures >= threshold:
            action = (
                _bootstrap_if_absent(label, plist_path, command_runner)
                if issue_code in bootstrap_issue_codes
                else _kickstart(label, command_runner)
            )
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
                    }
                )
                _push_notification("BrainLayer heal action", f"{action} for {issue_code}")
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


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _pause_sentinel_state(config: HealthCheckConfig, now: datetime) -> tuple[dict[str, Any], bool, bool]:
    payload = _load_json(config.pause_sentinel_path)
    if not payload:
        return {}, False, False
    expires_at = _parse_iso_datetime(payload.get("expires_at"))
    stale = expires_at is not None and now > expires_at
    return payload, not stale, stale


def _source_recent(config: HealthCheckConfig, now: datetime, window_seconds: int) -> bool:
    import glob

    cutoff = now.timestamp() - window_seconds
    for pattern in config.source_jsonl_globs:
        for raw_path in glob.iglob(str(Path(pattern).expanduser()), recursive=True):
            try:
                if Path(raw_path).stat().st_mtime >= cutoff:
                    return True
            except OSError:
                continue
    return False


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


def _plist_for_label(config: HealthCheckConfig, label: str) -> Path:
    if label == config.watch_label:
        return config.watch_plist_path
    if label == config.drain_label:
        return config.drain_plist_path
    if label == config.health_check_label:
        return config.health_check_plist_path
    if label == config.enrichment_label:
        return config.enrichment_plist_path
    if label == config.hotlane_label:
        return Path(f"~/Library/LaunchAgents/{label}.plist").expanduser()
    if label == config.brainbar_daemon_label:
        return Path(f"~/Library/LaunchAgents/{label}.plist").expanduser()
    return Path(f"~/Library/LaunchAgents/{label}.plist").expanduser()


def run_health_check(
    config: HealthCheckConfig,
    *,
    ps_output_fn: Callable[[], str] = _default_ps_output,
    socket_request_fn: SocketRequestFn = send_brainbar_search_canary,
    command_runner: CommandRunner = _default_command_runner,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> HealthCheckResult:
    now = now_fn()
    result = HealthCheckResult(checked_at=now.isoformat(), ok=True)
    state = _load_state(config.state_path)
    previous_heal_failures = _previous_heal_failures(state)
    previous_tripped = _previous_tripped_heals(state)
    heal_issue_labels: dict[str, tuple[str, Path]] = {}

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

    hotlane_processes = parse_hotlane_processes(ps_output_fn())
    result.hotlane_running = bool(hotlane_processes)
    if not hotlane_processes:
        add_issue("hotlane_dead", "critical", "hotlane BrainBar embedding daemon is not running")
        heal_issue_labels["hotlane_dead"] = (config.hotlane_label, _plist_for_label(config, config.hotlane_label))
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
        result.missing_vectors = count_missing_embeddings(config.db_path)
    except Exception as exc:
        add_issue(
            "missing_embeddings_count_failed",
            "critical",
            f"could not count missing embeddings: {exc}",
        )

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

    if config.heal:
        for label in (
            config.watch_label,
            config.drain_label,
            config.health_check_label,
            config.enrichment_label,
        ):
            if label:
                action = _bootstrap_if_absent(label, _plist_for_label(config, label), command_runner)
                if action.startswith(("bootstrap:", "bootstrap_failed:", "launchctl-unavailable:")):
                    result.actions.append(action)

    for label, issue_code, message in (
        (config.watch_label, "watch_unloaded", "watch launchd label is not loaded"),
        (config.drain_label, "drain_unloaded", "drain launchd label is not loaded"),
        (config.health_check_label, "health_check_unloaded", "health-check launchd label is not loaded"),
        (config.enrichment_label, "enrichment_unloaded", "enrichment launchd label is not loaded"),
    ):
        if not label:
            continue
        loaded = _launchd_label_loaded(label, command_runner)
        if loaded is False:
            add_issue(issue_code, "critical", message)
            heal_issue_labels[issue_code] = (label, _plist_for_label(config, label))

    pause_payload, pause_active, pause_stale = _pause_sentinel_state(config, now)
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

    watcher_health = _load_json(config.watcher_health_path)
    watcher_poll_count = watcher_health.get("poll_count")
    previous_watcher_poll_count = state.get("watcher_poll_count")
    source_recent = _source_recent(config, now, config.max_offsets_age_seconds)
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
    if queue_count > 0 and isinstance(drain_total, int) and drain_total == previous_drain_total:
        add_issue(
            "drain_no_progress",
            "critical",
            f"drain drained_total flat at {drain_total} while queue_count={queue_count}",
        )
        heal_issue_labels["drain_no_progress"] = (config.drain_label, config.drain_plist_path)

    heal_failures, heal_tripped = _apply_heals(
        result=result,
        issue_labels=heal_issue_labels,
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
    if pause_payload:
        state_payload["pause_sentinel"] = pause_payload
    _write_state(config.state_path, state_payload)
    result.ok = not result.issues
    return result
