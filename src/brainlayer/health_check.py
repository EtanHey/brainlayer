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

from .paths import get_db_path

DEFAULT_SOCKET_PATH = Path("/tmp/brainbar.sock")
DEFAULT_STATE_PATH = Path("~/.local/share/brainlayer/health-check-state.json").expanduser()
DEFAULT_CANARY_QUERY = "agentopology"
DEFAULT_HOTLANE_LABEL = "com.brainlayer.hotlane-brainbar"
DEFAULT_BRAINBAR_DAEMON_LABEL = "com.brainlayer.brainbar-daemon"
DEFAULT_BACKLOG_BATCH = 128
DEFAULT_HEAL_MIN_CONSECUTIVE_FAILURES = 2
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
    return subprocess.run(args, text=True, capture_output=True, check=False)


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


def _kickstart(label: str, command_runner: CommandRunner) -> str:
    target = f"gui/{os.getuid()}/{label}"
    command_runner(["launchctl", "kickstart", "-k", target])
    return f"kickstart:{label}"


def _kickstart_once(
    result: HealthCheckResult,
    label: str,
    issue_code: str,
    consecutive_failures: int,
    command_runner: CommandRunner,
) -> None:
    action = f"kickstart:{label}"
    if action in result.actions:
        return
    print(
        f"heal action label={label} issue={issue_code} consecutive_failures={consecutive_failures} action={action}",
        file=sys.stderr,
    )
    result.actions.append(_kickstart(label, command_runner))


def _previous_heal_failures(state: dict[str, Any]) -> dict[str, int]:
    raw_failures = state.get("heal_failures")
    if not isinstance(raw_failures, dict):
        return {}
    failures: dict[str, int] = {}
    for issue_code, count in raw_failures.items():
        if not isinstance(issue_code, str) or not isinstance(count, int):
            continue
        failures[issue_code] = max(0, count)
    return failures


def _updated_heal_failures(
    previous_failures: dict[str, int],
    current_issue_codes: set[str],
) -> dict[str, int]:
    return {issue_code: previous_failures.get(issue_code, 0) + 1 for issue_code in sorted(current_issue_codes)}


def _apply_heals(
    *,
    result: HealthCheckResult,
    issue_labels: dict[str, str],
    previous_failures: dict[str, int],
    config: HealthCheckConfig,
    command_runner: CommandRunner,
) -> dict[str, int]:
    current_issue_codes = {issue.code for issue in result.issues}
    heal_failures = _updated_heal_failures(previous_failures, current_issue_codes)
    if not config.heal:
        return heal_failures
    threshold = max(1, config.heal_min_consecutive_failures)
    for issue_code, label in issue_labels.items():
        consecutive_failures = heal_failures.get(issue_code, 0)
        if consecutive_failures >= threshold:
            _kickstart_once(result, label, issue_code, consecutive_failures, command_runner)
    return heal_failures


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
    heal_issue_labels: dict[str, str] = {}

    hotlane_processes = parse_hotlane_processes(ps_output_fn())
    result.hotlane_running = bool(hotlane_processes)
    if not hotlane_processes:
        result.issues.append(
            HealthIssue("hotlane_dead", "critical", "hotlane BrainBar embedding daemon is not running")
        )
        if config.heal:
            heal_issue_labels["hotlane_dead"] = config.hotlane_label
    else:
        result.backlog_batch = min(process.backlog_batch for process in hotlane_processes)
        if any(process.backlog_batch <= 0 for process in hotlane_processes):
            result.issues.append(
                HealthIssue("hotlane_backlog_disabled", "critical", "--backlog-batch is 0; embeddings will not drain")
            )
            if config.heal:
                heal_issue_labels["hotlane_backlog_disabled"] = config.hotlane_label

    previous_missing = state.get("missing_vectors")
    result.previous_missing_vectors = int(previous_missing) if isinstance(previous_missing, int) else None
    try:
        result.missing_vectors = count_missing_embeddings(config.db_path)
    except Exception as exc:
        result.issues.append(
            HealthIssue(
                "missing_embeddings_count_failed",
                "critical",
                f"could not count missing embeddings: {exc}",
            )
        )

    stalled_ticks = 0
    if result.missing_vectors is not None:
        if result.previous_missing_vectors is not None:
            prior_stalled_ticks = int(state.get("stalled_ticks", 0) or 0)
            if result.missing_vectors > result.previous_missing_vectors:
                result.issues.append(
                    HealthIssue(
                        "missing_embeddings_climbing",
                        "warning",
                        f"missing embeddings increased {result.previous_missing_vectors} -> {result.missing_vectors}",
                    )
                )
                stalled_ticks = 0
                if config.heal:
                    heal_issue_labels["missing_embeddings_climbing"] = config.hotlane_label
            elif result.missing_vectors == result.previous_missing_vectors and result.missing_vectors > 0:
                stalled_ticks = prior_stalled_ticks + 1
                if stalled_ticks >= config.max_stalled_ticks:
                    result.issues.append(
                        HealthIssue(
                            "missing_embeddings_not_draining",
                            "warning",
                            f"missing embeddings stayed at {result.missing_vectors} for {stalled_ticks} checks",
                        )
                    )
                    if config.heal:
                        heal_issue_labels["missing_embeddings_not_draining"] = config.hotlane_label
    result.stalled_ticks = stalled_ticks

    try:
        response = socket_request_fn(config.socket_path, config.canary_query, config.socket_timeout_seconds)
        canary_success, text = _canary_text(response)
        result.canary_result_count = _canary_count(text) if canary_success else 0
        result.canary_ok = canary_success and (result.canary_result_count or 0) > 0
        if not result.canary_ok:
            code = "brain_search_canary_failed" if not canary_success else "brain_search_canary_empty"
            result.issues.append(
                HealthIssue(
                    code,
                    "critical",
                    f"BrainBar brain_search canary returned no usable results: {text[:240]}",
                )
            )
            if config.heal:
                heal_issue_labels[code] = config.brainbar_daemon_label
    except Exception as exc:
        result.canary_ok = False
        result.issues.append(
            HealthIssue("brain_search_canary_failed", "critical", f"BrainBar brain_search canary failed: {exc}")
        )
        if config.heal:
            heal_issue_labels["brain_search_canary_failed"] = config.brainbar_daemon_label

    heal_failures = _apply_heals(
        result=result,
        issue_labels=heal_issue_labels,
        previous_failures=previous_heal_failures,
        config=config,
        command_runner=command_runner,
    )
    if result.missing_vectors is not None or heal_failures:
        state_payload: dict[str, Any] = dict(state)
        state_payload["heal_failures"] = heal_failures
        state_payload["ts"] = now.isoformat()
        if result.missing_vectors is not None:
            state_payload["missing_vectors"] = result.missing_vectors
            state_payload["stalled_ticks"] = result.stalled_ticks
        _write_state(config.state_path, state_payload)
    result.ok = not result.issues
    return result
