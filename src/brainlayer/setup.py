"""Installable setup helpers for BrainLayer."""

from __future__ import annotations

import hashlib
import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .cli.wizard import DEFAULT_BRAINLAYER_CONFIG, write_gemini_env_file
from .mcp_socket_config import (
    bridge_path_risk,
    iter_json_mcp_servers,
    iter_toml_mcp_servers,
    needs_socket_migration,
    owned_mcp_config_paths,
    socat_socket_conflict,
    socket_server_preserving,
    stable_mcp_bridge_command,
)
from .paths import get_canonical_db_path, resolve_db_path
from .spotlight import ensure_spotlight_excluded_layout as _ensure_spotlight_excluded_layout

DEFAULT_GOOGLE_API_KEY_OP_REF = "op://Private/Google AI/Gemini API key"
DEFAULT_MCP_PROTOCOL_VERSION = "2025-06-18"
_TOML_TRANSPORT_KEYS = frozenset({"command", "args"})


@dataclass(frozen=True)
class McpMigrationReport:
    changed: tuple[Path, ...] = ()
    skipped: tuple[tuple[Path, str], ...] = ()


def _config_backup_dir() -> Path:
    return Path.home() / ".local" / "share" / "brainlayer" / "config-backups"


def _backup_config_file(target_path: Path) -> Path:
    backup_dir = _config_backup_dir()
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d%H%M%S")
    digest = hashlib.sha256(str(target_path.resolve()).encode("utf-8")).hexdigest()[:12]
    backup = backup_dir / f"{target_path.name}.{digest}.{stamp}.{os.getpid()}.bak"
    backup.write_bytes(target_path.read_bytes())
    backup.chmod(target_path.stat().st_mode & 0o777)
    return backup


def ensure_spotlight_excluded_layout(
    *,
    data_dir: Path | None = None,
    env_file: Path | None = None,
    runtime_dir: Path | None = None,
    launchd_log_dir: Path | None = None,
    counter_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Create marker-backed roots for every high-churn BrainLayer runtime path."""
    return _ensure_spotlight_excluded_layout(
        data_dir=data_dir,
        env_file=env_file,
        runtime_dir=runtime_dir,
        launchd_log_dir=launchd_log_dir,
        counter_dir=counter_dir,
        resolve_db_path_fn=resolve_db_path,
        get_canonical_db_path_fn=get_canonical_db_path,
        home_fn=Path.home,
        ismount_fn=os.path.ismount,
    )


def get_default_env_file() -> Path:
    """Return the standard per-user BrainLayer env file."""
    return Path.home() / ".config" / "brainlayer" / "brainlayer.env"


def get_launchd_dir() -> Path:
    """Return the packaged launchd template directory, falling back to source checkout."""
    packaged = resources.files("brainlayer").joinpath("launchd")
    if packaged.is_dir():
        return Path(str(packaged))

    source = Path(__file__).resolve().parents[2] / "scripts" / "launchd"
    if source.is_dir():
        return source

    raise FileNotFoundError("BrainLayer launchd templates were not found")


def get_current_brainlayer_bin() -> str | None:
    """Return the current console-script path when setup was invoked by one."""
    argv0 = Path(sys.argv[0]).expanduser()
    if argv0.name == "brainlayer":
        candidate = argv0 if argv0.is_absolute() else Path.cwd() / argv0
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    found = shutil.which("brainlayer")
    if found:
        return found
    return None


def get_current_mcp_bridge_bin() -> str:
    """Return the stable opt path without resolving it to a versioned Cellar path."""
    return stable_mcp_bridge_command()


def _require_mcp_bridge(command: str) -> None:
    if risk := bridge_path_risk({"command": command}):
        raise FileNotFoundError(f"Install the Homebrew bridge at {command}: {risk['reason']}")


def get_default_mcp_config_paths() -> tuple[Path, ...]:
    """Configs BrainLayer owns and may rewrite on setup/upgrade."""
    return owned_mcp_config_paths()


def _validate_toml_text(text: str) -> None:
    import tomllib

    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"rewritten TOML is invalid: {exc}") from exc


def _strip_toml_table_assignments(body: str, keys: frozenset[str]) -> str:
    import tomllib

    lines = body.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(line)
            i += 1
            continue
        assignment = line
        while True:
            try:
                parsed = tomllib.loads(assignment)
                break
            except tomllib.TOMLDecodeError:
                i += 1
                if i == len(lines):
                    raise ValueError("could not safely parse a TOML assignment for migration") from None
                assignment += lines[i]
        if not keys.intersection(parsed):
            out.append(assignment)
        i += 1
    return "".join(out)


def _toml_literal(value: object) -> str:
    """Render parsed env values without treating their contents as source text."""
    if isinstance(value, (str, bool, int)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)} = {_toml_literal(v)}" for k, v in value.items()) + "}"
    from datetime import date, datetime
    from datetime import time as datetime_time

    if isinstance(value, (date, datetime, datetime_time)):
        return value.isoformat()
    raise ValueError("unsupported TOML env value; migration refused")


def _rewrite_codex_mcp_servers_to_socket(text: str, names: list[str], bridge_command: str) -> str:
    """Rewrite owned MCP blocks to the reconnecting bridge, retaining subtables."""
    import copy
    import tomllib

    payload = tomllib.loads(text)
    expected = copy.deepcopy(payload)
    result = text
    for name in names:
        for table in ("mcp_servers", "mcpServers"):
            server = payload.get(table, {}).get(name)
            if not needs_socket_migration(name, server) or socat_socket_conflict(server):
                continue
            pattern = re.compile(
                rf"(?ms)^(\[{re.escape(table)}\.{re.escape(name)}\]\s*\n)(.*?)(?=^\[|\Z)",
            )
            match = pattern.search(result)
            if not match:
                continue
            header, body = match.group(1), match.group(2)
            desired_env = socket_server_preserving(server, bridge_command=bridge_command).get("env")
            rewrite_env = desired_env != server.get("env")
            keys = _TOML_TRANSPORT_KEYS | ({"env"} if rewrite_env else set())
            kept_lines = _strip_toml_table_assignments(body, keys)
            env_line = f"env = {_toml_literal(desired_env)}\n" if rewrite_env else ""
            new_body = f"command = {json.dumps(bridge_command)}\n" + env_line + kept_lines
            result = result[: match.start()] + header + new_body + result[match.end() :]
            if rewrite_env:
                # Move a dedicated env table into the root's inline env map too.
                env_pattern = re.compile(rf"(?ms)^\[{re.escape(table)}\.{re.escape(name)}\.env\]\s*\n.*?(?=^\[|\Z)")
                result = env_pattern.sub("", result)
            changed_server = expected[table][name]
            changed_server["command"] = bridge_command
            changed_server.pop("args", None)
            if rewrite_env:
                changed_server["env"] = desired_env
    if tomllib.loads(result) != expected:
        raise ValueError("TOML migration could not preserve unrelated settings; no write performed")
    return result


def _rewrite_json_mcp_servers(payload: dict, bridge_command: str) -> bool:
    changed = False
    servers = payload.get("mcpServers")
    if isinstance(servers, dict):
        for name, server in list(servers.items()):
            if needs_socket_migration(str(name), server) and not socat_socket_conflict(server):
                servers[name] = socket_server_preserving(server, bridge_command=bridge_command)
                changed = True
    projects = payload.get("projects")
    if isinstance(projects, dict):
        for project_data in projects.values():
            if not isinstance(project_data, dict):
                continue
            nested = project_data.get("mcpServers")
            if not isinstance(nested, dict):
                continue
            for name, server in list(nested.items()):
                if needs_socket_migration(str(name), server) and not socat_socket_conflict(server):
                    nested[name] = socket_server_preserving(server, bridge_command=bridge_command)
                    changed = True
    return changed


def _toml_servers_needing_migration(text: str) -> list[str]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        return []
    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    return [name for name, server in iter_toml_mcp_servers(payload) if needs_socket_migration(name, server)]


def _socket_conflict_reasons(servers) -> list[str]:
    return [
        f"{name}: socat socket {risk['socat_socket_target']} conflicts with BRAINLAYER_MCP_SOCKET "
        f"{risk['bridge_socket_target']}; entry not migrated"
        for name, server in servers
        if needs_socket_migration(name, server) and (risk := socat_socket_conflict(server))
    ]


def migrate_legacy_mcp_configs(
    config_paths: list[Path] | tuple[Path, ...] | None = None,
    *,
    bridge_command: str | None = None,
) -> McpMigrationReport:
    """Rewrite owned BrainLayer MCP entries to the shipped reconnecting bridge.

    Name-matcher: only ``mcpServers`` keys in ``BRAINLAYER_SERVER_NAMES`` are
    rewritten unless already using the stable opt-path bridge.
    Each rewritten file is backed up under ~/.local/share/brainlayer/config-backups/.
    Unreadable/malformed inputs and failed writes are reported in ``skipped``.
    """
    bridge_command = bridge_command or get_current_mcp_bridge_bin()
    changed: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    paths = config_paths if config_paths is not None else get_default_mcp_config_paths()
    for config_path in paths:
        path = config_path.expanduser()
        if not path.is_file():
            continue
        try:
            target_path = path.resolve()
            original_stat = target_path.stat()
            original_text = target_path.read_text(encoding="utf-8")
        except OSError as exc:
            skipped.append((path, f"could not read config: {exc}"))
            continue

        suffix = target_path.suffix.lower()
        if suffix == ".toml" or target_path.name == "config.toml":
            names = _toml_servers_needing_migration(original_text)
            if not names:
                continue
            import tomllib

            servers = iter_toml_mcp_servers(tomllib.loads(original_text))
            skipped.extend((path, reason) for reason in _socket_conflict_reasons(servers))
            names = [
                name
                for name, server in servers
                if needs_socket_migration(name, server) and not socat_socket_conflict(server)
            ]
            if not names:
                continue
            try:
                new_text = _rewrite_codex_mcp_servers_to_socket(original_text, names, bridge_command)
            except ValueError as exc:
                skipped.append((path, str(exc)))
                continue
            if new_text == original_text:
                skipped.append((path, f"TOML servers {names!r} need migration but table headers were not rewritten"))
                continue
            try:
                _validate_toml_text(new_text)
            except ValueError as exc:
                skipped.append((path, str(exc)))
                continue
            _require_mcp_bridge(bridge_command)
            try:
                _backup_config_file(target_path)
                mode = target_path.stat().st_mode & 0o777
                if target_path.stat().st_mtime_ns != original_stat.st_mtime_ns:
                    skipped.append((path, "config changed during migration; skipped to avoid lost update"))
                    continue
                temporary = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
                try:
                    temporary.write_text(new_text, encoding="utf-8")
                    temporary.chmod(mode)
                    os.replace(temporary, target_path)
                finally:
                    temporary.unlink(missing_ok=True)
            except OSError as exc:
                skipped.append((path, f"could not write config: {exc}"))
                continue
            changed.append(path)
            continue

        try:
            payload = json.loads(original_text)
        except json.JSONDecodeError as exc:
            skipped.append((path, f"invalid JSON: {exc}"))
            continue
        if not isinstance(payload, dict):
            skipped.append((path, "MCP config root must be a JSON object"))
            continue
        skipped.extend((path, reason) for reason in _socket_conflict_reasons(iter_json_mcp_servers(payload)))
        try:
            rewritten = _rewrite_json_mcp_servers(payload, bridge_command)
        except ValueError as exc:
            skipped.append((path, str(exc)))
            continue
        if not rewritten:
            continue

        _require_mcp_bridge(bridge_command)
        try:
            _backup_config_file(target_path)
            mode = target_path.stat().st_mode & 0o777
            if target_path.stat().st_mtime_ns != original_stat.st_mtime_ns:
                skipped.append((path, "config changed during migration; skipped to avoid lost update"))
                continue
            temporary = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
            try:
                temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                temporary.chmod(mode)
                os.replace(temporary, target_path)
            finally:
                temporary.unlink(missing_ok=True)
        except OSError as exc:
            skipped.append((path, f"could not write config: {exc}"))
            continue
        changed.append(path)
    return McpMigrationReport(changed=tuple(changed), skipped=tuple(skipped))


def verify_mcp_transport(*, bridge_command: str | None = None, timeout_seconds: float = 10.0) -> int:
    """Require initialize + tools/list + one successful tool call on one fresh bridge."""
    resolved_bridge = bridge_command or get_current_mcp_bridge_bin()
    if not resolved_bridge:
        raise FileNotFoundError("brainlayer-mcp-stdio-bridge was not found at the pinned opt path")
    initialize_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": DEFAULT_MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "brainlayer-setup", "version": "1"},
        },
    }
    initialized_notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    deadline = time.monotonic() + timeout_seconds

    def send(process: subprocess.Popen[bytes], payload: dict[str, object]) -> None:
        assert process.stdin is not None
        process.stdin.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
        process.stdin.flush()

    def receive(process: subprocess.Popen[bytes], response_id: int) -> dict[str, object] | None:
        assert process.stdout is not None
        buffered = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                raise RuntimeError(f"MCP transport verification timed out after {timeout_seconds:g}s")
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                return None
            buffered.extend(chunk)
            while b"\n" in buffered:
                line, _, remainder = buffered.partition(b"\n")
                buffered = bytearray(remainder)
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("id") == response_id:
                    return payload

    try:
        process = subprocess.Popen(
            [resolved_bridge],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except OSError as exc:
        raise RuntimeError(f"could not start MCP bridge: {exc}") from exc

    try:
        send(process, initialize_request)
        initialize = receive(process, 1)
        if not initialize or not isinstance(initialize.get("result"), dict):
            raise RuntimeError("MCP initialize response missing or invalid")
        send(process, initialized_notification)
        send(process, tools_request)
        tools_response = receive(process, 2)
        tools_result = tools_response.get("result") if tools_response else None
        tools = tools_result.get("tools") if isinstance(tools_result, dict) else None
        if not isinstance(tools, list) or not tools:
            raise RuntimeError("MCP tools/list response missing tools")
        tool_names = {
            tool.get("name") for tool in tools if isinstance(tool, dict) and isinstance(tool.get("name"), str)
        }
        if "brain_recall" in tool_names:
            smoke_tool = "brain_recall"
            smoke_arguments = {"mode": "stats"}
        elif "expand_palette" in tool_names:
            smoke_tool = "expand_palette"
            smoke_arguments: dict[str, object] = {}
        else:
            raise RuntimeError("MCP tools/list has no safe deployment smoke tool")
        send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": smoke_tool, "arguments": smoke_arguments},
            },
        )
        call_response = receive(process, 3)
        call_result = call_response.get("result") if call_response else None
        if not isinstance(call_result, dict) or call_result.get("isError") is True:
            raise RuntimeError(f"MCP deployment tool call failed: {smoke_tool}")
        return len(tools)
    finally:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def ensure_brainlayer_env(
    env_file: Path | None = None,
    *,
    google_api_key_op_ref: str | None = None,
    overwrite_google_key: bool = False,
) -> Path:
    """Create or update brainlayer.env with defaults and an op-backed Google key."""
    target = env_file or get_default_env_file()

    if google_api_key_op_ref:
        write_gemini_env_file(
            target,
            google_api_key=google_api_key_op_ref,
            secret_source="1password",
            overwrite=overwrite_google_key,
            enrichment_env=DEFAULT_BRAINLAYER_CONFIG,
        )
        return target

    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BrainLayer private config.",
        "# Older installed jobs still require a 1Password-backed Google key until release:",
        f"# brainlayer setup --google-api-key-op-ref '{DEFAULT_GOOGLE_API_KEY_OP_REF}'",
        "",
    ]
    lines.extend(f"{key}={value}" for key, value in DEFAULT_BRAINLAYER_CONFIG.items())
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.chmod(0o600)
    return target


def install_launchd(
    target: str = "all",
    *,
    env_file: Path | None = None,
    launchd_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> None:
    """Run the packaged launchd installer, which owns bounded shutdown waits."""
    template_dir = launchd_dir or get_launchd_dir()
    install_script = template_dir / "install.sh"
    if not install_script.exists():
        raise FileNotFoundError(f"launchd installer not found: {install_script}")

    run_env = os.environ.copy()
    if extra_env:
        run_env.update(extra_env)
    run_env.setdefault("PYTHON_BIN", sys.executable)
    run_env.setdefault("BRAINLAYER_PYTHON", sys.executable)
    brainlayer_bin = get_current_brainlayer_bin()
    if brainlayer_bin is not None:
        run_env.setdefault("BRAINLAYER_BIN", brainlayer_bin)
    if env_file is not None:
        run_env["BRAINLAYER_ENV_FILE"] = str(env_file)

    try:
        subprocess.run([str(install_script), target], env=run_env, check=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        if timeout_seconds is None:
            raise TimeoutError("launchd installer timed out") from exc
        raise TimeoutError(f"launchd installer timed out after {timeout_seconds:g}s") from exc
