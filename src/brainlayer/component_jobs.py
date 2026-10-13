"""Component actions retain legacy jobs until their plists are archived."""

import os
import plistlib
from pathlib import Path

from .config import configured_brainlayer_env_value, get_user_env_path
from .jobs import _pid, _run
from .launchd_primitive import is_launchd_label_disabled
from .paths import supervisor_socket_path
from .pause import DEFAULT_PAUSE_SENTINEL_PATH, pause_applies_to_label, pause_sentinel_state
from .supervisor_adapters import ALIASES, SupervisorCommands
from .supervisor_jobs import load_components

NON_ENV_RUN = frozenset({"com.brainlayer.tier0-watchdog", "com.etanhey.brainlayer-fleet-watchdog"})


def _legacy_configured(env, env_path, *, contents=None, env_run=True):
    """Mirror env-run's disable gates for both restart and configured status."""

    if not env_run:
        return True

    def value(key):
        configured = configured_brainlayer_env_value(key, env_path, contents=contents)
        return env.get(key) if configured is None else configured

    if value("BRAINLAYER_SKIP_DISABLE_GATES") == "1":
        return True
    service = env.get("BRAINLAYER_LAUNCHD_SERVICE", "")
    keys = ["BRAINLAYER_SYSTEM_ENABLED"]
    if service:
        key = "HOTLANE" if service == "hotlane-brainbar" else service.replace("-", "_").upper()
        keys.append(f"BRAINLAYER_LAUNCHD_{key}_ENABLED")
    false_values = "0 false False FALSE no No NO off Off OFF disabled Disabled DISABLED".split()
    return not any(value(key) in false_values for key in keys)


def component_action(name, action, *, agents=None, manifest=None, path=None, runner=_run):
    label = name if name in ALIASES else next((k for k, v in ALIASES.items() if v == name), None)
    if label is None or action not in {"status", "restart", "env", "logs"}:
        raise ValueError("unknown component/action")
    name = ALIASES[label]
    agents = agents if agents is not None else Path.home() / "Library/LaunchAgents"
    path = path if path is not None else supervisor_socket_path()
    manifest = manifest if manifest is not None else path.with_name("jobs.json")
    legacy = agents / f"{label}.plist"
    target = f"gui/{os.getuid()}/{label}"
    if legacy.exists():
        with legacy.open("rb") as handle:
            plist = plistlib.load(handle)
        if not isinstance(plist, dict) or plist.get("Label") != label:
            raise ValueError("legacy plist label mismatch")
        env = plist.get("EnvironmentVariables", {})
        if not isinstance(env, dict):
            raise ValueError("legacy environment must be an object")
        source = "launchd"
        out, err = plist.get("StandardOutPath"), plist.get("StandardErrorPath")
    else:
        source = "supervisor"
        if action in {"env", "logs"}:
            child = load_components(manifest, path)[name]
            env, out, err = (
                child.env,
                str(child.stdout) if child.stdout else None,
                str(child.stderr) if child.stderr else None,
            )
        else:
            env, out, err = {}, None, None
    result = dict(
        source=source,
        label=label,
        component=name,
        provenance={"definition": str(legacy if source == "launchd" else manifest)},
    )
    env_path = Path(env["BRAINLAYER_ENV_FILE"]) if env.get("BRAINLAYER_ENV_FILE") else get_user_env_path()
    if action == "env":
        return result | dict(env_file=str(env_path))  # Paths only; never print credentials.
    if action == "logs":
        return result | dict(stdout=out, stderr=err)
    pause_path = Path(os.environ.get("BRAINLAYER_PAUSE_SENTINEL", str(DEFAULT_PAUSE_SENTINEL_PATH)))
    client = SupervisorCommands(path, runner, pause_path=pause_path)
    if action == "restart":
        if source == "launchd":
            if is_launchd_label_disabled(label, command_runner=runner) is not False:
                raise RuntimeError("component disabled state is disabled or unavailable")
            payload, active, _ = pause_sentinel_state(pause_path, strict=True)
            if active and pause_applies_to_label(payload, label):
                raise RuntimeError("component is paused")
            if not _legacy_configured(env, env_path, env_run=label not in NON_ENV_RUN):
                raise RuntimeError("component is configured OFF")
            command = runner(["launchctl", "kickstart", "-k", target])
        else:
            command = client(["launchctl", "kickstart", "-k", target])
        if command.returncode:
            raise RuntimeError(command.stderr or "component restart failed")
    if source == "supervisor":
        return result | dict(runtime=client.request()[name])
    printed = runner(["launchctl", "print", target])
    if printed.returncode not in {0, 113}:
        raise RuntimeError(printed.stderr or "legacy job status unavailable")
    return result | dict(
        runtime=dict(
            configured=_legacy_configured(env, env_path, env_run=label not in NON_ENV_RUN),
            loaded=printed.returncode == 0,
            pid=_pid(printed.stdout),
        )
    )
