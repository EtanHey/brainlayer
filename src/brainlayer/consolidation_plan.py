"""Read-only preflight for the launchd consolidation transaction."""

import json
import os
import plistlib
import re
import shlex
import stat
from pathlib import Path

from .component_jobs import NON_ENV_RUN, _legacy_configured
from .hook_python import DEFAULT_KEG_PYTHON, HOOK_PYTHON_ENV, is_pinned_interpreter
from .supervisor_adapters import ALIASES, ARCHIVED
from .supervisor_jobs import PLIST_KEYS, resource_policy

NEW = ("com.brainlayer.daemon", "com.brainlayer.maintenance")
# Setuptools' standard shim is the sole trusted executable .pth line.
DISTUTILS_PTH = (
    "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = os.environ.get(var, 'local') == 'local'; "
    "enabled and __import__('_distutils_hack').add_shim();"
)
UNMANAGED = frozenset({"com.brainlayer.brainbar", "com.brainlayer.brainbar-daemon", "com.mcplayer.brainlayer-proxy"})
REVIVERS = frozenset(
    "com.brainlayer.health-check com.brainlayer.throughput-watchdog "
    "com.brainlayer.tier0-watchdog com.etanhey.brainlayer-fleet-watchdog".split()
)


def read_owned(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise ValueError(f"refusing non-owned, non-regular or linked migration input: {path}")
        return stream.read()


def disabled_labels(text):
    if "disabled services" not in text or "{" not in text or "}" not in text:
        raise ValueError("unrecognized launchd disabled-state output")
    body = text.split("{", 1)[1].rsplit("}", 1)[0]
    entries = []
    for line in body.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'\s*"([^"]+)"\s*=>\s*(true|false|disabled|enabled)\s*', line)
        if match is None:
            raise ValueError("unrecognized launchd disabled-state entry")
        entries.append(match.groups())
    return {label for label, value in entries if value in {"true", "disabled"}}


def interpreter_site(python):
    python = Path(python)
    if (
        not is_pinned_interpreter(shlex.quote(str(python)))
        or str(python) not in {DEFAULT_KEG_PYTHON, os.environ.get(HOOK_PYTHON_ENV)}
        or not python.is_file()
        or not os.access(python, os.X_OK)
    ):
        raise ValueError("consolidated interpreter must be executable keg python or BRAINLAYER_HOOK_PYTHON override")
    sites = list((python.parent.parent / "lib").glob("python*/site-packages"))
    if len(sites) != 1:
        raise ValueError(f"interpreter site-packages missing or ambiguous: {python}")
    site = sites[0]
    if site.is_symlink():
        raise ValueError(f"linked site-packages refused: {site}")
    for path in site.iterdir():
        if "editable" in path.name.lower():
            raise ValueError(f"editable interpreter refused: {path}")
        if path.suffix == ".pth":
            for line in read_owned(path).decode().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(("import ", "import\t")):
                    if path.name == "distutils-precedence.pth" and line == DISTUTILS_PTH:
                        continue
                    raise ValueError(f"untrusted executable interpreter .pth refused: {path}")
                if "editable" in line.lower() or (not (site / line).resolve().is_relative_to(site.resolve())):
                    raise ValueError(f"editable/external interpreter path refused: {path}")
    package = site / "brainlayer"
    if not package.is_dir() or package.is_symlink():
        raise ValueError(f"interpreter requires an installed brainlayer package: {site}")
    return site


def rendered_launchers(templates, home, python, env_file):
    root = str(interpreter_site(python))
    values = {
        "__HOME__": str(home),
        "__BRAINLAYER_PYTHON__": str(python),
        "__BRAINLAYER_ENV_FILE__": str(env_file),
        "__BRAINLAYER_DIR__": root,
    }

    def replace(value):
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            for key, item in values.items():
                value = value.replace(key, item)
            if "__BRAINLAYER_ENV_RUN__" in value:
                raise ValueError("launcher wrapper must be supplied per launcher")
            if re.search(r"__[A-Z_]+__", value):
                raise ValueError(f"unresolved launcher placeholder: {value}")
        return value

    result = {}
    for label, title in zip(NEW, ("BrainLayer Daemon", "BrainLayer Maintenance")):
        plist = plistlib.loads(read_owned(templates / f"{label}.plist"))
        plist["ProgramArguments"][0] = str(home / ".local/lib/brainlayer" / title)
        plist = replace(plist)
        if plist["Label"] != label or plist["ExitTimeOut"] < 25:
            raise ValueError("invalid consolidated launcher identity/shutdown budget")
        if plist.get("AssociatedBundleIdentifiers") != ["com.brainlayer.brainbar"]:
            raise ValueError("invalid consolidated launcher association")
        if label == NEW[0]:
            valid = plist.get("KeepAlive") is True and plist.get("RunAtLoad") is True
            valid = valid and "StartCalendarInterval" not in plist and "StartInterval" not in plist
        else:
            valid = plist.get("StartCalendarInterval") == {"Hour": 3, "Minute": 15}
            valid = valid and not plist.get("KeepAlive") and not plist.get("RunAtLoad")
            valid = valid and "StartInterval" not in plist
        if not valid:
            raise ValueError(f"invalid consolidated launch behavior: {label}")
        result[label] = plistlib.dumps(plist, sort_keys=False)
    return result


def collect_legacy(agents, env_file, disabled):
    """Snapshot every owned legacy file/marker and reject unsupported resident policies."""
    records, files = {}, {}
    env_contents = {}
    known = set(ALIASES) | ARCHIVED | UNMANAGED
    for path in sorted(agents.iterdir()):
        name = path.name[:-11] if path.name.casefold().endswith(".loaded-keg") else path.name
        if "brainlayer" not in name.casefold() or not (
            name.casefold().endswith(".plist") or path.name.casefold().endswith(".loaded-keg")
        ):
            continue
        label = name[:-6] if name.casefold().endswith(".plist") else name
        if label not in known:
            raise ValueError(f"unknown BrainLayer label {label}: {path}")
        if path.name not in {f"{label}.plist", f"{label}.plist.loaded-keg"}:
            raise ValueError(f"noncanonical BrainLayer label path {label}: {path}")
        if path.name.endswith(".plist"):
            plist = plistlib.loads(read_owned(path))
            if not isinstance(plist, dict) or plist.get("Label") != label:
                raise ValueError(f"legacy plist identity mismatch: {path}")
    for label in sorted(set(ALIASES) | ARCHIVED):
        path = agents / f"{label}.plist"
        if not os.path.lexists(path):
            if os.path.lexists(path.with_name(path.name + ".loaded-keg")):
                raise ValueError(f"orphan loaded-keg marker: {path}")
            continue
        raw = read_owned(path)
        plist = plistlib.loads(raw)
        if not isinstance(plist, dict) or plist.get("Label") != label:
            raise ValueError(f"legacy plist identity mismatch: {path}")
        files[path.name] = raw
        marker = path.with_name(path.name + ".loaded-keg")
        if os.path.lexists(marker):
            files[marker.name] = read_owned(marker)
        if label in ALIASES:
            if unsupported := set(plist) - PLIST_KEYS:
                raise ValueError(f"unsupported resident plist keys: {sorted(unsupported)}")
            resource_policy(plist)
            env = plist.get("EnvironmentVariables", {})
            env_path = Path(env.get("BRAINLAYER_ENV_FILE") or env_file).expanduser()
            if not env_path.is_absolute():
                raise ValueError(f"selected env file must be absolute: {env_path}")
            if env_path not in env_contents:
                try:
                    env_contents[env_path] = read_owned(env_path).decode("utf-8")
                except FileNotFoundError:
                    env_contents[env_path] = ""  # Missing snapshot; do not reopen during parsing.
            records[label] = {
                "enabled": label not in disabled
                and _legacy_configured(
                    env, env_path, contents=env_contents[env_path], env_run=label not in NON_ENV_RUN
                ),
                "plist": plist,
            }
    if not records:
        raise ValueError("no legacy resident/interval jobs to consolidate")
    # Preserve serializable configuration; never execute templates during preflight.
    json.dumps(records)
    return records, files


def snapshot_legacy(agents, env_file, domain, run):
    """Capture all owned/archived states and bytes for a journal before any bootout."""
    result = run(["launchctl", "print-disabled", domain])
    if result.returncode:
        raise RuntimeError("cannot measure launchd disabled state")
    disabled = disabled_labels(result.stdout)
    records, files = collect_legacy(agents, env_file, disabled)
    states = {}
    for name in sorted(files):
        if not name.endswith(".plist"):
            continue
        label = name.removesuffix(".plist")
        result = run(["launchctl", "print", f"{domain}/{label}"])
        pid = None
        if result.returncode == 113 and "Could not find service" in result.stdout + result.stderr:
            loaded = False
        elif result.returncode == 0:
            loaded = True
            pids = [value.strip() for value in re.findall(r"(?m)^[ \t]*pid[ \t]*=[ \t]*(.*)$", result.stdout)]
            if len(pids) > 1 or pids and not re.fullmatch(r"[1-9][0-9]*", pids[0]):
                raise RuntimeError(f"cannot measure launchd process state: {label}")
            pid = int(pids[0]) if pids else None
        else:
            raise RuntimeError(f"cannot measure launchd loaded state: {label}")
        states[label] = {"loaded": loaded, "disabled": label in disabled, "pid": pid}
    return {
        "schema_version": 1,
        "components": records,
        "files": {k: v.hex() for k, v in files.items()},
        "states": states,
    }
