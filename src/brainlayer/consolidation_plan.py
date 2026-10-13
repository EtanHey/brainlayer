"""Read-only preflight for the launchd consolidation transaction."""

import json
import os
import plistlib
import re
import shlex
import stat
from pathlib import Path

from .component_jobs import _legacy_configured
from .hook_python import is_pinned_interpreter
from .supervisor_adapters import ALIASES, ARCHIVED
from .supervisor_jobs import PLIST_KEYS, resource_policy

NEW = ("com.brainlayer.daemon", "com.brainlayer.maintenance")
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


def rendered_launchers(templates, home, python, env_file):
    if (
        not is_pinned_interpreter(shlex.quote(str(python)))
        or not Path(python).is_file()
        or not os.access(python, os.X_OK)
    ):
        raise ValueError("consolidated interpreter must be an explicit executable keg/venv python")
    root = str(Path(python).parents[3])
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
        result[label] = plistlib.dumps(plist, sort_keys=False)
    return result


def collect_legacy(agents, env_file, disabled):
    """Snapshot every owned legacy file/marker and reject unsupported resident policies."""
    records, files = {}, {}
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
            records[label] = {
                "enabled": label not in disabled
                and _legacy_configured(plist.get("EnvironmentVariables", {}), env_file),
                "plist": plist,
            }
    if not records:
        raise ValueError("no legacy resident/interval jobs to consolidate")
    # Preserve serializable configuration; never execute templates during preflight.
    json.dumps(records)
    return records, files
