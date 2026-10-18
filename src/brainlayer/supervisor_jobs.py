"""Run an explicit, owner-only snapshot of installed resident/interval jobs.

PR5 writes this manifest before archiving the replaced plists. There is no discovery
or activation on import: argv, env gates, wrappers and logs come from rendered plists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import stat
import sys
import time
from pathlib import Path

from .paths import supervisor_socket_path
from .service_supervisor import Component, Supervisor
from .supervisor_adapters import ALIASES

PLIST_KEYS = set(
    "Label ProgramArguments EnvironmentVariables StandardOutPath StandardErrorPath "
    "WorkingDirectory StartInterval RunAtLoad KeepAlive ThrottleInterval ExitTimeOut "
    "AssociatedBundleIdentifiers SoftResourceLimits Nice LowPriorityIO ProcessType".split()
)


def resource_policy(plist):
    policy = {k: plist[k] for k in ("SoftResourceLimits", "Nice", "LowPriorityIO", "ProcessType") if k in plist}
    limits = policy.get("SoftResourceLimits", {})
    if not isinstance(limits, dict) or set(limits) - {"NumberOfFiles"}:
        raise ValueError(f"unsupported SoftResourceLimits: {limits}")
    if "NumberOfFiles" in limits and (type(limits["NumberOfFiles"]) is not int or limits["NumberOfFiles"] < 1):
        raise ValueError("invalid SoftResourceLimits.NumberOfFiles")
    if "Nice" in policy and (type(policy["Nice"]) is not int or not -20 <= policy["Nice"] <= 20):
        raise ValueError("invalid Nice")
    if "LowPriorityIO" in policy and type(policy["LowPriorityIO"]) is not bool:
        raise ValueError("invalid LowPriorityIO")
    if "ProcessType" in policy and (
        not isinstance(policy["ProcessType"], str) or policy["ProcessType"] not in {"Standard", "Background"}
    ):
        raise ValueError("unsupported ProcessType")
    return policy


def load_components(path: Path, socket_path: Path) -> dict[str, Component]:
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "r") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("job manifest must be an owner-only regular file")
        document = json.load(stream)
    if not isinstance(document, dict) or not document:
        raise ValueError("job manifest must be a nonempty object")
    components = {}
    for label, record in document.items():
        if label not in ALIASES or not isinstance(record, dict) or type(record.get("enabled")) is not bool:
            raise ValueError(f"unknown job or configured state: {label}")
        plist = record.get("plist")
        if not isinstance(plist, dict) or plist.get("Label") != label or "StartCalendarInterval" in plist:
            raise ValueError(f"invalid resident/interval plist: {label}")
        if unknown := set(plist) - PLIST_KEYS:
            raise ValueError(f"unsupported plist keys: {sorted(unknown)}")
        if "AssociatedBundleIdentifiers" in plist and plist["AssociatedBundleIdentifiers"] != [
            "com.brainlayer.brainbar"
        ]:
            raise ValueError("unsupported AssociatedBundleIdentifiers; consolidated launcher owns BrainBar association")
        argv, env = plist.get("ProgramArguments"), plist.get("EnvironmentVariables", {})
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(a, str) and a for a in argv)
            or not isinstance(env, dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
        ):
            raise ValueError(f"invalid argv/env: {label}")
        interval = plist.get("StartInterval", 0)
        run_at_load = plist.get("RunAtLoad", False)
        if type(run_at_load) is not bool:
            raise ValueError(f"invalid RunAtLoad: {label}")
        if not interval and plist.get("KeepAlive") is not True:
            raise ValueError(f"resident requires KeepAlive=true: {label}")
        if "KeepAlive" in plist and type(plist["KeepAlive"]) is not bool:
            raise ValueError(f"unsupported KeepAlive: {label}")
        if (
            type(interval) not in {int, float}
            or not math.isfinite(interval)
            or interval < 0
            or ("StartInterval" in plist and interval == 0)
        ):
            raise ValueError(f"invalid interval: {label}")
        grace = plist.get("ExitTimeOut", 60)
        if type(grace) not in {int, float} or not math.isfinite(grace) or grace < 0:
            raise ValueError(f"invalid exit timeout: {label}")
        timeout = record.get("run_timeout", max(300, grace))
        if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"invalid run timeout: {label}")
        throttle = plist.get("ThrottleInterval", 10)
        if type(throttle) not in {int, float} or not math.isfinite(throttle) or throttle <= 0:
            raise ValueError(f"invalid ThrottleInterval: {label}")

        def location(key):
            value = plist.get(key)
            if value is None:
                return None
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"{key} must be absolute: {label}")
            return Path(value)

        child = Component(
            tuple(argv),
            env | {"BRAINLAYER_SUPERVISOR_SOCKET": str(socket_path)},
            location("StandardOutPath"),
            location("StandardErrorPath"),
            enabled=record["enabled"],
            exit_timeout=grace,
            cwd=location("WorkingDirectory"),
            interval=interval,
            run_timeout=timeout,
            policy=resource_policy(plist),
            backoff=throttle,
        )
        if interval and not run_at_load:
            child.retry_at = time.monotonic() + interval
        components[ALIASES[label]] = child
    return with_brainbar_monitor(components, socket_path)


def with_brainbar_monitor(components, path):
    if sys.platform == "darwin":
        observability_env = components.get("observability", Component(("unused",))).env
        monitor_env = {
            key: observability_env[key]
            for key in ("BRAINLAYER_DB", "BRAINLAYER_JOB_ALERT_PATH")
            if key in observability_env
        }
        if "BRAINLAYER_FORBID_BRAINBAR_SOCKET" in os.environ:
            monitor_env["BRAINLAYER_FORBID_BRAINBAR_SOCKET"] = os.environ["BRAINLAYER_FORBID_BRAINBAR_SOCKET"]
        components["brainbar-activation"] = Component(
            (
                sys.executable,
                "-m",
                "brainlayer.brainbar_auto_activation",
                "--state",
                str(path.with_name("brainbar-activation.json")),
            ),
            env=monitor_env,
            interval=5,
            run_timeout=180,
            exit_timeout=2,
            stdout=path.with_name("brainbar-activation.log"),
            stderr=path.with_name("brainbar-activation.err.log"),
        )
    return components


def run_daemon(manifest: Path, socket: Path | None = None):
    path = socket if socket is not None else supervisor_socket_path()
    asyncio.run(Supervisor(load_components(manifest, path), path).run(install_signals=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--socket", type=Path, default=None)
    args = parser.parse_args()
    run_daemon(args.manifest, args.socket)


if __name__ == "__main__":
    main()
