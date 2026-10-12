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
import time
from pathlib import Path

from .paths import supervisor_socket_path
from .service_supervisor import Component, Supervisor
from .supervisor_adapters import ALIASES


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
        )
        if interval and not run_at_load:
            child.retry_at = time.monotonic() + interval
        components[ALIASES[label]] = child
    return components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--socket", type=Path, default=None)
    args = parser.parse_args()
    path = args.socket if args.socket is not None else supervisor_socket_path()
    asyncio.run(Supervisor(load_components(args.manifest, path), path).run(install_signals=True))


if __name__ == "__main__":
    main()
