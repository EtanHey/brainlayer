"""Explicit installer entrypoint; transaction/revert custody stays in PR-B/PR-C."""

import argparse
import fcntl
import os
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .consolidation_plan import read_owned, rendered_launchers
from .migration_custody import migration_lock

WRAPPER = b'#!/bin/sh\nset -eu\nrunner_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)\nexec "$runner_dir/brainlayer-env-run.sh" "$@"\n'


@contextmanager
def installer_lock(home):
    """Share directory custody with legacy install.sh without making runtime roots."""
    if home.resolve() != home.absolute():
        raise ValueError("linked installer HOME refused")
    fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ValueError("installer HOME custody refused")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def command(argv):
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def install(*, home, **options):
    with installer_lock(home):
        return _install(home=home, **options)


def _install(*, home, templates, env_file, python, keg, domain, run=command):
    from .launchd_consolidation import migrate

    if not keg:
        raise ValueError("consolidation requires an identified keg")
    # Rendering validates the installed interpreter, plist identity and schedule.
    prepared = rendered_launchers(templates, home, python, env_file)
    helpers = {
        name: read_owned(templates / name) for name in ("brainlayer-env-run.sh", "backup-daily.sh", "jsonl-backup.sh")
    }
    helpers.update({"BrainLayer Daemon": WRAPPER, "BrainLayer Maintenance": WRAPPER})
    for raw in helpers.values():
        subprocess.run(["/bin/bash", "-n"], input=raw, capture_output=True, check=True)
    if os.uname().sysname == "Darwin":
        for raw in prepared.values():
            subprocess.run(["/usr/bin/plutil", "-lint", "--", "-"], input=raw, capture_output=True, check=True)
    lib = home / ".local/lib/brainlayer"
    with migration_lock(home):
        if lib.resolve() != lib.absolute():
            raise ValueError("helper directory must not traverse symlinks")
        lib.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = lib.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ValueError("helper directory must be owned and not writable by others")
        # Check every shared helper before creating any. Never replace live legacy code.
        missing = []
        for name, raw in helpers.items():
            path = lib / name
            if os.path.lexists(path):
                if read_owned(path) != raw or stat.S_IMODE(path.stat().st_mode) != 0o755:
                    raise RuntimeError(f"helper differs; explicit operator reconciliation required: {path}")
            else:
                missing.append((path, raw))
        for path, raw in missing:
            fd, staging = tempfile.mkstemp(prefix=".install-", dir=lib)
            try:
                with os.fdopen(fd, "wb") as stream:
                    os.fchmod(stream.fileno(), 0o755)
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(staging, path)  # Refuse any concurrent arrival, including equal bytes.
            finally:
                os.unlink(staging)
        (home / "Library/LaunchAgents").mkdir(parents=True, exist_ok=True)
        (home / "Library/Logs/brainlayer").mkdir(parents=True, exist_ok=True)
    return migrate(home=home, templates=templates, env_file=env_file, python=python, keg=keg, domain=domain, run=run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("consolidate", "consolidation-rollback"))
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--domain", default=f"gui/{os.getuid()}")
    parser.add_argument("--templates", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--keg")
    args = parser.parse_args(argv)
    if args.action == "consolidation-rollback":
        from .launchd_consolidation import revert  # PR-C; never substitute a legacy reinstall.

        with installer_lock(args.home):
            revert(
                home=args.home,
                journal=args.home / ".local/share/brainlayer/supervisor/migration.json",
                domain=args.domain,
                run=command,
            )
    else:
        if any(value is None for value in (args.templates, args.env_file, args.python, args.keg)):
            parser.error("consolidate requires --templates, --env-file, --python and --keg")
        install(
            home=args.home,
            templates=args.templates,
            env_file=args.env_file,
            python=args.python,
            keg=args.keg,
            domain=args.domain,
        )
    print(f"{args.action}: verified transaction finished")


if __name__ == "__main__":
    main()
