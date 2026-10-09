#!/usr/bin/env python3
"""Render/install account LaunchAgents for an existing signed app; never activate jobs."""
import argparse
import json
import os
import plistlib
import pwd
import stat
import subprocess
import tempfile
from pathlib import Path

LABELS = {"com.brainlayer.brainbar": "BrainBar", "com.brainlayer.brainbar-daemon": "BrainBarDaemon"}


def owned(path, home, uid, missing=False, python_link=False, socket=False):
    """Check all home-relative components; an explicit venv Python may be a symlink."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(home):
        raise ValueError(f"path must be absolute and inside current account home: {path}")
    chain = [home, *reversed(list(path.parents))]
    for item in dict.fromkeys([p for p in chain if p.is_relative_to(home)] + [path]):
        try:
            info = item.lstat()
        except FileNotFoundError:
            if missing:
                continue
            raise ValueError(f"missing path: {item}") from None
        is_python_link = python_link and item == path and stat.S_ISLNK(info.st_mode)
        if info.st_uid != uid or (not is_python_link and
                (stat.S_ISLNK(info.st_mode) or info.st_mode & 0o022)):
            raise ValueError(f"foreign or unsafe path: {item}")
        allowed_leaf = stat.S_ISREG(info.st_mode) or (socket and stat.S_ISSOCK(info.st_mode))
        if not is_python_link and not stat.S_ISDIR(info.st_mode) and (item != path or not allowed_leaf or info.st_nlink != 1):
            raise ValueError(f"unsafe parent or multiply linked file: {item}")
    return path


def endpoint(path, home, uid):
    path = Path(path)
    if path == Path("/tmp/brainbar.sock"):
        for item in (path, Path(str(path) + ".lock")):
            if os.path.lexists(item):
                info = item.lstat()
                if info.st_uid != uid or info.st_mode & 0o077 or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
                    raise ValueError(f"foreign or unsafe default endpoint: {item}")
                if not (stat.S_ISSOCK(info.st_mode) if item == path else stat.S_ISREG(info.st_mode)):
                    raise ValueError(f"foreign or unsafe default endpoint: {item}")
    else:
        owned(path, home, uid, missing=True, socket=True)
        owned(Path(str(path) + ".lock"), home, uid, missing=True)
    if len(os.fsencode(path)) >= 104:
        raise ValueError("socket path exceeds macOS sockaddr_un limit")
    return path


def render(template, app, home, environment):
    """Reuse shipped plist keys; replace string placeholders structurally, never with sed."""
    def expand(value):
        if isinstance(value, str):
            return value.replace("/Applications/BrainBar.app", str(app)).replace("__HOME__", str(home))
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value
    result = expand(template)
    result["EnvironmentVariables"] = environment.copy()
    return result


def prepare(args, home, uid, verify):
    home = Path(home)
    if uid == 0 or uid != os.geteuid():
        raise ValueError("run as the intended non-root account without sudo")
    app = Path(args.app)
    if app.name != "BrainBar.app":
        raise ValueError("expected a named BrainBar.app bundle")
    if app == Path("/Applications/BrainBar.app"):
        info = app.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or info.st_mode & 0o022:
            raise ValueError("canonical app is foreign or unsafe")
    else:
        owned(app, home, uid)
    owned(app / "Contents/Info.plist", app, uid)
    verify(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)])
    with (app / "Contents/Info.plist").open("rb") as stream:
        if plistlib.load(stream).get("CFBundleIdentifier") != "com.brainlayer.brainbar":
            raise ValueError("unexpected app identity")
    python = owned(args.python, home, uid, python_link=True)
    cli = owned(args.cli, home, uid)
    if not python.is_file() or not cli.is_file() or not os.access(python, os.X_OK) or not os.access(cli, os.X_OK):
        raise ValueError("explicit installed Python/CLI must be executable")
    socket = endpoint(args.socket, home, uid)
    db = owned(args.db or home / ".local/share/brainlayer/brainlayer.db", home, uid, missing=True)
    envfile = owned(home / ".config/brainlayer/brainlayer.env", home, uid, missing=True)
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "BRAINBAR_SOCKET_PATH": str(socket),
           "BRAINLAYER_MCP_SOCKET": str(socket), "BRAINBAR_DB_PATH": str(db), "BRAINLAYER_DB": str(db),
           "BRAINBAR_PYTHON": str(python), "BRAINLAYER_CLI": str(cli), "BRAINLAYER_ENV_FILE": str(envfile)}
    plists = {}
    for label, binary in LABELS.items():
        executable = app / "Contents/MacOS" / binary
        owned(executable, app, uid)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"missing app executable: {binary}")
        template_path = owned(app / "Contents/Resources/LaunchAgents" / (label + ".plist"), app, uid)
        with template_path.open("rb") as stream:
            template = plistlib.load(stream)
        if template.get("Label") != label or template.get("ProgramArguments") != [f"/Applications/BrainBar.app/Contents/MacOS/{binary}"]:
            raise ValueError("unsupported LaunchAgent template")
        item = render(template, app, home, env)
        for key in ("StandardOutPath", "StandardErrorPath"):
            owned(item[key], home, uid, missing=True)
        owned(home / "Library/LaunchAgents" / (label + ".plist"), home, uid, missing=True)
        plists[label + ".plist"] = item
    return plists


def install(plists, home, uid):
    # No job operations or socket cleanup. Caller reviews rendered bytes before activation.
    directory = owned(home / "Library/LaunchAgents", home, uid, missing=True)
    parents = [directory]
    for item in plists.values():
        parents += [Path(item[key]).parent for key in ("StandardOutPath", "StandardErrorPath")]
        parents += [Path(item["EnvironmentVariables"][key]).parent for key in ("BRAINBAR_SOCKET_PATH", "BRAINBAR_DB_PATH")
                    if Path(item["EnvironmentVariables"][key]).is_relative_to(home)]
    for parent in parents:
        owned(parent, home, uid, missing=True)
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, item in plists.items():
        target = owned(directory / name, home, uid, missing=True)
        fd, temporary = tempfile.mkstemp(prefix=".brainbar-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                plistlib.dump(item, stream)
            owned(target, home, uid, missing=True)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("app", "python", "cli"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--socket", default="/tmp/brainbar.sock")
    parser.add_argument("--db")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render-only", action="store_true", help="default: JSON plists to stdout, no writes")
    mode.add_argument("--install", action="store_true", help="install own plists/directories only; no activation")
    args = parser.parse_args()
    home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    try:
        plists = prepare(args, home, os.getuid(), lambda command: subprocess.run(command, check=True))
        if args.install:
            install(plists, home, os.getuid())
        print(json.dumps({"domain": f"gui/{os.getuid()}", "activated": False, "plists": plists}, indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError, plistlib.InvalidFileException) as error:
        parser.exit(1, f"brainbar-install-services: {error}\n")


if __name__ == "__main__":
    main()
