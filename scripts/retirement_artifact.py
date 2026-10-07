"""Immutable candidate wheel fixture. Never builds from a mutable working tree."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private_env(home: Path) -> dict[str, str]:
    home.mkdir(parents=True, exist_ok=True)
    (home / "tmp").mkdir(exist_ok=True)
    return {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TMPDIR": str(home / "tmp"),
        "XDG_CONFIG_HOME": str(home / "config"),
        "XDG_CACHE_HOME": str(home / "cache"),
        "PIP_CONFIG_FILE": "/dev/null",
        "PIP_INDEX_URL": "https://pypi.org/simple",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "BRAINLAYER_FORBID_EMBEDDING_MODEL": "1",
        "BRAINLAYER_FORBID_BRAINBAR_SOCKET": "1",
        "BRAINLAYER_DB": str(home / "fixture.db"),
    }


def command(args: list[str], cwd: Path, env: dict, log: Path, timeout=1200) -> None:
    with log.open("w") as handle:
        result = subprocess.run(args, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log.name}")


def snapshot(repo: Path, sha: str, destination: Path) -> dict:
    env = private_env(destination.parent / "build-home")
    full = subprocess.check_output(["git", "rev-parse", sha + "^{commit}"], cwd=repo, env=env, text=True).strip()
    if full != sha or len(sha) != 40:
        raise ValueError("Snapshot requires full immutable commit SHA")
    destination.mkdir()
    archive = destination.parent / "source.tar"
    command(
        ["git", "archive", "--format=tar", "--output", str(archive), sha], repo, env, destination.parent / "archive.log"
    )
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--full-tree", sha], cwd=repo, env=env, text=True)
    manifest, links = {}, {}
    for line in tracked.splitlines():
        identity, filename = line.split("\t", 1)
        mode, kind, blob = identity.split()
        if kind != "blob":
            raise ValueError("Unsupported source tree entry: " + filename)
        path = destination / filename
        if path.is_symlink() and not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Source link escapes snapshot: " + filename)
        data = str(path.readlink()).encode() if mode == "120000" else path.read_bytes()
        if mode == "120000":
            links[filename] = {
                "target": str(path.readlink()),
                "resolved": path.resolve().relative_to(destination.resolve()).as_posix(),
            }
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != blob:
            raise ValueError("Source blob mismatch: " + filename)
        manifest[filename] = hashlib.sha256(data).hexdigest()
    tree = subprocess.check_output(["git", "rev-parse", sha + "^{tree}"], cwd=repo, env=env, text=True).strip()
    return {"sha": sha, "tree": tree, "archive_sha256": digest(archive), "files": manifest, "links": links}


def build_wheel(source: Path, work: Path, sha: str) -> dict:
    env = private_env(work / "build-home")
    build = work / "build-venv"
    command([sys.executable, "-I", "-m", "venv", str(build)], work, env, work / "build-venv.log")
    python = str(build / "bin/python")
    command([python, "-I", "-m", "pip", "install", "build", "hatchling"], work, env, work / "build-tools.log")
    stamp = source / "src/brainlayer/_build.py"
    if stamp.exists():
        raise ValueError("Archive unexpectedly contains generated stamp")
    stamp.write_text(f'BUILD_SHA = "{sha}"\n')
    command(
        [python, "-I", "-m", "build", "--wheel", "--no-isolation", "--outdir", str(work / "dist")],
        source,
        env,
        work / "build.log",
    )
    wheels = list((work / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Missing or ambiguous wheel")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as zipped:
        members = zipped.namelist()
        if "brainlayer/_build.py" not in members or any(
            "docs.local/" in name or name.endswith("brainlayer.db") for name in members
        ):
            raise ValueError("Wheel stamp/privacy inventory invalid")
    return {
        "path": str(wheel),
        "sha256": digest(wheel),
        "members": members,
        "stamp": "validation-generated _build.py, bound to source SHA",
    }


def install_profile(wheel: Path, profile: str, work: Path) -> dict:
    if profile not in {"default", "dev"}:
        raise ValueError("Unknown dependency profile")
    home = work / (profile + "-home")
    env = private_env(home)
    runtime = work / (profile + "-venv")
    command([sys.executable, "-I", "-m", "venv", str(runtime)], work, env, work / (profile + "-venv.log"))
    python = str(runtime / "bin/python")
    spec = str(wheel) + ("[dev]" if profile == "dev" else "")
    command([python, "-I", "-m", "pip", "install", spec], work, env, work / (profile + "-install.log"))
    command([python, "-I", "-m", "pip", "check"], work, env, work / (profile + "-pip-check.log"))
    return {"profile": profile, "python": python, "prefix": str(runtime), "home": str(home)}
