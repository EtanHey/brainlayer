"""Read-only source manifest shared by the native runner and Linux consumer."""

import hashlib
import os
import subprocess
from pathlib import Path


def source_identity(root: Path) -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"

    def git(*args: str) -> bytes:
        return subprocess.run(["/usr/bin/git", *args], cwd=root, env=env, check=True, capture_output=True).stdout

    actual = Path(os.fsdecode(git("rev-parse", "--show-toplevel").strip())).resolve()
    if actual != root.resolve():
        raise ValueError("source root must be the checkout root")
    names = sorted(filter(None, git("ls-files", "-z").split(b"\0")))
    manifest = hashlib.sha256()
    for name in names:
        path = actual / os.fsdecode(name)
        data = os.fsencode(os.readlink(path)) if path.is_symlink() else path.read_bytes()
        manifest.update(name + b"\0" + hashlib.sha256(data).hexdigest().encode() + b"\n")
    return {
        "schema_version": 1,
        "root": str(actual),
        "head": git("rev-parse", "HEAD").decode().strip(),
        "tree": git("rev-parse", "HEAD^{tree}").decode().strip(),
        "dirty": bool(git("status", "--porcelain", "-z", "--untracked-files=all")),
        "source_sha256": manifest.hexdigest(),
        "source_files": len(names),
    }
