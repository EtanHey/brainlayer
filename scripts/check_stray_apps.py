"""Reject stray apps and bulky docs.local binaries inside supplied repo/worktree trees.

Only .build.noindex is disposable build output. docs.local (including old archives and
.noindex subdirectories) has no app exemption. Binary files over 1 MiB are identified
by archive suffix, NUL bytes or non-UTF-8 data in their first 8 KiB. Never follow links.
"""

import codecs
import os
import stat
import sys
from pathlib import Path

MAX_BINARY_BYTES = 1024 * 1024
BINARY_SUFFIXES = {".zip", ".gz", ".bz2", ".xz", ".tar", ".dmg", ".pkg", ".a", ".so", ".dylib"}


def large_binary(path: Path) -> bool:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= MAX_BINARY_BYTES:
        return False
    if path.suffix.lower() in BINARY_SUFFIXES:
        return True
    with path.open("rb") as stream:
        header = stream.read(8192)
    if b"\x00" in header:
        return True
    try:
        codecs.getincrementaldecoder("utf-8")().decode(header, final=False)
    except UnicodeDecodeError:
        return True
    return False


def find_stray_apps(*roots: Path) -> list[Path]:
    """Return prohibited bundles and docs.local binaries; inspect no paths outside roots."""
    strays = set()
    for root in roots:
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"Expected a repo/worktree directory: {root}")
        if root.is_relative_to("/Applications"):
            continue
        for directory, children, files in os.walk(root, followlinks=False, onerror=_raise_walk_error):
            here = Path(directory)
            in_docs = root.name == "docs.local" or "docs.local" in here.relative_to(root).parts
            for name in children[:]:
                path = here / name
                if name == ".git" or (name == ".build.noindex" and not in_docs):
                    children.remove(name)
                elif name.endswith(".app"):
                    strays.add(path)
                    children.remove(name)
            for name in files:
                path = here / name
                if name.endswith(".app") or (in_docs and large_binary(path)):
                    strays.add(path)
    return sorted(strays)


def _raise_walk_error(error: OSError) -> None:
    raise error


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: check_stray_apps.py <repo-or-worktree-root> [...]")
    found = find_stray_apps(*(Path(arg) for arg in sys.argv[1:]))
    for path in found:
        print(f"Prohibited artifact (use .build.noindex; docs.local keeps text receipts): {path}")
    sys.exit(bool(found))
