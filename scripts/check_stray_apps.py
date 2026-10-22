"""Reject stray apps and all docs.local binaries inside supplied repo/worktree trees.

Only .build.noindex is disposable build output. docs.local (including old archives and
.noindex subdirectories) has no app exemption. Binary files of any size are identified
by known binary suffixes, non-text control bytes or non-UTF-8 data anywhere in the
file. Never follow links.
"""

import codecs
import os
import stat
import sys
from pathlib import Path

BINARY_SUFFIXES = {
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".tar",
    ".dmg",
    ".pkg",
    ".a",
    ".so",
    ".dylib",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".heic",
    ".ico",
    ".pdf",
    ".db",
    ".sqlite",
    ".mp3",
    ".mp4",
    ".mov",
    ".wav",
    ".bin",
}
# Tabs, line endings, form feed, backspace and ANSI escapes can occur in text logs.
NON_TEXT_BYTES = bytes(value for value in range(32) if value not in b"\t\n\r\f\b\x1b") + b"\x7f"


def is_binary(path: Path) -> bool:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        return False
    if path.suffix.lower() in BINARY_SUFFIXES:
        return True
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with path.open("rb") as stream:
            while block := stream.read(8192):
                if any(value in block for value in NON_TEXT_BYTES):
                    return True
                decoder.decode(block, final=False)
            decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return True
    return False


def find_stray_apps(*roots: Path, include_worktrees: bool = True) -> list[Path]:
    """Find prohibited artifacts; CLI sweeps include worktrees, pytest owns only its checkout."""
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
                elif not include_worktrees and here == root and name == ".worktrees":
                    children.remove(name)
                elif name.endswith(".app"):
                    strays.add(path)
                    children.remove(name)
            for name in files:
                path = here / name
                if name.endswith(".app") or (in_docs and is_binary(path)):
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
