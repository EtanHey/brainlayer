"""Reject indexable app bundles in explicitly supplied repo/temporary trees."""

import os
import sys
from pathlib import Path


def spotlight_excluded(path: Path) -> bool:
    return path.is_relative_to("/Applications") or any(part.endswith(".noindex") for part in path.parts)


def find_stray_apps(*roots: Path) -> list[Path]:
    strays = set()
    for root in roots:
        root = root.resolve()
        if spotlight_excluded(root):
            continue
        for directory, children, _ in os.walk(root, followlinks=False, onerror=_raise_walk_error):
            for name in children[:]:
                path = Path(directory) / name
                if name == ".git" or name.endswith(".noindex"):
                    children.remove(name)
                elif name.endswith(".app"):
                    strays.add(path)
                    children.remove(name)
    return sorted(strays)


def _raise_walk_error(error: OSError) -> None:
    raise error


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: check_stray_apps.py <repo-or-owned-temp-root> [...]")
    found = find_stray_apps(*(Path(arg) for arg in sys.argv[1:]))
    for path in found:
        print(f"Stray app bundle (move under *.noindex): {path}")
    sys.exit(bool(found))
