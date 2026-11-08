#!/usr/bin/env python3
"""Refresh release notes from merged PRs; fetch origin/main before running.

Usage: python scripts/release_changelog_sync.py
Only the selected release section is replaced. GitHub titles are fetched anew.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="1.5.92")
    parser.add_argument("--since", default="v1.5.91")
    parser.add_argument("--ref", default="origin/main")
    args = parser.parse_args()
    subjects = run("git", "log", "--format=%s", f"{args.since}..{args.ref}")
    numbers = list(dict.fromkeys(int(n) for n in re.findall(r"\(#(\d+)\)", subjects)))
    titles = {}
    for number in numbers:
        title = json.loads(run("gh", "pr", "view", str(number), "-R", "EtanHey/brainlayer", "--json", "title"))["title"]
        title = re.sub(r"^\w+(?:\([^)]*\))?!?:\s*", "", title)
        title = re.sub(r"\s*\((?:council|Refs|Fixes)[^)]*\)", "", title)
        titles[number] = title.rstrip(".")
    # Keep the user-facing fixes first; all other merged PRs remain included.
    preferred = [1677]
    preferred += [n for n in numbers if re.search(r"attribution|agent.*(?:count|row)", titles[n], re.I)]
    preferred += [1679, 1680]
    ordered = list(dict.fromkeys(n for n in preferred + numbers if n in titles))
    wording = {
        1677: "BrainBar no longer asks to reconnect Drive for a Production OAuth app",
        1679: "BrainBar reads recent history and tolerates partially written history lines",
        1680: "Indexing retries unresolved backlog files",
    }
    path = ROOT / "CHANGELOG.md"
    text = path.read_text()
    heading = f"## [{args.version}]"
    match = re.search(r"^" + re.escape(heading) + r"[^\n]*\n.*?(?=^## \[|\Z)", text, re.M | re.S)
    date = datetime.date.today().isoformat()
    if match:
        old_date = re.search(r"\d{4}-\d{2}-\d{2}", match.group())
        if old_date:
            date = old_date.group()
    section = (
        f"{heading} - {date}\n\n"
        + "".join(f"- {wording.get(n, titles[n][0].upper() + titles[n][1:])} (#{n}).\n" for n in ordered)
        + "\n"
    )
    if match:
        text = text[: match.start()] + section + text[match.end() :]
    else:
        position = re.search(r"^## \[\d", text, re.M)
        if position is None:
            raise SystemExit("No release heading found; refusing to modify CHANGELOG")
        text = text[: position.start()] + section + text[position.start() :]
    path.write_text(text)
    print(f"Updated {heading}: {len(ordered)} PRs from {args.since}..{args.ref}")


if __name__ == "__main__":
    main()
