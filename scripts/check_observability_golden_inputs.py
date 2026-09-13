#!/usr/bin/env python3
"""Check observability golden input receipts against their fixture artifacts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import tarfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures/observability"
SECTIONS = ("stores", "emitters", "author_unknown", "backups")


def _metadata(path: Path) -> tuple[int | None, str | None]:
    if not path.exists():
        return None, None
    if path.suffix in {".sqlite", ".db"}:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
        try:
            rows = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            connection.close()
        return rows, None
    if path.is_file():
        with path.open("rb") as handle:
            digest = hashlib.sha256(handle.read(65_536)).hexdigest()
        return path.stat().st_size, digest
    return len(list(path.iterdir())), None


def _golden_path(
    case: dict[str, Any],
    *,
    fixture_root: Path,
    heldout_root: Path | None,
) -> Path:
    root = heldout_root if case["split"] == "heldout" else fixture_root
    if root is None:
        raise ValueError("--heldout-golden-root is required for heldout cases")
    return root / case["golden"]


def seal_heldout(root: Path, cases: list[dict[str, Any]], output: Path) -> str:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(case["golden"] for case in cases if case["split"] == "heldout"):
            payload = (root / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.uid, info.gid = len(payload), 20260913, 0, 0
            info.uname, info.gname, info.mode = "", "", 0o644
            archive.addfile(info, io.BytesIO(payload))
    payload = stream.getvalue()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def check(
    *,
    fixture_root: Path,
    heldout_root: Path | None,
    split: str,
    write: bool,
) -> list[str]:
    manifest = json.loads((fixture_root / "cases.json").read_text(encoding="utf-8"))
    findings: list[str] = []
    for case in manifest["cases"]:
        if split != "all" and case["split"] != split:
            continue
        golden_path = _golden_path(case, fixture_root=fixture_root, heldout_root=heldout_root)
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        changed = False
        for section in SECTIONS:
            for index, item in enumerate(golden[section]["inputs"]):
                rows, digest = _metadata(fixture_root / item["path"])
                stale = []
                if item["rows_or_bytes"] != rows:
                    stale.append(f"rows_or_bytes {item['rows_or_bytes']!r} != {rows!r}")
                    item["rows_or_bytes"] = rows
                if item["sha256_first_64kb"] != digest:
                    stale.append(f"sha256_first_64kb {item['sha256_first_64kb']!r} != {digest!r}")
                    item["sha256_first_64kb"] = digest
                if stale:
                    findings.append(
                        f"{case['case_id']}: $.{section}.inputs[{index}] ({item['path']}): " + "; ".join(stale)
                    )
                    changed = True
        if write and changed:
            golden_path.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "heldout", "all"), default="dev")
    parser.add_argument("--fixture-root", type=Path, default=FIXTURES)
    parser.add_argument("--heldout-golden-root", type=Path)
    parser.add_argument("--heldout-tar", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        findings = check(
            fixture_root=args.fixture_root.resolve(),
            heldout_root=args.heldout_golden_root.resolve() if args.heldout_golden_root else None,
            split=args.split,
            write=args.write,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, sqlite3.Error) as exc:
        parser.error(str(exc))
    for finding in findings:
        print(finding)
    if args.heldout_tar:
        if not args.write or args.heldout_golden_root is None:
            parser.error("--heldout-tar requires --write and --heldout-golden-root")
        manifest = json.loads((args.fixture_root / "cases.json").read_text(encoding="utf-8"))
        digest = seal_heldout(args.heldout_golden_root, manifest["cases"], args.heldout_tar)
        print(f"heldout_goldens_sha256={digest}")
    if findings:
        print(f"{'updated' if args.write else 'stale'} input receipts: {len(findings)}")
    else:
        print("observability golden input receipts: clean")
    return 0 if args.write or not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
