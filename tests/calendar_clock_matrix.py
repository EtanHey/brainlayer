"""Run the focused calendar/backup files at five wall-clock boundaries.

Usage: python tests/calendar_clock_matrix.py [--root PRE_FIX_TREE] [--output DIR]
The separate root allows fail-first evidence without editing the working checkout.
"""

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

CLOCKS = (
    "2026-11-10T22:00+00:00",
    "2026-11-11T01:30+00:00",
    "2026-11-11T03:40+00:00",
    "2026-11-11T12:00+00:00",
    "2026-11-30T23:30+00:00",
)


def main():
    source = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=source)
    parser.add_argument("--output", type=Path, default=source / ".build.noindex/clock-matrix")
    parser.add_argument("--clock", action="append", choices=CLOCKS, help="Repeat to select clock rows")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = [
        str(path.relative_to(root))
        for path in sorted((root / "tests").glob("test_*.py"))
        if any(
            name in path.read_text()
            for name in (
                "maintenance_calendar",
                "backup_daily",
                "maintenance_upload_recovery",
                "local_backup_date",
                "maintenance_window",
                "index_admission",
            )
        )
    ]
    (output / "files.txt").write_text("\n".join(files) + "\n")
    failed = False
    summary = []
    for clock in args.clock or CLOCKS:
        name = clock[:16].replace(":", "-")
        report = output / f"{name}.xml"
        env = dict(os.environ, BRAINLAYER_TEST_CLOCK=clock, TZ="UTC")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(source / "tests/clock_support"), str(root / "src"), str(root), env.get("PYTHONPATH", "")]
        )
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", *files, "-q", "--tb=short", f"--junitxml={report}"],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        failures = []
        if report.exists():
            for test in ET.parse(report).iter("testcase"):
                if test.find("failure") is not None or test.find("error") is not None:
                    failures.append(f"{test.get('classname')}::{test.get('name')}")
        summary.append(f"{clock}: rc={result.returncode}\n" + "\n".join(failures))
        (output / "summary.txt").write_text("\n\n".join(summary) + "\n")
        print(summary[-1], flush=True)
        failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
