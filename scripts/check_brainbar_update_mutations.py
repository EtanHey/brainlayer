"""Replay the seven reviewed updater faults against isolated copies of the installer."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scratch = args.scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    source = (root / "scripts/brainbar-update.sh").read_text()
    signature = 'bash "$(dirname "$INSTALL")/release-verify-signatures.sh" "$K"\n'
    restore_start = source.index('    if [ -n "$BACKUP" ]')
    restore_end = source.index('    if [ "$MUTATED" = 0 ]; then', restore_start)
    guard_start = source.index('if [ -f "$ROOT/update-state" ]; then')
    guard_end = source.index("state preflight\n", guard_start)
    mutations = [
        (
            "second-preflight",
            source.replace("pin\npreflight\nSTEP=formula", "pin\nSTEP=formula"),
            "test_second_preflight_catches_work_started_during_update",
        ),
        (
            "late-signature",
            source.replace(signature, "").replace("STEP=activate\n", signature + "STEP=activate\n"),
            "test_failure_before_cask_preserves_old_app[signatures]",
        ),
        (
            "ignored-consolidation",
            source.replace('bash "$INSTALL" consolidate\n', 'bash "$INSTALL" consolidate || true\n'),
            "test_failure_before_cask_preserves_old_app[consolidate]",
        ),
        (
            "no-restore",
            source[:restore_start] + source[restore_end:],
            "test_swap_failure_restores_and_verifies_old_app[activate]",
        ),
        ("no-unsettled-guard", source[:guard_start] + source[guard_end:], "test_unsettled_attempt_refuses_retry"),
        (
            "noop-cask",
            source.replace("brew reinstall --cask", "brew upgrade --cask"),
            "test_retry_after_restoration_reinstalls_recorded_cask",
        ),
        ("nonexecutable", source, "test_update_script_is_executable"),
    ]
    rows = []
    for name, mutated, test in mutations:
        if name != "nonexecutable" and mutated == source:
            raise RuntimeError("Mutation did not change source: " + name)
        directory = scratch / name
        directory.mkdir(exist_ok=True)
        script = directory / "brainbar-update.sh"
        script.write_text(mutated)
        script.chmod(0o644 if name == "nonexecutable" else 0o755)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_brainbar_update.py::" + test,
                "--basetemp=" + str(directory / "pytest"),
                "-q",
            ],
            cwd=root,
            env=dict(os.environ, BRAINBAR_UPDATE_SCRIPT=str(script)),
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        (directory / "red.log").write_text(output)
        killed = result.returncode == 1 and ("FAILED tests/test_brainbar_update.py::" + test) in output
        rows.append({"mutation": name, "test": test, "exit": result.returncode, "killed": killed})
        print(json.dumps(rows[-1]), flush=True)
    (scratch / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0 if all(row["killed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
