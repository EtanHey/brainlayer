"""Keep all migration crash shards in a mandatory, bounded parallel CI gate."""

import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FILES = ["tests/test_launchd_consolidation_recovery.py"]
# PR-C is a later slice. Its entire sweep becomes mandatory with its implementation.
if (ROOT / "src/brainlayer/launchd_revert.py").exists():
    assert (ROOT / "tests/test_launchd_revert_crash.py").is_file()
    FILES.append("tests/test_launchd_revert_crash.py")


def collect(marker):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *FILES, "--collect-only", "-q", "-m", marker],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return {line for line in result.stdout.splitlines() if line.startswith("tests/") and "::" in line}


def test_marker_partition_preserves_every_exhaustive_shard_and_unit_control():
    expected = {
        f"{FILES[0]}::test_sigkill_every_os_and_launchctl_boundary_then_plain_rerun[shard-{shard:02}-{mode}]"
        for shard in range(32)
        for mode in ("first-install", "keg-restart")
    }
    if len(FILES) == 2:
        expected |= {
            f"{FILES[1]}::test_revert_sigkill_every_os_launchctl_cut_then_rerun[shard-{shard:02}]"
            for shard in range(64)
        }
    sweeps = collect("crash_sweep")
    controls = collect("not crash_sweep")
    assert sweeps == expected  # An independent 32×2 + 64 inventory; no subset may pass.
    assert controls and not controls.intersection(sweeps)
    assert controls | sweeps == collect("crash_sweep or not crash_sweep")
    assert any("test_probe_timeout" in node for node in controls)
    if len(FILES) == 2:
        assert any("test_timeout_identifies" in node for node in controls)


def test_ci_separates_sweeps_without_raising_the_unit_deadline_or_sampling():
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    unit = next(step["run"] for step in jobs["test"]["steps"] if step.get("name") == "Unit tests")
    args = shlex.split(unit)
    assert args[args.index("--seconds") + 1] == "1800"
    assert args[args.index("-m") + 1] == "not integration and not live and not crash_sweep"
    crash = jobs["crash-sweeps"]
    assert crash["name"] == "crash sweeps" and crash["if"] == "always()"
    assert "matrix" not in crash.get("strategy", {})  # One Python, all shards via xdist.
    assert any(step.get("with", {}).get("python-version") == "3.13" for step in crash["steps"])
    run = next(step["run"] for step in crash["steps"] if step.get("name") == "Exhaustive crash sweeps")
    args = shlex.split(run)
    assert args[args.index("pytest") + 1] == "tests/"
    assert args[args.index("-m") + 1] == "crash_sweep"
    assert args[args.index("-n") + 1] == "auto"
    assert args[args.index("--seconds") + 1] == "2400"
    assert not any(arg.split("=", 1)[0] in ("-k", "--ignore", "--deselect", "--lf", "--last-failed") for arg in args)
    assert any("pytest-xdist" in str(step.get("run", "")) for step in crash["steps"])
    assert any(
        step.get("uses", "").startswith("actions/upload-artifact")
        and step.get("if") == "always() && needs.changes.outputs.crash_sweeps == 'true'"
        for step in crash["steps"]
    )
