"""Bounded SIGKILL replay sweep with persisted independent launchctl state."""

import json
import plistlib
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.test_consolidation_plan import NEW, OLD, ROOT
from tests.test_launchd_consolidation_recovery import PROBE as MIGRATE_PROBE
from tests.test_launchd_consolidation_recovery import scratch
from tests.test_launchd_revert import CALENDAR, FLEET

IMPORT_ANCHOR = "from brainlayer.launchd_consolidation import migrate"
OVERLAP_ANCHOR = 'if "com.brainlayer.watch" in state["loaded"] and "com.brainlayer.daemon" in state["loaded"]:'


def revert_probe(migrate_probe):
    for anchor in (IMPORT_ANCHOR, OVERLAP_ANCHOR):
        if migrate_probe.count(anchor) != 1:
            raise ValueError(f"probe anchor missing or duplicated: {anchor}")
    code = migrate_probe.replace(IMPORT_ANCHOR, "from brainlayer.launchd_revert import revert")
    code = code.replace(
        OVERLAP_ANCHOR,
        'if {"com.brainlayer.watch", "com.etanhey.brainlayer-fleet-watchdog"}.intersection(state["loaded"]) and {"com.brainlayer.daemon", "com.brainlayer.maintenance"}.intersection(state["loaded"]):',
    )
    start, end = code.index("result = migrate("), code.index("print(json.dumps")
    return (
        code[:start]
        + """result = revert(home, home / ".local/share/brainlayer/supervisor/migration.json",
    "gui/424242", run, alive=lambda pid: False, clock=Clock())
"""
        + code[end:]
    )


REVERT_PROBE = revert_probe(MIGRATE_PROBE)


def probe(root, code, cut=0, *, context="baseline"):
    try:
        return subprocess.run(
            [sys.executable, "-c", code, str(root), "keg1", str(cut), str(ROOT)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        receipt = root / "home/.local/share/brainlayer/supervisor/migration.json"
        try:
            journal = receipt.read_text()[-2000:]
        except (OSError, UnicodeError) as missing:
            journal = f"unreadable/missing: {type(missing).__name__}"
        pytest.fail(
            f"revert {context} cut={cut} exceeded 30s; journal={journal}; stdout={error.stdout!r}; stderr={error.stderr!r}"
        )


@pytest.mark.crash_sweep
@pytest.mark.parametrize("shard", range(64), ids=lambda shard: f"shard-{shard:02}")
def test_revert_sigkill_every_os_launchctl_cut_then_rerun(tmp_path, shard):
    baseline_root = tmp_path / "baseline"
    scratch(baseline_root)
    agents = baseline_root / "home/Library/LaunchAgents"
    for path in agents.glob("*.plist"):
        doc = plistlib.loads(path.read_bytes())
        doc["ProgramArguments"] = ["/usr/bin/true"]
        path.write_bytes(plistlib.dumps(doc))
    originals = {p.name: p.read_bytes() for p in agents.iterdir()}
    migrated = probe(baseline_root, MIGRATE_PROBE)
    assert migrated.returncode == 0, migrated.stderr
    # Every clone retains its original absolute archive path, so translate only
    # that fixture field, independently from the implementation being measured.
    preimage = tmp_path / "preimage"
    shutil.copytree(baseline_root, preimage)
    baseline = probe(baseline_root, REVERT_PROBE)
    assert baseline.returncode == 0, baseline.stderr
    events = json.loads(baseline.stdout)["events"]
    cuts = [(cut, event) for cut, event in enumerate(events, 1) if (cut - 1) % 64 == shard]
    assert 0 < len(cuts) <= 30, len(events)
    timings = []
    for cut, event in cuts:
        root = tmp_path / str(cut)
        shutil.copytree(preimage, root)
        receipt = root / "home/.local/share/brainlayer/supervisor/migration.json"
        document = json.loads(receipt.read_bytes())
        document["archive"] = str(root / "home/Library/LaunchAgents" / Path(document["archive"]).name)
        document["group_custody"] = {
            label: str(root / "home/.local/share/brainlayer/supervisor" / ("calendar" if label == CALENDAR else ""))
            for label in (NEW, CALENDAR)
        }
        receipt.write_text(json.dumps(document))
        started = time.monotonic()
        killed = probe(root, REVERT_PROBE, cut, context=f"kill cut={cut} {event}")
        assert killed.returncode == -signal.SIGKILL, (cut, event, killed.stderr)
        rerun = probe(root, REVERT_PROBE, context=f"rerun cut={cut} {event}")
        assert rerun.returncode == 0, (cut, event, rerun.stderr)
        state = json.loads((root / "launchd.json").read_bytes())
        assert not state["violations"] and set(state["loaded"]) == {OLD, FLEET}, (cut, event, state)
        result = json.loads(receipt.read_bytes())
        assert result["phase"] == "reverted"
        agents = root / "home/Library/LaunchAgents"
        assert {p.name: p.read_bytes() for p in agents.iterdir() if p.is_file()} == originals
        assert not (receipt.parent / "jobs.json").exists()
        assert not list(agents.glob(".migration-*")) and not list(receipt.parent.glob(".migration-*"))
        assert result["states"][OLD]["disabled_override"] == "enabled"
        assert result["states"]["com.brainlayer.observability"]["disabled_override"] == "disabled"
        assert result["states"][FLEET]["disabled_override"] is None
        timings.append({"cut": cut, "boundary": event, "seconds": time.monotonic() - started})
    print(
        "REVERT_SWEEP "
        + json.dumps({"shard": shard, "cuts": len(cuts), "all_cuts": len(events), "bad": 0, "timings": timings}),
        flush=True,
    )


@pytest.mark.parametrize("receipt_present", [True, False])
def test_timeout_identifies_the_original_rerun_cut_and_streams(tmp_path, monkeypatch, receipt_present):
    receipt = tmp_path / "home/.local/share/brainlayer/supervisor/migration.json"
    receipt.parent.mkdir(parents=True)
    if receipt_present:
        receipt.write_text('{"phase":"reverting"}')

    def timeout(argv, **kwargs):
        assert kwargs["timeout"] == 30
        raise subprocess.TimeoutExpired(argv, 30, output=b"partial stdout", stderr=b"partial stderr")

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(pytest.fail.Exception) as error:
        probe(tmp_path, REVERT_PROBE, context="rerun cut=612 after:os.link")
    assert all(
        detail in str(error.value)
        for detail in (
            "rerun cut=612 after:os.link",
            "30s",
            "reverting" if receipt_present else "missing",
            "partial stdout",
            "partial stderr",
        )
    )
