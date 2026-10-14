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

REVERT_PROBE = MIGRATE_PROBE.replace(
    "from brainlayer.launchd_consolidation import migrate", "from brainlayer.launchd_revert import revert"
)
REVERT_PROBE = REVERT_PROBE.replace(
    'if "com.brainlayer.watch" in state["loaded"] and "com.brainlayer.daemon" in state["loaded"]:',
    'if {"com.brainlayer.watch", "com.etanhey.brainlayer-fleet-watchdog"}.intersection(state["loaded"]) and {"com.brainlayer.daemon", "com.brainlayer.maintenance"}.intersection(state["loaded"]):',
)
start = REVERT_PROBE.index("result = migrate(")
end = REVERT_PROBE.index("print(json.dumps", start)
REVERT_PROBE = (
    REVERT_PROBE[:start]
    + """result = revert(home, home / ".local/share/brainlayer/supervisor/migration.json",
    "gui/424242", run, alive=lambda pid: False, clock=Clock())
"""
    + REVERT_PROBE[end:]
)


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
        pytest.fail(
            f"revert {context} cut={cut} exceeded 30s; journal={receipt.read_text()[-2000:]}; stdout={error.stdout!r}; stderr={error.stderr!r}"
        )


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


def test_timeout_identifies_the_original_rerun_cut_and_streams(tmp_path, monkeypatch):
    receipt = tmp_path / "home/.local/share/brainlayer/supervisor/migration.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"phase":"reverting"}')

    def timeout(argv, **kwargs):
        assert kwargs["timeout"] == 30
        raise subprocess.TimeoutExpired(argv, 30, output=b"partial stdout", stderr=b"partial stderr")

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(pytest.fail.Exception) as error:
        probe(tmp_path, REVERT_PROBE, context="rerun cut=612 after:os.link")
    assert all(
        detail in str(error.value)
        for detail in ("rerun cut=612 after:os.link", "30s", "reverting", "partial stdout", "partial stderr")
    )
