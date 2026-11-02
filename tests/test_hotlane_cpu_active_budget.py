"""R2-B1: retain both reviewer scripts as executable regression fixtures."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from brainlayer.hotlane_budget import HotlaneCpuSampler
from scripts import hotlane_cpu_ratchet as ratchet


@pytest.mark.parametrize("case", ["dilution", "minduty_race"])
def test_reviewer_active_cpu_repros(tmp_path, case):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / f"tests/fixtures/hotlane_cpu/repro_{case}.py.txt"), str(tmp_path)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    print(result.stdout)
    assert "('GREEN'," not in result.stdout
    if case == "dilution":
        assert "('RED'," in result.stdout
        assert "150.0%" in result.stdout
    else:
        assert "backlog ran 1s, not enough to measure" in result.stdout


def test_cycle_bounds_exclude_yields_despite_background_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    clock, cpu = [0.0], [0.0]
    sampler = HotlaneCpuSampler(tmp_path / "x.db", clock=lambda: clock[0], cpu_clock=lambda: cpu[0])
    clock[0], cpu[0] = 540, 10.8
    sampler.observe("high load")
    clock[0], cpu[0] = 570, 11.4
    started = sampler.begin_cycle()
    clock[0], cpu[0] = 580, 26.4
    sampler.observe()  # independent sampler tick during the active cycle
    clock[0], cpu[0] = 600, 56.4
    sampler.observe("running", 16, cycle_start=started)
    payload = json.loads(sampler.path.read_text())
    assert payload["backlog_cpu_seconds"] == pytest.approx(45)
    assert payload["backlog_wall_seconds"] == 30
    assert ratchet.read_report(sampler.path)[0] == "RED"
    assert ratchet.daemon_verdict(payload)[0] == "RED"
    monkeypatch.setattr(ratchet, "measure", lambda *_: payload)
    output = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["ratchet", "--pid", "123", "--db", str(tmp_path / "x.db"), "--out", str(output)])
    assert ratchet.main() == 1
    clock[0], cpu[0] = 610, 56.6
    sampler.observe("high load")
    assert ratchet.read_report(sampler.path)[0] == "RED"


@pytest.mark.parametrize("active_wall,expected", [(0, "n/a"), (29.9, "n/a"), (30, "GREEN")])
def test_active_evidence_floor(tmp_path, monkeypatch, active_wall, expected):
    monkeypatch.setattr(sys, "platform", "darwin")
    clock, cpu = [0.0], [0.0]
    sampler = HotlaneCpuSampler(tmp_path / "x.db", clock=lambda: clock[0], cpu_clock=lambda: cpu[0])
    clock[0] = 600 - active_wall
    started = sampler.begin_cycle()
    clock[0], cpu[0] = 600, active_wall / 2
    sampler.observe("running", 16, cycle_start=started)
    assert ratchet.read_report(sampler.path)[0] == expected
