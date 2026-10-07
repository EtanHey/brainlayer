from pathlib import Path

import yaml


def test_retirement_selects_latest_xcode_unconditionally_before_build():
    root = Path(__file__).resolve().parents[1]
    steps = yaml.safe_load((root / ".github/workflows/ratchet.yml").read_text())["jobs"]["retirement"]["steps"]
    selectors = [(i, s) for i, s in enumerate(steps) if s.get("uses") == "maxim-lobanov/setup-xcode@v1"]
    assert len(selectors) == 1, "retirement must select Xcode exactly once"
    index, selector = selectors[0]
    assert "if" not in selector, "retirement Xcode selection must be unconditional"
    assert selector["with"]["xcode-version"] == "latest-stable"
    builds = [i for i, s in enumerate(steps) if "scripts/retirement_run.py" in s.get("run", "")]
    assert builds and all(index < i for i in builds), "select Xcode before every retirement native build"
