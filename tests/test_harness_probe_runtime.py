"""The live gate's observer may attribute only the spawned harness's descendants."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("harness_probe_runtime", SCRIPTS / "harness_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unix_peer_address_resolves_only_descendant_bridge(monkeypatch):
    module = load(monkeypatch)

    def run(argv):
        if argv[:2] == ["ps", "-axo"]:
            return "100 1 codex\n101 100 Python\n999 1 Python\n"
        if argv[0] == "ps":
            return "/keg/bin/brainlayer-mcp-stdio-bridge" if argv[2] in ["101", "999"] else "codex"
        return (
            "Python 101 user 3u unix 0xabc 0t0 ->0xdef\n"
            "Python 999 user 3u unix 0xaaa 0t0 ->0xbbb\n"
            "BrainBar 7 user 5u unix 0xdef 0t0 /tmp/brainbar.sock\n"
            "Other 8 user 5u unix 0xbbb 0t0 /tmp/other.sock\n"
        )

    monkeypatch.setattr(module, "run", run)
    assert module.observe(100) == [
        {"pid": 101, "command": ["/keg/bin/brainlayer-mcp-stdio-bridge"], "socket": "/tmp/brainbar.sock"}
    ]


def test_partial_selection_never_exits_green(monkeypatch, tmp_path):
    module = load(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(tmp_path / "receipt"), "--harnesses", "claude"])
    monkeypatch.setattr(module, "probe", lambda *a: {"harness": "claude", "status": "PASS"})
    assert module.main() == 1


def test_unavailable_app_surface_has_no_cli_substitution(monkeypatch, tmp_path):
    module = load(monkeypatch)
    from types import SimpleNamespace

    row = module.probe("codex-app", SimpleNamespace(output=tmp_path))
    assert row["status"] == "FAIL"
    assert "app-thread adapter" in row["reason"]


@pytest.mark.parametrize("error", [ValueError, KeyError, TypeError, AttributeError])
def test_bad_harness_data_keeps_failure_row_and_continues(monkeypatch, tmp_path, error):
    import json

    module = load(monkeypatch)
    output = tmp_path / "receipt"
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(output), "--harnesses", "claude", "codex-cli"])
    monkeypatch.setattr(module, "run", lambda _: "fixture-head")
    called = []

    def probe(harness, args):
        called.append(harness)
        if harness == "claude":
            raise error("private malformed payload")
        return {"harness": harness, "status": "PASS"}

    monkeypatch.setattr(module, "probe", probe)
    assert module.main() == 1
    assert called == ["claude", "codex-cli"]
    rows = json.loads((output / "results.json").read_text())
    assert rows[0] == {"harness": "claude", "status": "FAIL", "reason": error.__name__}
    assert rows[1]["harness"] == "codex-cli"
    assert "private malformed payload" not in (output / "table.md").read_text()
