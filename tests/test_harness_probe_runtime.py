"""The live gate's observer may attribute only the spawned harness's descendants."""

import importlib.util
import sys
from pathlib import Path

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

    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    row = module.probe("codex-app", SimpleNamespace(output=tmp_path))
    assert row["status"] == "FAIL"
    assert "executable missing" in row["reason"]


def test_all_required_harnesses_may_run_in_any_order(monkeypatch, tmp_path):
    module = load(monkeypatch)
    names = list(reversed(module.HARNESSES))
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(tmp_path / "receipt"), "--harnesses", *names])
    monkeypatch.setattr(module, "probe", lambda name, *a: {"harness": name, "status": "PASS"})
    assert module.main() == 0


def test_native_helper_gets_cleanup_margin_and_receipt_hash(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    module = load(monkeypatch)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/python")
    captured = {}

    class Proc:
        pid = 1
        returncode = 0

        def poll(self):
            return 0

    def spawn(*a, **kw):
        captured.update(kw)
        return Proc()

    monkeypatch.setattr(module.subprocess, "Popen", spawn)

    class Bridge:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def search(self, *a, **kw):
            return {}

    monkeypatch.setattr(module, "Bridge", Bridge)
    module.probe("codex-app", SimpleNamespace(output=tmp_path, timeout=60, ingest_timeout=0))
    assert captured["env"]["BRAINLAYER_NATIVE_PROBE_TIMEOUT"] == "45"
    monkeypatch.setattr(module, "run", lambda *_: "fixture-head")
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(out), "--harnesses", "codex-app"])
    monkeypatch.setattr(module, "probe", lambda *a: {"harness": "codex-app", "status": "FAIL"})
    module.main()
    assert "harness_probe_t3.py" in json.loads((out / "run.json").read_text())["script_sha256"]
