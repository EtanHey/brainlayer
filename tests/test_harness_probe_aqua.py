"""GUI probe admission uses one owned job and fails closed outside Aqua."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("probe_aqua", SCRIPTS / "harness_probe_aqua.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_session_admission_is_narrow_and_preserves_mcp_config(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from harness_probe_contract import command

    claude = command("claude", "probe")
    assert claude[claude.index("--allowedTools") + 1] == ("mcp__brainlayer__brain_search,mcp__brainlayer__brain_store")
    cursor = command("cursor", "probe")
    assert "--trust" in cursor
    for cmd in (claude, cursor):
        assert not set(cmd) & {"--mcp-config", "--settings", "--yolo", "--force", "--dangerously-skip-permissions"}


@pytest.mark.parametrize("manager", ["Background", "System", ""])
def test_required_gui_context_refuses_before_any_harness(monkeypatch, manager):
    module = load(monkeypatch)
    monkeypatch.setenv("BRAINLAYER_HARNESS_AQUA_REQUIRED", "1")
    monkeypatch.setattr(module, "run", lambda *_: manager)
    with pytest.raises(RuntimeError, match="Aqua"):
        module.require_aqua()


def test_owned_job_cleans_up_exact_label_after_terminal_receipt(monkeypatch, tmp_path):
    module = load(monkeypatch)
    output = tmp_path / "receipt"
    calls = []

    def run(argv):
        calls.append(argv)
        if argv[1] == "bootstrap":
            import plistlib

            spec = plistlib.loads(Path(argv[-1]).read_bytes())
            assert spec["LimitLoadToSessionType"] == "Aqua"
            assert spec["KeepAlive"] is False
            assert spec["EnvironmentVariables"]["BRAINLAYER_HARNESS_AQUA_REQUIRED"] == "1"
            assert "SSH_CONNECTION" not in spec["EnvironmentVariables"]
            assert spec["WorkingDirectory"] == str(Path.cwd())
            output.mkdir()
            (output / "aqua-exit.json").write_text(json.dumps({"exit_code": 1}))
        return ""

    monkeypatch.setattr(module, "run", run)
    assert module.launch(SCRIPTS / "harness_probe.py", ["--output", str(output)], output, 5) == 1
    assert calls[-1][1] == "bootout"
    assert calls[-1][2].startswith("gui/")
    assert "/com.brainlayer.harness-probe-" in calls[-1][2]
    assert (output / "aqua-launch.json").is_file()


def test_failed_bootstrap_never_boots_out_an_existing_job(monkeypatch, tmp_path):
    module = load(monkeypatch)
    calls = []

    def run(argv):
        calls.append(argv)
        raise RuntimeError("fixture bootstrap refused")

    monkeypatch.setattr(module, "run", run)
    with pytest.raises(RuntimeError, match="bootstrap refused"):
        module.launch(SCRIPTS / "harness_probe.py", [], tmp_path / "out", 1)
    assert [a[1] for a in calls] == ["bootstrap"]
