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


@pytest.mark.parametrize("harnesses", [("cursor",), ("t3",)])
@pytest.mark.parametrize(
    "output_args",
    [
        ("--output", "receipt"),
        ("--output=receipt",),
        ("--out", "receipt"),
        ("--output=stale", "--output", "receipt"),
    ],
)
def test_owned_job_cleans_up_exact_label_after_terminal_receipt(monkeypatch, tmp_path, harnesses, output_args):
    module = load(monkeypatch)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "receipt"
    calls, private = [], []
    caller = {"T3_ACP_MCP_ENDPOINT": "http://127.0.0.1:1/mcp", "T3_ACP_MCP_AUTHORIZATION": "synthetic-secret"}
    for key, value in {**caller, "UNRELATED_SECRET": "must-not-forward"}.items():
        monkeypatch.setenv(key, value)

    def run(argv):
        calls.append(argv)
        if argv[1] == "bootstrap":
            import plistlib

            plist = Path(argv[-1])
            private.append(plist)
            assert plist.stat().st_mode & 0o777 == 0o600 and plist.parent.stat().st_mode & 0o777 == 0o700
            spec = plistlib.loads(plist.read_bytes())
            assert spec["ProgramArguments"][2:] == ["--harnesses", *harnesses, "--output", str(output)]
            env = spec["EnvironmentVariables"]
            assert "UNRELATED_SECRET" not in env
            assert {k: env[k] for k in caller if k in env} == (caller if "t3" in harnesses else {})
            assert spec["LimitLoadToSessionType"] == "Aqua"
            assert spec["KeepAlive"] is False
            assert spec["EnvironmentVariables"]["BRAINLAYER_HARNESS_AQUA_REQUIRED"] == "1"
            assert "SSH_CONNECTION" not in spec["EnvironmentVariables"]
            assert spec["WorkingDirectory"] == str(Path.cwd())
            output.mkdir()
            (output / "aqua-exit.json").write_text(json.dumps({"exit_code": 1}))
        return ""

    monkeypatch.setattr(module, "run", run)
    argv = ["--harnesses", *harnesses, *output_args]
    assert module.launch(SCRIPTS / "harness_probe.py", argv, output, 5, harnesses=harnesses) == 1
    assert calls[-1][1] == "bootout"
    assert calls[-1][2].startswith("gui/")
    assert "/com.brainlayer.harness-probe-" in calls[-1][2]
    assert (output / "aqua-launch.json").is_file()
    assert not private[0].exists() and all("synthetic-secret" not in p.read_text() for p in output.iterdir())


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


@pytest.mark.parametrize("private_child", [False, True])
def test_aqua_refuses_shared_receipt_parent(monkeypatch, tmp_path, private_child):
    module = load(monkeypatch)
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o777)
    if private_child:
        parent = parent / "private"
        parent.mkdir(mode=0o700)
    monkeypatch.setattr(module, "run", lambda *_: pytest.fail("unsafe receipt parent reached launchctl"))
    with pytest.raises(RuntimeError, match="Unsafe Aqua receipt parent"):
        module.launch(SCRIPTS / "harness_probe.py", [], parent / "receipt", 5)


@pytest.mark.parametrize("kind", ["parent", "ancestor", "shared"])
def test_aqua_refuses_symlink_components(monkeypatch, tmp_path, kind):
    module = load(monkeypatch)
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    base = tmp_path
    suffix = Path("receipt")
    if kind == "ancestor":
        (target / "private").mkdir(mode=0o700)
        suffix = Path("private/receipt")
    if kind == "shared":
        base = tmp_path / "shared"
        base.mkdir()
        base.chmod(0o777)
    alias = base / "alias"
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(module, "run", lambda *_: pytest.fail("symlink path reached launchctl"))
    with pytest.raises(RuntimeError, match="Unsafe Aqua receipt parent"):
        module.launch(SCRIPTS / "harness_probe.py", [], alias / suffix, 5)


def test_plist_exclusive_creation_preserves_existing_file(monkeypatch, tmp_path):
    from contextlib import nullcontext

    module = load(monkeypatch)
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    plist = work / "probe.plist"
    plist.write_bytes(b"existing private file")
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda **kw: nullcontext(work))
    monkeypatch.setattr(module, "run", lambda *_: pytest.fail("existing plist reached launchctl"))
    with pytest.raises(FileExistsError):
        module.launch(SCRIPTS / "harness_probe.py", [], tmp_path / "receipt", 5)
    assert plist.read_bytes() == b"existing private file"
