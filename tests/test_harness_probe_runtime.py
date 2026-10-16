"""The live gate's observer may attribute only the spawned harness's descendants."""

import importlib.util
import os
import pwd
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(autouse=True)
def account_home(monkeypatch, tmp_path):
    home = tmp_path / "account-home"
    home.mkdir(mode=0o700)
    monkeypatch.setattr(pwd, "getpwuid", lambda _: SimpleNamespace(pw_dir=str(home)))
    return home


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("harness_probe_runtime", SCRIPTS / "harness_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("temp_parent", ["caller", "hostile"])
def test_cursor_scratch_ignores_caller_temp_parent(monkeypatch, tmp_path, account_home, temp_parent):
    module = load(monkeypatch)
    caller = tmp_path / "caller"
    caller.mkdir()
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    hostile.chmod(0o777)
    monkeypatch.chdir(caller)
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(caller if temp_parent == "caller" else hostile))
    monkeypatch.setenv("HOME", str(caller))
    monkeypatch.setattr(module.tempfile, "tempdir", None)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/cursor-agent")
    captured = []

    def spawn(*args, **kwargs):
        cwd = Path(kwargs["cwd"])
        captured.append(cwd)
        assert cwd.parent == account_home / ".brainlayer-harness-probe"
        for path in (cwd, cwd.parent):
            info = path.lstat()
            assert info.st_uid == os.getuid() and info.st_mode & 0o777 == 0o700
            assert not path.is_symlink()
        assert not list(cwd.iterdir())
        raise RuntimeError("test stopped before launch")

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError, match="test stopped before launch"):
        module.probe("cursor", SimpleNamespace(output=tmp_path, timeout=60, ingest_timeout=0))
    assert captured and not captured[0].exists()


@pytest.mark.parametrize("unsafe", ["writable", "symlink"])
def test_cursor_refuses_hostile_private_parent(monkeypatch, tmp_path, account_home, unsafe):
    module = load(monkeypatch)
    parent = account_home / ".brainlayer-harness-probe"
    if unsafe == "symlink":
        parent.symlink_to(tmp_path, target_is_directory=True)
    else:
        parent.mkdir(mode=0o700)
        parent.chmod(0o777)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/cursor-agent")
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: pytest.fail("unsafe parent launched Cursor"))
    with pytest.raises(RuntimeError, match="Unsafe probe scratch"):
        module.probe("cursor", SimpleNamespace(output=tmp_path, timeout=60, ingest_timeout=0))


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


def test_codex_app_skip_event_does_not_run_bridge_verification(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    module = load(monkeypatch)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/python")

    class Proc:
        pid = 123
        returncode = 0

        def poll(self):
            return 0

    def spawn(*args, **kwargs):
        kwargs["stdout"].write("[]\n")
        kwargs["stdout"].write(
            json.dumps(
                {
                    "type": "harness.skip",
                    "reason": "Codex App is not running; existing app context is unavailable",
                }
            )
            + "\n"
        )
        return Proc()

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    monkeypatch.setattr(module, "Bridge", lambda: pytest.fail("skip must not verify a bridge"))
    row = module.probe("codex-app", SimpleNamespace(output=tmp_path, timeout=5, ingest_timeout=0))
    assert row["status"] == "SKIP"
    assert row["reason"] == "Codex App is not running; existing app context is unavailable"
    assert row["exit_code"] == 0
    assert row["legs"] == {"search": "SKIPPED", "store": "SKIPPED", "refound": "SKIPPED", "ingest": "SKIPPED"}
    assert module.exit_code([row]) == 1


def test_codex_app_running_without_socket_remains_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace

    module = load(monkeypatch)
    monkeypatch.setattr(module.shutil, "which", lambda _: "/python")

    class Proc:
        pid = 123
        returncode = 1

        def poll(self):
            return 1

    def spawn(*args, **kwargs):
        kwargs["stdout"].write("")
        kwargs["stderr"].write("Existing app daemon control socket missing; no server started\n")
        return Proc()

    monkeypatch.setattr(module.subprocess, "Popen", spawn)

    class Bridge:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def search(self, *_args, **_kwargs):
            return {}

    monkeypatch.setattr(module, "Bridge", Bridge)
    row = module.probe("codex-app", SimpleNamespace(output=tmp_path, timeout=5, ingest_timeout=0))
    assert row["status"] == "FAIL"
    assert row["exit_code"] == 1
    assert "exit 1" in row["reason"]


def test_all_required_harnesses_may_run_in_any_order(monkeypatch, tmp_path):
    module = load(monkeypatch)
    names = list(reversed(module.HARNESSES))
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(tmp_path / "receipt"), "--harnesses", *names])
    monkeypatch.setattr(module, "probe", lambda name, *a: {"harness": name, "status": "PASS"})
    assert module.main() == 0


@pytest.mark.parametrize("harness", ["codex-app", "cursor"])
def test_native_helper_gets_cleanup_margin_and_receipt_hash(monkeypatch, tmp_path, harness):
    import json
    import os
    from types import SimpleNamespace

    module = load(monkeypatch)
    monkeypatch.setenv("T3_ACP_MCP_AUTHORIZATION", "synthetic-secret")
    monkeypatch.setattr(module.shutil, "which", lambda _: "/python")
    captured = {}

    class Proc:
        pid = 1
        returncode = 0

        def poll(self):
            return 0

    def spawn(*a, **kw):
        if harness == "cursor":
            cwd = Path(kw["cwd"])
            assert "--trust" in a[0] and cwd != Path.cwd() and not list(cwd.iterdir())
            assert cwd.stat().st_uid == os.getuid() and cwd.stat().st_mode & 0o777 == 0o700
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
    module.probe(harness, SimpleNamespace(output=tmp_path, timeout=60, ingest_timeout=0))
    if harness == "cursor":
        assert not Path(captured["cwd"]).exists()
        assert "T3_ACP_MCP_AUTHORIZATION" not in captured["env"]
    else:
        assert captured["env"]["BRAINLAYER_NATIVE_PROBE_TIMEOUT"] == "45"
    monkeypatch.setattr(module, "run", lambda *_: "fixture-head")
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["probe", "--output", str(out), "--harnesses", "codex-app"])
    monkeypatch.setattr(module, "probe", lambda *a: {"harness": "codex-app", "status": "FAIL"})
    module.main()
    assert "harness_probe_t3.py" in json.loads((out / "run.json").read_text())["script_sha256"]


def test_descendant_bridge_on_foreign_unix_socket_is_not_brainbar(monkeypatch):
    module = load(monkeypatch)

    def run(argv):
        if argv[:2] == ["ps", "-axo"]:
            return "100 1 codex\n101 100 Python\n"
        if argv[0] == "ps":
            return "/keg/bin/brainlayer-mcp-stdio-bridge" if argv[2] == "101" else "codex"
        return "Python 101 user 3u unix 0xabc 0t0 ->0xdef\nOther 7 user 5u unix 0xdef 0t0 /tmp/other.sock\n"

    monkeypatch.setattr(module, "run", run)
    assert module.observe(100) == []
