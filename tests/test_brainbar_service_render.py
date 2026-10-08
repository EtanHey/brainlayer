"""Synthetic install/render controls; no real signature, launchctl, app or socket action."""

import importlib.util
import json
import os
import plistlib
import shutil
import socket
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("service_render", ROOT / "brain-bar/install-services.py")
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


@pytest.fixture
def fixture():
    scratch = tempfile.TemporaryDirectory(prefix="bbs-", dir="/tmp")
    home = Path(scratch.name) / "home &| <case>"
    home.mkdir(mode=0o700)
    app = home / "Applications/BrainBar.app"
    resources = app / "Contents/Resources/LaunchAgents"
    resources.mkdir(parents=True)
    shutil.copy(ROOT / "brain-bar/bundle/Info.plist", app / "Contents/Info.plist")
    for label, binary in service.LABELS.items():
        shutil.copy(ROOT / "brain-bar/bundle" / (label + ".plist"), resources)
        executable = app / "Contents/MacOS" / binary
        executable.parent.mkdir(exist_ok=True)
        executable.write_text("not a real executable")
        executable.chmod(0o700)
    venv = home / "venv/bin"
    venv.mkdir(parents=True)
    for binary in ("python", "brainlayer"):
        (venv / binary).write_text("not executed")
        (venv / binary).chmod(0o700)
    args = Namespace(
        app=str(app), python=str(venv / "python"), cli=str(venv / "brainlayer"), socket=str(home / "s.sock"), db=None
    )
    yield home, app, args
    scratch.cleanup()


def prepare(fixture, monkeypatch):
    home, app, args = fixture
    calls = []
    plists = service.prepare(args, home, os.getuid(), calls.append)
    assert calls == [["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)]]
    return plists


def test_render_preserves_keys_metacharacters_and_ignores_inherited_secrets(fixture, monkeypatch):
    home, app, args = fixture
    monkeypatch.setenv("BRAINBAR_SOCKET_PATH", "/production/foreign.sock")
    monkeypatch.setenv("BRAINLAYER_DB", "/production/private.db")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-secret")
    plists = prepare(fixture, monkeypatch)
    for label, binary in service.LABELS.items():
        item = plistlib.loads(plistlib.dumps(plists[label + ".plist"]))
        assert item["ProgramArguments"] == [str(app / "Contents/MacOS" / binary)]
        assert item["AssociatedBundleIdentifiers"] == ["com.brainlayer.brainbar"]
        assert item["KeepAlive"] and item["Label"] == label
        env = item["EnvironmentVariables"]
        assert set(env) == {
            "PATH",
            "BRAINBAR_SOCKET_PATH",
            "BRAINLAYER_MCP_SOCKET",
            "BRAINBAR_DB_PATH",
            "BRAINLAYER_DB",
            "BRAINBAR_PYTHON",
            "BRAINLAYER_CLI",
            "BRAINLAYER_ENV_FILE",
        }
        assert env["BRAINBAR_SOCKET_PATH"] == env["BRAINLAYER_MCP_SOCKET"] == args.socket
        assert env["BRAINLAYER_DB"] == env["BRAINBAR_DB_PATH"] == str(home / ".local/share/brainlayer/brainlayer.db")
        assert str(home / "Library/Logs/brainlayer") in item["StandardErrorPath"]
        assert "synthetic-secret" not in str(item) and "production" not in str(item)
    assert not (home / "Library").exists(), "render must make no directories or installed artifacts"


def test_install_is_owned_private_and_does_not_activate(fixture, monkeypatch):
    home, _, _ = fixture
    plists = prepare(fixture, monkeypatch)
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: pytest.fail("no job/tool operation during install"))
    service.install(plists, home, os.getuid())
    for name, expected in plists.items():
        path = home / "Library/LaunchAgents" / name
        assert plistlib.loads(path.read_bytes()) == expected
        assert path.stat().st_mode & 0o777 == 0o600
    assert not Path(plists["com.brainlayer.brainbar.plist"]["EnvironmentVariables"]["BRAINBAR_SOCKET_PATH"]).exists()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "writable", "foreign", "outside", "fifo"])
def test_refuses_unsafe_paths_without_modifying_sentinel(tmp_path, kind):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    target = home / "target"
    sentinel = home / "sentinel"
    sentinel.write_bytes(b"unchanged")
    uid = os.getuid()
    if kind == "symlink":
        target.symlink_to(sentinel)
    elif kind == "hardlink":
        os.link(sentinel, target)
    elif kind == "fifo":
        os.mkfifo(target)
    else:
        target.write_bytes(b"unchanged")
    if kind == "writable":
        target.chmod(0o666)
    if kind == "foreign":
        uid += 1
    if kind == "outside":
        target = tmp_path / "elsewhere"
    with pytest.raises(ValueError):
        service.owned(target, home, uid, missing=True)
    assert sentinel.read_bytes() == b"unchanged"


def test_signature_failure_refuses_before_install(fixture, monkeypatch):
    home, _, args = fixture

    def reject(command):
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        service.prepare(args, home, os.getuid(), reject)
    assert not (home / "Library").exists()


def test_two_account_templates_keep_labels_and_default_socket():
    for uid in (501, 502):
        home = Path(f"/Users/account{uid}")
        for label in service.LABELS:
            template = plistlib.loads((ROOT / "brain-bar/bundle" / (label + ".plist")).read_bytes())
            item = service.render(
                template, home / "Applications/BrainBar.app", home, {"BRAINBAR_SOCKET_PATH": "/tmp/brainbar.sock"}
            )
            assert item["Label"] == label and f"account{uid}" in item["ProgramArguments"][0]
            assert item["EnvironmentVariables"]["BRAINBAR_SOCKET_PATH"] == "/tmp/brainbar.sock"
            assert item["StandardOutPath"].startswith(str(home))


def test_socket_limit_and_missing_explicit_cli_are_loud(fixture):
    home, _, args = fixture
    with pytest.raises(ValueError, match="sockaddr_un"):
        service.endpoint(home / ("x" * 104), home, os.getuid())
    args.cli = str(home / "absent")
    with pytest.raises(ValueError, match="missing path"):
        service.prepare(args, home, os.getuid(), lambda command: None)


def test_cli_render_only_has_current_uid_domain_and_no_activation(fixture, monkeypatch, capsys):
    home, app, args = fixture
    calls = []
    monkeypatch.setattr(service.pwd, "getpwuid", lambda uid: Namespace(pw_dir=str(home)))
    monkeypatch.setattr(service.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "install-services.py",
            "--app",
            str(app),
            "--python",
            args.python,
            "--cli",
            args.cli,
            "--socket",
            args.socket,
            "--render-only",
        ],
    )
    service.main()
    output = json.loads(capsys.readouterr().out)
    assert output["domain"] == f"gui/{os.getuid()}" and output["activated"] is False
    assert calls == [["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)]]
    assert not (home / "Library").exists()


@pytest.mark.parametrize("leaf", ["socket-file", "socket-directory", "lock-directory", "lock-fifo"])
def test_private_endpoint_leaf_types_are_distinct(fixture, leaf):
    home, _, args = fixture
    target = Path(args.socket + ".lock") if leaf.startswith("lock") else Path(args.socket)
    if leaf.endswith("directory"):
        target.mkdir(mode=0o700)
    elif leaf.endswith("fifo"):
        os.mkfifo(target, mode=0o600)
    else:
        target.write_bytes(b"keep data")
        target.chmod(0o600)
    with pytest.raises(ValueError):
        service.prepare(args, home, os.getuid(), lambda command: None)
    assert target.exists() and not (home / "Library").exists()
    if leaf == "socket-file":
        assert target.read_bytes() == b"keep data"


@pytest.mark.parametrize("endpoint_name", ["socket", "lock"])
@pytest.mark.parametrize("exists", [False, True])
def test_database_cannot_alias_either_endpoint_even_when_missing(fixture, endpoint_name, exists):
    home, _, args = fixture
    args.db = args.socket + (".lock" if endpoint_name == "lock" else "")
    target = Path(args.db)
    if exists:
        target.write_bytes(b"database sentinel")
        target.chmod(0o600)
    with pytest.raises(ValueError):
        service.prepare(args, home, os.getuid(), lambda command: None)
    assert target.exists() == exists and not (home / "Library").exists()
    if exists:
        assert target.read_bytes() == b"database sentinel"


def test_owned_private_socket_and_regular_lock_remain_valid(fixture, monkeypatch):
    home, _, args = fixture
    with socket.socket(socket.AF_UNIX) as private_socket:
        private_socket.bind(args.socket)
        Path(args.socket).chmod(0o600)
        lock = Path(args.socket + ".lock")
        lock.write_bytes(b"lock sentinel")
        lock.chmod(0o600)
        assert prepare(fixture, monkeypatch)
        assert lock.read_bytes() == b"lock sentinel" and not (home / "Library").exists()


def test_missing_default_endpoint_remains_valid_without_accessing_shared_state(monkeypatch):
    monkeypatch.setattr(service.os.path, "lexists", lambda path: False)
    assert service.endpoint("/tmp/brainbar.sock", Path("/Users/synthetic"), os.getuid()) == Path("/tmp/brainbar.sock")
