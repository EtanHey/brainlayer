"""Scratch-only migration invariants; the runner refuses every non-fixture domain."""

import plistlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = "gui/424242"
OLD = "com.brainlayer.watch"
NEW = "com.brainlayer.daemon"


@pytest.fixture
def setup(tmp_path):
    home = tmp_path / "home"
    agents = home / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    env = home / ".config/brainlayer/brainlayer.env"
    env.parent.mkdir(parents=True)
    env.write_text("BRAINLAYER_LAUNCHD_WATCH_ENABLED=1\n")
    env.chmod(0o600)
    for label in (OLD, "com.etanhey.brainlayer-fleet-watchdog"):
        plist = {"Label": label, "ProgramArguments": ["/bin/true"], "KeepAlive": True}
        (agents / f"{label}.plist").write_bytes(plistlib.dumps(plist))
        (agents / f"{label}.plist.loaded-keg").write_text("previous-keg\n")
    return home, agents, env


def test_preflight_preserves_off_flag_and_exact_marker_bytes(setup):
    from brainlayer.consolidation_plan import collect_legacy

    records, files = collect_legacy(setup[1], setup[2], {OLD})
    assert records[OLD]["enabled"] is False
    assert files[f"{OLD}.plist.loaded-keg"] == b"previous-keg\n"
    assert files[f"{OLD}.plist"] == (setup[1] / f"{OLD}.plist").read_bytes()


def test_preflight_rejects_linked_personal_file_before_any_mutation(setup, tmp_path):
    from brainlayer.consolidation_plan import collect_legacy

    path = setup[1] / f"{OLD}.plist"
    original = path.read_bytes()
    path.unlink()
    personal = tmp_path / "personal.plist"
    personal.write_bytes(original)
    path.symlink_to(personal)
    with pytest.raises(OSError):
        collect_legacy(setup[1], setup[2], set())
    assert personal.read_bytes() == original


@pytest.mark.parametrize("value", ["trueExtra", "unknown", "1"])
def test_disabled_parser_refuses_unknown_values(value):
    from brainlayer.consolidation_plan import disabled_labels

    with pytest.raises(ValueError):
        disabled_labels(f'disabled services = {{\n"{OLD}" => {value}\n}}')


def test_renderer_refuses_nonexistent_interpreter(setup):
    from brainlayer.consolidation_plan import rendered_launchers

    with pytest.raises(ValueError, match="interpreter"):
        rendered_launchers(ROOT / "scripts/launchd", setup[0], setup[0] / "missing/python", setup[2])


@pytest.mark.parametrize("custom", [False, True])
def test_preflight_env_off_overrides_enabled_label(setup, custom):
    from brainlayer.consolidation_plan import collect_legacy

    selected = setup[2].with_name("watch.env") if custom else setup[2]
    path = setup[1] / f"{OLD}.plist"
    plist = plistlib.loads(path.read_bytes())
    plist["EnvironmentVariables"] = dict(BRAINLAYER_LAUNCHD_SERVICE="watch", BRAINLAYER_ENV_FILE=str(selected))
    path.write_bytes(plistlib.dumps(plist))
    selected.write_text("BRAINLAYER_LAUNCHD_WATCH_ENABLED=disabled\n")
    records, _ = collect_legacy(setup[1], setup[2], set())
    assert records[OLD]["enabled"] is False


def test_preflight_refuses_fifo_without_waiting_for_a_writer(tmp_path):
    import os
    import sys

    fifo = tmp_path / "legacy.plist"
    os.mkfifo(fifo)
    code = (
        "import sys; from pathlib import Path; from brainlayer.consolidation_plan import read_owned; "
        "\ntry: read_owned(Path(sys.argv[1]))\nexcept ValueError: print('refused')"
    )
    result = subprocess.run([sys.executable, "-c", code, str(fifo)], capture_output=True, text=True, timeout=3)
    assert result.returncode == 0 and result.stdout.strip() == "refused", result.stderr


def test_renderer_preserves_xml_special_paths_and_named_launcher_args(setup, monkeypatch):
    from brainlayer.consolidation_plan import rendered_launchers

    home = setup[0] / "home & <scratch>"
    python = home / "opt/brainlayer/libexec/venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    site = python.parents[1] / "lib/python3.13/site-packages"
    (site / "brainlayer").mkdir(parents=True)
    monkeypatch.setenv("BRAINLAYER_HOOK_PYTHON", str(python))
    rendered = rendered_launchers(ROOT / "scripts/launchd", home, python, setup[2])
    daemon = plistlib.loads(rendered[NEW])
    assert daemon["ProgramArguments"][0] == str(home / ".local/lib/brainlayer/BrainLayer Daemon")
    assert daemon["ProgramArguments"][1] == str(python)
    assert daemon["EnvironmentVariables"]["BRAINLAYER_REPO_ROOT"] == str(site)
    calendar = plistlib.loads(rendered["com.brainlayer.maintenance"])
    assert calendar["ProgramArguments"][0].endswith("/BrainLayer Maintenance")
    assert calendar["StartCalendarInterval"] == {"Hour": 3, "Minute": 15}


@pytest.mark.parametrize("suffix", [".plist", ".plist.loaded-keg", ".loaded-keg"])
def test_unknown_brainlayer_label_is_refused_with_path(setup, suffix):
    from brainlayer.consolidation_plan import collect_legacy

    label = "com.brainlayer.reembed-backfill"
    path = setup[1] / (label + suffix)
    path.write_bytes(plistlib.dumps({"Label": label}) if suffix == ".plist" else b"previous-keg\n")
    with pytest.raises(ValueError, match=label) as error:
        collect_legacy(setup[1], setup[2], set())
    assert str(path) in str(error.value)


@pytest.mark.parametrize("editable", ["_editable_impl_brainlayer.pth", "__editable___brainlayer_finder.py", "path.pth"])
def test_renderer_refuses_editable_override(setup, monkeypatch, editable):
    from brainlayer.consolidation_plan import rendered_launchers

    python = setup[0] / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    site = python.parents[1] / "lib/python3.13/site-packages"
    site.mkdir(parents=True)
    (site / editable).write_text(str(ROOT / "src") + "\n")
    monkeypatch.setenv("BRAINLAYER_HOOK_PYTHON", str(python))
    with pytest.raises(ValueError, match="editable"):
        rendered_launchers(ROOT / "scripts/launchd", setup[0], python, setup[2])


def test_renderer_requires_explicit_override_for_non_keg(setup, monkeypatch):
    from brainlayer.consolidation_plan import rendered_launchers

    monkeypatch.delenv("BRAINLAYER_HOOK_PYTHON", raising=False)
    with pytest.raises(ValueError, match="interpreter"):
        rendered_launchers(ROOT / "scripts/launchd", setup[0], ROOT / ".venv/bin/python", setup[2])


def test_unmanaged_launchers_and_markers_are_untouched(setup):
    from brainlayer.consolidation_plan import UNMANAGED, collect_legacy

    for label in UNMANAGED:
        (setup[1] / f"{label}.plist").write_bytes(plistlib.dumps({"Label": label}))
        (setup[1] / f"{label}.plist.loaded-keg").write_bytes(b"unmanaged\n")
    _, files = collect_legacy(setup[1], setup[2], set())
    assert all(not name.startswith(label) for label in UNMANAGED for name in files)


@pytest.mark.parametrize("line", ["import sys; sys.path.insert(0, '/checkout')", "import external_finder"])
def test_renderer_refuses_executable_pth_without_editable_name(setup, monkeypatch, line):
    from brainlayer.consolidation_plan import interpreter_site

    python = setup[0] / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    site = python.parents[1] / "lib/python3.13/site-packages"
    (site / "brainlayer").mkdir(parents=True)
    (site / "injection.pth").write_text(line + "\n")
    monkeypatch.setenv("BRAINLAYER_HOOK_PYTHON", str(python))
    with pytest.raises(ValueError, match="executable"):
        interpreter_site(python)


def test_renderer_refuses_linked_site_packages(setup, tmp_path, monkeypatch):
    from brainlayer.consolidation_plan import interpreter_site

    python = setup[0] / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    external = tmp_path / "external"
    (external / "brainlayer").mkdir(parents=True)
    site = python.parents[1] / "lib/python3.13/site-packages"
    site.parent.mkdir(parents=True)
    site.symlink_to(external, target_is_directory=True)
    monkeypatch.setenv("BRAINLAYER_HOOK_PYTHON", str(python))
    with pytest.raises(ValueError, match="linked site-packages"):
        interpreter_site(python)


@pytest.mark.parametrize(
    "label,key,value",
    [
        (NEW, "KeepAlive", False),
        (NEW, "RunAtLoad", False),
        (NEW, "StartCalendarInterval", {"Hour": 4}),
        ("com.brainlayer.maintenance", "StartCalendarInterval", {"Hour": 4}),
        ("com.brainlayer.maintenance", "RunAtLoad", True),
        ("com.brainlayer.maintenance", "KeepAlive", True),
    ],
)
def test_renderer_refuses_changed_launcher_behavior(setup, tmp_path, monkeypatch, label, key, value):
    from brainlayer.consolidation_plan import rendered_launchers

    templates = tmp_path / "templates"
    templates.mkdir()
    for name in (NEW, "com.brainlayer.maintenance"):
        raw = (ROOT / "scripts/launchd" / f"{name}.plist").read_bytes()
        plist = plistlib.loads(raw)
        if name == label:
            plist[key] = value
        (templates / f"{name}.plist").write_bytes(plistlib.dumps(plist))
    python = setup[0] / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    (python.parents[1] / "lib/python3.13/site-packages/brainlayer").mkdir(parents=True)
    monkeypatch.setenv("BRAINLAYER_HOOK_PYTHON", str(python))
    with pytest.raises(ValueError, match="launch behavior"):
        rendered_launchers(templates, setup[0], python, setup[2])
