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


def test_preflight_env_off_overrides_enabled_label(setup):
    from brainlayer.consolidation_plan import collect_legacy

    # The actual installed service key controls the gate, as in PR1282 R2.
    path = setup[1] / f"{OLD}.plist"
    plist = plistlib.loads(path.read_bytes())
    plist["EnvironmentVariables"] = {"BRAINLAYER_LAUNCHD_SERVICE": "watch"}
    path.write_bytes(plistlib.dumps(plist))
    setup[2].write_text("BRAINLAYER_LAUNCHD_WATCH_ENABLED=disabled\n")
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


def test_renderer_preserves_xml_special_paths_and_named_launcher_args(setup):
    from brainlayer.consolidation_plan import rendered_launchers

    home = setup[0] / "home & <scratch>"
    python = home / "opt/brainlayer/libexec/venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    rendered = rendered_launchers(ROOT / "scripts/launchd", home, python, setup[2])
    daemon = plistlib.loads(rendered[NEW])
    assert daemon["ProgramArguments"][0] == str(home / ".local/lib/brainlayer/BrainLayer Daemon")
    assert daemon["ProgramArguments"][1] == str(python)
    assert daemon["EnvironmentVariables"]["BRAINLAYER_REPO_ROOT"] == str(python.parents[3])
    calendar = plistlib.loads(rendered["com.brainlayer.maintenance"])
    assert calendar["ProgramArguments"][0].endswith("/BrainLayer Maintenance")
    assert calendar["StartCalendarInterval"] == {"Hour": 3, "Minute": 15}
