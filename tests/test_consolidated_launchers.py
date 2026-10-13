"""Inactive consolidated launcher definitions preserve calendar/custody contracts."""

import plistlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,command", [("daemon", "daemon"), ("maintenance", "maintenance-calendar")])
def test_consolidated_launcher_template(name, command, tmp_path):
    path = ROOT / "scripts/launchd" / f"com.brainlayer.{name}.plist"
    plist = plistlib.loads(path.read_bytes())
    assert plist["Label"] == f"com.brainlayer.{name}"
    assert plist["AssociatedBundleIdentifiers"] == ["com.brainlayer.brainbar"]
    assert plist["ExitTimeOut"] >= 25
    argv = plist["ProgramArguments"]
    assert argv[:5] == ["__BRAINLAYER_ENV_RUN__", "__BRAINLAYER_PYTHON__", "-m", "brainlayer", command]
    assert plist["EnvironmentVariables"]["BRAINLAYER_SUPERVISOR_SOCKET"].endswith("/supervisor/control.sock")
    assert plist["EnvironmentVariables"]["BRAINLAYER_ENV_FILE"] == "__BRAINLAYER_ENV_FILE__"
    if name == "daemon":
        assert plist["KeepAlive"] is True and plist["RunAtLoad"] is True
        assert argv[5:] == ["--manifest", "__HOME__/.local/share/brainlayer/supervisor/jobs.json"]
        assert "StartCalendarInterval" not in plist
    else:
        assert plist["StartCalendarInterval"] == {"Hour": 3, "Minute": 15}
        assert not plist.get("KeepAlive") and not plist.get("RunAtLoad")
        assert "StartInterval" not in plist
        assert argv[argv.index("--state-dir") + 1].endswith("/supervisor/calendar")
        assert argv[argv.index("--lib-dir") + 1] == "__HOME__/.local/lib/brainlayer"
    # Round-trip renderer substitutes a scratch HOME and an explicit keg path,
    # never executes a launcher or consults this machine's LaunchAgents.
    rendered = path.read_text().replace("__HOME__", str(tmp_path))
    rendered = rendered.replace("__BRAINLAYER_PYTHON__", "/opt/homebrew/opt/brainlayer/libexec/venv/bin/python")
    rendered = rendered.replace("__BRAINLAYER_ENV_RUN__", str(tmp_path / "BrainLayer Launcher"))
    rendered = rendered.replace("__BRAINLAYER_ENV_FILE__", str(tmp_path / "brainlayer.env"))
    ready = plistlib.loads(rendered.encode())
    assert ready["ProgramArguments"][1] == "/opt/homebrew/opt/brainlayer/libexec/venv/bin/python"
    assert all("__" not in token for token in ready["ProgramArguments"])
