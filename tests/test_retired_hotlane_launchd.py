"""The packaged hotlane job requires only local embedding configuration."""

import plistlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hotlane_source_template_requires_only_local_embedding():
    config = plistlib.loads((REPO_ROOT / "scripts/launchd/com.brainlayer.hotlane-brainbar.plist").read_bytes())
    assert not any(arg.startswith("--enrich-") for arg in config["ProgramArguments"])
    env = config["EnvironmentVariables"]
    assert not any("GOOGLE" in key or "GEMINI" in key or "ENRICH" in key for key in env)
    assert "--backlog-batch" in config["ProgramArguments"]
    assert env["BRAINLAYER_LAUNCHD_SERVICE"] == "hotlane-brainbar"


def test_hotlane_env_runner_needs_no_google_credentials(tmp_path):
    config = plistlib.loads((REPO_ROOT / "scripts/launchd/com.brainlayer.hotlane-brainbar.plist").read_bytes())
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_SYSTEM_ENABLED=1\n")
    env_file.chmod(0o600)
    env = {
        **config["EnvironmentVariables"],
        "HOME": str(tmp_path),
        "BRAINLAYER_ENV_FILE": str(env_file),
        "PATH": "/usr/bin:/bin",
    }
    result = subprocess.run(
        ["/bin/bash", str(REPO_ROOT / "scripts/launchd/brainlayer-env-run.sh"), "/usr/bin/true"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_hotlane_install_action_needs_no_google_credentials(tmp_path):
    source = (REPO_ROOT / "scripts/launchd/install.sh").read_text()
    action = source.split("    hotlane|hotlane-brainbar)\n", 1)[1].split("        ;;", 1)[0]
    # Execute only the real action branch with fake functions; no launchctl, service or package writes.
    harness = (
        "set -euo pipefail\n"
        "verify_gemini_env_file() { echo 'retired key gate' >&2; return 78; }\n"
        "install_plist() { printf 'LOCAL %s\\n' \"$1\"; }\n" + action
    )
    result = subprocess.run(
        ["/bin/bash", "-c", harness],
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "LOCAL hotlane-brainbar\n"


def test_hotlane_template_arguments_match_local_cli(monkeypatch):
    import importlib
    import sys

    hotlane = importlib.import_module("scripts.hotlane_brainbar_daemon")
    config = plistlib.loads((REPO_ROOT / "scripts/launchd/com.brainlayer.hotlane-brainbar.plist").read_bytes())
    calls = []
    monkeypatch.setattr(hotlane, "run", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(hotlane.signal, "signal", lambda *args: None)
    local_args = config["ProgramArguments"][3:]
    old_1548_args = ["--enrich-interval", "10.0", "--enrich-limit", "0", "--enrich-since-hours", "87600"]
    for args in (local_args, [*local_args, *old_1548_args]):
        calls.clear()
        monkeypatch.setattr(sys, "argv", ["hotlane", *args])
        hotlane.main()
        assert len(calls) == 1
        assert calls[0]["backlog_batch"] == 16
        assert calls[0]["enrich_limit"] == 0
