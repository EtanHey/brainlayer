from __future__ import annotations

import os
import plistlib
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = "__HOME__/Library/Logs/brainlayer/"
RENDERED_LOG_ROOT = "/Users/etanheyman/Library/Logs/brainlayer/"
REQUIRED_PATH_PARTS = ["/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
DEV_SRC_PATH_RE = re.compile(r"/Users/[^:\s]+/Gits/[^:\s]+/src(?:$|:)")


def _load(path: str) -> dict:
    with (REPO_ROOT / path).open("rb") as handle:
        return plistlib.load(handle)


def _assert_common_hygiene(plist: dict) -> None:
    env = plist.get("EnvironmentVariables")
    assert isinstance(env, dict)
    path_dirs = [os.path.normpath(part) for part in env.get("PATH", "").split(os.pathsep) if part]
    for part in REQUIRED_PATH_PARTS:
        assert os.path.normpath(part) in path_dirs

    assert plist["StandardOutPath"].startswith((LOG_ROOT, RENDERED_LOG_ROOT))
    assert plist["StandardErrorPath"].startswith((LOG_ROOT, RENDERED_LOG_ROOT))

    limits = plist.get("SoftResourceLimits")
    assert isinstance(limits, dict)
    assert limits.get("NumberOfFiles", 0) >= 4096
    assert "ProcessType" in plist
    assert "ExitTimeOut" in plist


def _assert_no_dev_src_path_in_canonical_env(path: Path, plist: dict) -> None:
    label = plist.get("Label", "")
    if not isinstance(label, str) or not label.startswith("com.brainlayer."):
        return
    env = plist.get("EnvironmentVariables") or {}
    assert isinstance(env, dict), str(path)
    for key, value in env.items():
        assert not DEV_SRC_PATH_RE.search(str(value)), f"{path}: {key} leaks a concrete dev /src path"


def _assert_uses_installed_package_not_source_path(path: Path, plist: dict) -> None:
    env = plist.get("EnvironmentVariables") or {}
    assert "PYTHONPATH" not in env, f"{path}: canonical LaunchAgent must import installed brainlayer package"
    assert env.get("BRAINLAYER_REPO_ROOT") == "__BRAINLAYER_DIR__"


def test_active_daemon_launchd_hygiene_matrix():
    cases = {
        "brain-bar/bundle/com.brainlayer.brainbar.plist": {
            "ProcessType": "Interactive",
            "ExitTimeOut": 30,
            "LowPriorityIO": False,
            "KeepAlive": True,
        },
        "scripts/launchd/com.brainlayer.enrichment.plist": {
            "ProcessType": "Background",
            "ExitTimeOut": 120,
            "LowPriorityIO": False,
            "KeepAlive": True,
        },
        "scripts/launchd/com.brainlayer.watch.plist": {
            "ProcessType": "Background",
            "ExitTimeOut": 30,
            "LowPriorityIO": True,
            "KeepAlive": True,
        },
        "launchd/com.brainlayer.watch.plist": {
            "ProcessType": "Background",
            "ExitTimeOut": 30,
            "LowPriorityIO": True,
            "KeepAlive": True,
        },
        "scripts/launchd/com.brainlayer.drain.plist": {
            "ProcessType": "Background",
            "ExitTimeOut": 60,
            "LowPriorityIO": True,
            "KeepAlive": True,
            "ThrottleInterval": 10,
        },
        "scripts/launchd/com.brainlayer.backup-daily.plist": {
            "ProcessType": "Background",
            "ExitTimeOut": 300,
            "LowPriorityIO": True,
        },
    }

    for path, expected in cases.items():
        plist = _load(path)
        _assert_common_hygiene(plist)
        for key, value in expected.items():
            assert plist.get(key) == value, path

    drain = _load("scripts/launchd/com.brainlayer.drain.plist")
    assert "WatchPaths" not in drain
    assert "QueueDirectories" not in drain
    assert "--once" not in drain["ProgramArguments"]

    backup = _load("scripts/launchd/com.brainlayer.backup-daily.plist")
    assert "KeepAlive" not in backup
    assert "StartCalendarInterval" in backup


def test_all_script_launchd_plists_have_common_hygiene():
    for path in sorted((REPO_ROOT / "scripts/launchd").glob("com.brainlayer.*.plist")):
        _assert_common_hygiene(plistlib.loads(path.read_bytes()))


def test_index_launchagent_runs_nightly_without_keepalive_or_run_at_load():
    index = _load("scripts/launchd/com.brainlayer.index.plist")

    assert "KeepAlive" not in index
    assert "RunAtLoad" not in index
    assert "StartInterval" not in index
    assert index["StartCalendarInterval"] == {"Hour": 3, "Minute": 15}


def test_canonical_launchagent_env_has_no_concrete_dev_src_paths():
    plist_paths = [
        *sorted((REPO_ROOT / "scripts/launchd").glob("com.brainlayer.*.plist")),
        REPO_ROOT / "launchd/com.brainlayer.watch.plist",
        REPO_ROOT / "brain-bar/bundle/com.brainlayer.brainbar.plist",
        REPO_ROOT / "brain-bar/bundle/com.brainlayer.brainbar-daemon.plist",
    ]

    for path in plist_paths:
        _assert_no_dev_src_path_in_canonical_env(path, plistlib.loads(path.read_bytes()))


def test_script_launchagents_use_installed_package_imports():
    for path in sorted((REPO_ROOT / "scripts/launchd").glob("com.brainlayer.*.plist")):
        _assert_uses_installed_package_not_source_path(path, plistlib.loads(path.read_bytes()))


def test_enrichment_launchagent_sources_standard_env_file_without_embedded_google_key():
    plist = _load("scripts/launchd/com.brainlayer.enrichment.plist")
    env = plist["EnvironmentVariables"]
    args = plist["ProgramArguments"]

    assert env["BRAINLAYER_ENV_FILE"] == "__BRAINLAYER_ENV_FILE__"
    assert env["BRAINLAYER_REQUIRE_GOOGLE_API_KEY"] == "1"
    assert "GOOGLE_API_KEY" not in env
    assert "__GOOGLE_API_KEY__" not in plistlib.dumps(plist).decode("utf-8")
    assert args[:2] == ["__BRAINLAYER_ENV_RUN__", "__BRAINLAYER_BIN__"]
    assert args[2:] == ["enrich", "--mode", "realtime", "--supervisor"]


def test_all_script_launchagents_source_unified_config_file():
    for path in sorted((REPO_ROOT / "scripts/launchd").glob("com.brainlayer.*.plist")):
        plist = plistlib.loads(path.read_bytes())
        args = plist["ProgramArguments"]
        env = plist["EnvironmentVariables"]
        service = plist["Label"].removeprefix("com.brainlayer.")

        assert args[0] == "__BRAINLAYER_ENV_RUN__", str(path)
        assert env["BRAINLAYER_ENV_FILE"] == "__BRAINLAYER_ENV_FILE__", str(path)
        assert env["BRAINLAYER_LAUNCHD_SERVICE"] == service, str(path)


def test_launchd_env_loader_exists_and_sources_env_before_exec():
    loader = REPO_ROOT / "scripts/launchd/brainlayer-env-run.sh"
    content = loader.read_text(encoding="utf-8")

    assert 'ENV_FILE="${BRAINLAYER_ENV_FILE:-$HOME/.config/brainlayer/brainlayer.env}"' in content
    assert "BRAINLAYER_SYSTEM_ENABLED" in content
    assert "BRAINLAYER_LAUNCHD_SERVICE" in content
    assert "BRAINLAYER_LAUNCHD_${service_key}_ENABLED" in content
    assert "BRAINLAYER_ENRICH_ENABLED" in content
    assert "current user or root" in content
    assert "world-writable" in content
    assert "set -a" in content
    assert 'source "$ENV_FILE"' in content
    assert 'exec "$@"' in content
    assert "GOOGLE_API_KEY" in content


def test_launchd_env_loader_rejects_world_writable_env_file(tmp_path):
    loader = REPO_ROOT / "scripts/launchd/brainlayer-env-run.sh"
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("GOOGLE_API_KEY='test-secret'\n", encoding="utf-8")
    env_file.chmod(0o666)

    result = subprocess.run(
        [str(loader), "/usr/bin/true"],
        env={
            **os.environ,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_REQUIRE_GOOGLE_API_KEY": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "must not be world-writable" in result.stderr


def test_launchd_env_loader_honors_service_disable_toggle(tmp_path):
    loader = REPO_ROOT / "scripts/launchd/brainlayer-env-run.sh"
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text(
        "\n".join(
            [
                "BRAINLAYER_LAUNCHD_DRAIN_ENABLED=0",
                "BRAINLAYER_DISABLED_SLEEP_SECONDS=0",
            ]
        ),
        encoding="utf-8",
    )
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(loader), "/bin/sh", "-c", "exit 42"],
        env={
            **os.environ,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_SERVICE": "drain",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "disabled by config" in result.stderr


def test_launchd_env_loader_normalizes_auto_enrich_false_values(tmp_path):
    loader = REPO_ROOT / "scripts/launchd/brainlayer-env-run.sh"
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=off\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(loader), "/bin/sh", "-c", 'test "$BRAINLAYER_AUTO_ENRICH" = 0'],
        env={
            **os.environ,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_SERVICE": "watch",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0


def test_launchd_env_loader_skip_disable_gates_still_checks_required_key(tmp_path):
    loader = REPO_ROOT / "scripts/launchd/brainlayer-env-run.sh"
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text(
        "\n".join(
            [
                "BRAINLAYER_ENRICH_ENABLED=0",
                "BRAINLAYER_DISABLED_SLEEP_SECONDS=99",
            ]
        ),
        encoding="utf-8",
    )
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(loader), "/usr/bin/true"],
        env={
            **os.environ,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_SERVICE": "enrichment",
            "BRAINLAYER_REQUIRE_GOOGLE_API_KEY": "1",
            "BRAINLAYER_SKIP_DISABLE_GATES": "1",
        },
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert result.returncode != 0
    assert "GOOGLE_API_KEY not set" in result.stderr


def test_launchd_installer_rejects_key_only_enrichment_config(tmp_path):
    launchd_dir = tmp_path / "launchd"
    launchd_dir.mkdir()
    shutil.copy(REPO_ROOT / "scripts/launchd/brainlayer-env-run.sh", launchd_dir / "brainlayer-env-run.sh")
    install_source = (REPO_ROOT / "scripts/launchd/install.sh").read_text(encoding="utf-8")
    functions_only = install_source.split('case "${1:-all}" in', 1)[0]
    harness = launchd_dir / "verify-config.sh"
    harness.write_text(functions_only + "\nverify_gemini_env_file\n", encoding="utf-8")
    harness.chmod(0o755)

    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("GOOGLE_API_KEY='test-secret'\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(harness)],
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "BRAINLAYER_BIN": "/usr/bin/true",
            "PYTHON_BIN": "/usr/bin/true",
            "BRAINLAYER_ENV_FILE": str(env_file),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "missing BRAINLAYER_ENRICH_ENABLED" in result.stderr
    assert "missing required enrichment config keys" in result.stdout


def test_launchd_installer_wires_health_check_target():
    install_source = (REPO_ROOT / "scripts/launchd/install.sh").read_text(encoding="utf-8")

    assert "./scripts/launchd/install.sh health-check" in install_source
    assert "health-check)" in install_source
    assert "install_plist health-check" in install_source
    assert "remove_plist health-check" in install_source


def test_launchd_installer_uses_bootstrap_not_legacy_load_unload():
    install_source = (REPO_ROOT / "scripts/launchd/install.sh").read_text(encoding="utf-8")
    load_plist_body = install_source.split("load_plist() {", 1)[1].split("\nunload_plist() {", 1)[0]

    assert "launchctl enable" in load_plist_body
    assert "launchctl bootout" in load_plist_body
    assert "launchctl bootstrap" in load_plist_body
    assert "launchctl print" in load_plist_body
    assert "launchctl load" not in load_plist_body
    assert "launchctl unload" not in load_plist_body
