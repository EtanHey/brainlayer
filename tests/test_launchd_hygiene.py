from __future__ import annotations

import os
import plistlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from brainlayer.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = "__HOME__/Library/Logs/brainlayer/"
RENDERED_LOG_ROOT = "/Users/etanheyman/Library/Logs/brainlayer/"
REQUIRED_PATH_PARTS = ["/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
DEV_SRC_PATH_RE = re.compile(r"/Users/[^:\s]+/Gits/[^:\s]+/src(?:$|:)")
ENV_RUN_EXEMPT_LABELS = {
    "com.brainlayer.tier0-watchdog",
}
PACKAGE_IMPORT_EXEMPT_LABELS = ENV_RUN_EXEMPT_LABELS | {
    "com.brainlayer.throughput-watchdog",
}


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
        "scripts/launchd/com.brainlayer.hotlane-brainbar.plist": {
            "ProcessType": "Background",
            "ExitTimeOut": 30,
            "LowPriorityIO": True,
            "KeepAlive": True,
            "ThrottleInterval": 5,
        },
        "scripts/launchd/com.brainlayer.backup-daily.plist": {
            "ProcessType": "Background",
            "ExitTimeOut": 300,
            "LowPriorityIO": True,
        },
        "scripts/launchd/com.brainlayer.throughput-watchdog.plist": {
            "ProcessType": "Background",
            "ExitTimeOut": 30,
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

    hotlane = _load("scripts/launchd/com.brainlayer.hotlane-brainbar.plist")
    hotlane_args = hotlane["ProgramArguments"]
    assert hotlane_args[hotlane_args.index("--backlog-batch") + 1] == "4"
    assert hotlane_args[hotlane_args.index("--enrich-limit") + 1] == "0"

    backup = _load("scripts/launchd/com.brainlayer.backup-daily.plist")
    assert "KeepAlive" not in backup
    assert "StartCalendarInterval" in backup


def test_all_script_launchd_plists_have_common_hygiene():
    for path in sorted((REPO_ROOT / "scripts/launchd").glob("com.brainlayer.*.plist")):
        _assert_common_hygiene(plistlib.loads(path.read_bytes()))


def test_drain_launchagent_uses_stable_brainlayer_shim():
    drain = _load("scripts/launchd/com.brainlayer.drain.plist")

    assert drain["ProgramArguments"][:5] == [
        "__BRAINLAYER_ENV_RUN__",
        "/usr/bin/env",
        "BRAINLAYER_DRAIN_EMBED=0",
        "__BRAINLAYER_BIN__",
        "drain",
    ]
    assert "--daemon" in drain["ProgramArguments"]
    assert "BRAINLAYER_DRAIN_EMBED" not in drain["EnvironmentVariables"]
    assert "__PYTHON_BIN__" not in drain["ProgramArguments"]
    assert "-m" not in drain["ProgramArguments"]
    assert "brainlayer.drain" not in drain["ProgramArguments"]


def test_drain_launchagent_embed_override_survives_env_file_loading(tmp_path):
    loader = REPO_ROOT / "scripts/launchd/brainlayer-env-run.sh"
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_DRAIN_EMBED=1\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [
            str(loader),
            "/usr/bin/env",
            "BRAINLAYER_DRAIN_EMBED=0",
            "/bin/sh",
            "-c",
            'test "$BRAINLAYER_DRAIN_EMBED" = "0"',
        ],
        env={
            **os.environ,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_SERVICE": "drain",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_drain_cli_daemon_records_launch_provenance(monkeypatch):
    from brainlayer import deploy_drift, drain, parent_death

    calls = []
    monkeypatch.setattr(parent_death, "install_parent_death_watcher", lambda: calls.append("parent-death"))
    monkeypatch.setattr(deploy_drift, "record_launch_from_environment", lambda: calls.append("provenance"))
    monkeypatch.setattr(
        drain, "run_daemon", lambda interval, batch_size: calls.append(("daemon", interval, batch_size))
    )

    result = CliRunner().invoke(app, ["drain", "--daemon", "--interval", "0", "--batch-size", "7"])

    assert result.exit_code == 0
    assert calls == ["parent-death", "provenance", ("daemon", 0.0, 7)]


def test_p0_counter_launchagent_uses_brainlayer_console_shim():
    p0 = _load("scripts/launchd/com.brainlayer.p0-counter.plist")

    assert p0["ProgramArguments"] == [
        "__BRAINLAYER_ENV_RUN__",
        "__BRAINLAYER_BIN__",
        "p0-counter",
    ]
    assert p0["StartCalendarInterval"] == {"Hour": 5, "Minute": 0}
    assert "RunAtLoad" not in p0
    assert "StartInterval" not in p0
    assert "/usr/bin/python3" not in p0["ProgramArguments"]
    assert "__BRAINLAYER_PYTHON__" not in p0["ProgramArguments"]
    assert "__PYTHON_BIN__" not in p0["ProgramArguments"]
    assert "-m" not in p0["ProgramArguments"]
    assert "brainlayer.p0_longitudinal_count" not in p0["ProgramArguments"]
    assert "__BRAINLAYER_DIR__" not in p0["ProgramArguments"]


def test_index_launchagent_runs_nightly_without_keepalive_or_run_at_load():
    index = _load("scripts/launchd/com.brainlayer.index.plist")

    assert "KeepAlive" not in index
    assert "RunAtLoad" not in index
    assert "StartInterval" not in index
    assert index["StartCalendarInterval"] == {"Hour": 3, "Minute": 15}


def test_t3_ingest_launchagent_invokes_first_class_source():
    t3 = _load("scripts/launchd/com.brainlayer.t3-ingest.plist")

    assert t3["Label"] == "com.brainlayer.t3-ingest"
    assert t3["ProgramArguments"][:3] == ["__BRAINLAYER_ENV_RUN__", "__BRAINLAYER_BIN__", "ingest-t3"]
    assert t3["StartCalendarInterval"] == {"Hour": 3, "Minute": 45}
    assert t3["RunAtLoad"] is True
    assert "KeepAlive" not in t3


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
        plist = plistlib.loads(path.read_bytes())
        if plist["Label"] in PACKAGE_IMPORT_EXEMPT_LABELS:
            continue
        _assert_uses_installed_package_not_source_path(path, plist)


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

        if plist["Label"] in ENV_RUN_EXEMPT_LABELS:
            expected_args = {
                "com.brainlayer.tier0-watchdog": ["/bin/sh", "__TIER0_WATCHDOG_SCRIPT__"],
            }
            assert args == expected_args[plist["Label"]], str(path)
            assert "BRAINLAYER_ENV_FILE" not in env, str(path)
            assert "BRAINLAYER_LAUNCHD_SERVICE" not in env, str(path)
            continue

        assert args[0] == "__BRAINLAYER_ENV_RUN__", str(path)
        assert env["BRAINLAYER_ENV_FILE"] == "__BRAINLAYER_ENV_FILE__", str(path)
        expected_service = "watch" if plist["Label"] == "com.brainlayer.throughput-watchdog" else service
        assert env["BRAINLAYER_LAUNCHD_SERVICE"] == expected_service, str(path)


def test_launchd_env_loader_exists_and_loads_safe_env_before_exec():
    loader = REPO_ROOT / "scripts/launchd/brainlayer-env-run.sh"
    content = loader.read_text(encoding="utf-8")

    assert 'ENV_FILE="${BRAINLAYER_ENV_FILE:-$HOME/.config/brainlayer/brainlayer.env}"' in content
    assert "export PATH=" in content
    assert "BRAINLAYER_SYSTEM_ENABLED" in content
    assert "BRAINLAYER_LAUNCHD_SERVICE" in content
    assert "BRAINLAYER_LAUNCHD_${service_key}_ENABLED" in content
    assert "BRAINLAYER_ENRICH_ENABLED" in content
    assert "current user or root" in content
    assert "world-writable" in content
    assert "load_simple_env_file" in content
    assert "env_file_declares_google_key" in content
    assert 'source "$ENV_FILE"' not in content
    assert 'exec "$@"' in content
    assert "GOOGLE_API_KEY" in content


def test_launchd_env_loader_does_not_evaluate_env_command_substitution(tmp_path):
    loader = REPO_ROOT / "scripts/launchd/brainlayer-env-run.sh"
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text(
        "\n".join(
            [
                'GOOGLE_API_KEY="$(sleep 5)"',
                "BRAINLAYER_SYSTEM_ENABLED=1",
                "BRAINLAYER_LAUNCHD_DRAIN_ENABLED=1",
            ]
        ),
        encoding="utf-8",
    )
    env_file.chmod(0o600)

    result = subprocess.run(
        [
            str(loader),
            "/bin/sh",
            "-c",
            'test -z "${GOOGLE_API_KEY:-}" && test "$BRAINLAYER_SYSTEM_ENABLED" = "1"',
        ],
        env={
            **os.environ,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_SERVICE": "drain",
        },
        capture_output=True,
        text=True,
        timeout=1,
        check=False,
    )

    assert result.returncode == 0


def test_launchd_env_loader_required_google_key_allows_op_backed_declaration_without_shell_eval(tmp_path):
    loader = REPO_ROOT / "scripts/launchd/brainlayer-env-run.sh"
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text('GOOGLE_API_KEY="$(sleep 5)"\n', encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(loader), "/usr/bin/true"],
        env={
            **os.environ,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_REQUIRE_GOOGLE_API_KEY": "1",
            "BRAINLAYER_SKIP_DISABLE_GATES": "1",
        },
        capture_output=True,
        text=True,
        timeout=1,
        check=False,
    )

    assert result.returncode == 0


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

    child_env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "BRAINLAYER_ENRICH_ENABLED",
            "BRAINLAYER_ENRICH_MODE",
            "BRAINLAYER_ENRICH_PROVIDER",
            "BRAINLAYER_ENRICH_BACKEND",
            "BRAINLAYER_ENRICH_RATE",
            "BRAINLAYER_ENRICH_CONCURRENCY",
            "BRAINLAYER_MAX_COMMIT_BATCH",
            "BRAINLAYER_GEMINI_SERVICE_TIER",
        }
    }

    result = subprocess.run(
        [str(harness)],
        env={
            **child_env,
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


def test_launchd_installer_wires_t3_ingest_target():
    install_source = (REPO_ROOT / "scripts/launchd/install.sh").read_text(encoding="utf-8")

    assert "./scripts/launchd/install.sh t3-ingest" in install_source
    assert "t3-ingest)" in install_source
    assert "install_plist t3-ingest" in install_source
    assert "remove_plist t3-ingest" in install_source


def test_launchd_installer_wires_throughput_watchdog_target():
    install_source = (REPO_ROOT / "scripts/launchd/install.sh").read_text(encoding="utf-8")

    assert "./scripts/launchd/install.sh throughput-watchdog" in install_source
    assert "throughput-watchdog)" in install_source
    assert "install_throughput_watchdog" in install_source
    assert "remove_plist throughput-watchdog" in install_source


def test_launchd_installer_uses_bootstrap_not_legacy_load_unload():
    install_source = (REPO_ROOT / "scripts/launchd/install.sh").read_text(encoding="utf-8")
    load_plist_body = install_source.split("load_plist() {", 1)[1].split("\nunload_plist() {", 1)[0]

    assert "launchctl enable" in load_plist_body
    assert "launchctl bootout" in load_plist_body
    assert "launchctl bootstrap" in load_plist_body
    assert "launchctl print" in load_plist_body
    assert "launchctl load" not in load_plist_body
    assert "launchctl unload" not in load_plist_body


@pytest.mark.parametrize("form", ["disabled", "true"])
@pytest.mark.parametrize("name", ["watch", "hotlane-brainbar", "enrichment"])
def test_launchd_installer_load_plist_skips_operator_disabled_label(tmp_path, name, form):
    """An operator `launchctl disable` is a standing order: load_plist must not enable/bootstrap it."""
    install_source = (REPO_ROOT / "scripts/launchd/install.sh").read_text(encoding="utf-8")
    load_plist_body = (
        "load_plist() {" + install_source.split("\nload_plist() {", 1)[1].split("\nunload_plist() {", 1)[0]
    )
    helper_name = "label_disabled_by_operator() {"
    assert helper_name in install_source, "install.sh must define label_disabled_by_operator()"
    helper_body = helper_name + install_source.split("\n" + helper_name, 1)[1].split("\n}\n", 1)[0] + "\n}\n"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$LAUNCHCTL_LOG"\n'
        '[ "$1" = "print-disabled" ] && printf "%s\\n" "$LAUNCHCTL_LISTING"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    listing = (
        "\tdisabled services = {\n"
        f'\t\t"com.brainlayer.watch" => {form}\n\t\t"com.brainlayer.hotlane-brainbar" => {form}\n'
        f'\t\t"com.brainlayer.enrichment" => {form}\n\t\t"com.brainlayer.drain" => enabled\n'
        "\t}"
    )
    fake_launchctl.chmod(0o755)
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "set -euo pipefail\n"
        f'LAUNCH_DIR="{tmp_path}"\nPYTHON_BIN=/usr/bin/true\nLOAD_PLIST_SKIPPED=0\n'
        + helper_body
        + load_plist_body
        + '\nload_plist "$1"\n',
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "LAUNCHCTL_LOG": str(launchctl_log),
        "LAUNCHCTL_LISTING": listing,
    }

    result = subprocess.run(["/bin/bash", str(harness), name], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    uid = os.getuid()
    assert (
        f"SKIP: com.brainlayer.{name} disabled by operator (launchctl enable gui/{uid}/com.brainlayer.{name} to re-arm)"
        in result.stdout
    )
    calls = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert calls == [f"print-disabled gui/{uid}"], calls


def test_launchd_primitive_reads_current_print_disabled_vocabulary():
    """macOS 14+ prints `=> disabled|enabled`; older releases print `=> true|false`. Both must be honored."""
    from types import SimpleNamespace

    from brainlayer.launchd_primitive import is_launchd_label_disabled

    listing = (
        "\tdisabled services = {\n"
        '\t\t"com.brainlayer.watch" => disabled\n'
        '\t\t"com.brainlayer.drain" => enabled\n'
        '\t\t"com.legacy.off" => true\n'
        '\t\t"com.legacy.on" => false\n'
        "\t}\n"
    )
    runner = lambda _args: SimpleNamespace(returncode=0, stdout=listing, stderr="")  # noqa: E731

    assert is_launchd_label_disabled("com.brainlayer.watch", command_runner=runner) is True
    assert is_launchd_label_disabled("com.brainlayer.drain", command_runner=runner) is False
    assert is_launchd_label_disabled("com.legacy.off", command_runner=runner) is True
    assert is_launchd_label_disabled("com.legacy.on", command_runner=runner) is False
    assert is_launchd_label_disabled("com.absent", command_runner=runner) is False


def test_launchd_installer_hotlane_skip_bypasses_runtime_verification(tmp_path):
    """install.sh hotlane-brainbar on an operator-disabled label: rc 0, no verify_hotlane_runtime, no bootout."""
    install_source = (REPO_ROOT / "scripts/launchd/install.sh").read_text(encoding="utf-8")

    def _fn(name: str) -> str:
        head = name + "() {"
        return head + install_source.split("\n" + head, 1)[1].split("\n}\n", 1)[0] + "\n}\n"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    (fake_bin / "launchctl").write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$LAUNCHCTL_LOG"\n'
        '[ "$1" = "print-disabled" ] && printf \'\\t"com.brainlayer.hotlane-brainbar" => disabled\\n\'\n'
        "exit 0\n",
        encoding="utf-8",
    )
    (fake_bin / "launchctl").chmod(0o755)
    script_dir = tmp_path / "launchd"
    script_dir.mkdir()
    (script_dir / "com.brainlayer.hotlane-brainbar.plist").write_text("<plist/>\n", encoding="utf-8")
    verify_marker = tmp_path / "verify-ran"
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "set -euo pipefail\n"
        f'SCRIPT_DIR="{script_dir}"\nLAUNCH_DIR="{tmp_path}"\nLOG_DIR="{tmp_path}"\nBRAINLAYER_LOG_DIR="{tmp_path}"\n'
        "PYTHON_BIN=/usr/bin/true\nBRAINLAYER_BIN=x\nBRAINLAYER_DIR=x\nBRAINLAYER_LAUNCHD_DIR=x\nBRAINLAYER_PYTHON=x\n"
        "BRAINLAYER_ENV_FILE=x\nBRAINLAYER_ENV_RUN=x\nHOTLANE_BRAINBAR_DST=x\nLOAD_PLIST_SKIPPED=0\n"
        "install_hotlane_brainbar_daemon() { :; }\ninstall_env_runner() { :; }\nverify_config_file() { :; }\n"
        f'verify_hotlane_runtime() {{ touch "{verify_marker}"; return 1; }}\n'
        + _fn("label_disabled_by_operator")
        + _fn("load_plist")
        + _fn("install_plist")
        + "\ninstall_plist hotlane-brainbar\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "LAUNCHCTL_LOG": str(launchctl_log)}

    result = subprocess.run(["/bin/bash", str(harness)], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "SKIP: com.brainlayer.hotlane-brainbar disabled by operator" in result.stdout
    assert not verify_marker.exists(), "verify_hotlane_runtime must not run after an operator-disable skip"
    assert launchctl_log.read_text(encoding="utf-8").splitlines() == [f"print-disabled gui/{os.getuid()}"]
