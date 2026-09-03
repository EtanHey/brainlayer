"""Revival drills for the com.brainlayer.* fleet watchdog.

The watchdog is the reason `launchctl bootout` cannot be trusted during an upgrade: on
StartInterval 300 it re-bootstraps a booted-out label within five minutes. Only an
operator `launchctl disable` holds. Both directions are drilled here with a fake
`launchctl` first on PATH.
"""

from __future__ import annotations

import os
import plistlib
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "launchd" / "fleet-watchdog.sh"
LABEL = "com.etanhey.brainlayer-fleet-watchdog"
PLIST_PATH = REPO_ROOT / "scripts" / "launchd" / f"{LABEL}.plist"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "launchd" / "install.sh"

FAKE_LAUNCHCTL = """#!/bin/sh
printf '%s\\n' "$*" >> "$FAKE_LAUNCHCTL_LOG"
if [ "$1" = "print-disabled" ]; then
    if [ "${FAKE_PRINT_DISABLED_RC:-0}" != "0" ]; then
        exit "$FAKE_PRINT_DISABLED_RC"
    fi
    cat "$FAKE_DISABLED_LISTING"
    exit 0
fi
if [ "$1" = "print" ]; then
    label="${2##*/}"
    case ":$FAKE_LOADED_LABELS:" in
        *":$label:"*) exit 0 ;;
    esac
    exit 113
fi
if [ "$1" = "bootstrap" ]; then
    exit "${FAKE_BOOTSTRAP_RC:-0}"
fi
exit 0
"""


@dataclass(frozen=True)
class DrillResult:
    process: subprocess.CompletedProcess[str]
    commands: list[str]
    log: str
    skip_state: str

    def bootstrapped(self) -> set[str]:
        return {Path(command.split(" ")[-1]).stem for command in self.commands if command.startswith("bootstrap ")}

    def enabled(self) -> set[str]:
        return {command.split("/")[-1] for command in self.commands if command.startswith("enable ")}


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run_drill(
    tmp_path: Path,
    *,
    agent_labels: tuple[str, ...] = (),
    disabled_labels: tuple[str, ...] = (),
    loaded_labels: tuple[str, ...] = (),
    disabled_form: str = "disabled",
    print_disabled_rc: int = 0,
    bootstrap_rc: int = 0,
    home: Path | None = None,
) -> DrillResult:
    home = home or (tmp_path / "home")
    launch_dir = home / "Library" / "LaunchAgents"
    launch_dir.mkdir(parents=True, exist_ok=True)
    for label in agent_labels:
        (launch_dir / f"{label}.plist").write_bytes(
            plistlib.dumps({"Label": label, "ProgramArguments": ["/usr/bin/true"]})
        )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    _write_executable(fake_bin / "launchctl", FAKE_LAUNCHCTL)

    listing_path = tmp_path / "print-disabled.txt"
    listing_path.write_text(
        "".join(f'\t"{label}" => {disabled_form}\n' for label in disabled_labels),
        encoding="utf-8",
    )
    command_log = tmp_path / "launchctl.log"
    command_log.unlink(missing_ok=True)

    process = subprocess.run(
        ["/bin/sh", str(SCRIPT_PATH)],
        env={
            **os.environ,
            "HOME": str(home),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_LAUNCHCTL_LOG": str(command_log),
            "FAKE_DISABLED_LISTING": str(listing_path),
            "FAKE_LOADED_LABELS": ":".join(loaded_labels),
            "FAKE_PRINT_DISABLED_RC": str(print_disabled_rc),
            "FAKE_BOOTSTRAP_RC": str(bootstrap_rc),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    log_path = home / "Library" / "Logs" / "brainlayer" / "fleet-watchdog.log"
    skip_state = home / "Library" / "Logs" / "brainlayer" / "fleet-watchdog-skipped"
    return DrillResult(
        process=process,
        commands=command_log.read_text(encoding="utf-8").splitlines() if command_log.exists() else [],
        log=log_path.read_text(encoding="utf-8") if log_path.exists() else "",
        skip_state=skip_state.read_text(encoding="utf-8") if skip_state.exists() else "",
    )


def test_enabled_but_absent_label_is_bootstrapped(tmp_path: Path) -> None:
    result = _run_drill(tmp_path, agent_labels=("com.brainlayer.watch",))

    assert result.process.returncode == 0, result.process.stderr
    assert result.bootstrapped() == {"com.brainlayer.watch"}
    assert result.enabled() == {"com.brainlayer.watch"}
    assert "re-bootstrapped com.brainlayer.watch" in result.log


def test_operator_disabled_label_is_skipped_never_enabled_or_bootstrapped(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.watch", "com.brainlayer.drain"),
        disabled_labels=("com.brainlayer.watch",),
    )

    assert result.process.returncode == 0, result.process.stderr
    assert result.bootstrapped() == {"com.brainlayer.drain"}
    assert result.enabled() == {"com.brainlayer.drain"}
    assert "skipped com.brainlayer.watch (disabled by operator)" in result.log
    assert "re-bootstrapped com.brainlayer.watch" not in result.log
    assert "re-bootstrapped com.brainlayer.drain" in result.log


def test_disabled_form_true_is_honoured_like_disabled(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.watch",),
        disabled_labels=("com.brainlayer.watch",),
        disabled_form="true",
    )

    assert result.bootstrapped() == set()
    assert "skipped com.brainlayer.watch (disabled by operator)" in result.log


def test_prefix_sibling_of_a_disabled_label_is_still_revived(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.watch", "com.brainlayer.watch-extra"),
        disabled_labels=("com.brainlayer.watch",),
    )

    assert result.bootstrapped() == {"com.brainlayer.watch-extra"}


def test_loaded_label_is_left_alone(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.watch",),
        loaded_labels=("com.brainlayer.watch",),
    )

    assert result.bootstrapped() == set()
    assert result.enabled() == set()
    assert result.log == ""


def test_unreadable_disabled_state_fails_closed(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.watch",),
        print_disabled_rc=1,
    )

    assert result.process.returncode == 1
    assert result.bootstrapped() == set()
    assert result.enabled() == set()
    assert "refusing to revive anything" in result.log


def test_failed_bootstrap_is_logged_as_a_failure(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.watch",),
        bootstrap_rc=1,
    )

    assert "FAILED to re-bootstrap com.brainlayer.watch" in result.log


def test_standing_disable_does_not_relog_every_run(tmp_path: Path) -> None:
    home = tmp_path / "home"
    first = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.watch",),
        disabled_labels=("com.brainlayer.watch",),
        home=home,
    )
    second = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.watch",),
        disabled_labels=("com.brainlayer.watch",),
        home=home,
    )

    assert first.skip_state == "com.brainlayer.watch\n"
    assert first.log.count("skipped com.brainlayer.watch") == 1
    assert second.log.count("skipped com.brainlayer.watch") == 1
    assert second.bootstrapped() == set()


def test_re_disable_after_a_clean_run_logs_again(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.watch",),
        disabled_labels=("com.brainlayer.watch",),
        home=home,
    )
    cleared = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.watch",),
        loaded_labels=("com.brainlayer.watch",),
        home=home,
    )
    re_disabled = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.watch",),
        disabled_labels=("com.brainlayer.watch",),
        home=home,
    )

    assert cleared.skip_state == ""
    assert re_disabled.log.count("skipped com.brainlayer.watch") == 2


def test_watchdog_ignores_plists_outside_the_brainlayer_namespace(tmp_path: Path) -> None:
    result = _run_drill(tmp_path, agent_labels=("com.etanhey.brainlayer-fleet-watchdog", "com.other.thing"))

    assert result.bootstrapped() == set()


def test_script_header_states_that_bootout_does_not_hold() -> None:
    header = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "`launchctl bootout` DOES NOT HOLD" in header
    assert "StartInterval 300" in header
    assert "print-disabled" in header
    assert "install.sh fleet-watchdog-quiesce" in header
    assert "com.brainlayer.*, on purpose" in header


def test_plist_keeps_the_etanhey_namespace_and_templated_paths() -> None:
    plist = plistlib.loads(PLIST_PATH.read_bytes())

    assert plist["Label"] == LABEL
    assert plist["StartInterval"] == 300
    assert plist["ProgramArguments"] == ["/bin/sh", "__FLEET_WATCHDOG_SCRIPT__"]
    assert plist["StandardOutPath"].startswith("__HOME__/Library/Logs/brainlayer/")
    assert plist["StandardErrorPath"].startswith("__HOME__/Library/Logs/brainlayer/")
    assert plist["EnvironmentVariables"]["HOME"] == "__HOME__"
    assert plist["ProcessType"] == "Background"
    assert plist["ExitTimeOut"] == 15
    assert plist["SoftResourceLimits"]["NumberOfFiles"] >= 4096
    assert "/Users/happycampr" not in PLIST_PATH.read_text(encoding="utf-8")


def test_installer_documents_every_fleet_watchdog_action() -> None:
    installer = INSTALL_SCRIPT.read_text(encoding="utf-8")

    for action in ("fleet-watchdog", "fleet-watchdog-quiesce", "fleet-watchdog-resume"):
        assert f"./scripts/launchd/install.sh {action}" in installer
        assert f"|{action}|" in installer
    # `all` installs it and `remove` tears it down.
    assert "if ! install_fleet_watchdog; then" in installer
    assert "remove_fleet_watchdog 2>/dev/null || true" in installer


# A `launchctl` faithful enough to prove the quiesce actually holds: `disable` and
# `enable` mutate the print-disabled listing the way the real one does, and `print`
# only answers for a label that has been bootstrapped.
FAKE_INSTALLER_LAUNCHCTL = """#!/bin/sh
printf '%s\\n' "$*" >> "$FAKE_LAUNCHCTL_LOG"
case "$1" in
    print-disabled)
        cat "$FAKE_DISABLED_LISTING"
        exit 0
        ;;
    disable)
        if [ "${FAKE_DISABLE_IS_NOOP:-0}" = "0" ]; then
            printf '\\t"%s" => disabled\\n' "${2##*/}" >> "$FAKE_DISABLED_LISTING"
        fi
        exit "${FAKE_DISABLE_RC:-0}"
        ;;
    enable)
        grep -Fv "\\"${2##*/}\\" =>" "$FAKE_DISABLED_LISTING" > "$FAKE_DISABLED_LISTING.tmp" || true
        mv "$FAKE_DISABLED_LISTING.tmp" "$FAKE_DISABLED_LISTING"
        exit 0
        ;;
    print)
        if grep -Fq "${2##*/}.plist" "$FAKE_LAUNCHCTL_LOG"; then
            exit 0
        fi
        exit 113
        ;;
esac
exit 0
"""


@dataclass(frozen=True)
class InstallerResult:
    process: subprocess.CompletedProcess[str]
    commands: list[str]
    disabled_listing: str


class InstallerHarness:
    def __init__(self, tmp_path: Path) -> None:
        self.home = tmp_path / "installer-home"
        self.home.mkdir()
        self.fake_bin = tmp_path / "installer-bin"
        self.fake_bin.mkdir()
        _write_executable(self.fake_bin / "launchctl", FAKE_INSTALLER_LAUNCHCTL)
        self.brainlayer_bin = tmp_path / "brainlayer"
        _write_executable(self.brainlayer_bin, "#!/bin/sh\nexit 0\n")
        self.command_log = tmp_path / "installer-launchctl.log"
        self.disabled_listing = tmp_path / "installer-disabled.txt"
        self.disabled_listing.write_text("", encoding="utf-8")

    @property
    def plist_dst(self) -> Path:
        return self.home / "Library" / "LaunchAgents" / f"{LABEL}.plist"

    @property
    def script_dst(self) -> Path:
        return self.home / ".local" / "lib" / "brainlayer" / "fleet-watchdog.sh"

    def run(self, action: str, *, disable_is_noop: bool = False) -> InstallerResult:
        self.command_log.write_text("", encoding="utf-8")
        process = subprocess.run(
            [str(INSTALL_SCRIPT), action],
            env={
                **os.environ,
                "HOME": str(self.home),
                "PATH": f"{self.fake_bin}:{os.environ['PATH']}",
                "BRAINLAYER_BIN": str(self.brainlayer_bin),
                "FAKE_LAUNCHCTL_LOG": str(self.command_log),
                "FAKE_DISABLED_LISTING": str(self.disabled_listing),
                "FAKE_DISABLE_IS_NOOP": "1" if disable_is_noop else "0",
            },
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return InstallerResult(
            process=process,
            commands=self.command_log.read_text(encoding="utf-8").splitlines(),
            disabled_listing=self.disabled_listing.read_text(encoding="utf-8"),
        )


def test_installer_renders_the_plist_with_no_authoring_host_paths(tmp_path: Path) -> None:
    harness = InstallerHarness(tmp_path)

    result = harness.run("fleet-watchdog")

    assert result.process.returncode == 0, result.process.stdout + result.process.stderr
    rendered = plistlib.loads(harness.plist_dst.read_bytes())
    assert rendered["Label"] == LABEL
    assert rendered["ProgramArguments"] == ["/bin/sh", str(harness.script_dst)]
    assert rendered["StandardErrorPath"] == str(
        harness.home / "Library" / "Logs" / "brainlayer" / "fleet-watchdog.err.log"
    )
    assert "/Users/happycampr" not in harness.plist_dst.read_text(encoding="utf-8")
    assert harness.script_dst.exists()
    assert f"bootstrap gui/{os.getuid()} {harness.plist_dst}" in result.commands
    # The installer must say out loud that bootout does not hold.
    assert "NOT hold" in result.process.stdout
    assert "fleet-watchdog-quiesce" in result.process.stdout


def test_quiesce_disables_the_label_and_a_later_install_refuses_to_revive_it(tmp_path: Path) -> None:
    harness = InstallerHarness(tmp_path)
    harness.run("fleet-watchdog")

    quiesced = harness.run("fleet-watchdog-quiesce")

    assert quiesced.process.returncode == 0, quiesced.process.stdout + quiesced.process.stderr
    assert f"disable gui/{os.getuid()}/{LABEL}" in quiesced.commands
    assert f'"{LABEL}" => disabled' in quiesced.disabled_listing
    assert "bootout of com.brainlayer.* labels now holds" in quiesced.process.stdout

    reinstall = harness.run("fleet-watchdog")

    assert reinstall.process.returncode == 0, reinstall.process.stdout
    assert f"SKIP: {LABEL} disabled by operator" in reinstall.process.stdout
    assert not [command for command in reinstall.commands if command.startswith("bootstrap ")]


def test_quiesce_fails_loudly_when_the_disable_does_not_take(tmp_path: Path) -> None:
    harness = InstallerHarness(tmp_path)
    harness.run("fleet-watchdog")

    result = harness.run("fleet-watchdog-quiesce", disable_is_noop=True)

    assert result.process.returncode != 0
    assert "is not reported disabled after launchctl disable" in result.process.stderr


def test_resume_re_enables_and_bootstraps_after_a_quiesce(tmp_path: Path) -> None:
    harness = InstallerHarness(tmp_path)
    harness.run("fleet-watchdog")
    harness.run("fleet-watchdog-quiesce")

    result = harness.run("fleet-watchdog-resume")

    assert result.process.returncode == 0, result.process.stdout + result.process.stderr
    assert result.disabled_listing == ""
    assert f"enable gui/{os.getuid()}/{LABEL}" in result.commands
    assert f"bootstrap gui/{os.getuid()} {harness.plist_dst}" in result.commands


def test_resume_without_an_installed_plist_is_an_error(tmp_path: Path) -> None:
    harness = InstallerHarness(tmp_path)

    result = harness.run("fleet-watchdog-resume")

    assert result.process.returncode != 0
    assert "run" in result.process.stderr and "fleet-watchdog" in result.process.stderr


def test_remove_boots_out_and_deletes_both_artifacts(tmp_path: Path) -> None:
    harness = InstallerHarness(tmp_path)
    harness.run("fleet-watchdog")
    assert harness.plist_dst.exists()

    result = harness.run("remove")

    assert result.process.returncode == 0, result.process.stdout + result.process.stderr
    assert f"bootout gui/{os.getuid()}/{LABEL}" in result.commands
    assert not harness.plist_dst.exists()
    assert not harness.script_dst.exists()
