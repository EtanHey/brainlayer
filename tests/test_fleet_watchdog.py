"""Revival drills for the com.brainlayer.* fleet watchdog.

The watchdog is the reason `launchctl bootout` cannot be trusted during an upgrade: on
StartInterval 300 it re-bootstraps a booted-out label within five minutes. Only an
operator `launchctl disable` holds -- and, because `brainlayer.maintenance` pauses with a
bare bootout and no disable, so does an unexpired pause sentinel. Both signals are drilled
in both directions with a fake `launchctl` first on PATH; the sentinel is parsed by the real
`plutil`, not a stub.
"""

from __future__ import annotations

import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "launchd" / "fleet-watchdog.sh"
LABEL = "com.etanhey.brainlayer-fleet-watchdog"
PLIST_PATH = REPO_ROOT / "scripts" / "launchd" / f"{LABEL}.plist"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "launchd" / "install.sh"

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _has_bsd_date() -> bool:
    """True when `date -j -f` parses a stamp -- BSD date, not GNU."""
    try:
        probe = subprocess.run(
            ["date", "-j", "-f", "%Y-%m-%dT%H:%M:%S", "2026-01-01T00:00:00", "+%s"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


# The script reads the pause sentinel with `plutil` and BSD `date -j`, both macOS-only, so the
# sentinel drills cannot run on the ubuntu `test` matrix. The condition is the TOOLS, not
# sys.platform, so the skip branch is itself testable by manipulating PATH. Coverage does not
# vanish: the launchd (macos-15) CI job runs this file, and
# test_ci_runs_the_sentinel_drills_on_a_macos_runner -- which never skips -- enforces that.
SENTINEL_TOOLS_MISSING = shutil.which("plutil") is None or not _has_bsd_date()
needs_sentinel_tools = pytest.mark.skipif(
    SENTINEL_TOOLS_MISSING,
    reason="pause-sentinel drills need macOS `plutil` and BSD `date -j`; covered by the launchd (macos-15) CI job",
)

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
    sentinel: str | dict[str, object] | None = None,
) -> DrillResult:
    home = home or (tmp_path / "home")
    sentinel_path = home / ".local" / "share" / "brainlayer" / "pause.sentinel"
    if sentinel is not None:
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_path.write_text(
            sentinel if isinstance(sentinel, str) else json.dumps(sentinel),
            encoding="utf-8",
        )
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

    assert first.skip_state == "disabled by operator: com.brainlayer.watch\n"
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


# --- the pause sentinel -------------------------------------------------------------
# `brainlayer.maintenance` pauses services with a bare `launchctl bootout` and no
# `launchctl disable` (maintenance.py:_bootout_service), so mid-maintenance a label is
# absent-and-not-disabled: exactly the state the revival loop reverses. The sentinel is the
# signal maintenance.py:_resume_services honours, and the watchdog must honour it too.


def _sentinel(*labels: str, expires_in: timedelta = timedelta(hours=1)) -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "labels": list(labels),
        "created_at": now.isoformat(),
        "expires_at": (now + expires_in).isoformat(),
    }


@needs_sentinel_tools
def test_label_named_by_an_unexpired_pause_sentinel_is_left_down(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.enrichment", "com.brainlayer.index"),
        sentinel=_sentinel("com.brainlayer.enrichment"),
    )

    assert result.process.returncode == 0, result.process.stderr
    assert result.bootstrapped() == {"com.brainlayer.index"}
    assert result.enabled() == {"com.brainlayer.index"}
    assert "skipped com.brainlayer.enrichment (pause sentinel is active)" in result.log
    assert "re-bootstrapped com.brainlayer.enrichment" not in result.log


@needs_sentinel_tools
def test_label_not_named_by_the_sentinel_is_still_revived(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.drain",),
        sentinel=_sentinel("com.brainlayer.enrichment"),
    )

    assert result.bootstrapped() == {"com.brainlayer.drain"}


@needs_sentinel_tools
def test_every_label_in_a_multi_label_sentinel_is_left_down(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.enrichment", "com.brainlayer.drain", "com.brainlayer.watch"),
        sentinel=_sentinel("com.brainlayer.enrichment", "com.brainlayer.drain", "com.brainlayer.watch"),
    )

    assert result.bootstrapped() == set()
    for label in ("com.brainlayer.enrichment", "com.brainlayer.drain", "com.brainlayer.watch"):
        assert f"skipped {label} (pause sentinel is active)" in result.log


@needs_sentinel_tools
def test_expired_pause_sentinel_no_longer_holds_a_label_down(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.enrichment",),
        sentinel=_sentinel("com.brainlayer.enrichment", expires_in=timedelta(hours=-1)),
    )

    assert result.bootstrapped() == {"com.brainlayer.enrichment"}
    assert "re-bootstrapped com.brainlayer.enrichment" in result.log


@needs_sentinel_tools
def test_sentinel_without_an_expiry_never_goes_stale(tmp_path: Path) -> None:
    # pause.py: expires_at is None => stale is False => the pause is still active.
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.enrichment",),
        sentinel={"labels": ["com.brainlayer.enrichment"]},
    )

    assert result.bootstrapped() == set()
    assert "skipped com.brainlayer.enrichment (pause sentinel is active)" in result.log


@needs_sentinel_tools
def test_sentinel_with_an_unparseable_expiry_keeps_holding(tmp_path: Path) -> None:
    # pause.py returns stale=False for an unreadable expires_at; the pause keeps holding,
    # which is the safe direction for a reviver.
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.enrichment",),
        sentinel={"labels": ["com.brainlayer.enrichment"], "expires_at": "not-a-timestamp"},
    )

    assert result.bootstrapped() == set()
    assert "skipped com.brainlayer.enrichment (pause sentinel is active)" in result.log


@needs_sentinel_tools
def test_sentinel_expiry_offsets_and_fractional_seconds_are_understood(tmp_path: Path) -> None:
    future = (datetime.now(UTC) + timedelta(hours=1)).replace(tzinfo=None)
    for stamp in (
        f"{future.isoformat(timespec='microseconds')}+00:00",
        f"{future.isoformat(timespec='seconds')}Z",
        future.isoformat(timespec="seconds"),
    ):
        result = _run_drill(
            tmp_path,
            agent_labels=("com.brainlayer.enrichment",),
            sentinel={"labels": ["com.brainlayer.enrichment"], "expires_at": stamp},
            home=tmp_path / f"home-{abs(hash(stamp))}",
        )

        assert result.bootstrapped() == set(), stamp


@needs_sentinel_tools
@pytest.mark.parametrize(
    "payload",
    ['{"labels": ["com.brainlayer.enrichment"', "", "not json at all", "[]"],
    ids=["truncated", "empty", "garbage", "not-an-object"],
)
def test_unparseable_sentinel_fails_closed(tmp_path: Path, payload: str) -> None:
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.enrichment", "com.brainlayer.index"),
        sentinel=payload,
    )

    assert result.process.returncode == 1
    assert result.bootstrapped() == set()
    assert result.enabled() == set()
    assert "refusing to revive anything" in result.log


@needs_sentinel_tools
def test_no_sentinel_at_all_revives_normally(tmp_path: Path) -> None:
    result = _run_drill(tmp_path, agent_labels=("com.brainlayer.enrichment",))

    assert result.process.returncode == 0
    assert result.bootstrapped() == {"com.brainlayer.enrichment"}


@needs_sentinel_tools
def test_operator_disable_still_wins_over_a_sentinel_that_omits_the_label(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.watch",),
        disabled_labels=("com.brainlayer.watch",),
        sentinel=_sentinel("com.brainlayer.enrichment"),
    )

    assert result.bootstrapped() == set()
    assert "skipped com.brainlayer.watch (disabled by operator)" in result.log


@needs_sentinel_tools
def test_a_standing_pause_logs_its_skip_once(tmp_path: Path) -> None:
    home = tmp_path / "home"
    first = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.enrichment",),
        sentinel=_sentinel("com.brainlayer.enrichment"),
        home=home,
    )
    second = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.enrichment",),
        sentinel=_sentinel("com.brainlayer.enrichment"),
        home=home,
    )

    assert first.skip_state == "pause sentinel is active: com.brainlayer.enrichment\n"
    assert first.log.count("skipped com.brainlayer.enrichment") == 1
    assert second.log.count("skipped com.brainlayer.enrichment") == 1


@needs_sentinel_tools
def test_a_label_that_moves_from_paused_to_disabled_is_logged_again(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.enrichment",),
        sentinel=_sentinel("com.brainlayer.enrichment"),
        home=home,
    )
    disabled = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.enrichment",),
        disabled_labels=("com.brainlayer.enrichment",),
        home=home,
    )

    assert "skipped com.brainlayer.enrichment (disabled by operator)" in disabled.log


@needs_sentinel_tools
def test_sentinel_contract_matches_the_python_reader(tmp_path: Path) -> None:
    """The script and src/brainlayer/pause.py must agree on path, key names and expiry."""
    from brainlayer.pause import DEFAULT_PAUSE_SENTINEL_PATH, pause_applies_to_label, pause_sentinel_state

    assert DEFAULT_PAUSE_SENTINEL_PATH == Path("~/.local/share/brainlayer/pause.sentinel").expanduser()
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"$HOME/.local/share/brainlayer/pause.sentinel"' in script

    payload = _sentinel("com.brainlayer.enrichment")
    path = tmp_path / "pause.sentinel"
    path.write_text(json.dumps(payload), encoding="utf-8")
    parsed, active, stale = pause_sentinel_state(path, datetime.now(UTC))
    assert active and not stale
    assert pause_applies_to_label(parsed, "com.brainlayer.enrichment")
    assert not pause_applies_to_label(parsed, "com.brainlayer.index")

    # Same payload, same verdict from the shell script.
    result = _run_drill(
        tmp_path,
        agent_labels=("com.brainlayer.enrichment", "com.brainlayer.index"),
        sentinel=payload,
    )
    assert result.bootstrapped() == {"com.brainlayer.index"}


# --- the drills above must not be skipped everywhere -------------------------------
# A skip that makes CI green while testing nothing is the false-green class this whole PR
# exists to close, so neither lock below is allowed to skip.


def test_sentinel_tools_are_present_on_darwin() -> None:
    """On macOS the drills must RUN. A silent skip here would hide the sentinel logic."""
    if sys.platform != "darwin":
        pytest.skip("Darwin-only assertion; the CI-coverage lock below is the check that never skips")

    assert shutil.which("plutil") is not None, "macOS without plutil: the sentinel drills would silently skip"
    assert _has_bsd_date(), "macOS without BSD `date -j`: the sentinel drills would silently skip"
    assert not SENTINEL_TOOLS_MISSING


def test_ci_runs_the_sentinel_drills_on_a_macos_runner() -> None:
    """The ubuntu matrix skips the sentinel drills, so some macOS job must run this file.

    This test runs on every runner and never skips: delete the macOS job and it goes red.
    """
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = document["jobs"]

    macos_jobs = {name: job for name, job in jobs.items() if str(job.get("runs-on", "")).startswith("macos")}
    assert macos_jobs, "ci.yml has no macOS job, so the pause-sentinel drills run nowhere"

    this_file = "tests/test_fleet_watchdog.py"
    covering = [
        name
        for name, job in macos_jobs.items()
        for step in job.get("steps", [])
        if this_file in str(step.get("run", ""))
    ]
    assert covering, f"no macOS job in ci.yml runs {this_file}; the sentinel drills would be skipped everywhere"

    job = macos_jobs[covering[0]]
    # The drills import brainlayer.pause for the cross-checked contract test.
    assert any(
        "PYTHONPATH" in str(step.get("env", {})) or "pip install" in str(step.get("run", ""))
        for step in job.get("steps", [])
    ), f"{covering[0]} runs the drills but installs nothing to import brainlayer.pause from"


def test_ubuntu_matrix_still_collects_this_file() -> None:
    """The non-sentinel drills are portable and must keep running on the ubuntu matrix."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = document["jobs"]["test"]["steps"]

    assert str(document["jobs"]["test"]["runs-on"]).startswith("ubuntu")
    unit = next(step for step in steps if step.get("name") == "Unit tests")
    # `pytest tests/` collects this file; only the sentinel drills opt out, and they do it by
    # tool probe rather than by being excluded here.
    assert "tests/" in unit["run"]
    assert "test_fleet_watchdog" not in unit["run"]


def test_script_header_documents_the_sentinel_and_the_bare_bootout() -> None:
    header = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "pause.sentinel" in header
    assert "BARE bootout" in header
    assert "_resume_services" in header
    assert "fail CLOSED" in header


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
