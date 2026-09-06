"""Synthetic silent-death drills for the Tier-0 health-check watchdog."""

from __future__ import annotations

import os
import plistlib
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "tier0-watchdog.sh"
PLIST_PATH = REPO_ROOT / "scripts" / "launchd" / "com.brainlayer.tier0-watchdog.plist"

NOW_EPOCH = 10_000
STALE_SECONDS = 1_200
MISSED_RUN_GRACE_SECONDS = 600
DOMAIN = "gui/501"
LABEL = "com.example.brainlayer-health-check"
NOTIFY_ENDPOINT = "http://localhost:3847/notify"


@dataclass(frozen=True)
class DrillResult:
    process: subprocess.CompletedProcess[str]
    events: list[str]
    tier0_log: str
    alert_state: str
    run_state: str = ""


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _event_index(events: list[str], prefix: str) -> int:
    return next(index for index, event in enumerate(events) if event.startswith(prefix))


def _run_drill(
    tmp_path: Path,
    *,
    label_loaded: bool,
    state_mtime: int | None,
    curl_hangs: bool = False,
    osascript_hangs: bool = False,
    use_fake_wait_sleep: bool = False,
    last_alert_epoch: int | None = None,
    last_alert_reason: str = "",
    repeat_alert_seconds: int = 1_800,
    last_run_epoch: int | None = None,
    alert_timeout_seconds: int = 3,
    state_contents: str = "{}\n",
) -> DrillResult:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    events_path = tmp_path / "events.log"
    state_path = tmp_path / "health-check-state.json"
    health_plist_path = tmp_path / "com.example.brainlayer-health-check.plist"
    tier0_log_path = tmp_path / "logs" / "tier0-watchdog.log"
    alert_state_path = tmp_path / "tier0-watchdog-alert-state"
    run_state_path = tmp_path / "tier0-watchdog-last-run"
    health_plist_path.write_text("fixture\n", encoding="utf-8")

    if last_alert_epoch is not None:
        alert_state_path.write_text(
            f"{last_alert_epoch}\t{last_alert_reason}\n",
            encoding="utf-8",
        )

    if last_run_epoch is not None:
        run_state_path.write_text(f"{last_run_epoch}\n", encoding="utf-8")

    if state_mtime is not None:
        state_path.write_text(state_contents, encoding="utf-8")

    _write_executable(
        fake_bin / "launchctl",
        "\n".join(
            [
                "#!/bin/sh",
                'printf "launchctl:%s\\n" "$*" >> "$TIER0_DRILL_EVENTS"',
                'if [ "$1" = "print" ]; then exit "$FAKE_LAUNCHCTL_PRINT_EXIT"; fi',
                "exit 0",
                "",
            ]
        ),
    )
    _write_executable(
        fake_bin / "stat",
        "\n".join(
            [
                "#!/bin/sh",
                'printf "stat:%s\\n" "$*" >> "$TIER0_DRILL_EVENTS"',
                'printf "%s\\n" "$FAKE_STATE_MTIME"',
                "",
            ]
        ),
    )
    osascript_lines = [
        "#!/bin/sh",
        'printf "osascript:%s\\n" "$*" >> "$TIER0_DRILL_EVENTS"',
    ]
    if osascript_hangs:
        osascript_lines.append("exec /bin/sleep 30")
    else:
        osascript_lines.append("exit 0")
    osascript_lines.append("")
    _write_executable(fake_bin / "osascript", "\n".join(osascript_lines))
    curl_lines = [
        "#!/bin/sh",
        'printf "curl:%s\\n" "$*" >> "$TIER0_DRILL_EVENTS"',
    ]
    if curl_hangs:
        curl_lines.append("exec /bin/sleep 30")
    else:
        curl_lines.append("exit 0")
    curl_lines.append("")
    _write_executable(fake_bin / "curl", "\n".join(curl_lines))
    if use_fake_wait_sleep:
        _write_executable(
            fake_bin / "wait-sleep",
            "\n".join(
                [
                    "#!/bin/sh",
                    'printf "wait-sleep:%s\\n" "$*" >> "$TIER0_DRILL_EVENTS"',
                    "exit 0",
                    "",
                ]
            ),
        )

    env = {
        **os.environ,
        "FAKE_LAUNCHCTL_PRINT_EXIT": "0" if label_loaded else "113",
        "FAKE_STATE_MTIME": str(state_mtime or 0),
        "TIER0_ALERT_TIMEOUT_SECONDS": str(alert_timeout_seconds),
        "TIER0_ALERT_STATE_PATH": str(alert_state_path),
        "TIER0_CURL": str(fake_bin / "curl"),
        "TIER0_DOMAIN": DOMAIN,
        "TIER0_DRILL_EVENTS": str(events_path),
        "TIER0_HEALTH_PLIST_PATH": str(health_plist_path),
        "TIER0_LABEL": LABEL,
        "TIER0_LAUNCHCTL": str(fake_bin / "launchctl"),
        "TIER0_LOG_PATH": str(tier0_log_path),
        "TIER0_NOTIFY_ENDPOINT": NOTIFY_ENDPOINT,
        "TIER0_NOTIFY_TIMEOUT_SECONDS": "1",
        "TIER0_REPEAT_ALERT_SECONDS": str(repeat_alert_seconds),
        "TIER0_RUN_STATE_PATH": str(run_state_path),
        "TIER0_MISSED_RUN_GRACE_SECONDS": str(MISSED_RUN_GRACE_SECONDS),
        "TIER0_NOW_EPOCH": str(NOW_EPOCH),
        "TIER0_OSASCRIPT": str(fake_bin / "osascript"),
        "TIER0_SLEEP": str(fake_bin / "wait-sleep") if use_fake_wait_sleep else "/bin/sleep",
        "TIER0_STALE_SECONDS": str(STALE_SECONDS),
        "TIER0_STATE_PATH": str(state_path),
        "TIER0_STAT": str(fake_bin / "stat"),
    }
    process = subprocess.run(
        ["/bin/sh", str(SCRIPT_PATH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=6,
        check=False,
    )
    events = events_path.read_text(encoding="utf-8").splitlines() if events_path.exists() else []
    tier0_log = tier0_log_path.read_text(encoding="utf-8") if tier0_log_path.exists() else ""
    alert_state = alert_state_path.read_text(encoding="utf-8") if alert_state_path.exists() else ""
    run_state = run_state_path.read_text(encoding="utf-8") if run_state_path.exists() else ""
    return DrillResult(
        process=process,
        events=events,
        tier0_log=tier0_log,
        alert_state=alert_state,
        run_state=run_state,
    )


def _assert_alert_contract(result: DrillResult) -> None:
    osascript_index = _event_index(result.events, "osascript:")
    curl_index = _event_index(result.events, "curl:")
    kickstart_index = _event_index(result.events, f"launchctl:kickstart -k {DOMAIN}/{LABEL}")

    assert osascript_index < kickstart_index
    assert curl_index < kickstart_index
    curl_event = result.events[curl_index]
    assert "-fsS" in curl_event
    assert "--max-time 1" in curl_event
    assert f"-X POST {NOTIFY_ENDPOINT}" in curl_event
    assert "-H Content-Type: application/json" in curl_event
    assert '"title":"BrainLayer Tier-0 alert"' in curl_event
    assert '"source":"alerts"' in curl_event


def test_d1_unloaded_label_alerts_before_bootstrap_and_kickstart(tmp_path: Path) -> None:
    result = _run_drill(tmp_path, label_loaded=False, state_mtime=NOW_EPOCH - 60)

    assert result.process.returncode == 1, result.process.stdout + result.process.stderr
    _assert_alert_contract(result)
    bootstrap_index = _event_index(
        result.events,
        f"launchctl:bootstrap {DOMAIN} {tmp_path / 'com.example.brainlayer-health-check.plist'}",
    )
    kickstart_index = _event_index(result.events, f"launchctl:kickstart -k {DOMAIN}/{LABEL}")
    assert _event_index(result.events, "osascript:") < bootstrap_index < kickstart_index
    assert _event_index(result.events, "curl:") < bootstrap_index
    assert "label_unloaded" in result.tier0_log


def test_d2_stale_state_alerts_before_direct_kickstart(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        label_loaded=True,
        state_mtime=NOW_EPOCH - STALE_SECONDS - 1,
    )

    assert result.process.returncode == 1, result.process.stdout + result.process.stderr
    _assert_alert_contract(result)
    assert not any(event.startswith("launchctl:bootstrap ") for event in result.events)
    assert f"state_stale age={STALE_SECONDS + 1}s threshold={STALE_SECONDS}s" in result.tier0_log
    assert result.alert_state == f"{NOW_EPOCH}\tstate_stale\n"


def test_repeat_stale_alert_is_suppressed_during_cooldown_but_recovery_still_runs(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        label_loaded=True,
        state_mtime=NOW_EPOCH - STALE_SECONDS - 1,
        last_alert_epoch=NOW_EPOCH - 300,
        last_alert_reason="state_stale",
    )

    assert result.process.returncode == 1
    assert not any(event.startswith("osascript:") for event in result.events)
    assert not any(event.startswith("curl:") for event in result.events)
    assert result.tier0_log == ""
    assert any(event == f"launchctl:kickstart -k {DOMAIN}/{LABEL}" for event in result.events)
    assert result.alert_state == f"{NOW_EPOCH - 300}\tstate_stale\n"


def test_d3_hanging_notify_endpoint_cannot_suppress_local_alert_or_heal(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        label_loaded=True,
        state_mtime=NOW_EPOCH - STALE_SECONDS - 1,
        curl_hangs=True,
        alert_timeout_seconds=1,
    )

    assert result.process.returncode == 1, result.process.stdout + result.process.stderr
    _assert_alert_contract(result)
    assert "state_stale" in result.tier0_log
    assert any(event.startswith("osascript:") for event in result.events)
    assert any(event == f"launchctl:kickstart -k {DOMAIN}/{LABEL}" for event in result.events)


def test_d4_loaded_label_and_fresh_state_do_nothing(tmp_path: Path) -> None:
    result = _run_drill(tmp_path, label_loaded=True, state_mtime=NOW_EPOCH - 60)

    assert result.process.returncode == 0, result.process.stdout + result.process.stderr
    assert result.events == [
        f"launchctl:print {DOMAIN}/{LABEL}",
        f"stat:-f %m {tmp_path / 'health-check-state.json'}",
    ]
    assert result.tier0_log == ""


def test_fresh_state_resets_prior_incident_alert_cooldown(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        label_loaded=True,
        state_mtime=NOW_EPOCH - 60,
        last_alert_epoch=NOW_EPOCH - 300,
        last_alert_reason="state_stale",
    )

    assert result.process.returncode == 0
    assert result.alert_state == "0\tok\n"


def test_missing_state_alerts_and_kickstarts_without_bootstrap(tmp_path: Path) -> None:
    result = _run_drill(tmp_path, label_loaded=True, state_mtime=None)

    assert result.process.returncode == 1, result.process.stdout + result.process.stderr
    _assert_alert_contract(result)
    assert not any(event.startswith("stat:") for event in result.events)
    assert not any(event.startswith("launchctl:bootstrap ") for event in result.events)
    assert "state_missing" in result.tier0_log


def test_future_state_mtime_alerts_and_kickstarts(tmp_path: Path) -> None:
    result = _run_drill(tmp_path, label_loaded=True, state_mtime=NOW_EPOCH + 60)

    assert result.process.returncode == 1, result.process.stdout + result.process.stderr
    _assert_alert_contract(result)
    assert not any(event.startswith("launchctl:bootstrap ") for event in result.events)
    assert "state_mtime_future offset=60s" in result.tier0_log


def test_fresh_slow_check_state_alerts_and_kickstarts(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        label_loaded=True,
        state_mtime=NOW_EPOCH - 60,
        state_contents='{"slow_check": true, "slow_check_stage": "missing_embeddings"}\n',
    )

    assert result.process.returncode == 1, result.process.stdout + result.process.stderr
    _assert_alert_contract(result)
    assert not any(event.startswith("launchctl:bootstrap ") for event in result.events)
    assert "state_slow_check" in result.tier0_log
    assert result.alert_state == f"{NOW_EPOCH}\tstate_slow_check\n"


def test_alert_fanout_uses_one_shared_deadline(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        label_loaded=True,
        state_mtime=NOW_EPOCH - STALE_SECONDS - 1,
        curl_hangs=True,
        osascript_hangs=True,
        use_fake_wait_sleep=True,
        alert_timeout_seconds=1,
    )

    assert result.process.returncode == 1, result.process.stdout + result.process.stderr
    assert sum(event.startswith("wait-sleep:") for event in result.events) == 1
    assert any(event == f"launchctl:kickstart -k {DOMAIN}/{LABEL}" for event in result.events)


def test_tier0_launchagent_uses_bin_sh_without_python_wrapper() -> None:
    plist = plistlib.loads(PLIST_PATH.read_bytes())

    assert plist["Label"] == "com.brainlayer.tier0-watchdog"
    assert plist["ProgramArguments"] == ["/bin/sh", "__TIER0_WATCHDOG_SCRIPT__"]
    assert plist["StartInterval"] == 300
    assert plist["RunAtLoad"] is True
    assert plist["EnvironmentVariables"]["TIER0_STALE_SECONDS"] == "900"
    assert plist["EnvironmentVariables"]["TIER0_REPEAT_ALERT_SECONDS"] == "1800"
    assert plist["EnvironmentVariables"]["TIER0_ALERT_STATE_PATH"] == (
        "__HOME__/.local/share/brainlayer/tier0-watchdog-alert-state"
    )
    args = " ".join(plist["ProgramArguments"])
    assert "ENV_RUN" not in args
    assert "PYTHON" not in args.upper()


# --- sleep-blind staleness ----------------------------------------------------
# Measured on the M1 Pro, 2026-09-06: 13 notifications in one night, every one
# reason=state_stale, every age (2045-3635s) equal to the sleep span that preceded
# it. launchd runs no StartInterval job while the system sleeps, so neither this
# watchdog nor the health-check it guards gets its turn -- but state_stale measures
# wall-clock age and cannot tell a sleeping Mac from a dead job. The watchdog's own
# previous run dates the outage.
SLEPT_SECONDS = 2_364  # the first measured incident: alert 00:38:52, age=2364s


def test_stale_state_alert_is_withheld_when_the_watchdog_itself_missed_runs(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        label_loaded=True,
        state_mtime=NOW_EPOCH - SLEPT_SECONDS,
        last_run_epoch=NOW_EPOCH - SLEPT_SECONDS,
    )

    assert result.process.returncode == 0, result.process.stdout + result.process.stderr
    assert not any(event.startswith("osascript:") for event in result.events)
    assert not any(event.startswith("curl:") for event in result.events)
    assert f"state_stale_withheld_after_missed_runs gap={SLEPT_SECONDS}s" in result.tier0_log
    # Recovery still runs: on wake this refreshes the state file now rather than
    # waiting out the health-check's own interval.
    assert any(event == f"launchctl:kickstart -k {DOMAIN}/{LABEL}" for event in result.events)


def test_withheld_stale_alert_leaves_a_real_incident_cooldown_untouched(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        label_loaded=True,
        state_mtime=NOW_EPOCH - SLEPT_SECONDS,
        last_run_epoch=NOW_EPOCH - SLEPT_SECONDS,
        last_alert_epoch=NOW_EPOCH - 300,
        last_alert_reason="state_stale",
    )

    assert result.process.returncode == 0
    assert result.alert_state == f"{NOW_EPOCH - 300}\tstate_stale\n"


def test_stale_state_still_alerts_when_the_watchdog_ran_on_schedule(tmp_path: Path) -> None:
    """A genuinely dead health-check on an awake machine must still alert."""
    result = _run_drill(
        tmp_path,
        label_loaded=True,
        state_mtime=NOW_EPOCH - STALE_SECONDS - 1,
        last_run_epoch=NOW_EPOCH - 300,
    )

    assert result.process.returncode == 1, result.process.stdout + result.process.stderr
    _assert_alert_contract(result)
    assert f"state_stale age={STALE_SECONDS + 1}s threshold={STALE_SECONDS}s" in result.tier0_log


def test_first_ever_run_with_no_recorded_history_still_alerts_on_stale_state(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        label_loaded=True,
        state_mtime=NOW_EPOCH - STALE_SECONDS - 1,
        last_run_epoch=None,
    )

    assert result.process.returncode == 1, result.process.stdout + result.process.stderr
    _assert_alert_contract(result)


def test_missed_runs_never_withhold_a_failure_that_is_not_a_function_of_elapsed_time(
    tmp_path: Path,
) -> None:
    """label_unloaded is true whether the Mac slept or not, so sleep must not mask it."""
    result = _run_drill(
        tmp_path,
        label_loaded=False,
        state_mtime=NOW_EPOCH - SLEPT_SECONDS,
        last_run_epoch=NOW_EPOCH - SLEPT_SECONDS,
    )

    assert result.process.returncode == 1, result.process.stdout + result.process.stderr
    _assert_alert_contract(result)
    assert "label_unloaded" in result.tier0_log


def test_watchdog_records_its_own_run_epoch_on_every_path(tmp_path: Path) -> None:
    cases = (
        {"label_loaded": True, "state_mtime": NOW_EPOCH - 60},  # healthy, exits 0
        {"label_loaded": False, "state_mtime": NOW_EPOCH - 60},  # incident, exits 1
        {  # withheld after sleep, exits 0
            "label_loaded": True,
            "state_mtime": NOW_EPOCH - SLEPT_SECONDS,
            "last_run_epoch": NOW_EPOCH - SLEPT_SECONDS,
        },
    )
    for index, kwargs in enumerate(cases):
        case_dir = tmp_path / f"case{index}"
        case_dir.mkdir()
        result = _run_drill(case_dir, **kwargs)
        assert result.run_state == f"{NOW_EPOCH}\n", kwargs
