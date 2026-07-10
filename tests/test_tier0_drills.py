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
DOMAIN = "gui/501"
LABEL = "com.example.brainlayer-health-check"
NOTIFY_ENDPOINT = "http://localhost:3847/notify"


@dataclass(frozen=True)
class DrillResult:
    process: subprocess.CompletedProcess[str]
    events: list[str]
    tier0_log: str


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
) -> DrillResult:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    events_path = tmp_path / "events.log"
    state_path = tmp_path / "health-check-state.json"
    health_plist_path = tmp_path / "com.example.brainlayer-health-check.plist"
    tier0_log_path = tmp_path / "logs" / "tier0-watchdog.log"
    health_plist_path.write_text("fixture\n", encoding="utf-8")

    if state_mtime is not None:
        state_path.write_text("{}\n", encoding="utf-8")

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
        "TIER0_ALERT_TIMEOUT_SECONDS": "1",
        "TIER0_CURL": str(fake_bin / "curl"),
        "TIER0_DOMAIN": DOMAIN,
        "TIER0_DRILL_EVENTS": str(events_path),
        "TIER0_HEALTH_PLIST_PATH": str(health_plist_path),
        "TIER0_LABEL": LABEL,
        "TIER0_LAUNCHCTL": str(fake_bin / "launchctl"),
        "TIER0_LOG_PATH": str(tier0_log_path),
        "TIER0_NOTIFY_ENDPOINT": NOTIFY_ENDPOINT,
        "TIER0_NOTIFY_TIMEOUT_SECONDS": "1",
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
    return DrillResult(process=process, events=events, tier0_log=tier0_log)


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


def test_d3_hanging_notify_endpoint_cannot_suppress_local_alert_or_heal(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        label_loaded=True,
        state_mtime=NOW_EPOCH - STALE_SECONDS - 1,
        curl_hangs=True,
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


def test_alert_fanout_uses_one_shared_deadline(tmp_path: Path) -> None:
    result = _run_drill(
        tmp_path,
        label_loaded=True,
        state_mtime=NOW_EPOCH - STALE_SECONDS - 1,
        curl_hangs=True,
        osascript_hangs=True,
        use_fake_wait_sleep=True,
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
    args = " ".join(plist["ProgramArguments"])
    assert "ENV_RUN" not in args
    assert "PYTHON" not in args.upper()
