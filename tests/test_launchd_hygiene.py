from __future__ import annotations

import os
import plistlib
import re
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
            "ProcessType": "Adaptive",
            "ExitTimeOut": 60,
            "LowPriorityIO": True,
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
    assert "KeepAlive" not in drain
    assert drain["WatchPaths"] == ["__HOME__/.brainlayer/queue"]
    assert drain["QueueDirectories"] == ["__HOME__/.brainlayer/queue"]
    assert "--once" in drain["ProgramArguments"]

    backup = _load("scripts/launchd/com.brainlayer.backup-daily.plist")
    assert "KeepAlive" not in backup
    assert "StartCalendarInterval" in backup


def test_all_script_launchd_plists_have_common_hygiene():
    for path in sorted((REPO_ROOT / "scripts/launchd").glob("com.brainlayer.*.plist")):
        _assert_common_hygiene(plistlib.loads(path.read_bytes()))


def test_canonical_launchagent_env_has_no_concrete_dev_src_paths():
    plist_paths = [
        *sorted((REPO_ROOT / "scripts/launchd").glob("com.brainlayer.*.plist")),
        REPO_ROOT / "launchd/com.brainlayer.watch.plist",
        REPO_ROOT / "brain-bar/bundle/com.brainlayer.brainbar.plist",
        REPO_ROOT / "brain-bar/bundle/com.brainlayer.brainbar-daemon.plist",
    ]

    for path in plist_paths:
        _assert_no_dev_src_path_in_canonical_env(path, plistlib.loads(path.read_bytes()))
