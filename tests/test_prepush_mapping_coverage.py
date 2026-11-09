"""New source modules must have an executable pre-push test owner."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "scripts/prepush_unmapped_legacy.json"


def _clean_git_env():
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def unmapped(paths):
    # Execute the production mapper, stopping before the runner starts any gates.
    definitions = (ROOT / "scripts/run_tests.sh").read_text().split('\ncd "$ROOT_DIR"\n')[0]
    definitions = "\n".join(
        "ROOT_DIR=" + shlex.quote(str(ROOT)) if line.startswith("ROOT_DIR=") else line
        for line in definitions.splitlines()
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("GIT_", "BRAINLAYER_PREPUSH", "BRAINLAYER_CHANGED_FILES"))
    }
    env["BRAINLAYER_CHANGED_FILES"] = "\n".join(paths)
    command = (
        definitions
        + '\nROOT_DIR="$1"\nTEST_ROOT="$1/tests"\nmap_changed_files_to_pytests\nif [ ${#unmapped_changed_files[@]} -gt 0 ]; then printf "%s\\n" "${unmapped_changed_files[@]}"; fi\n'
    )
    return set(
        subprocess.check_output(["bash", "-c", command, "mapping-check", str(ROOT)], env=env, text=True).splitlines()
    )


def require_owners(paths, legacy):
    missing = unmapped(paths) - legacy
    assert not missing, f"Add pre-push test mappings: {sorted(missing)}"


def test_new_unmapped_module_is_rejected():
    with pytest.raises(AssertionError, match="new_module_without_owner"):
        require_owners(["src/brainlayer/new_module_without_owner.py"], set())
    require_owners(["src/brainlayer/paths.py"], set())


def test_every_source_module_has_a_test_owner():
    modules = {str(p.relative_to(ROOT)) for p in (ROOT / "src/brainlayer").rglob("*.py")}
    legacy = set(json.loads(ALLOWLIST.read_text()))
    require_owners(sorted(modules), legacy)
    missing = unmapped(sorted(modules))
    assert not legacy - missing, f"Remove resolved legacy exceptions: {sorted(legacy - missing)}"


def test_legacy_allowlist_can_only_shrink():
    legacy = set(json.loads(ALLOWLIST.read_text()))
    # PR CI supplies the immutable base SHA. Locally use the existing remote-tracking base;
    # no network fetch, and no dependence on a filename convention for source ownership.
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    event = json.loads(Path(event_path).read_text()) if event_path else {}
    base = event.get("pull_request", {}).get("base", {}).get("sha", "origin/main")
    previous = subprocess.run(
        ["git", "show", f"{base}:scripts/prepush_unmapped_legacy.json"],
        cwd=ROOT,
        env=_clean_git_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if previous.returncode == 0:
        assert legacy <= set(json.loads(previous.stdout)), "Legacy mapping exceptions may only shrink"
    else:
        # Bootstrap only: every exception must already exist on the base, never a new PR module.
        existing = set(
            subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", base, "src/brainlayer"],
                cwd=ROOT,
                env=_clean_git_env(),
                text=True,
            ).splitlines()
        )
        assert legacy <= existing, "New source modules cannot be grandfathered"
