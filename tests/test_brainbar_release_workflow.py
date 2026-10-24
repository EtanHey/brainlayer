from __future__ import annotations

import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from yaml.parser import ParserError
from yaml.scanner import ScannerError

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "brainbar-release.yml"


def _load_workflow() -> dict:
    with WORKFLOW_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _events(workflow: dict) -> dict:
    return workflow.get("on") or workflow.get(True) or {}


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _release_job(workflow: dict) -> dict:
    for job in workflow.get("jobs", {}).values():
        if job.get("runs-on") == "macos-15":
            return job
    raise AssertionError("BrainBar release workflow must run a job on macos-15")


def _step_run(step: dict) -> str:
    return str(step.get("run", ""))


def _step_runs_in_brain_bar(step: dict, command: str) -> bool:
    run = _step_run(step)
    working_directory = str(step.get("working-directory", ""))
    return command in run and (working_directory == "brain-bar" or "cd brain-bar" in run)


def _is_release_step(step: dict) -> bool:
    uses = str(step.get("uses", ""))
    run = _step_run(step)
    return "action-gh-release" in uses or "gh release" in run


def _is_no_zip_guardrail(step: dict) -> bool:
    run = _step_run(step)
    return all(
        token in run
        for token in (
            "BrainBar.zip",
            "-s",
            "exit 1",
            "::error",
        )
    )


def _assert_tag_triggered(workflow: dict) -> None:
    push = _events(workflow).get("push", {})
    assert "tags" in push, "BrainBar release workflow must be triggered by pushed tags"
    assert _as_list(push["tags"]), "BrainBar release workflow must define at least one pushed tag pattern"


def _assert_swift_build_contract(job: dict) -> None:
    steps = job.get("steps", [])
    assert any(
        isinstance(step, dict) and str(step.get("uses", "")).startswith("maxim-lobanov/setup-xcode@") for step in steps
    ), "BrainBar release workflow must select Xcode with maxim-lobanov/setup-xcode"
    assert any(
        isinstance(step, dict) and step.get("with", {}).get("xcode-version") == "latest-stable" for step in steps
    ), "BrainBar release workflow must use the latest stable Xcode for Swift 6.x"
    assert any(isinstance(step, dict) and _step_runs_in_brain_bar(step, "swift build -c release") for step in steps), (
        "BrainBar release workflow must run swift build -c release inside brain-bar"
    )


def _assert_no_zip_guardrail_before_release(job: dict) -> None:
    steps = job.get("steps", [])
    guard_indexes = [index for index, step in enumerate(steps) if isinstance(step, dict) and _is_no_zip_guardrail(step)]
    assert guard_indexes, "BrainBar release workflow must fail loudly when BrainBar.zip is missing or empty"

    release_indexes = [index for index, step in enumerate(steps) if isinstance(step, dict) and _is_release_step(step)]
    assert release_indexes, "BrainBar release workflow must attach BrainBar.zip to a GitHub Release"
    assert min(guard_indexes) < min(release_indexes), "BrainBar.zip guardrail must run before the release step"


def _assert_version_stamp(job: dict) -> None:
    version_steps = [
        step for step in job.get("steps", []) if isinstance(step, dict) and "Info.plist" in _step_run(step)
    ]
    assert any("GitCommit" in _step_run(step) for step in version_steps), "Info.plist must be stamped with GitCommit"
    assert any("BuildTimeUTC" in _step_run(step) for step in version_steps), (
        "Info.plist must be stamped with BuildTimeUTC"
    )
    assert any("CFBundleShortVersionString" in _step_run(step) for step in version_steps), (
        "Info.plist release version must come from the pushed tag"
    )
    assert any("BrainLayerReleaseVersion" in _step_run(step) for step in version_steps), (
        "Info.plist must preserve the full BrainLayer release identity"
    )
    assert any(
        "git rev-list --count HEAD" in _step_run(step) for step in job.get("steps", []) if isinstance(step, dict)
    ), "Info.plist CFBundleVersion must use an Apple-compatible monotonically increasing build number"
    metadata_steps = [
        step
        for step in job.get("steps", [])
        if isinstance(step, dict) and "BRAINBAR_RELEASE_VERSION" in _step_run(step)
    ]
    assert any(r"^[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?$" in _step_run(step) for step in metadata_steps), (
        "BrainBar release workflow must reject tags that cannot be stamped into macOS bundle versions"
    )


def _assert_signing_decode_is_macos_safe(job: dict) -> None:
    runs = "\n".join(_step_run(step) for step in job.get("steps", []) if isinstance(step, dict))
    assert "base64 --decode" not in runs, "macOS base64 does not support --decode in the signing path"
    assert "base64 -D" in runs, "BrainBar release workflow must decode signing certificates with macOS base64 -D"


def _assert_launchagents_are_packaged(job: dict) -> None:
    runs = "\n".join(_step_run(step) for step in job.get("steps", []) if isinstance(step, dict))
    assert "AppIcon.icns" in runs, "BrainBar.app must package the menu bar app icon resource"
    assert "Contents/Resources/LaunchAgents" in runs, "BrainBar.app must package launch agent templates"
    assert "com.brainlayer.brainbar.plist" in runs, "BrainBar.app must include the UI launch agent plist"
    assert "com.brainlayer.brainbar-daemon.plist" in runs, "BrainBar.app must include the daemon launch agent plist"


def _assert_gatekeeper_ready_release_contract(job: dict) -> None:
    runs = "\n".join(_step_run(step) for step in job.get("steps", []) if isinstance(step, dict))
    assert "::warning title=BrainBar signing skipped" not in runs, (
        "BrainBar release workflow must not upload unsigned app zips when Developer ID secrets are missing"
    )
    assert "::warning title=BrainBar notarization skipped" not in runs, (
        "BrainBar release workflow must not upload unnotarized app zips when notary secrets are missing"
    )
    assert "exit 1" in runs and "BrainBar signing unavailable" in runs, (
        "BrainBar release workflow must fail closed when Developer ID signing secrets are missing"
    )
    assert "exit 1" in runs and "BrainBar notarization unavailable" in runs, (
        "BrainBar release workflow must fail closed when notary secrets are missing"
    )
    assert "codesign --verify --deep --strict --verbose=4 dist/BrainBar.app" in runs, (
        "BrainBar release workflow must verify the signed app resource seal before notarization"
    )
    assert "xcrun stapler validate dist/BrainBar.app" in runs, (
        "BrainBar release workflow must validate the stapled notarization ticket"
    )
    assert "spctl --assess --type execute --verbose=4 dist/BrainBar.app" in runs, (
        "BrainBar release workflow must require Gatekeeper acceptance before packaging"
    )
    assert 'ditto -x -k dist/BrainBar.zip "$verify_dir"' in runs, (
        "BrainBar release workflow must unpack the final zip before publishing"
    )
    assert 'codesign --verify --deep --strict --verbose=4 "$verify_dir/BrainBar.app"' in runs, (
        "BrainBar release workflow must verify the packaged app signature before publishing"
    )
    assert 'spctl --assess --type execute --verbose=4 "$verify_dir/BrainBar.app"' in runs, (
        "BrainBar release workflow must require Gatekeeper acceptance on the packaged app before publishing"
    )


def _assert_action_refs_are_pinned(job: dict) -> None:
    action_steps = [step for step in job.get("steps", []) if isinstance(step, dict) and "uses" in step]
    assert action_steps, "BrainBar release workflow must declare action steps explicitly"
    for step in action_steps:
        uses = str(step["uses"])
        assert re.search(r"@[0-9a-f]{40}$", uses), f"BrainBar release action must be pinned to a SHA: {uses}"


def _assert_swift_cache_excludes_local_build(job: dict) -> None:
    cache_steps = [
        step
        for step in job.get("steps", [])
        if isinstance(step, dict) and str(step.get("uses", "")).startswith("actions/cache@")
    ]
    assert cache_steps, "BrainBar release workflow must cache SwiftPM dependencies"
    cache_paths = "\n".join(str(step.get("with", {}).get("path", "")) for step in cache_steps)
    assert "~/.swiftpm" in cache_paths, "BrainBar release workflow must cache external SwiftPM dependencies"
    assert "brain-bar/.build" not in cache_paths, "BrainBar release workflow must not restore local .build artifacts"


def _assert_release_workflow_contract(workflow: dict) -> None:
    _assert_tag_triggered(workflow)
    job = _release_job(workflow)
    _assert_action_refs_are_pinned(job)
    _assert_swift_build_contract(job)
    _assert_swift_cache_excludes_local_build(job)
    _assert_version_stamp(job)
    _assert_signing_decode_is_macos_safe(job)
    _assert_launchagents_are_packaged(job)
    _assert_gatekeeper_ready_release_contract(job)
    _assert_no_zip_guardrail_before_release(job)


def test_brainbar_release_workflow_is_valid_yaml() -> None:
    try:
        _load_workflow()
    except (ParserError, ScannerError) as exc:
        raise AssertionError(f"{WORKFLOW_PATH} is not valid YAML") from exc


def test_brainbar_release_workflow_contract() -> None:
    _assert_release_workflow_contract(_load_workflow())


def test_release_workflow_without_no_zip_guardrail_fails_contract() -> None:
    workflow = deepcopy(_load_workflow())
    job = _release_job(workflow)
    job["steps"] = [
        step for step in job.get("steps", []) if not (isinstance(step, dict) and _is_no_zip_guardrail(step))
    ]

    with pytest.raises(AssertionError, match="missing or empty"):
        _assert_no_zip_guardrail_before_release(job)


def test_release_workflow_with_signing_skip_fails_contract() -> None:
    workflow = deepcopy(_load_workflow())
    job = _release_job(workflow)
    for step in job.get("steps", []):
        if isinstance(step, dict) and step.get("name") == "Import Developer ID certificate":
            step["run"] += "\necho '::warning title=BrainBar signing skipped::bad'\n"

    with pytest.raises(AssertionError, match="unsigned app zips"):
        _assert_gatekeeper_ready_release_contract(job)


# This simulates ticket delivery, not Apple's network or PF itself. PATH contains only owned
# fakes (plus mkdir/rm for the private extraction directory); no host sudo/PF can be reached.
_FAKE_RELEASE_COMMAND = r"""
import os
import sys
from pathlib import Path

root = Path(os.environ["FAKE_ROOT"])
command = Path(sys.argv[0]).name
args = sys.argv[1:]
if command == "sudo":
    assert args in (["pfctl", "-f", "-"], ["pfctl", "-E"]), args
    os.execv(str(root / "bin" / "pfctl"), args)
if command == "ditto":
    if args[0] == "-c":
        Path(args[-1]).write_text("fake signed app archive")
    else:
        assert args[:3] == ["-x", "-k", "dist/BrainBar.zip"], args
        (Path(args[-1]) / "BrainBar.app").mkdir()
    sys.exit(0)
if command in ("ls", "stat"):
    print("21")
    sys.exit(0)

packaged = args[-1] != "dist/BrainBar.app"
if command == "xcrun":
    if args[:2] == ["notarytool", "submit"]:
        stage = "notary"
    else:
        assert args[:2] in (["stapler", "staple"], ["stapler", "validate"]), args
        stage = args[1] if not packaged else "packaged-validate"
elif command == "pfctl":
    assert (root / "notary").exists(), "PF must follow accepted notary"
    assert not (root / "staple").exists(), "PF must precede stapling"
    if args == ["-f", "-"]:
        assert sys.stdin.read() == "block return out quick proto udp from any to any port 443\n"
        stage = "pf-config"
    else:
        assert args == ["-E"] and (root / "pf-config").exists(), args
        stage = "pf-enable"
elif command == "codesign":
    if "--force" in args:
        stage = "sign"
    elif "--verify" not in args:
        sys.exit(0)
    else:
        stage = "signature" if (root / "notary").exists() else "initial-signature"
        if packaged:
            stage = "packaged-signature"
else:
    assert command == "spctl", command
    stage = "packaged-gatekeeper" if packaged else "gatekeeper"
with (root / "commands.log").open("a") as log:
    log.write(stage + "\n")
if stage == os.environ["FAIL_STAGE"]:
    print("status: Invalid" if stage == "notary" else "simulated " + stage + " failure")
    sys.exit(68 if stage == "staple" else 23)
if stage == "staple" and not (root / "pf-enable").exists():
    print("simulated ticket delivery: network -1005; stapler Error 68")
    sys.exit(68)
(root / stage).touch()
if stage == "notary":
    print("status: Accepted")
"""


def _run_fake_release(tmp_path: Path, fail_stage: str = "") -> tuple[int, list[str], str, str]:
    job = _release_job(_load_workflow())
    steps = job["steps"]
    start = next(i for i, step in enumerate(steps) if step.get("name") == "Sign BrainBar.app")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for command in ("xcrun", "sudo", "pfctl", "ditto", "codesign", "spctl", "ls", "stat"):
        path = fake_bin / command
        path.write_text(f"#!{sys.executable}\n" + _FAKE_RELEASE_COMMAND, encoding="utf-8")
        path.chmod(0o700)
    for command in ("mkdir", "rm"):
        (fake_bin / command).symlink_to(f"/bin/{command}")
    (tmp_path / "dist" / "BrainBar.app").mkdir(parents=True)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    github_env = tmp_path / "github-env"
    github_env.touch()
    env = {
        "PATH": str(fake_bin),
        "FAKE_ROOT": str(tmp_path),
        "FAIL_STAGE": fail_stage,
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_ENV": str(github_env),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "BRAINBAR_SIGNING_IDENTITY": "fake signing identity",
        "BRAINBAR_NOTARY_APPLE_ID": "fake@example.invalid",
        "BRAINBAR_NOTARY_PASSWORD": "fake-password",
        "BRAINBAR_NOTARY_TEAM_ID": "FAKE-TEAM",
    }
    output = ""
    returncode = 0
    for index, step in enumerate(steps[start:], start):
        # These are the GitHub default-success semantics modeled by this runner.
        # Reject overrides rather than silently simulating a safer workflow than YAML declares.
        assert not step.get("continue-on-error")
        if index > start + 1:
            assert "if" not in step
        else:
            assert step.get("if") == "steps.signing.outputs.signed == 'true'"
        if _is_release_step(step):
            with (tmp_path / "commands.log").open("a") as log:
                log.write("release\n")  # Reachability sentinel; no GitHub action is executed.
            break
        script = tmp_path / f"step-{index}.bash"
        script.write_text(_step_run(step), encoding="utf-8")
        result = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-e", "-o", "pipefail", str(script)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output += result.stdout + result.stderr
        returncode = result.returncode
        if returncode:
            break
    (tmp_path / "output.log").write_text(output, encoding="utf-8")
    events = (tmp_path / "commands.log").read_text(encoding="utf-8").splitlines()
    return returncode, events, github_env.read_text(encoding="utf-8"), output


_RELEASE_EVENTS = [
    "sign",
    "initial-signature",
    "notary",
    "pf-config",
    "pf-enable",
    "staple",
    "validate",
    "signature",
    "gatekeeper",
    "packaged-signature",
    "packaged-validate",
    "packaged-gatekeeper",
    "release",
]


def test_ticket_transport_fallback_before_stapling(tmp_path: Path) -> None:
    returncode, events, github_env, output = _run_fake_release(tmp_path)

    assert returncode == 0, output
    assert events == _RELEASE_EVENTS
    assert github_env == "BRAINBAR_NOTARIZATION_STATUS=notarized and stapled\n"


@pytest.mark.parametrize("fail_stage", _RELEASE_EVENTS[:-1])
def test_release_aborts_at_each_failed_gate(tmp_path: Path, fail_stage: str) -> None:
    returncode, events, github_env, output = _run_fake_release(tmp_path, fail_stage)

    assert returncode == (68 if fail_stage == "staple" else 23), output
    assert events == _RELEASE_EVENTS[: _RELEASE_EVENTS.index(fail_stage) + 1]
    if not fail_stage.startswith("packaged-"):
        assert "BRAINBAR_NOTARIZATION_STATUS" not in github_env
    assert "release" not in events
