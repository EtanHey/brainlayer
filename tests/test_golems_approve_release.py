"""Fail-closed release gates run without Apple credentials or platform commands."""

import importlib.util
import os
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/golems_approve_release.py"
spec = importlib.util.spec_from_file_location("golems_release", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def run(*args, env=None):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], env=env, capture_output=True, text=True)


@pytest.mark.parametrize("value", ["", "main", "30f54d07", "a" * 39, "g" * 40, "a" * 40 + "\n"])
def test_refuses_non_exact_sha(value):
    assert run("sha", value).returncode != 0


def test_accepts_full_sha():
    assert run("sha", "A" * 40).stdout.strip() == "a" * 40


def test_missing_secrets_refuses_without_values():
    env = dict(os.environ)
    for key in release.SECRETS:
        env.pop(key, None)
    env[release.SECRETS[0]] = "SENTINEL_PRIVATE_VALUE"
    result = run("secrets", env=env)
    assert result.returncode != 0
    assert "missing required secrets" in result.stderr
    assert "SENTINEL_PRIVATE_VALUE" not in result.stdout + result.stderr


def test_wrong_profile_refuses(tmp_path):
    profile = tmp_path / "wrong.provisionprofile"
    profile.write_bytes(b"wrong-profile")
    result = run("profile", profile)
    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr


def test_pinned_public_profile_passes():
    assert (
        run("profile", ROOT / ".github/signing/golems-approve/Golems_Approve_Developer_ID.provisionprofile").returncode
        == 0
    )


def test_missing_committed_entitlements_refuses_by_name(tmp_path):
    result = run("source", tmp_path)
    assert result.returncode != 0
    assert "missing committed native/golems-approve/GolemsApprove.entitlements" in result.stderr


def write_source(path, entitlements=None):
    (path / "GolemsApprove.entitlements").write_bytes(
        plistlib.dumps(release.ENTITLEMENTS if entitlements is None else entitlements)
    )
    (path / "Info.plist").write_bytes(
        plistlib.dumps(
            {
                "CFBundleIdentifier": release.IDENTIFIER,
                "CFBundleExecutable": "golems-approve",
                "CFBundlePackageType": "APPL",
            }
        )
    )


def test_exact_source_contract_passes(tmp_path):
    write_source(tmp_path)
    assert run("source", tmp_path).returncode == 0


def test_runtime_relaxation_refuses(tmp_path):
    write_source(tmp_path, {**release.ENTITLEMENTS, "com.apple.security.get-task-allow": True})
    assert run("source", tmp_path).returncode != 0


def test_missing_info_refuses_by_name(tmp_path):
    write_source(tmp_path)
    (tmp_path / "Info.plist").unlink()
    assert "missing committed native/golems-approve/Info.plist" in run("source", tmp_path).stderr


def test_workflow_separates_authority_and_cleans_before_launch():
    workflow = yaml.safe_load((ROOT / ".github/workflows/sign-golems-approve.yml").read_text())
    assert set(workflow.get("on", workflow.get(True))) == {"workflow_dispatch"}
    jobs = workflow["jobs"]
    assert workflow["permissions"] == {"contents": "read"}
    assert "secrets." not in str(jobs["build"])
    assert "secrets." not in str(jobs["publish"])
    assert jobs["publish"]["needs"] == ["build", "sign"]
    assert jobs["publish"]["permissions"] == {"contents": "write"}
    steps = jobs["sign"]["steps"]
    cleanup = next(i for i, s in enumerate(steps) if s.get("if") == "always()")
    probe = next(i for i, s in enumerate(steps) if "scripts/golems_approve_release.py verify" in s.get("run", ""))
    assert cleanup < probe
    assert "env" not in steps[probe]
    artifacts = [s["with"]["path"] for s in steps if s.get("uses", "").startswith("actions/upload-artifact@")]
    assert artifacts == ["dist/GolemsApprove.zip\ndist/SHA256SUMS\ndist/requirement.txt\n"]


def test_shell_entry_refuses_before_any_platform_command(tmp_path):
    env = dict(os.environ, RUNNER_TEMP=str(tmp_path))
    for key in release.SECRETS:
        env.pop(key, None)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/sign-golems-approve.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing required secrets" in result.stderr
    assert not list(tmp_path.iterdir())


def test_signing_requires_named_environment():
    workflow = yaml.safe_load((ROOT / ".github/workflows/sign-golems-approve.yml").read_text())
    assert workflow["jobs"]["sign"]["environment"] == "developer-id"


@pytest.fixture
def golems_history(tmp_path):
    upstream = tmp_path / "upstream"
    clone = tmp_path / "clone"

    def git(root, *args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    subprocess.run(["git", "init", "--initial-branch=master", str(upstream)], check=True, capture_output=True)
    git(
        upstream,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "--allow-empty",
        "-m",
        "master",
    )
    merged = git(upstream, "rev-parse", "HEAD")
    subprocess.run(["git", "clone", str(upstream), str(clone)], check=True, capture_output=True)
    git(clone, "checkout", "-b", "unmerged")
    git(
        clone,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "--allow-empty",
        "-m",
        "not merged",
    )
    unmerged = git(clone, "rev-parse", "HEAD")
    return upstream, clone, merged, unmerged, git


def test_unmerged_golems_sha_refuses(golems_history):
    _, clone, _, unmerged, _ = golems_history
    result = run("merged-sha", clone, unmerged)
    assert result.returncode != 0
    assert "not an ancestor of freshly fetched EtanHey/golems master" in result.stderr


def test_master_and_ancestor_pass_after_fresh_fetch(golems_history):
    upstream, clone, merged, _, git = golems_history
    assert run("merged-sha", clone, merged).returncode == 0
    git(
        upstream,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "--allow-empty",
        "-m",
        "new master",
    )
    new_master = git(upstream, "rev-parse", "HEAD")
    assert git(clone, "rev-parse", "origin/master") != new_master
    assert run("merged-sha", clone, new_master).returncode == 0
    assert git(clone, "rev-parse", "origin/master") == new_master
    assert run("merged-sha", clone, merged).returncode == 0


def protected_environment():
    return {
        "name": "developer-id",
        "protection_rules": [
            {
                "type": "required_reviewers",
                "prevent_self_review": True,
                "reviewers": [{"type": "User", "reviewer": {"login": "EtanHey"}}],
            }
        ],
        "deployment_branch_policy": {"protected_branches": False, "custom_branch_policies": True},
    }


def test_protected_environment_accepts_owner_and_main_only():
    release.environment_policy(protected_environment(), [{"name": "main", "type": "branch"}])


@pytest.mark.parametrize(
    "fault", ["missing", "no-review", "self-review", "other-reviewer", "all-branches", "tag", "extra-branch"]
)
def test_developer_id_environment_refuses_incomplete_protection(fault):
    metadata = protected_environment()
    policies = [{"name": "main", "type": "branch"}]
    if fault == "missing":
        metadata = {}
    elif fault == "no-review":
        metadata["protection_rules"] = []
    elif fault == "self-review":
        metadata["protection_rules"][0]["prevent_self_review"] = False
    elif fault == "other-reviewer":
        metadata["protection_rules"][0]["reviewers"].append({"type": "User", "reviewer": {"login": "someone"}})
    elif fault == "all-branches":
        metadata["deployment_branch_policy"] = None
    elif fault == "tag":
        policies[0]["type"] = "tag"
    else:
        policies.append({"name": "*", "type": "branch"})
    with pytest.raises(SystemExit, match="developer-id"):
        release.environment_policy(metadata, policies)


def test_environment_read_failure_refuses_without_response_leak(monkeypatch):
    def failed(command, **kwargs):
        assert command[:2] == ["gh", "api"]  # GET only, no settings mutation.
        return subprocess.CompletedProcess(command, 1, b"PRIVATE_SENTINEL", b"PRIVATE_SENTINEL")

    monkeypatch.setattr(release.subprocess, "run", failed)
    with pytest.raises(SystemExit, match="developer-id protected environment unavailable") as error:
        release.environment_ready()
    assert "PRIVATE_SENTINEL" not in str(error.value)


def test_workflow_admission_precedes_build_and_signing():
    workflow = yaml.safe_load((ROOT / ".github/workflows/sign-golems-approve.yml").read_text())
    build = workflow["jobs"]["build"]["steps"]
    runs = [s.get("run", "") for s in build]
    environment = next(i for i, s in enumerate(runs) if s.endswith(".py environment"))
    ancestry = next(i for i, s in enumerate(runs) if ".py merged-sha" in s)
    compile_step = next(i for i, s in enumerate(runs) if "swift build" in s)
    assert environment < ancestry < compile_step
    sign = workflow["jobs"]["sign"]["steps"]
    recheck = next(i for i, s in enumerate(sign) if s.get("run", "").endswith(".py environment"))
    credentials = next(i for i, s in enumerate(sign) if "secrets." in str(s))
    assert recheck < credentials
    assert "toolchain.txt" in runs[compile_step]
    assert "toolchain.txt >> dist/requirement.txt" in str(sign)


def test_master_fetch_failure_cannot_use_stale_ref(golems_history):
    _, clone, merged, _, git = golems_history
    git(clone, "remote", "set-url", "origin", str(clone / "missing-remote"))
    result = run("merged-sha", clone, merged)
    assert result.returncode != 0
    assert "cannot freshly fetch" in result.stderr


def test_shallow_checkout_fetches_complete_master(golems_history, tmp_path):
    upstream, _, merged, _, git = golems_history
    git(
        upstream,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "--allow-empty",
        "-m",
        "advance",
    )
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "--depth=1", upstream.as_uri(), str(shallow)], check=True, capture_output=True)
    assert git(shallow, "rev-parse", "--is-shallow-repository") == "true"
    assert run("merged-sha", shallow, merged).returncode == 0
    assert git(shallow, "rev-parse", "--is-shallow-repository") == "false"
