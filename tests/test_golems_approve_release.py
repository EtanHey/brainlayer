"""Fail-closed release gates run without Apple credentials or platform commands."""

import importlib.util
import json
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


def _clean_git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def run(*args, env=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        env=_clean_git_env() if env is None else env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("value", ["", "main", "30f54d07", "a" * 39, "g" * 40, "a" * 40 + "\n"])
def test_refuses_non_exact_sha(value):
    assert run("sha", value).returncode != 0


def test_accepts_full_sha():
    assert run("sha", "A" * 40).stdout.strip() == "a" * 40


def test_missing_secrets_refuses_without_values():
    env = _clean_git_env()
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
    assert jobs["publish"]["needs"] == ["build", "sign", "sign_environment"]
    assert jobs["publish"]["permissions"] == {"contents": "write"}
    steps = jobs["sign"]["steps"]
    cleanup = next(i for i, s in enumerate(steps) if s.get("if") == "always()")
    probe = next(i for i, s in enumerate(steps) if "scripts/golems_approve_release.py verify" in s.get("run", ""))
    assert cleanup < probe
    assert "env" not in steps[probe]
    artifacts = [s["with"]["path"] for s in steps if s.get("uses", "").startswith("actions/upload-artifact@")]
    assert artifacts == ["dist/GolemsApprove.zip\ndist/SHA256SUMS\ndist/requirement.txt\n"]


def test_shell_entry_refuses_before_any_platform_command(tmp_path):
    env = dict(_clean_git_env(), RUNNER_TEMP=str(tmp_path))
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


@pytest.mark.parametrize("selected", ["", "release-signing"])
def test_signing_environment_switch_routes_exclusive_jobs(selected):
    workflow = yaml.safe_load((ROOT / ".github/workflows/sign-golems-approve.yml").read_text())
    jobs = workflow["jobs"]
    assert run("environment-name", selected).stdout.strip() == selected
    assert jobs["sign"]["if"] == "needs.build.outputs.signing_environment == ''"
    assert jobs["sign_environment"]["if"] == "needs.build.outputs.signing_environment != ''"
    assert "environment" not in jobs["sign"]
    assert jobs["sign_environment"]["environment"] == "${{ needs.build.outputs.signing_environment }}"
    assert jobs["sign"]["steps"] is jobs["sign_environment"]["steps"]
    assert jobs["build"]["outputs"]["signing_environment"] == "${{ steps.signing_environment.outputs.name }}"
    resolve = next(s for s in jobs["build"]["steps"] if s.get("id") == "signing_environment")
    assert resolve["env"]["SIGNING_ENV"] == "${{ vars.GOLEMS_APPROVE_SIGNING_ENV }}"
    for name in ("sign", "sign_environment"):
        assert jobs[name]["permissions"] == {"contents": "read", "actions": "read"}


def test_publish_requires_exactly_one_successful_signing_job():
    workflow = yaml.safe_load((ROOT / ".github/workflows/sign-golems-approve.yml").read_text())
    condition = " ".join(workflow["jobs"]["publish"]["if"].split())
    assert condition == (
        "!cancelled() && needs.build.result == 'success' && "
        "((needs.sign.result == 'success' && needs.sign_environment.result == 'skipped') || "
        "(needs.sign.result == 'skipped' && needs.sign_environment.result == 'success'))"
    )


@pytest.fixture
def golems_history(tmp_path):
    upstream = tmp_path / "upstream"
    clone = tmp_path / "clone"

    def git(root, *args):
        return subprocess.check_output(["git", "-C", str(root), *args], env=_clean_git_env(), text=True).strip()

    subprocess.run(
        ["git", "init", "--initial-branch=master", str(upstream)], check=True, capture_output=True, env=_clean_git_env()
    )
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
    subprocess.run(["git", "clone", str(upstream), str(clone)], check=True, capture_output=True, env=_clean_git_env())
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


def configured_environment():
    return {
        "name": "release-signing",
        "deployment_branch_policy": {"protected_branches": False, "custom_branch_policies": True},
    }


def deployment_policies():
    return [{"name": "main", "type": "branch"}, {"name": "v*", "type": "tag"}]


def test_empty_switch_uses_repository_secrets_without_metadata_access(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Empty environment switch must not access environment metadata")

    monkeypatch.setattr(release.subprocess, "run", unexpected)
    release.environment_ready("")


def test_named_environment_accepts_main_and_version_tags_without_other_rules():
    release.environment_policy("release-signing", configured_environment(), deployment_policies())
    release.environment_policy("release-signing", configured_environment(), list(reversed(deployment_policies())))


@pytest.mark.parametrize(
    "fault", ["missing", "unrestricted", "missing-tags", "extra-branch", "wrong-tags", "tag-main", "duplicate"]
)
def test_named_environment_rejects_other_deployment_policies(fault):
    metadata = configured_environment()
    policies = deployment_policies()
    if fault == "missing":
        metadata = {}
    elif fault == "unrestricted":
        metadata["deployment_branch_policy"] = None
    elif fault == "missing-tags":
        policies.pop()
    elif fault == "extra-branch":
        policies.append({"name": "*", "type": "branch"})
    elif fault == "wrong-tags":
        policies[1]["name"] = "*"
    elif fault == "tag-main":
        policies[0]["type"] = "tag"
    else:
        policies[1] = policies[0]
    with pytest.raises(SystemExit, match="signing environment release-signing"):
        release.environment_policy("release-signing", metadata, policies)


def test_named_environment_reads_only_selected_metadata_with_pagination(monkeypatch):
    calls = []

    def get(command, **kwargs):
        calls.append(command)
        assert command[:4] == ["gh", "api", "--method", "GET"]
        assert "--paginate" in command and "--slurp" in command
        assert "/environments/release-signing" in command[-1]
        payload = (
            [{"branch_policies": deployment_policies()[:1]}, {"branch_policies": deployment_policies()[1:]}]
            if "deployment-branch-policies" in command[-1]
            else [configured_environment()]
        )
        return subprocess.CompletedProcess(command, 0, json.dumps(payload).encode(), b"")

    monkeypatch.setattr(release.subprocess, "run", get)
    release.environment_ready("release-signing")
    assert len(calls) == 2


def test_environment_read_failure_refuses_without_response_leak(monkeypatch):
    def failed(command, **kwargs):
        assert command[:4] == ["gh", "api", "--method", "GET"]
        return subprocess.CompletedProcess(command, 1, b"PRIVATE_SENTINEL", b"PRIVATE_SENTINEL")

    monkeypatch.setattr(release.subprocess, "run", failed)
    with pytest.raises(SystemExit, match="signing environment release-signing unavailable") as error:
        release.environment_ready("release-signing")
    assert "PRIVATE_SENTINEL" not in str(error.value)


@pytest.mark.parametrize("value", ["bad\nname", "../outside", "has space", "x" * 256])
def test_environment_switch_rejects_unsafe_names(value):
    assert run("environment-name", value).returncode != 0


def test_workflow_admission_precedes_build_and_signing():
    workflow = yaml.safe_load((ROOT / ".github/workflows/sign-golems-approve.yml").read_text())
    build = workflow["jobs"]["build"]["steps"]
    runs = [s.get("run", "") for s in build]
    environment = next(i for i, s in enumerate(runs) if ".py environment " in s)
    ancestry = next(i for i, s in enumerate(runs) if ".py merged-sha" in s)
    compile_step = next(i for i, s in enumerate(runs) if "swift build" in s)
    assert environment < ancestry < compile_step
    sign = workflow["jobs"]["sign"]["steps"]
    recheck = next(i for i, s in enumerate(sign) if ".py environment " in s.get("run", ""))
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
    subprocess.run(
        ["git", "clone", "--depth=1", upstream.as_uri(), str(shallow)],
        check=True,
        capture_output=True,
        env=_clean_git_env(),
    )
    assert git(shallow, "rev-parse", "--is-shallow-repository") == "true"
    assert run("merged-sha", shallow, merged).returncode == 0
    assert git(shallow, "rev-parse", "--is-shallow-repository") == "false"


def test_golems_action_checks_out_only_trusted_master_before_local_sha_selection():
    workflow = yaml.safe_load((ROOT / ".github/workflows/sign-golems-approve.yml").read_text())
    steps = workflow["jobs"]["build"]["steps"]
    checkout = next(s for s in steps if s.get("with", {}).get("repository") == "EtanHey/golems")
    assert checkout["with"]["ref"] == "master"
    assert checkout["with"]["fetch-depth"] == 0
    admission = next(s["run"] for s in steps if ".py merged-sha" in s.get("run", ""))
    assert admission.index(".py merged-sha") < admission.index('checkout --detach "$GOLEMS_SHA"')


def test_git_fixture_and_admission_ignore_inherited_git_env(tmp_path, monkeypatch):
    poisoned = tmp_path / "parent.git"
    monkeypatch.setenv("GIT_DIR", str(poisoned))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "parent-worktree"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "parent-index"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.bare")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    _, clone, merged, unmerged, git = golems_history.__wrapped__(tmp_path)
    assert git(clone, "rev-parse", "HEAD") == unmerged
    assert run("merged-sha", clone, merged).returncode == 0
    assert not poisoned.exists()
    assert not (tmp_path / "parent-index").exists()
