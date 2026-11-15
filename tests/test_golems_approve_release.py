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
