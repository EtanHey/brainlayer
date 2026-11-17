"""Signing credentials are gated by main's workflow and exact-tag refusals."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_tag_dispatch_has_no_secrets_and_only_main_can_sign():
    workflow = yaml.safe_load((ROOT / ".github/workflows/brainbar-release.yml").read_text())
    dispatch = workflow["jobs"]["dispatch-from-main"]
    assert "secrets." not in str(dispatch)
    assert dispatch["if"] == "github.event_name == 'push'"
    assert dispatch["permissions"] == {"actions": "write", "contents": "read"}
    assert 'gh workflow run brainbar-release.yml --ref main -f tag="$GITHUB_REF_NAME"' in str(dispatch)
    release = workflow["jobs"]["release"]
    assert release["environment"] == "signing"
    assert release["if"] == "github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'"
    assert workflow[True]["workflow_dispatch"]["inputs"]["tag"]["required"] is True
    assert "secrets." not in str(workflow.get("env", {}))
    assert "secrets." not in str(release.get("env", {}))
    steps = release["steps"]
    refusals = [
        next(i for i, s in enumerate(steps) if token in s.get("run", ""))
        for token in (
            "git merge-base --is-ancestor",
            "ratchet_gate.py verify-release",
            "crash_sweep_gate.py verify-release",
        )
    ]
    for i, step in enumerate(steps):
        if "secrets." in str(step):
            assert all(gate < i for gate in refusals)
            assert not any(s.get("continue-on-error") or s.get("if") for s in steps[: max(refusals) + 1])
    validation = next(s["run"] for s in steps if "git merge-base --is-ancestor" in s.get("run", ""))
    for pattern in workflow[True]["push"]["tags"]:
        if pattern.startswith("!"):
            assert pattern[1:] in validation


def test_tag_validation_refuses_exclusions_bad_tags_and_off_main_commits(tmp_path):
    """Execute the workflow's real bash gate against an isolated Git history."""
    import os
    import subprocess

    workflow = yaml.safe_load((ROOT / ".github/workflows/brainbar-release.yml").read_text())
    validation = next(s for s in workflow["jobs"]["release"]["steps"] if s.get("id") == "release")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, env=env, text=True).strip()

    git("init", "-q")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    git("config", "core.hooksPath", os.devnull)
    git("config", "commit.gpgsign", "false")
    git("config", "tag.gpgsign", "false")
    git("commit", "--allow-empty", "-qm", "main")
    main = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/main", main)
    git("tag", "-a", "v9.0.0", "-m", "annotated release")
    git("commit", "--allow-empty", "-qm", "off-main")
    git("tag", "v9.0.1")
    excluded = [tag[1:] for tag in workflow[True]["push"]["tags"] if tag.startswith("!")]
    for tag in excluded:
        git("tag", tag, main)
    for tag in ["v9.0.0", "v9.0.1", "v8.0.0", "v192.0.2.1", "v9.0.0-rc1", "main", "$(false)", *excluded]:
        output = tmp_path / "output"
        github_env = tmp_path / "env"
        output.write_text("")
        github_env.write_text("")
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", validation["run"]],
            cwd=tmp_path,
            env={**env, "RELEASE_TAG": tag, "GITHUB_OUTPUT": str(output), "GITHUB_ENV": str(github_env)},
            capture_output=True,
            text=True,
        )
        assert (result.returncode == 0) == (tag == "v9.0.0"), result.stderr
        assert output.read_text() == (f"tag=v9.0.0\nsha={main}\n" if tag == "v9.0.0" else "")
