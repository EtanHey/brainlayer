"""Real Git scratch repositories; never fetch, build, tag, or push a fleet repo."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/release_worktree.py"


@pytest.fixture
def scratch(tmp_path):
    repo = tmp_path / "scratch repo"
    repo.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], env=env, text=True, capture_output=True, check=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "Release Test")
    git("config", "user.email", "release@example.test")
    (repo / "tracked").write_text("original\n")
    hooks = repo / "boundary hooks"
    hooks.mkdir()
    hook_log = tmp_path / "hook.log"
    hook = hooks / "pre-push"
    hook.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" >> "$HOOK_LOG"\ncat >> "$HOOK_LOG"\n')
    hook.chmod(0o755)
    git("config", "core.hooksPath", str(hooks))
    (repo / ".gitignore").write_text(".worktrees/\n.build.noindex/\n")
    git("add", "tracked", ".gitignore")
    git("commit", "-qm", "fixture")
    git("tag", "-a", "v-test", "-m", "fixture release")
    env["HOOK_LOG"] = str(hook_log)
    return repo, env, git, hook_log


def invoke(scratch, *command, ref="v-test", extra_env=None):
    repo, env, _, _ = scratch
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo), ref, "--", *command],
        env={**env, **(extra_env or {})},
        text=True,
        capture_output=True,
    )


def test_detached_shared_objects_hook_runs_and_clean_worktree_is_removed(scratch, tmp_path):
    repo, env, git, hook_log = scratch
    sink = tmp_path / "sink.git"
    subprocess.run(["git", "init", "-q", "--bare", str(sink)], env=env, check=True)
    before = git("config", "--local", "--list")
    code = (
        "import json,os,pathlib,subprocess; "
        "p=pathlib.Path.cwd(); "
        "g=lambda *a:subprocess.check_output(['git',*a],text=True).strip(); "
        "assert subprocess.run(['git','symbolic-ref','-q','HEAD']).returncode == 1; "
        "print(json.dumps({'path':str(p),'common':g('rev-parse','--path-format=absolute','--git-common-dir'),"
        "'sha':g('rev-parse','HEAD'),'env':os.environ['BRAINLAYER_RELEASE_WORKTREE']})); "
        "subprocess.run(['git','push',os.environ['TEST_SINK'],'refs/tags/v-test'],check=True); "
        "(p/'.build.noindex').mkdir(); (p/'.build.noindex'/'fake-build').write_text('fixture')"
    )
    result = invoke(
        scratch,
        sys.executable,
        "-c",
        code,
        extra_env={
            "TEST_SINK": str(sink),
            "GIT_DIR": "/wrong/inherited/repo",
            "GIT_WORK_TREE": "/wrong/inherited/worktree",
        },
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    worktree = Path(receipt["path"])
    assert worktree.parent == repo / ".worktrees"
    assert receipt["env"] == str(worktree)
    assert Path(receipt["common"]) == repo / ".git"
    assert receipt["sha"] == git("rev-parse", "v-test^{commit}")
    hook_lines = hook_log.read_text().splitlines()
    assert hook_lines[0] == str(worktree)
    assert hook_lines[1].startswith("refs/tags/v-test ")
    assert git("config", "--local", "--list") == before
    assert not worktree.exists()
    assert str(worktree) not in git("worktree", "list", "--porcelain")
    assert git("branch", "--show-current") == "main"
    assert (repo / "tracked").read_text() == "original\n"


def test_failed_clean_command_is_removed_and_exit_code_propagates(scratch):
    result = invoke(scratch, sys.executable, "-c", "raise SystemExit(23)")
    assert result.returncode == 23, result.stderr
    assert not list((scratch[0] / ".worktrees").iterdir())
    assert scratch[2]("worktree", "list", "--porcelain").count("worktree ") == 1


@pytest.mark.parametrize("status,expected", [(17, 17), (0, 1)])
def test_dirty_failure_is_preserved_without_force(scratch, status, expected):
    result = invoke(
        scratch,
        sys.executable,
        "-c",
        f"from pathlib import Path; Path('tracked').write_text('keep my change'); raise SystemExit({status})",
    )
    assert result.returncode == expected, result.stderr
    retained = list((scratch[0] / ".worktrees").iterdir())
    assert len(retained) == 1
    assert (retained[0] / "tracked").read_text() == "keep my change"
    assert "preserved" in result.stderr
    assert str(retained[0]) in scratch[2]("worktree", "list", "--porcelain")


def test_unknown_ref_creates_no_worktree(scratch):
    result = invoke(scratch, sys.executable, "-c", "raise AssertionError('must not run')", ref="missing-tag")
    assert result.returncode != 0
    assert not (scratch[0] / ".worktrees").exists()


def test_invocation_from_worker_uses_canonical_container_and_expands_path_argument(scratch):
    repo, env, git, hook_log = scratch
    caller = repo / ".worktrees/caller"
    git("worktree", "add", "--detach", str(caller), "v-test")
    result = invoke(
        (caller, env, git, hook_log),
        sys.executable,
        "-c",
        "import pathlib,sys; assert pathlib.Path.cwd()==pathlib.Path(sys.argv[1]); print(sys.argv[1])",
        "{worktree}",
    )
    assert result.returncode == 0, result.stderr
    generated = Path(result.stdout.strip())
    assert generated.parent == repo / ".worktrees"
    assert not generated.exists()
    assert not (caller / ".worktrees").exists()
    assert (caller / "tracked").read_text() == "original\n"
