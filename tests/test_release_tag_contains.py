"""The tag-contains gate: a release may only claim fixes provably inside the tag.

The incident: v1.5.15 was tagged at 51a72a06 (2026-09-05 20:12:29); #778 merged at 78d92bcb
20:34:55 -- 22 minutes LATER. A deploy brief still said "1.5.15 carries #778". Nobody lied; the
claim was a memory of a same-evening merge. These tests pin the check that makes it provable.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "release_tag_contains.py"


def _clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """`.githooks/pre-push` runs this suite with GIT_DIR/GIT_INDEX_FILE exported.

    Those win over `cwd`, so an unscrubbed fixture commits into the REAL repo instead of its
    temp one — measured: every fixture died on `git commit` under the hook while passing bare.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(extra or {})
    return env


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    """Identity rides on `-c`, never `git config`.

    A `git config user.email` write goes to the COMMON config — shared by every worktree — so if
    the scrub above ever regresses, this fixture must still leave no trace in the real repo.
    Measured the hard way: an unscrubbed run wrote `user.name = Gate Test` and `core.bare = true`
    into ~/Gits/brainlayer/.git/config and committed `feat: one` onto the branch being pushed.
    """
    (repo / "file.txt").write_text(message)
    _git(repo, "add", "file.txt")
    _git(
        repo,
        "-c",
        "user.email=gate@example.com",
        "-c",
        "user.name=Gate Test",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        message,
    )
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """c1 <v1.0.0>  c2 <v1.0.1-rc1>  c3 <v1.1.0>  c4 (untagged, after the tag)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    # Fail LOUD before the first commit if anything still points at another repository.
    git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir")).resolve()
    assert tmp_path.resolve() in git_dir.parents, f"fixture escaped its tmp repo: {git_dir}"

    shas = {}
    shas["c1"] = _commit(repo, "feat: one")
    _git(repo, "tag", "v1.0.0")
    shas["c2"] = _commit(repo, "fix: two | with a pipe")
    _git(repo, "tag", "v1.0.1-rc1")
    shas["c3"] = _commit(repo, "fix: three")
    _git(repo, "tag", "v1.1.0")
    shas["c4"] = _commit(repo, "fix: four, merged after the tag")
    (repo / "shas.json").write_text(json.dumps(shas))
    return repo


def _shas(repo: Path) -> dict[str, str]:
    return json.loads((repo / "shas.json").read_text())


def _run(repo: Path, *args: str, env: dict[str, str] | None = None):
    return subprocess.run(
        ["python3", str(SCRIPT), *args],
        cwd=repo,
        env=_clean_env(env),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_exe(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


# --- the core gate --------------------------------------------------------------------------


def test_claimed_commit_inside_the_tag_passes(repo: Path) -> None:
    result = _run(repo, "v1.1.0", _shas(repo)["c3"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"| {_shas(repo)['c3'][:12]} | fix: three | IN |" in result.stdout
    assert "NOT IN" not in result.stdout


def test_claimed_commit_after_the_tag_fails_non_zero(repo: Path) -> None:
    """The v1.5.15/#778 shape: the fix merged AFTER the tag was cut."""
    result = _run(repo, "v1.1.0", _shas(repo)["c4"])

    assert result.returncode == 1
    assert f"| {_shas(repo)['c4'][:12]} | fix: four, merged after the tag | NOT IN |" in result.stdout
    assert "does NOT contain" in result.stdout


def test_one_bad_claim_among_good_ones_still_fails(repo: Path) -> None:
    shas = _shas(repo)
    result = _run(repo, "v1.1.0", shas["c2"], shas["c3"], shas["c4"])

    assert result.returncode == 1
    assert result.stdout.count("| IN |") == 2
    assert result.stdout.count("| NOT IN |") == 1


def test_commit_absent_from_this_repo_is_not_in_rather_than_a_crash(repo: Path) -> None:
    """An unfetched merge commit must read NOT IN, not blow up mid-table."""
    absent = "deadbeef" * 5
    result = _run(repo, "v1.1.0", absent)

    assert result.returncode == 1
    assert "NOT IN" in result.stdout
    assert "not in this repo" in result.stdout


def test_an_inherited_git_dir_cannot_redirect_the_gate(repo: Path, tmp_path: Path) -> None:
    """Run from a hook or `git rebase --exec`, GIT_DIR would answer for the WRONG repository.

    `other` has no v1.1.0, so an unscrubbed run exits 2 on "no such tag" — a release gate that
    silently graded a different repo is exactly the unfalsifiable claim this script closes.
    """
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q", "-b", "main")

    result = _run(repo, "v1.1.0", _shas(repo)["c3"], env={"GIT_DIR": str(other / ".git")})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "| IN |" in result.stdout


def test_unknown_tag_is_a_usage_error_not_a_pass(repo: Path) -> None:
    result = _run(repo, "v9.9.9", _shas(repo)["c3"])

    assert result.returncode == 2
    assert "v9.9.9" in result.stderr


def test_no_claims_still_reports_what_the_tag_carries(repo: Path) -> None:
    result = _run(repo, "v1.1.0")

    assert result.returncode == 0
    assert "no claims given" in result.stdout
    assert "fix: three" in result.stdout


# --- PR-number input ------------------------------------------------------------------------


def _fake_gh(tmp_path: Path, payload: dict[str, object]) -> Path:
    return _write_exe(
        tmp_path / "gh",
        f"#!/usr/bin/env bash\ncat <<'JSON'\n{json.dumps(payload)}\nJSON\n",
    )


def test_pr_number_resolves_to_its_merge_commit(repo: Path, tmp_path: Path) -> None:
    shas = _shas(repo)
    gh = _fake_gh(tmp_path, {"mergeCommit": {"oid": shas["c3"]}, "title": "fix: three"})

    result = _run(repo, "v1.1.0", "#778", env={"BRAINLAYER_GH_BIN": str(gh)})

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"| #778 | {shas['c3'][:12]} | fix: three | IN |" in result.stdout


def test_pr_number_whose_merge_commit_is_after_the_tag_fails(repo: Path, tmp_path: Path) -> None:
    shas = _shas(repo)
    gh = _fake_gh(tmp_path, {"mergeCommit": {"oid": shas["c4"]}, "title": "fix: four"})

    result = _run(repo, "v1.1.0", "778", env={"BRAINLAYER_GH_BIN": str(gh)})

    assert result.returncode == 1
    assert "| #778 |" in result.stdout
    assert "NOT IN" in result.stdout


def test_unmerged_pr_is_not_in(repo: Path, tmp_path: Path) -> None:
    gh = _fake_gh(tmp_path, {"mergeCommit": None, "title": "fix: still open"})

    result = _run(repo, "v1.1.0", "#999", env={"BRAINLAYER_GH_BIN": str(gh)})

    assert result.returncode == 1
    assert "not merged" in result.stdout
    assert "NOT IN" in result.stdout


def test_gh_failure_fails_the_gate_instead_of_passing(repo: Path, tmp_path: Path) -> None:
    gh = _write_exe(tmp_path / "gh", '#!/usr/bin/env bash\necho "boom" >&2\nexit 1\n')

    result = _run(repo, "v1.1.0", "#778", env={"BRAINLAYER_GH_BIN": str(gh)})

    assert result.returncode == 1
    assert "NOT IN" in result.stdout


def test_a_long_all_digit_claim_is_a_sha_not_a_pr_number(repo: Path, tmp_path: Path) -> None:
    """`gh pr view 1234567` would answer for an unrelated PR; a 7-digit claim is a SHA."""
    gh = _fake_gh(tmp_path, {"mergeCommit": {"oid": _shas(repo)["c3"]}, "title": "wrong answer"})

    result = _run(repo, "v1.1.0", "1234567", env={"BRAINLAYER_GH_BIN": str(gh)})

    assert result.returncode == 1
    assert "not in this repo" in result.stdout
    assert "wrong answer" not in result.stdout


# --- "what it actually carries" -------------------------------------------------------------


def test_carries_section_scopes_from_the_previous_FULL_release(repo: Path) -> None:
    """v1.0.1-rc1 sits between v1.0.0 and v1.1.0; a pre-release must not start the range."""
    result = _run(repo, "v1.1.0", _shas(repo)["c3"])

    assert "v1.0.0..v1.1.0" in result.stdout
    assert "v1.0.1-rc1" not in result.stdout
    assert "fix: two" in result.stdout
    assert "fix: three" in result.stdout
    assert "feat: one" not in result.stdout


def test_missing_previous_release_tag_is_loud_not_silent(repo: Path) -> None:
    result = _run(repo, "v1.0.0")

    assert result.returncode == 0
    assert "WARNING" in result.stdout
    assert "no previous full-release tag" in result.stdout


# --- --assert-module (the installed-keg half) -----------------------------------------------


def test_assert_module_passes_when_the_installed_keg_can_import_it(repo: Path, tmp_path: Path) -> None:
    keg_python = _write_exe(tmp_path / "python", "#!/usr/bin/env bash\nexit 0\n")

    result = _run(
        repo,
        "v1.1.0",
        _shas(repo)["c3"],
        "--assert-module",
        "brainlayer.index_watchdog",
        env={"BRAINLAYER_KEG_PYTHON": str(keg_python)},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "import brainlayer.index_watchdog | OK" in result.stdout


def test_assert_module_failure_fails_the_gate_even_when_every_claim_is_in(repo: Path, tmp_path: Path) -> None:
    keg_python = _write_exe(
        tmp_path / "python",
        "#!/usr/bin/env bash\necho \"ModuleNotFoundError: No module named 'brainlayer.index_watchdog'\" >&2\nexit 1\n",
    )

    result = _run(
        repo,
        "v1.1.0",
        _shas(repo)["c3"],
        "--assert-module",
        "brainlayer.index_watchdog",
        env={"BRAINLAYER_KEG_PYTHON": str(keg_python)},
    )

    assert result.returncode == 1
    assert "| IN |" in result.stdout
    assert "import brainlayer.index_watchdog | FAILED" in result.stdout
    assert "ModuleNotFoundError" in result.stdout


def test_missing_keg_python_fails_closed(repo: Path, tmp_path: Path) -> None:
    result = _run(
        repo,
        "v1.1.0",
        _shas(repo)["c3"],
        "--assert-module",
        "brainlayer",
        env={"BRAINLAYER_KEG_PYTHON": str(tmp_path / "nope" / "python")},
    )

    assert result.returncode == 1
    assert "FAILED" in result.stdout


# --- the wiring (AGENTS.md is where the rule lives, so a quiet drop must fail) ---------------


def test_the_script_is_executable() -> None:
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} must be runnable directly"


def test_agents_md_requires_the_gate_in_every_release_receipt() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text()

    assert "scripts/release_tag_contains.py" in agents
    assert "deploy brief may only name a fix that passed that gate" in agents
    assert "--assert-module" in agents
