"""Maintenance must refuse to run code that is not the merged code.

RED reproduces the 2026-08-05 deploy gap: `com.brainlayer.maintenance-nightly` runs
`~/Gits/brainlayer/.venv/bin/python -m brainlayer.maintenance`, and that venv is an
EDITABLE install — `import brainlayer` resolves to the working tree, not a built
artifact. On 2026-08-05 the tree was 8 commits behind origin/main, so #650's pause-
sentinel fix was merged on GitHub and absent on disk. The job would have run stale
code and reported success.

The property: it either runs the merged code, or it fails LOUDLY. It must never
silently run stale code and report ok.
"""

import pytest

from brainlayer import maintenance


def test_aborts_when_worktree_is_behind_merged_head(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(maintenance, "_git_head_sha", lambda root: "aaaaaaa")
    monkeypatch.setattr(maintenance, "_git_merged_head_sha", lambda root: "bbbbbbb")

    with pytest.raises(maintenance.MaintenanceAbort) as exc:
        maintenance._assert_running_merged_code(maintenance.Path("/repo"))

    msg = str(exc.value)
    assert "aaaaaaa" in msg and "bbbbbbb" in msg, "abort must name both SHAs"
    assert "stale" in msg.lower()


def test_proceeds_when_worktree_matches_merged_head(monkeypatch):
    monkeypatch.setattr(maintenance, "_git_head_sha", lambda root: "deadbee")
    monkeypatch.setattr(maintenance, "_git_merged_head_sha", lambda root: "deadbee")

    maintenance._assert_running_merged_code(maintenance.Path("/repo"))


def test_unverifiable_remote_does_not_silently_pass_when_drifted(monkeypatch):
    """A failed fetch must not become an excuse to run stale code."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(maintenance, "_git_head_sha", lambda root: "aaaaaaa")
    monkeypatch.setattr(maintenance, "_git_merged_head_sha", lambda root: None)

    # Unknown remote state: cannot prove freshness, so it must not claim success.
    with pytest.raises(maintenance.MaintenanceAbort):
        maintenance._assert_running_merged_code(maintenance.Path("/repo"), strict=True)


def test_explicit_override_allows_drift(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(maintenance, "_git_head_sha", lambda root: "aaaaaaa")
    monkeypatch.setattr(maintenance, "_git_merged_head_sha", lambda root: "bbbbbbb")
    monkeypatch.setenv("BRAINLAYER_MAINTENANCE_ALLOW_STALE", "1")

    maintenance._assert_running_merged_code(maintenance.Path("/repo"))


def test_guard_is_inert_under_pytest(monkeypatch):
    """Production keeps the guard; pytest opts out, matching this repo's convention.

    Without this, every run_maintenance test aborts as STALE simply because a feature
    branch differs from origin/main -- which is what a feature branch IS.
    """
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "x")
    monkeypatch.setattr(maintenance, "_git_head_sha", lambda root: "aaaaaaa")
    monkeypatch.setattr(maintenance, "_git_merged_head_sha", lambda root: "bbbbbbb")

    maintenance._assert_running_merged_code(maintenance.Path("/repo"))


def test_guard_fires_when_not_under_pytest(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(maintenance, "_git_head_sha", lambda root: "aaaaaaa")
    monkeypatch.setattr(maintenance, "_git_merged_head_sha", lambda root: "bbbbbbb")

    with pytest.raises(maintenance.MaintenanceAbort):
        maintenance._assert_running_merged_code(maintenance.Path("/repo"))
