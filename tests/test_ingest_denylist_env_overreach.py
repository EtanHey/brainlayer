"""The env override must never silently exclude what the class rule keeps."""

from __future__ import annotations

import pytest

from brainlayer.ingest_denylist import (
    BRAINLAYER_INGEST_DENYLIST_ENV,
    env_denylist_overreach,
)

# The blanket shape found in the deployed ~/.config/brainlayer/brainlayer.env on 2026-09-05.
DEPLOYED_BLANKET = (
    "~/.claude/projects/*/**/subagents/**,"
    "~/.claude/projects/**/wf_*/**,"
    "~/.cursor/**/agent-transcripts/**,"
    "~/.claude/projects/*matchmat*/**,"
    "~/Gits/matchmat/**"
)


def test_unset_env_has_no_overreach(monkeypatch):
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)

    assert env_denylist_overreach() == ()


@pytest.mark.parametrize(
    "pattern",
    [
        "~/.claude/projects/*/**/subagents/**",
        "~/.claude/projects/**/wf_*/**",
        "~/.cursor/**/agent-transcripts/**",
    ],
)
def test_class_blanket_patterns_are_reported_as_overreach(monkeypatch, pattern):
    monkeypatch.setenv(BRAINLAYER_INGEST_DENYLIST_ENV, pattern)

    findings = env_denylist_overreach()

    assert [finding.pattern for finding in findings] == [pattern]
    assert findings[0].kept_example


@pytest.mark.parametrize(
    "pattern",
    [
        "~/.claude/projects/*matchmat*/**",
        "~/Gits/matchmat/**",
    ],
)
def test_deployment_scoped_patterns_are_not_overreach(monkeypatch, pattern):
    monkeypatch.setenv(BRAINLAYER_INGEST_DENYLIST_ENV, pattern)

    assert env_denylist_overreach() == ()


def test_deployed_blanket_reports_exactly_the_class_overreaching_patterns(monkeypatch):
    monkeypatch.setenv(BRAINLAYER_INGEST_DENYLIST_ENV, DEPLOYED_BLANKET)

    assert [finding.pattern for finding in env_denylist_overreach()] == [
        "~/.claude/projects/*/**/subagents/**",
        "~/.claude/projects/**/wf_*/**",
        "~/.cursor/**/agent-transcripts/**",
    ]


def _invoke_setup(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from brainlayer.cli import app

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("sys.platform", "linux")
    return CliRunner().invoke(
        app,
        ["setup", "--no-migrate-mcp", "--no-launchd", "--env-file", str(tmp_path / "brainlayer.env")],
    )


def test_setup_warns_when_env_denylist_is_broader_than_the_class_rule(monkeypatch, tmp_path):
    monkeypatch.setenv(BRAINLAYER_INGEST_DENYLIST_ENV, DEPLOYED_BLANKET)

    result = _invoke_setup(monkeypatch, tmp_path)

    assert result.exit_code == 0, result.output
    assert BRAINLAYER_INGEST_DENYLIST_ENV in result.output
    assert "~/.claude/projects/**/wf_*/**" in result.output
    assert "~/.claude/projects/*/**/subagents/**" in result.output
    assert "~/.cursor/**/agent-transcripts/**" in result.output
    # A deployment-scoped pattern is legitimate and must not be named as overreach.
    assert "matchmat" not in result.output


def test_setup_is_quiet_when_the_env_denylist_matches_the_class_rule(monkeypatch, tmp_path):
    monkeypatch.setenv(BRAINLAYER_INGEST_DENYLIST_ENV, "~/.claude/projects/*matchmat*/**")

    result = _invoke_setup(monkeypatch, tmp_path)

    assert result.exit_code == 0, result.output
    assert BRAINLAYER_INGEST_DENYLIST_ENV not in result.output
