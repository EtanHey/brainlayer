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


def _flat(output: str) -> str:
    """Token-safe: rich may wrap mid-path, so rejoin without inserting whitespace."""
    return output.replace("\n", "")


def _prose(output: str) -> str:
    """Prose-safe: rich wraps between words, so collapse every run of whitespace to one space."""
    return " ".join(output.split())


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
    assert BRAINLAYER_INGEST_DENYLIST_ENV in _flat(result.output)
    assert "~/.claude/projects/**/wf_*/**" in _flat(result.output)
    assert "~/.claude/projects/*/**/subagents/**" in _flat(result.output)
    assert "~/.cursor/**/agent-transcripts/**" in _flat(result.output)
    # A deployment-scoped pattern is legitimate and must not be named as overreach.
    assert "matchmat" not in _flat(result.output)


def test_setup_is_silent_only_when_no_override_exists_at_all(monkeypatch, tmp_path):
    """The ONLY quiet case is no override anywhere.

    Round 0 of this file asserted that a deployment-scoped override was 'quiet because it matches
    the class rule'. That was wrong and the review caught it: `is_denylisted` disables the whole
    attribution branch when the env var is present at ALL, so a matchmat-only override still stops
    excluding brain-worker. Quiet must therefore mean 'no override', never 'a harmless override'.
    """
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)

    result = _invoke_setup(monkeypatch, tmp_path)

    assert result.exit_code == 0, result.output
    assert BRAINLAYER_INGEST_DENYLIST_ENV not in _flat(result.output)


def test_a_deployment_scoped_override_still_reports_that_it_replaces_the_class_rule(monkeypatch, tmp_path):
    monkeypatch.setenv(BRAINLAYER_INGEST_DENYLIST_ENV, "~/.claude/projects/*matchmat*/**")

    result = _invoke_setup(monkeypatch, tmp_path)

    assert result.exit_code == 0, result.output
    assert "REPLACES the ingest class rule" in _prose(result.output)
    assert "brain-worker" in _prose(result.output)
    # ...but it is not an over-exclusion, so no pattern is named as overreaching.
    assert "overreaching pattern" not in _prose(result.output)


# --- round 1 review: the warning must read the env FILE setup resolved, not only os.environ ---


def _env_file(tmp_path, value: str):
    target = tmp_path / "brainlayer.env"
    target.write_text(f'# comment\nBRAINLAYER_ENRICH_RATE="5.0"\nBRAINLAYER_INGEST_DENYLIST="{value}"\n')
    return target


def test_env_file_patterns_are_read_from_the_resolved_file(tmp_path):
    from brainlayer.ingest_denylist import env_file_denylist_patterns

    assert env_file_denylist_patterns(_env_file(tmp_path, DEPLOYED_BLANKET)) == (
        "~/.claude/projects/*/**/subagents/**",
        "~/.claude/projects/**/wf_*/**",
        "~/.cursor/**/agent-transcripts/**",
        "~/.claude/projects/*matchmat*/**",
        "~/Gits/matchmat/**",
    )


def test_env_file_without_the_key_reads_as_unset_not_empty(tmp_path):
    from brainlayer.ingest_denylist import env_file_denylist_patterns

    target = tmp_path / "brainlayer.env"
    target.write_text('BRAINLAYER_ENRICH_RATE="5.0"\n')

    assert env_file_denylist_patterns(target) is None
    assert env_file_denylist_patterns(tmp_path / "missing.env") is None


def test_overreach_accepts_explicit_patterns_independent_of_process_env(monkeypatch):
    from brainlayer.ingest_denylist import env_denylist_overreach

    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)

    findings = env_denylist_overreach(patterns=("~/.claude/projects/**/wf_*/**",))

    assert [f.pattern for f in findings] == ["~/.claude/projects/**/wf_*/**"]


def test_setup_warns_about_a_blanket_in_the_env_file_with_an_empty_process_env(monkeypatch, tmp_path):
    """The blanket lived in ~/.config/brainlayer/brainlayer.env, never in a fresh shell's environ."""
    from typer.testing import CliRunner

    from brainlayer.cli import app

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    target = _env_file(tmp_path, DEPLOYED_BLANKET)

    result = CliRunner().invoke(app, ["setup", "--no-migrate-mcp", "--no-launchd", "--env-file", str(target)])

    assert result.exit_code == 0, result.output
    assert "~/.claude/projects/**/wf_*/**" in _flat(result.output)
    assert "~/.claude/projects/*/**/subagents/**" in _flat(result.output)
    assert str(target) in _flat(result.output)
    assert "matchmat" not in _flat(result.output)


def test_setup_says_an_override_replaces_the_class_rule_even_when_no_pattern_overreaches(monkeypatch, tmp_path):
    """Quiet-on-overreach must not read as a clean bill: any override disables memory-reader exclusion."""
    from typer.testing import CliRunner

    from brainlayer.cli import app

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    target = _env_file(tmp_path, "~/.claude/projects/*matchmat*/**")

    result = CliRunner().invoke(app, ["setup", "--no-migrate-mcp", "--no-launchd", "--env-file", str(target)])

    assert result.exit_code == 0, result.output
    assert "brain-worker" in _prose(result.output)
    assert "replaces" in _prose(result.output).lower()


def test_every_reported_finding_is_one_the_matcher_actually_excludes(monkeypatch):
    """Lock probe<->matcher agreement: a reported kept_example must really be denylisted."""
    from brainlayer.ingest_denylist import env_denylist_overreach, is_denylisted

    monkeypatch.setenv(BRAINLAYER_INGEST_DENYLIST_ENV, DEPLOYED_BLANKET)

    findings = env_denylist_overreach()
    assert findings
    for finding in findings:
        assert is_denylisted(finding.kept_example), finding.pattern


def test_absolute_non_home_class_blanket_is_still_reported(monkeypatch):
    """A blanket anchored outside $HOME overrides ingest just as wholesale."""
    from brainlayer.ingest_denylist import env_denylist_overreach

    pattern = "/var/data/agents/.claude/projects/*/**/subagents/**"
    monkeypatch.setenv(BRAINLAYER_INGEST_DENYLIST_ENV, pattern)

    assert [f.pattern for f in env_denylist_overreach()] == [pattern]
