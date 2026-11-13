"""Known-agent census: a listed agent runs and writes where BrainLayer does not watch => ingest amber."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path

from brainlayer import health_facts
from brainlayer.agent_census import running_agents, unindexed_agents
from brainlayer.health_check import HealthCheckResult, HealthIssue
from brainlayer.health_verdict import AMBER, GREEN, UNOBSERVED, Facts, evaluate
from brainlayer.watcher import default_watch_roots

NOW = time.time()


def _roots(home: Path) -> list[Path]:
    return [root.resolved_path for root in default_watch_roots(home)]


def _write(path: Path, age_s: float = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
    os.utime(path, (NOW - age_s, NOW - age_s))


def test_gemini_cli_chats_outside_the_watched_sessions_root_are_reported(tmp_path):
    _write(tmp_path / ".gemini" / "tmp" / "3a50" / "chats" / "session-2026-11-13T09-00-ab12.json")
    ps = "4242 node /opt/homebrew/bin/gemini --yolo\n"
    assert unindexed_agents(ps, home=tmp_path, watch_roots=_roots(tmp_path), now=NOW) == [
        f"gemini: {tmp_path / '.gemini' / 'tmp'}"
    ]


def test_an_agent_writing_under_a_watched_root_is_not_reported(tmp_path):
    _write(tmp_path / ".claude" / "projects" / "-Users-example-proj" / "s.jsonl")
    _write(tmp_path / ".codex" / "sessions" / "2026" / "11" / "13" / "rollout.jsonl")
    ps = "1 /home/example/.local/bin/claude\n2 /opt/homebrew/bin/codex exec\n"
    assert unindexed_agents(ps, home=tmp_path, watch_roots=_roots(tmp_path), now=NOW) == []


def test_an_unlisted_root_is_reported_only_while_its_agent_runs_and_writes(tmp_path):
    _write(tmp_path / ".local" / "share" / "opencode" / "storage" / "session" / "ses_1.json")
    roots = _roots(tmp_path)
    assert unindexed_agents("9 /usr/local/bin/opencode\n", home=tmp_path, watch_roots=roots, now=NOW) == [
        f"opencode: {tmp_path / '.local' / 'share' / 'opencode' / 'storage'}"
    ]
    assert unindexed_agents("9 /bin/zsh\n", home=tmp_path, watch_roots=roots, now=NOW) == []
    _write(tmp_path / ".local" / "share" / "opencode" / "storage" / "session" / "ses_1.json", age_s=3 * 86400)
    assert unindexed_agents("9 /usr/local/bin/opencode\n", home=tmp_path, watch_roots=roots, now=NOW) == []


def test_process_matching_uses_argv0_or_argv1_basenames_only(tmp_path):
    ps = (
        "1 /Applications/Codex.app/Contents/MacOS/Codex\n"
        "2 /usr/bin/grep gemini\n"
        "3 /usr/bin/python3 /tmp/x.py --label gemini\n"
        "4 node /home/example/.nvm/versions/node/v22/bin/gemini\n"
    )
    assert [agent.name for agent in running_agents(ps)] == ["gemini"]


def _facts(**kwargs) -> Facts:
    now = datetime(2026, 11, 13, 9, 0, tzinfo=UTC)
    return Facts(now=now, watcher_updated_at=now, **kwargs)


def test_the_census_makes_a_fresh_ingest_check_amber_and_names_the_agent():
    found = ("gemini: /home/example/.gemini/tmp",)
    ingest = evaluate(_facts(unindexed_agents=found)).check("ingest")
    assert ingest.state == AMBER
    assert "an agent BrainLayer does not index is running: gemini" in ingest.note
    assert ingest.to_dict()["unindexed_agents"] == list(found)


def test_an_empty_or_unobserved_census_leaves_the_verdict_unchanged():
    baseline = evaluate(_facts()).check("ingest")
    for census in ((), UNOBSERVED):
        ingest = evaluate(_facts(unindexed_agents=census)).check("ingest")
        assert (ingest.state, ingest.note) == (baseline.state, baseline.note) == (GREEN, "")
    assert evaluate(_facts(unindexed_agents=())).check("ingest").to_dict()["unindexed_agents"] == []
    assert evaluate(_facts()).check("ingest").to_dict()["unindexed_agents"] is None


def test_the_census_issue_feeds_ingest_context_and_its_code():
    result = HealthCheckResult(
        checked_at="2026-11-13T09:00:00+00:00",
        ok=True,
        issues=[HealthIssue("unindexed_agent_running", "warning", "an agent BrainLayer does not index is running")],
        unindexed_agents=["gemini: /home/example/.gemini/tmp"],
    )
    blind, notes, context = health_facts.classify_issues(result)
    assert (blind, notes) == ((), ())
    assert context["ingest"].startswith("an agent BrainLayer does not index is running")
    assert health_facts.codes_by_check(result) == {"ingest": ("unindexed_agent_running",)}
