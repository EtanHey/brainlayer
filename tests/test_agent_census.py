"""Known-agent census: a listed agent writes, during its current run, where BrainLayer does not watch => ingest amber."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brainlayer import agent_census, health_facts
from brainlayer.agent_census import unindexed_agents
from brainlayer.health_check import HealthCheckResult, HealthIssue
from brainlayer.health_verdict import AMBER, GREEN, UNOBSERVED, Facts, evaluate
from brainlayer.watcher import default_watch_roots

NOW = time.time()
GEMINI = "4242 node /opt/homebrew/bin/gemini --yolo\n"


def _roots(home: Path) -> list[Path]:
    return [root.resolved_path for root in default_watch_roots(home)]


def _write(path: Path, age_s: float = 60) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
    os.utime(path, (NOW - age_s, NOW - age_s))
    return path


def _started(age_s: float):
    return lambda pids: {pid: NOW - age_s for pid in pids}


def _chat(home: Path, age_s: float = 60) -> Path:
    return _write(home / ".gemini" / "tmp" / "3a50" / "chats" / "session-2026-11-13T09-00-ab12.json", age_s)


def _r(census) -> tuple[list[str], list[str]]:
    return census.found, census.incomplete


def _census(home: Path, ps: str, started_age_s: float = 600):
    return unindexed_agents(ps, home=home, watch_roots=_roots(home), process_starts=_started(started_age_s))


# (1) An agent is its executable or the script an interpreter runs, never an argument.


def test_grep_gemini_alone_is_not_a_running_gemini(tmp_path):
    _chat(tmp_path)
    ps = "123 /usr/bin/grep gemini\n"
    assert _r(unindexed_agents(ps, home=tmp_path, watch_roots=_roots(tmp_path))) == ([], [])


def test_brainlayer_and_other_commands_naming_an_agent_never_count(tmp_path):
    _chat(tmp_path)
    ps = (
        "1 /opt/homebrew/opt/brainlayer/libexec/venv/bin/brainlayer gemini\n"
        "2 /usr/bin/python3 -m brainlayer.cli search gemini\n"
        "3 /usr/bin/python3 /srv/tools/report.py --agent gemini\n"
        "4 /Applications/Codex.app/Contents/MacOS/Codex\n"
        "5 /bin/zsh -c gemini --help\n"
    )
    assert _r(unindexed_agents(ps, home=tmp_path, watch_roots=_roots(tmp_path))) == ([], [])


def test_inline_code_and_modules_are_never_an_entry_script(tmp_path):
    _chat(tmp_path)
    ps = (
        "1 /usr/bin/python3 -c gemini\n"
        "2 /usr/bin/python3 -uc gemini\n"
        "3 /usr/bin/python3 -m gemini\n"
        "4 /usr/local/bin/node -e gemini\n"
        "5 node --eval=gemini\n"
        "6 node -p gemini\n"
        "7 /opt/homebrew/bin/bun -e gemini\n"
        "8 bun x gemini\n"
        "9 /opt/homebrew/bin/deno eval gemini\n"
    )
    assert agent_census.running_agents(ps) == {}
    assert _r(unindexed_agents(ps, home=tmp_path, watch_roots=_roots(tmp_path))) == ([], [])


def test_a_real_script_after_interpreter_options_is_matched():
    ps = (
        "20 node -r tsx/cjs /opt/homebrew/bin/gemini\n"
        "21 /usr/bin/python3 -W ignore -X dev /srv/agents/gemini\n"
        "22 bun run /home/example/.bun/bin/gemini\n"
        "23 deno run --allow-read /srv/agents/gemini\n"
        "24 node -- /usr/lib/node_modules/@google/gemini-cli/dist/index.js\n"
    )
    assert agent_census.running_agents(ps) == {"gemini": [20, 21, 22, 23, 24]}


def test_an_agent_binary_or_its_interpreter_entry_script_is_matched():
    ps = (
        "10 node /home/example/.nvm/versions/node/v22/bin/gemini\n"
        "11 node --no-warnings /usr/lib/node_modules/@google/gemini-cli/dist/index.js\n"
        "12 /home/example/.local/bin/claude --resume\n"
        "13 /opt/homebrew/bin/codex exec\n"
    )
    assert agent_census.running_agents(ps) == {"claude": [12], "codex": [13], "gemini": [10, 11]}


# (2) Only a transcript written during a live instance's current run counts.


def test_a_gemini_chat_written_during_the_current_run_is_reported(tmp_path):
    _chat(tmp_path, age_s=60)
    assert _r(_census(tmp_path, GEMINI, started_age_s=600)) == ([f"gemini: {tmp_path / '.gemini' / 'tmp'}"], [])


def test_a_chat_from_a_stopped_earlier_session_does_not_count(tmp_path):
    _chat(tmp_path, age_s=3600)  # written an hour ago by a session that ended
    _write(tmp_path / ".gemini" / "sessions" / "now.jsonl", age_s=30)  # today's run writes only to the watched root
    assert _r(_census(tmp_path, GEMINI, started_age_s=600)) == ([], [])


def test_an_unrelated_recent_file_under_the_root_is_not_a_transcript(tmp_path):
    _write(tmp_path / ".gemini" / "tmp" / "3a50" / "notes.txt", age_s=30)
    _write(tmp_path / ".gemini" / "tmp" / "3a50" / "chats" / "logs.json", age_s=30)
    assert _r(_census(tmp_path, GEMINI, started_age_s=600)) == ([], [])


def test_an_agent_writing_under_a_watched_root_is_not_reported(tmp_path):
    _write(tmp_path / ".claude" / "projects" / "-Users-example-proj" / "s.jsonl")
    _write(tmp_path / ".codex" / "sessions" / "2026" / "11" / "13" / "rollout-x.jsonl")
    ps = "1 /home/example/.local/bin/claude\n2 /opt/homebrew/bin/codex exec\n"
    assert _r(_census(tmp_path, ps)) == ([], [])


def test_parse_etime_reads_every_ps_shape():
    assert [agent_census.parse_etime(v) for v in ("00:07", "01:02:03", "2-00:00:01", "x", "1:2:3:4")] == [
        7,
        3723,
        172801,
        None,
        None,
    ]


# (4) A cut-short scan, a read error or an unknown start time is not measured, never "none found".


def test_a_scan_cut_short_by_the_entry_cap_is_not_measured(tmp_path, monkeypatch):
    for i in range(30):
        _write(tmp_path / ".gemini" / "tmp" / f"h{i:02}" / "other.txt")
    monkeypatch.setattr(agent_census, "MAX_ENTRIES", 10)
    assert _r(_census(tmp_path, GEMINI)) == ([], ["gemini"])


def test_a_transcript_below_the_depth_cap_is_not_measured(tmp_path, monkeypatch):
    _write(tmp_path / ".gemini" / "tmp" / "a" / "b" / "c" / "chats" / "session-1.json")
    monkeypatch.setattr(agent_census, "MAX_DEPTH", 2)
    assert _r(_census(tmp_path, GEMINI)) == ([], ["gemini"])


def test_an_unreadable_directory_is_not_measured(tmp_path):
    locked = tmp_path / ".gemini" / "tmp" / "locked"
    _write(locked / "chats" / "session-1.json")
    locked.chmod(0o000)
    try:
        assert _r(_census(tmp_path, GEMINI)) == ([], ["gemini"])
    finally:
        locked.chmod(0o755)


def test_a_running_agent_without_a_start_time_is_not_measured(tmp_path):
    _chat(tmp_path)
    census = unindexed_agents(GEMINI, home=tmp_path, watch_roots=_roots(tmp_path), process_starts=lambda p: None)
    assert _r(census) == ([], ["gemini"])


def test_ps_process_starts_parses_ps_and_refuses_a_failed_call():
    run = lambda args: type("R", (), {"returncode": 0, "stdout": "4242   01:40\n  77 2-00:00:00\n"})()  # noqa: E731
    assert agent_census.ps_process_starts(run, now=lambda: 1000.0)([4242, 77]) == {4242: 900.0, 77: 1000.0 - 172800}
    failed = lambda args: type("R", (), {"returncode": 2, "stdout": ""})()  # noqa: E731
    assert agent_census.ps_process_starts(failed)([1]) is None


# (3) A census that was not measured holds the last measured one; only a measured census clears it.

T0 = datetime(2026, 11, 13, 9, 0, tzinfo=UTC)


def _tick(prior, minutes, census):
    now = T0 + timedelta(minutes=minutes)
    return evaluate(Facts(now=now, watcher_updated_at=now, unindexed_agents=census), prior)


def test_an_unmeasured_census_never_clears_a_confirmed_one():
    found = ["gemini: /home/example/.gemini/tmp"]
    verdict = _tick(None, 0, tuple(found))
    assert verdict.check("ingest").state == AMBER
    for minute in (5, 10, 15, 20):  # ps keeps failing; the watcher heartbeat stays fresh
        verdict = _tick(verdict.state, minute, UNOBSERVED)
        ingest = verdict.check("ingest")
        assert ingest.state == AMBER, minute
        assert ingest.to_dict()["unindexed_agents"] == found
        assert "census not measured this check" in ingest.note
    for minute in (25, 30):  # a measured, empty census clears it through the normal recovery
        verdict = _tick(verdict.state, minute, ())
    assert verdict.check("ingest").state == GREEN


def test_an_empty_or_never_measured_census_leaves_the_verdict_unchanged():
    baseline = evaluate(Facts(now=T0, watcher_updated_at=T0)).check("ingest")
    for census in ((), UNOBSERVED):
        ingest = evaluate(Facts(now=T0, watcher_updated_at=T0, unindexed_agents=census)).check("ingest")
        assert (ingest.state, ingest.note) == (baseline.state, baseline.note) == (GREEN, "")
    assert (
        evaluate(Facts(now=T0, watcher_updated_at=T0, unindexed_agents=()))
        .check("ingest")
        .to_dict()["unindexed_agents"]
        == []
    )
    assert baseline.to_dict()["unindexed_agents"] is None


def test_the_census_issues_feed_ingest_context_and_codes():
    result = HealthCheckResult(
        checked_at="2026-11-13T09:00:00+00:00",
        ok=True,
        issues=[
            HealthIssue("unindexed_agent_running", "warning", "an agent BrainLayer does not index is running"),
            HealthIssue("agent_census_incomplete", "warning", "known-agent census not measured this check"),
        ],
        unindexed_agents=["gemini: /home/example/.gemini/tmp"],
    )
    blind, notes, context = health_facts.classify_issues(result)
    assert (blind, notes) == ((), ())
    assert context["ingest"].startswith("an agent BrainLayer does not index is running")
    assert health_facts.codes_by_check(result) == {"ingest": ("unindexed_agent_running", "agent_census_incomplete")}
