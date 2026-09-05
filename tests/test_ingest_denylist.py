import json
from pathlib import Path

import brainlayer.ingest_denylist as denylist
from brainlayer.ingest_denylist import BRAINLAYER_INGEST_DENYLIST_ENV, is_denylisted


def _write_subagent(path: Path, attribution: str | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "type": "user",
            "agentId": path.stem.removeprefix("agent-"),
            "message": {"role": "user", "content": "Investigate the assigned task."},
        }
    ]
    if attribution is not None:
        entries.append(
            {
                "type": "assistant",
                "attributionAgent": attribution,
                "message": {"role": "assistant", "content": "Attributed worker response."},
            }
        )
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    return path


def test_default_policy_allows_normal_provider_sessions(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "process-home"))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    backup_home = tmp_path / "backup-home"

    assert not is_denylisted(backup_home / ".codex" / "sessions" / "worker.jsonl")
    assert not is_denylisted(backup_home / ".gemini" / "sessions" / "worker.jsonl")
    assert not is_denylisted(
        backup_home / ".cursor" / "projects" / "repo" / "agent-transcripts" / "session" / "worker.jsonl"
    )
    assert not is_denylisted(backup_home / ".claude" / "projects" / "proj" / "direct-session.jsonl")


def test_default_policy_allows_ordinary_claude_subagents(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    projects = tmp_path / ".claude" / "projects" / "-Users-test-Gits-brainlayer" / "session-uuid"

    explore = _write_subagent(projects / "subagents" / "agent-explore.jsonl", "Explore")
    general = _write_subagent(projects / "subagents" / "agent-general.jsonl", "general-purpose")

    assert not is_denylisted(explore)
    assert not is_denylisted(general)


def test_default_policy_excludes_exact_brain_worker_but_keeps_raw_jsonl(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    worker = _write_subagent(
        tmp_path
        / ".claude"
        / "projects"
        / "-Users-test-Gits-brainlayer"
        / "session-uuid"
        / "subagents"
        / "agent-brain.jsonl",
        "brain-worker",
    )

    assert is_denylisted(worker)
    assert worker.exists()


def test_default_policy_excludes_all_memory_reader_attributions(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)

    for attribution in ("brain-worker", "session-miner", "weave"):
        worker = _write_subagent(
            tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / f"agent-{attribution}.jsonl",
            attribution,
        )
        assert is_denylisted(worker)
        assert worker.exists()


def test_default_policy_keeps_non_recon_workflow_subagent(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    workflow = _write_subagent(
        tmp_path
        / ".claude"
        / "projects"
        / "-Users-test-Gits-brainlayer"
        / "session-uuid"
        / "subagents"
        / "workflows"
        / "wf_c83ce37d"
        / "agent-workflow.jsonl",
        "workflow-subagent",
    )

    assert not is_denylisted(workflow)
    assert workflow.exists()


def test_subagent_attribution_becomes_known_after_append(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    worker = _write_subagent(
        tmp_path / ".claude" / "projects" / "proj" / "session-uuid" / "subagents" / "agent-new.jsonl",
        None,
    )

    denylist._SUBAGENT_ATTRIBUTION_CACHE.clear()
    assert denylist._claude_subagent_attribution(worker) is None

    with worker.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "assistant",
                    "attributionAgent": "general-purpose",
                    "message": {"role": "assistant", "content": "Identity is now available."},
                }
            )
            + "\n"
        )

    assert denylist._claude_subagent_attribution(worker) == "general-purpose"


def test_historical_policy_preserves_unverifiable_subagents(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    missing_worker = (
        tmp_path / ".claude" / "historical" / "projects" / "proj" / "session" / "subagents" / "agent-missing.jsonl"
    )

    assert is_denylisted(missing_worker)
    assert not is_denylisted(missing_worker, unknown_subagent_is_denylisted=False)


def test_subagent_attribution_skips_non_object_json_values(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    worker = tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent.jsonl"
    worker.parent.mkdir(parents=True)
    worker.write_text(
        "null\n[]\n" + json.dumps({"attributionAgent": "general-purpose"}) + "\n",
        encoding="utf-8",
    )

    assert denylist._claude_subagent_attribution(worker) == "general-purpose"


def test_subagent_attribution_is_found_after_legacy_scan_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    worker = tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent.jsonl"
    worker.parent.mkdir(parents=True)
    unattributed = json.dumps({"type": "progress"})
    worker.write_text(
        "\n".join([unattributed] * 300 + [json.dumps({"attributionAgent": "general-purpose"})]) + "\n",
        encoding="utf-8",
    )

    assert denylist._claude_subagent_attribution(worker) == "general-purpose"


def test_unattributed_appends_are_scanned_incrementally(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    denylist._SUBAGENT_ATTRIBUTION_CACHE.clear()
    worker = tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent.jsonl"
    worker.parent.mkdir(parents=True)
    progress = json.dumps({"type": "progress"})
    worker.write_text("\n".join([progress] * 200) + "\n", encoding="utf-8")
    real_loads = json.loads
    calls = 0

    def counting_loads(value):
        nonlocal calls
        calls += 1
        return real_loads(value)

    monkeypatch.setattr(denylist.json, "loads", counting_loads)

    assert denylist._claude_subagent_attribution(worker) is None
    with worker.open("a", encoding="utf-8") as handle:
        handle.write(progress + "\n")
    assert denylist._claude_subagent_attribution(worker) is None
    with worker.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"attributionAgent": "general-purpose"}) + "\n")
    assert denylist._claude_subagent_attribution(worker) == "general-purpose"
    assert calls == 202


def test_cached_attribution_is_invalidated_when_same_inode_is_rewritten_larger(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    denylist._SUBAGENT_ATTRIBUTION_CACHE.clear()
    worker = _write_subagent(
        tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent.jsonl",
        "general-purpose",
    )
    assert denylist._claude_subagent_attribution(worker) == "general-purpose"
    initial_stat = worker.stat()

    worker.write_text(
        json.dumps(
            {
                "type": "assistant",
                "attributionAgent": "brain-worker",
                "message": {"role": "assistant", "content": "rewritten" + "x" * 1_000},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rewritten_stat = worker.stat()

    assert rewritten_stat.st_ino == initial_stat.st_ino
    assert rewritten_stat.st_size > initial_stat.st_size
    assert denylist._claude_subagent_attribution(worker) == "brain-worker"


def test_explicit_empty_override_disables_default_subagent_policy(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(BRAINLAYER_INGEST_DENYLIST_ENV, "")
    worker = _write_subagent(
        tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent.jsonl",
        "brain-worker",
    )

    assert not is_denylisted(worker)


def test_explicit_environment_override_can_deny_an_otherwise_allowed_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(BRAINLAYER_INGEST_DENYLIST_ENV, "~/.codex/sessions/**")

    assert is_denylisted(tmp_path / ".codex" / "sessions" / "worker.jsonl")
    assert not is_denylisted(tmp_path / ".gemini" / "sessions" / "worker.jsonl")


def test_configured_pattern_match_is_cached_across_polls(monkeypatch, tmp_path):
    """The glob match is a pure function of (path, patterns, home) -- pay for it once, not every poll.

    Measured on the M4 (2026-09-05) with the deployed 5-pattern BRAINLAYER_INGEST_DENYLIST:
    one sweep over the 12,125-file corpus costs 0.35s CPU, every 30s poll, forever --
    60,625 glob expansions and 258,000 recursive `_match_parts` calls whose answers never
    change while the patterns do not. The per-poll memo added in #759 only stopped the
    3x-per-poll re-evaluation; the sweep itself still ran on every poll. A changed denylist
    must still be picked up promptly, which is why the patterns are part of the cache key.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(
        BRAINLAYER_INGEST_DENYLIST_ENV,
        "~/.claude/projects/*/**/subagents/**,~/.cursor/**/agent-transcripts/**",
    )
    denylist.clear_pattern_match_cache()
    path = tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent-1.jsonl"

    calls: list[int] = []
    real = denylist._match_parts

    def counting(path_parts, pattern_parts):
        calls.append(1)
        return real(path_parts, pattern_parts)

    monkeypatch.setattr(denylist, "_match_parts", counting)

    assert is_denylisted(path) is True
    first = len(calls)
    assert first > 0, "the first evaluation must actually run the glob match"

    assert is_denylisted(path) is True
    assert len(calls) == first, "the second evaluation of the same path must not re-run the glob match"

    # A changed denylist is picked up on the very next evaluation -- no restart, no TTL.
    monkeypatch.setenv(BRAINLAYER_INGEST_DENYLIST_ENV, "~/nowhere/**")
    assert is_denylisted(path) is False
    assert len(calls) > first, "new patterns must be evaluated, not served from the old cache entry"


def test_directory_is_denylisted_only_when_a_subtree_pattern_covers_all_descendants(monkeypatch, tmp_path):
    """A directory may be skipped wholesale only if every file under it would be denied.

    On the M4 the deployed denylist ends every pattern in `/**`, and `~/.cursor/projects`
    alone holds 5,086 directories, 4,315 of them under the 111 `agent-transcripts` dirs the
    pattern denies. Walking them costs ~1s of kernel time per poll on the efficiency cores
    to discover files that are then thrown away. A pattern that ends in `**` matches the
    directory AND everything below it, so the walk can stop there; any other pattern says
    nothing about descendants and must not prune.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    denylist.clear_pattern_match_cache()
    transcripts = tmp_path / ".cursor" / "projects" / "repo" / "agent-transcripts"
    weird_dir = tmp_path / ".codex" / "sessions" / "looks-like-a-file.jsonl"

    monkeypatch.setenv(BRAINLAYER_INGEST_DENYLIST_ENV, "~/.cursor/**/agent-transcripts/**,~/.codex/sessions/*.jsonl")
    assert denylist.is_directory_denylisted(transcripts) is True
    assert denylist.is_directory_denylisted(transcripts / "session-1") is True, "descendants are covered too"
    assert denylist.is_directory_denylisted(tmp_path / ".cursor" / "projects" / "repo") is False, "parent is not"
    assert denylist.is_directory_denylisted(weird_dir) is False, (
        "a non-`**` pattern matching a directory's name says nothing about the files inside it"
    )


def test_directory_is_never_denylisted_under_the_default_subagent_policy(monkeypatch, tmp_path):
    """Without configured patterns, subagent files are judged one by one (attribution), so no
    directory can be skipped -- ordinary subagents ingest, brain-workers do not, same dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    denylist.clear_pattern_match_cache()
    subagents = tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents"
    assert denylist.is_directory_denylisted(subagents) is False
