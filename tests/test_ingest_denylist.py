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


def test_default_policy_excludes_workflow_workers_by_path_and_keeps_raw_jsonl(monkeypatch, tmp_path):
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

    assert is_denylisted(workflow)
    assert workflow.exists()


def test_unattributed_subagent_is_deferred_until_identity_is_known(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    worker = _write_subagent(
        tmp_path / ".claude" / "projects" / "proj" / "session-uuid" / "subagents" / "agent-new.jsonl",
        None,
    )

    assert is_denylisted(worker)

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

    assert not is_denylisted(worker)


def test_historical_policy_preserves_unverifiable_subagents(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(BRAINLAYER_INGEST_DENYLIST_ENV, raising=False)
    missing_worker = tmp_path / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent-missing.jsonl"

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

    assert not is_denylisted(worker)


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

    assert not is_denylisted(worker)


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

    assert is_denylisted(worker)
    with worker.open("a", encoding="utf-8") as handle:
        handle.write(progress + "\n")
    assert is_denylisted(worker)
    with worker.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"attributionAgent": "general-purpose"}) + "\n")
    assert not is_denylisted(worker)
    assert calls == 202


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
