import json
from pathlib import Path

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


def test_explicit_environment_override_can_deny_an_otherwise_allowed_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(BRAINLAYER_INGEST_DENYLIST_ENV, "~/.codex/sessions/**")

    assert is_denylisted(tmp_path / ".codex" / "sessions" / "worker.jsonl")
    assert not is_denylisted(tmp_path / ".gemini" / "sessions" / "worker.jsonl")
