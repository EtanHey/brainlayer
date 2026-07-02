from pathlib import Path

from brainlayer.ingest_denylist import is_denylisted


def test_default_denylist_matches_alternate_home_without_process_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "real-home"))
    backup_home = tmp_path / "backup-home"

    assert is_denylisted(backup_home / ".codex" / "sessions" / "worker.jsonl")
    assert is_denylisted(backup_home / ".gemini" / "sessions" / "worker.jsonl")
    assert is_denylisted(
        backup_home / ".cursor" / "projects" / "repo" / "agent-transcripts" / "session" / "worker.jsonl"
    )
    assert is_denylisted(backup_home / ".claude" / "projects" / "proj" / "session" / "subagents" / "agent-a111.jsonl")


def test_default_denylist_is_segment_scoped(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    assert not is_denylisted(tmp_path / ".claude" / "projects" / "proj" / "direct-session.jsonl")
    assert is_denylisted(Path(tmp_path / ".cursor" / "projects" / "repo" / "agent-transcripts" / "worker.jsonl"))


def test_brain_worker_subagent_transcript_is_denylisted(monkeypatch, tmp_path):
    """Regression (Etan flag 2026-07-03): the brain-worker off-grid recon subagent
    (~/.claude/agents/brain-worker.md) writes its session JSONL like every Agent-tool
    subagent — under a `subagents/` dir — so it MUST be in the agent-output exclusion set.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    projects = tmp_path / ".claude" / "projects"

    # brain-worker (and any Agent-tool subagent) transcript path
    assert is_denylisted(
        projects / "-Users-etanheyman-Gits-brainlayer" / "session-uuid" / "subagents" / "agent-a25d6b3aa6880db8e.jsonl"
    )
    # workflow subagents nested a level deeper still excluded
    assert is_denylisted(
        projects / "-Users-x" / "session-uuid" / "subagents" / "workflows" / "wf_c83ce37d" / "agent-x.jsonl"
    )


def test_direct_control_transcripts_still_ingest(monkeypatch, tmp_path):
    """Regression (Etan flag 2026-07-03): DIRECT/CONTROL transcripts — a lead's or a
    top-level worker's own reasoning session — are NOT agent-output roots and MUST keep
    ingesting. The denylist targets agent-OUTPUT paths only, never direct sessions.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    projects = tmp_path / ".claude" / "projects"

    assert not is_denylisted(projects / "-Users-etanheyman-Gits-brainlayer" / "4220c177-8816-446d.jsonl")
    assert not is_denylisted(projects / "-Users-x" / "session-uuid" / "session-uuid.jsonl")
