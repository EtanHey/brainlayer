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
