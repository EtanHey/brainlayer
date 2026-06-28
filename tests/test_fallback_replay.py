from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
from pathlib import Path


def _git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _git_init(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(("git", "init", "-q"), cwd=path, env=_git_env(), check=True)


def _pending_file(repo: Path, relative: str, *, project: str | None = None) -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    project_line = f"project: {project}\n" if project else ""
    path.write_text(
        "---\n"
        "intended_brain_store: true\n"
        "importance: 8\n"
        "tags: [user-correction, systems]\n"
        f"{project_line}"
        "timestamp: 2026-06-12T19:54:07+03:00\n"
        "reason: transport_closed\n"
        "retry_attempted: true\n"
        "chunk_id:\n"
        "---\n"
        "original body stays byte-for-byte\n",
        encoding="utf-8",
    )
    return path


def test_attribution_uses_scopes_mapping_for_originating_repo(tmp_path):
    from brainlayer.fallback_replay import load_scope_map, parse_fallback_file

    systems = tmp_path / "systems"
    narration = tmp_path / "narrationlayer"
    _git_init(systems)
    _git_init(narration)
    scopes_path = tmp_path / "scopes.yaml"
    scopes_path.write_text(
        f'scopes:\n  {systems}: "systems"\n  {narration}: "narrationlayer"\ndefault: "all"\n',
        encoding="utf-8",
    )

    systems_entry = parse_fallback_file(
        _pending_file(systems, "docs.local/decisions/systems.md"),
        scope_map=load_scope_map(scopes_path),
    )
    narration_entry = parse_fallback_file(
        _pending_file(narration, "docs.local/decisions/narration.md"),
        scope_map=load_scope_map(scopes_path),
    )

    assert systems_entry.project == "systems"
    assert systems_entry.origin_repo_path == systems
    assert narration_entry.project == "narrationlayer"
    assert narration_entry.origin_repo_path == narration


def test_frontmatter_project_overrides_scopes_mapping(tmp_path):
    from brainlayer.fallback_replay import load_scope_map, parse_fallback_file

    repo = tmp_path / "systems"
    _git_init(repo)
    scopes_path = tmp_path / "scopes.yaml"
    scopes_path.write_text(f'scopes:\n  {repo}: "systems"\n', encoding="utf-8")

    entry = parse_fallback_file(
        _pending_file(repo, "docs.local/decisions/explicit.md", project="cmuxlayer"),
        scope_map=load_scope_map(scopes_path),
    )

    assert entry.project == "cmuxlayer"


def test_replay_preserves_attribution_and_updates_frontmatter_only(tmp_path):
    from brainlayer.fallback_replay import load_scope_map, parse_fallback_file, replay_entry

    repo = tmp_path / "narrationlayer"
    _git_init(repo)
    scopes_path = tmp_path / "scopes.yaml"
    scopes_path.write_text(f'scopes:\n  {repo}: "narrationlayer"\n', encoding="utf-8")
    path = _pending_file(repo, "docs.local/decisions/pending.md")
    original_body = path.read_text(encoding="utf-8").split("---\n", 2)[2]
    calls = []

    def store_func(**kwargs):
        calls.append(kwargs)
        return {"id": "manual-replayed123", "related": []}

    entry = parse_fallback_file(path, scope_map=load_scope_map(scopes_path))
    result = replay_entry(entry, store_func=store_func, replayed_by="phase-1-test")

    assert result.chunk_id == "manual-replayed123"
    assert calls == [
        {
            "content": "original body stays byte-for-byte\n",
            "memory_type": "note",
            "project": "narrationlayer",
            "tags": ["user-correction", "systems"],
            "importance": 8,
            "created_at": "2026-06-12T19:54:07+03:00",
            "fallback_source_path": str(path),
            "origin_repo_path": str(repo),
            "replayed_by": "phase-1-test",
        }
    ]

    updated = path.read_text(encoding="utf-8")
    assert "chunk_id: manual-replayed123" in updated
    assert updated.endswith(original_body)


def test_replay_wraps_scalar_frontmatter_tag_as_single_tag(tmp_path):
    from brainlayer.fallback_replay import parse_fallback_file, replay_entry

    repo = tmp_path / "systems"
    _git_init(repo)
    path = repo / "docs.local" / "decisions" / "pending.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nintended_brain_store: true\ntags: user-correction\nchunk_id:\n---\nscalar tag body\n",
        encoding="utf-8",
    )
    calls = []

    def store_func(**kwargs):
        calls.append(kwargs)
        return {"chunk_id": "manual-scalar-tag"}

    entry = parse_fallback_file(path)
    replay_entry(entry, store_func=store_func, replayed_by="phase-1-test")

    assert calls[0]["tags"] == ["user-correction"]


def test_parse_fallback_file_accepts_frontmatter_closing_delimiter_at_eof(tmp_path):
    from brainlayer.fallback_replay import parse_fallback_file

    repo = tmp_path / "systems"
    _git_init(repo)
    path = repo / "docs.local" / "decisions" / "frontmatter-only.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "intended_brain_store: true\n"
        "memory_type: decision\n"
        "project: systems\n"
        "tags: [fallback]\n"
        "timestamp: 2026-06-12T19:54:07+03:00\n"
        "---",
        encoding="utf-8",
    )

    entry = parse_fallback_file(path)

    assert entry.frontmatter["memory_type"] == "decision"
    assert entry.frontmatter["tags"] == ["fallback"]
    assert entry.frontmatter["timestamp"] == "2026-06-12T19:54:07+03:00"
    assert entry.body == ""


def test_parse_fallback_file_ignores_non_mapping_frontmatter(tmp_path):
    from brainlayer.fallback_replay import is_pending_entry, parse_fallback_file

    repo = tmp_path / "systems"
    _git_init(repo)
    path = repo / "docs.local" / "decisions" / "bad-frontmatter.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\n42\n---\nbody\n", encoding="utf-8")

    entry = parse_fallback_file(path)

    assert entry.frontmatter == {}
    assert entry.body == "body\n"
    assert is_pending_entry(entry) is False


def test_replay_errors_when_store_result_has_no_chunk_id(tmp_path):
    from brainlayer.fallback_replay import is_pending_entry, parse_fallback_file, replay_entry

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/missing-id.md")
    entry = parse_fallback_file(path)

    result = replay_entry(entry, store_func=lambda **_kwargs: {"related": []}, replayed_by="phase-1-test")
    updated_entry = parse_fallback_file(path)

    assert result.attempted is True
    assert result.chunk_id is None
    assert result.error == "store result did not include a chunk_id"
    assert is_pending_entry(updated_entry) is True
    assert updated_entry.frontmatter["retry_attempted"] is True


def test_replay_preserves_stored_chunk_id_when_frontmatter_update_fails(tmp_path, monkeypatch):
    import brainlayer.fallback_replay as fallback_replay

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/write-fails.md")
    entry = fallback_replay.parse_fallback_file(path)

    def fail_write(*_args, **_kwargs):
        raise OSError("read-only fallback file")

    monkeypatch.setattr(fallback_replay, "_write_frontmatter", fail_write)

    result = fallback_replay.replay_entry(
        entry, store_func=lambda **_kwargs: {"chunk_id": "manual-stored"}, replayed_by="phase-1-test"
    )

    assert result.attempted is True
    assert result.chunk_id == "manual-stored"
    assert (
        result.error
        == "stored chunk_id=manual-stored but failed to update fallback frontmatter: read-only fallback file"
    )


def test_replay_returns_missing_chunk_id_error_when_marker_write_fails(tmp_path, monkeypatch):
    import brainlayer.fallback_replay as fallback_replay

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/missing-id-and-write-fail.md")
    entry = fallback_replay.parse_fallback_file(path)

    def fail_write(*_args, **_kwargs):
        raise OSError("read-only fallback file")

    monkeypatch.setattr(fallback_replay, "_write_frontmatter", fail_write)

    result = fallback_replay.replay_entry(
        entry, store_func=lambda **_kwargs: {"related": []}, replayed_by="phase-1-test"
    )

    assert result.attempted is True
    assert result.chunk_id is None
    assert (
        result.error == "store result did not include a chunk_id; write_replay_attempt failed: read-only fallback file"
    )


def test_replay_returns_store_error_when_failed_attempt_marker_write_fails(tmp_path, monkeypatch):
    import brainlayer.fallback_replay as fallback_replay

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/store-and-write-fail.md")
    entry = fallback_replay.parse_fallback_file(path)

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    def fail_store(**_kwargs):
        raise RuntimeError("store failed")

    monkeypatch.setattr(fallback_replay, "_write_frontmatter", fail_write)

    result = fallback_replay.replay_entry(entry, store_func=fail_store, replayed_by="phase-1-test")

    assert result.attempted is True
    assert result.chunk_id is None
    assert result.error == "store failed: store failed; write_replay_attempt failed: disk full"


def test_inventory_fallbacks_reports_structured_pending_and_legacy_debt(tmp_path):
    from brainlayer.fallback_replay import inventory_fallbacks

    structured_repo = tmp_path / "structured"
    legacy_repo = tmp_path / "legacy"
    pending = _pending_file(structured_repo, "docs.local/decisions/pending.md")
    replayed = _pending_file(structured_repo, "docs.local/decisions/replayed.md")
    replayed.write_text(replayed.read_text(encoding="utf-8").replace("chunk_id:\n", "chunk_id: manual-existing\n"))
    legacy = legacy_repo / "docs.local" / "brain-store-fallback" / "transport-closed.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy fallback body\n", encoding="utf-8")

    inventory = inventory_fallbacks(tmp_path, scope_map={})
    summary = inventory.summary()

    assert [entry.path for entry in inventory.pending] == [pending]
    assert summary["green"] is False
    assert summary["structured_count"] == 2
    assert summary["pending_count"] == 1
    assert summary["legacy_count"] == 1
    assert summary["pending_sample"] == [str(pending)]
    assert summary["legacy_sample"] == [str(legacy)]


def test_inventory_treats_stale_fallback_chunk_id_as_pending(tmp_path):
    from brainlayer.fallback_replay import _fallback_chunk_id, inventory_fallbacks, parse_fallback_file

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/edited-after-queue.md")
    old_chunk_id = _fallback_chunk_id(parse_fallback_file(path))
    path.write_text(
        path.read_text(encoding="utf-8")
        .replace("chunk_id:\n", f"chunk_id: {old_chunk_id}\n")
        .replace("original body stays byte-for-byte\n", "edited body should still be pending\n"),
        encoding="utf-8",
    )

    inventory = inventory_fallbacks(tmp_path, scope_map={})

    assert [entry.path for entry in inventory.pending] == [path]


def test_queue_entry_enqueues_with_stable_chunk_id_but_keeps_fallback_pending(tmp_path):
    from brainlayer.fallback_replay import is_pending_entry, parse_fallback_file, queue_entry

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/pending.md")
    calls = []

    def enqueue_func(**kwargs):
        calls.append(kwargs)
        queue_path = tmp_path / "queue" / "fallback-replay.jsonl"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text("queued\n", encoding="utf-8")
        return queue_path

    entry = parse_fallback_file(path)
    first = queue_entry(entry, enqueue_func=enqueue_func, replayed_by="phase-1-test")
    updated_entry = parse_fallback_file(path)
    second = queue_entry(updated_entry, enqueue_func=enqueue_func, replayed_by="phase-1-test")
    second_entry = parse_fallback_file(path)

    assert first.error is None
    assert first.chunk_id is not None
    assert first.chunk_id.startswith("fallback-")
    assert second.chunk_id == first.chunk_id
    assert calls[0]["source"] == "fallback-replay"
    assert calls[0]["chunk_id"] == first.chunk_id
    assert calls[0]["fallback_source_path"] == str(path)
    assert calls[0]["origin_repo_path"] == str(repo.resolve())
    assert len(calls) == 1
    assert updated_entry.frontmatter["project"] == "systems"
    assert updated_entry.frontmatter["queued_chunk_id"] == first.chunk_id
    assert updated_entry.frontmatter["queued_queue_path"] == str(tmp_path / "queue" / "fallback-replay.jsonl")
    assert updated_entry.frontmatter["chunk_id"] is None
    assert is_pending_entry(updated_entry) is True
    assert second_entry.frontmatter["queued_chunk_id"] == first.chunk_id
    assert second_entry.frontmatter["chunk_id"] is None


def test_queue_entry_requeues_when_recorded_queue_file_is_missing(tmp_path):
    from brainlayer.fallback_replay import parse_fallback_file, queue_entry

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/missing-queue-file.md")
    calls = []

    def enqueue_func(**kwargs):
        calls.append(kwargs)
        queue_path = tmp_path / "queue" / f"fallback-replay-{len(calls)}.jsonl"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text("queued\n", encoding="utf-8")
        return queue_path

    first = queue_entry(parse_fallback_file(path), enqueue_func=enqueue_func, replayed_by="phase-1-test")
    first_queue_path = Path(parse_fallback_file(path).frontmatter["queued_queue_path"])
    first_queue_path.unlink()

    second = queue_entry(parse_fallback_file(path), enqueue_func=enqueue_func, replayed_by="phase-1-test")
    updated = parse_fallback_file(path)

    assert first.chunk_id == second.chunk_id
    assert len(calls) == 2
    assert Path(updated.frontmatter["queued_queue_path"]).exists()


def test_queue_entry_clears_stale_queue_path_when_requeue_returns_no_path(tmp_path):
    from brainlayer.fallback_replay import parse_fallback_file, queue_entry

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/missing-queue-file-no-new-path.md")
    calls = []

    def enqueue_func(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            queue_path = tmp_path / "queue" / "fallback-replay.jsonl"
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text("queued\n", encoding="utf-8")
            return queue_path
        return None

    first = queue_entry(parse_fallback_file(path), enqueue_func=enqueue_func, replayed_by="phase-1-test")
    first_queue_path = Path(parse_fallback_file(path).frontmatter["queued_queue_path"])
    first_queue_path.unlink()

    second = queue_entry(parse_fallback_file(path), enqueue_func=enqueue_func, replayed_by="phase-1-test")
    updated = parse_fallback_file(path)

    assert first.chunk_id == second.chunk_id
    assert len(calls) == 2
    assert updated.frontmatter["queued_chunk_id"] == first.chunk_id
    assert updated.frontmatter.get("queued_queue_path") is None


def test_mark_fallback_stored_persists_scoped_project_for_chunk_id_parity(tmp_path):
    from brainlayer.fallback_replay import (
        is_pending_entry,
        load_scope_map,
        mark_fallback_stored,
        parse_fallback_file,
        queue_entry,
    )

    repo = tmp_path / "brainlayer"
    _git_init(repo)
    scopes_path = tmp_path / "scopes.yaml"
    scopes_path.write_text(f'scopes:\n  {repo}: "systems"\n', encoding="utf-8")
    path = _pending_file(repo, "docs.local/decisions/scoped.md")
    calls = []

    def enqueue_func(**kwargs):
        calls.append(kwargs)
        return tmp_path / "queue" / "fallback-replay.jsonl"

    entry = parse_fallback_file(path, scope_map=load_scope_map(scopes_path))
    queued = queue_entry(entry, enqueue_func=enqueue_func, replayed_by="phase-1-test")
    mark_fallback_stored(
        path,
        chunk_id=queued.chunk_id or "",
        project=entry.project,
        origin_repo_path=entry.origin_repo_path,
    )
    updated_entry = parse_fallback_file(path)

    assert calls[0]["project"] == "systems"
    assert updated_entry.frontmatter["project"] == "systems"
    assert updated_entry.frontmatter["chunk_id"] == queued.chunk_id
    assert "queued_chunk_id" not in updated_entry.frontmatter
    assert is_pending_entry(updated_entry) is False


def test_queue_entry_recomputes_chunk_id_after_fallback_body_changes(tmp_path):
    from brainlayer.fallback_replay import parse_fallback_file, queue_entry

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/edited.md")
    calls = []

    def enqueue_func(**kwargs):
        calls.append(kwargs)
        return tmp_path / "queue" / "fallback-replay.jsonl"

    first = queue_entry(parse_fallback_file(path), enqueue_func=enqueue_func, replayed_by="phase-1-test")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "original body stays byte-for-byte\n",
            "edited fallback body must receive a fresh queue id\n",
        ),
        encoding="utf-8",
    )
    second = queue_entry(parse_fallback_file(path), enqueue_func=enqueue_func, replayed_by="phase-1-test")

    assert first.chunk_id != second.chunk_id
    assert calls[0]["chunk_id"] == first.chunk_id
    assert calls[1]["chunk_id"] == second.chunk_id


def test_replay_entry_trusts_returned_fallback_chunk_id_from_direct_store(tmp_path):
    from brainlayer.fallback_replay import is_pending_entry, parse_fallback_file, queue_entry, replay_entry

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/direct-after-queue.md")
    queued = queue_entry(parse_fallback_file(path), enqueue_func=lambda **_kwargs: None, replayed_by="phase-1-test")

    result = replay_entry(
        parse_fallback_file(path),
        store_func=lambda **_kwargs: {"chunk_id": "brainbar-direct-store-id"},
        replayed_by="phase-1-test",
    )
    updated = parse_fallback_file(path)

    assert queued.chunk_id != result.chunk_id
    assert result.error is None
    assert updated.frontmatter["chunk_id"] == "brainbar-direct-store-id"
    assert updated.frontmatter["replayed_body_sha256"]
    assert "queued_chunk_id" not in updated.frontmatter
    assert is_pending_entry(updated) is False
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "original body stays byte-for-byte\n",
            "edited after trusted direct replay\n",
        ),
        encoding="utf-8",
    )
    edited = parse_fallback_file(path)
    assert is_pending_entry(edited) is True

    requeued_calls = []

    def requeue_func(**kwargs):
        requeued_calls.append(kwargs)
        queue_path = tmp_path / "queue" / "edited-direct.jsonl"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text("queued\n", encoding="utf-8")
        return queue_path

    requeued = queue_entry(edited, enqueue_func=requeue_func, replayed_by="phase-1-test")
    requeued_entry = parse_fallback_file(path)

    assert requeued.error is None
    assert requeued.chunk_id is not None
    assert requeued.chunk_id.startswith("fallback-")
    assert requeued.chunk_id != result.chunk_id
    assert requeued_calls[0]["chunk_id"] == requeued.chunk_id
    assert requeued_entry.frontmatter["chunk_id"] is None
    assert requeued_entry.frontmatter["queued_chunk_id"] == requeued.chunk_id


def test_mark_fallback_stored_handles_legacy_path_metadata(tmp_path):
    from brainlayer.fallback_replay import (
        _fallback_chunk_id,
        inventory_fallbacks,
        legacy_entry_from_path,
        mark_fallback_stored,
    )

    repo = tmp_path / "orchestrator"
    _git_init(repo)
    path = repo / "docs.local" / "brain-store-fallback" / "2026-05-29-gen10-boot" / "pending-stores.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# legacy fallback\nBody stays intact.\n", encoding="utf-8")
    chunk_id = _fallback_chunk_id(legacy_entry_from_path(path, scope_map={}))

    mark_fallback_stored(path, chunk_id=chunk_id, project="orchestrator", origin_repo_path=repo.resolve())

    inventory = inventory_fallbacks(tmp_path, scope_map={})
    updated = path.read_text(encoding="utf-8")
    assert inventory.legacy == []
    assert f"chunk_id: {chunk_id}" in updated
    assert "legacy_brain_store_fallback: true" in updated


def test_queue_entry_serializes_yaml_date_tags_for_jsonl_queue(tmp_path):
    from brainlayer.fallback_replay import parse_fallback_file, queue_entry

    repo = tmp_path / "systems"
    _git_init(repo)
    path = repo / "docs.local" / "decisions" / "date-tag.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "intended_brain_store: true\n"
        "importance: 7\n"
        "tags:\n"
        "- brainlayer\n"
        "- 2026-06-04\n"
        "timestamp: '2026-06-04T15:31:57Z'\n"
        "chunk_id:\n"
        "---\n"
        "body with a yaml date-shaped tag\n",
        encoding="utf-8",
    )
    calls = []

    def jsonl_enqueue_func(**kwargs):
        import json

        json.dumps(kwargs)
        calls.append(kwargs)

    result = queue_entry(parse_fallback_file(path), enqueue_func=jsonl_enqueue_func, replayed_by="phase-1-test")

    assert result.error is None
    assert calls[0]["tags"] == ["brainlayer", "2026-06-04"]


def test_queue_legacy_entry_synthesizes_metadata_and_marks_replayed(tmp_path):
    from brainlayer.fallback_replay import legacy_entry_from_path, queue_legacy_entry

    repo = tmp_path / "orchestrator"
    _git_init(repo)
    path = repo / "docs.local" / "brain-store-fallback" / "2026-05-29-gen10-boot" / "pending-stores.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# gen-10 pending-stores\n"
        "## [imp:8] USER CORRECTION -- use BrainLayer, NOT harness file-memory\n"
        "Tags: etan-correction, frustration, brainlayer\n"
        "Body stays intact.\n",
        encoding="utf-8",
    )
    calls = []

    def enqueue_func(**kwargs):
        calls.append(kwargs)

    result = queue_legacy_entry(
        legacy_entry_from_path(path, scope_map={}),
        enqueue_func=enqueue_func,
        replayed_by="phase-1-test",
    )

    updated = path.read_text(encoding="utf-8")
    assert result.error is None
    assert result.chunk_id is not None
    assert calls[0]["importance"] == 8
    assert calls[0]["tags"] == ["legacy-fallback", "brain-store-fallback", "orchestrator"]
    assert calls[0]["created_at"] == "2026-05-29T00:00:00Z"
    assert calls[0]["source"] == "fallback-replay"
    assert updated.startswith("---\n")
    assert "legacy_brain_store_fallback: true" in updated
    assert "chunk_id: " + result.chunk_id in updated
    assert updated.endswith("Body stays intact.\n")


def test_inventory_ignores_replayed_legacy_fallback_markers(tmp_path):
    from brainlayer.fallback_replay import _fallback_chunk_id, inventory_fallbacks, legacy_entry_from_path

    repo = tmp_path / "orchestrator"
    _git_init(repo)
    path = repo / "docs.local" / "brain-store-fallback" / "replayed.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\n"
        "intended_brain_store: true\n"
        "legacy_brain_store_fallback: true\n"
        "retry_attempted: true\n"
        "chunk_id:\n"
        "---\n"
        "legacy body\n",
        encoding="utf-8",
    )
    chunk_id = _fallback_chunk_id(legacy_entry_from_path(path, scope_map={}))
    path.write_text(path.read_text(encoding="utf-8").replace("chunk_id:\n", f"chunk_id: {chunk_id}\n"))

    inventory = inventory_fallbacks(tmp_path, scope_map={})

    assert inventory.legacy == []
    assert inventory.summary()["legacy_count"] == 0
    assert inventory.summary()["green"] is True


def test_inventory_treats_stale_legacy_fallback_chunk_id_as_debt(tmp_path):
    from brainlayer.fallback_replay import _fallback_chunk_id, inventory_fallbacks, legacy_entry_from_path

    repo = tmp_path / "orchestrator"
    _git_init(repo)
    path = repo / "docs.local" / "brain-store-fallback" / "edited-legacy.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\n"
        "intended_brain_store: true\n"
        "legacy_brain_store_fallback: true\n"
        "retry_attempted: true\n"
        "chunk_id:\n"
        "---\n"
        "legacy body before edit\n",
        encoding="utf-8",
    )
    old_chunk_id = _fallback_chunk_id(legacy_entry_from_path(path, scope_map={}))
    path.write_text(
        path.read_text(encoding="utf-8")
        .replace("chunk_id:\n", f"chunk_id: {old_chunk_id}\n")
        .replace("legacy body before edit\n", "legacy body after edit\n"),
        encoding="utf-8",
    )

    inventory = inventory_fallbacks(tmp_path, scope_map={})

    assert inventory.legacy == [path]
    assert inventory.summary()["green"] is False


def test_fallback_chunk_id_is_stable_across_checkout_roots(tmp_path):
    from brainlayer.fallback_replay import _fallback_chunk_id, parse_fallback_file

    first_repo = tmp_path / "first" / "brainlayer"
    second_repo = tmp_path / "second" / "brainlayer"
    for repo in (first_repo, second_repo):
        _git_init(repo)
        _pending_file(repo, "docs.local/decisions/stable.md")

    first = parse_fallback_file(first_repo / "docs.local" / "decisions" / "stable.md")
    second = parse_fallback_file(second_repo / "docs.local" / "decisions" / "stable.md")

    assert _fallback_chunk_id(first) == _fallback_chunk_id(second)


def test_queue_attempt_does_not_clobber_committed_chunk_marker(tmp_path):
    from brainlayer.fallback_replay import _write_queue_attempt, parse_fallback_file

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/race.md")
    stale_entry = parse_fallback_file(path)
    path.write_text(path.read_text(encoding="utf-8").replace("chunk_id:\n", "chunk_id: manual-stored\n"))

    _write_queue_attempt(stale_entry, chunk_id="fallback-queued")

    updated = parse_fallback_file(path)
    assert updated.frontmatter["chunk_id"] == "manual-stored"
    assert "queued_chunk_id" not in updated.frontmatter


def test_queue_attempt_does_not_resurrect_removed_structured_intent(tmp_path):
    from brainlayer.fallback_replay import _fallback_chunk_id, _write_queue_attempt, parse_fallback_file

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/cancelled.md")
    stale_entry = parse_fallback_file(path)
    stale_chunk_id = _fallback_chunk_id(stale_entry)
    path.write_text(
        path.read_text(encoding="utf-8").replace("intended_brain_store: true\n", ""),
        encoding="utf-8",
    )

    _write_queue_attempt(stale_entry, chunk_id=stale_chunk_id)

    updated = parse_fallback_file(path)
    assert "intended_brain_store" not in updated.frontmatter
    assert "queued_chunk_id" not in updated.frontmatter


def test_fallback_marker_writes_are_serialized_per_file(tmp_path):
    from brainlayer import fallback_replay
    from brainlayer.fallback_replay import _fallback_chunk_id, _write_queue_attempt, parse_fallback_file

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/serialized.md")
    entry = parse_fallback_file(path)
    chunk_id = _fallback_chunk_id(entry)
    lock_held = threading.Event()
    release_lock = threading.Event()
    write_done = threading.Event()

    def hold_lock():
        with fallback_replay._fallback_marker_file_lock(path):
            lock_held.set()
            release_lock.wait(timeout=2)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert lock_held.wait(timeout=1)

    writer = threading.Thread(target=lambda: (_write_queue_attempt(entry, chunk_id=chunk_id), write_done.set()))
    writer.start()
    assert not write_done.wait(timeout=0.1)

    release_lock.set()
    holder.join(timeout=1)
    writer.join(timeout=1)

    assert write_done.is_set()
    updated = parse_fallback_file(path)
    assert updated.frontmatter["queued_chunk_id"] == chunk_id


def test_queue_entry_serializes_enqueue_with_marker_check(tmp_path):
    from brainlayer.fallback_replay import parse_fallback_file, queue_entry

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/serialized-queue.md")
    entry = parse_fallback_file(path)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    calls: list[dict] = []
    release_first_enqueue = threading.Event()

    def enqueue_func(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            release_first_enqueue.wait(timeout=0.2)
        queue_path = queue_dir / f"{len(calls)}.jsonl"
        queue_path.write_text("queued\n", encoding="utf-8")
        return queue_path

    results = []
    errors = []

    def worker():
        try:
            results.append(queue_entry(entry, enqueue_func=enqueue_func, replayed_by="phase-1-test"))
        except Exception as exc:  # pragma: no cover - assertion includes unexpected thread errors
            errors.append(exc)

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    first.start()
    second.start()
    release_first_enqueue.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert len(results) == 2
    assert len(calls) == 1
    updated = parse_fallback_file(path)
    assert updated.frontmatter["queued_chunk_id"] == results[0].chunk_id


def test_fallback_marker_file_lock_fails_closed_on_flock_error(tmp_path, monkeypatch):
    import fcntl

    from brainlayer.fallback_replay import _fallback_chunk_id, _write_queue_attempt, parse_fallback_file

    repo = tmp_path / "systems"
    _git_init(repo)
    path = _pending_file(repo, "docs.local/decisions/flock-fails.md")
    entry = parse_fallback_file(path)
    chunk_id = _fallback_chunk_id(entry)

    def fail_flock(*_args, **_kwargs):
        raise OSError("synthetic flock failure")

    monkeypatch.setattr(fcntl, "flock", fail_flock)

    try:
        _write_queue_attempt(entry, chunk_id=chunk_id)
    except OSError as exc:
        assert "synthetic flock failure" in str(exc)
    else:
        raise AssertionError("fallback marker lock should fail closed when flock fails")

    updated = parse_fallback_file(path)
    assert "queued_chunk_id" not in updated.frontmatter


def test_inventory_reports_unparseable_legacy_fallback_files_as_debt(tmp_path):
    from brainlayer.fallback_replay import inventory_fallbacks

    repo = tmp_path / "orchestrator"
    _git_init(repo)
    path = repo / "docs.local" / "brain-store-fallback" / "bad-frontmatter.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\n: bad yaml\n---\nlegacy body\n", encoding="utf-8")

    inventory = inventory_fallbacks(tmp_path, scope_map={})

    assert inventory.legacy == [path]
    assert inventory.summary()["legacy_count"] == 1
    assert inventory.summary()["green"] is False


def test_inventory_reports_unparseable_structured_fallback_files_as_debt(tmp_path):
    from brainlayer.fallback_replay import inventory_fallbacks

    repo = tmp_path / "orchestrator"
    _git_init(repo)
    path = repo / "docs.local" / "decisions" / "bad-frontmatter.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\n: bad yaml\n---\nstructured body\n", encoding="utf-8")

    inventory = inventory_fallbacks(tmp_path, scope_map={})

    assert inventory.legacy == [path]
    assert inventory.summary()["legacy_count"] == 1
    assert inventory.summary()["green"] is False


def test_replay_cli_apply_exits_nonzero_when_any_replay_errors(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "replay_brain_store_fallbacks.py"
    spec = importlib.util.spec_from_file_location("replay_brain_store_fallbacks_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from brainlayer.fallback_replay import parse_fallback_file

    repo = tmp_path / "systems"
    path = _pending_file(repo, "docs.local/decisions/failing.md")
    entry = parse_fallback_file(path)

    class DummyStore:
        def close(self):
            pass

    monkeypatch.setattr(module, "load_scope_map", lambda _path: {})

    class DummyInventory:
        structured = [entry]
        legacy = []
        pending = [entry]

    monkeypatch.setattr(module, "inventory_fallbacks", lambda _root, *, scope_map: DummyInventory())
    monkeypatch.setattr(module, "VectorStore", lambda _db: DummyStore())
    monkeypatch.setattr(module, "store_memory", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("store failed")))
    monkeypatch.setattr(
        sys,
        "argv",
        ["replay_brain_store_fallbacks.py", "--apply", "--direct-db-write", "--gits-root", str(tmp_path)],
    )

    assert module.main() == 1


def test_replay_cli_apply_queues_by_default(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "replay_brain_store_fallbacks.py"
    spec = importlib.util.spec_from_file_location("replay_brain_store_fallbacks_queue_default_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from brainlayer.fallback_replay import parse_fallback_file

    repo = tmp_path / "systems"
    path = _pending_file(repo, "docs.local/decisions/default-queue.md")
    entry = parse_fallback_file(path)
    calls = []

    class DummyInventory:
        structured = [entry]
        legacy = []
        pending = [entry]

    monkeypatch.setattr(module, "load_scope_map", lambda _path: {})
    monkeypatch.setattr(module, "inventory_fallbacks", lambda _root, *, scope_map: DummyInventory())
    monkeypatch.setattr(module, "VectorStore", lambda _db: (_ for _ in ()).throw(AssertionError("direct DB open")))
    monkeypatch.setattr(module, "enqueue_store", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["replay_brain_store_fallbacks.py", "--apply", "--gits-root", str(tmp_path)])

    assert module.main() == 0
    assert calls[0]["content"] == "original body stays byte-for-byte\n"
    assert calls[0]["source"] == "fallback-replay"


def test_replay_cli_apply_uses_custom_db_queue_dir_for_queued_replay(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "replay_brain_store_fallbacks.py"
    spec = importlib.util.spec_from_file_location("replay_brain_store_fallbacks_custom_queue_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from brainlayer.fallback_replay import parse_fallback_file

    repo = tmp_path / "systems"
    path = _pending_file(repo, "docs.local/decisions/custom-db-queue.md")
    entry = parse_fallback_file(path)
    db_path = tmp_path / "sandbox" / "brainlayer.db"
    calls = []

    class DummyInventory:
        structured = [entry]
        legacy = []
        pending = [entry]

    monkeypatch.setattr(module, "load_scope_map", lambda _path: {})
    monkeypatch.setattr(module, "inventory_fallbacks", lambda _root, *, scope_map: DummyInventory())
    monkeypatch.setattr(module, "enqueue_store", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        ["replay_brain_store_fallbacks.py", "--apply", "--gits-root", str(tmp_path), "--db", str(db_path)],
    )

    assert module.main() == 0
    assert calls[0]["queue_dir"] == db_path.parent / "queue"


def test_replay_cli_apply_queue_legacy_reports_malformed_legacy_files(tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "replay_brain_store_fallbacks.py"
    spec = importlib.util.spec_from_file_location("replay_brain_store_fallbacks_legacy_malformed_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    repo = tmp_path / "orchestrator"
    _git_init(repo)
    bad = repo / "docs.local" / "brain-store-fallback" / "bad.md"
    good = repo / "docs.local" / "brain-store-fallback" / "good.md"
    bad.parent.mkdir(parents=True)
    bad.write_text("---\n: bad yaml\n---\nmalformed legacy body\n", encoding="utf-8")
    good.write_text("valid legacy body\n", encoding="utf-8")
    calls = []

    def enqueue_func(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(module, "load_scope_map", lambda _path: {})
    monkeypatch.setattr(module, "enqueue_store", enqueue_func)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_brain_store_fallbacks.py",
            "--apply",
            "--queue",
            "--legacy",
            "--limit",
            "1",
            "--gits-root",
            str(tmp_path),
        ],
    )

    assert module.main() == 1
    output = capsys.readouterr().out
    assert len(calls) == 1
    assert calls[0]["content"] == "valid legacy body\n"
    assert "legacy parse failed" in output
    assert "replay_count" not in output


def test_replay_cli_apply_queue_legacy_marks_legacy_files(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "replay_brain_store_fallbacks.py"
    spec = importlib.util.spec_from_file_location("replay_brain_store_fallbacks_legacy_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    repo = tmp_path / "orchestrator"
    _git_init(repo)
    path = repo / "docs.local" / "brain-store-fallback" / "legacy.md"
    path.parent.mkdir(parents=True)
    path.write_text("legacy body\n", encoding="utf-8")
    calls = []

    def enqueue_func(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(module, "load_scope_map", lambda _path: {})
    monkeypatch.setattr(module, "enqueue_store", enqueue_func)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_brain_store_fallbacks.py",
            "--apply",
            "--queue",
            "--legacy",
            "--gits-root",
            str(tmp_path),
        ],
    )

    assert module.main() == 0
    assert calls[0]["source"] == "fallback-replay"
    assert "legacy_brain_store_fallback: true" in path.read_text(encoding="utf-8")
