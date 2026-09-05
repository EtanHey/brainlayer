"""The fallback queue drains itself.

The write half of the brain_store fallback worked for months while the replay half
only ever ran when a human typed the script: 122 memories sat in
`docs.local/decisions` from June and July, invisible to `brain_search`, while every
agent that said "brainlayer's down, it'll replay" was misinforming Etan. A queue
nothing drains on its own is the hidden-crash-loop class, so the drain daemon sweeps
it at startup — under a bound, so a pathological backlog cannot flood the queue on
every restart.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def _git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _repo(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(("git", "init", "-q"), cwd=path, env=_git_env(), check=True)
    return path


def _pending(repo: Path, name: str) -> Path:
    path = repo / "docs.local" / "decisions" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "intended_brain_store: true\n"
        "importance: 8\n"
        "tags: [correction]\n"
        "timestamp: 2026-06-28T10:00:00+03:00\n"
        "reason: transport_closed\n"
        "chunk_id:\n"
        "---\n"
        f"body of {name}\n",
        encoding="utf-8",
    )
    return path


def test_enqueue_pending_fallbacks_reports_deferred_and_leaves_nothing_unseen(tmp_path):
    from brainlayer.fallback_replay import enqueue_pending_fallbacks

    repo = _repo(tmp_path / "gits", "systems")
    _pending(repo, "one.md")
    _pending(repo, "two.md")
    queued = []

    def enqueue(**kwargs):
        target = tmp_path / "queue" / f"{len(queued)}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")
        queued.append(kwargs)
        return target

    summary = enqueue_pending_fallbacks(
        gits_root=tmp_path / "gits",
        scope_map={},
        enqueue_func=enqueue,
        replayed_by="test",
        limit=10,
    )

    assert len(queued) == 2
    assert summary["pending_before"] == 2
    assert summary["attempted"] == 2
    assert summary["remaining"] == 0
    assert summary["outcome_counts"] == {"deferred": 2}


def test_enqueue_pending_fallbacks_stops_at_the_batch_bound(tmp_path):
    from brainlayer.fallback_replay import enqueue_pending_fallbacks

    repo = _repo(tmp_path / "gits", "systems")
    for index in range(5):
        _pending(repo, f"file-{index}.md")
    queued = []

    def enqueue(**kwargs):
        target = tmp_path / "queue" / f"{len(queued)}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")
        queued.append(kwargs)
        return target

    summary = enqueue_pending_fallbacks(
        gits_root=tmp_path / "gits",
        scope_map={},
        enqueue_func=enqueue,
        replayed_by="test",
        limit=2,
    )

    assert len(queued) == 2
    assert summary["pending_before"] == 5
    assert summary["attempted"] == 2
    assert summary["limit"] == 2
    assert summary["remaining"] == 3


def test_enqueue_pending_fallbacks_rejects_a_nonpositive_bound(tmp_path):
    from brainlayer.fallback_replay import enqueue_pending_fallbacks

    with pytest.raises(ValueError):
        enqueue_pending_fallbacks(
            gits_root=tmp_path,
            scope_map={},
            enqueue_func=lambda **_kwargs: tmp_path / "unused.jsonl",
            replayed_by="test",
            limit=0,
        )


def test_enqueue_pending_fallbacks_reports_a_per_file_error_without_abandoning_the_rest(tmp_path):
    from brainlayer.fallback_replay import enqueue_pending_fallbacks

    repo = _repo(tmp_path / "gits", "systems")
    _pending(repo, "aaa-boom.md")
    _pending(repo, "zzz-fine.md")
    queued = []

    def enqueue(**kwargs):
        if "aaa-boom" in str(kwargs["fallback_source_path"]):
            raise RuntimeError("queue write failed")
        target = tmp_path / "queue" / "ok.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")
        queued.append(kwargs)
        return target

    summary = enqueue_pending_fallbacks(
        gits_root=tmp_path / "gits",
        scope_map={},
        enqueue_func=enqueue,
        replayed_by="test",
        limit=10,
    )

    assert len(queued) == 1
    assert summary["outcome_counts"] == {"deferred": 1, "error": 1}
    assert summary["errors"] == 1


def test_run_daemon_sweeps_the_fallback_queue_once_before_draining(tmp_path):
    from brainlayer.drain import run_daemon

    calls = []

    run_daemon(
        interval=0,
        batch_size=10,
        health_path=tmp_path / "drain-health.json",
        drain_once_fn=lambda **_kwargs: calls.append("drain") or 0,
        sleep_fn=lambda _seconds: None,
        max_cycles=3,
        replay_fallbacks_fn=lambda: calls.append("fallback-sweep"),
    )

    assert calls[0] == "fallback-sweep"
    assert calls.count("fallback-sweep") == 1
    assert calls.count("drain") == 3


def test_run_daemon_keeps_draining_when_the_fallback_sweep_raises(tmp_path):
    from brainlayer.drain import run_daemon

    drained = []

    def boom():
        raise RuntimeError("inventory blew up")

    run_daemon(
        interval=0,
        batch_size=10,
        health_path=tmp_path / "drain-health.json",
        drain_once_fn=lambda **_kwargs: drained.append("drain") or 0,
        sleep_fn=lambda _seconds: None,
        max_cycles=2,
        replay_fallbacks_fn=boom,
    )

    assert drained == ["drain", "drain"]


def test_fallback_sweep_on_start_is_opt_out_by_env(monkeypatch, tmp_path):
    from brainlayer import drain

    monkeypatch.setenv("BRAINLAYER_FALLBACK_REPLAY_ON_START", "0")
    monkeypatch.setattr(
        drain,
        "enqueue_pending_fallbacks_for_start",
        lambda **_kwargs: pytest.fail("swept while disabled"),
        raising=False,
    )

    assert drain.replay_fallbacks_on_start(log_path=tmp_path / "drain.log") is None


def test_fallback_sweep_on_start_defaults_to_enabled_and_logs_its_receipt(monkeypatch, tmp_path):
    from brainlayer import drain, fallback_replay

    monkeypatch.delenv("BRAINLAYER_FALLBACK_REPLAY_ON_START", raising=False)
    seen = {}

    def fake_sweep(**kwargs):
        seen.update(kwargs)
        return {"pending_before": 3, "attempted": 3, "remaining": 0, "outcome_counts": {"deferred": 3}, "errors": 0}

    monkeypatch.setattr(fallback_replay, "enqueue_pending_fallbacks", fake_sweep)

    log_path = tmp_path / "drain.log"
    summary = drain.replay_fallbacks_on_start(log_path=log_path)

    assert summary["attempted"] == 3
    assert seen["limit"] == drain.DEFAULT_FALLBACK_REPLAY_ON_START_LIMIT
    assert seen["replayed_by"] == "brainlayer-drain-start"
    assert "fallback replay on start" in log_path.read_text(encoding="utf-8")


def test_fallback_sweep_bound_is_configurable_and_ignores_junk(monkeypatch):
    from brainlayer import drain

    monkeypatch.setenv("BRAINLAYER_FALLBACK_REPLAY_ON_START_LIMIT", "7")
    assert drain._fallback_replay_on_start_limit() == 7

    monkeypatch.setenv("BRAINLAYER_FALLBACK_REPLAY_ON_START_LIMIT", "not-a-number")
    assert drain._fallback_replay_on_start_limit() == drain.DEFAULT_FALLBACK_REPLAY_ON_START_LIMIT

    monkeypatch.setenv("BRAINLAYER_FALLBACK_REPLAY_ON_START_LIMIT", "-4")
    assert drain._fallback_replay_on_start_limit() == drain.DEFAULT_FALLBACK_REPLAY_ON_START_LIMIT
