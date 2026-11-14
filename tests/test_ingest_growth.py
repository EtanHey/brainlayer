"""Invented transcript growth through the real tailer, health payload and verdict."""

import json

import pytest

import brainlayer.watcher as wm
from brainlayer.watcher import JSONLWatcher, WatchRoot
from tests.test_ingest_stall_alerts import _Clock, _ingest_states

PROVIDERS = ("claude", "codex", "cursor", "cursor-agent-transcripts", "gemini")


@pytest.mark.parametrize("provider", PROVIDERS)
def test_growth_window_restart_and_recovery(tmp_path, monkeypatch, provider):
    clock = _Clock(wm.time)
    monkeypatch.setattr(wm, "time", clock)
    health = tmp_path / "watcher-health.json"
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "-proj" / "s.jsonl"
    source.parent.mkdir()

    def watcher():
        return JSONLWatcher(
            watch_roots=[WatchRoot(provider, root)],
            registry_path=tmp_path / "offsets.json",
            health_path=health,
            max_lines_per_file=1000,
            registry_flush_interval_s=0,
        )

    def append(record, n):
        with source.open("a") as f:
            f.write((json.dumps(record) + "\n") * n)

    def poll(w, seconds=0):
        clock.mono += seconds
        clock.wall_shift += seconds
        w.poll_once()
        return json.loads(health.read_text())["alert_reasons"]

    w = watcher()
    append({"kind": "new-format", "body": "unrecognised"}, 199)
    assert poll(w) == []
    assert poll(w, 900) == []
    w = watcher()  # checkpoint and the half-window survive a process restart
    append({"kind": "new-format"}, 1)
    assert poll(w, 899) == []
    assert f"growth_not_indexed:{provider}" in poll(w, 1)
    assert f"growth_not_indexed:{provider}" in poll(watcher(), 30)
    append({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "read"}]}}, 1)
    assert poll(w, 30) == []  # known tool-only message clears before content filtering


@pytest.mark.parametrize("provider", PROVIDERS)
def test_recognised_noise_idle_and_below_threshold(tmp_path, monkeypatch, provider):
    clock = _Clock(wm.time)
    monkeypatch.setattr(wm, "time", clock)
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "-proj" / "s.jsonl"
    source.parent.mkdir()
    health = tmp_path / "watcher-health.json"
    w = JSONLWatcher(
        watch_roots=[WatchRoot(provider, root)],
        registry_path=tmp_path / "offsets.json",
        health_path=health,
        max_lines_per_file=1000,
    )
    for record, count in [
        (None, 0),
        ({"kind": "unknown"}, 199),
        ({"type": "assistant", "message": {"content": []}}, 250),
    ]:
        if record:
            with source.open("a") as f:
                f.write((json.dumps(record) + "\n") * count)
        w.poll_once()
        clock.mono += 1800
        clock.wall_shift += 1800
        w.poll_once()
        assert not any(r.startswith("growth_not_indexed:") for r in json.loads(health.read_text())["alert_reasons"])


def test_growth_reason_reaches_ingest_verdict(tmp_path):
    assert _ingest_states(tmp_path, 8, {"alert_reasons": ["growth_not_indexed:codex"]}) == (
        ["amber"] * 5 + ["red"] * 3 + ["red", "green", "green"]
    )


@pytest.mark.parametrize(
    "provider,record",
    [
        ("claude", {"type": "progress", "data": {"type": "tool_progress"}}),
        ("codex", {"type": "response_item", "payload": {"type": "function_call", "name": "read"}}),
        ("cursor", {"type": "tool", "content": "noise"}),
        ("cursor-agent-transcripts", {"type": "tool", "content": "noise"}),
        ("gemini", {"type": "GENERIC", "step_index": 1}),
    ],
)
def test_native_tool_records_are_recognised_before_content_filters(provider, record):
    assert wm.recognised_provider_entry(record, provider)


def test_only_the_growing_provider_alarms_and_scalar_lines_count(tmp_path, monkeypatch):
    clock = _Clock(wm.time)
    monkeypatch.setattr(wm, "time", clock)
    roots = [WatchRoot(p, tmp_path / p) for p in ("codex", "cursor", "unknown")]
    for root in roots:
        root.path.mkdir()
        (root.path / "s.jsonl").write_text("[]\n" * 200)
    health = tmp_path / "health.json"
    w = JSONLWatcher(watch_roots=roots, registry_path=tmp_path / "offsets.json", health_path=health)
    w.poll_once()
    (roots[1].path / "s.jsonl").write_text('{"type":"tool"}\n' * 200)
    w.poll_once()
    clock.wall_shift += 1800
    w.poll_once()
    reasons = json.loads(health.read_text())["alert_reasons"]
    assert reasons == ["growth_not_indexed:codex"]


def test_expired_samples_start_a_fresh_observation_window(tmp_path, monkeypatch):
    clock = _Clock(wm.time)
    monkeypatch.setattr(wm, "time", clock)
    root = tmp_path / "sessions"
    root.mkdir()
    source = root / "s.jsonl"
    source.write_text('{"kind":"unknown"}\n')
    health = tmp_path / "health.json"
    w = JSONLWatcher(
        watch_roots=[WatchRoot("codex", root)],
        registry_path=tmp_path / "offsets.json",
        health_path=health,
        max_lines_per_file=1000,
    )
    w.poll_once()
    clock.wall_shift += 2100  # old sample expires, then a new burst starts
    with source.open("a") as f:
        f.write('{"kind":"unknown"}\n' * 200)
    w.poll_once()
    assert json.loads(health.read_text())["alert_reasons"] == []
    clock.wall_shift += 1800
    w.poll_once()
    assert json.loads(health.read_text())["alert_reasons"] == ["growth_not_indexed:codex"]


ROLE_NOISE = {
    "claude": {"message": {"role": "assistant", "content": [{"type": "tool_use", "name": "read"}]}},
    "codex": {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": []}},
    "cursor": {"role": "assistant", "content": [{"type": "tool_use", "name": "read"}]},
    "cursor-agent-transcripts": {"speaker": "assistant", "content": []},
    "gemini": {"author": {"role": "model"}, "content": []},
}


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("field", ("role", "sender", "speaker", "author", "message"))
def test_role_only_recognition_precedes_text_extraction(provider, field):
    role = "model" if provider == "gemini" else "assistant"
    record = {field: {"role": role} if field in ("author", "message") else role, "content": []}
    assert wm.recognised_provider_entry(record, provider)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_role_only_tool_noise_never_alarms_on_the_real_poll_path(tmp_path, monkeypatch, provider):
    clock = _Clock(wm.time)
    monkeypatch.setattr(wm, "time", clock)
    root = tmp_path / "sessions"
    source = root / "-proj" / "s.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text((json.dumps(ROLE_NOISE[provider]) + "\n") * 200)
    health = tmp_path / "health.json"
    w = JSONLWatcher(
        watch_roots=[WatchRoot(provider, root)],
        registry_path=tmp_path / "offsets.json",
        health_path=health,
        max_lines_per_file=1000,
    )
    w.poll_once()
    assert json.loads(health.read_text())["source_growth"][provider]["recognised_lines"] == 200
    clock.wall_shift += 1800
    w.poll_once()
    assert not any(r.startswith("growth_not_indexed:") for r in json.loads(health.read_text())["alert_reasons"])


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("entry_type", ("user", "assistant"))
def test_typed_empty_messages_keep_the_normalisers_early_branch(provider, entry_type):
    record = {"type": entry_type, "message": {}}
    assert wm.normalize_provider_entry(record, provider) is not None
    assert wm.recognised_provider_entry(record, provider)


@pytest.mark.parametrize("role", ("user", "assistant", "model", "gemini", "ai", "bot"))
def test_all_normaliser_role_aliases_are_recognised_without_text(role):
    assert wm.recognised_provider_entry({"role": role.upper(), "content": []}, "gemini")


@pytest.mark.parametrize("provider", PROVIDERS)
def test_empty_payload_falls_back_to_the_top_level_role(provider):
    record = {"role": "assistant", "content": "tool output", "payload": {}}
    assert wm.normalize_provider_entry(record, provider) is not None
    assert wm.recognised_provider_entry(record, provider)
