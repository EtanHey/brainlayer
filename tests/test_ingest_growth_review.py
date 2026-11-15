"""Review regressions: invent formats; never inspect a real source or database."""

import json
import math

import pytest

import brainlayer.watcher as wm
from brainlayer.watcher import JSONLWatcher, WatchRoot
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_ingest_growth import PROVIDERS, ROLE_NOISE
from tests.test_ingest_stall_alerts import _Clock

compress = _compress

CHANGED = {
    "claude": {"type": "assistant", "msg": {"role": "assistant", "content": "lost"}},
    "codex": {"type": "response_item", "payload": {"kind": "message", "body": "lost"}},
    "cursor": {"type": "assistant", "message": {"role": "assistant", "parts": ["lost"]}},
    "cursor-agent-transcripts": {"type": "assistant", "message": "lost"},
    "gemini": {"type": "USER_INPUT", "content": "lost"},
}


@pytest.fixture
def rig(tmp_path, monkeypatch):
    clock = _Clock(wm.time)
    monkeypatch.setattr(wm, "time", clock)
    root = tmp_path / "sessions"
    source = root / "-proj" / "s.jsonl"
    source.parent.mkdir(parents=True)
    source.touch()
    health = tmp_path / "health.json"

    def make(provider, limit=1000):
        return JSONLWatcher(
            watch_roots=[WatchRoot(p, root) for p in ([provider] if isinstance(provider, str) else provider)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: {i["_source_file"]: i["_line_end_offset"] for i in items},
            health_path=health,
            max_lines_per_file=limit,
            registry_flush_interval_s=0,
        )

    def append(record, count=1):
        with source.open("a") as f:
            f.write((json.dumps(record) + "\n") * count)

    def poll(w, seconds=0):
        clock.wall_shift += seconds
        clock.mono += seconds
        w.poll_once()
        return json.loads(health.read_text())

    return make, append, poll, source, health, clock


@pytest.mark.parametrize("provider", PROVIDERS)
def test_known_type_changed_body_and_metadata_cannot_mask_loss(rig, provider):
    make, append, poll, *_ = rig
    w = make(provider)
    poll(w)  # finish first ingest before observing new content
    append(CHANGED[provider], 200)
    append({"type": "summary", "summary": "bookkeeping"}, 200)
    if provider == "codex":
        append({"type": "event_msg", "payload": {"type": "token_count"}}, 200)
    poll(w)
    assert f"growth_not_indexed:{provider}" in poll(w, 1800)["alert_reasons"]
    append(ROLE_NOISE[provider])
    assert not poll(w)["alert_reasons"]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_first_ingest_backfill_is_excluded_until_caught_up_and_after_restart(rig, provider):
    make, append, poll, _, _, _ = rig
    append(CHANGED[provider], 300)
    w = make(provider, 100)
    assert not poll(w)["alert_reasons"]
    w = make(provider, 100)
    assert not poll(w, 1800)["alert_reasons"]
    assert not poll(w, 30)["alert_reasons"]
    assert not poll(w, 1800)["alert_reasons"]
    append(CHANGED[provider], 200)
    poll(w)
    poll(w, 30)
    assert f"growth_not_indexed:{provider}" in poll(w, 1800)["alert_reasons"]


@pytest.mark.parametrize("blank", (False, True))
def test_alarm_expires_after_six_hours_without_source_lines(rig, blank):
    make, append, poll, source, *_ = rig
    w = make("codex")
    poll(w)
    append(CHANGED["codex"], 200)
    poll(w)
    assert poll(w, 1800)["alert_reasons"]
    # Metadata is still activity, although it cannot mask malformed content.
    if blank:
        with source.open("a") as f:
            f.write("\n")
    else:
        append({"type": "session_meta", "payload": {"id": "invented"}})
    poll(w)
    assert poll(w, 21599)["alert_reasons"]
    assert not poll(w, 1)["alert_reasons"]


@pytest.mark.parametrize("field", ("since", "last_line_at", "sample"))
@pytest.mark.parametrize("invalid", (True, False, math.nan, math.inf, -math.inf, 10**400))
def test_invalid_restored_timestamps_reset_only_that_agent(rig, field, invalid):
    make, _, _, _, health, clock = rig
    state = {"since": clock.time(), "last_line_at": clock.time(), "samples": [[clock.time(), 200]], "alerting": True}
    valid = json.loads(json.dumps(state))
    if field == "sample":
        state["samples"][0][0] = invalid
    else:
        state[field] = invalid
    health.write_text(json.dumps({"source_growth": {"codex": state, "cursor": valid}}))
    restored = make(["codex", "cursor"])._growth
    assert "codex" not in restored and "cursor" in restored


def test_blank_lines_are_neutral(rig):
    make, _, poll, source, *_ = rig
    w = make("codex")
    poll(w)
    with source.open("a") as f:
        f.write("\n" * 250)
    assert poll(w)["source_growth"]["codex"]["lines_read"] == 0
    assert not poll(w, 1800)["alert_reasons"]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_changed_content_block_type_is_not_recognised(provider):
    record = {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "renamed_text", "text": "lost"}]},
    }
    assert not wm.recognised_provider_entry(record, provider)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_body_selection_cannot_fall_through_a_truthy_broken_content_field(provider):
    record = {"role": "assistant", "content": {"parts": ["lost"]}, "text": "unused fallback"}
    assert wm.normalize_provider_entry(record, provider) is None
    assert not wm.recognised_provider_entry(record, provider)


def test_backfill_mixed_batch_counts_only_new_records(rig):
    make, append, poll, *_ = rig
    append(CHANGED["codex"], 150)
    w = make("codex", 100)
    w.max_read_bytes_per_file = len((json.dumps(CHANGED["codex"]) + "\n").encode()) * 100
    poll(w)
    append(CHANGED["codex"], 200)
    w.max_lines_per_file = 1000
    w.max_read_bytes_per_file = 1048576
    assert poll(w)["source_growth"]["codex"]["lines_read"] == 200
    assert poll(w, 1800)["alert_reasons"]


def test_replacement_during_backfill_does_not_keep_an_obsolete_frontier(rig):
    make, append, poll, source, *_ = rig
    append(CHANGED["codex"], 1000)
    w = make("codex", 100)
    poll(w)
    source.write_text("")
    append(CHANGED["codex"], 200)
    poll(w)
    poll(w, 30)
    assert poll(w, 1800)["alert_reasons"]


def test_restart_at_zero_offset_preserves_the_original_backfill_frontier(rig):
    make, append, poll, *_ = rig
    append(CHANGED["claude"], 150)
    w = make("claude", 100)
    w.indexer.on_flush = lambda _items: None  # no confirmation: registry offset remains zero
    poll(w)
    append(CHANGED["claude"], 200)
    w = make("claude", 1000)
    assert poll(w)["source_growth"]["claude"]["lines_read"] == 200


def test_compressed_history_uses_decoded_backfill_boundary(tmp_path, monkeypatch, compress):

    from tests.test_codex_zst_watcher import watcher
    from tests.test_codex_zstd_stream import rollout

    clock = _Clock(wm.time)
    monkeypatch.setattr(wm, "time", clock)
    history = rollout() + (json.dumps(CHANGED["codex"]) + "\n").encode() * 300
    compress("history.jsonl.zst", history)
    w, _ = watcher(tmp_path, monkeypatch, max_lines=100)
    w.health_path = tmp_path / "health.json"
    for _ in range(5):
        w.poll_once()
        clock.wall_shift += 1800
        clock.mono += 1800
        state = json.loads(w.health_path.read_text())
        assert state["source_growth"]["codex"]["lines_read"] == 0
        assert not any(r.startswith("growth_not_indexed:") for r in state["alert_reasons"])


@pytest.mark.parametrize("new_lines", (False, True))
def test_backfill_survives_plain_to_archive_representation_change(tmp_path, monkeypatch, compress, new_lines):
    from tests.test_codex_zst_watcher import watcher
    from tests.test_codex_zstd_stream import rollout

    clock = _Clock(wm.time)
    monkeypatch.setattr(wm, "time", clock)
    history = rollout() + (json.dumps(CHANGED["codex"]) + "\n").encode() * 1000
    plain = tmp_path / "history.jsonl"
    plain.write_bytes(history)
    w, _ = watcher(tmp_path, monkeypatch, max_lines=100)
    w.health_path = tmp_path / "health.json"
    w.poll_once()
    extra = (json.dumps(CHANGED["codex"]) + "\n").encode() * 200 if new_lines else b""
    compress("history.jsonl.zst", history + extra)
    plain.unlink()
    w.max_lines_per_file = 2000
    w.poll_once()
    assert json.loads(w.health_path.read_text())["source_growth"]["codex"]["lines_read"] == (200 if new_lines else 0)
    clock.wall_shift += 1800
    clock.mono += 1800
    w.poll_once()
    assert (
        any(r.startswith("growth_not_indexed:") for r in json.loads(w.health_path.read_text())["alert_reasons"])
        == new_lines
    )


def test_restored_idle_alarm_expires_before_neutral_activity(rig):
    make, append, poll, source, health, clock = rig
    w = make("codex")
    poll(w)
    append(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "confirmed before restart " * 30}],
            },
        }
    )
    poll(w, 1)
    assert w.registry.get(str(source))[0] > 0
    saved = json.loads(health.read_text())
    old = clock.time() - 7 * 3600
    saved["source_growth"]["codex"].update(since=old, last_line_at=old, samples=[], alerting=True)
    health.write_text(json.dumps(saved))
    w = make("codex")
    append({"type": "session_meta", "payload": {"id": "invented"}})
    state = poll(w)
    assert "growth_not_indexed:codex" not in state["alert_reasons"]
    assert state["source_growth"]["codex"]["last_line_at"] == pytest.approx(clock.time(), rel=0, abs=1)


@pytest.mark.parametrize("restart", (False, True))
@pytest.mark.parametrize("extra_lines", (0, 200))
def test_backfill_survives_plain_archive_plain_with_new_inode(tmp_path, monkeypatch, compress, restart, extra_lines):
    from tests.test_codex_zst_watcher import watcher
    from tests.test_codex_zstd_stream import rollout

    clock = _Clock(wm.time)
    monkeypatch.setattr(wm, "time", clock)
    history = rollout() + (json.dumps(CHANGED["codex"]) + "\n").encode() * 1000
    plain = tmp_path / "history.jsonl"
    plain.write_bytes(history)
    replacement = tmp_path / "replacement.tmp"
    replacement.write_bytes(history + (json.dumps(CHANGED["codex"]) + "\n").encode() * extra_lines)
    assert replacement.stat().st_ino != plain.stat().st_ino
    w, _ = watcher(tmp_path, monkeypatch, max_lines=100)
    w.health_path = tmp_path / "health.json"
    w.poll_once()
    compress("history.jsonl.zst", history)
    plain.unlink()
    w.poll_once()
    if restart:
        w.stop()
        w = JSONLWatcher(
            watch_roots=[WatchRoot("codex", tmp_path)],
            registry_path=tmp_path / "offsets.json",
            on_flush=lambda items: {i["_source_file"]: i["_line_end_offset"] for i in items},
            health_path=tmp_path / "health.json",
            batch_size=1,
            max_lines_per_file=100,
        )
    replacement.rename(plain)
    w.max_lines_per_file = 2000
    w.poll_once()
    assert json.loads(w.health_path.read_text())["source_growth"]["codex"]["lines_read"] == extra_lines
    clock.wall_shift += 1800
    clock.mono += 1800
    w.poll_once()
    assert ("growth_not_indexed:codex" in json.loads(w.health_path.read_text())["alert_reasons"]) == bool(extra_lines)
