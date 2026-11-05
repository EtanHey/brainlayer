"""#655 realtime half: the Codex watcher must attribute each assistant turn to the
``turn_context`` model in force when it was written -- across polls, restarts, replays,
rewinds and replaced files -- and never to a stale earlier model.

Drives the production path: JSONLWatcher -> watcher_bridge flush -> synthetic scratch DB
(or the arbitrated queue). Every model and provider name is a synthetic fixture string.
"""

import json
import os
import sqlite3
from pathlib import Path

import pytest

from brainlayer.watcher import JSONLWatcher, WatchRoot, scan_codex_turn_context
from brainlayer.watcher_bridge import create_flush_callback

MODEL_A = "codex-fixture-model-a"
MODEL_B = "codex-fixture-model-b"
MODEL_C = "codex-fixture-model-c"

ASSISTANT_TEXTS = [
    "Decision {n}: keep the fixture retry budget at three attempts because the backend flakes under load.",
    "Root cause {n}: the fixture cache key omitted the tenant id, so two tenants shared one cached answer.",
    "Fix {n}: the fixture scheduler now drains the queue before rotating the lease, which stops double runs.",
]


def _say(n: int) -> str:
    return ASSISTANT_TEXTS[n % len(ASSISTANT_TEXTS)].format(n=n)


USER_TEXT = "Why does the fixture cache return another tenant's answer when both tenants query at once?"


def _meta(provider: str = "fixture-provider") -> dict:
    return {
        "timestamp": "2026-11-05T00:00:00Z",
        "type": "session_meta",
        "payload": {"id": "codex-fixture", "cwd": "/fixture/Gits/proj", "model_provider": provider},
    }


def _turn(model) -> dict:
    return {"timestamp": "2026-11-05T00:00:01Z", "type": "turn_context", "payload": {"model": model}}


def _assistant(text: str) -> dict:
    return {
        "timestamp": "2026-11-05T00:00:02Z",
        "type": "response_item",
        "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
    }


def _user(text: str) -> dict:
    return {
        "timestamp": "2026-11-05T00:00:03Z",
        "type": "response_item",
        "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
    }


def _encode(lines: list[dict]) -> bytes:
    return b"".join((json.dumps(line) + "\n").encode() for line in lines)


def _append(path: Path, lines: list[dict]) -> None:
    with path.open("ab") as handle:
        handle.write(_encode(lines))


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    sessions = home / ".codex" / "sessions"
    (sessions / "2026" / "11").mkdir(parents=True)
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(tmp_path / "no-t3-state.db"))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    return tmp_path, sessions


def _source(sessions: Path, name: str = "a") -> Path:
    return sessions / "2026" / "11" / f"rollout-2026-11-05-{name}.jsonl"


def _watcher(tmp_path: Path, sessions: Path, *, arbitrated: bool = False) -> JSONLWatcher:
    return JSONLWatcher(
        watch_roots=[WatchRoot("codex", sessions)],
        registry_path=tmp_path / "offsets.json",
        on_flush=create_flush_callback(db_path=tmp_path / "scratch.db", arbitrated=arbitrated),
        batch_size=1,
        flush_interval_ms=0,
        registry_flush_interval_s=0,
        health_path=tmp_path / "watcher-health.json",
    )


def _drain(watcher: JSONLWatcher) -> None:
    watcher.poll_once()
    watcher.indexer.flush()
    watcher.registry.flush()


def _rows(tmp_path: Path) -> dict[str, dict]:
    with sqlite3.connect(tmp_path / "scratch.db") as conn:
        rows = conn.execute("SELECT content, metadata FROM chunks").fetchall()
    return {content: json.loads(metadata) for content, metadata in rows}


def _models(tmp_path: Path) -> dict[str, str | None]:
    return {content: meta.get("model") for content, meta in _rows(tmp_path).items()}


def test_mixed_model_turns_in_one_read_attribute_each_assistant_to_its_own_turn(env):
    tmp_path, sessions = env
    source = _source(sessions)
    source.write_bytes(
        _encode([_meta(), _turn(MODEL_A), _user(USER_TEXT), _assistant(_say(0)), _turn(MODEL_B), _assistant(_say(1))])
    )

    _drain(_watcher(tmp_path, sessions))

    rows = _rows(tmp_path)
    assert rows[_say(0)]["model"] == MODEL_A
    assert rows[_say(1)]["model"] == MODEL_B
    assert rows[_say(0)]["model_source"] == "codex_turn_context"
    assert rows[_say(1)]["model_provider"] == "fixture-provider"
    # A human turn has no author model, but it does come from the session's provider.
    assert "model" not in rows[USER_TEXT] and "model_source" not in rows[USER_TEXT]
    assert rows[USER_TEXT]["model_provider"] == "fixture-provider"


def test_resume_mid_file_after_restart_keeps_the_latest_turn_model_not_the_first(env):
    tmp_path, sessions = env
    source = _source(sessions)
    source.write_bytes(_encode([_meta(), _turn(MODEL_A), _assistant(_say(0)), _turn(MODEL_B), _assistant(_say(1))]))
    _drain(_watcher(tmp_path, sessions))

    # The process restarts; the next assistant turn's turn_context is behind the offset.
    _append(source, [_assistant(_say(2))])
    restarted = _watcher(tmp_path, sessions)
    assert restarted.registry.get(str(source))[0] > 0
    _drain(restarted)

    assert _models(tmp_path)[_say(2)] == MODEL_B
    assert _rows(tmp_path)[_say(2)]["model_provider"] == "fixture-provider"


def test_resume_from_a_legacy_offset_entry_rebuilds_context_from_the_file(env):
    """A pre-generation registry entry ({offset, inode, mtime}) still resumes correctly."""
    tmp_path, sessions = env
    source = _source(sessions)
    prefix = _encode([_meta(), _turn(MODEL_C), _assistant(_say(0))])
    source.write_bytes(prefix + _encode([_assistant(_say(1))]))
    (tmp_path / "offsets.json").write_text(
        json.dumps({str(source): {"offset": len(prefix), "inode": os.stat(source).st_ino, "mtime": 1.0}})
    )

    _drain(_watcher(tmp_path, sessions))

    assert _models(tmp_path) == {_say(1): MODEL_C}


def test_replay_after_a_failed_read_does_not_reuse_the_context_from_the_end_of_that_read(env, monkeypatch):
    tmp_path, sessions = env
    source = _source(sessions)
    source.write_bytes(_encode([_meta(), _turn(MODEL_A), _assistant(_say(0))]))
    watcher = _watcher(tmp_path, sessions)
    _drain(watcher)

    _append(source, [_assistant(_say(1)), _turn(MODEL_B), _assistant(_say(2))])
    original = watcher._normalize_lines
    calls = 0

    def fail_once(filepath, lines):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("forced normalization failure")
        return original(filepath, lines)

    monkeypatch.setattr(watcher, "_normalize_lines", fail_once)
    monkeypatch.setattr("brainlayer.watcher.raise_alarm", lambda *_args, **_kwargs: None)
    _drain(watcher)  # fails after the read; the tailer is rolled back to the old offset
    assert _say(1) not in _models(tmp_path)
    _drain(watcher)  # replays the same bytes

    models = _models(tmp_path)
    assert models[_say(1)] == MODEL_A
    assert models[_say(2)] == MODEL_B


def test_truncation_rewind_drops_context_that_only_existed_in_the_removed_bytes(env):
    tmp_path, sessions = env
    source = _source(sessions)
    source.write_bytes(_encode([_meta(), _turn(MODEL_A), _assistant(_say(0)), _turn(MODEL_B), _assistant(_say(1))]))
    watcher = _watcher(tmp_path, sessions)
    _drain(watcher)

    # Checkpoint restore: the file shrinks to a version whose only turn has no turn_context.
    source.write_bytes(_encode([_meta(), _assistant(_say(2))]))
    _drain(watcher)

    assert _say(2) in _rows(tmp_path)
    assert _models(tmp_path)[_say(2)] is None


def test_replaced_inode_starts_from_the_new_file_not_the_old_context(env):
    tmp_path, sessions = env
    source = _source(sessions)
    source.write_bytes(_encode([_meta(), _turn(MODEL_A), _assistant(_say(0))]))
    watcher = _watcher(tmp_path, sessions)
    _drain(watcher)
    old_size = source.stat().st_size

    replacement = source.with_name("replacement.tmp")
    body = [_meta("fixture-provider-2"), _assistant(_say(1) + " " + "padding " * (old_size // 8))]
    replacement.write_bytes(_encode(body))
    assert replacement.stat().st_size > old_size  # not a shrink: only the inode tells
    os.replace(replacement, source)
    _drain(watcher)

    new_row = next(meta for content, meta in _rows(tmp_path).items() if content.startswith(_say(1)))
    assert "model" not in new_row
    assert new_row["model_provider"] == "fixture-provider-2"


def test_context_is_per_file(env):
    tmp_path, sessions = env
    first, second = _source(sessions, "a"), _source(sessions, "b")
    first.write_bytes(_encode([_meta(), _turn(MODEL_A), _assistant(_say(0))]))
    second.write_bytes(_encode([_meta("fixture-provider-2"), _assistant(_say(1))]))
    watcher = _watcher(tmp_path, sessions)
    _drain(watcher)

    _append(second, [_turn(MODEL_B), _assistant(_say(2))])
    _append(first, [_assistant(_say(3))])
    _drain(_watcher(tmp_path, sessions))

    rows = _rows(tmp_path)
    assert {content: meta.get("model") for content, meta in rows.items()} == {
        _say(0): MODEL_A,
        _say(1): None,
        _say(2): MODEL_B,
        _say(3): MODEL_A,
    }
    assert rows[_say(1)]["model_provider"] == rows[_say(2)]["model_provider"] == "fixture-provider-2"
    assert rows[_say(3)]["model_provider"] == "fixture-provider"


def test_turn_context_without_a_usable_model_makes_the_next_turn_unknown(env):
    tmp_path, sessions = env
    source = _source(sessions)
    source.write_bytes(_encode([_meta(), _turn(MODEL_A), _assistant(_say(0)), _turn("<synthetic>"), _assistant(_say(1))]))

    _drain(_watcher(tmp_path, sessions))

    assert _models(tmp_path) == {_say(0): MODEL_A, _say(1): None}


def test_prefix_scan_treats_an_unparseable_turn_context_as_unknown(tmp_path):
    source = tmp_path / "rollout.jsonl"
    source.write_bytes(_encode([_meta(), _turn(MODEL_A)]) + b'{"type": "turn_context", "payload": {"mod\n')

    context = scan_codex_turn_context(str(source), source.stat().st_size, source.stat().st_ino)

    assert context.model is None
    assert context.model_provider == "fixture-provider"


def test_prefix_scan_refuses_a_file_that_is_not_the_inode_being_read(tmp_path):
    source = tmp_path / "rollout.jsonl"
    source.write_bytes(_encode([_meta(), _turn(MODEL_A)]))

    context = scan_codex_turn_context(str(source), source.stat().st_size, source.stat().st_ino + 1)

    assert context.model is None and context.model_provider is None


def test_arbitrated_queue_carries_the_same_attribution(env):
    tmp_path, sessions = env
    source = _source(sessions)
    source.write_bytes(_encode([_meta(), _turn(MODEL_A), _assistant(_say(0))]))
    _drain(_watcher(tmp_path, sessions, arbitrated=True))
    _append(source, [_user(USER_TEXT), _assistant(_say(1)), _turn(MODEL_B), _assistant(_say(2))])
    _drain(_watcher(tmp_path, sessions, arbitrated=True))

    events = [
        json.loads(line)
        for queue_file in sorted((tmp_path / "queue").glob("*.jsonl"))
        for line in queue_file.read_text(encoding="utf-8").splitlines()
    ]
    by_content = {event["content"]: event["metadata"] for event in events}
    assert {content: meta.get("model") for content, meta in by_content.items()} == {
        _say(0): MODEL_A,
        USER_TEXT: None,
        _say(1): MODEL_A,
        _say(2): MODEL_B,
    }
    assert by_content[_say(2)]["model_source"] == "codex_turn_context"
    assert all(meta["model_provider"] == "fixture-provider" for meta in by_content.values())
