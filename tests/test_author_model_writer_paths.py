"""Peer falsification of #1639's author-model contract (Refs #655) on BOTH watcher writer paths.

#1639 proved ``metadata.model`` only through the direct (``arbitrated=False``) writer. Here every
contract test runs the same synthetic transcript through the direct writer AND through the
arbitrated writer queue (``enqueue_watcher_chunk`` -> queue JSONL -> ``drain_once``), then compares
what each one persisted.

Tests marked ``xfail(strict=True)`` are FINDINGS: each pins an invariant the head violates today.
They are strict, so a fix makes them XPASS and fails the suite until the marker is removed.

All model names, keys and transcripts are synthetic fixture strings. No real JSONL, no real DB.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import apsw
import pytest

from brainlayer.dedupe import SIMHASH_THRESHOLD, compute_dedupe_fields, hamming_distance
from brainlayer.drain import drain_once
from brainlayer.ingest.codex import parse_codex_session
from brainlayer.vector_store import VectorStore
from brainlayer.watcher import normalize_provider_entry
from brainlayer.watcher_bridge import create_flush_callback

MODEL_A = "claude-fixture-model-a"
MODEL_B = "claude-fixture-model-b"
CODEX_MODEL = "codex-fixture-model-x"

TEXT_USER = "Why does the fixture cache return another tenant's answer when both tenants query it at once?"
TEXT_A = "Decision: keep the fixture retry budget at three attempts because the staging backend flakes under load."
TEXT_B = "Root cause: the fixture cache key omitted the tenant id, so two tenants shared a single cached answer."
TEXT_SYNTH = "API Error: the fixture request was interrupted before any model produced a reply for this turn."
TEXT_NOMODEL = "Follow-up: the fixture dashboard keeps rendering stale rows until the next scheduled refresh closes."
TEXT_TOOL = "fixture tool output line: 42 rows scanned, 0 tenants mismatched, cache warm, checksum verified ok"
TEXT_CODEX = "Codex answer: the fixture migration needs a backfill step before the new tenant index can be trusted."

_WORDS = (
    "root cause analysis shows the retry budget stays at three attempts because the fixture backend flakes "
    "under heavy load and the cache key must include the tenant identifier so that two tenants never share "
    "one cached answer during concurrent queries while the dashboard keeps rendering stale rows until the "
    "next scheduled refresh window closes for everyone involved so we will ship the patch behind a flag then "
    "watch the error budget graphs closely before widening the rollout further"
).split()
NEAR_A = " ".join(_WORDS)
NEAR_B = " ".join(_WORDS[:51] + ["incorrectly"] + _WORDS[52:])


# ── transcript fixtures ─────────────────────────────────────────────────────


def _claude_path(tmp_path: Path, stem: str) -> Path:
    path = tmp_path / "home" / ".claude" / "projects" / "-Users-fixture-Gits-proj" / f"{stem}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _codex_path(tmp_path: Path, model_line: dict | None = None) -> Path:
    path = tmp_path / "home" / ".codex" / "sessions" / "2026" / "11" / "05" / "rollout-fixture.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [{"type": "session_meta", "payload": {"id": "cx", "cwd": "/Users/fixture/Gits/proj"}}]
    lines.append(model_line or {"type": "turn_context", "payload": {"model": CODEX_MODEL}})
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def _turn(kind: str, source: Path, content, offset: int, *, model=..., session: str = "fixture-session") -> dict:
    message = {"role": kind, "content": content if isinstance(content, list) else [{"type": "text", "text": content}]}
    if model is not ...:
        message["model"] = model
    return {
        "type": kind,
        "sessionId": session,
        "message": message,
        "timestamp": f"2026-11-05T00:00:{offset:02d}Z",
        "_source_file": str(source),
        "_line_end_offset": offset * 100,
    }


def _codex_reply(text: str) -> dict:
    payload = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
    return {"type": "response_item", "payload": payload}


def _codex_turn(source: Path, text: str, offset: int) -> dict:
    """A Codex rollout response_item as the realtime watcher normalizes it."""
    entry = normalize_provider_entry({**_codex_reply(text), "timestamp": f"2026-11-05T00:00:{offset:02d}Z"}, "codex")
    entry["_source_file"] = str(source)
    entry["_line_end_offset"] = offset * 100
    return entry


# ── the two writer paths ────────────────────────────────────────────────────


def _rows(db_path: Path) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, content, metadata, sender, seen_count, enrichment_model, enrichment_backend FROM chunks"
        ).fetchall()
    return [{**dict(row), "metadata": json.loads(row["metadata"])} for row in rows]


def _authors(db_path: Path) -> dict[str, tuple]:
    return {row["content"]: (row["sender"], row["metadata"].get("model", "<absent>")) for row in _rows(db_path)}


def _direct(tmp_path: Path, *batches: list[dict]) -> Path:
    db_path = tmp_path / "direct.db"
    VectorStore(db_path).close()
    flush = create_flush_callback(db_path=db_path, arbitrated=False)
    for batch in batches:
        flush(batch)
    return db_path


def _drain(tmp_path: Path, db_path: Path, queue_dir: Path) -> int:
    return drain_once(
        db_path=db_path,
        queue_dir=queue_dir,
        log_path=tmp_path / "drain.log",
        pause_sentinel_path=tmp_path / "no-pause-sentinel",
    )


def _queued_events(queue_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in sorted(queue_dir.glob("watcher-*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture
def arbitrated(tmp_path, monkeypatch):
    """Run batches through the arbitrated writer, return (db_path, queued events before drain)."""
    queue_dir = tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")

    def run(*batches: list[dict]) -> tuple[Path, list[dict]]:
        db_path = tmp_path / "arbitrated.db"
        VectorStore(db_path).close()
        flush = create_flush_callback(db_path=db_path, arbitrated=True)
        events: list[dict] = []
        for batch in batches:
            flush(batch)
            events.extend(_queued_events(queue_dir))
            assert _drain(tmp_path, db_path, queue_dir) >= 1
        return db_path, events

    return run


# ── contract: holds at head, fails on the pre-#1639 code ────────────────────


def test_mixed_claude_codex_batch_persists_identical_authors_through_direct_and_arbitrated_writers(
    tmp_path, arbitrated
):
    claude = _claude_path(tmp_path, "aaaaaaaa-mixed")
    codex = _codex_path(tmp_path)
    tool_result = [{"type": "tool_result", "tool_use_id": "toolu_fixture", "content": TEXT_TOOL}]
    batch = [
        _turn("user", claude, TEXT_USER, 1, model=MODEL_A),  # a user entry claiming a model is not authored by it
        _turn("assistant", claude, TEXT_A, 2, model=MODEL_A),
        _turn("assistant", claude, TEXT_B, 3, model=f"  {MODEL_B}\t"),
        _turn("assistant", claude, TEXT_SYNTH, 4, model="<synthetic>"),
        _turn("assistant", claude, TEXT_NOMODEL, 5),
        _turn("user", claude, tool_result, 6, model=MODEL_B),  # returned tool content is never model-authored
        _codex_turn(codex, TEXT_CODEX, 7),  # rollout named CODEX_MODEL on turn_context; watcher has no state
    ]
    expected = {
        TEXT_USER: ("user", "<absent>"),
        TEXT_A: ("assistant", MODEL_A),
        TEXT_B: ("assistant", MODEL_B),
        TEXT_SYNTH: ("assistant", "<absent>"),
        TEXT_NOMODEL: ("assistant", "<absent>"),
        TEXT_CODEX: ("assistant", "<absent>"),
    }

    direct_db = _direct(tmp_path, [dict(entry) for entry in batch])
    arbitrated_db, events = arbitrated([dict(entry) for entry in batch])

    direct, drained = _authors(direct_db), _authors(arbitrated_db)
    assert direct == drained
    assert {content: direct[content] for content in expected} == expected
    for content, (sender, model) in direct.items():
        if content not in expected:  # the tool_result, however it classified
            assert (sender, model) == ("user", "<absent>")
    # The queue event itself carries the author: a drain-side default cannot be what passes this.
    queued = {event["content"]: event["metadata"].get("model", "<absent>") for event in events}
    assert queued[TEXT_A] == MODEL_A and queued[TEXT_B] == MODEL_B
    assert {queued[TEXT_USER], queued[TEXT_SYNTH], queued[TEXT_NOMODEL], queued[TEXT_CODEX]} == {"<absent>"}
    for db_path in (direct_db, arbitrated_db):
        assert {(row["enrichment_model"], row["enrichment_backend"]) for row in _rows(db_path)} == {(None, None)}


def test_busy_direct_write_spills_author_model_through_queue_and_drain(tmp_path, monkeypatch):
    import brainlayer.watcher_bridge as bridge

    queue_dir = tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    monkeypatch.setenv("BRAINLAYER_WATCHER_WRITE_DEADLINE_S", "0")
    monkeypatch.setattr(bridge, "find_duplicate", lambda *_a, **_k: (_ for _ in ()).throw(apsw.BusyError()))
    claude = _claude_path(tmp_path, "aaaaaaaa-busy")
    db_path = tmp_path / "busy.db"
    VectorStore(db_path).close()

    create_flush_callback(db_path=db_path, arbitrated=False)(
        [_turn("assistant", claude, TEXT_A, 2, model=MODEL_A), _turn("user", claude, TEXT_USER, 3, model=MODEL_A)]
    )

    assert _authors(db_path) == {}  # rolled back: nothing direct-written
    assert {e["content"]: e["metadata"].get("model", "<absent>") for e in _queued_events(queue_dir)} == {
        TEXT_A: MODEL_A,
        TEXT_USER: "<absent>",
    }
    assert _drain(tmp_path, db_path, queue_dir) == 2
    assert _authors(db_path) == {TEXT_A: ("assistant", MODEL_A), TEXT_USER: ("user", "<absent>")}


def test_replayed_flush_and_replayed_queue_file_keep_one_row_and_its_author(tmp_path, monkeypatch):
    queue_dir = tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue_dir))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    claude = _claude_path(tmp_path, "aaaaaaaa-replay")
    turn = _turn("assistant", claude, TEXT_A, 2, model=MODEL_A)

    direct_db = _direct(tmp_path, [dict(turn)], [dict(turn)])  # offsets lost -> watcher replays the line

    arbitrated_db = tmp_path / "arbitrated.db"
    VectorStore(arbitrated_db).close()
    create_flush_callback(db_path=arbitrated_db, arbitrated=True)([dict(turn)])
    (queued,) = queue_dir.glob("watcher-*.jsonl")
    kept = tmp_path / "replay-copy.jsonl"
    shutil.copy(queued, kept)
    assert _drain(tmp_path, arbitrated_db, queue_dir) == 1
    shutil.copy(kept, queue_dir / queued.name)  # the same intent drained twice (crash before unlink)
    assert _drain(tmp_path, arbitrated_db, queue_dir) == 1

    for db_path in (direct_db, arbitrated_db):
        (row,) = _rows(db_path)
        assert (row["content"], row["metadata"]["model"], row["seen_count"]) == (TEXT_A, MODEL_A, 2)


def test_author_model_never_bleeds_across_interleaved_sessions_or_batches(tmp_path, arbitrated):
    s1, s2, s3 = (_claude_path(tmp_path, f"{c * 8}-session") for c in "abc")
    first = [
        _turn("assistant", s1, TEXT_A, 1, model=MODEL_A, session="s1"),
        _turn("assistant", s2, TEXT_NOMODEL, 2, session="s2"),
        _turn("assistant", s3, TEXT_B, 3, model=MODEL_B, session="s3"),
    ]
    second = [_turn("assistant", s2, TEXT_SYNTH, 4, model="<synthetic>", session="s2")]
    expected = {
        TEXT_A: ("assistant", MODEL_A),
        TEXT_NOMODEL: ("assistant", "<absent>"),
        TEXT_B: ("assistant", MODEL_B),
        TEXT_SYNTH: ("assistant", "<absent>"),
    }

    assert _authors(_direct(tmp_path, first, second)) == expected
    assert _authors(arbitrated(first, second)[0]) == expected


_LONG = "m" * 1_048_576


@pytest.mark.parametrize(
    ("raw", "stored"),
    [
        ("x" * 128, "x" * 128),
        ("x" * 129, None),
        (_LONG, None),
        ("fixture‮model", None),  # RIGHT-TO-LEFT OVERRIDE
        ("fixture⁦model⁩", None),  # LEFT-TO-RIGHT ISOLATE ... POP
        ("fixture‍model", None),  # ZERO WIDTH JOINER
        ("﻿fixture-model", None),  # BOM
        ("fixture model", None),  # NO-BREAK SPACE inside
        ("fixture\udc00model", None),  # lone surrogate
        ("fixture\x00model", None),
        ("fixture\tmodel", None),
        (True, None),
        ({"id": MODEL_A}, None),
        ("аnthropic-fixture-ḿodel", "аnthropic-fixture-ḿodel"),  # homoglyph+combining: verbatim
    ],
    ids=[
        "len128-kept",
        "len129",
        "1MiB",
        "rlo",
        "isolate",
        "zwj",
        "bom",
        "nbsp",
        "surrogate",
        "nul",
        "tab",
        "bool",
        "dict",
        "homoglyph-verbatim",
    ],
)
def test_adversarial_model_names_are_stored_verbatim_or_unknown_identically_on_both_paths(
    tmp_path, arbitrated, raw, stored
):
    claude = _claude_path(tmp_path, "aaaaaaaa-names")
    batch = [_turn("assistant", claude, TEXT_A, 2, model=raw)]

    expected = {TEXT_A: ("assistant", "<absent>" if stored is None else stored)}
    assert _authors(_direct(tmp_path, [dict(batch[0])])) == expected
    assert _authors(arbitrated([dict(batch[0])])[0]) == expected


def test_secret_shaped_model_string_is_scrubbed_on_both_paths(tmp_path, arbitrated):
    fake_key = "sk-" + "ant-" + "SyntheticOnlyAbCdEf0123456789xyz"  # assembled at runtime; not a real key
    claude = _claude_path(tmp_path, "aaaaaaaa-secret")
    batch = [_turn("assistant", claude, TEXT_A, 2, model=fake_key)]

    for db_path in (_direct(tmp_path, [dict(batch[0])]), arbitrated([dict(batch[0])])[0]):
        with sqlite3.connect(db_path) as conn:
            dumped = "\n".join(str(row) for row in conn.execute("SELECT * FROM chunks"))
        assert fake_key not in dumped


def test_copied_content_keeps_each_turns_own_author(tmp_path, arbitrated):
    """A verbatim copy in the same session folds into the FIRST writer's row (same chunk_id), on both paths.

    So an assistant quoting the user never makes the user's text model-authored, and a user pasting the
    answer back never strips or reassigns the answer's author. First writer wins; nothing is upgraded.
    """
    claude = _claude_path(tmp_path, "aaaaaaaa-copied")
    batch = [
        _turn("user", claude, TEXT_USER, 1),
        _turn("assistant", claude, TEXT_USER, 2, model=MODEL_A),
        _turn("assistant", claude, TEXT_B, 3, model=MODEL_B),
        _turn("user", claude, TEXT_B, 4),
    ]

    for db_path in (_direct(tmp_path, [dict(e) for e in batch]), arbitrated([dict(e) for e in batch])[0]):
        rows = sorted(
            (r["content"], r["sender"], r["metadata"].get("model", "<absent>"), r["seen_count"]) for r in _rows(db_path)
        )
        assert rows == sorted([(TEXT_USER, "user", "<absent>", 2), (TEXT_B, "assistant", MODEL_B, 2)])


# ── FINDINGS: invariants the head violates (strict xfail) ───────────────────


@pytest.mark.parametrize("path", ["direct", "arbitrated"])
@pytest.mark.parametrize("second_author", ["codex-unknown", "claude-model-b"])
@pytest.mark.xfail(
    strict=True,
    reason="F1: a near-duplicate merge folds another author's text into a row that keeps the first author's "
    "metadata.model (dedupe.merge_duplicate_chunk never reconciles metadata)",
)
def test_near_duplicate_merge_never_attributes_another_authors_text_to_the_first_model(
    tmp_path, arbitrated, path, second_author
):
    first_distance = hamming_distance(
        compute_dedupe_fields(NEAR_A, "2026-11-05T00:00:01Z").simhash,
        compute_dedupe_fields(NEAR_B, "2026-11-05T00:00:01Z").simhash,
    )
    assert 0 < first_distance <= SIMHASH_THRESHOLD  # precondition: a real near-duplicate, not an exact one
    first = [_turn("assistant", _claude_path(tmp_path, "aaaaaaaa-first"), NEAR_A, 1, model=MODEL_A, session="s1")]
    if second_author == "codex-unknown":
        second = [_codex_turn(_codex_path(tmp_path), NEAR_B, 5)]
    else:
        second = [_turn("assistant", _claude_path(tmp_path, "bbbbbbbb-other"), NEAR_B, 5, model=MODEL_B, session="s2")]

    db_path = _direct(tmp_path, first, second) if path == "direct" else arbitrated(first, second)[0]

    (holder,) = [row for row in _rows(db_path) if "incorrectly" in row["content"]]
    assert holder["metadata"].get("model") != MODEL_A, holder["metadata"]


@pytest.mark.xfail(
    strict=True,
    reason="F2: codex.parse_codex_session resets session_id on a later session_meta but keeps the previous "
    "session's turn_context model",
)
def test_codex_session_boundary_does_not_inherit_the_previous_sessions_model(tmp_path):
    lines = [
        {"type": "session_meta", "payload": {"id": "S1", "model_provider": "openai"}},
        {"type": "turn_context", "payload": {"model": CODEX_MODEL}},
        _codex_reply(TEXT_A),
        {"type": "session_meta", "payload": {"id": "S2", "model_provider": "openai"}},
        _codex_reply(TEXT_B),
    ]
    session = tmp_path / "rollout.jsonl"
    session.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

    by_text = {chunk["content"]: chunk["metadata"] for chunk in parse_codex_session(session)}

    assert by_text[TEXT_A]["model"] == CODEX_MODEL  # control: the first session is attributed
    assert by_text[TEXT_B]["session_id"] == "S2"
    assert by_text[TEXT_B]["model"] is None


@pytest.mark.parametrize(
    "provider",
    [..., {"name": "x" * 300}, "fixture‮provider", "p" * 4096],
    ids=["absent-defaults-to-openai", "dict", "bidi", "oversized"],
)
@pytest.mark.xfail(
    strict=True,
    reason="F3: codex model_provider is stored unvalidated on every chunk, and an absent one is inferred as 'openai'",
)
def test_codex_model_provider_is_validated_like_model_and_never_inferred(tmp_path, provider):
    meta = {"id": "S1"}
    if provider is not ...:
        meta["model_provider"] = provider
    lines = [
        {"type": "session_meta", "payload": meta},
        _codex_reply(TEXT_A),
    ]
    session = tmp_path / "rollout.jsonl"
    session.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

    (chunk,) = list(parse_codex_session(session))

    assert chunk["metadata"].get("model_provider") is None
