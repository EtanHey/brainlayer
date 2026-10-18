"""Opus convergence regressions: real temp SQLite, synthetic vectors, no live services."""

import json
from dataclasses import replace

import pytest
from typer.testing import CliRunner

import brainlayer.cli as cli
import brainlayer.index_new as adapter
from brainlayer.runtime_store import ReadonlyStore
from tests.test_index_checkpoint_boundary import cap_after_commit, checkpoint, synthetic_vectors
from tests.test_index_completion import calendar_fixture as calendar_fixture
from tests.test_index_completion import readback


def invoke(source):
    return CliRunner().invoke(cli.app, ["index", str(source)])


def test_checkpointed_backlog_is_persisted_and_shrinks(calendar_fixture, monkeypatch):
    source, db, chunks, clock = calendar_fixture
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    cap_after_commit(monkeypatch, clock)
    for remaining in (2, 0):
        clock[0] = 100.0
        result = invoke(source)
        assert result.exit_code == 0, result.output
        assert f"chunks_remaining={remaining}" in result.output
        assert f"files_remaining={int(remaining > 0)}" in result.output
        state = json.loads(next(db.parent.glob("index-backlog-*.json")).read_text())
        assert state["chunks_remaining"] == remaining
        assert state["stalled_passes"] == 0


def test_stagnation_is_loud_after_configured_consecutive_passes(calendar_fixture, monkeypatch):
    source, db, chunks, clock = calendar_fixture
    monkeypatch.setenv("BRAINLAYER_INDEX_STALL_PASSES", "3")
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    cap_after_commit(monkeypatch, clock)
    # Source keeps growing faster than one 64-input pass: same remaining backlog each time.
    for number in range(4):
        if number:
            with (source / "a.jsonl").open("a") as f:
                f.write("{}\n")
            chunks.extend(replace(chunks[0], content=f"Growing input {len(chunks) + i}") for i in range(64))
        clock[0] = 100.0
        result = invoke(source)
        assert result.exit_code == (1 if number == 3 else 0), result.output
    assert "INDEX_BACKLOG_STALLED" in result.output
    state = json.loads(next(db.parent.glob("index-backlog-*.json")).read_text())
    assert state["stalled_passes"] == 3


@pytest.mark.parametrize("mutation", ["append", "rewrite", "truncate", "replace_inode"])
def test_append_resumes_but_mutation_replays_real_rows(calendar_fixture, monkeypatch, mutation):
    source, db, chunks, clock = calendar_fixture
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    with monkeypatch.context() as cap:
        cap_after_commit(cap, clock)
        first = invoke(source)
    assert first.exit_code == 0, first.output
    assert next(iter(checkpoint(db).values()))["next_chunk"] == 64
    path = source / "a.jsonl"
    if mutation == "append":
        with path.open("a") as f:
            f.write("{}\n")
        chunks.extend(replace(chunks[0], content=f"Appended input {i}") for i in range(2))
    elif mutation == "rewrite":
        path.write_text("[]\n")
    elif mutation == "truncate":
        path.write_text("")
    else:
        replacement = path.with_suffix(".replacement")
        replacement.write_text("{}\n")
        replacement.replace(path)
    embedded = []

    def trace(cs, **kw):
        embedded.extend(cs)
        return synthetic_vectors(cs)

    monkeypatch.setattr(adapter, "embed_chunks", trace)
    clock[0] = 100.0
    second = invoke(source)
    assert second.exit_code == 0, second.output
    assert embedded == (chunks[64:] if mutation == "append" else chunks)
    readback(db, chunks)
    clock[0] = 100.0
    assert invoke(source).exit_code == 0
    readback(db, chunks)


@pytest.mark.parametrize("corruption", ["", "{", "[]", '{"version":2,"files":{}}'])
def test_unreadable_checkpoint_warns_and_replays_idempotently(calendar_fixture, monkeypatch, corruption):
    source, db, chunks, _ = calendar_fixture
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    assert invoke(source).exit_code == 0
    next(db.parent.glob("index-checkpoint-*.json")).write_text(corruption)
    result = invoke(source)
    assert result.exit_code == 0, result.output
    assert "checkpoint" in result.output.lower() and "replay" in result.output.lower()
    readback(db, chunks)


def test_all_rejected_tail_is_consumed_and_counted(calendar_fixture, monkeypatch):
    source, db, chunks, _ = calendar_fixture
    chunks[:] = chunks[:65]
    chunks[-1] = replace(chunks[-1], content='MCP frame {"jsonrpc": "2.0", "method": "tools/call"}')
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    for _ in range(2):
        result = invoke(source)
        assert result.exit_code == 0, result.output
        saved = next(iter(checkpoint(db).values()))
        assert saved["done"] and saved["next_chunk"] == 65
    with ReadonlyStore(db) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (64,)
        assert store.conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    # The persisted pass record keeps admission accounting even when a repeat skips the file.
    assert next(iter(checkpoint(db).values()))["rejected_chunks"] == 1


def test_identical_encode_failure_quarantines_file_and_visits_later_files(calendar_fixture, monkeypatch):
    source, db, chunks, clock = calendar_fixture
    (source / "b.jsonl").write_text("{}\n")
    monkeypatch.setenv("BRAINLAYER_INDEX_QUARANTINE_PASSES", "3")
    calls = []

    def encode(cs, **kw):
        calls.append(True)
        if len(calls) <= 3:
            raise RuntimeError("identical synthetic encode failure")
        return synthetic_vectors(cs)

    monkeypatch.setattr(adapter, "embed_chunks", encode)
    for _ in range(3):
        clock[0] = 100.0
        result = invoke(source)
        assert result.exit_code == 1, result.output
    assert "quarantined_files=1" in result.output
    assert len(checkpoint(db)) == 1  # Later b.jsonl completes; poison a.jsonl has no skip offset.
    assert next(iter(checkpoint(db))).endswith("b.jsonl")
    state = json.loads(next(db.parent.glob("index-backlog-*.json")).read_text())
    assert len(state["quarantined"]) == 1
    # Rewriting the failing prefix retries it; appending cannot cure that prefix.
    (source / "a.jsonl").write_text("[]\n")
    clock[0] = 100.0
    result = invoke(source)
    assert result.exit_code == 0, result.output
    assert len(checkpoint(db)) == 2


def test_changed_failure_resets_quarantine_streak(calendar_fixture, monkeypatch):
    source, db, chunks, clock = calendar_fixture
    errors = iter(["first", "second", "first"])

    def broken(cs, **kw):
        raise RuntimeError(next(errors))

    monkeypatch.setattr(adapter, "embed_chunks", broken)
    for _ in range(3):
        clock[0] = 100.0
        assert invoke(source).exit_code == 1
    state = json.loads(next(db.parent.glob("index-backlog-*.json")).read_text())
    assert not state["quarantined"]
    assert next(iter(state["failures"].values()))["count"] == 1


def test_growing_poison_prefix_does_not_reset_quarantine(calendar_fixture, monkeypatch):
    source, db, chunks, clock = calendar_fixture

    def broken(cs, **kw):
        raise RuntimeError("stable encode failure")

    monkeypatch.setattr(adapter, "embed_chunks", broken)
    for _ in range(3):
        with (source / "a.jsonl").open("a") as f:
            f.write("{}\n")
        clock[0] = 100.0
        assert invoke(source).exit_code == 1
    state = json.loads(next(db.parent.glob("index-backlog-*.json")).read_text())
    assert len(state["quarantined"]) == 1


def test_corrupt_entry_and_backlog_are_discarded(calendar_fixture, monkeypatch):
    source, db, chunks, _ = calendar_fixture
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    assert invoke(source).exit_code == 0
    path = next(db.parent.glob("index-checkpoint-*.json"))
    saved = json.loads(path.read_text())
    next(iter(saved["files"].values()))["signature"] = 7
    path.write_text(json.dumps(saved))
    next(db.parent.glob("index-backlog-*.json")).write_text('{"catalog": [], "failures": 3}')
    result = invoke(source)
    assert result.exit_code == 0, result.output
    assert "replay required" in result.output
    readback(db, chunks)


@pytest.mark.parametrize("sidecar", ["checkpoint", "backlog"])
def test_invalid_saved_signature_warns_and_recovers(calendar_fixture, monkeypatch, sidecar):
    source, db, chunks, _ = calendar_fixture
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    assert invoke(source).exit_code == 0
    path = next(db.parent.glob(f"index-{sidecar}-*.json"))
    saved = json.loads(path.read_text())
    key = str((source / "a.jsonl").resolve())
    signature = cli._CalendarIndexCheckpoint.signature(source / "a.jsonl")
    if sidecar == "checkpoint":
        # A negative prefix length must never reach fingerprint()'s read loop.
        saved["files"][key]["signature"][1] = -1
    else:
        # Shape-valid history with a corrupt scalar used to fail on every begin().
        signature[1] = "invalid-size"
        saved["failures"][key] = dict(signature=signature, count=3, prefix_sha256="invalid")
    path.write_text(json.dumps(saved))
    result = invoke(source)
    assert result.exit_code == 0, result.output
    assert "Warning: unreadable index" in result.output
    readback(db, chunks)
    assert invoke(source).exit_code == 0


def test_non_newline_prefix_requires_full_replay(calendar_fixture, monkeypatch):
    source, db, chunks, clock = calendar_fixture
    path = source / "a.jsonl"
    path.write_text("{}")
    monkeypatch.setattr(adapter, "embed_chunks", synthetic_vectors)
    assert invoke(source).exit_code == 0
    with path.open("a") as f:
        f.write("\n{}\n")
    seen = []
    monkeypatch.setattr(adapter, "embed_chunks", lambda cs, **kw: seen.extend(cs) or synthetic_vectors(cs))
    clock[0] = 100.0
    assert invoke(source).exit_code == 0
    assert seen == chunks


def test_all_rejected_batch_never_loads_encoder(calendar_fixture, monkeypatch):
    source, db, chunks, _ = calendar_fixture
    chunks[:] = [replace(chunks[0], content='Recursive {"jsonrpc":"2.0","id":1,"method":"tools/call"}')]

    def forbidden(*a, **kw):
        pytest.fail("rejected input must not be embedded")

    monkeypatch.setattr(adapter, "embed_chunks", forbidden)
    result = invoke(source)
    assert result.exit_code == 0, result.output
    assert "rejected_chunks=1" in result.output
    assert next(iter(checkpoint(db).values()))["next_chunk"] == 1
    with ReadonlyStore(db) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone() == (0,)
