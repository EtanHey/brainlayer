"""Calendar index admission waits for a writer-idle window inside its own budget."""

import json
from types import SimpleNamespace

import apsw
import pytest
from typer.testing import CliRunner

import brainlayer.cli as cli
import brainlayer.index_new as index_new
import brainlayer.vector_store as storage
from brainlayer.index_admission import AdmissionBackoff, IndexAdmissionExhausted
from tests.test_cli_index_calendar import prepare


def busy(committed=0):
    """A busy rollback as upsert_chunks reports it: nothing past ``committed`` is durable."""
    exc = apsw.BusyError("database is locked")
    exc.committed_count = exc.committed_input_count = committed
    return exc


def backoff(clock, *, deadline=1000.0, rng=lambda: 1.0, **kw):
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    policy = AdmissionBackoff(deadline, monotonic=lambda: clock[0], sleep=sleep, rng=rng, **kw)
    return policy, sleeps


def test_backoff_is_exponential_jittered_and_capped():
    clock = [0.0]
    policy, sleeps = backoff(clock, base_s=1.0, cap_s=8.0, rng=lambda: 0.0)
    for _ in range(6):
        policy.wait(apsw.BusyError("database is locked"))
    # Jitter keeps half of the exponential step at rng=0; the cap bounds the step.
    assert sleeps == [0.5, 1.0, 2.0, 4.0, 4.0, 4.0]
    assert policy.retries == 6
    assert policy.waited_s == pytest.approx(15.5)


def test_admission_resets_the_backoff_streak():
    clock = [0.0]
    policy, sleeps = backoff(clock, base_s=1.0, cap_s=8.0)
    policy.wait(apsw.BusyError("busy"))
    policy.wait(apsw.BusyError("busy"))
    policy.admitted()
    policy.wait(apsw.BusyError("busy"))
    assert sleeps == [1.0, 2.0, 1.0]
    assert policy.retries == 3


def test_backoff_never_sleeps_into_the_attempt_reserve():
    clock = [0.0]
    policy, sleeps = backoff(clock, deadline=20.0, base_s=4.0, cap_s=64.0, attempt_reserve_s=10.0)
    cause = apsw.BusyError("database is locked")
    policy.wait(cause)  # 0 + 4 + 10 <= 20
    with pytest.raises(IndexAdmissionExhausted) as raised:
        policy.wait(cause)  # 4 + 8 + 10 > 20
    assert sleeps == [4.0]
    assert raised.value.cause is cause
    assert raised.value.retries == 1
    assert raised.value.waited_s == pytest.approx(4.0)


def test_an_expired_deadline_reraises_the_cause_for_cap_attribution():
    clock = [50.0]
    policy, sleeps = backoff(clock, deadline=50.0)
    cause = apsw.BusyError("write busy deadline exceeded")
    with pytest.raises(apsw.BusyError) as raised:
        policy.wait(cause)
    assert raised.value is cause
    assert sleeps == []


def test_run_retries_writer_in_use_then_admits():
    clock = [0.0]
    policy, sleeps = backoff(clock)
    calls = []

    def attempt():
        calls.append(clock[0])
        if len(calls) < 3:
            raise storage.WriterInUseError("resident writer")
        return "store"

    assert policy.run(attempt) == "store"
    assert len(calls) == 3 and len(sleeps) == 2


def _real_chunks(monkeypatch, count):
    from brainlayer.embeddings import EmbeddedChunk
    from brainlayer.pipeline.chunk import Chunk
    from brainlayer.pipeline.classify import ContentType, ContentValue

    chunks = [
        Chunk(f"Synthetic admission fixture {i}", ContentType.USER_MESSAGE, ContentValue.HIGH, {}, 30)
        for i in range(count)
    ]
    embeds = []

    def embed(batch, **kw):
        embeds.append(len(batch))
        return [EmbeddedChunk(chunk=c, embedding=[0.1] * 4) for c in batch]

    monkeypatch.setattr("brainlayer.index_new.embed_chunks", embed)
    return chunks, embeds


def test_busy_upsert_retries_without_reembedding(tmp_path, monkeypatch):
    chunks, embeds = _real_chunks(monkeypatch, 3)
    source = tmp_path / "s.jsonl"
    source.write_text("{}\n")
    upserts, waits = [], []

    class Store:
        def upsert_chunks(self, data, vectors, **kw):
            upserts.append(len(data))
            if len(upserts) < 3:
                raise busy()
            return len(data)

    count = index_new.index_chunks_to_sqlite(chunks, source_file=str(source), store=Store(), on_busy=waits.append)
    assert count == 3
    assert upserts == [3, 3, 3]
    assert embeds == [3]
    assert len(waits) == 2 and all(isinstance(e, apsw.BusyError) for e in waits)


def test_busy_upsert_without_policy_still_raises(tmp_path, monkeypatch):
    chunks, _ = _real_chunks(monkeypatch, 1)
    source = tmp_path / "s.jsonl"
    source.write_text("{}\n")

    class Store:
        def upsert_chunks(self, *a, **kw):
            raise apsw.BusyError("database is locked")

    with pytest.raises(apsw.BusyError):
        index_new.index_chunks_to_sqlite(chunks, source_file=str(source), store=Store())


def _calendar(tmp_path, monkeypatch, *, chunks, busy_upserts, budget_s=600):
    """A calendar pass over fakes: real adapter, fake store, fake sleep advancing the clock."""
    source, clock, alarms = prepare(tmp_path, monkeypatch)
    monkeypatch.setenv("BRAINLAYER_INDEX_MAX_RUNTIME_S", str(budget_s))
    real, _ = _real_chunks(monkeypatch, chunks)
    monkeypatch.setattr("brainlayer.pipeline.chunk.chunk_content", lambda entry: real)
    sleeps, upserts, committed = [], [], []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("brainlayer.index_admission._sleep", sleep)

    class Store:
        conn = SimpleNamespace(execute=lambda sql: None)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def upsert_chunks(self, data, vectors, **kw):
            upserts.append(len(data))
            if busy_upserts(len(upserts)):
                raise busy()
            committed.extend(d["id"] for d in data)
            return len(data)

    def open_store(*a, **kw):
        return Store()

    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", open_store)
    return source, clock, alarms, sleeps, committed, open_store


def test_busy_rollbacks_inside_the_budget_back_off_and_complete(tmp_path, monkeypatch):
    # The 10-19 shape: one batch commits, then upsert rollbacks while drain/hotlane write.
    source, _, alarms, sleeps, committed, _ = _calendar(
        tmp_path, monkeypatch, chunks=130, busy_upserts=lambda n: n in {2, 3, 4, 5, 6}
    )
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 0, result.output
    assert "INDEX_DEFERRED" not in result.output
    assert len(committed) == 130
    assert len(sleeps) == 5
    assert result.output.count("INDEX_ADMISSION_WAIT") == 5
    flat = " ".join(result.output.split())
    assert "backlog_before_chunks=130" in flat
    assert "backlog_delta_chunks=-130" in flat
    assert not alarms


def test_no_admission_within_budget_defers_with_receipt(tmp_path, monkeypatch):
    source, clock, alarms, sleeps, committed, _ = _calendar(
        tmp_path, monkeypatch, chunks=130, busy_upserts=lambda n: n >= 2
    )
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 75, result.output
    # Rich wraps receipts at the console width; tokens never split, so compare them flat.
    deferred = " ".join(result.output.split()).split("INDEX_DEFERRED ", 1)[1].split(" Index admission:")[0]
    assert deferred.split()[0] == "resident_writer_busy"
    assert "committed_chunks=64" in deferred
    assert f"admission_retries={len(sleeps)}" in deferred
    assert "backlog_before_chunks=130" in deferred
    assert "backlog_delta_chunks=-64" in deferred
    assert len(committed) == 64 and len(sleeps) > 5
    # The wait consumed the budget instead of quitting at the first rollback burst,
    # and the last sleep still left one attempt's reserve before the deadline.
    assert 100.0 + 600 - 60 <= clock[0] < 100.0 + 600
    checkpoint = json.loads(next(tmp_path.glob("index-checkpoint-*.json")).read_text())
    assert next(iter(checkpoint["files"].values()))["next_chunk"] == 64
    backlog = json.loads(next(tmp_path.glob("index-backlog-*.json")).read_text())
    assert backlog["backlog_before"]["chunks_remaining"] == 130
    assert backlog["chunks_remaining"] == 66
    assert backlog["deferred"] is True
    assert not alarms


def test_open_contention_waits_for_admission(tmp_path, monkeypatch):
    source, _, _, sleeps, committed, real_open = _calendar(
        tmp_path, monkeypatch, chunks=2, busy_upserts=lambda n: False
    )
    opens = []

    def contended(*a, **kw):
        opens.append(1)
        if len(opens) < 3:
            raise storage.WriterInUseError("resident writer")
        return real_open(*a, **kw)

    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", contended)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 0, result.output
    assert len(opens) == 3 and len(sleeps) == 2
    assert len(committed) == 2


def test_admission_keeps_the_bounded_busy_timeout(tmp_path, monkeypatch):
    source, *_ = _calendar(tmp_path, monkeypatch, chunks=1, busy_upserts=lambda n: n == 1)
    seen = []
    original = storage.temporary_write_busy_timeout_ms

    def recording(timeout_ms, **kw):
        seen.append(timeout_ms)
        return original(timeout_ms, **kw)

    monkeypatch.setattr(storage, "temporary_write_busy_timeout_ms", recording)
    result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 0, result.output
    assert seen == [1000]
