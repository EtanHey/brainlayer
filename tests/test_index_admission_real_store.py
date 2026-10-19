"""Admission retries on a REAL store replay only the uncommitted suffix.

upsert_chunks commits BRAINLAYER_INDEX_TXN_BATCH-sized sub-batches independently.
A busy rollback on a later sub-batch keeps the committed prefix, so retrying the
whole call would replay it: same-ID merges bump seen_count, alias merges bump the
canonical count and add audit rows. Every case here compares against a
contention-free control on an isolated temp DB, with a real APSW lock contender.
"""

import json
from dataclasses import replace

import apsw
import pytest
from typer.testing import CliRunner

import brainlayer.cli as cli
import brainlayer.index_new as adapter
import brainlayer.runtime_store as runtime
import brainlayer.vector_store as storage
from brainlayer.embeddings import EmbeddedChunk
from brainlayer.index_admission import AdmissionBackoff, IndexAdmissionExhausted
from brainlayer.pipeline.chunk import Chunk
from brainlayer.pipeline.classify import ContentType, ContentValue
from brainlayer.vector_store import VectorStore

TXN = 16
COUNT = 40
# In the committed prefix: an alias of input 2 and of input 3 (one crosses into the suffix),
# plus an alias pair entirely inside the suffix.
DUPLICATES = {5: 2, 20: 3, 33: 21}


def inputs():
    chunks = [
        Chunk(f"Synthetic admission replay input {i}", ContentType.USER_MESSAGE, ContentValue.HIGH, {}, 34)
        for i in range(COUNT)
    ]
    for dup, original in DUPLICATES.items():
        chunks[dup] = replace(chunks[dup], content=chunks[original].content)
    return chunks


def vectors(chunks, **kw):
    return [EmbeddedChunk(c, [0.1] * 1024) for c in chunks]


def fingerprint(db):
    """Stored state with ids reduced to input positions, so twin DBs compare directly."""

    def ids(rows):
        return sorted(
            tuple(v.rsplit(":", 1)[1] if isinstance(v, str) and ".jsonl:" in v else v for v in r) for r in rows
        )

    with runtime.ReadonlyStore(db) as store:
        conn = store.conn
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        state = dict(
            chunks=ids(
                conn.execute(
                    "SELECT id, position, content, COALESCE(seen_count, 1), archived_at IS NOT NULL FROM chunks"
                )
            ),
            aliases=ids(conn.execute("SELECT old_chunk_id, canonical_chunk_id FROM chunk_id_alias")),
            audit=ids(conn.execute("SELECT chunk_id_dropped, chunk_id_kept, mechanism FROM dedupe_audit")),
            vectors=conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0],
        )
    fts = runtime.check_fts_counts(db)
    assert fts["status"] == "ok", fts
    return state


@pytest.fixture
def store_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(tmp_path / "pidfiles"))
    monkeypatch.setenv("BRAINLAYER_INDEX_TXN_BATCH", str(TXN))
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_PATH", str(tmp_path / "telemetry.jsonl"))
    # A real lock, but millisecond waits: keep the real busy path fast.
    monkeypatch.setattr(storage, "_write_busy_timeout_ms", lambda: 5)
    monkeypatch.setattr(runtime, "_write_busy_timeout_ms", lambda: 5)
    monkeypatch.setattr(storage, "_sleep", lambda seconds: None)
    monkeypatch.setattr(adapter, "embed_chunks", vectors)

    def make(name):
        root = tmp_path / name
        (root / "projects").mkdir(parents=True)
        (root / "projects" / "a.jsonl").write_text("{}\n")
        db = root / "brainlayer.db"
        with VectorStore(db) as store:
            store.conn.execute("CREATE TABLE contender_log(n INTEGER)")
        return root, db

    return make


class Contender:
    """A drain-like writer that takes the real write lock as sub-batch N begins."""

    def __init__(self, monkeypatch, db, *, at_span):
        self.conn = apsw.Connection(str(db))
        self.holding = False
        self.commits = 0
        spans = []
        real_span = storage.start_writer_span

        def span(*args, **kwargs):
            if kwargs.get("operation") == "upsert_chunks":
                spans.append(1)
                if len(spans) == at_span:
                    self.conn.execute("BEGIN IMMEDIATE")
                    self.holding = True
            return real_span(*args, **kwargs)

        monkeypatch.setattr(storage, "start_writer_span", span)

    def release(self, *_):
        # Prove the contender really writes while the index is backing off.
        if self.holding:
            self.conn.execute("INSERT INTO contender_log VALUES (1)")
            self.conn.execute("COMMIT")
            self.holding, self.commits = False, self.commits + 1

    def close(self):
        if self.holding:
            self.conn.execute("ROLLBACK")
        self.conn.close()


def run_adapter(db, source, chunks, *, on_busy=None):
    with runtime.open_writer_store(db) as store:
        return adapter.index_chunks_to_sqlite(chunks, source_file=str(source), store=store, on_busy=on_busy)


def test_adapter_retry_replays_only_the_uncommitted_suffix(store_env, monkeypatch):
    chunks = inputs()
    control_root, control_db = store_env("control")
    assert run_adapter(control_db, control_root / "projects" / "a.jsonl", chunks) == COUNT

    root, db = store_env("contended")
    contender = Contender(monkeypatch, db, at_span=2)
    waits = []
    try:
        count = run_adapter(
            db, root / "projects" / "a.jsonl", chunks, on_busy=lambda exc: (waits.append(exc), contender.release())
        )
    finally:
        contender.close()
    assert count == COUNT
    assert len(waits) == 1 and contender.commits == 1
    # The wait came after the first sub-batch committed, and it was durable.
    assert waits[0].committed_count == TXN
    control = fingerprint(control_db)
    assert fingerprint(db) == control
    # The control itself merges the legitimate duplicates exactly once each.
    seen = {row[0]: row[3] for row in control["chunks"]}
    assert all(seen[str(original)] == 2 for original in DUPLICATES.values())
    assert len(control["audit"]) == len(DUPLICATES)


def test_adapter_exhaustion_reports_the_durable_prefix(store_env, monkeypatch):
    chunks = inputs()
    root, db = store_env("contended")
    contender = Contender(monkeypatch, db, at_span=2)
    clock = [0.0]
    policy = AdmissionBackoff(30.0, monotonic=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0] + s))
    try:
        with pytest.raises(IndexAdmissionExhausted) as raised:
            run_adapter(db, root / "projects" / "a.jsonl", chunks, on_busy=policy.wait)
    finally:
        contender.close()
    assert policy.retries >= 1
    assert raised.value.cause.committed_count == TXN
    assert raised.value.cause.next_position == TXN
    state = fingerprint(db)
    assert [row[0] for row in state["chunks"]] == sorted(str(i) for i in range(TXN) if i != 5)
    assert all(row[3] == (2 if row[0] == "2" else 1) for row in state["chunks"])


def calendar(monkeypatch, root, db, clock, *, budget_s=60):
    monkeypatch.setattr(cli, "get_db_path", lambda: db)
    monkeypatch.setattr(cli, "_calendar_index_budget_s", lambda: budget_s)
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("BRAINLAYER_LAUNCHD_SERVICE", "index")
    monkeypatch.setattr("brainlayer.pipeline.extract.parse_jsonl", lambda p: [{}])
    monkeypatch.setattr("brainlayer.pipeline.classify.classify_content", lambda x: x)
    chunks = inputs()
    monkeypatch.setattr("brainlayer.pipeline.chunk.chunk_content", lambda x: chunks)
    return root / "projects"


def checkpoint_offset(db):
    return next(iter(json.loads(next(db.parent.glob("index-checkpoint-*.json")).read_text())["files"].values()))


def flat(output):
    return " ".join(output.split())


def control_pass(store_env, monkeypatch, chunks_upto=COUNT):
    root, db = store_env(f"control-{chunks_upto}")
    with monkeypatch.context() as m:
        clock = [100.0]
        source = calendar(m, root, db, clock)
        truncated = inputs()[:chunks_upto]
        m.setattr("brainlayer.pipeline.chunk.chunk_content", lambda x: truncated)
        result = CliRunner().invoke(cli.app, ["index", str(source)])
    assert result.exit_code == 0, result.output
    return fingerprint(db)


def test_calendar_contention_that_clears_matches_control(store_env, monkeypatch):
    control = control_pass(store_env, monkeypatch)
    root, db = store_env("contended")
    clock = [100.0]
    source = calendar(monkeypatch, root, db, clock)
    contender = Contender(monkeypatch, db, at_span=2)

    def sleep(seconds):
        clock[0] += seconds
        contender.release()

    monkeypatch.setattr("brainlayer.index_admission._sleep", sleep)
    try:
        result = CliRunner().invoke(cli.app, ["index", str(source)])
    finally:
        contender.close()
    assert result.exit_code == 0, result.output
    assert "INDEX_ADMISSION_WAIT" in result.output and contender.commits == 1
    assert f"Indexed {COUNT} chunks" in flat(result.output)
    assert checkpoint_offset(db)["done"] is True
    assert fingerprint(db) == control


def test_calendar_persistent_contention_checkpoints_the_durable_prefix(store_env, monkeypatch):
    prefix_control = control_pass(store_env, monkeypatch, chunks_upto=TXN)
    full_control = control_pass(store_env, monkeypatch)
    root, db = store_env("contended")
    clock = [100.0]
    source = calendar(monkeypatch, root, db, clock)
    contender = Contender(monkeypatch, db, at_span=2)
    monkeypatch.setattr("brainlayer.index_admission._sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    try:
        first = CliRunner().invoke(cli.app, ["index", str(source)])
        assert first.exit_code == 75, first.output
        receipt = flat(first.output)
        assert f"INDEX_DEFERRED resident_writer_busy committed_chunks={TXN}" in receipt
        assert f"backlog_delta_chunks=-{TXN}" in receipt
        saved = checkpoint_offset(db)
        assert saved["next_chunk"] == TXN and saved["done"] is False
        backlog = json.loads(next(db.parent.glob("index-backlog-*.json")).read_text())
        assert backlog["failures"] == {} and backlog["quarantined"] == []
        assert fingerprint(db) == prefix_control
    finally:
        contender.close()
    # The writer-idle window opens: the next pass resumes at the durable offset only.
    clock[0] = 100.0
    second = CliRunner().invoke(cli.app, ["index", str(source)])
    assert second.exit_code == 0, second.output
    assert f"Indexed {COUNT - TXN} chunks" in flat(second.output)
    assert fingerprint(db) == full_control


def test_busy_without_known_progress_is_not_retried(store_env, monkeypatch):
    root, db = store_env("unknown")
    waits = []

    class Store:
        def upsert_chunks(self, *a, **kw):
            raise apsw.BusyError("database is locked")  # no committed-prefix annotation

    with pytest.raises(apsw.BusyError):
        adapter.index_chunks_to_sqlite(
            inputs()[:2], source_file=str(root / "projects" / "a.jsonl"), store=Store(), on_busy=waits.append
        )
    assert waits == []
