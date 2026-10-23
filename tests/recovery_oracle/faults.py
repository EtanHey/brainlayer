"""Owned child-process fault cuts; no service lifecycle or resident socket access."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from brainlayer import drain
from brainlayer.embeddings import EmbeddedChunk
from brainlayer.index_new import index_chunks_to_sqlite
from brainlayer.pipeline.chunk import Chunk
from brainlayer.pipeline.classify import ContentType, ContentValue
from brainlayer.vector_store import VectorStore

from .core import Universe, private_path, validate_environment


def _pause(root):
    private_path(root, root / "cut-ready").write_text("controlled cut reached")
    while True:
        time.sleep(0.01)


def _index(root, seed, store, cut=False):
    owner = SimpleNamespace(seed=seed)
    source = private_path(root, root / "input.jsonl")
    source.write_text('{"timestamp":"2026-10-22T00:00:00Z"}\n')
    chunks = [
        Chunk(
            Universe.text(owner, name),
            ContentType.USER_MESSAGE,
            ContentValue.HIGH,
            {"chunk_id": name, "source": "oracle-index", "source_class": "cli-agent"},
            160,
        )
        for name in ("indexa", "indexb")
    ]
    if cut:

        def trace(_cursor, statement, bindings):
            if "INSERT INTO CHUNKS" in statement.upper() and bindings and bindings[0] == "indexb":
                _pause(root)
            return True

        store.conn.setexectrace(trace)
    return index_chunks_to_sqlite(
        chunks,
        str(source),
        project="oracle",
        db_path=root / "private.db",
        store=store,
        precomputed_embeddings=[EmbeddedChunk(c, Universe.encode(owner, c.content)) for c in chunks],
    )


def index_batch(world):
    world.check()
    assert _index(world.root, world.seed, world.writer) == 2
    for name in ("indexa", "indexb"):
        assert world.writer.conn.execute("SELECT COUNT(*) FROM chunks WHERE id=?", (name,)).fetchone() == (1,)
        assert (name,) in list(
            world.writer.conn.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ?", (f"oracle{name}",))
        )
        result = world.writer.search(query_embedding=world.encode(world.text(name)), n_results=1)
        assert result["ids"][0] == [name]
        assert result["distances"][0][0] < 1e-5


def cut_worker(world, action):
    world.check()
    ready = world.guard(world.root / "cut-ready")
    log_path = world.guard(world.root / "child.log")
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo), str(repo / "src"), env.get("PYTHONPATH", "")])
    env["BRAINLAYER_INDEX_TXN_BATCH"] = "1"
    world.writer.close()
    process = None
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "tests.recovery_oracle.faults", str(world.root), str(world.seed), action],
                cwd=repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 8
            while not ready.exists():
                assert process.poll() is None, log_path.read_text()
                assert time.monotonic() < deadline, "owned child did not reach controlled cut"
                time.sleep(0.01)
            process.kill()
            assert process.wait(timeout=3) == -signal.SIGKILL
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        world.writer = VectorStore(world.db)


def backlog(world):
    world.check()
    # Exact over this bounded fixture, including queued acknowledgements not yet committed.
    missing = world.writer.conn.execute(
        "SELECT count(*) FROM chunks c WHERE NOT EXISTS (SELECT 1 FROM chunk_vectors_rowids v WHERE v.id=c.id)"
    ).fetchone()[0]
    return len(list(world.queue.glob("*.jsonl"))), missing


def catch_up(world, timeout):
    started = time.monotonic()
    before = backlog(world)
    while True:
        world.drain()
        world.hotlane()
        depth, missing = backlog(world)
        elapsed = time.monotonic() - started
        assert elapsed <= timeout, f"backlog did not converge in {timeout}s: queue={depth}, missing={missing}"
        if depth == missing == 0:
            world.assert_roundtrip()
            return {
                "queue_before": before[0],
                "missing_before": before[1],
                "queue_after": depth,
                "missing_after": missing,
                "catch_up_ms": round(elapsed * 1000, 3),
            }
        time.sleep(0.01)


def worker(root, seed, action):
    validate_environment(root)

    def encode(text):
        return Universe.encode(SimpleNamespace(seed=seed), text)

    if action == "index":
        with VectorStore(private_path(root, root / "private.db")) as store:
            _index(root, seed, store, cut=True)
    else:
        if action == "encoder_stall":

            def encode(_text):
                return _pause(root)
        elif action == "drain_restart":
            drain._unlink_processed_file = lambda *_args: _pause(root)
        else:
            raise ValueError(action)
        drain.drain_once(
            db_path=private_path(root, root / "private.db"),
            queue_dir=private_path(root, root / "queue"),
            batch_size=100,
            log_path=private_path(root, root / "drain.log"),
            embed_fn=encode,
            pause_sentinel_path=private_path(root, root / "pause"),
        )
    raise AssertionError("worker missed its controlled cut")


if __name__ == "__main__":
    worker(Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3])
