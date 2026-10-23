"""Production write/search paths inside a fail-closed, disposable test universe."""

import asyncio
import hashlib
import json
import math
import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import apsw

from brainlayer import drain
from brainlayer.mcp import store_handler
from brainlayer.vector_store import VectorStore
from scripts import hotlane_brainbar_daemon as hotlane

PRIVATE_PATHS = {
    "BRAINLAYER_DB": "private.db",
    "BRAINBAR_DB_PATH": "private.db",
    "BRAINBAR_SOCKET_PATH": "private.sock",
    "BRAINLAYER_MCP_SOCKET": "private.sock",
    "BRAINLAYER_QUEUE_DIR": "queue",
    "BRAINLAYER_DRAIN_LOG_PATH": "drain.log",
    "BRAINLAYER_DRAIN_ERROR_LOG_PATH": "drain.err.log",
    "BRAINLAYER_DRAIN_HEALTH_PATH": "drain-health.json",
    "BRAINLAYER_WRITER_TELEMETRY_PATH": "writer.jsonl",
    "BRAINLAYER_WRITER_HEARTBEAT_DIR": "heartbeats",
    "BRAINLAYER_WRITER_PIDFILE_DIR": "pidfiles",
    "BRAINLAYER_TRANSPORT_HEALTH_PATH": "transport.json",
    "BRAINLAYER_OBSERVABILITY_PATH": "observability.json",
}


def private_path(root, path):
    root, path = Path(root).resolve(), Path(path).expanduser().resolve()
    if not path.is_relative_to(root) or path == root or ".local/share/brainlayer" in str(path):
        raise ValueError(f"outside disposable universe: {path}")
    return path


def validate_environment(root):
    for key, suffix in PRIVATE_PATHS.items():
        if private_path(root, os.environ.get(key, "/tmp/brainbar.sock")) != root / suffix:
            raise ValueError(f"outside disposable universe: {key}")


class Universe:
    ack_bound_ms = 500
    held_bound_ms = 500

    def __init__(self, root, monkeypatch, seed):
        self.root, self.seed = root.resolve(), seed
        build = Path(__file__).resolve().parents[2] / ".build.noindex"
        if not self.root.is_relative_to(build.resolve()):
            raise ValueError("outside disposable universe: root must be below .build.noindex")
        self.receipts, self.threads = [], []
        self.hotlane_busy = 0
        self.names = ["otter", "ibis", "lynx", "mole", "puma", "yak"]
        random.Random(seed).shuffle(self.names)
        for key, suffix in PRIVATE_PATHS.items():
            monkeypatch.setenv(key, str(self.guard(self.root / suffix)))
        for key, value in {
            "BRAINLAYER_FORBID_BRAINBAR_SOCKET": "1",
            "BRAINLAYER_FORBID_EMBEDDING_MODEL": "1",
            "BRAINLAYER_WRITER_TELEMETRY": "1",
            "BRAINLAYER_STORE_BUSY_BUDGET_MS": "25",
            "BRAINLAYER_DRAIN_BUSY_TIMEOUT_MS": "50",
            "BRAINLAYER_HOTLANE_WRITE_BUSY_TIMEOUT_MS": "50",
            "BRAINLAYER_DRAIN_POST_COMMIT_YIELD_MS": "0",
            "BRAINLAYER_DRAIN_EMBED": "1",
            "BRAINLAYER_ARBITRATED": "0",
            "BRAINLAYER_INTERACTIVE_STORE_QUEUE": "0",
        }.items():
            monkeypatch.setenv(key, value)
        self.check()
        self.db = self.root / "private.db"
        self.queue = self.root / "queue"
        self.queue.mkdir()
        self.writer = VectorStore(self.db)
        monkeypatch.setattr(store_handler, "_get_vector_store", lambda **_kw: self.writer)
        model = SimpleNamespace(embed_query=self.encode, embed_texts=lambda texts, **_: [self.encode(t) for t in texts])
        monkeypatch.setattr(store_handler, "_get_embedding_model", lambda: model)

        def thread(*args, **kwargs):
            result = threading.Thread(*args, **kwargs)
            self.threads.append(result)
            return result

        monkeypatch.setattr(store_handler, "threading", SimpleNamespace(Thread=thread))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for thread in self.threads:
            thread.join(timeout=3)
            assert not thread.is_alive(), "owned background writer did not stop"
        self.writer.close()

    def guard(self, path):
        return private_path(self.root, path)

    def check(self):
        validate_environment(self.root)

    def connection(self):
        self.check()
        return apsw.Connection(str(self.guard(self.db)))

    def text(self, name):
        words = "river forest meadow mountain island desert ocean garden canyon marsh prairie valley stone willow birch cedar maple fern moss amber violet copper silver orange indigo scarlet green yellow azure quartz granite basalt marble limestone shale pebble shell branch leaf root bloom seed fruit rain wind snow frost cloud sunlight moonlight dawn dusk summer winter spring autumn".split()
        rng = random.Random(f"{self.seed}:{name}")
        return f"oracle{name} " + " ".join(rng.sample(words, 18))

    def encode(self, text):
        rng = random.Random(hashlib.sha256(f"{self.seed}:{text}".encode()).digest())
        values = [rng.uniform(-1, 1) for _ in range(1024)]
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values]

    def store(self, name, content=None):
        self.check()
        started = time.monotonic()
        _, result = asyncio.run(
            store_handler._store(content=content or self.text(name), memory_type="note", project="oracle")
        )
        elapsed = (time.monotonic() - started) * 1000
        assert result["status"] in {"STORED", "DUPLICATE", "MERGED", "DEFERRED"}, result
        self.receipts.append({**result, "name": name, "ack_ms": elapsed})
        for thread in self.threads:
            thread.join(timeout=3)
            assert not thread.is_alive()
        return result

    def drain(self, embed_fn=None):
        self.check()
        return drain.drain_once(
            db_path=self.guard(self.db),
            queue_dir=self.guard(self.queue),
            batch_size=100,
            log_path=self.guard(self.root / "drain.log"),
            embed_fn=embed_fn or self.encode,
            pause_sentinel_path=self.guard(self.root / "pause"),
        )

    def hotlane(self, embed_fn=None):
        self.check()
        with VectorStore(self.db, readonly=True) as reader:
            candidates = hotlane._pending_chunk_rows(reader, limit=100)
        vectors = hotlane._embed_candidates(candidates, embed_fn=embed_fn or self.encode)
        try:
            return hotlane._write_embedded_vectors(self.db, vectors)
        except apsw.BusyError:
            self.hotlane_busy += 1
            return 0

    def canonical(self, reader, chunk_id):
        alias = reader.conn.execute(
            "SELECT canonical_chunk_id FROM chunk_id_alias WHERE old_chunk_id=?", (chunk_id,)
        ).fetchone()
        return alias[0] if alias else chunk_id

    def assert_roundtrip(self):
        self.check()
        with VectorStore(self.db, readonly=True) as reader:
            for receipt in self.receipts:
                cid = self.canonical(reader, receipt["chunk_id"])
                row = reader.conn.execute("SELECT content FROM chunks WHERE id=?", (cid,)).fetchone()
                assert row, f"missing acknowledged identity: {cid}"
                token = f"oracle{receipt['name']}"
                fts = list(reader.conn.execute("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ?", (token,)))
                assert (cid,) in fts, f"FTS invisible: {cid}"
                public = reader.hybrid_search(query_embedding=None, query_text=token, n_results=100)
                assert cid in public["ids"][0], f"public search invisible: {cid}"
                vector = reader.search(query_embedding=self.encode(row[0]), n_results=1)
                assert vector["ids"][0] == [cid], f"vector invisible: {cid}"
                assert vector["distances"][0][0] < 1e-5, f"stale vector: {cid}"

    def assert_acknowledged(self, extra_ids=()):
        self.assert_roundtrip()
        with VectorStore(self.db, readonly=True) as reader:
            ids = {self.canonical(reader, r["chunk_id"]) for r in self.receipts} | set(extra_ids)
            assert reader.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == len(ids), "duplicate rows"
            assert not list(reader.conn.execute("SELECT content FROM chunks GROUP BY content HAVING count(*)>1"))

    def lock_holds(self):
        spans = [json.loads(line) for line in (self.root / "writer.jsonl").read_text().splitlines()]
        held = []
        for span in spans:
            if (
                span.get("event") != "txn_finished"
                or span.get("producer") != "drain"
                or span.get("outcome") != "commit"
            ):
                continue
            assert "statements" in span, "missing committed drain statement telemetry"
            statements = {s["normalized_sql"]: s for s in span["statements"]}
            assert {"BEGIN IMMEDIATE", "COMMIT"} <= statements.keys(), "missing drain lock boundary statements"
            begin, commit = statements["BEGIN IMMEDIATE"], statements["COMMIT"]
            elapsed = (
                datetime.fromisoformat(commit["started_at"]) - datetime.fromisoformat(begin["started_at"])
            ).total_seconds() * 1000
            held.append(elapsed + commit["total_duration_ms"] - begin["total_duration_ms"])
        assert held, "missing committed drain telemetry"
        return held

    def assert_latency(self):
        acks = sorted(r["ack_ms"] for r in self.receipts)
        p95 = acks[math.ceil(len(acks) * 0.95) - 1]
        held = max(self.lock_holds())
        assert p95 <= self.ack_bound_ms, f"ack p95 {p95}ms"
        assert 0 <= held <= self.held_bound_ms, f"drain lock held {held}ms"
        return {"acks": len(acks), "ack_p95_ms": round(p95, 3), "drain_held_max_ms": round(held, 3)}
