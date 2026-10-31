# Disposable recovery oracle

Run from this worktree (no resident processes, database, sockets, or LaunchAgents):

```sh
PYTHONPATH=$PWD/src BRAINBAR_SOCKET_PATH=/nonexistent BRAINLAYER_FORBID_BRAINBAR_SOCKET=1 \
  ~/Gits/brainlayer/.venv/bin/python -m pytest tests/recovery_oracle -m recovery_oracle -s
```

`pytest -m recovery_oracle` also selects the gate. Observe the surge brief's load gate
before launching locally. All artifacts belong below `.build.noindex/pytest/`.

The gate uses the production Python store handler's actual structured acknowledgements,
queue_io, drain, hotlane candidate/embedding/write functions, full VectorStore schema,
FTS5 MATCH, public hybrid search, sqlite-vec nearest-neighbor search, alias resolution,
and writer telemetry. The encoder alone is deterministic and seeded; this does not
measure real model throughput or semantic quality. Check every acknowledged identity,
not just the first canary; replayed aliases must resolve to a single canonical row.

The lock fault holds a separate real BEGIN IMMEDIATE for at least 300 ms while stores,
drain and hotlane attempt work. Test budgets: store busy 25 ms, drain/hotlane busy 50 ms;
ack p95 <=500 ms; committed drain BEGIN-success through COMMIT <=500 ms. These bounds
apply to this tiny fixture, not the resident corpus. Telemetry must exist; missing
measurements fail. Negative controls skip replay, remove FTS/vectors, duplicate a row,
and sleep in the real queue admission/locked drain path; each must trip an oracle.

## Ack sampling (2026-10-31)

The 500 ms ceilings are unchanged. Fault scenarios now measure 100 fresh store
requests; lock scenarios measure those 100 contended requests plus two seeds.
Nearest-rank p95 over the former six/eight receipts was exactly the maximum.
At 100 samples, up to five isolated slow receipts can occur without making the
95th percentile exceed its bound. All receipts still undergo replay, alias,
FTS/vector round-trip and no-loss/duplicate checks; none are discarded as warmup.
The all-status contract keeps its four receipts and strict latency check.

The lock producer's three-second deadline applies to each batch of at most six
requests as the workload grows. The drain/hotlane future deadlines, 30-second
test watchdog, five-second catch-up bound and 500 ms drain-held bound remain.
Two real-queue controls inject 600 ms into the first one/six acknowledgements:
one outlier must pass p95; six outliers must fail. The existing slow-ack and
long-lock negative controls remain unchanged. Receipts also expose `ack_max_ms`
so p95 can be distinguished from the slowest operation.

The timer includes input generation, `asyncio.run`, validation and durable queue
publication. Fixture construction, background joins and the subsequently injected
index/encoder/drain cuts are outside it. In a separate local profile, durable
syncs consumed 426 of 445 ms across six handler calls; they belong inside ack
latency and must not be mocked away. This identifies the local I/O cost, not the
precise cause of historical Ubuntu stalls.

Week-of-October-24 CI logs contained ten ack overruns (500.201–588.459 ms), five
already on October 26, including one on main. Passing CI output captures the
receipt JSON, so a historical uncensored distribution cannot be reconstructed.
Ten unchanged-head local runs reproduced an index overrun at 508.014 ms; seven
runs passed, one failed that ack bound, and one had two separate all-status
precondition failures. The PR body records the per-run measurements.

Native scope: BrainBarDaemon supports private BRAINBAR_DB_PATH/BRAINBAR_SOCKET_PATH,
but starts a lifecycle watchdog that can kickstart the resident UI. Never launch it
unguarded here. Existing `DaemonTerminationProcessTests` use `daemon_shutdown_guard.c`
to forbid peer signals and launchctl/open, and emit `real_daemon_termination_receipt`.
Existing `MCPRouterTests` cover native deferred queue/replay. This Python gate exercises
the shared queue/drain side; it does not attest a Swift MCP wire round-trip. Process
crash recovery is distinct from power-loss durability: queue_io on the BEFORE revision
has no file/directory fsync. Do not infer power-loss safety from these tests.

## BEFORE: unchanged main 1a713400, 2026-10-22 (M1, seed 22026)

First gated run (load 3.929): **11 passed, 1 failed in 3.34 s**. Lock fault:
8 real acks; FTS/vector round-trips and no duplicate/lost rows passed; ack p95
293.300 ms; drain lock-held maximum 11.863 ms (bounds 500/500 ms). All six broken
variants and four isolation checks passed. Keep this short gate in default CI:
it is already selected by CI's ordinary marker expression. No new heavy job.

The all-success-status scenario is deliberately **RED on main**: STORED, DUPLICATE,
MERGED and DEFERRED occurred, but MERGED changed canonical content while retaining
its previous vector. Independent readonly measurement of the disposable failed DB
found cosine distance to its own current content **1.011185884**, versus 0 for the
other chunk. `store.py:248-267` does not invalidate an existing vector when a merge
uses embed_fn=None; `embed_hot_chunk` selects only absent vectors. This acceptance
failure is tracked in #1450. BL lead authorized a strict xfail for this one case;
XPASS fails CI, and the separate fix PR must remove the marker. The assertions
and the recorded red baseline remain unchanged.
These are synthetic main-code measurements, not resident performance numbers.

## Fault expansion and bounded catch-up (PR 2)

Child cuts use SIGKILL on Popen handles owned by this test, after a file barrier:
index call's second transaction (first chunk committed), encoder before writing,
and drain immediately after commit but before queue unlink. Reopen the DB and
replay real work; retrying index inputs must leave exactly one row per identity.
Provider exceptions preserve FTS rows with absent vectors until hotlane repairs.
Catch-up measures exact fixture debt and requires queue=0 and missing vectors=0
within 5 s. Skipping replay or hotlane must fail that deadline oracle.

Latest complete main-code run: **17 passed, 1 failed in 6.41 s**, load 4.461;
the sole failure remains MERGED stale-vector retrieval. Eight negative controls
passed. No production source was changed. All fault scenarios below pass a/b/c/d:

| Fault | Acks | Ack p95 ms | Drain held max ms | Catch-up ms | Final queue / missing |
|---|---:|---:|---:|---:|---:|
| Lock | 8 | 38.762 | 10.420 | 65.488 | 0 / 0 |
| Interrupted index | 6 | 4.042 | 4.198 | 57.175 | 0 / 0 |
| Encoder raises | 6 | 1.322 | 3.356 | 77.633 | 0 / 0 |
| Encoder stalls, killed | 6 | 1.267 | 13.655 | 73.303 | 0 / 0 |
| Drain restart | 6 | 1.254 | 15.486 | 64.517 | 0 / 0 |

Run with `-s` for `recovery_oracle_receipt` JSON. The earlier direct suites plus
PR 1 measured 152 passed / 1 failed in 8.53 s. Resident recon baseline remains
separate: drain held up to 180.584 s, queue age 116.35 min, 2,902 missing vectors,
index rc124, deferred-row creation-to-ingest up to 228 s; total DEFERRED loss and
receipt-latency denominators were unmeasurable. These test results do not replace it.

## #1450 fix follow-up

The standalone repro outside this harness (`tests/test_store_vector_invalidation.py`)
plus the index regression failed **9 tests / passed 4** on unchanged main. The fix
invalidates both vector indexes in the content-merge savepoint; drain accepts a
precomputed embedding only for the exact canonical body and retains valid vectors
on unchanged merges; index defers changed canonicals instead of blending vectors.
No encoder runs while the writer lock is held. FTS remains available during repair.

With that fix and the strict marker still present, the all-status case produced
**XPASS(strict)** and failed, proving the baseline flips. The fix removes that one
marker without changing the oracle assertions. Final focused validation, including
all fault scenarios: **187 passed in 16.08 s**. The independent pre-harness suite
passed **169 tests**; a temporary source overlay of #1446 passed **41 tests** and
was removed. #1446 is compatible and is neither a cause nor a dependency of #1450.
