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
failure must be fixed in a separate hardening PR, not xfailed or weakened here.
These are synthetic main-code measurements, not resident performance numbers.
