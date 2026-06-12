# Phase 1 Findings

## 2026-06-12 Happy Campr Readiness - Fleet Write Reliability

### Base and integrated commits

- Worktree: `/Users/etanheyman/Gits/brainlayer.wt/hc-phase-1`
- Branch: `hc/phase-1-fleet-write-reliability`
- Phase 0 base: `425e8645fa299386427e6d59a817af94c57ee7d7` (`origin/main`)
- Integrated provenance/deadlock lineage from `feat/provenance-resolution`:
  - `faa0d242` resolver core + eval
  - `025b4d66` enrichment integration
  - `883fc806` resolver edge coverage
  - `38026199` auto-supersession detector
  - `41e6d651` answer-leg annotations
  - `c0c7499b` autosupersede dry-run hook
  - `980c6f16` hybrid-search FTS provenance expressions
  - `a1bceb01` BrainBar 30s busy budget + flush-pending-on-queued-store
  - `2eecf61d` provenance regression fixes

### Implemented on this phase branch

- `brain_store` queued/busy responses are loud `DEFERRED` receipts:
  - text includes `DEFERRED`
  - structured response includes `status: "DEFERRED"`, `queued: true`, durable `chunk_id`, queue path, reason, and action
  - compatibility `queued: true` remains for existing callers
- Added `BRAINLAYER_MCP_QUERY_TIMEOUT` so fleet smoke can prove MCP calls with an explicit budget instead of failing at the hard-coded 15s wrapper during cold model/search startup.
- Added fallback replay module and CLI:
  - `src/brainlayer/fallback_replay.py`
  - `scripts/replay_brain_store_fallbacks.py`
  - structured YAML-frontmatter files replay with original body, timestamp, tags, importance, and originating project attribution
  - replay updates frontmatter `chunk_id` only after a concrete store ID
  - legacy free-form ledgers are inventoried but not auto-replayed
- Added representative MCP transport smoke:
  - `scripts/smoke/repogolem_mcp_transport_smoke.py`
  - runs actual MCP `tools/list`, `brain_search`, and `brain_store` from `brainlayer`, `systems`, and `narrationlayer` CWDs
  - fails if MCP tool calls return `isError`, `Transport closed`, or `Connection closed`
  - does not treat sockets/processes/DB rows as success

### Fallback inventory and attribution

Dry-run command:

```bash
scripts/replay_brain_store_fallbacks.py --json
```

Artifact with full pending and legacy path lists:

```text
docs.local/plans/2026-06-12-happy-campr-readiness/phase-1-fleet-write-reliability/artifacts/fallback-inventory-2026-06-12.json
```

Result:

- Structured fallback files: 59
- Pending structured files with empty `chunk_id`: 52
- Legacy fallback ledgers: 22
- Replay was not applied in this run.

Attribution rules implemented:

- Origin repo is the fallback file's `git rev-parse --show-toplevel`.
- Project comes from frontmatter `project` or `scope` if present.
- Otherwise project comes from the longest matching `~/.config/brainlayer/scopes.yaml` prefix.
- If no scope mapping exists, project falls back to the origin repo basename.
- `replayed_by` is stored separately in chunk metadata and does not replace the primary project.
- Stored replay metadata includes `fallback_source_path` and `origin_repo_path`.

Representative test fixtures prove:

- systems fallback path maps to `project="systems"`.
- narrationlayer fallback path maps to `project="narrationlayer"`.
- explicit frontmatter project overrides scopes mapping.
- replay preserves body byte-for-byte and raw ISO timestamp text.

### Transport smoke output

Command:

```bash
scripts/smoke/repogolem_mcp_transport_smoke.py --timeout 90
```

Artifact with exact JSON output:

```text
docs.local/plans/2026-06-12-happy-campr-readiness/phase-1-fleet-write-reliability/artifacts/transport-smoke-2026-06-12.json
```

Result: PASS for all three representative contexts.

- `brainlayer`: `brain_search` returned a real result; `brain_store` returned `DEFERRED` with chunk `manual-0c19f3e6e041453a`.
- `systems`: `brain_search` returned a real result; `brain_store` returned `DEFERRED` with chunk `manual-dce47dfe1fec4c19`.
- `narrationlayer`: `brain_search` returned a real result; `brain_store` returned `DEFERRED` with chunk `manual-123e091898014b39`.

Initial smoke without the configurable server timeout failed on actual MCP `brain_search` with:

```text
BrainLayer timeout (15s): DB may be locked by enrichment pipeline.
```

That failure is why the smoke now passes `BRAINLAYER_MCP_QUERY_TIMEOUT` into the server and records tool-call failure rather than relying on liveness.

Process-output sanitation note: no raw process listings were recorded in this findings file. Smoke output records MCP tool results only, with no process command arguments.

### Launcher-level repoGolem smoke

Launcher availability check:

```bash
zsh -ic 'source ~/.config/ralphtools/golem-dispatch.zsh; source ~/.config/ralphtools/launchers.zsh 2>/dev/null || true; type brainlayerCodex systemsCodex narrationlayerCodex'
```

Result: all three wrappers resolved from `/Users/etanheyman/.config/ralphtools/launchers.zsh`.

Headless launcher smoke used each wrapper's `-p` path, which dispatches through repoGolem into `codex exec`:

```bash
brainlayerCodex -p '<BrainLayer MCP brain_search + brain_store smoke prompt>'
systemsCodex -p '<BrainLayer MCP brain_search + brain_store smoke prompt>'
narrationlayerCodex -p '<BrainLayer MCP brain_search + brain_store smoke prompt>'
```

Sanitized artifact:

```text
docs.local/plans/2026-06-12-happy-campr-readiness/phase-1-fleet-write-reliability/artifacts/launcher-smoke-2026-06-12.json
```

Result: PASS for all three repoGolem Codex launchers. Each nested Codex session completed actual BrainLayer MCP `brain_search` and `brain_store`; liveness alone was not accepted.

- `brainlayerCodex`: `TASK_DONE search=ok store=brainbar-06015eb3-92a`
- `systemsCodex`: `TASK_DONE search=ok store=brainbar-22fcf2f2-b4b`
- `narrationlayerCodex`: `TASK_DONE search=ok store=brainbar-885db366-f0c`

Note: the nested sessions inherited global skill rules, so some sessions read required skill files before MCP calls. They did not edit files, commit, or push.

### Verification

Targeted Python:

```bash
pytest tests/test_brainstore.py::TestStoreMemory::test_changed_duplicate_embedding_runs_after_write_transaction \
  tests/test_brainstore.py::TestStoreMemory::test_store_persists_fallback_replay_metadata \
  tests/test_store_handler.py \
  tests/test_write_queue.py::TestStoreRetryOnLock \
  tests/test_fallback_replay.py \
  tests/test_provenance.py \
  tests/test_provenance_integration.py \
  tests/test_provenance_autosupersede.py \
  tests/test_enrichment_controller.py::test_apply_enrichment_sets_content_hash -q
```

Result: `85 passed in 9.57s`.

Focused Swift:

```bash
swift test --package-path brain-bar --filter 'DatabaseTests/testStoreOrQueueWithinDefaultBudgetStoresAfterBriefWriteLock|DatabaseTests/testFlushPendingStoresUsesQueuedAtAsCreatedAtAndPreservesProject|MCPRouterTests/testBrainStoreFlushesPendingQueueWhenCurrentStoreQueues'
```

Result: 3 selected tests passed, 0 failures. Build emitted pre-existing Swift warnings unrelated to this phase.

Full Python attempt:

```bash
pytest -q
```

Result: collection blocked by `tests/regression/test_drift_detection.py` importing Deepchecks, which calls `sklearn.metrics.get_scorer("max_error")`; this sklearn install rejects that scorer.

Broad Python excluding that collection blocker:

```bash
pytest -q --ignore=tests/regression/test_drift_detection.py
```

Result: `2788 passed, 48 skipped, 5 xfailed, 326 warnings in 274.56s`.

### Phase 1.5 verification refresh

Commands rerun from this worktree on 2026-06-12:

```bash
pytest tests/test_brainstore.py::TestStoreMemory::test_changed_duplicate_embedding_runs_after_write_transaction \
  tests/test_brainstore.py::TestStoreMemory::test_store_persists_fallback_replay_metadata \
  tests/test_store_handler.py \
  tests/test_write_queue.py::TestStoreRetryOnLock \
  tests/test_fallback_replay.py \
  tests/test_provenance.py \
  tests/test_provenance_integration.py \
  tests/test_provenance_autosupersede.py \
  tests/test_enrichment_controller.py::test_apply_enrichment_sets_content_hash -q
```

Result: `85 passed in 9.45s`.

```bash
pytest -q --ignore=tests/regression/test_drift_detection.py
```

Result: `2788 passed, 48 skipped, 5 xfailed, 326 warnings in 285.26s (0:04:45)`.

```bash
swift test --package-path brain-bar --filter 'DatabaseTests/testStoreOrQueueWithinDefaultBudgetStoresAfterBriefWriteLock|DatabaseTests/testFlushPendingStoresUsesQueuedAtAsCreatedAtAndPreservesProject|MCPRouterTests/testBrainStoreFlushesPendingQueueWhenCurrentStoreQueues'
```

Result: selected XCTest suite executed 3 tests with 0 failures in 0.300 seconds. Swift Testing also reported 0 tests in 0 suites for this filter.

```bash
scripts/replay_brain_store_fallbacks.py --json
```

Result: dry-run only; `structured_count=60`, `pending_count=53`, `legacy_count=22`, `replayed=[]`. This was not applied.

```bash
scripts/smoke/repogolem_mcp_transport_smoke.py --timeout 90
```

Result: PASS for all three contexts with loud deferred receipts:

- `brainlayer`: `manual-e8ceb65b47cc434f`
- `systems`: `manual-aa7b273e926a4635`
- `narrationlayer`: `manual-93b685a4c74b4bdd`

Bounded deferred-store landing check:

- First attempted with `python`, which failed immediately because `python` is not on PATH in this shell.
- Reran with `python3` using a temporary `BRAINLAYER_QUEUE_DIR`, MCP `brain_store` in arbitrated mode, `drain_once(batch_size=1)` with `BRAINLAYER_DRAIN_EMBED=0`, then MCP `brain_search` for the unique marker.
- Result: `store_text` contained `DEFERRED`, queued chunk `manual-495c38102cf149de`, `drained=1`, and MCP search returned the marker `PHASE1_5_DEFERRED_LANDING_1781296941`.

Current full-suite blocker check:

```bash
pytest tests/regression/test_drift_detection.py -q
```

Result: collection failed with 1 error in 1.65s because Deepchecks imports `get_scorer("max_error")`, and this sklearn install reports `'max_error' is not a valid scoring value`. This blocker was intentionally not fixed in Phase 1.5.

### Phase 1.5 report hygiene

- `findings.md` now contains only Happy Campr Phase 1 / Phase 1.5 content.
- Unrelated old notes that were previously appended to `findings.md` were moved to ignored local salvage: `docs.local/plans/2026-06-12-happy-campr-readiness/phase-1-fleet-write-reliability/salvage-old-findings.md`.
- Phase 1 artifacts remain ignored under `docs.local/`; this tracked report records their paths, counts, and verdicts instead of force-adding them.

### Residual risk / not done

- Structured fallback files were inventoried but not replayed; the fresh Phase 1.5 dry-run found 53 still have empty `chunk_id`.
- Legacy `docs.local/brain-store-fallback/` ledgers are inventoried separately and need a manual conversion/budget before replay.
- The representative three-context MCP smoke used arbitrated store mode, so those three smoke writes prove MCP transport plus loud deferred receipts. A separate bounded temp-queue check verified one deferred write could later be drained and found via MCP search.
- Full `pytest -q` remains blocked by the Deepchecks/sklearn collection issue unless that dependency/test is fixed or isolated.
- Happy Campr user was not created.
