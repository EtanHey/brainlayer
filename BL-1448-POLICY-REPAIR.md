# BL-1448 policy repair

Status: PREPARED ONLY / NOT GREEN. No M1 admission or push.

## Identity and purpose

- Worktree: `/Users/etanheyman/.t3/worktrees/brainlayer/t3-9d28a37c`.
- Private branch: `private/bl-1448-policy-t3-9d28a37c`.
- Direct parent: `d16e5d892b1b90effa694dcf882b20455123abf7`.
- Eventual child PR base: `wt/surge-health-missing-embeddings-bound`.
- Original checkout branch `fix/policy-repair` remains at
  `f365286b9ab915da4526b1ea3fa0a4e6a2d60d89`; no original branch update.
- Prepared identity is the commit containing this report; obtain its full SHA
  and tree with `git rev-parse HEAD HEAD^{tree}`.
- Scope: `scripts/retirement_policy.json` plus this receipt only.

The two supplied R1 reports identify the missing policy classification at
`health_check.py:652`, matching published red job `119908821149`, artifact
`12437548642`. Those CI results are peer evidence, not refreshed by this worker.

Exact added purpose map:

```json
{
  "a97548889728cc55a200d9ee298d04394c1c761b6ab9d7acef288afab8ac99cf": "Local bounded read-only missing-embedding sampler: SQLite mode=ro and query_only, one snapshot per tick with page/instruction/time budgets; no model or network operation."
}
```

Site: `src/brainlayer/health_check.py`, owner
`<module>.sample_missing_embeddings`, kind `generic_send`, target
`sqlite3.connect`, call SHA-256
`3fbb77487a9ccb9a26c61a4670ecf0f9095e05c17a0b25d32ba49dd1fde20208`.
The scanner's generic kind includes SQLite connect; it does not imply networking.

## Preparation evidence

Goal: classify exactly this bounded local reader. Round: one sequential slice.
Tools: assigned Codex gpt-6.1-sol/high; static reasoning for exact-purpose scope.
Depends on: pinned source and supplied R1 purpose reviews. Time: not measured.
Steps completed: read brief/handoff/R1s, inspect generator/source, generate policy.
Commit/static-diff receipt is returned separately by the worker.

Read the generator, inventory and imported source-scanner module before running.
Their generation path uses standard-library parsing/read operations and does not
execute scanned source, load a model, open a DB/socket, run jobs, tests or builds.
Ran only code generation via `retirement_policy_generate.render(root, purposes)`
under `python3 -I -S -B` with this worktree explicitly on `sys.path`; startup site
processing and bytecode writes were disabled. No policy check/test was run.
Generation required exactly one new site, no stale sites, unchanged existing
entries and unchanged top-level metadata before writing the result.
Inventory increased 588 to 589 entries; generated policy diff is +10/-0 lines.

Source inspection: `mode=ro`, `PRAGMA query_only`, one read transaction per tick,
50 ms busy timeout, page/instruction budgets and at most a one-second deadline.
SQLite callbacks do not prove a bound on filesystem stalls. No runtime claim.
Scanners, unknown/stale refusal, mutation tests, probes and table schema unchanged.
No test-running commit hooks were present in the configured `.githooks` directory;
only the pre-push hook was present. Hooks are not bypassed.

## Missing evidence and proposed next gate

All execution below needs a later fresh M1 grant; none was run here:

1. Verify the admitted candidate's full SHA/tree and direct pinned parent, then
   `python scripts/retirement_policy_generate.py --check` and focused
   `pytest tests/test_retirement_policy_generate.py tests/test_retirement_ast_portability.py tests/test_retirement_inventory.py tests/test_retirement_guard.py`.
   Require exact inventory/purpose equality, deterministic output, unknown-site
   refusal and semantic-mutation refusal; never weaken a failing gate.
2. Run `pytest tests/test_health_check.py tests/test_stability_health_check.py`.
   Preserve incomplete versus exact zero, page/instruction/time limits, snapshot
   wrap without stale totals, lifecycle exclusions, retained heal history,
   canary reachability and exact doctor parity.
3. Parent-owned copied-real-DB gate: realistic schema, WAL/concurrent writes,
   contention/deadline behavior and exact parity where deliberately measured.
   Keep canonical data/services untouched; no copy or contention proof exists here.
4. Normal admitted M1 pre-push before any push/new PR, then required CI on the
   corrected final head. Full retirement installed/native evidence and deployed
   real request acceptance remain separate parent gates. No GREEN/copy/native claim.

No MBP pytest/collection/lint/build/pre-push, M1 command, DB/socket/model access,
push/PR, reviewer/child/watch, person message/audio, config/service/GUI/Drive
operation, cleanup or deletion was performed. Peers/files/refs are preserved.
