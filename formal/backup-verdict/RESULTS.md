# Measured run

- Date: 2026-11-02T07:03Z, on the M4: `formal/backup-verdict/check.sh --require-sync`, one TLC worker.
- Tree: this branch on main 67d23f14 (main has not moved since the last merge). S1a-5
  (`fix/verdict-hysteresis-observed`, 4e07651b) is **not** on main.
- TLC: release v1.7.4 jar (sha256 `936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88`);
  TLC's own banner reads `TLC2 Version 2.19 of 08 August 2024`.
- Result: **34 / 34 cases matched, no stale source, exit 0.** Total TLC time 194.4 s. Probes: 28 passed.

| Case | Expected = got | Distinct states | s | Trace steps | Replay or conformance (SHA) |
|---|---|---:|---:|---:|---|
| current-backup | pass | 1,352,435 | 93.2 | — |  |
| current-vacuum | pass | 61,103 | 2.0 | — |  |
| current-pipes-drive | pass | 6,396 | 1.1 | — |  |
| current-pipes-stores | pass | 125,988 | 14.3 | — |  |
| current-pipes-embedding | pass | 29,713 | 2.9 | — |  |
| current-dot | pass | 9,224 | 9.6 | — |  |
| current-dot-liveness | pass | 9,224 | 36.1 | — |  |
| bug-1583-B1-canary-latch | RedHoldsUntilEvidence | 2,982 | 0.9 | 10 |  |
| bug-1583-B2-resume-grace | RedClearsToGreenOnlyOnEvidence | 3,850 | 0.8 | 11 | 672c9f97 reproduces / fixed |
| bug-1583-B3-renewable-grace | NoFalseGreen | 3,039 | 0.8 | 10 | 672c9f97 reproduces / fixed |
| bug-1583-B4-drive-latch | RedHoldsUntilEvidence | 67 | 0.7 | 5 |  |
| bug-1583-B5-persisted-max | NoFalseGreen | 179 | 0.7 | 6 | 672c9f97 reproduces / fixed |
| bug-1589-B5a-grace-without-clock | NoFalseGreen | 502 | 0.7 | 7 | 7f7f6245 reproduces / fixed |
| bug-1589-B5b-shared-revocation | NoFalseGreen | 1,178 | 0.8 | 8 | 7f7f6245 reproduces / fixed |
| bug-1589-B5c-legacy-identity | NoFalseGreen | 239 | 0.7 | 6 | 7f7f6245 reproduces / fixed |
| bug-1595-local-removed | TruthfulSnapshotLabel | 35 | 0.7 | 5 | d9d7bff1 reproduces / fixed |
| bug-1595-recovery-reset | RedHoldsUntilEvidence | 8,538 | 1.2 | 10 |  |
| bug-1596-abandoned | NoFalseGreen | 17 | 0.6 | 4 | 12b9ddd4 reproduces / fixed |
| bug-1596-superseded | NoFalseGreen | 413 | 0.7 | 7 | 12b9ddd4 reproduces / fixed |
| bug-1587-B4-backlog-ages-out | RedHoldsUntilEvidence | 708 | 0.7 | 8 |  |
| bug-1587-B5-stale-canary-success | RedHoldsUntilEvidence | 11,151 | 1.3 | 11 |  |
| bug-1583-S1-alert-lights-dot | NoStuckAlert | 386 | 0.7 | 2 |  |
| bug-s1a5-F1-red-to-amber | RedHoldsUntilEvidence | 3,918 | 0.9 | 11 | 67d23f14 reproduces / **still on main** |
| bug-s1a5-F2-unobserved-tick-keeps-count | RedToGreenNeedsConsecutive | 10,700 | 1.2 | 13 | 67d23f14 reproduces / **still on main** |
| bug-s1a5-F3-graced-stale-keeps-count | RedClearsToGreenOnlyOnEvidence | 20,949 | 2.0 | 16 | 67d23f14 reproduces / **still on main** |
| conform-ingest-prefix | conforms | simulated | 2.9 | — | 3,000 traces, 8,934 ticks vs 67d23f14: 0 diverging |
| conform-ingest-fixed | diverges | simulated | 3.0 | — | 3,000 traces, 8,934 ticks vs working tree: 2 diverging |
| finding-ingest-red-to-amber | pass | 21,840 | 2.1 | — |  |
| finding-ingest-graced-stale-keeps-count | pass | 21,840 | 2.3 | — |  |
| finding-ingest-unobserved-tick-keeps-count | pass | 21,840 | 2.8 | — |  |
| limit-unreadable-log | NoFalseGreenStrict | 314 | 0.8 | 6 |  |
| scratch-vacuum-any-receipt | VacuumNeedsFreshSnapshot | 25 | 0.8 | 4 |  |
| scratch-deferred-advances | VacuumNeedsFreshSnapshot | 5 | 0.7 | 2 |  |
| scratch-clean-run-keeps-alert | AlertClears | 3,930 | 3.7 | 4 |  |

Every pre-fix replay matched the model on every tick (`conforms: true`). Conformance traces come from
`tlc -simulate file=…,num=3000 -depth 60 -seed 7`. Each tick's heartbeat observation is fed to the
real `evaluate()`, and the `ingest` state is compared on every tick.

## Against the S1a-5 head (`--src-ref 4e07651b`)

`check.sh --require-sync --src-ref 4e07651b --case conform-ingest-fixed --case bug-s1a5`, 2026-11-02T07:06Z:

| Case | Replay at 67d23f14 (main) | At 4e07651b (S1a-5) |
|---|---|---|
| bug-s1a5-F1-red-to-amber | reproduces, tick for tick | does not reproduce |
| bug-s1a5-F2-unobserved-tick-keeps-count | reproduces, tick for tick | does not reproduce |
| bug-s1a5-F3-graced-stale-keeps-count | reproduces, tick for tick | does not reproduce |
| conform-ingest-fixed (S1a-5 model) | 2 of 3,000 traces diverge | **0 of 3,000 traces diverge (8,934 ticks)** |

The S1a-5 model matches the S1a-5 code on every tick of 3,000 random behaviours, and the pre-S1a-5
model matches main on every tick of the same 3,000 (`conform-ingest-prefix`). That run exits 1 by
design: the four results are the "fix is present" side of the ratchet. When S1a-5 merges, flip
`fixed_here` on the three `bug-s1a5-*` cases and set `conform-ingest-fixed` to `"conforms"`.

## Scratch break (the skill's proof step, round 1)

The working tree's `_verified_age` was edited to take `max(persisted, observed)` again (#1583 B5).
Then `check.sh --case bug-1583-B5 --case bug-1596-ab` printed
`FAIL bug-1583-B5-persisted-max … replay at working tree: reproduces … (UNEXPECTED …)` and
**exit 1**. After `git checkout -- src/brainlayer/health_verdict.py` the sha256 was back to its
committed value, and the case passes.
