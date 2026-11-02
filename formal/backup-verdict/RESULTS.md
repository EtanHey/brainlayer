# Measured run

- Date: 2026-11-02T06:44Z, on the M4: `formal/backup-verdict/check.sh --require-sync`, one TLC worker, sequential.
- Tree: this branch merged with main 67d23f14 (#1595, #1596, #1587 and #1588 all merged).
- TLC: release v1.7.4 jar (sha256 `936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88`).
  TLC's own banner reads `TLC2 Version 2.19 of 08 August 2024`.
- Result: **29 / 29 cases matched, no stale source, exit 0.** Total TLC time 203.7 s.
  Probes (`test_check.py`): 28 passed.

| Case | Expected = got | Distinct states | s | Trace steps | Replay: pre-fix ref / this tree |
|---|---|---:|---:|---:|---|
| current-backup | pass | 1,352,435 | 103.2 | — |  |
| current-vacuum | pass | 61,103 | 2.3 | — |  |
| current-pipes-drive | pass | 6,094 | 1.2 | — |  |
| current-pipes-stores | pass | 83,480 | 8.2 | — |  |
| current-pipes-embedding | pass | 28,434 | 2.1 | — |  |
| current-dot | pass | 9,224 | 7.3 | — |  |
| current-dot-liveness | pass | 9,224 | 56.7 | — |  |
| bug-1583-B1-canary-latch | RedHoldsUntilEvidence | 2,337 | 1.0 | 10 |  |
| bug-1583-B2-resume-grace | RedClearsToGreenOnlyOnEvidence | 3,729 | 1.0 | 11 |  |
| bug-1583-B3-renewable-grace | NoFalseGreen | 3,039 | 0.9 | 10 | 672c9f97 reproduces / fixed |
| bug-1583-B4-drive-latch | RedHoldsUntilEvidence | 67 | 0.7 | 5 |  |
| bug-1583-B5-persisted-max | NoFalseGreen | 179 | 0.7 | 6 | 672c9f97 reproduces / fixed |
| bug-1589-B5a-grace-without-clock | NoFalseGreen | 502 | 0.7 | 7 | 7f7f6245 reproduces / fixed |
| bug-1589-B5b-shared-revocation | NoFalseGreen | 1,178 | 0.8 | 8 | 7f7f6245 reproduces / fixed |
| bug-1589-B5c-legacy-identity | NoFalseGreen | 239 | 0.7 | 6 | 7f7f6245 reproduces / fixed |
| bug-1595-local-removed | TruthfulSnapshotLabel | 35 | 0.7 | 5 | d9d7bff1 reproduces / fixed |
| bug-1595-recovery-reset | RedHoldsUntilEvidence | 6,444 | 1.2 | 10 |  |
| bug-1596-abandoned | NoFalseGreen | 17 | 0.7 | 4 | 12b9ddd4 reproduces / fixed |
| bug-1596-superseded | NoFalseGreen | 413 | 0.8 | 7 | 12b9ddd4 reproduces / fixed |
| bug-1587-B4-backlog-ages-out | RedHoldsUntilEvidence | 707 | 0.7 | 8 |  |
| bug-1587-B5-stale-canary-success | RedHoldsUntilEvidence | 8,951 | 1.2 | 11 |  |
| bug-1583-S1-alert-lights-dot | NoStuckAlert | 386 | 0.8 | 2 |  |
| finding-ingest-red-to-amber | RedHoldsUntilEvidence | 3,780 | 0.9 | 11 |  |
| finding-ingest-graced-stale-keeps-count | RedClearsToGreenOnlyOnEvidence | 19,377 | 2.1 | 16 |  |
| finding-ingest-unobserved-tick-keeps-count | RedToGreenNeedsConsecutive | 10,109 | 1.2 | 13 |  |
| limit-unreadable-log | NoFalseGreenStrict | 314 | 0.8 | 6 |  |
| scratch-vacuum-any-receipt | VacuumNeedsFreshSnapshot | 25 | 0.7 | 4 |  |
| scratch-deferred-advances | VacuumNeedsFreshSnapshot | 5 | 0.7 | 2 |  |
| scratch-clean-run-keeps-alert | AlertClears | 3,871 | 3.7 | 4 |  |

Every pre-fix replay matched the model on every tick (`conforms: true` in `results.json`). Before
#1596 merged, the same run on main 12b9ddd4 showed `bug-1596-*` as "still in this source". With
`--src-ref 7505f8b1` (the #1596 head), both read "does not reproduce".

## Scratch break (the skill's proof step)

The working tree's `_verified_age` was edited to take `max(persisted, observed)` again (#1583 B5).
Then `check.sh --case bug-1583-B5 --case bug-1596-ab` printed
`FAIL bug-1583-B5-persisted-max … replay at working tree: reproduces … (UNEXPECTED …)` and
**exit 1**. After `git checkout -- src/brainlayer/health_verdict.py` the sha256 was back to its
committed value, and the case passes.
