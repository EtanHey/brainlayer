# Measured run

- Date: 2026-11-02T06:39Z, on the M4 (`formal/backup-verdict/check.sh`, one TLC worker, sequential).
- Tree: main 12b9ddd4 (#1595 merged, #1596 not yet), plus this PR's files.
- TLC: release v1.7.4 jar (sha256 `936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88`).
  TLC's own banner reads `TLC2 Version 2.19 of 08 August 2024`.
- Result: **29 / 29 cases matched, exit 0.** Total TLC time 166.5 s. Probes (`test_check.py`): 28 passed.
- `SYNC STALE: src/brainlayer/health_verdict.py`, as expected: the pin is #1596's head 7505f8b1.

| Case | Expected → got | Distinct states | s | Trace steps | Replay: pre-fix ref / this tree |
|---|---|---:|---:|---:|---|
| current-backup | pass | 1,352,435 | 83.1 | — | |
| current-vacuum | pass | 61,103 | 2.0 | — | |
| current-pipes-drive | pass | 6,094 | 1.1 | — | |
| current-pipes-stores | pass | 83,480 | 7.6 | — | |
| current-pipes-embedding | pass | 28,434 | 2.1 | — | |
| current-dot | pass | 9,224 | 6.5 | — | |
| current-dot-liveness | pass (AlertClears) | 9,224 | 40.2 | — | |
| bug-1583-B1-canary-latch | RedHoldsUntilEvidence | 2,337 | 0.9 | 10 | |
| bug-1583-B2-resume-grace | RedClearsToGreenOnlyOnEvidence | 3,729 | 0.8 | 11 | |
| bug-1583-B3-renewable-grace | NoFalseGreen | 3,039 | 0.8 | 10 | 672c9f97 reproduces / fixed |
| bug-1583-B4-drive-latch | RedHoldsUntilEvidence | 67 | 0.7 | 5 | |
| bug-1583-B5-persisted-max | NoFalseGreen | 179 | 0.7 | 6 | 672c9f97 reproduces / fixed |
| bug-1589-B5a-grace-without-clock | NoFalseGreen | 502 | 0.8 | 7 | 7f7f6245 reproduces / fixed |
| bug-1589-B5b-shared-revocation | NoFalseGreen | 1,178 | 0.8 | 8 | 7f7f6245 reproduces / fixed |
| bug-1589-B5c-legacy-identity | NoFalseGreen | 239 | 0.7 | 6 | 7f7f6245 reproduces / fixed |
| bug-1595-local-removed | TruthfulSnapshotLabel | 35 | 0.7 | 5 | d9d7bff1 reproduces / fixed |
| bug-1595-recovery-reset | RedHoldsUntilEvidence | 6,444 | 1.1 | 10 | |
| bug-1596-abandoned | NoFalseGreen | 17 | 0.7 | 4 | 12b9ddd4 reproduces / **still on main** (fixed at 7505f8b1) |
| bug-1596-superseded | NoFalseGreen | 413 | 0.7 | 7 | 12b9ddd4 reproduces / **still on main** (fixed at 7505f8b1) |
| bug-1587-B4-backlog-ages-out | RedHoldsUntilEvidence | 707 | 0.7 | 8 | |
| bug-1587-B5-stale-canary-success | RedHoldsUntilEvidence | 8,951 | 1.3 | 11 | |
| bug-1583-S1-alert-lights-dot | NoStuckAlert | 386 | 0.7 | 2 | |
| finding-ingest-red-to-amber (F1) | RedHoldsUntilEvidence | 3,780 | 0.9 | 11 | |
| finding-ingest-graced-stale-keeps-count (F3) | RedClearsToGreenOnlyOnEvidence | 19,377 | 2.3 | 16 | |
| finding-ingest-unobserved-tick-keeps-count (F2) | RedToGreenNeedsConsecutive | 10,109 | 1.6 | 13 | |
| limit-unreadable-log | NoFalseGreenStrict | 314 | 1.3 | 6 | |
| scratch-vacuum-any-receipt | VacuumNeedsFreshSnapshot | 25 | 0.8 | 4 | |
| scratch-deferred-advances | VacuumNeedsFreshSnapshot | 5 | 0.8 | 2 | |
| scratch-clean-run-keeps-alert | AlertClears | 3,822 | 4.1 | 4 | |

Every pre-fix replay matched the model on every tick (`conforms: true` in `results.json`).
`--src-ref 7505f8b1` confirms the #1596 head fixes both `bug-1596-*` cases. It reports them as
UNEXPECTED, so their `fixed_here` must flip when #1596 merges.

## Scratch break (the skill's proof step)

The working tree's `_verified_age` was edited to take `max(persisted, observed)` again (#1583 B5).
Then `check.sh --case bug-1583-B5 --case bug-1596-ab` printed
`FAIL bug-1583-B5-persisted-max … replay at working tree: reproduces … (UNEXPECTED …)` and
**exit 1**. After `git checkout -- src/brainlayer/health_verdict.py` the sha256 was back to
`3b5ef96c…aa786`, and the case passes.
