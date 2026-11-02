# Measured run (F-TLA1 R3)

- Date: 2026-11-02T07:45Z, on the M4, with `TLC_WORKERS=4 formal/backup-verdict/check.sh --require-sync`.
  Models run one at a time; simulation cases always use one worker.
- Tree: this branch merged with main de545db2. #1598 is test-only, so no covered source changed
  since 67d23f14. S1a-5 (#1600, 4e07651b) and S1a-6 (`fix/wake-grace-once`, 8eecb5ee) are **not**
  on main.
- TLC: release v1.7.4 jar (sha256 `936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88`);
  TLC's own banner reads `TLC2 Version 2.19 of 08 August 2024`.
- Result: **42 / 42 cases matched, no stale source, exit 0.** Total TLC time 184.6 s.
  Probes (`test_check.py`): 36 passed.
- State counts: pass cases explore the whole bounded space, so their counts are exact. Counterexample
  cases stop at the first violation, so their counts vary with worker count and search order.
  `current-backup` is 1,193,217 states, down from 1,352,435 in R2: `Supersede` now requires a newer
  verified upload, as the producer does, which removes states the producer cannot reach.
- Simulation receipts are exact for this setup: one worker, `num=3000`, depth 60, seed 7. REV-FTLA1-R1
  observed 6,000 traces / 17,957 ticks with two workers, because `num` is per worker. The checker now
  pins simulation to one worker.

| Case | Expected = got | Distinct states | s | Replay or conformance (SHA) |
|---|---|---:|---:|---|
| current-backup | pass | 1,193,217 | 21.3 |  |
| current-vacuum | pass | 61,103 | 1.4 |  |
| current-pipes-drive | pass | 6,396 | 1.1 |  |
| current-pipes-stores | pass | 956,058 | 58.2 |  |
| current-pipes-embedding | pass | 38,064 | 2.6 |  |
| current-dot | pass | 9,224 | 3.3 |  |
| current-dot-liveness | pass | 9,224 | 19.3 |  |
| bug-1583-B1-canary-latch | RedHoldsUntilEvidence | 3,436 | 1.2 |  |
| bug-1583-B2-resume-grace | RedClearsToGreenOnlyOnEvidence | 2,084 | 1.5 | 672c9f97 reproduces, every tick / fixed |
| bug-1583-B3-renewable-grace | NoFalseGreen | 3,797 | 1.2 | 672c9f97 reproduces, every tick / fixed |
| bug-1583-B4-drive-latch | RedHoldsUntilEvidence | 187 | 0.8 |  |
| bug-1583-B5-persisted-max | NoFalseGreen | 378 | 1.0 | 672c9f97 reproduces, every tick / fixed |
| bug-1589-B5a-grace-without-clock | NoFalseGreen | 880 | 1.1 | 7f7f6245 reproduces, every tick / fixed |
| bug-1589-B5b-shared-revocation | NoFalseGreen | 1,525 | 1.1 | 7f7f6245 reproduces, every tick / fixed |
| bug-1589-B5c-legacy-identity | NoFalseGreen | 434 | 1.1 | 7f7f6245 reproduces, every tick / fixed |
| bug-1595-local-removed | TruthfulSnapshotLabel | 87 | 1.0 | d9d7bff1 reproduces, every tick / fixed |
| bug-1595-recovery-reset | RedHoldsUntilEvidence | 12,781 | 1.2 |  |
| bug-1596-abandoned | NoFalseGreen | 53 | 0.7 | 12b9ddd4 reproduces, every tick / fixed |
| bug-1596-superseded | NoFalseGreen | 1,587 | 0.8 | 12b9ddd4 reproduces, every tick / fixed |
| bug-1587-B4-backlog-ages-out | RedHoldsUntilEvidence | 679 | 0.8 |  |
| bug-1587-B5-stale-canary-success | RedHoldsUntilEvidence | 127,464 | 4.4 |  |
| bug-1583-S1-alert-lights-dot | NoStuckAlert | 965 | 0.7 |  |
| bug-s1a5-F1-red-to-amber | RedHoldsUntilEvidence | 3,378 | 1.0 | 67d23f14 reproduces, every tick / **still on main** |
| bug-s1a5-F2-unobserved-tick-keeps-count | RedToGreenNeedsConsecutive | 7,303 | 1.1 | 67d23f14 reproduces, every tick / **still on main** |
| bug-s1a5-F3-graced-stale-keeps-count | RedClearsToGreenOnlyOnEvidence | 20,235 | 1.5 | 67d23f14 reproduces, every tick / **still on main** |
| bug-renewable-grace | NoFalseGreenStall | 330 | 0.8 | 67d23f14 reproduces, every tick / **still on main** |
| bug-renewable-grace-ingest | NoFalseGreenStall | 5,637 | 1.3 | 67d23f14 reproduces, every tick / **still on main** |
| bug-renewable-grace-stores | NoFalseGreenStall | 52,411 | 2.2 | 67d23f14 reproduces, every tick / **still on main** |
| conform-ingest-main | conforms | simulated | 5.0 | 3,000 traces, 8,934 ticks vs 67d23f14: 0 diverging |
| conform-embedding-main | conforms | simulated | 5.1 | 3,000 traces, 17,955 ticks vs 67d23f14: 0 diverging |
| conform-stores-main | conforms | simulated | 5.6 | 3,000 traces, 11,861 ticks vs 67d23f14: 0 diverging |
| conform-ingest-s1a5 | diverges | simulated | 4.4 | 3,000 traces, 8,934 ticks vs working tree: 2 diverging |
| conform-ingest-s1a6 | diverges | simulated | 5.1 | 3,000 traces, 8,934 ticks vs working tree: 218 diverging |
| conform-embedding-s1a6 | diverges | simulated | 7.6 | 3,000 traces, 17,955 ticks vs working tree: 160 diverging |
| conform-stores-s1a6 | diverges | simulated | 6.1 | 3,000 traces, 11,861 ticks vs working tree: 212 diverging |
| finding-ingest-red-to-amber | pass | 26,138 | 2.0 |  |
| finding-ingest-graced-stale-keeps-count | pass | 26,138 | 1.8 |  |
| finding-ingest-unobserved-tick-keeps-count | pass | 26,138 | 1.9 |  |
| limit-unreadable-log | NoFalseGreenStrict | 592 | 0.9 |  |
| scratch-vacuum-any-receipt | VacuumNeedsFreshSnapshot | 70 | 0.7 |  |
| scratch-deferred-advances | VacuumNeedsFreshSnapshot | 35 | 0.7 |  |
| scratch-clean-run-keeps-alert | AlertClears | 6,367 | 4.0 |  |

## Against S1a-6 (`--src-ref 8eecb5ee`): the second, independent check of S1a-6

Exit 1 by design: all nine results are the "fix is present" side of the ratchet.

| Case | Result |
|---|---|
| bug-s1a5-F1-red-to-amber | 67d23f14: reproduces, every tick; 8eecb5ee: **does not reproduce** |
| bug-s1a5-F2-unobserved-tick-keeps-count | 67d23f14: reproduces, every tick; 8eecb5ee: **does not reproduce** |
| bug-s1a5-F3-graced-stale-keeps-count | 67d23f14: reproduces, every tick; 8eecb5ee: **does not reproduce** |
| bug-renewable-grace | 67d23f14: reproduces, every tick; 8eecb5ee: **does not reproduce** |
| bug-renewable-grace-ingest | 67d23f14: reproduces, every tick; 8eecb5ee: **does not reproduce** |
| bug-renewable-grace-stores | 67d23f14: reproduces, every tick; 8eecb5ee: **does not reproduce** |
| conform-ingest-s1a6 | 3,000 traces, 8,934 ticks: **0 diverging** |
| conform-embedding-s1a6 | 3,000 traces, 17,955 ticks: **0 diverging** |
| conform-stores-s1a6 | 3,000 traces, 11,861 ticks: **0 diverging** |

## Against S1a-5 (`--src-ref 4e07651b`)

| Case | Result |
|---|---|
| bug-s1a5-F1-red-to-amber | 67d23f14: reproduces, every tick; 4e07651b: **does not reproduce** |
| bug-s1a5-F2-unobserved-tick-keeps-count | 67d23f14: reproduces, every tick; 4e07651b: **does not reproduce** |
| bug-s1a5-F3-graced-stale-keeps-count | 67d23f14: reproduces, every tick; 4e07651b: **does not reproduce** |
| conform-ingest-s1a5 | 3,000 traces, 8,934 ticks: **0 diverging** |

## Scratch break (the skill's proof step, round 1)

The working tree's `_verified_age` was edited to take `max(persisted, observed)` again (#1583 B5).
`check.sh --case bug-1583-B5` then printed `FAIL … replay at working tree: reproduces … (UNEXPECTED …)`
and exited 1. With the source restored, the case passes.

In round 3, the fail-closed replay caught a real checker regression. A shadowed helper turned every
backup receipt's `attempted_at` into a list, the reader saw no receipt, and `bug-1596-superseded`
failed instead of passing. Probe `test_backup_replay_receipts_carry_iso_times` now guards it.
