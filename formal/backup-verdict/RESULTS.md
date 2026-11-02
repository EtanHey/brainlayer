# Measured run (F-TLA1 sync round)

- Date: 2026-11-02T09:28Z, on the M4, with `TLC_WORKERS=4 formal/backup-verdict/check.sh --require-sync`.
  Models run one at a time; simulation and temporal cases always use one worker.
- Tree: this branch merged with main **62f55b09**. #1600 (S1a-5, 2f983832) and #1602 (S1a-6,
  62f55b09) are both merged, so every fixed-here case runs against this working tree.
- TLC: release v1.7.4 jar (sha256 `936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88`);
  TLC's own banner reads `TLC2 Version 2.19 of 08 August 2024`.
- Result: **42 / 42 cases matched, no stale source, exit 0.** Total TLC time 189.9 s.
  Probes (`test_check.py`): 43 passed.
- State counts: pass cases explore the whole bounded space (exact). Counterexample cases stop at the
  first violation, so their counts vary with worker count and search order.

| Case | Expected = got | Distinct states | s | Replay or conformance (SHA) |
|---|---|---:|---:|---|
| current-backup | pass | 1,193,217 | 22.5 |  |
| current-vacuum | pass | 61,103 | 1.5 |  |
| current-pipes-drive | pass | 6,396 | 1.0 |  |
| current-pipes-stores | pass | 956,058 | 55.6 |  |
| current-pipes-embedding | pass | 38,064 | 2.0 |  |
| current-dot | pass | 9,224 | 2.7 |  |
| current-dot-liveness | pass | 9,224 | 39.7 |  |
| bug-1583-B1-canary-latch | RedHoldsUntilEvidence | 2,652 | 0.8 |  |
| bug-1583-B2-resume-grace | RedClearsToGreenOnlyOnEvidence | 2,632 | 0.8 | 672c9f97 reproduces, every tick / fixed in working tree |
| bug-1583-B3-renewable-grace | NoFalseGreen | 4,248 | 0.9 | 672c9f97 reproduces, every tick / fixed in working tree |
| bug-1583-B4-drive-latch | RedHoldsUntilEvidence | 193 | 0.7 |  |
| bug-1583-B5-persisted-max | NoFalseGreen | 402 | 0.7 | 672c9f97 reproduces, every tick / fixed in working tree |
| bug-1589-B5a-grace-without-clock | NoFalseGreen | 861 | 0.7 | 7f7f6245 reproduces, every tick / fixed in working tree |
| bug-1589-B5b-shared-revocation | NoFalseGreen | 1,373 | 0.8 | 7f7f6245 reproduces, every tick / fixed in working tree |
| bug-1589-B5c-legacy-identity | NoFalseGreen | 561 | 0.7 | 7f7f6245 reproduces, every tick / fixed in working tree |
| bug-1595-local-removed | TruthfulSnapshotLabel | 66 | 0.7 | d9d7bff1 reproduces, every tick / fixed in working tree |
| bug-1595-recovery-reset | RedHoldsUntilEvidence | 18,686 | 1.5 |  |
| bug-1596-abandoned | NoFalseGreen | 46 | 0.7 | 12b9ddd4 reproduces, every tick / fixed in working tree |
| bug-1596-superseded | NoFalseGreen | 1,394 | 0.7 | 12b9ddd4 reproduces, every tick / fixed in working tree |
| bug-1587-B4-backlog-ages-out | RedHoldsUntilEvidence | 629 | 0.8 |  |
| bug-1587-B5-stale-canary-success | RedHoldsUntilEvidence | 128,366 | 4.3 |  |
| bug-1583-S1-alert-lights-dot | NoStuckAlert | 965 | 0.7 |  |
| bug-s1a5-F1-red-to-amber | RedHoldsUntilEvidence | 3,885 | 1.0 | 67d23f14 reproduces, every tick / fixed in working tree |
| bug-s1a5-F2-unobserved-tick-keeps-count | RedToGreenNeedsConsecutive | 7,228 | 1.1 | 67d23f14 reproduces, every tick / fixed in working tree |
| bug-s1a5-F3-graced-stale-keeps-count | RedClearsToGreenOnlyOnEvidence | 20,295 | 1.5 | 67d23f14 reproduces, every tick / fixed in working tree |
| bug-renewable-grace | NoFalseGreenStall | 320 | 0.8 | 67d23f14 reproduces, every tick / fixed in working tree |
| bug-renewable-grace-ingest | NoFalseGreenStall | 5,008 | 1.0 | 67d23f14 reproduces, every tick / fixed in working tree |
| bug-renewable-grace-stores | NoFalseGreenStall | 67,498 | 2.3 | 67d23f14 reproduces, every tick / fixed in working tree |
| conform-ingest-main | conforms | simulated | 3.8 | 3,000 traces, 8,934 ticks vs 67d23f14: 0 diverging |
| conform-embedding-main | conforms | simulated | 4.6 | 3,000 traces, 17,955 ticks vs 67d23f14: 0 diverging |
| conform-stores-main | conforms | simulated | 5.2 | 3,000 traces, 11,861 ticks vs 67d23f14: 0 diverging |
| conform-ingest-s1a5 | conforms | simulated | 3.7 | 3,000 traces, 8,934 ticks vs 4e07651b: 0 diverging |
| conform-ingest-s1a6 | conforms | simulated | 3.9 | 3,000 traces, 8,934 ticks vs working tree: 0 diverging |
| conform-embedding-s1a6 | conforms | simulated | 4.8 | 3,000 traces, 17,955 ticks vs working tree: 0 diverging |
| conform-stores-s1a6 | conforms | simulated | 5.2 | 3,000 traces, 11,861 ticks vs working tree: 0 diverging |
| finding-ingest-red-to-amber | pass | 26,138 | 1.6 |  |
| finding-ingest-graced-stale-keeps-count | pass | 26,138 | 1.6 |  |
| finding-ingest-unobserved-tick-keeps-count | pass | 26,138 | 1.5 |  |
| limit-unreadable-log | NoFalseGreenStrict | 514 | 0.7 |  |
| scratch-vacuum-any-receipt | VacuumNeedsFreshSnapshot | 68 | 0.7 |  |
| scratch-deferred-advances | VacuumNeedsFreshSnapshot | 40 | 0.7 |  |
| scratch-clean-run-keeps-alert | AlertClears | 4,003 | 3.7 |  |

The two fixes are proven in this tree three ways:
- The three S1a-5 counterexamples and the three renewable-grace counterexamples reproduce, tick for
  tick, at 67d23f14 (before both fixes), and none reproduces in the working tree.
- 3,000 random S1a-5+S1a-6 behaviours per check agree with the working tree on every tick:
  ingest 8,934, embedding 17,955 and stores 11,861 ticks, 0 diverging.
- The historical cases still agree with their own code: the main model with 67d23f14, and the S1a-5
  model with 4e07651b.

## Release-gate flake (F-TLA1-FLAKE): `current-dot-liveness`

CI reported `FAIL current-dot-liveness: expected pass, got error [9224 states, …]` in 2 of 3 runs of
the TLA+ job: job 125151087872 at e2cfe4a0 and job 125158268946 at 7f197e6b. Both ran with
`TLC_WORKERS=2` on GitHub's 2-vCPU ubuntu runner, and both errored after the full 9,224-state
exploration, i.e. in TLC's liveness phase. The run at bffbd5d7 passed. The checker did not print
TLC's output, so CI gives no further evidence.

Local runs on the M4, 40 in total:

| Setup | Results | Pass |
|---|---|---|
| direct TLC, local wrapper, -workers 2 | pass, pass, pass, pass, pass, pass, pass, pass, pass, pass | 10/10 |
| direct TLC, java -XX:ActiveProcessorCount=2 -Xmx512m, -workers 2 | pass, pass, pass, pass, pass, pass, pass, pass, pass, pass | 10/10 |
| pre-fix (b), TLC_JAR java -Xmx512m, TLC_WORKERS=2, through check.py | pass, pass, pass, pass, pass, pass, pass, pass, pass, pass | 10/10 |
| with fix (b), TLC_JAR java -Xmx512m, TLC_WORKERS=2, through check.py | pass, pass, pass, pass, pass, pass, pass, pass, pass, pass | 10/10 |

**Root cause: not established.** The error has never occurred outside the GitHub runner, including
with the CI launch line (`java -Xmx512m`, two workers) and with the JVM limited to two processors.
What the evidence pins down is the phase: after full exploration, in multi-worker temporal checking.

**Fixes:**
- (a) An `error` outcome prints TLC's exit code and the last 40 lines of its output. The CI job also
  uploads every case's `.log`, `.trace.txt` and `results.json` as an artifact on failure.
- (b) Every temporal case (a config with `PROPERTIES`) runs on one TLC worker, whatever
  `TLC_WORKERS` says. TLC's banner confirms "with 1 worker" in all ten post-fix runs. This removes
  the multi-worker liveness path from the gate.

If it recurs on CI, the printed tail and the artifact will name the cause.

## Scratch break (the skill's proof step, round 1)

The working tree's `_verified_age` was edited to take `max(persisted, observed)` again (#1583 B5).
`check.sh --case bug-1583-B5` then printed `FAIL … replay at working tree: reproduces … (UNEXPECTED …)`
and exited 1. With the source restored, the case passes.
