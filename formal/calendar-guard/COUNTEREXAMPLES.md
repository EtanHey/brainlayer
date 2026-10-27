# Counterexamples in calendar terms

Status: historical reproductions and current findings; no runtime fixes in this PR.
Source lines refer to the commits in `coverage.json`. Rerun logs contain full TLC states.

## Historical defects reproduced

- **`historical-sync` — `NoFalseKill`:** acquire custody; launch the guard;
  start a finite three-tick runner-record/fsync operation on the event loop;
  H cannot run; after the two-tick heartbeat deadline the guard SIGKILLs the
  healthy parent. The operation would finish before its four-tick budget.
  This reproduces the heartbeat-starvation mechanism consistent with the
  2026-10-27 01:25:55Z kill, **not proof that fsync caused that incident**.
  Source: pre-#1519 main `maintenance_calendar.py` runner record; current
  `supervisor_guard.py:251-276`. Off-loop policy is derived from the historical source.
- **`historical-missing-terminal` — `Custody`:** write runner; runner and guard
  disappear; the next run takes custody and reaps stale groups; pre-#1517 has
  no abandonment scan, so it proceeds with zero terminal outcomes for that run.
  The source at `54c580ca` has no `_recover_abandoned`.
- **`historical-worker-wedge` — `BoundedTermination`:** runner and child startup
  succeed; custody persistence wedges in an executor; #1519 round 1 keeps
  emitting H indefinitely; the guard never expires and custody stays held at
  the forty-tick horizon. The next invocation cannot acquire it.
  The old real `pulse` body emits H above the operation budget.

## Current-source findings retained

- **`finding-wedge-pre` — `BoundedTermination`:** custody is acquired; recovery
  or journal scanning wedges before `start_guard`; the operations tracker has no
  pulse task to enforce its deadline. Lock still held at tick 40.
  `maintenance_calendar.py:703-728,749-757`.
- **`finding-wedge-summary` — `BoundedTermination`:** child finishes; heartbeat
  stops; lease channel and Q/EOF let the guard reap and exit; summary fsync
  wedges; the parent still owns custody with no living guard.
  `maintenance_calendar.py:895-946`.
- **`finding-recovery` — `Custody`:** write `transcript-recovery`, then die;
  the next run ignores it as a nightly outcome and emits no abandoned/summary
  record for it. `maintenance_calendar.py:189-190,762-768`.
- **`finding-late` — `Custody`:** write nightly runner, then die; the next
  successful recovery occurs two days later; the date filter skips the pending
  run forever. `maintenance_calendar.py:223-227`.
- **`finding-two-pending` — `Custody`:** two runner identities have no terminal
  row; a later scan overwrites `latest` with the second; only that second run
  receives an abandonment. `maintenance_calendar.py:188-190,211-227`.
  Actual function probes confirm the last three findings on disposable files.
- **`finding-lost-start` — `Custody`:** the guard or environment kills the
  runner before its initial journal append completes; only torn bytes remain;
  the next scanner skips that line, finds no runner identity, and cannot publish
  its terminal outcome. `maintenance_calendar.py:115-124,180-212,758-769`.
  The passing custody cases require a durable initial runner marker.
- **`finding-guard-killed` — `BoundedTermination`:** an executor wedges and the
  environment kills the guard; suppressing H cannot kill the parent anymore.
  This exposes the necessary surviving-guard assumption, not a regression
  introduced by #1519. `maintenance_calendar.py:558-569,749-757`.
- **`registration-ack` — `RegistrationAcknowledged`:** durable record finishes;
  parent writes `+pid`; parent releases gate and the child executes before the
  guard parses `+pid`. `maintenance_calendar.py:456-458` and
  `supervisor_guard.py:296-314`. The existing promise is enqueue ordering;
  a stronger actual-registration-before-exec promise would need an acknowledgment.
  The ordinary `ExactlyOneChild` check preserves the implemented enqueue contract.

## Real-code scratch break

`--scratch-break` copies #1519's entire calendar source outside the PR and
replaces `if not overdue:` in the real `pulse` with `if True:`. An executed
pulse above budget then emits H. The derived `Budgeted=FALSE` config makes
`scratch-worker-wedge` fail `BoundedTermination`. No TLA+ change is used.
The original body suppresses H and `current-wedge-runner` passes. Both source
versions, generated configs and full traces are retained under `.build.noindex/`.

## What historical passes do not prove

The old synchronous runner can still satisfy custody ordering and one-child
safety: those fixes were already present. Pre-#1517 can still preserve healthy
heartbeats and spawn ordering; it lacks terminal recovery. Round 1's worker can
preserve no-false-kill and one-child safety while failing termination. Properties
unrelated to each historical defect are not claimed as newly discovered failures.
The historical configs focus on the named failing property (plus listed common
invariants); they are not a cross-product verification of every historical source.
There is no all-property current PASS: the stronger brief is contradicted by the
source. Fault-free bounded checks stay separate from those findings.
