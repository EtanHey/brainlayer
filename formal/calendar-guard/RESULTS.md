# Measured TLC results

Status: **NOT_GREEN**. All 23 expected outcomes reproduced; this is not an all-property pass.

Run: 2026-10-27 07:25:29Z, TLC 1.7.4, one worker, 512 MiB heap, fingerprint 0.
Command: `formal/calendar-guard/check.sh --scratch-break` (exit 1: current contract findings remain).
`--expect-known` accepts these findings with exit 0; exit 2 means unexpected results.
Local logs: `.build.noindex/calendar-guard/20261027T072529Z/` (excluded from Git; rerun to regenerate).

| Config | TLC exit | Distinct states | Depth | Wall seconds | Result |
|---|---:|---:|---:|---:|---|
| current-healthy | 0 | 1421 | 39 | 1.211 | PASS within stated scope |
| current-wedge-runner | 0 | 95 | 16 | 0.684 | PASS within stated scope |
| current-wedge-custody | 0 | 215 | 22 | 0.702 | PASS within stated scope |
| current-wedge-finish | 0 | 779 | 33 | 0.808 | PASS within stated scope |
| current-loop-wedge | 0 | 143 | 18 | 0.785 | PASS within stated scope |
| current-q-timeout | 0 | 2477 | 41 | 0.795 | PASS within stated scope |
| current-parent-killed | 0 | 1529 | 39 | 0.808 | PASS within stated scope |
| current-journal | 0 | 70 | 23 | 0.795 | PASS within stated scope |
| current-completed | 0 | 13 | 13 | 0.681 | PASS within stated scope |
| finding-wedge-pre | 12 | 122 | 42 | 0.715 | BoundedTermination |
| finding-wedge-summary | 12 | 5340 | 58 | 0.819 | BoundedTermination |
| finding-recovery | 12 | 27 | 9 | 0.828 | Custody |
| finding-late | 12 | 27 | 9 | 0.675 | Custody |
| finding-two-pending | 12 | 40 | 12 | 0.751 | Custody |
| finding-lost-start | 12 | 7 | 7 | 0.696 | Custody |
| registration-ack | 12 | 67 | 11 | 0.726 | RegistrationAcknowledged |
| finding-terminal-record-count | 12 | 27 | 9 | 1.081 | TerminalRecordUnique |
| finding-guard-killed | 12 | 3432 | 49 | 0.817 | BoundedTermination |
| historical-sync | 12 | 34 | 8 | 0.747 | NoFalseKill |
| historical-missing-terminal | 12 | 27 | 9 | 0.789 | Custody |
| historical-worker-wedge | 12 | 726 | 48 | 0.795 | BoundedTermination |
| historical-worker-liveness | 13 | 743 | 48 | 0.801 | EventuallyReleased |
| scratch-worker-wedge | 12 | 363 | 45 | 0.749 | BoundedTermination |

Depth is the complete BFS depth or TLC's final `Progress(N)` depth for temporal failures.
Invariant counterexamples stop exploration early. Runtimes are process wall measurements, not a performance comparison.

## Checked properties by config

- `current-healthy`: TypeOK, ExactlyOneChild, NoFalseKill, BoundedTermination, EventuallyReleased
- `current-wedge-runner`: TypeOK, ExactlyOneChild, NoFalseKill, BoundedTermination, EventuallyReleased
- `current-wedge-custody`: TypeOK, ExactlyOneChild, NoFalseKill, BoundedTermination, EventuallyReleased
- `current-wedge-finish`: TypeOK, ExactlyOneChild, NoFalseKill, BoundedTermination, EventuallyReleased
- `current-loop-wedge`: TypeOK, ExactlyOneChild, NoFalseKill, BoundedTermination, EventuallyReleased
- `current-q-timeout`: TypeOK, ExactlyOneChild, NoFalseKill, BoundedTermination, EventuallyReleased
- `current-parent-killed`: TypeOK, ExactlyOneChild, NoFalseKill, BoundedTermination, EventuallyReleased
- `current-journal`: TypeOK, AtMostOneTerminal, PidReuseSafe, Custody, TornJournalLive
- `current-completed`: TypeOK, AtMostOneTerminal, PidReuseSafe, Custody, TornJournalLive
- `finding-wedge-pre`: TypeOK, ExactlyOneChild, NoFalseKill, BoundedTermination
- `finding-wedge-summary`: TypeOK, ExactlyOneChild, NoFalseKill, BoundedTermination
- `finding-recovery`: TypeOK, AtMostOneTerminal, PidReuseSafe, Custody
- `finding-late`: TypeOK, AtMostOneTerminal, PidReuseSafe, Custody
- `finding-two-pending`: TypeOK, AtMostOneTerminal, PidReuseSafe, Custody
- `finding-lost-start`: TypeOK, AtMostOneTerminal, PidReuseSafe, Custody
- `registration-ack`: RegistrationAcknowledged
- `finding-terminal-record-count`: TerminalRecordUnique
- `finding-guard-killed`: TypeOK, ExactlyOneChild, NoFalseKill, BoundedTermination
- `historical-sync`: NoFalseKill
- `historical-missing-terminal`: TypeOK, AtMostOneTerminal, PidReuseSafe, Custody
- `historical-worker-wedge`: TypeOK, ExactlyOneChild, NoFalseKill, BoundedTermination
- `historical-worker-liveness`: TypeOK, ExactlyOneChild, EventuallyReleased
- `scratch-worker-wedge`: TypeOK, ExactlyOneChild, NoFalseKill, BoundedTermination

## Real-source probes

- `pulse`: historical synchronous/R1 sources send H above budget; current suppresses it; scratch sends it again.
- `record` + `_recover_abandoned`, two scans each: recent nightly yields 1 abandoned + 1 synthetic summary despite a torn line; recovery and two-day-old runs yield 0; two pending identities yield only identity 2.
- `run_guard`: H at logical time 1, Q at 2, repeated H at 3; Q expires at 5, invokes fake SIGKILL and reap. No OS signals are sent.
- `Custody.recover`: a reused numeric PID with a different birth is absent from the reap set.
- Scratch source SHA-256: `9ad28230d86cb3f8e279c773c56e1afc68a18323ea42b1acba0a5b64661b7e8d`.
- Original calendar SHA-256: `dbdca628b3991a7f1ddd437619fb0e57d4bb0ff63362c9c00f600296f8fc7cef`.

Additional scoped verification at checkout main `f01469cf`: `PYTHONPATH=src pytest tests/test_maintenance_calendar.py -q -k "scan_damage or identity_and_interrupted_receipt or multiday_journal"` → **13 passed, 53 deselected, 1 existing RequestsDependencyWarning in 1.66s**.
`ruff check formal/calendar-guard/check.py`, `sh -n formal/calendar-guard/check.sh`, and `git diff --check` passed.

No production DB, live socket, live service, install, deploy, full suite or CI wiring was used.

Initial pre-push at `32128ae6` was RED: registration tests had 3 passes and a teardown error because shared `origin/main` changed during pytest. Isolated tests had 40 passes, bun had 5 passes, and shell regression passed. This adverse result is retained; no bypass or timeout change was used.
