# Latest round-1 TLC receipt

Status: **NOT_GREEN**, wrapper **exit 1**, **25/25 expected outcomes reproduced**.
One invocation only: `formal/calendar-guard/check.sh --scratch-break`,
2026-10-27 08:45:34Z. TLC version observed from each log:
`2.19 of 08 August 2024 (rev: 5a47802)` (the 1.7.4 jar distribution used by
prior receipts). No hardcoded version banner. Load gate observed 22.16–23.74,
no admission wait. One worker and 512 MiB heap, as before.
Local evidence: `.build.noindex/calendar-guard/20261027T084534Z-49ea8ifs/`.

All 23 prior case verdicts remain unchanged; state counts/depths below supersede
the earlier counts for the revised models. Ten bounded current passes, nine
retained current findings (including labeled abstraction/policy assumptions),
four historical failures, the scratch failure, and one deliberately false
reachability assertion total 25 expected outcomes. The reachability assertion
is not an additional product finding.

| Config | TLC exit | Distinct states | Depth | Wall seconds | Violation |
|---|---:|---:|---:|---:|---|
| current-healthy | 0 | 1055 | 39 | 1.262 | PASS within scope |
| current-wedge-runner | 0 | 95 | 16 | 0.833 | PASS within scope |
| current-wedge-custody | 0 | 215 | 22 | 0.864 | PASS within scope |
| current-wedge-finish | 0 | 752 | 33 | 0.886 | PASS within scope |
| current-loop-wedge | 0 | 143 | 18 | 0.817 | PASS within scope |
| current-q-timeout | 0 | 1718 | 41 | 0.930 | PASS within scope |
| current-parent-killed | 0 | 1217 | 39 | 0.878 | PASS within scope |
| current-journal | 0 | 70 | 23 | 1.011 | PASS within scope |
| current-completed | 0 | 13 | 13 | 0.779 | PASS within scope |
| current-completed-then-pending | 0 | 28 | 26 | 0.909 | PASS within scope |
| probe-parent-before-receipt | 12 | 85 | 12 | 0.796 | ParentDeathAfterReceipt |
| finding-wedge-pre | 12 | 122 | 42 | 0.815 | BoundedTermination |
| finding-wedge-summary | 12 | 3042 | 59 | 0.883 | BoundedTermination |
| finding-recovery | 12 | 27 | 9 | 0.834 | Custody |
| finding-late | 12 | 27 | 9 | 0.809 | Custody |
| finding-two-pending | 12 | 40 | 12 | 0.756 | Custody |
| finding-lost-start | 12 | 7 | 7 | 0.773 | Custody |
| registration-ack | 12 | 67 | 11 | 0.814 | RegistrationAcknowledged |
| finding-terminal-record-count | 12 | 27 | 9 | 0.790 | TerminalRecordUnique |
| finding-guard-killed | 12 | 3432 | 49 | 0.854 | BoundedTermination |
| historical-sync | 12 | 34 | 8 | 0.892 | NoFalseKill |
| historical-missing-terminal | 12 | 27 | 9 | 0.806 | Custody |
| historical-worker-wedge | 12 | 726 | 48 | 0.999 | BoundedTermination |
| historical-worker-liveness | 13 | 743 | 48 | 0.963 | EventuallyReleased |
| scratch-worker-wedge | 12 | 363 | 45 | 0.844 | BoundedTermination |

Round-1 changes and regression evidence:

- Configs for parent death before receipt and a completed run followed by a
  pending run were written before model edits. TLC was reserved for this one
  final run; no pre-fix model run is claimed.
- `probe-parent-before-receipt` fails the deliberately false reachability
  assertion at depth 12: after ReleaseGate, `parent=FALSE`, `sent=TRUE`,
  `registered=FALSE`, child running. Buffered receipt precedes reap. The
  parent-killed liveness config still passes. This is distinct from lost-start.
- `current-completed-then-pending` passes Custody (28 states, depth 26) after
  AbandonSummary credits scanned terminal outcomes too. The missing-terminal
  two-pending finding remains, explicitly labeled as an injected abstraction.
- Python probe red→green: **12 failed, 6 passed** before checker edits;
  **18 passed in 0.10 s** after. Launch-failure/nonzero-exit tests passed
  already and retain their existing correct behavior.
- Relative TLC paths are absolute before changing cwd; versions are observed
  or null; ordinary probe crashes exit 2 while parser SystemExit is preserved;
  duplicated coverage pins are checked; stale pre-#1519 checkout is refused.
- All four Opus wording fixes applied; external nightly/hourly bounds and
  hard-coded BirthChecked/ExactlyOneChild limitations are explicit in README.
- Ruff lint/format, shell syntax and diff whitespace checks passed. No runtime
  code changed, no CI wiring, no M1 access or full suite, no new reviewer.

---

# Earlier MBP/M1 receipts (models before round 1)

Status: **NOT_GREEN**. All 23 expected outcomes reproduced; this is not an all-property pass.

Run: 2026-10-27 07:33:06Z, TLC 1.7.4, one worker, 512 MiB heap, fingerprint 0.
Command: `formal/calendar-guard/check.sh --scratch-break` (exit 1: current contract findings remain).
`--expect-known` accepts these findings with exit 0; exit 2 means unexpected results.
Local logs: `.build.noindex/calendar-guard/20261027T073306Z/` (excluded from Git; rerun to regenerate).

| Config | TLC exit | Distinct states | Depth | Wall seconds | Result |
|---|---:|---:|---:|---:|---|
| current-healthy | 0 | 1421 | 39 | 1.875 | PASS within stated scope |
| current-wedge-runner | 0 | 95 | 16 | 0.979 | PASS within stated scope |
| current-wedge-custody | 0 | 215 | 22 | 0.947 | PASS within stated scope |
| current-wedge-finish | 0 | 779 | 33 | 0.916 | PASS within stated scope |
| current-loop-wedge | 0 | 143 | 18 | 0.872 | PASS within stated scope |
| current-q-timeout | 0 | 2477 | 41 | 1.105 | PASS within stated scope |
| current-parent-killed | 0 | 1529 | 39 | 0.961 | PASS within stated scope |
| current-journal | 0 | 70 | 23 | 0.850 | PASS within stated scope |
| current-completed | 0 | 13 | 13 | 0.906 | PASS within stated scope |
| finding-wedge-pre | 12 | 122 | 42 | 0.818 | BoundedTermination |
| finding-wedge-summary | 12 | 5340 | 58 | 1.377 | BoundedTermination |
| finding-recovery | 12 | 27 | 9 | 1.083 | Custody |
| finding-late | 12 | 27 | 9 | 0.890 | Custody |
| finding-two-pending | 12 | 40 | 12 | 0.957 | Custody |
| finding-lost-start | 12 | 7 | 7 | 0.873 | Custody |
| registration-ack | 12 | 67 | 11 | 0.851 | RegistrationAcknowledged |
| finding-terminal-record-count | 12 | 27 | 9 | 0.836 | TerminalRecordUnique |
| finding-guard-killed | 12 | 3432 | 49 | 1.168 | BoundedTermination |
| historical-sync | 12 | 34 | 8 | 0.976 | NoFalseKill |
| historical-missing-terminal | 12 | 27 | 9 | 0.828 | Custody |
| historical-worker-wedge | 12 | 726 | 48 | 0.909 | BoundedTermination |
| historical-worker-liveness | 13 | 743 | 48 | 0.904 | EventuallyReleased |
| scratch-worker-wedge | 12 | 363 | 45 | 0.925 | BoundedTermination |

Depth is the complete BFS depth or TLC's final `Progress(N)` depth for temporal failures.
Invariant counterexamples stop exploration early. Runtimes are process wall measurements, not a performance comparison.

Every TLC process first ran `sysctl -n vm.loadavg`; observed one-minute load ranged from 25.74 to 29.26, so none required a pressure wait. The runner now refuses startup after ten minutes above 40. A stdlib-only gate probe verified waiting above 40, admission at 40, and refusal at 600 seconds; no pytest or Swift suite was manually run in this resumed lane.

The recovered branch was merged with `origin/main` at `a93f28f9f7f4b6edad107a23986bccfe178d9483` before this run. Its three modeled runtime files match the pinned composite byte-for-byte.

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

Prior child verification (not rerun under HOST-PRESSURE) at checkout main `f01469cf`: `PYTHONPATH=src pytest tests/test_maintenance_calendar.py -q -k "scan_damage or identity_and_interrupted_receipt or multiday_journal"` → **13 passed, 53 deselected, 1 existing RequestsDependencyWarning in 1.66s**.
`ruff check formal/calendar-guard/check.py`, `sh -n formal/calendar-guard/check.sh`, and `git diff --check` passed.

No production DB, live socket, live service, install, deploy, full suite or CI wiring was used.

Initial pre-push at `32128ae6` was RED: registration tests had 3 passes and a teardown error because shared `origin/main` changed during pytest. Isolated tests had 40 passes, bun had 5 passes, and shell regression passed. This adverse result is retained; no bypass or timeout change was used.

Resumed-lane CodeRabbit pre-commit pass was unavailable: CLI 0.7.6 returned `Rate limit exceeded` and requested a paid `--use-credits` rerun (maximum 200 cents). No paid rerun, CLI upgrade, reviewer child, or approval bypass was used. Claude pair review remains lead-routed after PR open.

## M1 handoff after the local run

The lead requested M1 offload after the 07:33Z run above had finished. No TLC
process was started after that instruction. The final runner adds `--case`,
`--prepare-only` and `--scratch-ref`; the 23 reference configs are committed.
All five handoff selections passed source/config preparation with TLC disabled.
The current selection contains nine passing cases and nine retained findings.
An unknown config returns exit 2. The final selector changes still need the M1
TLC run; the earlier local results do not claim that later execution.

The separate scratch branch `scratch/t2-calendar-guard-budget-20261027` at
`8a30073b9555d99bb0cc485eac798fe6a3b4b8c3` carries the patched full module
copy at `formal/calendar-guard/scratch/maintenance_calendar.py`. Runtime paths
are unchanged on both branches. The source probe verifies that this copy differs
from the pinned real module only by the pulse budget break. README.md lists the
five exact runner commands and expected failures.

The first resumed push at `ac9b80bf` passed its queued scoped hook: unit suite
correctly skipped because formal files map no pytest targets; registration 3
passed, isolated routing/eval 40 passed, bun 5 passed, shell regression passed.
This does not erase the earlier RED push recorded above.

## M1 confirmation and launcher override

The lead-supplied M1 receipt (`T2-m1-run.md`, 2026-10-27) confirms all 23
outcomes above at model head `489fdd89840b6e233e82086a00b3dd9a2f37872c`
and scratch input `8a30073b9555d99bb0cc485eac798fe6a3b4b8c3`.
One run, exit **1 / NOT_GREEN**, no timeout, no rerun. Its command was
`formal/calendar-guard/check.sh --scratch-break --scratch-ref 8a30073b9555d99bb0cc485eac798fe6a3b4b8c3`,
under pgrun and `/usr/bin/time -l`. Real **18.61 s**, user 23.73 s,
sys 2.03 s; maximum RSS **125222912 bytes (125 MB)**. Admission load 5.63;
per-case load 5.36–5.63. State counts and depths match the table above.
This is a report receipt from the assigned M1 lane, not a new local run.

The M1 preflight exposed the hardcoded executable path. The confirmation run
used an explicitly authorized `~/.local/bin/tlc` wrapper. `TLC` now selects an
alternate executable without requiring that installation; the default remains
`~/.local/bin/tlc`.

T2B verification under HOST PRESSURE (no TLC):

- First `pytest -q formal/calendar-guard/test_check.py`: **2 failed, 1 passed**;
  both override paths still selected the hardcoded default.
- After the launcher fix: **3 passed in 0.02 s**. The probe intercepts
  `subprocess.run`; no TLC process can start. It checks the default, an absolute
  path containing spaces, and tilde expansion.
- `check.sh --prepare-only --scratch-break --scratch-ref 8a30073b9555d99bb0cc485eac798fe6a3b4b8c3`:
  **23 configs prepared, exit 0, TLC NOT RUN**. This proves preparation only;
  it does not replace or extend the earlier model verdict.
- Ruff lint/format, shell syntax and diff whitespace checks passed.
- Runtime diff against `origin/main` (`src brain-bar tests scripts`) is empty.

No runtime code changed. No CI wiring (follow-up). Lead-routed review pending.

T2B CodeRabbit pre-commit pass: CLI 0.7.6 returned `Rate limit exceeded`
(exit 1), offering a paid rerun capped at 775 cents for 31 files. No paid
rerun or CLI upgrade was performed. No independent review verdict is claimed.
