---
name: verify-backup-verdict
description: Model-check BrainLayer's backup → snapshot → VACUUM → BrainBar-dot logic with TLA+. Use before merging any change to src/brainlayer/health_verdict.py, backup_daily.py receipts, maintenance.py's weekly VACUUM gate, maintenance_calendar.py's last_weekly, or job_alerts.py, or when someone asks whether the dot can be falsely green, stuck red, or cleared without evidence.
---

# Verify the backup verdict

This skill checks one promise: the dot (plan section 3d) is red exactly when memory is at risk or a
pipe stopped. It is never green without a verified copy, never red only because of a job alert,
and never clears red without the evidence section 3d names. Everything lives in
`formal/backup-verdict/`. Read its README first.

## 1. Does your change touch the verified code?

Run `python3 formal/backup-verdict/check.sh --list` and open `coverage.json`. Your change is in
scope if it edits any file under `sources`, or any `path:line` under `actions`. Look out
especially for `_verified_age`, `evaluate()`'s unknown / hold / hysteresis / wake-grace branches,
`backup_facts`, `badge_alerts`, `_recent_local_verified_backup`, the VACUUM `deferred` paths in
`run_maintenance`, `maintenance.main`'s `report(...)` calls and `last_weekly`.

## 2. Run it

```sh
formal/backup-verdict/check.sh            # about 3 min; exit 0 = every case matched
python3 -m pytest -q formal/backup-verdict/test_check.py -p no:cacheprovider -o pythonpath=
```

This was proved once: a scratch edit that put #1583 B5's persisted max back into `_verified_age`
made `bug-1583-B5` FAIL (exit 1), and restoring the line made it pass.

Each line reads `ok|FAIL <case>: expected X, got Y`. Under a bug case, `replay at <ref>` says
whether the real code at the pre-fix commit reproduces the counterexample, and at the current tree
whether it still does. Exit 0 with a `SYNC STALE` line means the models still hold, but someone must
confirm the model describes the changed source (step 4).

## 3. A real bug, or a modelling error?

Open `.build.noindex/backup-verdict/<run>/<case>.trace.txt` (the shortest path, changed variables
only), or read the result with the pstack `/statechart` skill, which tells each violation as its
shortest path in plain words.

- **A `current-*` case fails:** the change broke the promise. For a `BackupVerdict` trace, confirm
  it against the real code: copy the bug-case pattern (a config plus a `prefix` ref in `CASES`), or
  run `check.sh --case <case> --src-ref <your commit>`. If the real code reproduces the trace, it is
  a real bug. Report it in the code's words (receipts, ticks, check states), as `COUNTEREXAMPLES.md`
  does.
- **A `bug-*` case FAILs with "UNEXPECTED" on the working-tree replay:** either your change
  brought a fixed bug back (a real regression; fix the code), or the bug's PR just merged and its
  `fixed_here` is still `"no"` (flip it to `"yes"`).
- **A `bug-*` case passes, or its replay "does not reproduce" at its pre-fix ref:** the model
  drifted from the code it claims to describe. Fix the model. Never delete the case.
- **A trace walks a path the real code cannot take** (for example, an unobserved fact the
  collectors always observe): that is a modelling error. Fix the action's precondition, and cite the
  code that makes it impossible.
- Never weaken a property until it passes. If the code breaks section 3d, add a `finding-*` case,
  record it in `COUNTEREXAMPLES.md`, and tell the lead (F1–F3 are the current ones).

## 4. Intended behaviour change

1. Update the action in the `.tla` file, and add a constant if the old behaviour is a fixed bug,
   so it stays replayable as `bug-<PR>-*`.
2. Regenerate or hand-edit `configs/<case>.cfg`. Every config belongs to exactly one `CASES`
   entry (the probes enforce it).
3. Rerun the full suite. Then update `coverage.json`: move each changed source's `sha` and
   `sha256` (`shasum -a 256 <file>`), and its line refs. Move the pin **only after** you have
   re-read the mapped lines and confirmed the model describes them at the new commit.
4. Add the counterexample to `COUNTEREXAMPLES.md` and a red box to `STATE-DIAGRAM.md`, labelled
   with the PR.

## Coverage record

- Sources and sync: `coverage.json` (`health_verdict.py` at 7505f8b1, #1596 head; the rest at main
  12b9ddd4).
- Checker: `check.py` (`CASES` is the table), `replay.py` (real-code replay), `test_check.py` (probes).
- Bounds: 2 snapshots, 5 receipts, 6 logical hours (backups); 7 hours, one check per run (pipes);
  5 classes, unbounded (dot).
