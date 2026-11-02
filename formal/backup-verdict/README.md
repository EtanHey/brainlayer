# Backup verdict verification (TLA+)

Status: **all-fixes models pass. Every bug fixed tonight reproduces when its pre-fix switch is on,
including F1–F3 (`ingest` hysteresis), which this model found and S1a-5 (4e07651b) fixes. Main
still has F1–F3 until S1a-5 merges.**

The promise: the BrainBar dot is red exactly when memory is at risk or a pipe has stopped (plan
section 3d). The harm when it breaks: a green dot with no restorable backup, or a red dot that
never clears or clears with no evidence.

## Run

From the repository root, with Python 3.11+ and Java 17:

```sh
formal/backup-verdict/check.sh                 # every case, about 3 minutes, one TLC worker
formal/backup-verdict/check.sh --case bug-1596 # cases whose name starts with this
formal/backup-verdict/check.sh --list          # the case table
formal/backup-verdict/check.sh --src-ref <ref>     # replay counterexamples against another ref
formal/backup-verdict/check.sh --require-sync     # also fail when a covered source changed (CI runs this)
```

TLC is `~/.local/bin/tlc` (wrapper around `~/.local/share/tlaplus/tla2tools-1.7.4.jar`), or `TLC=<path>`,
or `TLC_JAR=<jar>` (runs `java -Xmx512m`). `TLC_WORKERS` defaults to 1. Before each TLC process the
checker waits while the one-minute load is above 40, for at most ten minutes.

Exit **0**: every case matched its expectation. **1**: a case did not (a fix regressed, or a pre-fix
case stopped reproducing), or `--require-sync` and a covered source changed. **2**: incomplete (no
TLC, a parse error, a crash). Output: `.build.noindex/backup-verdict/<UTC>-<pid>/`, holding per-case
`.log`, the shortest trace as `.trace.txt` (changed variables only), and `results.json`.

CI: job `TLA+ backup verdict` in `.github/workflows/ci.yml` runs the probes and every case when
this folder or a covered source changes. It downloads TLC v1.7.4 from the tlaplus release,
digest-pinned (sha256 `936a2620…0e88`, the same jar as the local one).

## What is checked

| Module | Models | Properties |
|---|---|---|
| `BackupVerdict.tla` | backup-daily receipts (snapshot, upload, removal after a verified upload, abandoned, superseded, revoked, local-only re-verify), `backup_facts()`, `_verified_age`, `evaluate()` for `backup_local` and `backup_offsite` with wake grace, monitoring gaps, unreadable-log ticks, and the weekly VACUUM gate | `NoFalseGreen`, `RedHoldsUntilEvidence`, `TruthfulSnapshotLabel`, `VacuumNeedsFreshSnapshot`, and `NoFalseGreenStrict` (a documented design limit) |
| `PipeVerdict.tla` | `evaluate()` for `drive_auth` (latch), `ingest` (heartbeat, wake grace, hysteresis), `stores` (pending queue + latched round-trip canary + hysteresis), `embedding` (anchored stall) | `RedHoldsUntilEvidence` (section 3d's "what clears red"), `RedToGreenNeedsConsecutive`, `RedClearsToGreenOnlyOnEvidence` (the docstring's weaker reading) |
| `DotTable.tla` | the dot over all 11 checks (five behaviour classes), `badge_alerts()`, and the job-alert lifecycle (`maintenance.main` → `job_alerts.report`) | `NoStuckAlert` (a job alert never lights the dot; dot red ⇔ some check red; all green ⇒ dot green), `MonitorNeverRed`, `AlertClears` (liveness: a clean run clears its alert) |

The properties in one sentence each:
- **NoFalseGreen**: a non-red backup check means a verified copy of its kind still exists within the red age, as far as the last readable receipts show. The only exception is one bounded, non-renewable wake-grace window for the same real copy aging out during a gap.
- **RedHoldsUntilEvidence**: a red check leaves red only on the evidence section 3d names (a new verified receipt; an observed Drive `ok`; vectors written; N consecutive good ticks).
- **TruthfulSnapshotLabel**: the snapshot check never claims a local copy that is gone, and its "local copy removed" note is true when shown.
- **VacuumNeedsFreshSnapshot**: VACUUM runs only with a verified local snapshot at most 6 h old, and a deferred weekly run (exit 75) never advances `last_weekly`.
- **NoStuckAlert**: a job alert can never make the dot red, and the dot cannot be red while every outcome check is green.

**Replay against real code.** Each `BackupVerdict` counterexample is turned into real receipt dicts
and run through the real `backup_facts()` and `evaluate()` (`replay.py`). The tree used is either
the named pre-fix commit or the current one. `health_verdict.py` is loaded on its own, and
`backup_daily` is replaced by the real `_snapshot_created_time` / `_parse_snapshot_date` bodies,
AST-extracted. Nothing touches a file, a database or a socket. A bug case passes only when its
counterexample reproduces at the pre-fix commit, **and** the current tree reproduces it exactly
when the case says the fix is not here yet (`fixed_here`). So a code change that brings a fixed
bug back fails the default run (exit 1). This was checked by putting `#1583 B5`'s persisted max
back into the working tree: `bug-1583-B5` failed, and passed again once restored. A case for a fix
not yet merged is `fixed_here: "no"`, and fails once the fix lands until someone flips it.

## Scope and bounds

- `BackupVerdict`: 2 snapshots, 5 receipts, 6 logical hours (all-fixes run: 1,352,435 distinct
  states). The receipt log is kept as its summaries (last local / Drive receipt kind per snapshot,
  plus the reader's running fold), so logs with equal summaries are one state.
- Thresholds are ordering-preserving logical hours, not production seconds: Grace 1 ≤ VacAge 1 <
  RedSnap 2 < RedOff 3. Backup amber ages and the canary's real count (3, model 2) are not modelled.
  Amber and green are both "not red".
- `PipeVerdict` explores one check's world per config (`Focus`). `evaluate()` couples checks only
  through `ticked_at` / `resumed_at`, which every config keeps. Pending-store amber ages are not
  modelled, so F1's store variant is checked only through S1a-5's shared rule, not by a trace.
- `DotTable` takes each class's state freely from its section 3d columns. How each class gets
  there is checked in the two modules above. `transcripts`, `integrity`, `search` and `disk` are
  covered only at this class level.

## What is not covered

- The collectors that build `Facts` from files (`health_facts.py`, #1587) are modelled only for
  #1587 B4 and B5, and are not on main. File reads, timeouts and the 4 MiB tail are out of scope.
- Only `ingest` pipe traces are replayed against code: they carry the tick's observation
  (`lastObs`). `drive_auth`, `stores`, `embedding` and dot traces are not replayed yet.
- BrainBar's Swift rendering, the installed app, and real launchd timing.

## Keeping it in step

`coverage.json` pins each covered source by sha256, maps each modelled action to path:line, and
records bounds and atomicity. Every source is pinned at main 67d23f14. CI runs `--require-sync`,
so any change to a covered source fails this job until someone re-confirms the model and moves the
pin. When covered code changes:
re-read the mapped lines, update the model or add a switch, rerun, and only then move the pin.
See [SKILL.md](SKILL.md).

[COUNTEREXAMPLES.md](COUNTEREXAMPLES.md) tells every trace in the code's words.
[STATE-DIAGRAM.md](STATE-DIAGRAM.md) draws them, with tonight's bugs in red.
[RESULTS.md](RESULTS.md) is the measured run.
