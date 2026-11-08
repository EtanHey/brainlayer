# Counterexamples

Each entry below is TLC's shortest path to a broken rule. It is told in the code's words, with the
model's logical hours (RedSnap 2 h stands for 54 h, RedOff 3 h for 78 h, Grace 1 h for 2 h). Every
case re-runs with `formal/backup-verdict/check.sh --case <name>`, which writes the full trace to
`.build.noindex/backup-verdict/<run>/<name>.trace.txt`.

**Replay.** Every `BackupVerdict` counterexample also runs, receipt by receipt and tick by tick,
through the real `backup_facts()` and `evaluate()` of the named pre-fix commit, then of the
current tree (`replay.py`). "Reproduces at the pre-fix ref" means the real code reaches the same
false state, and every tick's backup states equal the model's. Counterexamples on the three stall
checks (`ingest`, `embedding`, `stores`) replay the same way through the real `evaluate()`, fact by
fact. That covers #1583 B2, S1a-5 F1–F3 and the renewable-grace bug. A replayed case passes only
when the pre-fix ref reproduces it **and** agrees on every tick. The current tree must reproduce it
exactly while the fix is not there (`fixed_here: "no"`). A missing trace, a trace without a tick,
or a failed replay is an error, never a pass (REV-FTLA1-R1 B2). The `drive_auth` and dot
counterexamples are not replayed (see README, "What is not covered").

**Conformance.** Beyond the counterexamples, `conform-*` cases replay 3,000 random TLC behaviours per
check (`-simulate`, one worker, depth 60, seed 7) through real code. These are the counts per check:

| Model | Code | ingest (8,934 ticks) | embedding (17,955) | stores (11,861) |
|---|---|---|---|---|
| main (renewable grace, no S1a-5) | 67d23f14 | 0 diverging | 0 diverging | 0 diverging |
| S1a-5 (renewable grace) | 4e07651b | 0 diverging | — | — |
| S1a-5 + S1a-6 (one window per episode) | **8eecb5ee** | **0 diverging** | **0 diverging** | **0 diverging** |
| S1a-5 + S1a-6 | **working tree, main 62f55b09** (both merged) | **0 diverging** | **0 diverging** | **0 diverging** |
| S1a-5 + S1a-6 | 67d23f14 (before both) | 218 of 3,000 | 160 of 3,000 | 212 of 3,000 |

Zero trace files, zero parsed ticks, or a file that parses to nothing is an error.

## Tonight's bugs: each pre-fix switch gives a counterexample

| Case | PR | Rule broken | Replay: pre-fix ref → current tree |
|---|---|---|---|
| `bug-1583-B1-canary-latch` | #1583 R1 B1 | RedHoldsUntilEvidence | (pipe model) |
| `bug-1583-B2-resume-grace` | #1583 R1 B2 | RedClearsToGreenOnlyOnEvidence | 672c9f97 reproduces → fixed |
| `bug-1583-B3-renewable-grace` | #1583 R1 B3 | NoFalseGreen | 672c9f97 reproduces → fixed |
| `bug-1583-B4-drive-latch` | #1583 R1 B4 | RedHoldsUntilEvidence | (pipe model) |
| `bug-1583-B5-persisted-max` | #1583 R1 B5 | NoFalseGreen | 672c9f97 reproduces → fixed |
| `bug-1589-B5a-grace-without-clock` | #1589 B5a | NoFalseGreen | 7f7f6245 reproduces → fixed |
| `bug-1589-B5b-shared-revocation` | #1589 B5b | NoFalseGreen | 7f7f6245 reproduces → fixed |
| `bug-1589-B5c-legacy-identity` | #1589 B5c | NoFalseGreen | 7f7f6245 reproduces → fixed |
| `bug-1595-local-removed` | #1595 | TruthfulSnapshotLabel | d9d7bff1 reproduces → fixed |
| `bug-1595-recovery-reset` | #1595 | RedHoldsUntilEvidence | (pipe model) |
| `bug-1596-abandoned` | #1596 | NoFalseGreen | 12b9ddd4 reproduces → fixed (main 67d23f14) |
| `bug-1596-superseded` | #1596 | NoFalseGreen | 12b9ddd4 reproduces → fixed (main 67d23f14) |
| `bug-1685-lenient-weekly-gate` | #1685 | VacuumNeedsFreshSnapshot | ff3a8b35 reproduces → fixed (main 5fda86d2) |
| `bug-1587-B4-backlog-ages-out` | #1587 B4 | RedHoldsUntilEvidence | (pipe model) |
| `bug-1587-B5-stale-canary-success` | #1587 B5 | RedHoldsUntilEvidence | (pipe model) |
| `bug-1583-S1-alert-lights-dot` | #1583 S1 | NoStuckAlert | (dot model) |
| `bug-renewable-grace` | S1a-6 (8eecb5ee) | NoFalseGreenStall | 67d23f14 reproduces → fixed (main 62f55b09, #1602) |
| `bug-renewable-grace-ingest` | S1a-6 (8eecb5ee) | NoFalseGreenStall | 67d23f14 reproduces → fixed (main 62f55b09, #1602) |
| `bug-renewable-grace-stores` | S1a-6 (8eecb5ee) | NoFalseGreenStall | 67d23f14 reproduces → fixed (main 62f55b09, #1602) |
| `bug-s1a5-F1-red-to-amber` | S1a-5 (4e07651b) | RedHoldsUntilEvidence | 67d23f14 reproduces → fixed (main 62f55b09, #1600) |
| `bug-s1a5-F2-unobserved-tick-keeps-count` | S1a-5 (4e07651b) | RedToGreenNeedsConsecutive | 67d23f14 reproduces → fixed (main 62f55b09, #1600) |
| `bug-s1a5-F3-graced-stale-keeps-count` | S1a-5 (4e07651b) | RedClearsToGreenOnlyOnEvidence | 67d23f14 reproduces → fixed (main 62f55b09, #1600) |

### #1583 R1 B1: an empty queue repays a failed store canary
1. The store path breaks, and the nightly round trip writes a failed receipt. The tick reads it: canary failed once, `stores` amber.
2. The next night fails again. Two failures (the model's red count), `stores` red.
3. The next tick has no canary run and sees an empty pending queue. Pre-fix, the unrun canary is left out, so the empty queue alone is GREEN and counts as good tick 1 of 2.
4. The tick after that does the same. Good tick 2 of 2, so `stores` turns **green**. No store has ever succeeded since the failures.

Fix (#1583): the failure latches until a canary succeeds (`_counter(latch=True)`), and an outstanding failure is only repaid by a successful round trip (`canary_owed`).

### #1583 R1 B2: a tick with no heartbeat counts as a good tick
1. The watcher stops at hour 0. At hour 3 its heartbeat is 3 h old, and `ingest` is red.
2. The watcher writes one heartbeat at hour 3, and the hour-4 tick sees it fresh. Good tick 1 of 2.
3. The hour-5 tick observes no heartbeat. The persisted heartbeat is 1 h old, so the raw state is GREEN, and pre-fix that counts as good tick 2 of 2. `ingest` turns **green** with one real observation.

The reviewer's B2 repro is the same rule seen through a wake. After a gap, `resumed_at` makes an unobserved or stale clock look fresh, and that tick counted as good. Fix (#1583): a tick without a new observation, or whose age looks fresh only because of the resume, is `evidence=False`. It holds a known amber or red and never counts as good.

### #1583 R1 B3: every monitoring gap renews backup grace
1. Snapshot 1 is verified at hour 0. The hour-1 tick sees it: `backup_local` green.
2. The Mac sleeps. At the hour-3 tick the copy is 3 h old, past red (2 h). It is a resume, so grace makes it amber.
3. The Mac sleeps again. At the hour-5 tick (another resume) grace is granted again: still **amber**, 5 h old, and 2 h past the first catch-up deadline (hour 4). Nothing bounds this.

Replay at 672c9f97: the real code gives the same states on every tick. Fix (#1583 final, kept by #1589): one persisted catch-up deadline per episode, never moved by a later gap.

### #1583 R1 B4: a failed refresh clears confirmed Drive loss
1. Drive access is lost. The tick sees `consent_needed`: `drive_auth` red.
2. The next tick sees `refresh_failed`. Pre-fix that is amber, so `drive_auth` steps **red → amber** and the badge goes out. Access is still lost.

Fix (#1583): confirmed loss latches (`lost`) until an observed `ok`.

### #1583 R1 B5: the persisted max keeps a revoked copy
1. Snapshot 1 is locally verified at hour 0. The tick sees it: `backup_local` green.
2. A later receipt revokes snapshot 1 (`verification_error`).
3. At the hour-1 tick, `backup_facts()` correctly reports no verified copy. But `_verified_age` takes `max(observed, persisted)`, so the persisted clock (snapshot 1) survives: `backup_local` stays **green** with nothing verified on record.

Replay at 672c9f97: reproduces, tick for tick. Fix (#1589): readable receipts own the clock. The persisted clock is only an unreadable-source fallback.

### #1589 B5a: wake grace hides the revocation of the only copy
1. Snapshot 1 is verified, and the tick sees green. A receipt then revokes it.
2. The Mac sleeps for 2 h. At the next tick (a resume) there is no verified copy at all, but the #1583-final grace graced any newly red backup check: `backup_local` **amber**, deadline 3 h, no clock.

Replay at 7f7f6245: reproduces. Fix (#1589): grace only for the *same* verified clock that crossed red during the gap (`same_copy`, `ticked <= expired_at < resumed_at`).

### #1589 B5b: a local-only re-verify restores a revoked Drive copy
1. Snapshot 1 is uploaded and verified. The tick sees both checks green.
2. A Drive md5 mismatch revokes it. Then a local-only success re-verifies the local file.
3. At the next tick the reader correctly reports no Drive copy. But the local-only success removed snapshot 1 from the shared revocation set, so the persisted Drive clock is not recognised as revoked: `backup_offsite` stays **green**.

Replay at 7f7f6245: reproduces. Fix (#1589): receipts are authoritative, so no persisted clock survives a readable tick.

### #1589 B5c: a filename-only revocation misses legacy persisted state
1. Snapshot 1 is locally verified, and the tick sees green. The persisted state has no `clock_id` (written before it existed).
2. A receipt revokes snapshot 1 but carries no `snapshot_created_at`. `_snapshot_created_time` falls back to the filename's midnight, which does not equal the exact persisted time.
3. At the next tick, `_is_revoked` cannot match by time: `backup_local` stays **green**.

Replay at 7f7f6245 (with `clock_id` stripped from the persisted state each tick): reproduces. Fix (#1589): receipts are authoritative, and exact creation times survive from history.

### #1595: a removed local copy still reads as a local backup
1. Snapshot 1 is uploaded and verified. `backup_daily` then deletes the local archive (`local_removed`).
2. The tick shows `backup_local`, labelled "Local database backup", **green**, with snapshot 1 as its clock. No local copy exists.

Replay at d9d7bff1: reproduces (label and note unchanged). Fix (#1595): the check is "Database snapshot (verified)" and counts a Drive-verified copy, and the note says "local copy removed after Drive verification".

### #1595: an unknown tick keeps the recovery count
1. The pending-store queue stalls: `stores` red.
2. A good tick (canary ok): recovering, 1 of 2.
3. A tick that observes nothing: raw UNKNOWN. Pre-fix it held red without resetting `good_ticks`.
4. Another good tick makes 2 of 2: **green**, without two consecutive good ticks.

Fix (#1595): `if r.state != GREEN: good_ticks = 0` before every branch. The Macroscope shape (a failed canary held red mid-recovery) is the same line.

### #1596: an abandoned snapshot still counts
1. Snapshot 1 is locally verified at hour 0 and never uploaded.
2. Its upload finds the local file missing: `upload_state: abandoned, local_verified: false`.
3. The tick shows `backup_local` **green** on snapshot 1. No verified copy exists anywhere.

Replay at 12b9ddd4 (main before #1596): reproduces. At main 67d23f14: does not.

### #1596: a deleted superseded stall comes back as the clock
1. Snapshot 1 is a local-only stall. Snapshot 2 is taken at hour 1, and is uploaded and verified.
2. That verified upload lets `_reclaim_superseded_local` delete snapshot 1
   (`upload_state: superseded`); it runs only after a newer verified upload
   (`backup_daily.py:3209-3211`). Snapshot 2's Drive copy is then revoked (md5 mismatch).
3. The tick shows `backup_local` **green** on snapshot 1, which was deleted.

Replay at 12b9ddd4: reproduces, tick for tick. At main: does not. (REV-FTLA1-R1 noted that R2's
version superseded without a newer upload, which the producer cannot emit. `Supersede` now requires
the newer verified upload, and the trace above is producer-reachable.)

### #1685: a damaged revoke line lets the weekly gate VACUUM on a revoked snapshot
1. Snapshot 1 is locally verified at hour 0 (`local_verified`, `local_integrity: ok`, `local_artifact`).
2. Its verification is revoked (`drive_md5_match: false`), but that receipt's line in
   `backup-daily.log` is damaged: newline-terminated, not JSON, so it is neither progress noise
   nor a torn final record.
3. Weekly maintenance `--full` asks `_recent_local_verified_backup` for a fresh verified snapshot.
   The lenient read skips the damaged line, so snapshot 1's latest readable receipt is the
   verified one. The gate admits it, and VACUUM runs with no verified fresh snapshot.

Replay (`replay.py --run-weekly`): real snapshot files and a real log under `.build.noindex/`,
read by the real gate extracted from each ref's `backup_daily.py`. At ff3a8b35 (main before
#1685) the gate returns snapshot 1: reproduces. At main 5fda86d2 `_read_backup_receipts(strict=True)`
raises `BackupReceiptUnreadableError`, the gate returns `None`, and the run defers with
`last_weekly` kept. That is fail-closed: VACUUM then stays deferred for as long as the damaged line
stays in the log. `_append_json_log` truncates a torn tail before it appends, so the writer itself
does not leave a mid-file damaged line. The case covers damage from anything else.

### #1587 B4: an embedding stall clears when its rows age out
1. The embed backlog stalls at hour 0. At hour 3 it has not shrunk for 3 h: `embedding` red.
2. With the 24 h window not anchored, the stalled rows age out of the query and the backlog reads as empty. The next tick shows **green**, and no vector was written.

Fix (#1587): once a backlog appears, the window's lower bound is anchored until the backlog reaches 0.

### #1587 B5: a failed round trip leaves the older success readable
1. A store round trip succeeds and writes its receipt. The store path then breaks, and the next attempt fails, but its receipt write also fails, so the old success stays.
2. A pending-store stall turns `stores` red. The queue then clears.
3. Two ticks read the stale "ok" receipt: recovering, then **green**. The latest attempt failed.

Fix (#1587): every attempt replaces the receipt. If it cannot be written, the old one is removed and the canary reads unobserved.

### #1583 S1: a job alert lights the badge with every check green
1. `maintenance-full` fails: `job-alerts.json` gains its key.
2. Every outcome check is green, yet the pre-#1583 badge treated the job alert as critical, so the badge is **red**. It stays red until that exact job succeeds again ("red until that exact job succeeds again").

Fix (#1583; plan section 3d): job alerts are `job_alert_*` warning notes, and only `outcome_<check>` for a red check is critical.

## Found by this model, fixed by S1a-5 (#1600, merged as 2f983832)

TLC found these three with every earlier fix on. S1a-5 changes `evaluate()` so that `good_ticks`
resets when `r.state != GREEN or not r.evidence or r.graced`, and a hysteresis check that is red
and observes amber stays red. The model mirrors this behind `ObservedHysteresis`. With it on, the
three `finding-ingest-*` cases now **pass** (21,840 states). With it off, the `bug-s1a5-*` cases
reproduce each trace below, and the real code at main 67d23f14 reproduces each one tick for tick.

### F1 `bug-s1a5-F1-red-to-amber`: one aging heartbeat steps ingest from red to amber
1. The watcher stops: at hour 3, `ingest` is red (heartbeat 3 h old, red after 2 h).
2. The watcher writes once, then stops again. The next tick observes that heartbeat 2 h old (amber age).
3. Raw AMBER, observed, not graced: the hysteresis branch only guards the step to GREEN, so `ingest` goes **red → amber** on one observation. The badge goes out.

Section 3d says red clears on "2 consecutive good ticks, with an advancing heartbeat". The pending-store age (15–45 min amber) has the same shape in code, but the model does not include amber store ages. Code: `health_verdict.py` (7505f8b1) `evaluate`, the `_HYSTERESIS and state == GREEN` branch.

### F2 `bug-s1a5-F2-unobserved-tick-keeps-count`: an unobserved tick does not reset recovery
1. `ingest` is red. One fresh heartbeat is observed: good tick 1 of 2.
2. The next tick observes no heartbeat. The persisted clock is fresh, so raw is GREEN with `evidence=False`, the state is held red, and `good_ticks` is **not** reset (#1595 resets only on a non-green raw).
3. The next fresh observation makes 2 of 2: **green**, with a blind tick between the two good ones.

The old module docstring allowed this ("no bad one between"); section 3d's "consecutive" did not. The lead ruled for section 3d, and S1a-5 rewrote the docstring.

### F3 `bug-s1a5-F3-graced-stale-keeps-count`: a stale heartbeat seen inside wake grace does not reset recovery
1. `ingest` is red. One fresh heartbeat is observed: good tick 1 of 2. The watcher then dies.
2. The Mac sleeps. The next tick **observes** the dead watcher's 3 h-old heartbeat, but wake grace makes the effective age 0, so raw is GREEN (graced), held red, and `good_ticks` stays 1.
3. The watcher restarts. One fresh tick makes 2 of 2: **green**.

This broke the code's own rule as well: an observed bad heartbeat sat between the two good ones. S1a-5's `or r.graced` is the fix.

## Found by review, fixed by S1a-6 (#1602, merged as 62f55b09): renewable wake grace

REV-FTLA1-R1 B1 found that the model computed embedding (and pending stores) from the raw clock,
while the real code routes them through `_stall_age` → `_graced_age`. That path softens the age
after a monitoring gap, and every later gap renews it. The plan's section 3d wake-grace ruling sets
one window per stall episode, keyed by the clock value and never renewed. All three stall checks now
take the real path in the model (`GracedAge`, evidence suppressed while graced). `GraceOnce` mirrors
S1a-6's persisted `grace_clock` / `grace_started_at` / `grace_until`. New invariant
**NoFalseGreenStall**: a stall whose true age is past red is non-red only inside one wake window,
the window of the first resume seen while it was over red.

### `bug-renewable-grace`: the reviewer's trace, green ×5 on a stopped embedder
Scripted schedule (`TickAt = {0, 1, 3, 5, 7}`, every tick observes the backlog):
1. Hour 0: the tick sees a clear backlog (green). The backlog stalls at hour 0, and no vector is written after that.
2. Hour 1: stall age 1 h, green.
3. Hour 3: true age 3 h, past red (2 h). It is a resume, so grace counts from 3: **green**. The window runs to 6.
4. Hour 5: another gap, and today's code renews `resumed_at` to 5: **green**.
5. Hour 7: past the first window, but the next gap renews it again: **green**. NoFalseGreenStall breaks.

Real code at 67d23f14 gives green at all five ticks (tick for tick). At 8eecb5ee the hour-7 tick is
red. `bug-renewable-grace-ingest` and `-stores` are the same bug on a dead watcher and on a stuck
pending store (shortest TLC paths); both reproduce at 67d23f14 and not at 8eecb5ee.

## Found by this model on main 5fda86d2, not fixed here

### `finding-damaged-receipt-dot`: a damaged abandon line leaves the dot green
1. Snapshot 1 is locally verified at hour 0.
2. Its upload finds the local file missing (`upload_state: abandoned`), but that receipt's line is damaged.
3. The tick reads the log through `health_facts.read_receipts`, which skips any line it cannot
   parse (`health_facts.py:120-126`). `backup_local` is **green** on snapshot 1, which no longer exists.

This is the same uncertainty #1685 closed for the VACUUM gate. The dot still reads leniently.
Failing closed there would hold the check `unknown` (amber after three ticks) while the damaged line
stays in the log. Whether that trade is wanted is a product call, so the case expects
`NoFalseGreen` to fail, as a ratchet: it turns into a mismatch the day the dot's read changes. Not
replayed: the real reader is the file parser in `health_facts.py`, which `replay.py` does not load.

## Design limit and scratch breaks

| Case | Rule | Meaning |
|---|---|---|
| `limit-unreadable-log` | NoFalseGreenStrict | While `backup-daily.log` is unreadable, a revocation appended after the last readable tick is invisible. The check keeps the last known copy until that copy ages past red. This is the documented unreadable-fallback design, bounded by the red age, not a bug. |
| `scratch-vacuum-any-receipt` | VacuumNeedsFreshSnapshot | A gate that trusts any old `local_ok` receipt runs VACUUM on a snapshot revoked since. Shows the property can fail. |
| `scratch-deferred-advances` | VacuumNeedsFreshSnapshot | A deferred weekly run that advanced `last_weekly` would skip the week. Shows the property can fail. |
| `scratch-pause-downgrades` | RedHoldsUntilEvidence | #1686 reads AMBER only on recent progress; a load pause with no batch only adds a note. If the pause also read AMBER (`Daemon = "pause"`), an embedding stall past red would step red → amber with no vector written. Shows the property guards #1686's choice. |
| `scratch-clean-run-keeps-alert` | AlertClears | If a clean run did not clear its alert, the alert would never clear. Shows the liveness property can fail. |
