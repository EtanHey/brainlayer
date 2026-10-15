# Explicit launchd consolidation

Install the release containing migration, revert and wiring before activation.
Run installed `brainlayer/launchd/install.sh consolidate` for BrainLayer Daemon and Maintenance. Signatures, Spotlight layout, interpreter, rendered plists and helper syntax are checked first.
Same-keg replay leaves writers running; a changed keg restarts the supervisor. Differing shared helpers require explicit operator reconciliation.
Use `install.sh consolidation-rollback` to restore journaled legacy definitions, markers and runtime states; the journal remains. Interrupts print recovery guidance and return nonzero.
Legacy targets (`all`, individual jobs, `load`, `unload`, `remove`, watchdog commands) refuse supervisor-owned or unknown phases and share installer/migration custody. They reopen after `reverted` or `rolled_back`.
Activation requires review, merged prerequisites and real serving/sleep-wake proof on both Macs. Scratch tests do not establish installed or native acceptance.

## Upgrade order on consolidated Macs

1. Before the formula can move `opt/brainlayer`, confirm the maintenance calendar is idle:

   ```bash
   launchctl print "gui/$(id -u)/com.brainlayer.maintenance"
   ```

   Require a successful print of the loaded job with no `pid = ...` line. Any PID means the calendar is active: wait for it to finish and check again; do not interrupt its backup upload. A failed print is unknown, not idle: stop and reconcile it. Recheck immediately before the formula upgrade. This snapshot does not reserve the maintenance window; if a calendar run can start before consolidation finishes, defer the upgrade.
2. Explicitly upgrade the formula with `brew upgrade --formula etanhey/layers/brainlayer`. The cask's `depends_on formula: "brainlayer"` can otherwise bundle this upgrade; do not let it bypass the calendar check or consolidation.
3. Immediately run the installed `brainlayer/launchd/install.sh consolidate`. Its signature/interpreter preflight must pass before any restart. A keg change re-pins the daemon interpreter and restarts the supervisor onto the new keg. Do not start another maintenance task in between. If consolidation refuses because the calendar became active, stop the deploy; do not force a restart or continue to the cask.

   With #1316, the post-flip/pre-consolidate window intentionally shows critical `job_scan_failed` ("keg mapping unavailable"), with no restarts; consolidate clears the ambiguity, and kickstarting the daemon is not a remedy.

   If the daemon's `runs` count climbs with exit 0 after the flip (inspect `launchctl print "gui/$(id -u)/com.brainlayer.daemon"`), stop the deploy and repoint opt to the prior installed Cellar keg that matches the receipt. Replace `<prior>` with that version:

   ```bash
   ln -sfn "../Cellar/brainlayer/<prior>" /opt/homebrew/opt/brainlayer
   ```

   Never kickstart the daemon. Leave the calendar/upload untouched and reconcile the failed deploy before retrying.
4. Verify actual installed keg signatures/imports and the new supervisor: the lifecycle health scan must have `scan_error: null`, and `brainlayer jobs component watch status` must report `source: supervisor`. Confirm the receipt names the new keg, resolving a relative `keg` against the parent of `opt/brainlayer`. Stop on any gate failure.
5. **Required harness gate, after installed verification and before the cask.** From a trusted
   BrainLayer worktree at the release commit, with the operator's normal login environment, run:

   ```bash
   /opt/homebrew/opt/brainlayer/libexec/venv/bin/python scripts/harness_probe.py --output "$HOME/harness-probe-$(date -u +%Y%m%dT%H%M%SZ)"
   ```

   Require exit 0 and all seven rows PASS. Any FAIL, missing harness, missing app/T3 adapter,
   partial selection, or missing ingest proof blocks the cask and the convergence claim. Preserve
   the private `results.json` and `table.md` in the release receipt; never publish raw harness logs.
   Run first on the MBP, then over `ssh m1` in that user's normal login environment. This runs fresh
   sessions with real user config and tiny tagged stores; it must not change services or config.
   See [the receipt template](release-receipt-template.md) for proof limits and required fields.
6. Upgrade the cask only after verification and the harness gate pass; its postflight runs afterward. For the formula-only 1.5.58 release, the cask remains 1.5.57.

`brainlayer jobs restart --all --verify` skips `com.brainlayer.daemon` and `com.brainlayer.maintenance` while the migration journal phase is `migrating`, `migrated` or `reverting`, with reason `supervisor-owned: keg changes go through install.sh consolidate`. It does not restart or verify those labels. A skip receipt does not prove supervisor health or a successful keg upgrade; consolidation and the preceding gates own that proof.
With no journal, or `reverted` / `rolled_back`, legacy restart behavior remains. An unreadable journal or unknown phase refuses restart before commands. Other jobs retain their restart and verification checks.
