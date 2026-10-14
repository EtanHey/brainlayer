# Explicit launchd consolidation

Install the release containing migration, revert and wiring before activation.
Run installed `brainlayer/launchd/install.sh consolidate` for BrainLayer Daemon and Maintenance. Signatures, Spotlight layout, interpreter, rendered plists and helper syntax are checked first.
Same-keg replay leaves writers running; a changed keg restarts the supervisor. Differing shared helpers require explicit operator reconciliation.
Use `install.sh consolidation-rollback` to restore journaled legacy definitions, markers and runtime states; the journal remains. Interrupts print recovery guidance and return nonzero.
Legacy targets (`all`, individual jobs, `load`, `unload`, `remove`, watchdog commands) refuse supervisor-owned or unknown phases and share installer/migration custody. They reopen after `reverted` or `rolled_back`.
Activation requires review, merged prerequisites and real serving/sleep-wake proof on both Macs. Scratch tests do not establish installed or native acceptance.

## 1.5.57 upgrade order on consolidated Macs

1. Explicitly upgrade the formula first with `brew upgrade --formula etanhey/layers/brainlayer`, then run the installed keg's signature and import gates. The cask's `depends_on formula: "brainlayer"` can upgrade the formula during a cask upgrade; do not combine these steps or let that bypass consolidation. Stop on any gate failure.
2. Run the installed `brainlayer/launchd/install.sh consolidate`. A keg change re-pins the daemon interpreter and restarts the supervisor onto the new keg.
3. Verify the new supervisor: the lifecycle health scan must have `scan_error: null`, and `brainlayer jobs component watch status` must report `source: supervisor`. Confirm the receipt names the new keg, resolving a relative `keg` against the parent of `opt/brainlayer`.
4. Upgrade the cask only after those gates pass; its postflight runs afterward.

`brainlayer jobs restart --all --verify` skips `com.brainlayer.daemon` and `com.brainlayer.maintenance` while the migration journal phase is `migrating`, `migrated` or `reverting`, with reason `supervisor-owned: keg changes go through install.sh consolidate`. It does not restart or verify those labels. A skip receipt does not prove supervisor health or a successful keg upgrade; consolidation and the preceding gates own that proof.
With no journal, or `reverted` / `rolled_back`, legacy restart behavior remains. An unreadable journal or unknown phase refuses restart before commands. Other jobs retain their restart and verification checks.
