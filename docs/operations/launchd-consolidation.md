# Explicit launchd consolidation

Install the release containing migration, revert and wiring before activation.
Run installed `brainlayer/launchd/install.sh consolidate` for BrainLayer Daemon and Maintenance. Signatures, Spotlight layout, interpreter, rendered plists and helper syntax are checked first.
Same-keg replay leaves writers running; a changed keg restarts the supervisor. Differing shared helpers require explicit operator reconciliation.
Use `install.sh consolidation-rollback` to restore journaled legacy definitions, markers and runtime states; the journal remains. Interrupts print recovery guidance and return nonzero.
Legacy targets (`all`, individual jobs, `load`, `unload`, `remove`, watchdog commands) refuse supervisor-owned or unknown phases and share installer/migration custody. They reopen after `reverted` or `rolled_back`.
Activation requires review, merged prerequisites and real serving/sleep-wake proof on both Macs. Scratch tests do not establish installed or native acceptance.
