# Explicit launchd consolidation

Install the release containing migration, revert and installer wiring before activation.
Run the installed `brainlayer/launchd/install.sh consolidate` to hand legacy jobs to
BrainLayer Daemon and BrainLayer Maintenance. The installer checks release signatures,
Spotlight layout, the installed interpreter, rendered plists and helper syntax before handoff.
Same-keg replay leaves writers running; a changed keg restarts the supervisor through the
checked transaction. Differing shared helpers require explicit operator reconciliation.

Use `install.sh consolidation-rollback` to restore the journaled legacy definitions,
markers and runtime states through the checked revert API. The migration journal remains.
Legacy targets (`all`, individual jobs, `load`, `unload`, `remove`, watchdog commands)
refuse supervisor-owned or unknown journal phases and share the migration lock.
They are available after a verified `reverted` or `rolled_back` journal state.

Activation requires source review, merged prerequisites and real serving/sleep-wake proof
on both Macs. Scratch tests do not establish installed or native acceptance.
