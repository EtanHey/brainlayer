# Supervisor policy reconcile (S24)

Run `TLC_JAR=/path/to/tla2tools.jar sh formal/policy-reconcile/check.sh`.
Logs and counterexamples stay under `.build.noindex/tlc/policy-reconcile/`.

`Intent`/`IntentSync` model the restart journal's durable old/new bytes and SHA;
`Stage`/`Rename`/`ManifestSync` model `archive_replace(jobs.json)` after supervisor
shutdown. `Complete` follows bootstrap, readiness and marker publication.
Rollback restores/fsyncs the old manifest before bootstrapping the fallback.
If completion published a new receipt before failing, rollback first restores
durable pending intent so both manifest versions remain recoverable.
Recovery accepts only journaled versions; unrelated bytes refuse. One crash may
occur at any step, including unsynced directory entries. The bug config omits
durable intent and reproduces an unowned new manifest at `Rename`.
Recovery of matching old/new state explicitly starts that version and waits
for readiness. Normal restart and rollback also observe readiness before
completion. `DoneHasReadyService` requires a running, ready service for the
selected manifest at `done`. `bug-recovery-skips-readiness.cfg` starts recovery
but skips its readiness check, reproducing a violation of that invariant.

The abstraction assumes the existing migration lock, archive, launcher, marker,
and runtime custody checks; it does not prove those implementations. The fake-home
SIGKILL test exercises real journal/manifest fsync and rename boundaries.

Launcher plist policy reconciliation is outside this model's state space: the
modeled durable value is the supervisor component manifest, while launchers and
their loaded runtime are explicit custody assumptions. Modeling one or more
plist publications plus per-label bootout/bootstrap requires separate launcher
state and invariants (including maintenance-lock admission), so the S24 model is
not extended to claim coverage it does not provide. The launcher transition's
old/new hashes and crash recovery are exercised by fake-home tests instead.
