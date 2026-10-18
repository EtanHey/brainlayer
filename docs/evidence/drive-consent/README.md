# Drive consent reminder — implementation evidence

Status: source implementation evidence; pending independent review and installed verification.
Implementer: Codex, gpt-6.1-sol (DRIVE lane).

The seven-day consent gate remains unchanged. The backend now reports `expiring`
at 48 hours remaining, including offline status. BrainBar presents the deadline
at 49 hours to allow for its 30-minute refresh interval and show the reminder at
least 48 hours before expiry while running. Expired consent remains rejected.
Backup warning/error text includes the exact consent expiry timestamp.

## Before and after

All four PNGs are screenshots of visible, running DEBUG BrainBar windows with
**synthetic consent and recovery data**, captured using macOS `screencapture`.
The harness exits before `BrainBarApp.main()`: it opens no production database,
installs no credential runner, and performs no consent/account action.
These are source UI evidence, **not installed/live-health proof on either Mac**.
The fixture expiry is 36 hours after each capture's clock, so the displayed
minutes differ between before and after.

| Page | Before | After |
|---|---|---|
| Dashboard | [No Drive reminder](dashboard-before.png) | [Yellow reconnect deadline and button](dashboard-after.png) |
| Backups | [Green connected, no reminder](backups-before.png) | [Yellow reconnect deadline and button](backups-after.png) |

The Backups images show the actual Backups body. An earlier mislabeled capture
showed Dashboard and is rejected; it is not included here. The Backups fixture's
6/6 check badge and stale example dates are synthetic. Dashboard's other
attention items and missing observability data are also unrelated fixture
state, not observations about either installed Mac.

## Nonvisual behavior

Synthetic probe uses expiry `2026-10-24T18:50:00+00:00`. Both versions expose
`expires_at`; the change makes the warning early enough and names it in logs.

| Case | Base | Implementation |
|---|---|---|
| 48/36 hours remaining | `valid`, allowed, no warning | `expiring`, allowed, exact expiry warning |
| Day seven | `invalid`, rejected, error has no timestamp | `invalid`, rejected, error names exact expiry |
| DB-backup failure recorder | Generic consent-expired error | Error names `2026-10-24T18:50:00+00:00` |
| Token bytes after probes | Unchanged | Unchanged |

The backup failure-recorder probe stubs artifact creation and credentials;
no database is opened and no upload or retention operation runs.

## Verification

- Python base-module RED: 6 failed, 14 passed.
- Retained original Swift RED: 11 tests, 16 failed assertions in 2 tests.
- Poll-margin Swift RED: 11 tests, 3 failed assertions in 1 test.
- Python GREEN: 239 passed across Drive credentials, DB backup, JSONL backup.
- Swift GREEN: 25 passed across Drive auth, backup checks, and backups health.

Commands, logs, exact frozen SHA, hashes and the surface ledger are recorded in
`lane-DRIVE.md` under the lead's assigned local stability report directory.
Independent review, PR creation, release, installation and per-Mac acceptance
remain with the parent stability lead.
