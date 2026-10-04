# Maintenance job alert: once per screen, Show log, cleared by a clean run

Etan saw "BrainLayer light maintenance failed; check the maintenance log" four times (the 2026-10-02
04:00 nightly light pass). These are DEBUG render-harness captures (`renderMaintenanceAlert`) from
fixtures: `BrainBarDashboardFixture.maintenanceAlertObservabilityResult` (otherwise fresh, verified
backups) and, for "cleared", the same document after a clean run. No real DB, socket or log is read.

```sh
DEVELOPER_DIR=/Applications/Devtools/Xcode.app/Contents/Developer swift build --product BrainBar
BRAINBAR_DB_PATH=<scratch>/brainlayer.db BRAINBAR_RENDER_ONLY=<scratch>/out .build/debug/BrainBar
```

| Surface | Before | After (alert) | After a clean run |
|---|---|---|---|
| Backups page | [twice: badge reason + red row](before-alert-backups-default-dark.png) | once, as an alert card with **Show log**; the badge keeps "Needs attention" without repeating it: [760](after-alert-backups-compact-dark.png) · [960](after-alert-backups-default-dark.png) · [1280](after-alert-backups-wide-dark.png) | [no alert, Healthy](after-cleared-backups-default-dark.png) |
| Dashboard ("1 thing needs you") | [once, no action](before-alert-dashboard-expanded-default-dark.png) | once, with **Show log** on the item: [760](after-alert-dashboard-expanded-compact-dark.png) · [960](after-alert-dashboard-expanded-default-dark.png) · [1280](after-alert-dashboard-expanded-wide-dark.png) | [no attention strip](after-cleared-dashboard-default-dark.png) |
| Menu-bar menu | [status line only](before-alert-menu-dark.png) | status line + **Show log** ([dark](after-alert-menu-dark.png), [light](after-alert-menu-light.png)) | [Show log hidden](after-cleared-menu-dark.png) |

**The menu** cannot be captured off-screen. Its renders draw the controller's real row titles
(`BrainBarStatusPopoverController.menuRowTitles`), and the menu tests check the live `NSMenu`.

**Light and dark:** the harness renders every Dashboard and Backups capture under both the
`darkAqua` and `aqua` system appearances. They are byte-identical, because BrainBar pins its own
dark scheme (`.environment(\.colorScheme, .dark)`), so only the dark set is kept here. Only the menu
follows the system appearance.

**The alert's words** come from the job (`job-alerts.json` → `observability.json`
`backups.error_type = "job_alert:<reason>"`). BrainBar renders whatever reason it carries and
hardcodes none. Backend PR A changes the reason to say what failed and what to do.

**Show log** opens the failing job's log in its default app (Console): `maintenance-*` →
`maintenance.log`, `backup-daily` → `backup-daily.log`, `jsonl-backup` → `jsonl-backup.log`, each
path as the job's own LaunchAgent environment resolves it. When the log does not exist yet, or the
job is unknown, it reveals the logs folder instead.

**Clearing:** a clean run removes the job's key from `job-alerts.json` (`job_alerts.report(key, None)`).
BrainBar now reconciles `observability.json` with that file on every read
(`ObservabilityReader.readReconciled`). A cleared alert disappears right away, not at the next
observability write. When that file is missing or unreadable, the document is shown as read
(unknown, never "no alerts").

## Addendum (lead, from the #1061 review)

**Show log opens `~/.local/share/brainlayer/logs/maintenance.log`** for every `maintenance-*`
alert. That is the file #1061 makes hold every abort and failure reason, never the LaunchAgent's
`.out`/`.err`. The path is the one the job itself writes: `BRAINLAYER_MAINTENANCE_LOG_PATH` from
the job's LaunchAgent environment, which on this Mac is unset (checked read-only on 2026-10-04 for
both maintenance LaunchAgents), so the default applies. Pinned by
`test_show_log_defaults_to_the_shared_maintenance_log_never_the_launch_agent_stdout`.

**N1, a run that succeeds with warnings:** #1061's `{"status": "ok", "warnings": [...]}` record is
now read. When the latest run's own record carries warnings, the Maintenance card shows a quiet
amber note, e.g. "Nightly last run OK · search slower than target (62 ms)". The badge stays
Healthy (never red, never "Needs attention"):
[with a warning](after-warning-jobs-default-dark.png) ·
[after a later clean run](after-cleared-jobs-default-dark.png) (the note is gone: 307 amber pixels
→ 0). An unrecognised warning is shown as written.
