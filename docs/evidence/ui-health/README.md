# Dashboard calendar health — synthetic native evidence

Status: source rendering inspected; installed/live acceptance not performed.
Implementer: bl-stability-lead-w5, Codex gpt-6.1-sol.

These are original screencapture PNGs of source-built native BrainBar windows.
The title labels the synthetic fixture; AppDelegate, database and services are not started.
Fixture dates and numbers are synthetic, and the Jobs footer has no watcher health file.

- `before-dashboard.png`: retained original pre-health implementation; incorrectly says All good.
- `after-dashboard.png`: corrected source, attention expanded; index failure is visible.
- `after-jobs.png`: same synthetic index failure, matching Dashboard.

The before image is retained from the original implementation run; after images were captured
from the corrected health branch. Only disclosure expansion differs between capture setups.
Capture uses the existing DEBUG BRAINBAR_JOB_HEALTH_QA entrypoint and
`/usr/sbin/screencapture -x -l <fixture-window-id>`; fixture process exits itself.

Lifecycle/loading correction: the production-root navigation test changed synthetic observability
and configuration while Dashboard was selected, then entered Backups with the retained model.
Before correction: 2 tests, 4 assertion failures. After correction: 62 focused Swift tests pass,
including initial insertion, repeated entry, recovery/failure, drafts and newest-read protection.
Settings refresh now runs on initial insertion as well as window activation; Checking adds no
Dashboard attention, while disabled/failing jobs still do.
