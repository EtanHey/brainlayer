# Calendar maintenance explanations — synthetic native evidence

Status: source rendering inspected; installed/live acceptance not performed.
Implementer: bl-stability-lead-w5, Codex gpt-6.1-sol.

Original screencapture PNGs show source-built native BrainBar with synthetic records.
Window titles identify the fixture; no AppDelegate, canonical database or services start.
The before images are retained from the original pre-maintenance consumer capture.
Fresh after images use the production history reader and private synthetic runs.jsonl:

- `before-missed-jobs.png`: rc75 with reason unavailable.
- `after-missed-jobs.png`: recorded index window reason plus weekly scheduled instead.
- `after-missed-dashboard.png`: expanded attention visibly names the index window reason.
- `before-success-jobs.png`: generic calendar result, no substitution explanation.
- `after-success-jobs.png`: Nightly skipped: weekly ran instead.

The missed Dashboard reason is distinct from the index-failure health fixture. An incidental
chart hover tooltip does not cover the reason. Fixture dates/counts are not current health.
Successful weekly substitution adds no actionable Dashboard item; its proof is the Jobs image.
The old collapsed/duplicate Dashboard captures remain retained locally and are not accepted here.
Capture: existing DEBUG BRAINBAR_JOB_HEALTH_QA plus BRAINBAR_MAINTENANCE_QA=missed|success,
then `/usr/sbin/screencapture -x -l <fixture-window-id>`; each fixture exits itself.

The corrected health parent is merged forward, preserving both round-1 reviewed commits.
No reason/replaces producer changes are made by this correction; IDX custody/budget hunks
remain separate and must be reconciled by the lead at integration.
