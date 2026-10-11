# Backups schedule and Reveal in Finder: render proof (#968)

These renders come from the DEBUG render harness (`BRAINBAR_RENDER_ONLY`, `renderUnifiedSettings`) with fixture rows, at 760, 960 and 1280 pt. They are pixels from a source build, not proof from the installed app. The fixture times are built in the local calendar, so the group card and the schedule rows agree in any time zone.

| Variant | 760 | 960 | 1280 |
| --- | --- | --- | --- |
| Every schedule known, both local copies present | [PNG](backups-760.png) | [PNG](backups-960.png) | [PNG](backups-1280.png) |
| Transcript LaunchAgent missing | [PNG](backups-unknown-760.png) | [PNG](backups-unknown-960.png) | [PNG](backups-unknown-1280.png) |

Each row under **Schedule** shows its cadence ("daily at 03:17", "weekly on Sunday at 04:00"), its last run (from the job's own log), and its next run.

- **Reveal in Finder** and **Copy path** appear only where a local copy exists: the newest DB snapshot and the newest transcript archive. Weekly maintenance has no local copy of its own.
- In the unknown variant, the Transcripts row says why ("Schedule unknown — no LaunchAgent installed at …"), shows "No run recorded" and "Next run unknown", and has no actions.

All six PNGs were inspected.
