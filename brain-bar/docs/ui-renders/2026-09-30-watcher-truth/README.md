# One watcher-health truth: render proof (#966)

These renders come from the DEBUG render harness (`BRAINBAR_RENDER_ONLY`, `renderWatcherTruth`) with fixture data. They are pixels from a source build, not proof from the installed app or live data. No DB, socket or launchctl call is made during the run.

```sh
DEVELOPER_DIR=/Applications/Devtools/Xcode.app/Contents/Developer swift build --product BrainBar
BRAINBAR_DB_PATH=<scratch>/brainlayer.db BRAINBAR_RENDER_ONLY=<scratch>/out .build/debug/BrainBar
```

Each state is rendered on both surfaces that show watcher health, at 760, 960 and 1280 pt, for 30 PNGs in total:
- **Dashboard:** the "things need you" strip (expanded) and the Ingest card.
- **Settings → Jobs:** the Ingest group card and the footer.

Before writing, the harness asserts that the Dashboard's `WatcherHealthStatus` title and the Settings footer title both equal the expected title for the state. All 30 PNGs were inspected, through the two contact sheets. The 960 pt renders are committed individually.

| State | Watcher evidence | Dashboard (960) | Settings → Jobs (960) |
| --- | --- | --- | --- |
| Running | launchd running, heartbeat 70 s old | [PNG](running-dashboard-960.png): All good | [PNG](running-settings-960.png): Ingest Healthy, footer "Watcher running" |
| Idle with replay debt (the #966 repro) | running, fresh heartbeat, 0 watcher chunks, BrainBar replay debt > 0 | [PNG](idle-replay-debt-dashboard-960.png): **All good**. It used to say "Watcher flow needs attention." | [PNG](idle-replay-debt-settings-960.png): Watcher running |
| Degraded | running, `file_ingestion_failure` ×2 since 2 h | [PNG](degraded-dashboard-960.png): "2 transcript files could not be ingested · since 2h ago · See file_ingestion_failures in watcher-health.json" | [PNG](degraded-settings-960.png): the same line on the Ingest card and in the footer |
| Stopped | launchd: not loaded / not running | [PNG](stopped-dashboard-960.png): "Watcher is not running (…). Restart it from Settings → Jobs → Ingest, or check ~/Library/Logs/brainlayer/watch.err.log." | [PNG](stopped-settings-960.png): Ingest Needs attention, footer "Watcher stopped" |
| Unknown | running, health file missing | [PNG](unknown-dashboard-960.png): "Watcher is running, but its health file is missing at …/watcher-health.json." | [PNG](unknown-settings-960.png): Ingest **Status unknown** (not "Needs attention"), footer "Watcher status unknown" |

- Contact sheets covering all three widths: [dashboard](contact-sheet-dashboard-760-960-1280.png), [settings](contact-sheet-settings-760-960-1280.png).
- At 760 pt the footer wraps its reason to two lines and truncates the rest, with the full text in the tooltip. The Ingest card above it always shows the full line.
