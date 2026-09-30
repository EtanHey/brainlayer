# One window: Dashboard in the sidebar (#963 PR 2)

These are DEBUG render-harness captures from fixture data, not the installed app.

```sh
DEVELOPER_DIR=/Applications/Devtools/Xcode.app/Contents/Developer swift build --product BrainBar
BRAINBAR_DB_PATH=<scratch>/brainlayer.db BRAINBAR_RENDER_ONLY=<scratch>/out .build/debug/BrainBar
```

## What the window looks like now
- **No top tabs.** The header keeps the brand, "Refresh now" (on Dashboard only) and the app menu.
- **One sidebar:** Dashboard, then a "Settings" caption, then Jobs, Backups and Advanced. The selected row is tinted with the accent colour.
- **General is gone.** It only said "Monitor memory activity on Dashboard. Manage services in Jobs and Backups", so it is absorbed into Dashboard.
- The Settings pages no longer draw their own inner sidebar; there is one sidebar per window.

## Captures
- **Dashboard:** the real `BrainBarMainWindow` with its title bar at 760, 960 and 1280 × 640: [760](dashboard-compact.png), [960](dashboard-default.png), [1280](dashboard-wide.png). The dashboard reflows to one column at 760, where it has 583 pt beside the 176 pt sidebar.
- **Settings pages at 960:** [Jobs](jobs-default.png), [Backups](backups-default.png), [Advanced](advanced-default.png).
- **Contact sheets** covering all three widths: [Dashboard](contact-sheet-dashboard-760-960-1280.png), and [Settings](contact-sheet-settings-760-960-1280.png) (Jobs, Backups, Advanced, and the save-receipt state).

The "unavailable" rows come from the fixture runtime, which has no DB path or backup status.

These renders are re-generated on `main` after #1014 and #1016 merged. The Backups page shows #1016's Schedule block with Reveal in Finder / Copy path, and the Jobs page and footer show #1014's watcher-health line, all inside the one sidebar.
