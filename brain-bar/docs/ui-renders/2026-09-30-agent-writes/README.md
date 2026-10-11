# "Agent writes (24 h)" = agent brain_store writes (#965, ruling A)

These are DEBUG render-harness captures of the real window, in the sidebar Dashboard, from fixture data. They are not the installed app.

```sh
DEVELOPER_DIR=/Applications/Devtools/Xcode.app/Contents/Developer swift build --product BrainBar
BRAINBAR_DB_PATH=<scratch>/brainlayer.db BRAINBAR_RENDER_ONLY=<scratch>/out .build/debug/BrainBar
```

## What the Today card shows
- **Measured:** the count, "brain_store writes (24 h)", and a definition line under it: "brain_store calls by agents, last 24 h". Hovering shows the full definition. Captures at [760](dashboard-compact.png), [960](dashboard-default.png) and [1280](dashboard-wide.png); the fixture count is 175.
- **Unknown:** "brain_store writes (24 h) unknown — <reason>", never 0. [960 capture](dashboard-agent-writes-unknown-960.png), with the reason "database not open".

## The count
- It covers rows in `chunks` where `COALESCE(LOWER(TRIM(source)), '') = 'mcp'` and `created_at` falls in the rolling 24 h ending now, normalised to UTC epoch seconds.
- It is read live from BrainBar's own DB, the same database `brain_store` writes to, each time the Dashboard stats refresh. It does not come from the periodic observability document, so it can't go stale.
