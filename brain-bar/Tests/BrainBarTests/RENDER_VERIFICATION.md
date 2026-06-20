# BrainBar Render-Verification Infra

Deterministic PNG renders of BrainBar's real UI so **any agent can visually verify
the dashboard without launching the app**.

## Why this exists

BrainBar is a menu-bar / `LSUIElement` app. That breaks the usual visual-check tools:

- `computer-use` MCP returns **`not_installed`** for it.
- Full-screen `screencapture` grabs the **wrong window**.

So there was no reliable way to confirm "the UI actually looks right" — which led to
repeated **over-claims** (asserting the UI rendered correctly with no proof). This
infra makes visual verification deterministic, cheap, and offline: run a snapshot
test, then `Read` the emitted PNG.

## How to render + verify (the agent flow)

```bash
cd brain-bar
swift test --filter BrainBarDashboardSnapshotTests
```

That writes PNGs to **`brain-bar/docs.local/brainbar-render/`**:

| File | What it shows | Layout breakpoint |
|------|---------------|-------------------|
| `dashboard-compact.png`  | Full dashboard (hero/overview + pipeline + diagnostics) | width 760 — compact cards, 1 chart column |
| `dashboard-default.png`  | Full dashboard | width 960 — roomy cards, 1 chart column |
| `dashboard-wide.png`     | Full dashboard | width 1280 — 2 chart columns |
| `settings.png`           | Settings panel (enrichment config, launchd jobs) | width 700 |

Then `Read` any PNG to inspect the UI. Override the output directory with
`BRAINBAR_RENDER_DIR=/abs/path swift test --filter BrainBarDashboardSnapshotTests`.

## Why the renders are deterministic (byte-stable)

The dashboard is rendered from a fixed fixture, not live state:

1. **`BrainBarDashboardFixture`** (`Sources/BrainBar/Dashboard/BrainBarDashboardFixture.swift`,
   `#if DEBUG`) — canonical `DashboardStats` / `DaemonHealthSnapshot` /
   `AgentActivitySnapshot` with literal counts and buckets. **No live DB, daemon,
   or clock.**
2. **`StatsCollector.fixture(...)`** (`#if DEBUG`, in `StatsCollector.swift`) — loads
   that state into a collector with **no `start()`**: no database open, no Darwin
   observer, no refresh timers.
3. **No relative-time text.** Every `Date?` that drives "Xm ago" strings
   (`lastWriteAt`, `lastEnrichedAt`, `pendingStoreOldestQueuedAt`) is `nil`. The only
   timestamp rendered (`lastDataFetchedAt`) shows through an **absolute** formatter.
   That's what makes the output independent of the wall clock.
4. **`accessibilityReduceMotion = true`** — SwiftUI animations resolve to their final
   state immediately, so there's no mid-animation capture.

## How it's wired (extends the existing snapshot seam)

This extends the same render seam used by `BrainBarSettingsSnapshotTests` and the
`BrainBarFlowLaneCardPreview` debug hook — `NSHostingView` + `cacheDisplay` to a
`NSBitmapImageRep` → PNG. (That AppKit path is used instead of SwiftUI `ImageRenderer`
because the dashboard's glass/material backgrounds composite correctly through
`cacheDisplay` but render blank through `ImageRenderer`.)

- **`BrainBarDashboardPreview.make(collector:)`** (`#if DEBUG`, in
  `BrainBarWindowRootView.swift`) exposes the otherwise-`private`
  `BrainBarDashboardView`, wrapped in the dark app background with reduceMotion on.
- **`BrainBarDashboardSnapshotTests`** renders it at each breakpoint and asserts the
  PNG is non-trivial (size + distinct-color floor → catches blank/clipped renders).

All render code is `#if DEBUG` only and never ships in a release build.

## Extending

- **New view** → add a `#if DEBUG` `…Preview.make(...)` seam exposing the private
  view, then a test case that renders it. Keep relative-time dates `nil` and
  reduceMotion on.
- **New breakpoint** → add a case to `Breakpoint` in the test with its size.
- **New state** (e.g. degraded, idle, backlogged) → add a fixture variant in
  `BrainBarDashboardFixture` and a test case; the determinism rules above still apply.
