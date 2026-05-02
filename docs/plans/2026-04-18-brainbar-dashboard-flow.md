# BrainBar Dashboard Flow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the contradictory BrainBar dashboard hero with a synchronized, visual flow board that shows writes, enrichment, backlog, and live/idle state from one coherent time model.

**Architecture:** Extend `DashboardStats` so the UI gets explicit window metadata and real last-event timestamps instead of inferring freshness from buckets. Derive a richer flow summary from daemon health, write activity, enrichment activity, and backlog, then rebuild the dashboard hero into dual visual lanes that share a single time window and separate “live right now” from “trend over the last window.”

**Tech Stack:** Swift, SwiftUI, AppKit, XCTest, SQLite-backed `BrainDatabase`, existing BrainBar dashboard components.

---

### Task 1: Add explicit dashboard time semantics

**Files:**
- Modify: `brain-bar/Sources/BrainBar/BrainDatabase.swift`
- Modify: `brain-bar/Sources/BrainBar/Dashboard/StatsCollector.swift`
- Test: `brain-bar/Tests/BrainBarTests/DashboardTests.swift`

**Step 1: Write the failing tests**

Add tests that assert `DashboardStats` carries:
- `activityWindowMinutes`
- `bucketCount`
- `liveWindowMinutes`
- `lastWriteAt`
- `lastEnrichedAt`

Also add a database test proving:
- a chunk written now and enriched 90 seconds ago reports `lastWriteAt != nil`
- `lastEnrichedAt != nil`
- `enrichmentRatePerMinute == 0`
- enrichment trend still contains the older completion

**Step 2: Run test to verify it fails**

Run: `swift test --filter DashboardTests`
Expected: compile failure or assertion failure because `DashboardStats` does not yet expose the new time fields.

**Step 3: Write minimal implementation**

Update `DashboardStats` and `dashboardStats(...)` to include the new metadata and fetch `MAX(created_at)` / `MAX(enriched_at)` timestamps using the same SQLite date parsing already used for bucket generation.

**Step 4: Run test to verify it passes**

Run: `swift test --filter DashboardTests`
Expected: the new stats tests pass and existing dashboard database tests still pass.

### Task 2: Replace the single enum with a richer flow model

**Files:**
- Modify: `brain-bar/Sources/BrainBar/Dashboard/PipelineState.swift`
- Modify: `brain-bar/Sources/BrainBar/BrainBarWindowState.swift`
- Modify: `brain-bar/Sources/BrainBar/Dashboard/DashboardMetricFormatter.swift`
- Test: `brain-bar/Tests/BrainBarTests/DashboardTests.swift`
- Test: `brain-bar/Tests/BrainBarTests/BrainBarUXLogicTests.swift`
- Test: `brain-bar/Tests/BrainBarTests/BrainBarWindowStateTests.swift`

**Step 1: Write the failing tests**

Add tests for a derived flow summary that can represent:
- writes live + enrichments live at the same time
- writes idle + enrichments draining backlog
- writes idle + enrichments idle + backlog present
- daemon unavailable

Add formatter tests that require:
- real last-event strings from actual timestamps, not bucket position
- a stable “live now” badge that only reflects the configured live window

**Step 2: Run test to verify it fails**

Run: `swift test --filter BrainBarUXLogicTests`
Run: `swift test --filter BrainBarWindowStateTests`
Expected: failures because the richer flow summary and timestamp-driven formatting do not exist yet.

**Step 3: Write minimal implementation**

Introduce a derived dashboard flow model with:
- a write lane state
- an enrichment lane state
- a backlog/queue state
- a top-level narrative for the hero

Keep `PipelineState` available for legacy consumers, but make it derive from the richer summary instead of directly from raw buckets.

Replace bucket-inferred `lastCompletionString(...)` with a timestamp-based formatter and add a generic “time ago” helper for both writes and enrichments.

**Step 4: Run test to verify it passes**

Run: `swift test --filter BrainBarUXLogicTests`
Run: `swift test --filter BrainBarWindowStateTests`
Expected: new derivation and formatter tests pass without breaking the existing layout/token tests.

### Task 3: Rebuild the dashboard into a dual-lane flow board

**Files:**
- Modify: `brain-bar/Sources/BrainBar/BrainBarWindowRootView.swift`
- Modify: `brain-bar/Sources/BrainBar/Dashboard/SparklineRenderer.swift`
- Possibly modify: `brain-bar/Sources/BrainBar/Dashboard/StatusPopoverView.swift`
- Test: `brain-bar/Tests/BrainBarTests/BrainBarUXLogicTests.swift`

**Step 1: Write the failing tests**

Add logic-level tests for any new copy helpers or layout-state helpers needed by the dashboard, including:
- synchronized window labels
- lane summaries using the same window value
- idle copy that no longer contradicts a recent-trend graph

**Step 2: Run test to verify it fails**

Run: `swift test --filter BrainBarUXLogicTests`
Expected: failures because the new helpers or derived labels are not implemented yet.

**Step 3: Write minimal implementation**

Rebuild `BrainBarDashboardView` so the hero becomes a flow board with:
- a left lane for writes
- a center queue/backlog card
- a right lane for enrichments
- shared window labeling
- real last-write / last-enriched timestamps
- live badges that are explicitly “now”
- charts with stronger fill, endpoint emphasis, and separate colors for ingress vs enrichment

Reuse the same derived flow summary everywhere in the window instead of mixing independent heuristics.

If `StatusPopoverView` still shows contradictory wording after the model change, update it to consume the same derived summary.

**Step 4: Run test to verify it passes**

Run: `swift test --filter BrainBarUXLogicTests`
Expected: the helper/copy tests pass and the dashboard remains readable under compact layout tests.

### Task 4: Keep legacy/menu surfaces consistent

**Files:**
- Modify: `brain-bar/Sources/BrainBar/BrainBarApp.swift`
- Modify: `brain-bar/Sources/BrainBar/Dashboard/StatusPopoverView.swift`
- Test: `brain-bar/Tests/BrainBarTests/DashboardTests.swift`

**Step 1: Write the failing tests**

Add or extend tests that verify the status popover still loads and that the live/menu surfaces can continue rendering with the updated state/formatter interfaces.

**Step 2: Run test to verify it fails**

Run: `swift test --filter DashboardTests`
Expected: failures if any legacy consumer still relies on removed copy or raw bucket inference.

**Step 3: Write minimal implementation**

Adapt menu-bar and popover consumers to:
- use timestamp-safe formatting
- use the derived summary or derived compatibility state
- keep their visual contract simple and backward-safe

**Step 4: Run test to verify it passes**

Run: `swift test --filter DashboardTests`
Expected: dashboard and popover tests pass together.

### Task 5: Full verification and finish

**Files:**
- No code changes required unless regressions appear

**Step 1: Run targeted Swift verification**

Run:
- `swift test --filter DashboardTests`
- `swift test --filter BrainBarUXLogicTests`
- `swift test --filter BrainBarWindowStateTests`

Expected: all targeted dashboard/state suites pass.

**Step 2: Run broader BrainBar verification**

Run: `swift test --filter BrainBar`
Expected: BrainBar package tests pass, or any unrelated pre-existing failures are called out explicitly before claiming completion.

**Step 3: Summarize and store the decision**

Use BrainLayer memory to store:
- what changed
- why synchronized time semantics were required
- why fail-rate was excluded from this pass unless a truthful source is added

**Step 4: Commit**

```bash
git add brain-bar/Sources/BrainBar/BrainDatabase.swift \
        brain-bar/Sources/BrainBar/Dashboard/StatsCollector.swift \
        brain-bar/Sources/BrainBar/Dashboard/PipelineState.swift \
        brain-bar/Sources/BrainBar/BrainBarWindowState.swift \
        brain-bar/Sources/BrainBar/Dashboard/DashboardMetricFormatter.swift \
        brain-bar/Sources/BrainBar/BrainBarWindowRootView.swift \
        brain-bar/Sources/BrainBar/Dashboard/SparklineRenderer.swift \
        brain-bar/Sources/BrainBar/Dashboard/StatusPopoverView.swift \
        brain-bar/Sources/BrainBar/BrainBarApp.swift \
        brain-bar/Tests/BrainBarTests/DashboardTests.swift \
        brain-bar/Tests/BrainBarTests/BrainBarUXLogicTests.swift \
        brain-bar/Tests/BrainBarTests/BrainBarWindowStateTests.swift \
        docs/plans/2026-04-18-brainbar-dashboard-flow.md
git commit -m "feat: redesign BrainBar dashboard flow board"
```
