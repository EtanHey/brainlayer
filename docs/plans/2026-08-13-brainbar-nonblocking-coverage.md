# BrainBar Non-Blocking Coverage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make BrainBar publish dashboard and menubar graph data immediately while exact lifecycle-filtered coverage computes independently in the background.

**Architecture:** Split `BrainDatabase.dashboardStats` into a hot snapshot that can omit signal coverage and a separate exact coverage provider. `StatsCollector` publishes the hot snapshot first, retains the last exact coverage result as an in-memory cache, and refreshes coverage on an independent generation-checked task. The UI represents missing coverage as `computing…`; no schema or trigger changes are included.

**Tech Stack:** Swift 6, Swift concurrency, Combine/SwiftUI, SQLite3, XCTest.

---

### Task 1: RED — define the truthful hot-snapshot contract

**Files:**
- Modify: `brain-bar/Tests/BrainBarTests/DashboardTests.swift`
- Modify: `brain-bar/Tests/BrainBarTests/StatsCollectorTests.swift`

**Step 1: Replace the rejected map-count regression**

Add `testDashboardStatsCanSkipSignalCoverageForHotSnapshot`. Insert ordinary chunks, call
`dashboardStats(..., includeSignalCoverage: false)`, and assert:

```swift
XCTAssertFalse(stats.signalCoverageIsAvailable)
XCTAssertEqual(stats.signalEligibleChunkCount, 0)
XCTAssertEqual(stats.vectorIndexedChunkCount, 0)
XCTAssertEqual(stats.ftsIndexedChunkCount, 0)
XCTAssertEqual(stats.trigramIndexedChunkCount, 0)
```

Keep the existing default-path test that proves exact eligible/indexed/backlog values.

**Step 2: Add the collector ordering regression**

Inject an immediate hot stats provider and a coverage provider held by a `DispatchSemaphore`. Start
the collector and prove the snapshot becomes `.live` with `signalCoverageIsAvailable == false`
before releasing coverage. Release the provider and prove the exact snapshot is merged later without
changing the hot snapshot's graph buckets.

**Step 3: Run RED tests**

Run:

```bash
cd brain-bar
swift test --filter DashboardTests/testDashboardStatsCanSkipSignalCoverageForHotSnapshot
swift test --filter StatsCollectorTests/testHotSnapshotPublishesBeforeExactSignalCoverage
```

Expected: compile failures for the missing `includeSignalCoverage`, availability state, providers,
and exact coverage snapshot API.

### Task 2: Implement the split database contract

**Files:**
- Modify: `brain-bar/Sources/BrainBar/BrainDatabase.swift`

**Step 1: Add exact coverage value state**

Create `SignalCoverageSnapshot: Sendable, Equatable` with eligible, vector, FTS, and trigram counts.
Add `signalCoverageIsAvailable` to `DashboardStats` with a default of `true` for existing fixtures.
Add copy helpers that preserve availability and one helper that merges an exact snapshot.

**Step 2: Add the hot-path switch**

Change the database API to:

```swift
func dashboardStats(
    activityWindowMinutes: Int = 30,
    bucketCount: Int = 12,
    includeSignalCoverage: Bool = true
) throws -> DashboardStats
```

When false, do not call `dashboardSignalCoverageCounts`; publish zero placeholders marked
unavailable. Keep the default true so existing exact callers and tests preserve behavior.

**Step 3: Expose exact coverage separately**

Add `dashboardSignalCoverageSnapshot()` that runs the existing lifecycle-filtered
`dashboardSignalCoverageCounts()` inside its own read transaction and returns the snapshot. Restore
the pre-PR virtual-table joins; remove all direct `chunk_fts_rowids` counting code.

**Step 4: Run database tests**

Run:

```bash
swift test --filter DashboardTests/testDashboardStatsCanSkipSignalCoverageForHotSnapshot
swift test --filter DashboardTests/testDashboardStatsReportsPerSignalCoverageAndBacklogs
```

Expected: both pass.

### Task 3: Publish hot stats before background coverage

**Files:**
- Modify: `brain-bar/Sources/BrainBar/Dashboard/StatsCollector.swift`
- Test: `brain-bar/Tests/BrainBarTests/StatsCollectorTests.swift`

**Step 1: Add injectable providers**

Add `DashboardStatsProvider` and `SignalCoverageProvider` sendable closures. Defaults open separate
short-lived database handles; the hot provider calls `dashboardStats(includeSignalCoverage: false)`
and the coverage provider calls `dashboardSignalCoverageSnapshot()`.

**Step 2: Add independent coverage lifecycle**

Add a coverage task, generation, last-attempt time, error, refresh interval, and refreshing flag.
Cancel/invalidate it in `stop()`. Never cancel a running coverage task because another hot refresh
arrives.

Failed exact attempts are throttled by the same interval so a broken or busy database cannot turn
ordinary hot refreshes into an expensive retry storm.

**Step 3: Preserve cached exact coverage**

On hot success, merge the previous exact coverage into the new hot snapshot when available, publish
and clear global `.loading`, then request coverage only when no task is active and the periodic
interval is due. Coverage failure must not set `lastFetchError` or global snapshot freshness.

**Step 4: Run the ordering test**

Run:

```bash
swift test --filter StatsCollectorTests/testHotSnapshotPublishesBeforeExactSignalCoverage
```

Expected: pass; the hot assertion completes before the semaphore is released.

### Task 4: Render unknown coverage honestly

**Files:**
- Modify: `brain-bar/Sources/BrainBar/BrainBarWindowRootView.swift`
- Modify: `brain-bar/Tests/BrainBarTests/DashboardTests.swift`

**Step 1: Write the UI RED test**

Add a source-contract or model test proving unavailable coverage produces the exact visible string
`computing…` and does not expose a numeric percent/backlog.

**Step 2: Thread availability through the coverage model**

Add `isAvailable` to `BrainBarSignalCoverage`; pass `stats.signalCoverageIsAvailable` for Vector,
FTS5, and Trigram. Make `percentText` and `backlogText` return `computing…` while unavailable.
Replace the animated foreground bar with a neutral background-only capsule while unavailable.

**Step 3: Run focused UI/dashboard tests**

Run:

```bash
swift test --filter DashboardTests
swift test --filter StatsCollectorTests
```

Expected: all focused tests pass.

### Task 5: Review, publish, and deploy

**Files:**
- Update: `docs/plans/2026-08-13-brainbar-nonblocking-coverage-design.md`
- Update: `docs/plans/2026-08-13-brainbar-nonblocking-coverage.md`

**Step 1: Verify and commit**

Run `git diff --check`, focused Swift suites, the full Swift package, and `coderabbit review --agent`.
Commit the replacement implementation and push with `BRAINLAYER_PREPUSH_SCOPE=changed-only`.

**Step 2: Resolve review threads**

Reply to MF-1/MF-2 with the replacement design and exact test evidence. Re-request the same Claude
pair reviewer and `@codex review`; wait for CI and actionable review findings.

**Step 3: Merge and deploy**

Merge with a merge commit after clean review. Build/install the canonical app from merged `main`,
restart the GUI and daemon, and prove new PIDs execute `/Applications/BrainBar.app`, launchd is live,
and the embedded Git commit matches the merge.

**Step 4: Live UI proof**

Prove the installed collector logs a completed hot refresh before exact coverage, the dashboard is
not `.loading`, menubar buckets carry real non-flat values when activity exists, and coverage either
shows `computing…` or a later exact lifecycle-filtered value without blocking the hot snapshot.

**Step 5: File the trigger-gap follow-up**

Create a signed GitHub issue for lifecycle columns (`archived_at`, `superseded_by`,
`aggregated_into`, `status`) not updating FTS trigger state. Do not implement it in this PR.
