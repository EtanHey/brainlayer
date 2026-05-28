# BugBot Review: Phase 2 Optimization — Injections + Graph + Heartbeat

**PR**: feat/brainbar-phase2-injections-graph-heartbeat  
**Reviewer**: @bugbot  
**Date**: 2026-05-28  
**Commit**: 8a395f4b273d977fc8317f93af31501e580869c6

## Executive Summary

**Risk Level**: 🟡 **MEDIUM RISK**

This PR introduces important lifecycle optimizations for BrainBar's Injections tab, Knowledge Graph, and dashboard heartbeat system. The changes reduce background CPU usage when tabs are inactive or windows are hidden, while maintaining UI responsiveness through lightweight heartbeat signals.

**Key Safety Improvements**:
- ✅ InjectionStore now uses **read-only DB handle** (prevents write contention)
- ✅ Proper active/inactive gating with complete resource cleanup
- ✅ Darwin observer cleanup in deinit prevents use-after-free
- ✅ Debounced refresh prevents DB read storms during mutation bursts
- ✅ Comprehensive test coverage for new lifecycle paths

**Primary Concerns**:
- ⚠️ Read contention during heavy enrichment (known issue, mitigated by retries)
- ⚠️ Complexity in graph lifecycle state machine
- ℹ️ Window occlusion during in-flight refresh (minor, acceptable)

---

## Critical Path Review

### 1. Database Safety & Concurrency ✅

#### InjectionStore Read-Only Access (EXCELLENT)

```swift
// InjectionStore.swift:72-76
init(databasePath: String) throws {
    self.reader = BrainDatabase(
        path: databasePath,
        openConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
    )
```

**Finding**: ✅ **SAFE** - InjectionStore now opens database in read-only mode, eliminating write contention risk with the Python enrichment pipeline.

**Rationale**: 
- Prevents SQLITE_BUSY errors from competing writers
- Aligns with BrainBar's read-heavy UI workload
- Consistent with agent guidelines: "Flag risky DB or concurrency changes explicitly"

#### Darwin Observer Lifecycle (SAFE with caveats)

```swift
// InjectionStore.swift:141-156
deinit {
    // Always remove — CFNotificationCenterRemoveObserver is a no-op on a
    // never-registered observer, so this is safe regardless of state.
    let center = CFNotificationCenterGetDarwinNotifyCenter()
    CFNotificationCenterRemoveObserver(
        center,
        Unmanaged.passUnretained(self).toOpaque(),
        CFNotificationName(BrainDatabase.dashboardDidChangeNotification as CFString),
        nil
    )
}
```

**Finding**: ✅ **SAFE** - Good defensive programming. The deinit unconditionally removes the observer even if `observerInstalled` flag is false, preventing use-after-free crashes.

**Minor Note**: The `observerInstalled` flag in `removeDarwinObserver()` could theoretically get out of sync, but the deinit safety net catches this case.

---

### 2. Active/Inactive Gating ✅

#### InjectionStore Lifecycle

```swift
// InjectionStore.swift:123-139
func setActive(_ active: Bool) {
    guard isRunning else { return }
    guard isActive != active else { return }

    isActive = active
    if active {
        installDarwinObserver()
        scheduleRefresh(force: events.isEmpty, immediate: true)
        startPolling()
    } else {
        pendingRefreshTask?.cancel()
        pendingRefreshTask = nil
        pollTask?.cancel()
        pollTask = nil
        removeDarwinObserver()
    }
}
```

**Finding**: ✅ **CORRECT** - Clean activation/deactivation with proper task cancellation and resource cleanup.

**Verification**: Test coverage confirms inactive stores don't poll:

```swift
// InjectionStoreTests.swift:135-156
func testInactiveStoreDoesNotPollUntilActivated() async throws {
    store.start(active: false)
    try await Task.sleep(for: .milliseconds(180))
    XCTAssertEqual(reader.dataVersionCallCount, 0)
    XCTAssertTrue(store.events.isEmpty)
    
    store.setActive(true)
    try await Task.sleep(for: .milliseconds(80))
    XCTAssertEqual(store.events.map(\.query), ["active event"])
}
```

#### Graph Simulation Lifecycle

```swift
// KGSimulationController.swift:57-62
func setActive(_ active: Bool) {
    isActive = active
    if !active {
        stop()
    }
}
```

**Finding**: ✅ **CORRECT** - Simple but effective. When tab becomes inactive, simulation stops immediately.

**Integration Point**:

```swift
// KGCanvasView.swift:118-124
.onChange(of: isActive) { _, active in
    if active {
        startSimulation()
    } else {
        stopSimulation()
    }
}
```

---

### 3. Refresh Debouncing ✅

```swift
// InjectionStore.swift:189-212
private func scheduleRefresh(force: Bool, immediate: Bool = false) {
    guard isActive else { return }
    pendingRefreshTask?.cancel()
    pendingRefreshTask = nil

    if immediate || refreshDebounceInterval <= 0 {
        refresh(force: force)
        return
    }

    let delay = refreshDebounceInterval
    pendingRefreshTask = Task { [weak self] in
        do {
            try await Task.sleep(for: .seconds(delay))
        } catch {
            return
        }
        await MainActor.run {
            guard let self, self.isActive else { return }
            self.pendingRefreshTask = nil
            self.refresh(force: force)
        }
    }
}
```

**Finding**: ✅ **EXCELLENT** - Debouncing prevents DB read storms during Darwin notification bursts (e.g., rapid enrichment writes).

**Test Coverage**:

```swift
// InjectionStoreTests.swift:159-179
func testActiveRefreshDebouncesMutationBursts() async throws {
    let store = InjectionStore(reader: reader, refreshDebounceInterval: 0.05)
    store.start(active: true)
    store.scheduleRefreshForTesting(force: false)
    store.scheduleRefreshForTesting(force: false)
    store.scheduleRefreshForTesting(force: false)
    
    try await Task.sleep(for: .milliseconds(120))
    
    XCTAssertEqual(reader.listCallCount, 2)  // Initial + final debounced
}
```

**Default**: 150ms debounce interval is reasonable for UI responsiveness vs. DB load trade-off.

---

### 4. Heartbeat vs Stats Refresh Separation ✅

#### Lightweight Heartbeat Path

```swift
// StatsCollector.swift:357-372
private func handleBrainBusEvent(_ event: BrainBusEvent) {
    recordHeartbeat(
        event: event,
        trigger: "brain_bus",
        timestamp: event.generatedAt
    )

    switch event.type {
    case .healthTick:
        daemon = daemonMonitor.sample()
        refreshAgentActivity(force: false, now: Date())
        state = PipelineState.derive(daemon: daemon, stats: stats)
    case .queueDepth, .enrichStatus, .lastChunkID, .dbBusy:
        break
    }
}
```

**Finding**: ✅ **GOOD DESIGN** - Most BrainBus events now only update the heartbeat timestamp without triggering full DB stats queries. Only `healthTick` refreshes lightweight daemon/agent activity.

#### Database Notifications Schedule Coalesced Refresh

```swift
// StatsCollector.swift:348-355
fileprivate func handleDatabaseMutationNotification() {
    recordHeartbeat(
        event: nil,
        trigger: "darwin_db_notification",
        timestamp: Date()
    )
    schedulePendingStatsRefresh(after: statsRefreshCoalesceInterval)
}
```

**Finding**: ✅ **CORRECT** - DB change notifications update heartbeat immediately but schedule a coalesced stats refresh (default 5s), preventing excessive `dashboardStats()` queries.

---

### 5. Knowledge Graph Cached Load ✅

```swift
// KGViewModel.swift:103-111
@discardableResult
func loadGraphIfNeeded(
    retrySleep: (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
) async -> Bool {
    if hasLoadedGraph {
        return true
    }
    return await loadGraph(retrySleep: retrySleep)
}
```

**Finding**: ✅ **CORRECT** - Graph loads once when tab becomes active, eliminating repeated polling. The `hasLoadedGraph` flag prevents redundant loads.

**Integration**:

```swift
// KGCanvasView.swift:90-106
.task(id: isActive) {
    guard isActive else {
        stopSimulation()
        return
    }
    guard !hasStartedGraphPolling else { return }
    hasStartedGraphPolling = true
    defer { hasStartedGraphPolling = false }
    if await viewModel.loadGraphIfNeeded() {
        hasLoadedGraph = true
        if reduceMotion {
            _ = viewModel.tick(reduceMotionEnabled: true)
        } else {
            startSimulation()
        }
    }
}
```

**Note**: This replaces the previous `loadGraphRepeatedly()` pattern. Graph data is now static per tab activation, which is acceptable for Phase 2. Future phases may want live graph updates.

---

## Risk Analysis

### 🟡 Medium Risk: Read Contention During Heavy Enrichment

**Context**: PR description lists "DB locking during enrichment" and "WAL can grow to 4.7GB" as known issues.

**Mitigation in Place**:

```swift
// KGViewModel.swift:71-95
// Retry once on transient ReadOnly/busy/locked failures from the writer
// pidfile contention (PR #309). The Python enrich-supervisor + drain
// hold the writer briefly; one retry after a short backoff usually
// suffices. Persistent failures surface as a degradation badge.
let attemptLimit = 2
var lastError: Error?
for attempt in 1...attemptLimit {
    do {
        let graph = try await Self.fetchGraphRows(reader: graphReader)
        applyGraph(entityRows: graph.entities, relationRows: graph.relations)
        degradationState = .healthy
        hasLoadedGraph = true
        return true
    } catch {
        lastError = error
        if attempt < attemptLimit {
            do {
                try await retrySleep(.milliseconds(200))
            } catch {
                markDegraded(from: lastError)
                return false
            }
        }
    }
}
```

**Assessment**: 
- ✅ Retry logic is appropriate for transient locks
- ✅ Degradation badge surfaces persistent failures
- ⚠️ Read-only handles can still block on WAL checkpoint or heavy write traffic
- ⚠️ 750ms polling in InjectionStore + 30s auto-refresh in StatsCollector could amplify contention during enrichment

**Recommendation**: Acceptable for Phase 2, but monitor for:
- Increased degradation badge occurrences during bulk enrichment
- UI freezes if multiple readers hit checkpoint WAL flush simultaneously

### 🟢 Low Risk: Graph Lifecycle State Machine Complexity

**State Variables**:
- `hasLoadedGraph` (KGViewModel)
- `hasStartedGraphPolling` (KGCanvasView)
- `isActive` (passed from parent)
- `simulationController.timerActive`

**Assessment**: 
- ✅ Test coverage for lifecycle transitions is good
- ✅ State guards prevent redundant work
- ⚠️ Four interacting flags create complexity, but well-tested

**Recommendation**: Add a test for rapid tab switching (active → inactive → active) to verify state consistency.

### 🟢 Low Risk: In-Flight Refresh During Window Occlusion

**Scenario**: User hides window while dashboard stats refresh is in progress.

**Current Behavior**: Refresh completes and updates published state even if window is now hidden.

**Assessment**: 
- ℹ️ This is acceptable — the read is already in flight, and the result will be fresh when window re-appears
- ℹ️ The background task uses `.utility` priority which is appropriate for this case
- ✅ Active gating prevents *new* refreshes from starting when inactive

**Recommendation**: No action needed for Phase 2.

---

## Test Coverage Review ✅

### New Tests Added

1. **InjectionStoreTests.swift**
   - `testInactiveStoreDoesNotPollUntilActivated` ✅
   - `testActiveRefreshDebouncesMutationBursts` ✅
   - `testDegradedStoreStaysDegradedUntilEventQuerySucceeds` ✅ (regression guard)

2. **KnowledgeGraphTests.swift**
   - `testKGSimulationControllerPausesWhenInactive` ✅
   - `testKGCanvasViewLoadsGraphOnceWhenActivated` ✅
   - `testKGViewModelLoadGraphIfNeededSkipsSecondLoad` ✅

3. **DashboardTests.swift**
   - `testHeartbeatUpdatesWithoutStatsRefresh` ✅
   - `testDarwinNotificationSchedulesCoalescedRefresh` ✅

**Assessment**: ✅ **EXCELLENT** - Test coverage directly addresses the new lifecycle behaviors introduced in Phase 2.

---

## Code Quality Observations

### ✅ Strengths

1. **Clear separation of concerns**: Heartbeat vs. stats refresh, active vs. inactive
2. **Defensive programming**: deinit observer cleanup, retry logic, degradation states
3. **Good naming**: `loadGraphIfNeeded`, `setActive`, `scheduleRefresh`
4. **Comprehensive tests**: Lifecycle edge cases covered

### ⚠️ Minor Notes

1. **Polling intervals**: 750ms (InjectionStore) and 30s (StatsCollector auto-refresh) are hard-coded in several places. Consider extracting to a shared configuration if these need tuning.

2. **Label decluttering logic**: 

```swift
// KGCanvasView.swift:242-262
private func labelledNodeIDs(for nodes: [KGNode]) -> Set<String> {
    let zoomThreshold: CGFloat = 0.8
    let importanceThreshold: Double = scale < zoomThreshold ? 6.0 : 4.0
    return Set(nodes.filter { $0.importance >= importanceThreshold }.map(\.id))
}
```

This is clever but the thresholds are magic numbers. A comment explaining the rationale would help future maintainers.

3. **Window visibility tracking**: The `BrainBarWindowObserver` pattern is good, but the attachment via `WindowAttachmentView` is indirect. Direct window observation might be clearer.

---

## Compliance with Agent Guidelines

### ✅ Retrieval Correctness
- No changes to search quality or MCP tool contracts
- Read-only DB access preserves data integrity

### ✅ Write Safety
- No writes in this PR (read-only handles)
- Proper task cancellation prevents partial updates

### ✅ Lock Handling
- Read-only DB connections reduce lock contention
- Darwin observer cleanup prevents UAF
- Retry logic handles transient lock failures

### ✅ MCP Stability
- No changes to MCP tool implementations
- No changes to brain_search, brain_store, etc.

---

## Verdict

### ✅ APPROVE with Monitoring

This PR introduces well-designed lifecycle optimizations with appropriate test coverage. The primary concern (read contention during enrichment) is a known issue with existing mitigations (retries, degradation badges).

**Pre-Merge Checklist**:
- ✅ Swift tests: 502 pass, 0 failures (verified per PR description)
- ✅ Python tests: 2193 pass (verified per PR description)
- ✅ Computer Use validation: completed (per PR description)
- ✅ CPU idle smoke test: 0.00% hidden-window steady-state (per PR description)

**Post-Merge Monitoring**:
- Monitor for increased degradation badge occurrences during bulk enrichment
- Watch for user reports of UI freezes during heavy watcher activity
- Consider telemetry for Darwin notification burst frequency and debounce effectiveness

**User Merge Control**:
Per PR description: "Do not auto-merge. Etan will inspect the final v4 demo before merge greenlight."

---

## Summary for User

I've completed a thorough review of this PR focusing on the critical concerns outlined in the agent guidelines: **retrieval correctness, write safety, MCP stability, DB operations, and concurrency**.

**✅ Major Safety Improvements:**
- InjectionStore now uses a **read-only DB handle**, eliminating write contention
- Proper active/inactive gating stops all background work when tabs are hidden
- Darwin observer cleanup in deinit prevents use-after-free crashes
- Debounced refresh prevents DB read storms during mutation bursts

**🟡 Primary Risk (Medium, Mitigated):**
Read contention during heavy enrichment is still possible, but mitigated by:
- Retry logic with backoff
- Degradation badges for persistent failures  
- Read-only access reduces lock severity

**✅ Test Coverage:**
Comprehensive tests cover inactive polling, debounced refresh, cached graph reload, and heartbeat-without-stats-refresh.

**Recommendation**: **APPROVE** with post-merge monitoring for degradation badge frequency during bulk enrichment.

The PR description states you'll inspect the final v4 demo before merge — this review confirms the code changes are sound from a correctness and safety perspective.
