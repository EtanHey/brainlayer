# Bugbot Review Notes - PR #273

## Summary
This PR addresses three Bugbot findings from PR #268 plus adds startup database retry logic for migration DDL lock contention self-healing.

## Changes Reviewed

### 1. BrainBarServer.swift - Database Startup Retry Logic ✅

**Added:**
- `DatabaseRecoveryPolicy` struct with exponential backoff configuration
- `attemptDatabaseOpen()` method that checks `db.isOpen` and schedules retries on failure
- `scheduleDatabaseRetry()` with exponential backoff (default 1s → 30s max)
- `databaseRetryWorkItem` and `lastDatabaseRetryDelayMillis` state tracking
- `onDatabaseReady` callback for test coordination

**Analysis:**
- ✅ Proper cancellation of retry work items in cleanup
- ✅ Guards against retry storms (nil check on `providedDatabase`, `listenSource`, `database`)
- ✅ Socket binds BEFORE database open (correct ordering for connection queueing)
- ✅ Router created first (no DB dependency for initialize/tools/list)
- ✅ Policy validation in init (busyTimeout, delays clamped to >= 1ms)
- ✅ nextDelay correctly implements exponential backoff with max clamp

**No issues found.**

### 2. BrainDatabase.swift - Trigram Rebuild Fixes ✅

**Change 1: Cancellation state propagation (line ~2070-2078)**
```swift
if shouldCancel() {
    let cancelled = TrigramMaintenanceProgress(
        state: .cancelled,
        processed: processed,
        total: total,
        etaSeconds: nil
    )
    progress(cancelled)
    return cancelled
}
```

**Analysis:**
- ✅ Correctly checks `shouldCancel()` between batches
- ✅ Returns cancelled state with accurate processed count
- ✅ Fires progress callback before returning (observable state)

**Change 2: Sparse rowid tracking (line ~2110)**
```swift
processed = min(processed + batch.rowCount, total)
```

**Analysis:**
- ✅ Uses `batch.rowCount` (actual rows) instead of rowid delta
- ✅ Clamps with `min(_, total)` to prevent overshoot
- ✅ `nextChunkBatch` returns actual row count (verified in implementation)

**No issues found.**

### 3. MCPRouter.swift - Cancellation Closure Documentation ✅

**Changed (line ~614):**
```swift
let final = try db.triggerTrigramRebuild(
    batchSize: batchSize,
    // Preflight-only: cancellation never reaches the inner rebuild loop here.
    shouldCancel: { false },
    progress: { event in
```

**Analysis:**
- ✅ Comment clarifies intent (preflight-only check, always false)
- ✅ No semantic change (still returns `false`)
- ✅ Correct behavior: MCPRouter doesn't support mid-rebuild cancellation

**No issues found.**

### 4. Tests - BrainBarStartupRecoveryTests.swift ✅

**New test file adds:**
- `testServerRecoversAfterStartupMigrationLockContention()`
- Opens external lock with `BEGIN IMMEDIATE`
- Releases lock after 250ms
- Verifies BrainBar retries and becomes operational
- Tests actual MCP request (brain_search) succeeds after recovery

**Analysis:**
- ✅ Uses isolated temp directory
- ✅ Seeds valid database before locking
- ✅ Waits for both lock release AND database ready
- ✅ End-to-end validation (MCP protocol + tool execution)
- ✅ Proper cleanup in tearDown

**No issues found.**

### 5. Tests - DatabaseTests.swift additions ✅

**New tests:**
- `testLargeTrigramDesyncDoesNotForceSynchronousStartupRebuild()` - validates skipBackfill decision
- `testTrigramMaintenanceBatchSizeIsClampedForExternalInput()` - validates normalization
- `testTriggerTrigramRebuildBackfillsInBatchesWithProgress()` - validates batch rebuild
- `testTriggerTrigramRebuildHonorsCancellationBetweenBatches()` - validates cancellation
- `testTriggerTrigramRebuildCancellationPreservesUnprocessedLiveRows()` - validates partial state
- `testTriggerTrigramRebuildAllowsWritersBetweenBatches()` - validates lock release
- `testTriggerTrigramRebuildProgressTracksActualRowsAcrossSparseRowIDs()` - validates rowCount fix
- `testTriggerTrigramRebuildDoesNotDuplicateRowsWhenChunkUpdatesBetweenBatches()` - validates idempotency

**Analysis:**
- ✅ Comprehensive test coverage for all three bugbot nits
- ✅ Edge cases covered (sparse rowids, concurrent writes, cancellation timing)
- ✅ Helper methods reused appropriately (seedTrigramMaintenanceRows, sqliteCount, etc.)

**No issues found.**

## Overall Assessment

### Code Quality: ✅ Excellent
- Clean separation of concerns
- Proper error handling throughout
- No resource leaks detected
- Thread-safe design (serial queue in BrainBarServer, transaction batching in BrainDatabase)

### Test Coverage: ✅ Comprehensive
- 9 new tests covering startup retry, cancellation, progress tracking, and concurrency
- Integration test validates end-to-end recovery
- Edge cases explicitly tested (sparse rowids, concurrent updates)

### Safety: ✅ Strong
- All retry guards in place (prevents infinite loops)
- Cancellation properly propagates state
- Resource cleanup in all error paths
- Lock release between batches prevents writer starvation

### Documentation: ✅ Clear
- PR description references findings.md and PR #268
- Code comments explain non-obvious behavior (preflight cancellation, skipBackfill logic)
- Test names clearly describe intent

## Recommendations

None. This PR is production-ready.

## Test Results

According to PR description:
- ✅ `swift test --package-path brain-bar`: 337 passed, 0 failed
- ✅ `./scripts/run_tests.sh`: 1823 pytest passed, 3 MCP passed, 32 isolated passed, 1 bun passed, 1 regression passed

## Conclusion

All three Bugbot nits from PR #268 are correctly addressed:
1. ✅ Trigram rebuild cancellation state properly propagates via progress callback and return value
2. ✅ Progress tracking uses actual row count (`batch.rowCount`) instead of rowid delta
3. ✅ MCPRouter cancellation closure clarified as preflight-only with explicit comment

Additionally:
4. ✅ Startup database retry with exponential backoff self-heals DDL lock contention

**Recommendation: Approve and merge.**
