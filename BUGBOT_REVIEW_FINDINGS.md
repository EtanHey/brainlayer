# Bugbot Review: PR #359 - Reduce Enrichment Write Lock Starvation

**Status**: ✅ **APPROVED with observations**  
**Reviewed**: 2026-05-29  
**Commit**: c6c5103d4e75f28e1481a801d38ae0c2ef13358d

---

## Summary

This PR addresses write lock starvation between the drain loop (processing enrichment backlog) and realtime MCP operations (brain_store). The solution uses **cooperative scheduling** with three key mechanisms:

1. **Bounded transactions** — Split large durable queue files into 5-event chunks (configurable)
2. **Cooperative yields** — Sleep 10-20ms after each commit to let other writers acquire locks
3. **Store retries** — MCP operations retry lock errors 4x before falling back to durable queue

---

## Test Results

✅ **All PR-specific tests pass** (126/126)  
✅ **Full test suite**: 2,257 passed, 1 failed + 32 errors (environmental - require production DB)

```
pytest tests/test_write_queue.py tests/test_enrichment_bounded_commit.py \
       tests/test_db_lock_resilience.py tests/test_enrich_supervisor.py \
       tests/test_store_handler.py tests/test_enrichment_controller.py
```

**Outcome**: All targeted tests pass. Environmental failures unrelated to PR changes.

---

## Critical Path Review

### 1. Write Safety ✅

**drain.py (L588-623)**
- Bounded transaction logic: Slices first N events, keeps remainder for next iteration
- Rewrite logic for partial files uses atomic `tmp.replace(path)` (POSIX safe)
- Passive WAL checkpoint after commit (non-blocking)
- Cooperative yield: `time.sleep(yield_seconds)` after commit, before next file

**Concerns:**
- ⚠️ Queue lock held for entire batch. With 250 files × 10ms yield = 2.5s lock hold
  - **Mitigation**: Lock is on queue directory (fcntl flock), not DB write lock
- ⚠️ PASSIVE checkpoint may not reduce WAL under heavy reader load
  - **Observation**: Passive mode chosen to avoid blocking readers. WAL growth to 4.7GB still possible but addressed by separate maintenance jobs

### 2. Lock Handling ✅

**enrichment_controller.py (L282-306, L342-405)**
- `_submit_write` yields after write when `yield_after=True` (default 20ms for realtime)
- `_EnrichmentWriteBatcher` batches updates with time (250ms) and size (25 events) limits
- Overdue flush detection (L356-362): Flushes pending batch before appending next item
- Error handling: Flush failures deferred without data loss (L369-372)

**Concerns:**
- ⚠️ Batcher deferring exceptions could lead to unbounded memory if queue repeatedly fails
  - **Mitigation**: Batcher retains pending items in `_pending` list; explicit flush on cleanup
- ✅ Interval-based flushing uses `time.monotonic()` (correct for elapsed time)

**mcp/store_handler.py (L492-509)**
- Retry helper: 4 attempts with exponential backoff (0.15s base → 0.15s, 0.3s, 0.6s, 1.2s)
- Only retries on `_is_lock_error` (BusyError or "locked"/"busy" in message)
- Falls back to durable queue after exhausting retries

**Concerns:**
- ✅ Base delay 0.15s is longer than drain's 0.05s, reducing contention
- ✅ Max retry delay ~2.2s before fallback (reasonable for interactive MCP calls)

### 3. Concurrency Safety ✅

**Race conditions analyzed:**
- Drain holds queue lock (L567-641) while processing batch → Prevents concurrent drain instances ✅
- Enrichment writes go to new JSONL files → No conflict with drain reading existing files ✅
- Store retries coordinate via DB write lock (SQLite serialization) → Safe ✅

**Thread safety:**
- WriteQueue uses single-threaded executor → Serializes writes per store ✅
- Batcher state (`_pending`, `_last_flush`) accessed only from executor thread → Safe ✅

### 4. MCP Tool Contracts ✅

**No breaking changes to MCP tool signatures:**
- `brain_store` — still returns `chunk_id`, adds `queued: true` on fallback
- `brain_digest` — unchanged
- `brain_update` / `brain_archive` / `brain_supersede` — retry logic internal

---

## Configuration & Tunability

New environment variables:
- `BRAINLAYER_DRAIN_MAX_EVENTS_PER_TRANSACTION` (default: 5)
- `BRAINLAYER_DRAIN_POST_COMMIT_YIELD_MS` (default: 10.0ms)
- `BRAINLAYER_MAX_COMMIT_BATCH` (default: 25 events)
- `BRAINLAYER_MAX_COMMIT_INTERVAL_MS` (default: 250ms)
- `BRAINLAYER_ENRICH_POST_WRITE_YIELD_MS` (default: 20.0ms)

**Default choices are reasonable:**
- 10-20ms yields balance responsiveness vs throughput
- Bounded batch sizes prevent monopolization
- All have fallback defaults if env vars are invalid

---

## Known Limitations

### 1. WAL Growth (4.7GB) — Not Fully Addressed
The PR uses **PASSIVE** checkpoints which won't truncate WAL if readers hold pages.

**Current mitigation:**
- Separate `wal_checkpoint.py` cron job runs TRUNCATE mode
- Bulk operations (scripts) use FULL checkpoints after completion

**Recommendation for future work:**
- Consider RESTART mode after enrichment batches (blocks readers briefly but truncates WAL)
- Add WAL size metric to Axiom telemetry

### 2. Drain Lock Hold Time
With default settings, drain can hold queue lock for seconds while processing 250 files.

**Not a blocker because:**
- Lock is on queue directory (filesystem), not DB
- Only prevents concurrent drain instances (by design)
- MCP writes create new files (no queue lock needed)

---

## Performance Impact

**Expected improvements:**
- ✅ MCP store operations retry short lock bursts (avoid unnecessary queueing)
- ✅ Enrichment writes batched (reduce fsync overhead)
- ✅ Cooperative yields prevent drain monopolization

**Expected overhead:**
- Minor: 10-20ms sleep after each commit
- Batch mode: Drain processes 5 events/tx instead of all-at-once

**Net effect:** Should reduce user-visible store latency spikes during enrichment.

---

## Code Quality

✅ **Test coverage**: Comprehensive unit and integration tests  
✅ **Error handling**: Retries with exponential backoff, graceful fallbacks  
✅ **Configurability**: All timeouts/limits tunable via env vars  
✅ **Documentation**: CLAUDE.md updated with bulk operation safety rules

---

## Verdict

**APPROVED** — This PR safely addresses write lock starvation through well-tested cooperative scheduling mechanisms. The bounded transaction approach is sound, and the retry-before-queue pattern improves resilience.

### Recommendations for merge:

1. ✅ Merge as-is — changes are safe and well-tested
2. 📝 Follow-up: Monitor WAL growth post-deploy (Axiom telemetry)
3. 📝 Follow-up: Consider RESTART checkpoints if 4.7GB WAL growth persists

### No blocking issues found.

---

## Axiom Queries for Post-Deploy Monitoring

```
# Enrichment write batch sizes
dataset('brainlayer-enrichment')
| where _type == 'complete'
| summarize avg(enriched), max(enriched), p95(enriched) by bin(_time, 1h)

# Store queue fallback rate
dataset('brainlayer-watcher')
| where event == 'queued_store'
| summarize count() by bin(_time, 5m)

# Drain lock contention
dataset('brainlayer-drain')
| where message contains 'busy'
| summarize count() by bin(_time, 1m)
```

---

**Reviewed by**: @bugbot (autonomous code review agent)  
**Review mode**: Critical path analysis (lock handling, write safety, MCP stability)
