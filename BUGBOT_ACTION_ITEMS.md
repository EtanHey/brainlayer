# Bugbot Action Items - BrainBar Helper Fast Profile PR

## 🔴 Critical (Fix Before Merge)

### 1. Add FTS Timeout Telemetry
**File**: `src/brainlayer/search_repo.py`  
**Line**: ~1593

**Current Code**:
```python
except apsw.InterruptError:
    if timeout_ms is not None:
        return []  # Silent degradation
    raise
```

**Add This Line**:
```python
except apsw.InterruptError:
    if timeout_ms is not None:
        search_profile.emit(profile_scope, "fts_timeout", profile_query_id, timeout_ms=timeout_ms)
        return []
    raise
```

### 2. Add Sparse Results Warning
**File**: `src/brainlayer/search_repo.py`  
**Line**: ~1199

**Current Code**:
```python
results = list(cursor.execute(query, params))
if len(results) < n_results and not brainbar_helper_fast_profile:
    retry_k = self._effective_knn_k(...)
    if retry_k > effective_k:
        results = list(cursor.execute(query, [query_bytes, retry_k] + filter_params))
results = results[:n_results]
```

**Add After Line 1190**:
```python
results = list(cursor.execute(query, params))
if len(results) < n_results and not brainbar_helper_fast_profile:
    retry_k = self._effective_knn_k(...)
    if retry_k > effective_k:
        results = list(cursor.execute(query, [query_bytes, retry_k] + filter_params))
elif brainbar_helper_fast_profile and len(results) < n_results * 0.5:
    from . import logger
    logger.warning(
        "BrainBar helper returned sparse results: got %d, requested %d (K=%d)",
        len(results), n_results, effective_k
    )
results = results[:n_results]
```

### 3. Document Cache Fragmentation
**File**: `src/brainlayer/search_repo.py`  
**Line**: ~1346 (before `_hybrid_cache_key` function)

**Add This Comment**:
```python
# ── hybrid_search result cache ───────────────────────────────────────────────
# Caches identical (store, query_text, filters) → results for 60s.
# ...existing comment...
#
# NOTE: BrainBar helper and MCP use separate cache entries even for identical
# queries because brainbar_helper_fast_profile differs in the cache key. This
# is intentional to avoid returning fast-profile results (which may have
# incomplete FTS or K=400 limits) to default MCP callers. However, this
# doubles cache pressure in mixed workloads. Monitor cache_hit_rate_* metrics.
```

---

## 🟡 High Priority (Post-Merge)

### 4. Add Progress Handler Logging
**File**: `src/brainlayer/search_repo.py`  
**Line**: ~1555

**Add After Line 1556** (`if connection is None:`):
```python
if connection is None:
    logger.debug(
        "FTS progress handler unavailable on cursor type=%s, timeout won't be enforced",
        type(cursor).__name__
    )
    return None
```

### 5. Make Constants Configurable
**File**: `src/brainlayer/search_repo.py`  
**Line**: ~45-48

**Change From**:
```python
_BRAINBAR_HELPER_FAST_K = 400
_BRAINBAR_HELPER_FTS_BUDGET_MS = 50.0
```

**Change To**:
```python
try:
    _BRAINBAR_HELPER_FAST_K = int(os.environ.get("BRAINLAYER_HELPER_FAST_K", "400"))
except (TypeError, ValueError):
    _BRAINBAR_HELPER_FAST_K = 400

try:
    _BRAINBAR_HELPER_FTS_BUDGET_MS = float(os.environ.get("BRAINLAYER_HELPER_FTS_BUDGET_MS", "50.0"))
except (TypeError, ValueError):
    _BRAINBAR_HELPER_FTS_BUDGET_MS = 50.0
```

### 6. Increase Cache Size
**File**: `src/brainlayer/search_repo.py`  
**Line**: ~35

**Change From**:
```python
_HYBRID_CACHE_MAX = 128  # max entries (LRU eviction)
```

**Change To**:
```python
_HYBRID_CACHE_MAX = 256  # max entries (LRU eviction) - increased to absorb helper/MCP cache split
```

---

## 🟢 Medium Priority (Future Work)

### 7. Add Quality Gate for Retry
**File**: `src/brainlayer/search_repo.py`  
**Line**: After line 1199

**Add This Logic**:
```python
# Quality recovery: if helper path returns very sparse results, retry once with 2x K
if brainbar_helper_fast_profile and len(results) < n_results * 0.5:
    recovery_k = min(effective_k * 2, _SQLITE_VEC_MAX_K)
    if recovery_k > effective_k:
        logger.info("Helper sparse results (%d < %d), retrying with K=%d", 
                    len(results), n_results, recovery_k)
        results = list(cursor.execute(query, [query_bytes, recovery_k] + filter_params))
        search_profile.emit(
            profile_scope,
            "helper_quality_recovery",
            profile_query_id,
            initial_k=effective_k,
            recovery_k=recovery_k,
            initial_count=len(results),
        )
```

### 8. Add Monitoring Metrics

**Create New File**: `src/brainlayer/metrics_helper.py`

```python
"""Helper-specific metrics for monitoring fast profile performance."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class HelperSearchMetrics:
    """Metrics for a single helper search request."""
    query_id: str
    query_text: str
    n_results_requested: int
    n_results_returned: int
    fts_timed_out: bool
    fts_timeout_ms: Optional[float]
    semantic_k: int
    latency_ms: float
    
    @property
    def is_sparse(self) -> bool:
        return self.n_results_returned < self.n_results_requested * 0.5
    
    @property
    def quality_score(self) -> float:
        """0-1 quality score based on result completeness."""
        if self.n_results_requested == 0:
            return 1.0
        return min(self.n_results_returned / self.n_results_requested, 1.0)
```

Then emit these metrics in `search_repo.py` at the end of `hybrid_search()` when `brainbar_helper_fast_profile=True`.

---

## 📊 Monitoring Setup (Production)

### Metrics to Add to Dashboards

```python
# In search_profile.py or telemetry.py

# FTS timeout rate
fts_timeout_rate = fts_timeout_count / total_helper_searches
# Alert if > 10%

# Sparse results rate  
sparse_results_rate = sparse_results_count / total_helper_searches
# Alert if > 5%

# Cache hit rates (separate tracking)
cache_hit_rate_helper = helper_cache_hits / helper_cache_lookups
cache_hit_rate_mcp = mcp_cache_hits / mcp_cache_lookups
# Alert if helper < 40% or MCP drops > 10%

# Quality comparison
avg_quality_helper = sum(quality_scores_helper) / len(quality_scores_helper)
avg_quality_mcp = sum(quality_scores_mcp) / len(quality_scores_mcp)
# Alert if helper < mcp * 0.8
```

### Alert Thresholds

```yaml
alerts:
  - name: BrainBarHelperFTSTimeoutHigh
    condition: fts_timeout_rate > 0.10
    severity: warning
    message: "BrainBar helper FTS timing out >10% of queries"
    
  - name: BrainBarHelperSparseResults
    condition: sparse_results_rate > 0.05
    severity: warning
    message: "BrainBar helper returning sparse results >5% of queries"
    
  - name: BrainBarHelperCacheHitLow
    condition: cache_hit_rate_helper < 0.40
    severity: info
    message: "BrainBar helper cache hit rate below 40%"
    
  - name: BrainBarHelperQualityDegradation
    condition: avg_quality_helper < avg_quality_mcp * 0.80
    severity: critical
    message: "BrainBar helper quality 20% worse than MCP"
```

---

## ✅ Testing Additions

### Test #1: FTS Timeout Recovery
**File**: `tests/test_brainbar_helper_fast_profile.py`

```python
def test_brainbar_fast_profile_emits_telemetry_when_fts_times_out():
    """Verify search_profile.emit() called when FTS timeout fires."""
    clear_hybrid_search_cache()
    
    class InterruptingCursor(RecordingCursor):
        def execute(self, sql, params=()):
            if "FROM chunks_fts" in sql:
                raise apsw.InterruptError("timeout fired")
            return super().execute(sql, params)
    
    cursor = InterruptingCursor()
    store = RecordingHybridStore(cursor)
    
    # Mock search_profile.emit
    emitted_events = []
    original_emit = search_profile.emit
    search_profile.emit = lambda *args, **kwargs: emitted_events.append((args, kwargs))
    
    try:
        results = store.hybrid_search(
            query_embedding=[0.1, 0.2, 0.3],
            query_text="timeout test",
            n_results=5,
            brainbar_helper_fast_profile=True,
        )
        
        # Should still return semantic results
        assert len(results["ids"][0]) > 0
        
        # Should emit fts_timeout event
        timeout_events = [e for e in emitted_events if "fts_timeout" in str(e)]
        assert len(timeout_events) > 0, "FTS timeout should emit telemetry"
    finally:
        search_profile.emit = original_emit
```

### Test #2: Sparse Results Warning
**File**: `tests/test_brainbar_helper_fast_profile.py`

```python
def test_brainbar_fast_profile_warns_when_results_sparse():
    """Verify warning logged when binary KNN returns < 50% of requested."""
    store = RecordingBinaryStore()
    store.cursor = RecordingCursor()  # Returns empty results
    
    with pytest.warns(UserWarning, match="sparse results"):
        results = store._binary_search(
            query_embedding=[0.1, 0.2, 0.3],
            n_results=10,  # Request 10
            brainbar_helper_fast_profile=True,
        )
        # Returns 0, which is < 10 * 0.5 = 5, should warn
```

### Test #3: Cache Fragmentation
**File**: `tests/test_brainbar_helper_fast_profile.py`

```python
def test_helper_and_mcp_use_different_cache_keys():
    """Verify helper and MCP paths don't share cache entries."""
    clear_hybrid_search_cache()
    
    cursor = RecordingCursor(rows_by_table={"chunks_fts": [_fts_row("test-1")]})
    store = RecordingHybridStore(cursor)
    
    # First call with fast_profile=True
    store.hybrid_search(
        query_embedding=[0.1, 0.2, 0.3],
        query_text="cache test",
        n_results=5,
        brainbar_helper_fast_profile=True,
    )
    
    # Reset cursor to detect cache miss
    cursor.calls = []
    
    # Second call with fast_profile=False should MISS cache
    store.hybrid_search(
        query_embedding=[0.1, 0.2, 0.3],
        query_text="cache test",
        n_results=5,
        brainbar_helper_fast_profile=False,
    )
    
    # If cache was shared, cursor.calls would be empty (cache hit)
    # Since cache is split, should have new calls (cache miss)
    assert len(cursor.calls) > 0, "Helper and MCP should use separate cache entries"
```

---

## 🎯 Success Criteria

**Before marking this PR as complete**:

- [x] All tests pass (2355 passed, 49 skipped, 5 xfailed) ✅
- [ ] FTS timeout telemetry added (Critical #1)
- [ ] Sparse results warning added (Critical #2)
- [ ] Cache fragmentation documented (Critical #3)
- [ ] New tests added for timeout, sparse, cache (Testing #1-3)
- [ ] Monitoring metrics defined (Monitoring section)
- [ ] Alert thresholds configured (Alert section)

**Post-merge within 1 week**:

- [ ] Monitor FTS timeout rate (<10% target)
- [ ] Monitor sparse results rate (<5% target)
- [ ] Monitor cache hit rates (helper >40%, MCP no drop)
- [ ] Compare helper vs. MCP quality on sample queries
- [ ] Verify latency stays <800ms p99
- [ ] Collect user feedback on BrainBar search quality

**If metrics cross thresholds**:

1. **FTS timeout >10%**: Increase `BRAINLAYER_HELPER_FTS_BUDGET_MS` to 100ms
2. **Sparse results >5%**: Add quality gate (Priority #7)
3. **Cache hit <40%**: Increase `_HYBRID_CACHE_MAX` to 512
4. **Quality gap >20%**: Disable helper route, investigate deeper

---

**Priority Order**:
1. Complete Critical #1-3 (4 lines total) ✅ **REQUIRED BEFORE MERGE**
2. Add Test #1-3 (90 lines) ⚡ **HIGHLY RECOMMENDED**
3. Configure Monitoring (30 min setup) 📊 **DAY 1 POST-MERGE**
4. Implement Priority #4-8 (100 lines) 🔧 **WEEK 1 POST-MERGE**
