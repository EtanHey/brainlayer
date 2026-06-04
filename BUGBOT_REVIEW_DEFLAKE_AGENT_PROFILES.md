# Bugbot Review: test: deflake agent profile recency intent

**PR:** fix/deflake-agent-profiles-recency  
**Commit:** aeea669ad4a3ad1901239dcfbc61713fcca478ea  
**Review Date:** 2026-06-04  
**Risk Level:** ✅ **LOW** (test-only change)

---

## Summary

Fixes time-dependent flakiness in `test_hybrid_search_agent_profile_scales_recency_intent_neutral_point` by replacing hard-coded 2026 timestamps with UTC-now-relative values. Clean, focused fix with no runtime behavior impact.

---

## Changes Analysis

### What Changed

```diff
+ from datetime import datetime, timedelta, timezone

+ def _created_at_days_ago(days: int) -> str:
+     return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds").replace("+00:00", "Z")

- created_at="2026-01-01T00:00:00Z",  # old chunk
+ created_at=_created_at_days_ago(150),

- created_at="2026-05-28T00:00:00Z",  # recent chunk  
+ created_at=_created_at_days_ago(2),
```

### Why This Matters

The test validates recency-intent behavior in hybrid search. The recency scoring uses a ~7-day window. Fixed dates like "2026-01-01" and "2026-05-28" would cause the test to fail or behave unexpectedly once real time advances past those dates.

**Solution:** Use relative offsets (150 days ago vs 2 days ago) to ensure:
- Recent chunk (2 days) **always** stays inside the ~7-day window
- Old chunk (150 days) **always** stays far outside the window  
- Test remains deterministic regardless of when it runs

---

## Review Findings

### ✅ Core Functionality Preserved

- Test logic **unchanged** — still validates `recency_intent` scaling behavior
- Same assertions and expectations
- Same embedding similarity setup (`_embed(0.1)` vs `_embed(0.3)`)
- Same importance values (both 1.0)

### ✅ Implementation Quality

**Helper function is clean and correct:**
```python
def _created_at_days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds").replace("+00:00", "Z")
```

✓ Proper UTC timezone handling (`timezone.utc`)  
✓ Consistent ISO format with Z suffix (matches existing format)  
✓ Uses `timespec="seconds"` for precision consistency  
✓ Single-purpose, reusable helper

**Offset choices are sound:**
- `_created_at_days_ago(150)` → ~5 months old (well outside 7-day window)
- `_created_at_days_ago(2)` → 2 days old (inside 7-day window)
- Large gap ensures clear separation for recency scoring

### ✅ Test Robustness

**Before (flaky):**
- Hard-coded `2026-01-01` and `2026-05-28`
- Would fail/change behavior after May 2026
- Not future-proof

**After (robust):**
- Always relative to current time
- Maintains same relative positions (old vs recent)
- Works indefinitely as calendar advances

### ✅ Style & Formatting

```bash
$ ruff check tests/test_agent_profiles.py
All checks passed!

$ ruff format --check tests/test_agent_profiles.py  
1 file already formatted
```

Clean imports follow existing patterns in the file.

### ✅ Test Coverage

```bash
$ pytest tests/test_agent_profiles.py::test_hybrid_search_agent_profile_scales_recency_intent_neutral_point -v
PASSED [100%]

$ pytest tests/test_agent_profiles.py -v
10 passed in 2.42s
```

All agent profile tests pass, including:
- Migration tests
- CLI roundtrip tests  
- Profile validation tests
- Hybrid search with agent profiles
- Recency intent scaling (the fixed test)

---

## Risk Assessment

**Risk Level: LOW ✅**

| Dimension | Assessment | Rationale |
|-----------|-----------|-----------|
| **Runtime Impact** | None | Test-only change, no production code modified |
| **Database Schema** | None | No schema changes, no migrations |
| **MCP Tools** | None | No changes to MCP tool contracts |
| **Search Behavior** | None | No changes to `search_repo.py`, `vector_store.py`, or `agent_profiles.py` |
| **Concurrency** | N/A | Test uses `tmp_path` fixture (isolated per-test DB) |
| **Backwards Compat** | N/A | Test-only change |

**Files modified:**
- `tests/test_agent_profiles.py` (test-only)

**Files NOT modified:**
- `src/brainlayer/search_repo.py`
- `src/brainlayer/vector_store.py`  
- `src/brainlayer/agent_profiles.py`
- Any MCP server code
- Any CLI code

---

## Test Plan Verification

From PR description:

- [x] ✅ `pytest tests/test_agent_profiles.py::test_hybrid_search_agent_profile_scales_recency_intent_neutral_point -q` → **PASSED**
- [x] ✅ Future-date probe (test would work in 2035) → **Confirmed** (uses relative dates)
- [x] ✅ `pytest tests/test_agent_profiles.py -q` → **10 passed in 2.42s**
- [x] ✅ `ruff format --check tests/test_agent_profiles.py` → **1 file already formatted**
- [x] ✅ `ruff check tests/test_agent_profiles.py` → **All checks passed!**
- [ ] ⏳ `pytest` (full suite) → Deferred due to env setup cost (929 tests)
- [ ] ⏳ Pre-push BrainLayer test gate → Depends on full suite

---

## Recommendations

### ✅ Ready to Merge

This PR is **clean, focused, and low-risk**. No issues found.

**Optional follow-ups (not blockers):**

1. **Other time-dependent tests:** Search for other hard-coded dates in test suite:
   ```bash
   grep -r "2026-" tests/ | grep -v ".pyc"
   ```
   May find similar flakiness issues in other tests.

2. **Documentation:** Consider adding a comment in the test explaining the offset choices:
   ```python
   # Use 150 days (well outside 7-day recency window) vs 2 days (inside window)
   created_at=_created_at_days_ago(150),
   ```
   But this is minor — the offsets are self-explanatory.

---

## Conclusion

**LGTM ✅**

This is a textbook example of a clean test fix:
- Identifies root cause (hard-coded dates)
- Implements minimal, focused solution (relative dates)
- Preserves test intent and coverage
- Adds no complexity
- Passes all checks

**No blockers. Ready for merge.**

---

**Reviewed by:** Bugbot (via Cloud Agent)  
**Review method:** Code inspection + test execution  
**Test results:** 10/10 agent_profiles tests passing  
**Style checks:** ruff check + ruff format both passing
