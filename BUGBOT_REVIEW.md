# Bugbot Review: PR feat/p2b-fixture-corpus

## Summary
This PR adds a comprehensive test fixture for FTS5 ranking regression detection. The hermetic test fix addresses a **critical network dependency issue** found by @codex.

---

## 🐛 Critical Bugs Fixed

### 1. **Network Dependency in FTS Test (CRITICAL)**
**Severity: HIGH** ⚠️ **Found by @codex**  
**Location**: `tests/stale_index_query.test.ts`

**Issue**: The test originally used `uvx --from sqlite-utils` which requires network access to PyPI on every run, breaking in offline/CI environments.

**Fix**: ✅ **FIXED** - Refactored to use native Bun SQLite (`db.query()`) for FTS assertions. The FTS ranking test is now fully hermetic. The embedding drift test still legitimately requires `uv run python3` for the embedding model.

---

### 2. **Missing Dependency Check**
**Severity: MEDIUM**  
**Location**: `scripts/generate-fixtures.sh`

**Issue**: Script assumes `uvx` is available without checking.

**Fix**: ✅ **FIXED** - Added preflight check with helpful error message.

---

### 3. **Missing `.gitattributes`**
**Severity: LOW**

**Fix**: ✅ **FIXED** - Added entry to mark fixtures as linguist-generated for better GitHub stats.

---

## ⚠️ Design Notes

### Cosine Similarity Threshold
The fixture uses 0.999 threshold. Consider relaxing to 0.995 if you see false positives across different CPU architectures (x86 vs ARM) or BLAS implementations.

### Test Timeout
Current 120s timeout may be tight on slow CI. Consider making it configurable via env var if needed.

---

## ✅ What's Excellent

1. **Hermetic FTS Test**: Core ranking assertions now require zero network access
2. **Fixture Provenance**: README and embedded metadata make regeneration transparent
3. **Dual Language Coverage**: Bun + pytest ensure the fixture works across ecosystems
4. **Control Document**: `orchard-ml-004` is a proper negative control

---

## Test Results

- ✅ **pytest**: `tests/test_stale_index_fixture.py` passes
- ✅ **Bun FTS test**: Now hermetic (no network required)
- ⚠️ **Bun embedding test**: Requires `uv` (acceptable - needs embedding model)

---

## Verdict

**✅ READY TO MERGE** - Critical network dependency fixed, test infrastructure is solid.

---

**Reviewed by**: Bugbot (Claude Agent) + @codex  
**Date**: 2026-04-27  
**Commits**: ea23dcf (+ bugbot fixes)
