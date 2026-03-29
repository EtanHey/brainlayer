# @bugbot Review Summary

**PR #135:** feat: implement all 6 BrainBar stub tools  
**Branch:** `feat/brainbar-implement-stub-tools`  
**Reviewed:** 2026-03-29  
**Status:** ⚠️ **DO NOT MERGE**

---

## 🔴 CRITICAL FINDINGS

Found **7 critical bugs** that will cause runtime failures, data corruption, or security issues.

### Top 3 Blockers:

1. **brain_update Schema Mismatch** (Lines 479-490 in MCPRouter.swift)
   - Schema requires `action` parameter but implementation ignores it
   - Schema missing `importance` and `tags` properties that implementation uses
   - Will fail when Claude tries to call the tool

2. **brain_expand Incorrect Logic** (Lines 919-920 in BrainDatabase.swift)
   - Uses `before * 10` multiplier for rowid range (arbitrary magic number)
   - Fails in sparse rowid spaces, returns wrong context chunks
   - No test coverage for this edge case

3. **brain_entity SQL Injection** (Line 994 in BrainDatabase.swift)
   - LIKE query doesn't escape `%` and `_` wildcards
   - User input like "test_entity" will match "testXentity"
   - Security vulnerability

---

## 📊 Bug Breakdown

| Severity | Count | Examples |
|----------|-------|----------|
| Critical | 7 | Schema mismatch, SQL injection, incorrect logic |
| Moderate | 5 | Performance, error handling, data loss |
| Low | 0 | - |

---

## 📝 Full Report

See [`BUG_REPORT.md`](./BUG_REPORT.md) for:
- Detailed analysis of all 12 issues
- Code examples showing the bugs
- Recommended fixes with code snippets
- Test coverage gaps
- Priority-ordered fix list

---

## ✅ What's Working

- Consistent error handling with ToolError enum
- Good retry logic for SQLITE_BUSY (3 retries)
- Proper transaction handling
- FTS5 integration is solid
- Basic test coverage for happy paths

---

## 🔧 Required Actions

**Before merging:**
1. Fix brain_update schema (add missing properties, implement all actions)
2. Fix brain_expand rowid calculation (use proper SQL subqueries)
3. Fix brain_entity SQL injection (escape LIKE wildcards)
4. Add error handling to brain_digest regex compilation
5. Add truncation warning to brain_digest results
6. Fix brain_tags case preservation
7. Add database indexes for kg_entities and kg_relations

**Estimated effort:** 2-4 hours to fix all critical bugs

---

## 📎 Files Changed

- `BUG_REPORT.md` — Comprehensive bug analysis (committed)
- `REVIEW_SUMMARY.md` — This summary (committed)

Both files have been committed to the branch and pushed to remote.

---

**Next Steps:**
1. Review the detailed bug report in `BUG_REPORT.md`
2. Fix critical bugs #1-6
3. Add test coverage for edge cases
4. Re-run tests
5. Request another review

**Note:** Swift environment was not available in this cloud agent, so analysis is based on static code review only. Bugs were identified through careful examination of the implementation against the schema definitions and documented requirements in CLAUDE.md.
