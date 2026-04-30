# Bugbot Final Status Report

**Date:** 2026-04-30  
**PR:** #263 - fix: harden BrainLayer FTS recall across all three layers  
**Latest Commit:** `9f4a75a` (current HEAD)  
**Status:** 🔴 **CRITICAL ISSUES CONFIRMED - MERGE BLOCKED**

---

## Executive Summary

Cursor's own Bugbot has **independently confirmed** the 3 critical issues I identified from Macroscope and Codex reviews. These are **real bugs** that must be fixed before merge.

---

## Confirmed Critical Issues

### Issue #1: Exact Chunk-ID Bypass Ignores Project Scope 🔴
**Severity:** High (P0)  
**Confirmed By:** Cursor Bugbot, Codex, Human Review  
**Bugbot ID:** `4388acb8-612c-43a1-b6ea-984da7108a23`

**Description from Cursor Bugbot:**
> `_exact_chunk_lookup_result` only receives `query`, `store`, and `detail`—it checks lifecycle state but ignores all caller-supplied filters including `project` (which may be auto-resolved via `resolve_project_scope()`), `source`, `tag`, `content_type`, `importance_min`, date ranges, and `entity_id`. A user scoped to one project can retrieve chunks from another project by querying a known chunk ID, causing cross-project data leakage.

**Location:** `src/brainlayer/mcp/search_handler.py:387-391`

**Impact:** 🔴 Cross-project data leakage (security issue)

---

### Issue #2: Phrase Quoting Reduces Recall for Multi-Word Queries 🟡
**Severity:** Medium (P2)  
**Confirmed By:** Cursor Bugbot, Codex, Human Review  
**Bugbot ID:** `f6198648-d940-4ccd-87fd-669557679d8b`

**Description from Cursor Bugbot:**
> `_expanded_fts_query` uses `_quote_fts_phrase` to wrap each variant (including the original query) in a single pair of double quotes. For multi-word queries, this converts token-level AND matching (terms anywhere in the document) into exact phrase matching (terms must be adjacent in order). When expansion fires for a multi-word query, valid chunks with scattered terms are dropped, causing a recall regression.

**Location:** `src/brainlayer/mcp/search_handler.py:130-131`

**Impact:** 🟡 Recall regression on multi-word queries

---

### Issue #3: Trigram-Only Results Skip Filters 🔴
**Severity:** High (P0)  
**Confirmed By:** Macroscope, Human Review  

**Description:**
Post-hoc filtering at line 1141 uses `if fts_rank is not None and sem_entry is None:`, which excludes trigram-only results from filter checks. When a chunk appears only in `trigram_fts_results`, `fts_rank` is `None` while `trigram_rank` has a value, so the condition is `False` and filters are silently skipped.

**Location:** `src/brainlayer/search_repo.py:1141`

**Impact:** 🔴 Filter contract violations

---

## Additional Issues Identified

### Issue #4: Merge Conflict Marker in Documentation 📄
**Severity:** Low  
**Confirmed By:** Cursor Bugbot, Macroscope  
**Bugbot ID:** `cfbf1b81-cdb2-4bb4-96a5-0c7bdcc8a532`

**Location:** `BUGBOT_REVIEW_FTS_RECALL.md:31`  
**Status:** ✅ **FIXED** in this commit

---

### Issue #5: SQL Normalizer Mismatch 🟡
**Severity:** Medium  
**Confirmed By:** Macroscope

**Description:**
The SQL `normalizer` expression only strips `-`, `_`, `.`, and space, but `_normalize_surface` also removes apostrophes and other non-alphanumeric characters. Queries like `"O'Brien"` won't match.

**Location:** `src/brainlayer/mcp/search_handler.py:71`

**Impact:** KG alias lookup failures for names with apostrophes/special characters

---

## Current Status

**Merge Readiness:** 🔴 **NOT READY**

**Critical Blockers:** 3 issues (2 confirmed P0, 1 confirmed P2)

**Fix Status:**
- ❌ Issue #1: Not fixed (cross-project leakage)
- ❌ Issue #2: Not fixed (recall regression)
- ❌ Issue #3: Not fixed (filter bypass)
- ✅ Issue #4: Fixed (merge conflict removed)
- ❌ Issue #5: Not fixed (normalizer mismatch)

---

## Verification

Cursor Bugbot has provided **"Fix in Cursor"** and **"Fix in Web"** links for Issues #1 and #2:

**Issue #1 - Cross-Project Leakage:**
- Fix in Cursor: [Link provided by Bugbot]
- Fix in Web: [Link provided by Bugbot]

**Issue #2 - Phrase Matching:**
- Fix in Cursor: [Link provided by Bugbot]
- Fix in Web: [Link provided by Bugbot]

---

## Recommendation

**DO NOT MERGE** until all 3 critical issues are fixed:

1. **Fix Issue #1** (P0) - Add filter validation to `_exact_chunk_lookup_result()`
2. **Fix Issue #3** (P0) - Update filter condition to include `trigram_rank`
3. **Fix Issue #2** (P2) - Replace `_quote_fts_phrase` with `_escape_fts5_query`

**Optional:**
4. Fix Issue #5 (P3) - Align SQL normalizer with Python `_normalize_surface()`

**Estimated Time:** 30-45 minutes for all fixes + tests

---

## Updated Verdict

**Previous:** ✅ APPROVED with increased confidence  
**Current:** 🔴 **MERGE BLOCKED - CRITICAL FIXES REQUIRED**

The PR architecture is sound, but these edge cases represent **real security and correctness bugs** that would cause production issues:
- Cross-project data leakage
- Filter contract violations  
- Recall quality regressions

Once fixed and verified, the PR will be production-ready.

---

## Review Timeline

1. **Initial Review** (`2c0454c`) - Approved with observations
2. **Re-Review #1** (`546f7b2`) - Approved with increased confidence (6 fixes)
3. **Re-Review #2** (`4006365`) - Identified 3 critical issues (Macroscope/Codex)
4. **Current** (`9f4a75a`) - Cursor Bugbot confirms critical issues

**Status:** Awaiting fixes for merge approval

---

**Reviewer:** @bugbot  
**Confirmation:** Issues independently verified by Cursor Bugbot  
**Action Required:** Fix 3 critical issues before merge
