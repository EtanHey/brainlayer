# Bugbot Re-Review: Critical Issues Identified

**Review Date:** 2026-04-30 (Final)  
**Branch:** `fix/fts-recall-all-three`  
**Latest Commit:** `546f7b2`  
**Status:** ⚠️ **APPROVE WITH CRITICAL FIXES REQUIRED**

---

## Executive Summary

Macroscope and Codex have identified **3 critical correctness issues** that must be addressed before merge. These issues were not caught in my initial review and represent genuine retrieval correctness bugs.

---

## Critical Issue #1: Trigram-Only Results Skip Filters 🔴

**Severity:** P0 - Critical Correctness Bug  
**Source:** Macroscope  
**Location:** `search_repo.py:1141`

### Problem

Post-RRF filter guard only checks `fts_rank is not None` but ignores `trigram_rank`. When a chunk appears **only** in trigram results (not in main FTS), it bypasses all post-RRF filters.

**Current Code:**
```python
if fts_rank is not None and sem_entry is None:
    if source_filter and meta.get("source") != source_filter:
        continue
    # ... other filters
```

**Bug:** If `fts_rank=None` and `trigram_rank=42`, the chunk passes through unfiltered.

### Impact

- Trigram-only hits bypass `source_filter`, `project_filter`, `content_type_filter`, `sender_filter`, `language_filter`
- Cross-project data leakage possible
- Filter contracts violated for substring identifier matches

### Fix Required

```python
# Change condition to include trigram_rank
if (fts_rank is not None or trigram_rank is not None) and sem_entry is None:
    if source_filter and meta.get("source") != source_filter:
        continue
    # ... rest of filters
```

**Verification:** Add test case:
```python
# Insert chunk with trigram-only match + wrong source
# Query with source_filter
# Assert: chunk does NOT appear in results
```

---

## Critical Issue #2: Exact Chunk-ID Bypass Ignores Filters 🔴

**Severity:** P1 - Critical Scope Violation  
**Source:** Codex  
**Location:** `search_handler.py:389-391`

### Problem

`_exact_chunk_lookup_result()` only checks lifecycle state but **ignores all other filters** passed to `_brain_search()`:
- `project` (including auto-scoped project from `resolve_project_scope()`)
- `source`, `tag`, `intent`, `importance_min`
- `date_from`, `date_to`, `sentiment`
- `entity_id`, `source_filter`, `correction_category`

**Current Flow:**
```python
# Line 389-391
exact_chunk_hit = _exact_chunk_lookup_result(query, store, detail)
if exact_chunk_hit is not None:
    return exact_chunk_hit  # Returns without checking ANY filters!
```

### Impact

- **Cross-project data leakage:** User with project scope can access chunks from other projects via direct ID
- **Filter bypass:** All MCP tool filters are ignored for chunk-id queries
- **Security implication:** Breaks project isolation guarantees

### Fix Required

**Option A:** Pass filters to `_exact_chunk_lookup_result()` and verify chunk matches

```python
def _exact_chunk_lookup_result(
    query: str, 
    store: Any, 
    detail: str,
    project: str | None = None,
    source: str | None = None,
    # ... all other filters
) -> tuple[list[TextContent], dict] | None:
    # ... existing lookup ...
    
    # Add filter checks after lookup
    if project and chunk.get("project") != project:
        return None
    if source and chunk.get("source") != source:
        return None
    # ... etc for all filters
```

**Option B:** Disable exact bypass when ANY filter is active

```python
# Only bypass if no filters active
has_filters = any([project, source, tag, intent, ...])
if not has_filters:
    exact_chunk_hit = _exact_chunk_lookup_result(query, store, detail)
    if exact_chunk_hit is not None:
        return exact_chunk_hit
```

**Recommendation:** Implement Option A for consistency, Option B as fallback if implementation is complex.

---

## Critical Issue #3: Alias Expansion Changes FTS Semantics 🟡

**Severity:** P2 - Recall Regression Risk  
**Source:** Codex  
**Location:** `search_handler.py:131`

### Problem

`_expanded_fts_query()` wraps each variant in `_quote_fts_phrase()`, converting token-level matching to **phrase matching**. For multi-word queries, this breaks FTS5 semantics.

**Example:**
- Original query: `"brain search layer"` → FTS5: `brain AND search AND layer` (token-level)
- With expansion: `"Hershkovitz"` variant → `"brain search layer" OR "Hershkovitz"` (phrase match)
- Problem: Now requires exact phrase `"brain search layer"`, dropping valid hits with non-adjacent terms

### Impact

- **Recall regression** for multi-word queries with variants
- Valid chunks with scattered terms (e.g., `"brain ... search ... layer"`) are dropped
- Contradicts FTS5 token-level matching behavior

### Fix Required

Build OR expression from **escaped tokens**, not quoted phrases:

```python
def _expanded_fts_query(query: str, store: Any) -> str | None:
    variants = _lexical_defense_variants(query)
    seen = {value.casefold().strip() for value in variants if value.strip()}
    for variant in _kg_alias_variants(query, store):
        dedupe_key = variant.casefold().strip()
        if dedupe_key and dedupe_key not in seen:
            seen.add(dedupe_key)
            variants.append(variant)

    if len(variants) <= 1:
        return None
    
    # FIX: Escape each variant instead of phrase-quoting
    from ._helpers import _escape_fts5_query
    return " OR ".join(_escape_fts5_query(variant) for variant in variants)
```

**Verification:** Test multi-word query with variant expansion against chunk with scattered terms.

---

## Revised Verdict

**⚠️ APPROVE WITH MANDATORY FIXES**

The PR delivers valuable recall improvements, but the 3 critical issues above **must be fixed** before production deployment:

1. **MUST FIX:** Trigram-only filter bypass (P0 correctness)
2. **MUST FIX:** Exact chunk-ID filter bypass (P1 security/scope)
3. **SHOULD FIX:** Alias expansion phrase matching (P2 recall)

### Recommended Fix Order

1. **Trigram filter guard** (5 min) - Add `trigram_rank` to condition
2. **Exact bypass filters** (15 min) - Implement Option A or B
3. **Alias expansion semantics** (10 min) - Replace `_quote_fts_phrase` with `_escape_fts5_query`

### Testing Checklist Before Merge

- [ ] Add test: Trigram-only hit with wrong source_filter → 0 results
- [ ] Add test: Exact chunk-ID with wrong project → 0 results  
- [ ] Add test: Multi-word query with expansion → scattered terms still match
- [ ] Verify: All existing tests still pass
- [ ] Verify: Behavioral receipts still work

---

## Updated Risk Assessment

**Pre-Fix:** 🔴 High Risk
- Cross-project data leakage via exact bypass
- Filter contracts violated for trigram hits
- Potential recall regression on multi-word queries

**Post-Fix:** 🟢 Low Risk
- All retrieval paths respect filters consistently
- FTS semantics preserved
- Production-ready with high confidence

---

## Approval

**⚠️ CONDITIONALLY APPROVED**

Fix the 3 critical issues above, verify with tests, then **merge with confidence**.

The underlying architecture is sound. These are fixable edge cases in the filter application logic, not fundamental design flaws.

---

**Reviewer:** @bugbot  
**Date:** 2026-04-30  
**Priority:** P0 - Block merge until fixed
