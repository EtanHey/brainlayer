# Bugbot Review: fix/hooks-fail-loud-db-error

**PR**: fix: make BrainLayer hook DB failures loud  
**Commit**: a756baa41bf76dca03b19ad7f5fc89d83af64d32  
**Review Date**: 2026-06-04  
**Reviewer**: Bugbot (Cloud Agent)

---

## Summary

This PR makes DB failures in BrainLayer hooks visible to users instead of silently failing. When SessionStart or UserPromptSubmit hooks cannot access the BrainLayer database (missing path, connect failure, or query error), they now emit a stable `⚠️ DEGRADED: BrainLayer` notice to stdout explaining the session runs without long-term memory.

**Risk Level**: **LOW**  
- Hook stdout and telemetry mode changes only
- No DB writes, schema changes, or concurrency modifications
- Comprehensive test coverage (39 tests, all passing)
- Exit code 0 preserved in all cases (keeps session alive)

---

## Implementation Review

### 1. Core Changes

Both hooks (`brainlayer-session-start.py` and `brainlayer-prompt-search.py`) now include:

```python
DEGRADED_PREFIX = "⚠️ DEGRADED: BrainLayer"

def degraded_notice(reason):
    return f"{DEGRADED_PREFIX} memory unavailable this session ({reason}) - operating without long-term memory."

def emit_degraded(reason):
    print(degraded_notice(reason))
```

### 2. Failure Modes Handled

#### SessionStart Hook
1. **Missing DB path** (line 231): Emits degraded notice, exits 0
2. **Connection error** (line 240): Emits degraded notice, exits 0  
3. **Query error** (line 297): Appends degraded notice to lines, still exits 0

#### UserPromptSubmit Hook
1. **Missing DB path** (line 1123): Emits degraded notice, records telemetry mode="degraded"
2. **Connection error** (line 1143): Emits degraded notice, records telemetry mode="degraded"
3. **Query error** (line 1243): Appends degraded notice, sets telemetry_mode="degraded"

### 3. Behavioral Contracts Preserved

✅ **Silent skip paths unchanged**:
- `BRAINLAYER_HOOKS_DISABLED=1` → no output, exit 0
- `CLAUDE_NON_INTERACTIVE=1` → no output, exit 0
- Command/casual prompts → no output, exit 0

✅ **Exit code 0 in all cases** → session continues

✅ **Telemetry mode tracking**:
- `skip` → intentional skip (disabled, non-interactive, casual)
- `degraded` → DB failure (new mode)
- `normal`, `deep`, `entity` → successful injection

---

## Test Coverage Analysis

**Test file**: `tests/test_conditional_hooks.py`  
**Total tests**: 39 (all passing)

### SessionStart Hook Tests (11 tests)
- ✅ Default activation
- ✅ Disabled env var skip
- ✅ Non-interactive skip
- ✅ Light mode
- ✅ Hebrew style guidance
- ✅ **Degraded notice on connect error** (NEW)
- ✅ **Degraded notice on query error** (NEW)
- ✅ **Degraded notice on missing DB path** (NEW)
- ✅ **Silent behavior when disabled** (NEW)

### UserPromptSubmit Hook Tests (28 tests)
- ✅ Entity detection (6 tests)
- ✅ Injection event telemetry (4 tests)
- ✅ **Degraded notice on connect error** (NEW)
- ✅ **Degraded notice on query error** (NEW)
- ✅ **Degraded notice on missing DB path** (NEW)
- ✅ **Silent behavior when disabled** (NEW)
- ✅ Search-before-assume warnings (2 tests)
- ✅ Correction detection (2 tests)

### Key Test Verifications
1. **Exit code 0** verified in all degraded scenarios
2. **Stdout content** verified to contain `⚠️ DEGRADED: BrainLayer`
3. **Telemetry mode** verified as `degraded` for DB failures
4. **PRAGMA execution** verified (`busy_timeout=1000`, `query_only=true`)
5. **Connection closure** verified on error paths

---

## Critical Path Concerns

### ✅ Retrieval Correctness
- No changes to search logic, ranking, or result selection
- Degraded mode simply skips injection instead of silently failing
- Test coverage confirms no change to working retrieval paths

### ✅ Write Safety
- **No DB writes in hooks** (read-only mode via `query_only=true`)
- Injection event telemetry writes handled by best-effort try/except
- Telemetry mode now records `degraded` state for monitoring

### ✅ MCP Stability
- Hooks are pre-MCP execution (Claude Desktop hook contract)
- No MCP tool changes
- No impact on MCP server runtime

### ✅ Concurrency
- **No concurrency changes**
- Hooks remain stateless, single-threaded
- Each hook execution uses its own read-only SQLite connection
- No new lock files or coordination mechanisms

---

## Risk Analysis

### Low Risk Items (Approved)
1. **Stdout changes**: Adding degraded notices is low-risk observability
2. **Telemetry mode**: New `degraded` value aids monitoring
3. **Test coverage**: Comprehensive regression coverage (39 tests)
4. **Exit code**: Always 0 (session continues)

### No Risk Items
1. **DB schema**: No schema changes
2. **Concurrency**: No lock changes
3. **Hook timing**: No performance impact (still fails fast)
4. **Backward compat**: Existing behavior unchanged (silent skip paths preserved)

---

## Code Quality

### ✅ Strengths
1. **DRY**: `degraded_notice()` helper avoids duplication
2. **Consistent**: Both hooks use same notice format
3. **Testable**: Pure functions, easy to mock
4. **Clear intent**: `DEGRADED_PREFIX` makes notice scannable

### ⚠️ Minor Observations (Non-blocking)
1. **Duplicate helpers**: `degraded_notice()` defined in both hooks → acceptable for standalone scripts
2. **Reason strings**: "DB not found" vs "DB error" → clear and concise
3. **No i18n**: Warning message is English-only → acceptable for system notices

---

## Syntax & Style Compliance

```bash
$ ruff check hooks/brainlayer-session-start.py hooks/brainlayer-prompt-search.py tests/test_conditional_hooks.py
All checks passed!

$ python3 -m py_compile hooks/brainlayer-session-start.py hooks/brainlayer-prompt-search.py
(no errors)

$ pytest tests/test_conditional_hooks.py -v
39 passed in 1.20s
```

---

## Deployment Notes

✅ **Production hook sync required**  
The PR description correctly notes that production hook copies live at `~/.claude/hooks/` and need re-syncing post-merge. This PR only updates the repo copies.

**Recommended deployment steps**:
1. Merge PR to main
2. Sync hooks to `~/.claude/hooks/` on user machines:
   ```bash
   cp hooks/brainlayer-session-start.py ~/.claude/hooks/
   cp hooks/brainlayer-prompt-search.py ~/.claude/hooks/
   ```
3. Monitor telemetry for `mode="degraded"` events in `injection_events` table

---

## Regression Risk Assessment

### Pre-Existing Failure (Acknowledged)
- `test_agent_profiles.py::test_hybrid_search_agent_profile_scales_recency_intent_neutral_point` fails on clean origin/main 7a4ea3cc
- **NOT introduced by this change** (verified by PR author on detached origin/main worktree)
- No action required for this PR

### New Regressions
**None detected** after 39 tests passed.

---

## Recommendations

### ✅ Approve for Merge
This PR is **low-risk, well-tested, and correctly scoped**.

**Approval conditions met**:
1. ✅ No DB writes or schema changes
2. ✅ No concurrency changes
3. ✅ Comprehensive test coverage (39 tests)
4. ✅ Exit code 0 preserved (session continues)
5. ✅ Silent skip paths unchanged (no false positives)
6. ✅ Syntax/style checks pass
7. ✅ Pre-existing baseline failure acknowledged and unrelated

### 📋 Post-Merge Actions
1. **Sync production hooks** (see Deployment Notes)
2. **Monitor telemetry**: Watch for `mode="degraded"` spike in `injection_events`
3. **User education**: Consider mentioning degraded mode in BrainLayer docs

### 🔍 Future Enhancements (Out of Scope)
1. Add retry logic for transient `SQLITE_BUSY` errors
2. Expose hook health metrics in BrainBar UI
3. Add structured logging (JSONL) for hook degraded events

---

## Final Verdict

**APPROVED** ✅

This PR successfully makes BrainLayer hook DB failures visible without introducing risk. The implementation is clean, well-tested, and preserves all existing behavioral contracts.

**Merge recommendation**: **Approve and merge**  
**Follow-up review needed**: **No**

---

**Bugbot Sign-Off**  
Reviewed commit: `a756baa41bf76dca03b19ad7f5fc89d83af64d32`  
Review profile: BrainLayer Critical Path Review (AGENTS.md)  
Risk: **LOW** | Tests: **39/39 PASSED** | Concurrency: **NO CHANGES** | DB: **NO CHANGES**
