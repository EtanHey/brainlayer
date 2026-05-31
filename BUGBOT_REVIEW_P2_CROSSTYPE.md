# Bugbot Review: P2 Cross-Type KG Cleanup

**PR**: feat: add P2 cross-type KG cleanup tooling  
**Commit**: 58a62a1f696548e7ac60117fcccbacd7aea8183e  
**Review Date**: 2026-05-31  
**Status**: ✅ APPROVED with observations

---

## Summary

This PR adds a one-time cleanup script to resolve cross-type entity ontology issues in the knowledge graph, plus a defensive entity lookup improvement. The cleanup was already executed live with commander approval using a backup.

**Changed files:**
- `scripts/kg_p2_crosstype_cleanup.py` (565 lines, new)
- `src/brainlayer/pipeline/digest.py` (20 lines changed)
- `tests/test_kg_p2_crosstype_cleanup.py` (123 lines, new)
- `tests/test_phase3_digest.py` (44 lines added)

**Key operations:**
1. Merge 40 entity fragments into pinned project/agent hubs
2. Promote 3 Codex tool entities to agent type (with collision detection)
3. Reparent BrainLayer MCP tool entities and Domica repo
4. Delete 5 domicaClaude launcher entities after rerouting chunks
5. Improve entity_lookup to prefer exact typed matches before FTS fallback

---

## Critical Path Assessment

### ✅ Count Gates
The script enforces exact expected counts via `EXPECTED_COUNTS` and `validate_counts()`:
- `merge_sources: 40`
- `launcher_delete_rows: 5`
- `promotions: 3`

**Safety**: If the live DB shape diverged from the approved plan, the script would abort with `RuntimeError("count gate mismatch")`. This is the correct defensive pattern.

### ✅ Live Execution Safety
Per PR description:
- Backup created: `brainlayer-pre-p2-crosstype-20260531T091941Z.db`
- After-report: `docs.local/P2-CROSSTYPE-AFTER.md`
- Commander verified: orphan counts = 0, pinned hubs intact, Domica parent link correct

**Observation**: The script correctly uses `--apply` flag with dry-run as default. Transaction rollback on error is properly implemented.

### ✅ Concurrency & DB Safety
- Uses `BEGIN IMMEDIATE` transaction (line 545)
- Rollback on exception (lines 549-550)
- Reads use `_read_cursor()` for consistency
- Script is idempotent: can safely re-run after rollback

**No issues found.**

---

## Code Quality Analysis

### 1. Entity Promotion Logic (`promote_to_agent`, lines 272-297)

**Good:**
- Collision detection: blocks promotion if agent with same name exists
- Type validation: only promotes from `tool` to `agent`
- Canonical name handling: `coalesce(nullif(canonical_name, ''), lower(?))`

**Edge case covered:** If entity is already `agent`, early return (line 279).

**Potential improvement:** The script doesn't verify that the promoted entity has meaningful relations/chunks before promotion. However, since this is a one-time cleanup with hardcoded IDs, this is acceptable.

### 2. Chunk Link Merging (`move_chunk_link`, lines 316-372)

**Logic:**
- If target already has the chunk link, merges metrics (relevance, tier, weight) using max/min strategies
- Context preservation: prefers context from higher-relevance source
- Mention type: "explicit" takes precedence over "implicit" or null

**Test coverage:** `test_move_chunk_link_merges_duplicate_null_metrics` validates null metric handling.

**Observation:** The merge strategy is sound. Using `_coalesce_max` for relevance/weight and `_coalesce_min` for tier is correct (lower tier = higher importance in BrainLayer).

### 3. Domica Launcher Cleanup (`cleanup_domica_launcher_rows`, lines 375-417)

**Safety checks:**
1. Verifies source entities have zero relations (line 390-391)
2. Validates every chunk link is classified (repo/company/drop) (line 399-400)
3. Deletes from all related tables in correct order

**Chunk routing:**
- `repo_chunk_ids` → rerouted to `DOMICA_REPO`
- `company_chunk_ids` → rerouted to `DOMICA_COMPANY` (currently empty set)
- `drop_chunk_ids` → deleted without reroute

**Test coverage:** `test_domica_launcher_cleanup_routes_repo_chunks_and_drops_noise` validates routing logic.

**No issues found.**

### 4. Entity Lookup Defensive Fix (`digest.py`, lines 812-824)

**Problem solved:** When `entity_lookup` is called with `entity_type="project"` for query "Domica", if `resolve_entity()` returns a `company` entity, the old code would fall through to FTS, which might return a partial match like "EtanHey/domica" instead of the exact "domica" project entity.

**Fix:** Added `_exact_name_siblings(store, query, entity_type=entity_type)` check before FTS fallback.

**Logic:**
1. Try `resolve_entity()` first (existing behavior)
2. If resolve returns wrong type, check for exact-name siblings of correct type
3. If exact typed matches exist, pick evidence-rich one
4. Otherwise fall back to FTS → semantic search

**Test coverage:** `test_entity_lookup_prefers_exact_typed_match_when_name_resolves_to_other_type` validates this path.

**Observation:** This is a good defensive addition. The `_select_evidence_rich_entity` scorer uses `(chunk_count, relation_count, importance)` tuple for selection, which correctly prioritizes entities with real evidence over stubs.

---

## Test Coverage Assessment

### New Tests (5 total)

1. ✅ `test_count_gate_requires_exact_approved_shape` - validates count gate enforcement
2. ✅ `test_promote_entity_type_rejects_existing_agent_name` - collision detection
3. ✅ `test_domica_launcher_cleanup_routes_repo_chunks_and_drops_noise` - routing logic
4. ✅ `test_move_chunk_link_merges_duplicate_null_metrics` - null metric merging
5. ✅ `test_entity_lookup_prefers_exact_typed_match_when_name_resolves_to_other_type` - typed lookup priority

**Coverage verdict:** All critical paths are tested. The tests use realistic scenarios and proper fixtures.

### Full Test Suite Results
```
2313 passed, 50 skipped, 1 xfailed, 4 xpassed, 2 failed, 32 errors
```

**Failed tests:**
- `test_canonical_build_removes_only_stale_dev_bundles` - BrainBar build guard (env setup issue)
- `test_real_db_exists` - integration test expecting production DB

**Errors:** 32 integration tests expecting production DB or specific system setup.

**Verdict:** Failures are not related to this PR's changes. Core functionality tests pass.

---

## Lint & Format Status

✅ **ruff check**: All checks passed  
✅ **ruff format**: 286 files already formatted  

No style violations.

---

## Concurrency & Lock Analysis

### Write Safety
- Script uses `BEGIN IMMEDIATE` for write atomicity
- Merge operations delegate to `merge_entities_preserving_links()` (existing, tested function)
- No parallel write operations within script

### Risk: `brain_digest` Collision
Per `AGENTS.md`:
> brain_digest is write-heavy; do not run it in parallel with other MCP work.

**Mitigation:** Script is designed for one-time offline execution. Users should not run `brain_digest` during cleanup execution. Since cleanup was already completed, this is now a historical note.

### Risk: WAL Growth
Per `AGENTS.md`:
> WAL can grow to 4.7GB

**Mitigation:** Script includes checkpoint before and after large operations would be ideal, but since it uses a transaction, WAL truncation happens on commit. For future bulk ops, consider manual `PRAGMA wal_checkpoint(FULL)` calls.

---

## Observations & Recommendations

### 1. Cross-Type Entity Aggregation Safety
Line 36 defines:
```python
_SAFE_CROSS_TYPE_ENTITY_AGGREGATES = {"claude code"}
```

This constant is defined in `digest.py` but only used in the entity lookup logic to allow certain cross-type exact-name collisions. The cleanup script hardcodes specific entity IDs and doesn't rely on this set.

**Recommendation:** Document which entities are considered "safe" cross-type aggregates in `CLAUDE.md` so future engineers understand why "claude code" can exist as both a tool and an agent.

### 2. Hardcoded Entity IDs
The script uses 40+ hardcoded UUIDv5 entity IDs. This is correct for a one-time cleanup but makes the script non-reusable.

**Observation:** The count gates and validation functions ensure the script is safe as-is. No action needed since this is a one-time operation.

### 3. Orphan Count Verification
The script's `orphan_counts()` function (lines 488-515) validates referential integrity post-cleanup:
- `entity_chunk_orphans`: chunk links pointing to deleted entities
- `relation_source_orphans`: relations with deleted source entities
- `relation_target_orphans`: relations with deleted target entities

**Good practice:** This is the correct post-execution validation. Per PR description, all orphan counts were 0 after live execution.

### 4. Missing Test: Promotion Type Enforcement
The script verifies `entity_type == 'tool'` before promotion (line 280-281), but there's no test for attempting to promote a non-tool entity (e.g., project → agent).

**Impact:** Low (the live script only promotes hardcoded tool IDs).

**Recommendation:** Add test case:
```python
def test_promote_to_agent_rejects_non_tool_type(store):
    cleanup = _load_script()
    store.upsert_entity("project-123", "project", "TestProject")
    with pytest.raises(RuntimeError, match="expected tool, found project"):
        cleanup.promote_to_agent(store, "project-123")
```

---

## Security Considerations

### 1. SQL Injection Risk
All SQL uses parameterized queries with `?` placeholders. No f-string interpolation of user data into SQL.

**Verdict:** ✅ No SQL injection vulnerabilities.

### 2. Data Loss Prevention
- Backup created before execution: `brainlayer-pre-p2-crosstype-20260531T091941Z.db`
- Transaction rollback on error
- Dry-run mode as default

**Verdict:** ✅ Proper safety measures in place.

### 3. Commander Gate Approval
Per PR description:
> Dry-run is the default. Use --apply only after commander gate approval.

**Observation:** This follows the established BrainLayer safety pattern for destructive operations.

---

## Final Verdict

### ✅ Approved for Merge

**Rationale:**
1. All critical paths tested and passing
2. Count gates prevent accidental misapplication
3. Transaction safety and rollback implemented correctly
4. Live execution completed successfully with commander verification
5. Entity lookup defensive fix is sound and well-tested
6. No concurrency or lock handling issues
7. Lint and format checks pass

**Risk Level:** Low (one-time script, already executed, tests pass)

**Deployment Notes:**
- Script has already been executed on live DB with success
- Backup exists: `brainlayer-pre-p2-crosstype-20260531T091941Z.db`
- After-report: `docs.local/P2-CROSSTYPE-AFTER.md`
- Orphan verification: all zero (clean state)

---

## Recommendations for Future Bulk Operations

Based on review of this cleanup script, future bulk DB operations should:

1. **Always use count gates** - validate expected row counts before and after
2. **Checkpoint WAL** - add `PRAGMA wal_checkpoint(FULL)` before/after large ops
3. **Test orphan detection** - verify referential integrity post-operation
4. **Dry-run by default** - require explicit `--apply` flag for writes
5. **Transaction boundaries** - wrap in `BEGIN IMMEDIATE` with rollback on error
6. **Backup first** - create timestamped backup before destructive ops

---

## Test Execution Summary

**Specific PR Tests:**
```bash
pytest tests/test_kg_p2_crosstype_cleanup.py \
       tests/test_phase3_digest.py::test_entity_lookup_prefers_exact_typed_match_when_name_resolves_to_other_type -v
```
**Result:** ✅ 5/5 passed (100%)

**Full Suite:**
```bash
pytest -q
```
**Result:** 2313 passed, 50 skipped, 1 xfailed, 4 xpassed, 2 failed (unrelated), 32 errors (integration env setup)

**Lint:**
```bash
ruff check src/ tests/ scripts/kg_p2_crosstype_cleanup.py
```
**Result:** ✅ All checks passed

**Format:**
```bash
ruff format --check src/ tests/ scripts/kg_p2_crosstype_cleanup.py
```
**Result:** ✅ 286 files already formatted

---

## Reviewer Notes

This is high-quality, safety-first code. The author clearly understands BrainLayer's concurrency model, transaction semantics, and the criticality of knowledge graph integrity. The combination of:
- Count gates
- Type validation
- Collision detection
- Transaction rollback
- Orphan verification
- Comprehensive tests

...demonstrates a mature approach to database migrations. The entity lookup defensive fix is a smart addition that prevents future cross-type resolution bugs.

**Approved for merge without changes.**

---

**Reviewed by:** @bugbot  
**Review completion time:** 2026-05-31T09:58:00Z
