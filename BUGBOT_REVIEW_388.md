# Bugbot Review: PR #388 - feat: prepare BrainBar and launchd engine cutover

**Review Date**: 2026-05-29  
**Reviewer**: Bugbot (Cloud Agent)  
**Branch**: `feat/brain-engine-extraction`  
**Risk Level**: **MEDIUM** (Installation dependency, breaking change)

---

## Executive Summary

This PR successfully shifts BrainBar and launchd services from source-tree imports (`PYTHONPATH=<repo>/src`) to installed-package imports. The implementation is technically sound and well-tested, but introduces a **hard dependency on package installation** that creates deployment risk if not carefully managed.

**Critical Path Impact**: ✅ Retrieval correctness unaffected, ✅ MCP stability preserved, ⚠️ Write safety requires careful rollout.

**Status**: **APPROVED WITH STRONG DEPLOYMENT WARNINGS**

---

## Critical Findings

### 🔴 CRITICAL: Silent Breakage on Partial Deployment

**Severity**: High  
**Component**: Launchd plists, BrainBar hybrid helper  
**Risk**: Service degradation if plists updated without package install

#### The Issue

The PR removes `PYTHONPATH=__BRAINLAYER_DIR__/src` from all `com.brainlayer.*` plists and changes Swift `resolvePythonPath()` to return `nil` (use installed package) by default. Source fallback now requires explicit `BRAINLAYER_SOURCE_FALLBACK=1`.

**Before (main branch)**:
```xml
<!-- scripts/launchd/com.brainlayer.enrichment.plist -->
<key>PYTHONPATH</key>
<string>__BRAINLAYER_DIR__/src</string>
```

**After (this PR)**:
```xml
<!-- PYTHONPATH key removed entirely -->
<key>BRAINLAYER_REPO_ROOT</key>
<string>__BRAINLAYER_DIR__</string>
```

**Swift change** (`brain-bar/Sources/BrainBar/HybridSearchHelperClient.swift:78-93`):
```swift
static func resolvePythonPath(environment: [String: String]) -> String? {
    if let existing = environment["PYTHONPATH"], !existing.isEmpty {
        return existing
    }
    guard environment["BRAINLAYER_SOURCE_FALLBACK"] == "1" else {
        return nil  // NEW: Default to installed package
    }
    // Only use source tree if explicitly requested
    guard let repoRoot = normalizedRepoRoot(environment: environment) else {
        return nil
    }
    let sourcePath = "\(repoRoot)/src"
    guard FileManager.default.fileExists(atPath: sourcePath) else {
        return nil
    }
    return sourcePath
}
```

#### Impact

If a user:
1. Updates launchd plists (via install script or manual copy)
2. Restarts BrainBar or reloads `com.brainlayer.*` agents
3. **Has not run `pip install -e .`**

**Result**: All services fail with `ModuleNotFoundError: No module named 'brainlayer'`.

- Enrichment stops
- Watcher stops  
- BrainBar search fails (hybrid helper crashes on startup)
- Daily backup fails
- Drain queue processing stops

**Affected systems**: Every background service + BrainBar UI.

#### Why This Matters

The PR description says:
> "PR only prepares cutover. It does not cut over live services."

But this is **misleading**. The plists ARE live config. Anyone running:
```bash
bash brain-bar/build-app.sh
# or
scripts/install-launchd.sh
```

Will **immediately cut over** to the new behavior and break if the package isn't installed.

#### Recommendations

1. **Add explicit pre-flight check to install scripts**:
   ```bash
   # In brain-bar/build-app.sh and scripts/install-launchd.sh
   if ! python3 -c "import brainlayer" 2>/dev/null; then
       echo "[ERROR] brainlayer package not installed."
       echo "Run: pip install -e . (or pip install brainlayer)"
       exit 1
   fi
   ```

2. **Add migration warning to README.md**:
   ```markdown
   ## ⚠️ Breaking Change (v0.x.x)
   
   BrainLayer now requires package installation for all services:
   
   ```bash
   pip install -e .  # Development
   # or
   pip install brainlayer  # Production
   ```
   
   **Before** updating launchd agents or rebuilding BrainBar, ensure the package is installed.
   ```

3. **Update PR description** to clarify: "Prepares cutover but does NOT reload live agents. Manual `launchctl unload/load` required to activate."

---

### 🟡 MODERATE: Missing Failure Mode Documentation

**Severity**: Medium  
**Component**: Error handling, user experience

#### The Issue

When the package is not installed and services try to start, users get:
```python
ModuleNotFoundError: No module named 'brainlayer'
```

This error message doesn't tell the user:
- That this is expected if the package isn't installed
- How to fix it (`pip install -e .`)
- Whether they should use source fallback (`BRAINLAYER_SOURCE_FALLBACK=1`)

#### Recommendation

Add a `try/except` wrapper in `src/brainlayer/brainbar_hybrid_helper.py:main()`:

```python
def main(argv: list[str] | None = None) -> int:
    try:
        from brainlayer.parent_death import install_parent_death_watcher
    except ImportError as exc:
        print(
            "ERROR: brainlayer package not found.\n"
            "Install with: pip install -e . (or pip install brainlayer)\n"
            "For source-tree fallback: export BRAINLAYER_SOURCE_FALLBACK=1",
            file=sys.stderr,
        )
        return 1
    
    install_parent_death_watcher()
    # ... rest of main()
```

---

### 🟢 POSITIVE: Contract Test Improvement

**Severity**: N/A (enhancement)  
**Component**: `tests/test_hybrid_helper_contract.py`

#### The Change

Lines 139-142 now fail fast on EOF instead of hanging:

```python
while not payload.endswith(b"\n"):
    chunk = client.recv(65536)
    if not chunk:
        raise AssertionError("helper closed socket before sending a full NDJSON line")
    payload += chunk
```

**Before**: Test could hang indefinitely if helper crashed without writing response.  
**After**: Test fails immediately with clear error message.

**Impact**: Faster test failures, clearer CI output. ✅

---

### 🟢 POSITIVE: Missing `agent_id` Parameter Added

**Severity**: N/A (bug fix)  
**Component**: `src/brainlayer/brainbar_hybrid_helper.py:191`

#### The Change

The hybrid helper now accepts and forwards `agent_id`:

```python
search_kwargs = {
    # ... other args
    "agent_id": arguments.get("agent_id"),  # NEW
    # ...
}
```

**Before**: BrainBar search couldn't scope results by `agent_id`.  
**After**: BrainBar search respects per-agent memory isolation.

**Impact**: Fixes missing feature parity between BrainBar search and Python MCP `brain_search`. ✅

---

### 🟢 POSITIVE: Launchd Hygiene Tests

**Severity**: N/A (test coverage)  
**Component**: `tests/test_launchd_hygiene.py:125-128`

#### The Addition

New test validates all canonical plists use installed package:

```python
def test_script_launchagents_use_installed_package_imports():
    for path in sorted((REPO_ROOT / "scripts/launchd").glob("com.brainlayer.*.plist")):
        _assert_uses_installed_package_not_source_path(path, plistlib.loads(path.read_bytes()))
```

Where:
```python
def _assert_uses_installed_package_not_source_path(path: Path, plist: dict) -> None:
    env = plist.get("EnvironmentVariables") or {}
    assert "PYTHONPATH" not in env, f"{path}: canonical LaunchAgent must import installed brainlayer package"
    assert env.get("BRAINLAYER_REPO_ROOT") == "__BRAINLAYER_DIR__"
```

**Impact**: Prevents regression. Any future plist that leaks `PYTHONPATH` will fail this test. ✅

---

### 🟢 POSITIVE: Swift Test Coverage for Fallback Behavior

**Severity**: N/A (test coverage)  
**Component**: `brain-bar/Tests/BrainBarTests/HybridSearchHelperClientTests.swift`

#### The Tests

Three new tests validate the PYTHONPATH resolution contract:

1. **`testResolvePythonPathPrefersInstalledPackageWhenUnset`** (lines 27-40):
   - Validates that `resolvePythonPath` returns `nil` when `BRAINLAYER_SOURCE_FALLBACK` is not set
   - Even if `BRAINLAYER_REPO_ROOT` and `<repo>/src` exist

2. **`testResolvePythonPathUsesRepoSourceDirectoryOnlyWhenFallbackRequested`** (lines 42-56):
   - Validates that `resolvePythonPath` returns `<repo>/src` when `BRAINLAYER_SOURCE_FALLBACK=1`

3. **`testResolvePythonPathPreservesExistingPythonPath`** (lines 58-65):
   - Validates that existing `PYTHONPATH` is always preserved (highest priority)

**Impact**: Complete coverage of the new fallback behavior. ✅

---

## MCP Stability Analysis

### Socket Protocol Unchanged ✅

The hybrid helper still speaks the same NDJSON protocol:
- Request: `{"method": "brain_search", "arguments": {...}}\n`
- Response: `{"ok": true, "text": "...", "metadata": {...}}\n`

No changes to wire format, so **BrainBar ↔ Python contract is stable**.

### Tool Schemas Unchanged ✅

The `brain_search` tool arguments accepted by the helper remain the same (with `agent_id` added):
- `query`, `project`, `source`, `tag`, `importance_min`, `agent_id`, `num_results`, `max_results`, `detail`, `_profile_query_id`

Documented in `contracts/engine-ui-contract.md:50`. ✅

---

## Retrieval Correctness Analysis

### No Changes to Search Logic ✅

The PR only changes **how** the Python code is loaded (via PYTHONPATH vs installed package). The actual search implementation in `src/brainlayer/mcp/search_handler.py` is **unchanged**.

Retrieval quality, ranking, and result filtering are unaffected. ✅

---

## Write Safety Analysis

### No Changes to Write Path ✅

Enrichment, watcher, drain, and `brain_store` write handlers are unchanged. The only difference is import resolution.

**However**: If services crash on startup due to missing package, writes will **queue** instead of processing. Not data loss, but operational degradation.

**Mitigation**: Pre-flight install check (see recommendation above). ✅

---

## Lock Handling Analysis

### No Concurrency Changes ✅

The PR does not touch:
- DB connection management (`vector_store.py`)
- WAL checkpoint logic
- Enrichment write locks
- Queue arbitration

Lock behavior is **identical** before and after. ✅

---

## Edge Cases & Subtle Risks

### 1. Environment Variable Priority (Minor)

**File**: `brain-bar/Sources/BrainBar/HybridSearchHelperClient.swift:78-93`

The priority order is:
1. Existing `PYTHONPATH` (line 79-80) — always wins
2. `BRAINLAYER_SOURCE_FALLBACK=1` (line 82-83) — checked second
3. `nil` (line 83) — default

**Edge case**: If someone has `PYTHONPATH` set to an **old** source tree (different repo), the helper will use that instead of the installed package, even if they **didn't** set `BRAINLAYER_SOURCE_FALLBACK=1`.

**Severity**: Low (user misconfiguration, not a code bug)  
**Recommendation**: Document this in `contracts/engine-ui-contract.md`.

### 2. `resolvePythonPath` Called After Modifying `env` (Minor)

**File**: `brain-bar/Sources/BrainBar/HybridSearchHelperClient.swift:216-220`

```swift
var env = environment
env["BRAINLAYER_DB"] = dbPath  // Line 217
if let pythonPath = Self.resolvePythonPath(environment: env) {  // Line 218
    env["PYTHONPATH"] = pythonPath
}
```

`resolvePythonPath` is called **after** `BRAINLAYER_DB` is added to the env dict. This is fine because `resolvePythonPath` only reads `PYTHONPATH`, `BRAINLAYER_SOURCE_FALLBACK`, and `BRAINLAYER_REPO_ROOT`.

**Potential issue**: If future code adds logic to `resolvePythonPath` that depends on other env vars, the order might matter.

**Severity**: Low (hypothetical future issue)  
**Recommendation**: Add a comment:
```swift
// Note: BRAINLAYER_DB is set before resolvePythonPath, but resolvePythonPath
// only reads PYTHONPATH, BRAINLAYER_SOURCE_FALLBACK, and BRAINLAYER_REPO_ROOT.
```

### 3. `backup-daily.sh` No Longer Sets `PYTHONPATH` (Good)

**File**: `scripts/launchd/backup-daily.sh:13`

**Before**: Script exported `PYTHONPATH=__BRAINLAYER_DIR__/src`  
**After**: Script runs `python3 -m brainlayer.backup_daily` directly

This is **correct**. The script relies on the installed package. ✅

The plist sets `BRAINLAYER_REPO_ROOT=__BRAINLAYER_DIR__` but **not** `PYTHONPATH`, which is the new contract.

**Test coverage**: `tests/test_backup_daily.py:108` (`test_launchd_installer_knows_backup_target`) validates the launchd hygiene.

---

## Test Coverage Summary

| Test File | Coverage |
|-----------|----------|
| `test_engine_package_boundary.py` | Validates `pyproject.toml` excludes CLI/dashboard from wheel ✅ |
| `test_hybrid_helper_contract.py` | NDJSON protocol + EOF fast-fail ✅ |
| `test_launchd_hygiene.py` | Validates no `PYTHONPATH` in canonical plists ✅ |
| `HybridSearchHelperClientTests.swift` | Swift PYTHONPATH fallback behavior ✅ |

**Missing coverage**:
- ❌ Test for failure mode when package is **not** installed (error message quality)
- ❌ Test for partial deployment scenario (old plists + new code)

---

## Recommendations Summary

### Must-Have Before Merge

1. **Add pre-flight install check** to `brain-bar/build-app.sh` and any install scripts
2. **Add migration warning** to README.md or MIGRATION.md
3. **Clarify PR description** about what "prepares cutover" means

### Nice-to-Have

4. Add better error message in `brainbar_hybrid_helper.py:main()` for missing package
5. Add comment in Swift about env var order (line 218)
6. Document `PYTHONPATH` priority in `contracts/engine-ui-contract.md`
7. Add test for failure mode when package is not installed

---

## Deployment Checklist (for Cutover)

When actually deploying this change to live systems:

- [ ] Verify `pip install -e .` (or `pip install brainlayer`) is run
- [ ] Test BrainBar hybrid helper starts without errors: `tail -f ~/Library/Logs/brainlayer/brainbar.err.log`
- [ ] Test enrichment starts: `launchctl list | grep com.brainlayer.enrichment`
- [ ] Test watcher starts: `launchctl list | grep com.brainlayer.watch`
- [ ] Run `brain_search` from BrainBar search panel to validate hybrid helper
- [ ] Check no errors in: `~/Library/Logs/brainlayer/*.err.log`
- [ ] Verify fallback works: `BRAINLAYER_SOURCE_FALLBACK=1 python3 -m brainlayer.brainbar_hybrid_helper --socket-path /tmp/test.sock --db-path /tmp/test.db`

---

## Conclusion

The PR is **technically sound** and implements the cutover correctly. The new contract is well-tested, and the code changes are minimal and focused. The risk is **operational**, not technical: if users update plists without installing the package, services will break.

**Approval**: ✅ **APPROVED** with deployment warnings.

**Blockers before merge**: Add pre-flight install checks to build/install scripts.

**Recommended follow-up**: Better error messages for missing package, additional edge-case tests.

---

**Reviewed by**: Bugbot (Cursor Cloud Agent)  
**Review type**: Static analysis + contract validation  
**Focus areas**: Retrieval correctness, MCP stability, write safety, lock handling, deployment risk
