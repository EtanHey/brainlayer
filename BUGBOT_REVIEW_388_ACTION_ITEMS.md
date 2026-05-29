# Bugbot Review Action Items - PR #388

**Review**: BUGBOT_REVIEW_388.md  
**Status**: Approved with required changes before merge

---

## 🔴 REQUIRED BEFORE MERGE

### 1. Add Pre-Flight Package Check to `build-app.sh`

**File**: `brain-bar/build-app.sh`  
**Location**: After line 333, before `configure_launchagent_environment` function  
**Severity**: Critical - prevents service breakage

#### Change

```bash
# Add this function before configure_launchagent_environment (after line 333)

check_brainlayer_package_installed() {
    local repo_root="$1"
    local python_path="$repo_root/.venv/bin/python"
    
    # Use repo venv if available, otherwise system python3
    local python_exec="python3"
    if [ -x "$python_path" ]; then
        python_exec="$python_path"
    fi
    
    if ! "$python_exec" -c "import brainlayer" 2>/dev/null; then
        echo "[build-app] ERROR: brainlayer package not installed" >&2
        echo "" >&2
        echo "This build requires the brainlayer package to be installed." >&2
        echo "BrainBar and launchd services no longer use PYTHONPATH for imports." >&2
        echo "" >&2
        echo "Install with:" >&2
        echo "  cd $repo_root" >&2
        if [ -x "$python_path" ]; then
            echo "  $python_path -m pip install -e ." >&2
        else
            echo "  python3 -m pip install -e ." >&2
        fi
        echo "" >&2
        echo "For temporary source-tree fallback:" >&2
        echo "  export BRAINLAYER_SOURCE_FALLBACK=1" >&2
        echo "" >&2
        return 1
    fi
    
    echo "[build-app] ✓ brainlayer package is installed"
    return 0
}
```

#### Call Site

Update `configure_launchagent_environment` to call the check first:

```bash
configure_launchagent_environment() {
    local plist_path="$1"
    local repo_root="$2"
    local python_path="$repo_root/.venv/bin/python"
    
    # NEW: Pre-flight check
    if ! check_brainlayer_package_installed "$repo_root"; then
        exit 1
    fi

    "$PLIST_BUDDY" -c "Delete :EnvironmentVariables" "$plist_path" >/dev/null 2>&1 || true
    "$PLIST_BUDDY" -c "Add :EnvironmentVariables dict" "$plist_path"
    "$PLIST_BUDDY" -c "Add :EnvironmentVariables:BRAINLAYER_REPO_ROOT string \"$repo_root\"" "$plist_path"
    if [ -x "$python_path" ]; then
        "$PLIST_BUDDY" -c "Add :EnvironmentVariables:BRAINBAR_PYTHON string \"$python_path\"" "$plist_path"
    fi
}
```

#### Rationale

Prevents silent breakage if someone runs `build-app.sh` without installing the package first. The script will fail immediately with a clear, actionable error message.

---

### 2. Add Migration Notice to README.md

**File**: `README.md`  
**Location**: After line 28 (after `pip install brainlayer`)  
**Severity**: High - user communication

#### Change

```markdown
pip install brainlayer
```

**⚠️ Migration Notice (v0.x.x)**: If upgrading from a version that used `PYTHONPATH` for imports, you must install the package **before** updating launchd agents or rebuilding BrainBar:

```bash
# Development install
pip install -e .

# Or production install
pip install brainlayer
```

Services will fail to start if the package is not installed. For temporary source-tree fallback during development:

```bash
export BRAINLAYER_SOURCE_FALLBACK=1
```

Add to your MCP config (`~/.claude.json` for Claude Code):
```

#### Rationale

Users need to know this is a breaking change and must take action before updating. Without this notice, existing users will experience silent breakage.

---

## 🟡 STRONGLY RECOMMENDED

### 3. Improve Error Message in Hybrid Helper

**File**: `src/brainlayer/brainbar_hybrid_helper.py`  
**Location**: Lines 252-266 (wrap main() imports)  
**Severity**: Medium - user experience

#### Change

```python
def main(argv: list[str] | None = None) -> int:
    # Validate package is importable before attempting startup
    try:
        from brainlayer.parent_death import install_parent_death_watcher
    except ImportError as exc:
        print(
            "ERROR: brainlayer package not found.\n"
            "\n"
            "The hybrid search helper requires the brainlayer package to be installed.\n"
            "Install with:\n"
            "  pip install -e .  (development)\n"
            "  pip install brainlayer  (production)\n"
            "\n"
            "For source-tree fallback:\n"
            "  export BRAINLAYER_SOURCE_FALLBACK=1\n"
            "\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    install_parent_death_watcher()
    
    args = parse_args(argv)
    if args.db_path:
        os.environ["BRAINLAYER_DB"] = args.db_path

    from brainlayer.paths import get_db_path

    helper = HybridSearchHelper(socket_path=Path(args.socket_path), db_path=get_db_path())
    signal.signal(signal.SIGINT, helper.stop)
    helper.serve_forever()
    return 0
```

#### Rationale

When BrainBar tries to start the helper and the package isn't installed, the error should be immediately obvious. Currently users would see a bare `ModuleNotFoundError` which doesn't explain the fix.

---

### 4. Document PYTHONPATH Priority in Contract

**File**: `contracts/engine-ui-contract.md`  
**Location**: After line 42 (in "Environment contract" section)  
**Severity**: Medium - documentation completeness

#### Change

```markdown
Environment contract:

- `BRAINBAR_PYTHON` overrides Python executable resolution at `brain-bar/Sources/BrainBar/HybridSearchHelperClient.swift:53`.
- Installed package imports are the default. `BRAINLAYER_REPO_ROOT` is used to resolve `<repo>/src` for `PYTHONPATH` only when `BRAINLAYER_SOURCE_FALLBACK=1`, preserving a deliberate source-tree fallback at `brain-bar/Sources/BrainBar/HybridSearchHelperClient.swift:78`.
- **PYTHONPATH priority**: If `PYTHONPATH` is already set in the environment, it is always preserved and takes precedence over both the fallback and the default installed-package behavior. Order of precedence: (1) existing `PYTHONPATH`, (2) `BRAINLAYER_SOURCE_FALLBACK=1` → `<repo>/src`, (3) nil → use installed package.
- `PYTHONPATH` is passed through or set before launch at `brain-bar/Sources/BrainBar/HybridSearchHelperClient.swift:213`.
- `BRAINLAYER_DB` is set to the selected DB path at `brain-bar/Sources/BrainBar/HybridSearchHelperClient.swift:214` and by the helper at `src/brainlayer/brainbar_hybrid_helper.py:258`.
```

#### Rationale

The priority order is implicit in the code but not documented. Users debugging import issues need to know that an existing PYTHONPATH will override everything else.

---

### 5. Add Swift Code Comment About Env Var Order

**File**: `brain-bar/Sources/BrainBar/HybridSearchHelperClient.swift`  
**Location**: Line 216 (before `env["BRAINLAYER_DB"] = dbPath`)  
**Severity**: Low - code maintainability

#### Change

```swift
var env = environment
// Note: BRAINLAYER_DB is set before resolvePythonPath, but resolvePythonPath
// only reads PYTHONPATH, BRAINLAYER_SOURCE_FALLBACK, and BRAINLAYER_REPO_ROOT,
// so order does not matter for current implementation.
env["BRAINLAYER_DB"] = dbPath
if let pythonPath = Self.resolvePythonPath(environment: env) {
    env["PYTHONPATH"] = pythonPath
}
```

#### Rationale

Prevents future maintainers from accidentally introducing order-dependent bugs if `resolvePythonPath` logic changes.

---

## 🟢 NICE-TO-HAVE (Post-Merge)

### 6. Add Test for Missing Package Failure Mode

**File**: `tests/test_hybrid_helper_contract.py` (new test)  
**Severity**: Low - test coverage

#### Change

```python
def test_brainbar_hybrid_helper_fails_gracefully_when_package_not_installed(tmp_path):
    """Verify the helper exits with a clear error when brainlayer isn't importable."""
    socket_path = Path(f"/tmp/brainbar-missing-pkg-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock")
    
    # Use a PYTHONPATH that doesn't contain brainlayer
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)  # Empty dir, no brainlayer
    env["BRAINLAYER_SOURCE_FALLBACK"] = ""  # Disable fallback
    env.pop("BRAINLAYER_REPO_ROOT", None)  # Remove repo root
    
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "brainlayer.brainbar_hybrid_helper",
            "--socket-path",
            str(socket_path),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    try:
        stdout, stderr = process.communicate(timeout=2)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        raise AssertionError("helper should exit immediately when package is missing")
    
    assert returncode != 0, "helper should fail when package is not installed"
    assert "brainlayer package not found" in stderr.lower() or "modulenotfounderror" in stderr.lower()
```

#### Rationale

Validates that the failure mode is graceful and produces useful error messages.

---

### 7. Update PR Description Clarity

**File**: PR description on GitHub  
**Severity**: Low - communication

#### Change

Current text:
> PR only prepares cutover. It does not cut over live services.

Should be:
> PR prepares cutover by updating plists and build scripts to use installed package imports. **Live services will cut over when launchd agents are reloaded** (via `launchctl unload/load` or `build-app.sh`). Manual reload required; automatic reload from KeepAlive will NOT pick up new plists until next restart.

#### Rationale

"Does not cut over live services" is ambiguous. Clarify that the cutover happens on agent reload, not on merge.

---

## Testing Checklist

Before merging, verify:

- [ ] Run modified `build-app.sh` without package installed → fails with clear error
- [ ] Run modified `build-app.sh` with package installed → succeeds
- [ ] BrainBar hybrid helper starts successfully after build
- [ ] Enrichment service starts: `launchctl list | grep enrichment`
- [ ] Watch service starts: `launchctl list | grep watch`
- [ ] No errors in `~/Library/Logs/brainlayer/*.err.log`
- [ ] Source fallback works: `BRAINLAYER_SOURCE_FALLBACK=1 python3 -m brainlayer.brainbar_hybrid_helper --socket-path /tmp/test.sock --db-path /tmp/test.db` (should start without error)

---

## Post-Merge Deployment Plan

When deploying to production:

1. **Before reloading agents**:
   - Verify package install: `python3 -c "import brainlayer" || echo "INSTALL REQUIRED"`
   - If missing: `pip install -e .` (or `pip install brainlayer`)

2. **Reload agents** (choose one):
   - Run `bash brain-bar/build-app.sh` (rebuilds + reloads)
   - Or manual reload:
     ```bash
     launchctl bootout gui/$(id -u)/com.brainlayer.enrichment
     launchctl bootout gui/$(id -u)/com.brainlayer.watch
     # ... other agents
     launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.brainlayer.*.plist
     ```

3. **Verify services started**:
   ```bash
   launchctl list | grep com.brainlayer
   tail -f ~/Library/Logs/brainlayer/*.err.log
   ```

4. **Test BrainBar search**: Open BrainBar → run a search → verify no hybrid helper errors

---

## Summary

**Required Changes**: 3 (pre-flight check, README notice, clarify PR description)  
**Recommended Changes**: 2 (error message, contract doc)  
**Nice-to-Have**: 2 (test coverage, code comment)

**Estimated Implementation Time**: 1-2 hours for required changes.

**Risk After Changes**: Low. Pre-flight check prevents the main failure mode (services breaking on partial deployment).
