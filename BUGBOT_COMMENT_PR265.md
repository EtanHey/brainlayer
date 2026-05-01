## 🐛 BugBot Review - CRITICAL ISSUE FOUND

I've completed my review of PR #265 and found **1 CRITICAL BUG** that breaks the core functionality.

---

### 🔴 CRITICAL: Canonical Path Resolution Bug

**Location:** `brain-bar/build-app.sh:62-64`

The canonical path comparison fails when `CANONICAL_REPO_ROOT` doesn't exist (which is the **default configuration**). The script only canonicalizes the path if the directory exists:

```bash
if [ -d "$CANONICAL_REPO_ROOT" ]; then
    CANONICAL_REPO_ROOT="$(cd "$CANONICAL_REPO_ROOT" && pwd)"
fi
```

**Impact:**
- Default `CANONICAL_REPO_ROOT` is `$HOME/Gits/brainlayer`
- For new users, this directory doesn't exist
- Path canonicalization is **skipped**
- The guard compares canonicalized `CURRENT_REPO_ROOT` against non-canonicalized `CANONICAL_REPO_ROOT`
- Result: Path comparison is **broken** (compares `/workspace` vs `$HOME/Gits/brainlayer`)

**This breaks:**
1. ❌ Builds from canonical repo with tilde paths (`~/Gits/brainlayer`)
2. ❌ Builds from symlinked canonical repos
3. ❌ Default configuration for new users
4. ❌ Any scenario where canonical path doesn't pre-exist

**Proof:**

```bash
# Test with default config (directory doesn't exist)
export HOME="/tmp"
CANONICAL_REPO_ROOT="${HOME}/Gits/brainlayer"  # Default

# Current code:
if [ -d "$CANONICAL_REPO_ROOT" ]; then
    CANONICAL_REPO_ROOT="$(cd "$CANONICAL_REPO_ROOT" && pwd)"
fi
# → Skipped! Directory doesn't exist

CURRENT_REPO_ROOT="$(git rev-parse --show-toplevel)"  # Always canonicalized
# → "/workspace"

# Compare:
if [ "$CURRENT_REPO_ROOT" != "$CANONICAL_REPO_ROOT" ]; then
    echo "PATHS DON'T MATCH"
fi
# → "/workspace" != "/tmp/Gits/brainlayer" → REJECTED (wrong!)
```

---

### 💡 Recommended Fix

Either fail fast when canonical path doesn't exist, OR expand/normalize paths even when directory is missing:

```bash
CANONICAL_REPO_ROOT="${BRAINBAR_CANONICAL_REPO_ROOT:-$HOME/Gits/brainlayer}"
CANONICAL_REPO_ROOT="${CANONICAL_REPO_ROOT/#\~/$HOME}"  # Expand tilde

if [ ! -d "$CANONICAL_REPO_ROOT" ]; then
    echo "[build-app] WARNING: BRAINBAR_CANONICAL_REPO_ROOT not found: $CANONICAL_REPO_ROOT" >&2
    echo "[build-app] All builds will be treated as non-canonical" >&2
    CANONICAL_REPO_ROOT=""
fi

if [ -n "$CANONICAL_REPO_ROOT" ]; then
    CANONICAL_REPO_ROOT="$(cd "$CANONICAL_REPO_ROOT" && pwd)"
fi
```

---

### 🟡 Additional Issues Found

**Moderate Issues:**
1. Confusing error message when using `--force-worktree-build` with dirty tree (needs both flags but error doesn't say so)
2. Missing test coverage for path edge cases (symlinks, tilde, non-existent canonical path)

---

### 📄 Full Review

See `BUGBOT_REVIEW_PR265_BRAINBAR_GUARDS.md` for complete analysis with:
- Full reproduction steps
- All moderate issues detailed
- Test coverage gaps
- Complete recommended fixes
- Severity assessment table

---

### ⚠️ Recommendation

**🔴 BLOCK MERGE** until path resolution is fixed.

**Rationale:**
- The bug breaks the guard in the default configuration
- New users will hit this immediately
- The entire PR purpose is build guards - if they don't work correctly, the PR doesn't deliver value
- Fix is straightforward (see above)

**After Fix:**
- Add test for non-existent canonical path scenario
- Consider adding tests for symlinks and tilde paths
- Improve error messages (can be follow-up PR)

---

**Reviewed by:** @bugbot  
**Date:** 2026-05-01  
**Files Reviewed:** `brain-bar/build-app.sh`, `tests/test_brainbar_build_app_guards.py`  
**Commit:** 243a897ee7b4355e4e0ae47ba3e118e1ffaa9501
