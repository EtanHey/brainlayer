#!/usr/bin/env bash

set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="${BRAINLAYER_TEST_ROOT:-$ROOT_DIR/tests}"
BRAINLAYER_USE_UV="${BRAINLAYER_USE_UV:-1}"
# `not embedding_model` is suite hygiene, not a speed tweak: a marked test loads a real embedding
# model (2.5 GB RSS for BGE-M3), and this script is what pre-push runs on Etan's Macs. CI passes its
# own -m expression in .github/workflows/ci.yml and still runs them, on a runner that warms the HF
# cache on purpose. tests/conftest.py fails any UNMARKED test that loads one.
UNIT_MARK_EXPR="${BRAINLAYER_PYTEST_MARK_EXPR:-not integration and not live and not embedding_model}"
BRAINLAYER_PREPUSH="${BRAINLAYER_PREPUSH:-0}"
BRAINLAYER_PREPUSH_SCOPE="${BRAINLAYER_PREPUSH_SCOPE:-full}"
exit_status=0
declare -a targeted_pytest_files=()
declare -a unmapped_changed_files=()
changed_source_unmapped=0
changed_files_seen=0
# Empty when the changed set was established; otherwise the reason it could not be.
changed_files_scope_error=""

default_prepush_cache_dir() {
  local git_dir
  git_dir="$(git -C "$ROOT_DIR" rev-parse --git-path brainlayer-prepush-cache 2>/dev/null || true)"
  if [ -n "$git_dir" ]; then
    printf '%s\n' "$git_dir"
  else
    printf '%s\n' "$ROOT_DIR/.git/brainlayer-prepush-cache"
  fi
}

BRAINLAYER_PREPUSH_CACHE_DIR="${BRAINLAYER_PREPUSH_CACHE_DIR:-$(default_prepush_cache_dir)}"

REAL_DB_TEST_FILES=(
  "test_vector_store.py"
  "test_engine.py"
)

run_step() {
  local label="$1"
  shift

  echo "==> $label"
  "$@"
  local rc=$?
  exit_status=$(( exit_status | rc ))

  if [ "$rc" -eq 0 ]; then
    echo "PASS: $label"
  else
    echo "FAIL ($rc): $label"
  fi

  echo
}

collect_bun_tests() {
  if [ ! -d "$TEST_ROOT" ]; then
    return 0
  fi

  find "$TEST_ROOT" -type f -name "*.test.ts" | sort
}

collect_isolated_pytest_files() {
  if [ ! -d "$TEST_ROOT" ]; then
    return 0
  fi

  local candidate
  for candidate in \
    "$TEST_ROOT/test_eval_framework.py" \
    "$TEST_ROOT/test_follow_up_rewrite.py" \
    "$TEST_ROOT/test_prompt_classification.py"
  do
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
    fi
  done
}

collect_regression_shell_tests() {
  if [ ! -d "$TEST_ROOT" ]; then
    return 0
  fi

  find "$TEST_ROOT" -type f -path "*/regression/*.sh" | sort
}

prepush_tree_hash() {
  if [ -n "${BRAINLAYER_PREPUSH_TREE_HASH:-}" ]; then
    printf '%s\n' "$BRAINLAYER_PREPUSH_TREE_HASH"
    return 0
  fi
  git rev-parse HEAD^{tree} 2>/dev/null || true
}

prepush_cache_file() {
  local tree_hash="$1"
  mkdir -p "$BRAINLAYER_PREPUSH_CACHE_DIR"
  printf '%s/%s.full.ok\n' "$BRAINLAYER_PREPUSH_CACHE_DIR" "$tree_hash"
}

# Returns the changed paths on stdout and a MEANINGFUL exit code: non-zero says "this script could
# not determine what changed", which is a different thing from "nothing changed" and must never be
# collapsed into it. The old `2>/dev/null || true` swallowed a git failure into an empty list with
# rc 0, so a missing origin/main or a broken diff read as "nothing to test -- pass". That is a
# fail-OPEN on a gate, and silent is worse than expensive.
changed_files() {
  if [ -n "${BRAINLAYER_CHANGED_FILES:-}" ]; then
    printf '%s\n' "$BRAINLAYER_CHANGED_FILES" | tr ',' '\n' | sed '/^$/d'
    return 0
  fi
  # A TAG push has no branch to diff against: `git push origin v1.5.14` reached this function with
  # no origin/main relationship worth reading, took the default full scope, and ran all 4,386 tests
  # on the M4 for a ref whose content is exactly one range. .githooks/pre-push resolves that range
  # from the pushed ref and hands it over here. The rc is git's, deliberately: a range git cannot
  # resolve is a scope that was never established, which is not the same thing as an empty one.
  if [ -n "${BRAINLAYER_CHANGED_FILES_RANGE:-}" ]; then
    git diff --name-only "$BRAINLAYER_CHANGED_FILES_RANGE"
    return $?
  fi
  if git rev-parse --verify origin/main >/dev/null 2>&1; then
    git diff --name-only origin/main...HEAD
    return $?
  fi
  if git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
    git diff --name-only HEAD~1...HEAD
    return $?
  fi
  return 1
}

is_real_db_test_file() {
  local candidate_name
  candidate_name="$(basename "$1")"
  local real_db_test
  for real_db_test in "${REAL_DB_TEST_FILES[@]}"; do
    if [ "$candidate_name" = "$real_db_test" ]; then
      return 0
    fi
  done
  return 1
}

append_unique() {
  local value="$1"
  local existing
  if [ "${#targeted_pytest_files[@]}" -gt 0 ]; then
    for existing in "${targeted_pytest_files[@]}"; do
      if [ "$existing" = "$value" ]; then
        return 0
      fi
    done
  fi
  targeted_pytest_files+=("$value")
}

map_changed_files_to_pytests() {
  targeted_pytest_files=()
  unmapped_changed_files=()
  changed_source_unmapped=0
  changed_files_seen=0
  changed_files_scope_error=""
  local changed rel test_path module_name mapped changed_list detect_rc=0
  local init_version_test init_build_sha_test
  # Command substitution, not process substitution: `done < <(changed_files)` throws away the exit
  # code, which is the whole signal that tells a failed detection from an empty one.
  changed_list="$(changed_files)" || detect_rc=$?
  if [ "$detect_rc" -ne 0 ]; then
    changed_files_scope_error="git could not name the changed files (no origin/main and no HEAD~1, or the diff failed)"
    return 0
  fi
  if [ -n "${BRAINLAYER_CHANGED_FILES:-}" ] && [ -z "$changed_list" ]; then
    changed_files_scope_error="BRAINLAYER_CHANGED_FILES was set but named no paths"
    return 0
  fi
  while IFS= read -r changed; do
    [ -z "$changed" ] && continue
    changed_files_seen=1
    mapped=0
    case "$changed" in
      src/brainlayer/watcher.py)
        test_path="$TEST_ROOT/test_jsonl_watcher.py"
        if [ -f "$test_path" ]; then
          append_unique "$test_path"
          mapped=1
        fi
        ;;
      src/brainlayer/mcp/store_handler.py|src/brainlayer/queue_io.py|src/brainlayer/drain.py|src/brainlayer/store.py)
        for rel in test_store_handler.py test_write_queue.py test_brainstore.py; do
          test_path="$TEST_ROOT/$rel"
          if [ -f "$test_path" ] && ! is_real_db_test_file "$test_path"; then
            append_unique "$test_path"
            mapped=1
          fi
        done
        ;;
      src/brainlayer/vector_store.py|src/brainlayer/search_repo.py)
        for rel in test_source_class.py test_vector_store_schema_flags.py test_vector_store_upsert_transactions.py test_hybrid_search.py test_vector_store_readonly.py test_search_trigram_fts.py test_precompact_chunk_origin.py; do
          test_path="$TEST_ROOT/$rel"
          if [ -f "$test_path" ]; then
            append_unique "$test_path"
            mapped=1
          fi
        done
        ;;
      src/brainlayer/__init__.py)
        # A release bump touches this file and nothing else in src/. The generic
        # src/brainlayer/*.py rule below looks for tests/test___init__.py, finds nothing, and
        # escalates the whole 4,386-test suite for a one-line version string.
        #
        # This file has exactly two behaviours, and BOTH are named here so the mapping is not a
        # fail-open. test_version_consistency.py reads __version__ with `ast` and never imports
        # the package, so on its own it would pass while `import brainlayer` was broken --
        # e.g. a botched `from ._build import BUILD_SHA` fallback. test_build_sha.py runs
        # `import brainlayer` in a subprocess and asserts __build_sha__ both stamped and unstamped,
        # which is the import path the other suite cannot see.
        # BOTH must exist to count as mapped. A per-file `mapped=1` would let a deleted or
        # renamed test_build_sha.py silently narrow this back to the AST-only suite -- the same
        # fail-open, arriving through a missing sibling instead of an incomplete case list.
        # Partial coverage here is worse than none: it looks mapped and gates nothing.
        init_version_test="$TEST_ROOT/test_version_consistency.py"
        init_build_sha_test="$TEST_ROOT/test_build_sha.py"
        if [ -f "$init_version_test" ] && [ -f "$init_build_sha_test" ]; then
          append_unique "$init_version_test"
          append_unique "$init_build_sha_test"
          mapped=1
        fi
        ;;
      src/brainlayer/index_new.py)
        for rel in test_source_class.py test_ingest_t3.py test_context_pipeline.py; do
          test_path="$TEST_ROOT/$rel"
          if [ -f "$test_path" ]; then
            append_unique "$test_path"
            mapped=1
          fi
        done
        ;;
      tests/*.py|tests/**/*.py)
        test_path="$TEST_ROOT/${changed#tests/}"
        if [ -f "$test_path" ]; then
          if is_real_db_test_file "$test_path"; then
            changed_source_unmapped=1
            unmapped_changed_files+=("$changed")
          else
            append_unique "$test_path"
            mapped=1
          fi
        fi
        ;;
      src/brainlayer/*.py)
        module_name="$(basename "$changed" .py)"
        test_path="$TEST_ROOT/test_${module_name}.py"
        if [ -f "$test_path" ] && ! is_real_db_test_file "$test_path"; then
          append_unique "$test_path"
          mapped=1
        fi
        ;;
      scripts/run_tests.sh|.githooks/pre-push)
        test_path="$TEST_ROOT/test_run_tests_script.py"
        if [ -f "$test_path" ]; then
          append_unique "$test_path"
          mapped=1
        fi
        ;;
    esac
    if [ "$mapped" -eq 0 ]; then
      case "$changed" in
        src/brainlayer/*.py|src/brainlayer/**/*.py)
          changed_source_unmapped=1
          unmapped_changed_files+=("$changed")
          ;;
      esac
    fi
  done <<< "$changed_list"
}

run_pytest() {
  if [ "$BRAINLAYER_USE_UV" = "1" ] && command -v uv >/dev/null 2>&1; then
    uv run --extra dev pytest "$@"
  else
    pytest "$@"
  fi
}

cd "$ROOT_DIR"

prepush_cache_path=""
if [ "$BRAINLAYER_PREPUSH" = "1" ] && [ "$BRAINLAYER_PREPUSH_SCOPE" = "full" ]; then
  tree_hash="$(prepush_tree_hash)"
  if [ -n "$tree_hash" ]; then
    prepush_cache_path="$(prepush_cache_file "$tree_hash")"
    if [ -f "$prepush_cache_path" ]; then
      echo "SKIP: pre-push tree hash $tree_hash already passed"
      exit 0
    fi
  fi
fi

isolated_pytest_files=()
while IFS= read -r test_file; do
  isolated_pytest_files+=("$test_file")
done < <(collect_isolated_pytest_files)

if [ "$BRAINLAYER_PREPUSH_SCOPE" = "changed-only" ]; then
  map_changed_files_to_pytests
fi

if [ "$BRAINLAYER_PREPUSH_SCOPE" = "changed-only" ] && [ -n "$changed_files_scope_error" ]; then
  # Fail CLOSED. The skip below is only legitimate when this script actually established that
  # nothing changed; a scope it could not determine is not a scope it may call empty.
  echo "==> pytest unit suite"
  echo "FAIL: changed-only scope could not be determined: $changed_files_scope_error"
  echo "FAIL: refusing to call this run green for a scope that was never established."
  echo
  exit_status=1
  pytest_unit_cmd=()
elif [ "$BRAINLAYER_PREPUSH_SCOPE" = "changed-only" ] && [ "$changed_files_seen" -eq 0 ] && [ -n "${BRAINLAYER_PREPUSH_TAG:-}" ]; then
  # A RELEASE TAG is the one caller for whom the skip below is fail-open. Before tag scoping existed
  # a tag push ran the whole suite; a bump whose range maps nothing must not end up gating nothing
  # at all. So for a tag -- and only for a tag -- an empty measured scope escalates instead.
  echo "==> pytest unit suite"
  echo "WARNING: release tag $BRAINLAYER_PREPUSH_TAG scoped to an EMPTY change set."
  echo "WARNING: a release gate does not skip; running the FULL pytest unit suite."
  pytest_unit_cmd=(run_pytest "$TEST_ROOT/" -v --tb=short -m "$UNIT_MARK_EXPR")
elif [ "$BRAINLAYER_PREPUSH_SCOPE" = "changed-only" ] && [ "$changed_files_seen" -eq 0 ]; then
  # A changed-only run that found NOTHING has nothing to test, and escalating that to the full
  # suite was a fail-open: the most expensive thing this script can do, chosen precisely when the
  # evidence says there is nothing to do. On the M4 that full suite spawns
  # scripts/reembed_bgem3.py --test (a 2.5 GB embedding model holding fds on the production DB) --
  # the verified cause of a 14:22 UI stall. Skip loudly instead; the caller decides what to widen.
  echo "==> pytest unit suite"
  echo "WARNING: changed-only scope MEASURED an empty change set; SKIPPING the pytest unit suite."
  echo "WARNING: nothing was measured here. Set BRAINLAYER_CHANGED_FILES explicitly, or run with"
  echo "WARNING: BRAINLAYER_PREPUSH_SCOPE=full to ask for the whole suite on purpose."
  echo
  pytest_unit_cmd=()
elif [ "$BRAINLAYER_PREPUSH_SCOPE" = "changed-only" ] && [ "$changed_source_unmapped" -eq 1 ]; then
  # An unmapped SOURCE change is the opposite case: there IS something to test and no targeted way
  # to test it, so the escalation stays. It just no longer happens quietly -- it names what forced
  # it, so the fix (add a mapping) is visible instead of paid for on every push.
  echo "WARNING: changed-only scope found an unmapped source change; falling back to full pytest unit suite"
  echo "WARNING: unmapped: ${unmapped_changed_files[*]}"
  pytest_unit_cmd=(run_pytest "$TEST_ROOT/" -v --tb=short -m "$UNIT_MARK_EXPR")
elif [ "$BRAINLAYER_PREPUSH_SCOPE" = "changed-only" ] && [ "${#targeted_pytest_files[@]}" -gt 0 ]; then
  pytest_unit_cmd=(run_pytest "${targeted_pytest_files[@]}" -v --tb=short -m "$UNIT_MARK_EXPR")
elif [ "$BRAINLAYER_PREPUSH_SCOPE" = "changed-only" ] && [ -n "${BRAINLAYER_PREPUSH_TAG:-}" ]; then
  # Same release-gate rule as the empty-scope case above: a docs/changelog-only bump maps no pytest
  # target, and for a tag "nothing mapped" may not become "nothing gated".
  echo "==> pytest unit suite"
  echo "WARNING: release tag $BRAINLAYER_PREPUSH_TAG mapped no pytest targets in its range."
  echo "WARNING: a release gate does not skip; running the FULL pytest unit suite."
  pytest_unit_cmd=(run_pytest "$TEST_ROOT/" -v --tb=short -m "$UNIT_MARK_EXPR")
elif [ "$BRAINLAYER_PREPUSH_SCOPE" = "changed-only" ]; then
  echo "==> pytest unit suite"
  echo "SKIP: changed-only scope found no mapped pytest targets"
  echo
  pytest_unit_cmd=()
else
  pytest_unit_cmd=(run_pytest "$TEST_ROOT/" -v --tb=short -m "$UNIT_MARK_EXPR")
fi
if [ "${#isolated_pytest_files[@]}" -gt 0 ]; then
  for isolated_test in "${isolated_pytest_files[@]}"; do
    if [ "${#pytest_unit_cmd[@]}" -gt 0 ]; then
      pytest_unit_cmd+=("--ignore=$isolated_test")
    fi
  done
fi
if [ "$BRAINLAYER_PREPUSH" = "1" ] && [ "${#pytest_unit_cmd[@]}" -gt 0 ]; then
  for real_db_test in "${REAL_DB_TEST_FILES[@]}"; do
    if [ -f "$TEST_ROOT/$real_db_test" ]; then
      pytest_unit_cmd+=("--ignore=$TEST_ROOT/$real_db_test")
    fi
  done
fi

if [ "${#pytest_unit_cmd[@]}" -gt 0 ]; then
  run_step "pytest unit suite" "${pytest_unit_cmd[@]}"
fi
run_step \
  "pytest MCP tool registration" \
  run_pytest "$TEST_ROOT/test_think_recall_integration.py::TestMCPToolCount" -v --tb=short

if [ "${#isolated_pytest_files[@]}" -gt 0 ]; then
  run_step \
    "pytest isolated eval and hook routing" \
    run_pytest "${isolated_pytest_files[@]}" -v --tb=short
else
  echo "==> pytest isolated eval and hook routing"
  echo "SKIP: no isolated pytest files found under $TEST_ROOT"
  echo
fi

bun_tests=()
while IFS= read -r test_file; do
  bun_tests+=("$test_file")
done < <(collect_bun_tests)

if [ "${#bun_tests[@]}" -gt 0 ]; then
  if command -v bun >/dev/null 2>&1; then
    run_step "bun test suite" bun test "${bun_tests[@]}"
  else
    echo "FAIL (1): bun not found but TypeScript tests exist under $TEST_ROOT"
    echo
    exit_status=$(( exit_status | 1 ))
  fi
else
  echo "==> bun test suite"
  echo "SKIP: no .test.ts files found under $TEST_ROOT"
  echo
fi

shell_tests=()
while IFS= read -r test_file; do
  shell_tests+=("$test_file")
done < <(collect_regression_shell_tests)

if [ "${#shell_tests[@]}" -gt 0 ]; then
  for shell_test in "${shell_tests[@]}"; do
    run_step "regression shell $(basename "$shell_test")" bash "$shell_test"
  done
else
  echo "==> regression shell suite"
  echo "SKIP: no regression shell scripts found under $TEST_ROOT"
  echo
fi

if [ "$exit_status" -ne 0 ]; then
  echo "BrainLayer test gate failed."
else
  echo "BrainLayer test gate passed."
  if [ -n "$prepush_cache_path" ]; then
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$prepush_cache_path"
  fi
fi

exit "$exit_status"
