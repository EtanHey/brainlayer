#!/usr/bin/env bash

set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="${BRAINLAYER_TEST_ROOT:-$ROOT_DIR/tests}"
BRAINLAYER_USE_UV="${BRAINLAYER_USE_UV:-1}"
UNIT_MARK_EXPR="${BRAINLAYER_PYTEST_MARK_EXPR:-not integration and not live}"
BRAINLAYER_PREPUSH="${BRAINLAYER_PREPUSH:-0}"
BRAINLAYER_PREPUSH_SCOPE="${BRAINLAYER_PREPUSH_SCOPE:-full}"
BRAINLAYER_PREPUSH_CACHE_DIR="${BRAINLAYER_PREPUSH_CACHE_DIR:-$ROOT_DIR/.git/brainlayer-prepush-cache}"
exit_status=0
declare -a targeted_pytest_files=()
changed_source_unmapped=0

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

changed_files() {
  if [ -n "${BRAINLAYER_CHANGED_FILES:-}" ]; then
    printf '%s\n' "$BRAINLAYER_CHANGED_FILES" | tr ',' '\n' | sed '/^$/d'
    return 0
  fi
  if git rev-parse --verify origin/main >/dev/null 2>&1; then
    git diff --name-only origin/main...HEAD
  else
    git diff --name-only HEAD~1...HEAD 2>/dev/null || true
  fi
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
  changed_source_unmapped=0
  local changed rel test_path module_name mapped
  while IFS= read -r changed; do
    [ -z "$changed" ] && continue
    mapped=0
    case "$changed" in
      src/brainlayer/mcp/store_handler.py|src/brainlayer/queue_io.py|src/brainlayer/drain.py|src/brainlayer/store.py)
        for rel in test_store_handler.py test_write_queue.py test_brainstore.py; do
          test_path="$TEST_ROOT/$rel"
          if [ -f "$test_path" ] && ! is_real_db_test_file "$test_path"; then
            append_unique "$test_path"
            mapped=1
          fi
        done
        ;;
      tests/test_*.py)
        test_path="$TEST_ROOT/$(basename "$changed")"
        if [ -f "$test_path" ] && ! is_real_db_test_file "$test_path"; then
          append_unique "$test_path"
          mapped=1
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
          ;;
      esac
    fi
  done < <(changed_files)
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

if [ "$BRAINLAYER_PREPUSH_SCOPE" = "changed-only" ] && [ "${#targeted_pytest_files[@]}" -gt 0 ]; then
  pytest_unit_cmd=(run_pytest "${targeted_pytest_files[@]}" -v --tb=short -m "$UNIT_MARK_EXPR")
elif [ "$BRAINLAYER_PREPUSH_SCOPE" = "changed-only" ] && [ "$changed_source_unmapped" -eq 1 ]; then
  echo "changed-only scope found an unmapped source change; falling back to full pytest unit suite"
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
