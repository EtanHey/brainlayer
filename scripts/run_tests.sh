#!/usr/bin/env bash

set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="${BRAINLAYER_TEST_ROOT:-$ROOT_DIR/tests}"
BRAINLAYER_USE_UV="${BRAINLAYER_USE_UV:-1}"
exit_status=0

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

run_pytest() {
  if [ "$BRAINLAYER_USE_UV" = "1" ] && command -v uv >/dev/null 2>&1; then
    uv run pytest "$@"
  else
    pytest "$@"
  fi
}

cd "$ROOT_DIR"

run_step "pytest unit suite" run_pytest "$TEST_ROOT/" -v --tb=short -m "not integration" -x
run_step \
  "pytest MCP tool registration" \
  run_pytest "$TEST_ROOT/test_think_recall_integration.py::TestMCPToolCount" -v --tb=short

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

if [ "$exit_status" -ne 0 ]; then
  echo "BrainLayer test gate failed."
else
  echo "BrainLayer test gate passed."
fi

exit "$exit_status"
