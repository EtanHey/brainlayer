#!/usr/bin/env bash
# Smoke test: prove the BrainLayer MCP transport exposes brain_search on the
# first tools/list and can serve a query quickly over the BrainBar daemon socket.
#
# Usage:
#   scripts/smoke/firstturn-brainlayer-smoke.sh [socket_path]
#
# Environment:
#   DEADLINE_SECS=8    Response deadline for tools/list + query
#   QUERY=agent-html   Query passed to brain_search

set -euo pipefail

SOCK="${1:-/tmp/brainbar.sock}"
DEADLINE_SECS="${DEADLINE_SECS:-8}"
QUERY="${QUERY:-agent-html}"

for cmd in date grep python3 socat timeout; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "FAIL: required command not found: $cmd"
    exit 2
  fi
done

if [[ ! -S "$SOCK" ]]; then
  echo "FAIL: socket not found: $SOCK"
  exit 2
fi

# MCP stdio framing = newline-delimited JSON-RPC.
init='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"firstturn-smoke","version":"1.0"}}}'
inited='{"jsonrpc":"2.0","method":"notifications/initialized"}'
listtools='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
json_query="$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$QUERY")"
callsearch='{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"brain_search","arguments":{"query":'"$json_query"',"num_results":1}}}'

echo "== first-turn BrainLayer smoke =="
echo "socket: $SOCK   deadline: ${DEADLINE_SECS}s   query: $QUERY"

start=$(date +%s)
set +e
out="$(printf '%s\n%s\n%s\n%s\n' "$init" "$inited" "$listtools" "$callsearch" \
  | timeout "$DEADLINE_SECS" socat - "UNIX-CONNECT:$SOCK" 2>/dev/null)"
rc=$?
set -e
end=$(date +%s)
elapsed=$((end - start))

if [[ -z "$out" ]]; then
  echo "FAIL: no output from MCP socket (rc=$rc, ${elapsed}s)"
  exit 2
fi

if [[ "$rc" -eq 124 ]]; then
  echo "FAIL: MCP socket command timed out after ${DEADLINE_SECS}s"
  echo "--- raw (first 600 chars) ---"
  echo "${out:0:600}"
  exit 2
fi

# CHECK 1: brain_search present on the first tools/list response (id:2).
if grep -q '"brain_search"' <<<"$out"; then
  echo "PASS check1: brain_search exposed on first-turn tools/list (${elapsed}s)"
else
  echo "FAIL check1: brain_search NOT in first tools/list (rc=$rc, ${elapsed}s)"
  echo "--- raw (first 400 chars) ---"
  echo "${out:0:400}"
  exit 1
fi

# CHECK 2: the brain_search tools/call (id:3) returned a result, not a timeout/error.
if grep -q '"id":3' <<<"$out" \
  && ! grep -q '"isError":true' <<<"$out" \
  && ! grep -qi 'timeout\|DB may be locked\|Error:' <<<"$out"; then
  echo "PASS check2: brain_search('$QUERY') returned without timeout (${elapsed}s total)"
else
  echo "FAIL check2: brain_search call timed out or errored"
  echo "--- raw (first 600 chars) ---"
  echo "${out:0:600}"
  exit 1
fi

echo "RESULT: PASS - first-turn BrainLayer parity OK on $SOCK"
