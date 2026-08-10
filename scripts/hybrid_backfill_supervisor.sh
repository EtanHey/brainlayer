#!/usr/bin/env bash
set -u

ROOT="/Users/etanheyman/Gits/brainlayer-prod"
ARTIFACT_DIR="$ROOT/artifacts/hybrid-backfill-sample-2026-06-16-r1"
LOG="$ARTIFACT_DIR/logs/supervisor.log"
PIDFILE="$ARTIFACT_DIR/logs/supervisor.pid"
LOCKDIR="$ARTIFACT_DIR/logs/supervisor.lock"
SLEEP_SECONDS="${1:-180}"

mkdir -p "$ARTIFACT_DIR/logs"
cd "$ROOT" || exit 2

if mkdir "$LOCKDIR" 2>/dev/null; then
  printf '%s\n' "$$" > "$PIDFILE"
  trap 'rm -rf "$LOCKDIR"; rm -f "$PIDFILE"' EXIT INT TERM
else
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    printf '[%s] supervisor already running pid=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$(cat "$PIDFILE")" >> "$LOG"
    exit 0
  fi
  rm -rf "$LOCKDIR"
  mkdir "$LOCKDIR" || exit 3
  printf '%s\n' "$$" > "$PIDFILE"
  trap 'rm -rf "$LOCKDIR"; rm -f "$PIDFILE"' EXIT INT TERM
fi

printf '[%s] supervisor detached poller start pid=%s sleep=%ss root=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$$" "$SLEEP_SECONDS" "$ROOT" >> "$LOG"

while true; do
  printf '[%s] supervisor step start\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$LOG"
  python3 scripts/hybrid_backfill_monitor.py step >> "$LOG" 2>&1
  rc=$?
  printf '[%s] supervisor step exit=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$rc" >> "$LOG"

  if [ "$rc" -ne 0 ]; then
    printf '[%s] supervisor exiting after failure\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$LOG"
    exit "$rc"
  fi

  if [ -f "$ARTIFACT_DIR/judge-hybrid-vs-baseline-summary.json" ]; then
    printf '[%s] supervisor exiting after sample verdict artifact\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$LOG"
    exit 0
  fi

  sleep "$SLEEP_SECONDS"
done
