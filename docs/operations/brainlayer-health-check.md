# BrainLayer Health Check

`brainlayer health-check` is a lightweight launchd tick for live BrainLayer stability.

The scheduled LaunchAgent is `scripts/launchd/com.brainlayer.health-check.plist`. It runs every 300 seconds, at load, through the standard `brainlayer-env-run.sh` wrapper:

```bash
brainlayer health-check --json --heal
```

## Checks

- Hotlane BrainBar embedding daemon is running.
- No running hotlane command line is disabling the embedding backlog with `--backlog-batch 0`.
- Active chunks missing semantic vectors are decreasing across ticks. One unchanged tick is tolerated; the second unchanged tick alarms.
- BrainBar's served MCP socket can answer a `brain_search` canary with at least one result.
- The JSONL backup attempt log has a fresh terminal receipt. A verified upload and a
  verified `no-op` are healthy but remain distinct in `jsonl_backup.state`; failed,
  malformed, missing, or older-than-36-hour receipts are critical issues.

The backup process appends a timezone-aware `attempted_at` to every normal result and
to caught terminal failures. The independent five-minute health check reads that
receipt, so a backup process that never starts cannot make its own absence look
healthy. The threshold can be changed with
`BRAINLAYER_JSONL_BACKUP_MAX_AGE_SECONDS`; the monitored log follows
`BRAINLAYER_JSONL_BACKUP_LOG_PATH` when set. Legacy receipts without
`attempted_at` are interpreted at the launchd schedule of 05:00 in the producer
machine's local timezone; this fallback disappears after the next stamped run.

The missing-vector count is exact, but it computes the ID difference through the
covering `chunks` and `chunk_vectors_rowids` indexes before reading chunk payloads.
This keeps the scheduled check independent of database payload size.

The check has a 45-second internal deadline. A query that exhausts that budget is
interrupted and the state file is still refreshed with `slow_check: true`, the
stage name, and the elapsed duration.

## Self-Heal

With `--heal`, the check uses `launchctl kickstart -k` for cheap recovery:

- `com.brainlayer.hotlane-brainbar` when hotlane is dead, backlog is disabled, or missing vectors are climbing/stalled.
- `com.brainlayer.brainbar-daemon` when the BrainBar MCP canary fails or returns zero results.

Before a kickstart, the check reads the launchd process state. Processes in an
uninterruptible `U`/`D` state are not kickstarted repeatedly; the check records a
`heal_backoff` action and the existing circuit breaker escalates after repeated
failed ticks.

The command writes the latest successfully measured missing-vector count to
`~/.local/share/brainlayer/health-check-state.json`; a timed-out tick preserves
the previous count and marks the state as slow.

The independent Tier-0 watchdog treats `slow_check: true` as unhealthy even
when the state file was just refreshed, and treats any state as stale after 900
seconds. Its first alert is immediate; repeat alerts for the same failure class
are suppressed for 1,800 seconds while recovery attempts continue.

## Logs

Launchd output goes to:

- `~/Library/Logs/brainlayer/health-check.out.log`
- `~/Library/Logs/brainlayer/health-check.err.log`

Manual dry run:

```bash
brainlayer health-check --json --no-heal
```
