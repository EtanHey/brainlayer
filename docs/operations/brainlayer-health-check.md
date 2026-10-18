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
- BrainBar's served MCP socket can answer a `brain_search` canary successfully, including zero results.
- The JSONL backup attempt log has a fresh terminal receipt. A verified upload and a
  verified `no-op` are healthy but remain distinct in `jsonl_backup.state`; failed,
  malformed, missing, or older-than-36-hour receipts are critical issues. A
  separate critical issue identifies two or more complete 24-hour windows with
  no durable attempt receipt.

The backup process appends a timezone-aware `attempted_at` to every normal result and
to caught terminal failures. The independent five-minute health check reads that
receipt, so a backup process that never starts cannot make its own absence look
healthy. The threshold can be changed with
`BRAINLAYER_JSONL_BACKUP_MAX_AGE_SECONDS`; the monitored log follows
`BRAINLAYER_JSONL_BACKUP_LOG_PATH` when set. Multi-night escalation defaults
to two nights and can be raised with
`BRAINLAYER_JSONL_BACKUP_ABSENCE_ALERT_NIGHTS` (values below two clamp to two).
It detects an absent attempt receipt, not a bundle that was uploaded and later
removed by retention. Legacy receipts without
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

- `com.brainlayer.hotlane-brainbar` when hotlane is dead or backlog is disabled.
  Missing-vector warnings remain visible. Backlog-based healing requires no
  decrease in missing vectors for `max_stalled_ticks` comparisons (default two),
  at least 300 seconds apart, followed by the existing heal threshold. One heal
  counter spans rising and flat counts. A decrease resets the episode. Vector
  growth is not hotlane progress: drain can embed queued events independently.
  Transient SQLite busy failures are warnings and preserve the last successful
  baseline, stall counter and heal episode; a later successful sample continues
  the comparison. An unavailable current count cannot itself trigger that heal.
- `com.brainlayer.brainbar-daemon` when the BrainBar MCP canary fails retrieval or transport.

Before a kickstart, the check reads the launchd process state. Processes in an
uninterruptible `U`/`D` state are not kickstarted repeatedly; the check records a
`heal_backoff` action and the existing circuit breaker escalates after repeated
failed ticks.

The command writes the latest successfully measured missing-vector count to
`~/.local/share/brainlayer/health-check-state.json`; a timed-out tick preserves
the previous count and marks the state as slow.

Missing-progress comparisons live in the private state file; the public health
response schema is unchanged. A fresh, healthy drain heartbeat whose
`drained_total` advances resets the queue-backlog heal counter and circuit breaker
only when measured queue depth also stays level or shrinks. A trickle of writes
under a growing backlog does not reset escalation. Backlog alerts remain
visible. Stale, unknown or stalled drain health cannot reset the breaker; a later
stall can trigger healing again. Job alerts clear through their producer's next
successful run, never through this policy.

The independent Tier-0 watchdog treats `slow_check: true` as unhealthy even
when the state file was just refreshed, and treats any state as stale after 900
seconds. Its first alert is immediate; repeat alerts for the same failure class
are suppressed for 1,800 seconds while recovery attempts continue.

## Idle machines

No new transcripts, zero indexed today, an old last-store timestamp, unchanged
offsets, and a flat `drained_total` with an empty queue are normal during inactivity.
Watcher polls and drain cycles should keep advancing while their processes run.
A fresh verified transcript-backup `no-op` receipt is healthy even when the last
archive upload is old; the upload timestamp remains historical evidence.

Real failures still need attention: stopped processes, stalled watcher polling,
undrained queued work, failing scheduled jobs, and missing, stale, malformed or
failed backup attempts. A recent source write is evidence of work to ingest,
never a requirement for keeping an otherwise healthy idle machine green.

The [2026-10-18 audit](idle-machine-audit-2026-10-18.md) records both Macs' observed
state and the remaining native backup presentation follow-up. These are read-only
observations, not deployment or native visual verification.

## Logs

Launchd output goes to:

- `~/Library/Logs/brainlayer/health-check.out.log`
- `~/Library/Logs/brainlayer/health-check.err.log`

Manual dry run:

```bash
brainlayer health-check --json --no-heal
```
