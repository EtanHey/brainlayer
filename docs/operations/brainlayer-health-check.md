# BrainLayer Health Check

`brainlayer health-check` is a lightweight launchd tick for live BrainLayer stability.

The scheduled LaunchAgent is `scripts/launchd/com.brainlayer.health-check.plist`. It runs every 300 seconds, at load, through the standard `brainlayer-env-run.sh` wrapper:

```bash
brainlayer health-check --json --heal
```

## Checks

- Hotlane BrainBar embedding daemon is running.
- Hotlane command line is not disabling the embedding backlog with `--backlog-batch 0`.
- Active chunks missing semantic vectors are decreasing across ticks. One unchanged tick is tolerated; the second unchanged tick alarms.
- BrainBar's served MCP socket can answer a `brain_search` canary with at least one result.

## Self-Heal

With `--heal`, the check uses `launchctl kickstart -k` for cheap recovery:

- `com.brainlayer.hotlane-brainbar` when hotlane is dead, backlog is disabled, or missing vectors are climbing/stalled.
- `com.brainlayer.brainbar` when the BrainBar MCP canary fails or returns zero results.

The command always writes the latest missing-vector count to `~/.local/share/brainlayer/health-check-state.json`.

## Logs

Launchd output goes to:

- `~/Library/Logs/brainlayer/health-check.out.log`
- `~/Library/Logs/brainlayer/health-check.err.log`

Manual dry run:

```bash
brainlayer health-check --json --no-heal
```
