# Badge alert evidence

`badge-state.json` keeps every unsuppressed issue in `alerts.active`, including
warnings. Only critical (or unknown-severity) issues turn on the red dot. A
healthy check has empty active and suppressed lists. BrainBar retains warning
messages in its informational menu row when the dot is off.

A critical is visible by default. The only currently supported exception is a
search canary failure explicitly expected during an owned snapshot/maintenance
interval. The canary response must carry `error.data.expected_episode`, with
`kind` (`snapshot` or `maintenance`), nonempty `owner` and `episode_id`, timezone
qualified `started_at`, `observed_at`, `deadline`, and `expected_codes` containing
`brain_search_canary_failed`. The responding owner must attest its own live
episode; a static configuration, message substring, host load, or guessed owner
is never evidence. Existing BrainBar responses without this record stay visible.

The observation must be at most 300 seconds old, within its interval, and not
in the future. The fixed deadline is exclusive and at most 3600 seconds after
the episode started. Both the producer and BrainBar reader enforce these bounds;
sleep grace does not extend suppression. Suppressed entries retain the episode
record for inspection. Missing, malformed, stale or expired proof restores the
critical, even before another producer run. Owners must retain the original
start and deadline when reobserving an episode.

Queue age/depth, watcher stalls, drain/hotlane failures, lock wedges, and data-loss
alerts have no exception here. Positive throughput alone cannot establish timely
recovery. Existing health-check escalation thresholds continue to apply; no
recovering-episode exception is inferred from net queue or embedding counts.

Ship the Python producer and Swift reader together: the reader now distinguishes
warning visibility from critical badge eligibility and retains that distinction
when it clears a recovered `job_alert_<key>`.
