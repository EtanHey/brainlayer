# Watcher heartbeat and T3 linkage alarms

BrainBar checks `updated_at` in the canonical `watcher-health.json`, beside the
watcher's resolved database (or `BRAINLAYER_WATCHER_HEALTH_PATH`). Settings rereads
that file every 30 seconds while mounted, independently of supervisor job sampling.
A missing heartbeat for more than five minutes remains a real degraded state.
This reader fix requires a BrainBar app release.

For T3-origin Codex files missing projection linkage, every retry still raises
`t3_runtime_linkage_pending` and ingestion retains the deferred input. Only alarm
emission backs off: once per file/state transition, with hourly reminders. Appending
transcript data does not reset backoff; a changed session or projection selection does.

Status is persisted beside the resolved BrainLayer DB in
`t3-linkage-status/<sha256-of-absolute-source-path>.json`. Each record includes `path`,
`code`, `state` (`pending` or `resolved`), `fingerprint`, `first_seen_at`, and (when pending)
`last_alert_at` in Unix seconds. Status contains no transcript content. Resolution
updates the record; recurrence immediately alerts again. Corrupt records are reported
and repaired; failed status writes or a busy status lock stay loud and never suppress
the pending exception. The alarm change ships with the Python formula.
