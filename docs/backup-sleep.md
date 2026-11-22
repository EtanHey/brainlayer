# Backup sleep observations

Backup progress and terminal receipts keep the existing active-time measurements. They also record
`upload_wall_seconds` (sleep-inclusive elapsed time), `upload_slept_seconds`, and `sleep_intervals`
with `from`, `to`, and `seconds`. Snapshot receipts carry `snapshot_wall_seconds`,
`snapshot_slept_seconds`, and `snapshot_sleep_intervals`; the terminal `sleep_intervals` list includes
both phases. A run-level observer also covers archive hashes, fsyncs, credential refresh, folder
checks and other work between phases. `run_wall_seconds` and `run_slept_seconds` measure this
invocation, including failed or abandoned attempts; run and phase seconds overlap, so do not add
them together. On recovery, historical intervals follow the reused artifact in `sleep_intervals`.
Switching to a fresh snapshot drops earlier-invocation history while preserving pauses observed
in this invocation. Failed snapshots and uploads retain
observed sleep even without another progress tick. Verification-only recovery keeps completed
upload counters and intervals; its note is rendered with the current span rules.

The observer compares deltas from two monotonic clocks. Darwin's `clock_gettime(3)` documents
`CLOCK_MONOTONIC_RAW` as sleep-inclusive and immune to time/frequency adjustments, and
`CLOCK_UPTIME_RAW` as sleep-exclusive (the converted `mach_absolute_time` value). The SDK's
`mach/mach_time.h` also documents that `mach_continuous_time` advances during sleep. Linux uses
`CLOCK_BOOTTIME` with `CLOCK_MONOTONIC`. Adjustable wall timestamps only label the inferred bounds;
an NTP/manual jump cannot trigger detection. A difference of at least 60 seconds records a pause.
Supported clock pairs are required explicitly. Missing clocks or a failed clock read raise a clear
`RuntimeError` explaining that sleep observation is unavailable, rather than an attribute error or
an invented zero. Darwin and Linux are supported; other platforms need a proven clock pair first.
Uploads without telemetry (including the JSONL callers) construct no sleep observer.
The final run sample is best-effort: a clock failure records `sleep_observation_error` and preserves
the original outcome/error and terminal receipt, rather than letting observation prevent the receipt.
DarkWake time when the CPU is awake counts as awake time, so the observed suspended duration can
be shorter than the span between closing the lid and full wake.

Each raw interval carries `bounds: "inferred"`. Bounds are estimates between observation ticks, not exact kernel sleep events: awake time after
wake can shift an inferred boundary by up to the surrounding progress tick. `sleep_note` is absent
without a detected pause. For the note only, fragments separated by **20 minutes or less** join
into one outer span, covering typical DarkWakes. The duration still sums suspended seconds only;
raw intervals stay unmerged. Separate sleeps more than 20 minutes apart keep separate spans, and
spans crossing midnight include dates. A transfer pause says the upload resumed; only a completed
upload that itself observed sleep says “uploaded the rest after wake”. Other phases say the backup
continued. These labels remain estimates, including in the single joined span.

An upload shares one phase observer across in-process session replacements. `upload_wall_seconds`
therefore spans those attempts; the existing `upload_elapsed_seconds` still starts at the latest
uploader call. Never subtract those fields to infer suspended time: use `upload_slept_seconds`.
The run observer retains pauses from an abandoned upload when a fresh snapshot is prepared.

The observability producer forwards the note and finite nonnegative snapshot, upload and run
wall/slept seconds under `backups.local_snapshot`; these are optional schema-v2 fields. This supplies
BrainBar's reader without claiming an installed UI change.

This changes observation only. It adds no admission gate, deadline, retry, or upload scheduling rule.

Historical logs predating these fields contain wall timestamps and sleep-exclusive elapsed time,
but no continuous-clock samples. Their replay therefore reconstructs a clock axis from timestamps
and corroborates pauses against `pmset -g log`; that historical reconstruction alone cannot exclude
a wall-clock adjustment. Injected-clock tests prove the production observer ignores wall jumps.
