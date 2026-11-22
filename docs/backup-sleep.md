# Backup sleep observations

Backup progress and terminal receipts keep the existing active-time measurements. They also record
`upload_wall_seconds` (sleep-inclusive elapsed time), `upload_slept_seconds`, and `sleep_intervals`
with `from`, `to`, and `seconds`. Snapshot receipts carry `snapshot_wall_seconds`,
`snapshot_slept_seconds`, and `snapshot_sleep_intervals`; the terminal `sleep_intervals` list includes
both phases. Failed snapshots and uploads retain observed sleep even without another progress tick. Verification-only recovery keeps the
completed upload's measurements, and replacing an expired upload session preserves the observer.

The observer compares deltas from two monotonic clocks. Darwin's `clock_gettime(3)` documents
`CLOCK_MONOTONIC_RAW` as sleep-inclusive and immune to time/frequency adjustments, and
`CLOCK_UPTIME_RAW` as sleep-exclusive (the converted `mach_absolute_time` value). The SDK's
`mach/mach_time.h` also documents that `mach_continuous_time` advances during sleep. Linux uses
`CLOCK_BOOTTIME` with `CLOCK_MONOTONIC`. Adjustable wall timestamps only label the inferred bounds;
an NTP/manual jump cannot trigger detection. A difference of at least 60 seconds records a pause.
DarkWake time when the CPU is awake counts as awake time, so the observed suspended duration can
be shorter than the span between closing the lid and full wake.

Bounds are estimates between observation ticks, not exact kernel sleep events: awake time after
wake can shift an inferred boundary by up to the surrounding progress tick. `sleep_note` is absent
without a detected pause. During transfer it says the upload resumed; after Drive accepts the file,
it says, for example, “paused 6 h 11 min while the Mac slept (07:09-13:20); uploaded the rest after wake”.

This changes observation only. It adds no admission gate, deadline, retry, or upload scheduling rule.

Historical logs predating these fields contain wall timestamps and sleep-exclusive elapsed time,
but no continuous-clock samples. Their replay therefore reconstructs a clock axis from timestamps
and corroborates pauses against `pmset -g log`; that historical reconstruction alone cannot exclude
a wall-clock adjustment. Injected-clock tests prove the production observer ignores wall jumps.
