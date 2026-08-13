# BrainBar Non-Blocking Coverage Design

Date: 2026-08-13
Status: approved by lead ruling

## Problem

`StatsCollector` publishes nothing until `BrainDatabase.dashboardStats` completes. That method runs
four full coverage scans after the ordinary dashboard metrics. On the production 15 GB database,
each coverage query can exceed five seconds. The panel therefore remains in `.loading`, and the
menubar graphs stay flat even though their own bucket data is available.

A direct `chunk_fts_rowids` count is fast but not truthful. Lifecycle-only updates do not fire the
FTS triggers, so archived or superseded chunks remain mapped and coverage can exceed 100 percent.
Rejoining the map to `chunks` preserves correctness but exceeded an eight-second production fence
for both FTS signals.

## Considered approaches

1. **Two-phase in-memory refresh (selected).** Publish cheap dashboard metrics first, then compute
   exact coverage on a separate connection and task. This requires no schema change and restores
   both UI surfaces immediately.
2. **Persist a materialized coverage snapshot.** A worker would periodically write one small stats
   row. This gives fast restart-time coverage but adds schema and writer ownership, which are outside
   this hotfix's bounds.
3. **Stream individual foreground coverage signals.** This can reveal partial results but still
   couples refresh scheduling to four expensive scans and complicates truth/error presentation.

## Selected data flow

1. `StatsCollector` starts a hot refresh using `dashboardStats(includeSignalCoverage: false)`.
2. `BrainDatabase` skips every coverage query and returns the pipeline, queue, health, and graph
   buckets normally.
3. `StatsCollector` publishes that snapshot, clears `.loading`, and preserves a previously cached
   exact coverage snapshot when one exists.
4. After a short scheduling gap, an independent utility-priority task opens its own read-only
   database connection and runs the existing lifecycle-filtered coverage queries. The gap lets a
   collector that has already stopped cancel before entering synchronous SQLite work.
5. Until the first exact result arrives, coverage rows and chips render `computing…` and do not
   imply zero backlog or zero-percent coverage.
6. A successful exact result is merged into the current hot snapshot. The result remains cached
   across ordinary hot refreshes and is recomputed periodically rather than on every mutation.
7. A coverage failure leaves the hot snapshot live. Coverage remains unknown on first failure or
   keeps the last exact cached value after a later failure; it never changes global snapshot
   freshness to `.error`.

## State and cancellation

- The collector owns a separate coverage task, generation, last-attempt timestamp, and refresh
  interval.
- `stop()` cancels both hot and coverage tasks.
- A forced manual refresh may request coverage, but it does not cancel a coverage computation that
  is already in flight.
- Coverage publication is generation-checked so a late task cannot update a stopped collector.

## UI contract

- `DashboardStats` carries whether exact signal coverage is available.
- Coverage chips and expanded rows render `computing…` while unavailable.
- Coverage bars do not animate a fabricated zero value while unavailable.
- Menubar sparklines require no UI change: they already subscribe to the shared collector's hot
  activity buckets and recover as soon as the first hot snapshot publishes.

## Verification

- RED-first collector test holds the coverage provider open and proves the hot snapshot becomes live
  before coverage finishes, then proves exact coverage publishes after release.
- Database test proves the hot stats path omits coverage work and marks it unavailable.
- UI contract tests prove `computing…` is rendered for unavailable coverage.
- Existing exact coverage tests remain unchanged and green.
- Live deployment proof samples the installed app: dashboard refresh completes, the panel is not
  `.loading`, menubar bucket values are non-flat when activity exists, and coverage transitions from
  computing to exact without blocking the hot snapshot.

## Follow-up

File the lifecycle-trigger gap separately: changes to `archived_at`, `superseded_by`,
`aggregated_into`, and `status` must keep FTS lifecycle state synchronized. Do not fold that schema
work into this hotfix.
