# Changelog

## [Unreleased]

### Added
- `brain_expand` MCP tool — expand a chunk_id with N surrounding chunks for full context retrieval
- Groq enrichment backend support (`GROQ_API_KEY`, `BRAINLAYER_GROQ_URL`, `BRAINLAYER_GROQ_MODEL`)
- `[style]` optional extra: ChromaDB vector store as alternative backend
- `faiss-cpu` added to `[brain]` optional extra for fast ANN search
- Deferred embedding in `brain_store` — async embedding pipeline for faster writes (#76)
- Eval suite + entity injection in UserPromptSubmit hook (#72)
- C6 eval coverage expansion — 18 new test cases for search scenarios (#75)
- C7 failed query mining script — automated discovery of low-score searches (#74)
- C1+C2 lifecycle hardening — stale MCP process cleanup + WAL checkpoint on start (#73)
- Post-RRF importance and recency reranking (US-002) — boosts high-importance and recent chunks
- FTS5 expansion to index `summary`, `tags`, `resolved_query` fields (US-001)
- KG entity quality — validation, prompts, and cleanup pipeline (#69)
- KG rebuild pipeline with audit fixes (119 entities, 153K entity-chunk links) (#67)
- Groq rate limiter for enrichment backend (#68)

### Fixed
- BrainBar's backup busy rejection now carries a structured `error.data.brainlayer_error_code = "backup_snapshot_busy"` (`admitted: false`). The backup client treats that code, the legacy message, or an `isError` tool result that wraps either as busy, so a wrapped rejection no longer unlinks a still-running earlier VACUUM INTO target. `initialize` reports `serverInfo.backupSnapshotInProgress`. An aged, unmarked attempt from the current writer epoch is reclaimed only when BrainBar positively reports no snapshot in progress in that same epoch. A missing, unknown or restarted status keeps the attempt (Refs #1413).
- `brain_graph.py`: replaced correlated subquery with literal `'claude_code'` — eliminates N+1 subquery on large DBs (PR #62)
- Removed `tree-sitter>=0.21.0` from core dependencies
- `brainlayer serve` docs: removed non-existent `--http` flag
- `brain_expand` for manual chunk IDs — fixed lookup path (#78)
- Consolidated DB paths — single canonical `~/.local/share/brainlayer/brainlayer.db` across all scripts (#77)
- `brain_search` now returns `chunk_id` in results (#66)
- DB lock resilience — retry + queue for MCP writes and reads (#65)
- `format` parameter renamed to avoid Python builtin shadowing (BUG-005)
- Entity merge in cleanup script — UNIQUE constraint violation (#a125d68)
- Search validation hardened with backfill coverage (a108aff)

### Changed
- Service JSON compatibility: doctor and drain-liveness alarm details retain retired `enrichment_backlog: null` for one release. Doctor code `queue_paused_enrichment` becomes `legacy_queue_held`; drain-health state `drain_paused` becomes `drain_legacy_queue_held`. Removed codes: `enrichment_unloaded`, `enrichment_idle_with_backlog`, `enrichment_backlog_count_failed`. Enrichment CLI options `--enrichment-label` and `--enrichment-plist-path` are removed; callers must stop passing them.
- MCP tool count updated from 7 to 8 (brain_expand added)
- Brain graph clustering documented as Leiden + UMAP (was incorrectly listed as HDBSCAN + UMAP)
- Test count: 698 → 929 (after eval suite expansion and search hardening)
- `[kg]` optional extra documented in README and Optional Extras section
- Search: hybrid RRF reranking now incorporates importance and recency signals
- Embeddings: deferred from synchronous to async pipeline

## [1.5.96] - 2026-11-10

### Hardening

- fix(alerts): a replacing step's success clears the replaced step's alert (#1744)
- fix(ci): the allowlist ratchet reads its base from history it actually has (#1747)
- fix(calendar): retry a pending database backup upload the same day (#1743)
- fix(doctor): version check compares what is installed, not dev checkouts (#1746)
- fix(supervisor): keep status responsive and defer transport outages (#1749)

## [1.5.95] - 2026-11-09

### Search

- BrainBar's production server now uses the router's 6 s hybrid budget. It still passed 0.8 s, so installed 1.5.94 served 0/20 live agent searches through hybrid (#1741).

## [1.5.94] - 2026-11-09

### Search

- Restore native hybrid routing with bounded vector reads: unscoped BrainBar calls run as the orchestrator consumer instead of deny-all, point reads replace vec0 IN-scans, and the helper budget rises to 6 s (#1725).
- Bound filtered KNN sizing and expose the fallback route in compact `_meta` (#1717).
- Scope linked worktrees to their canonical repository, so search from inside a worktree finds the repo's memory (#1713).

### Health

- Doctor: per-probe synthetic vectors, so residual probes no longer tie with the round-trip probe and raise a false FATAL (#1739).

### CI and tests

- Take the required swift (macos-15) skip on Linux when brain-bar is unchanged (#1738).
- Gate the macOS retirement and no-enrichment ratchets to the paths they guard (#1709).
- Require pre-push test mappings for new source modules (#1723).
- Preserve watcher WAL across Codex scratch reads in tests (#1735).

## [1.5.93] - 2026-11-09

### Codex .jsonl.zst ingest

- Bounded decoder and read-only reconciliation (#1708).
- Isolate bounded tailer and terminal record quarantine (#1710).
- Preserve archive resume and author attribution (#1727).
- Persist confirmed archive admission and report held checkpoints (#1728).
- Validate streamed provenance without retaining header payloads (#1729).
- Reconcile manifest candidates and dedup outcomes atomically (#1730).
- Require durable complete manifest replay receipts (#1731).
- Expose bounded backfill CLI and retain archive headers (#1732).
- Preserve Codex author context in manifest replay (#1736).
- test(zst): retain bounded acceptance guards (#1733; landed via #1736)

### Other changes

- fix(backup): resume committed bytes after transient Drive HTTP failures (#1726)
- Close backup supervisor signal lifecycle windows (#1705).
- Bound unmapped changed-only checks with smoke tests (#1721).
- Retry bounded supervisor status reads before steps (#1711).
- Align legacy author-model fixture with safe resume (#1720).
- Carry Codex turn_context author model across realtime reads (#1651).
- Surface failed progress with a fresh heartbeat (#1707).
- Deleted JSONL is not a parse alarm; verify bytes before resuming a same-inode path; name real log files (#1656).
- Bound EOF drain while requests remain unsent (#1691).
- Compact status menu and Dashboard attention list (#1698).

## [1.5.92] - 2026-11-08

- BrainBar no longer asks to reconnect Drive for a Production OAuth app (#1677).
- Count Codex threads hosted by app-server in the Agents row (#1687).
- BrainBar reads recent history and tolerates partially written history lines (#1679).
- Indexing retries unresolved backlog files (#1680).
- Model #1685's strict weekly receipt read and re-pin backup-verdict at main 5fda86d2 (#1699).
- Pending-store replay drops queue rows only after a WAL durability barrier (#1653).
- Retry brief staging lock contention (#1684).
- Give the replacement supervisor a load-tolerant readiness grace (#1697).
- Don't judge p95 latency from 4 samples (#1695).
- Pure-deletion filter requires patch structure, so prose is kept (S45) (#1672).
- Leave supervised timeout to parent (#1688).
- Tell a stalled embedding backlog apart from one paused by machine load (#1686).
- Bound T3 alarm counters to retained hourly totals (S43) (#1673).
- A full-suite branch push gets the release-gate budget (#1694).
- Copy-only realtime writer rehearsal script for #1661 (S44b) (#1674).
- Durably replace trimmed queue files (#1689).
- Rotate runs.jsonl before the history reader's 4 MiB window (#1693).
- Publish merged files exclusively from hashed bytes (#1690).
- Reject damaged weekly cleanup receipts (#1685).
- Record who holds the maintenance lock (names the holder on a busy hold) (#1671).
- Unify consent deadline parsing (#1683).
- Admit one helper worker at a time (#1682).
- Preserve instance lock on release (#1678).
- Stop hotlane tests renicing the suite process; guard the whole suite (#1676).
- Give a tag-only push a 5400s release-gate budget (#1675).

## [1.5.91] - 2026-11-07

### BrainBar
- The status card shows one plain line per problem, each saying what happens next and whether anything is needed from you (#1663). A failed nightly index used to appear three times (job alert, calendar result, retry line); it is now one line. A disk measurement that could not be taken reads "Couldn't check backup disk space. Rechecks every 5 min. Nothing to do."
- `brain_entity` resolves names case-insensitively before falling back to substring matches (#1632); served `brain_store` accepts an optional self-reported `caller_model` (#1643); compact `brain_search` rows stay under the wire-size gate while probes read the served shape (#1648, includes #1638).

### Index
- A backlog that drains every night no longer fails the nightly index (#1662). The stall alarm compared each pass with the all-time minimum, so four passes that each indexed more than arrived (M1, 11-02→11-05) still ended "index exited 1". A pass now fails only when it made no progress; a backlog that keeps outgrowing indexing prints a non-failing `INDEX_BACKLOG_GROWING` line.
- The first calendar run refuses a missing database explicitly; `bootstrap-db` creates one atomically (#1640). Per-turn authoring model is recorded for Claude Code and Codex chunks (#1639).

### Health
- A busy Mac no longer turns the backup disk check amber (#1666): when the measurement times out under load, the last real measurement (up to 30 min old) is reused and labelled as such.
- Maintenance exits 70 for real failures and 75 only for deliberate deferrals, and the calendar and BrainBar match each outcome to its run (#1641). `jobs restart --verify` allows for ExitTimeOut and startup settling (#1636).

### Data
- Runtime schema changes are recorded in `migration_events` in the same transaction as the change, and the FTS repair (tables, triggers, backfill, row-id map) is atomic (#1642, Refs #618). The startup WAL checkpoint still runs only after the one-time chunk-origin backfill.
- `content_hash` is rewritten by every content-changing dedupe merge, with an explicit copy-only repair command (#1637, Refs #623); DB-copy proof and runbook in #1649 (rehearsed on a real M1 copy: 166 stale → 0, fingerprints equal).

### Backup and Drive
- A busy maintenance lock at 04:00 defers the maintenance step instead of aborting the night (#1668). On the M4 on 11-07 that abort skipped jsonl-backup, wal-checkpoint and backup-daily, so no backup ran; an inherited lease that cannot be re-locked still aborts.
- One elapsed deadline covers the calendar kill through the snapshot timeout receipt (#1647). A retryable Google token-endpoint failure (5xx/429) is temporary, not "invalid" (#1633). Scrub-at-rest refusals name a value-free cause and gate (#1645).

### Tests, CI and formal
- CalendarGuard models the supervisor's stall-grace extensions and the TLA+ source pins are current (#1665); CI forwards INT/TERM/HUP to the suite group (#1646); pre-push maps `t3_provenance.py` (#1630); flaky admission-wait bound fixed (#1664); test-only hardening #1631, #1634, #1635.

### Also in this release
- fix(scrub-at-rest): refuse quiesce without quiet-window run budget (Refs #1073) (#1644).
- fix(backup): structured BrainBar busy contract + same-epoch snapshot status (Refs #1413) (#1650).
- fix(backup): hash retained stalls only where used; surface the disk-full deadlock in BrainBar (Refs #1385) (#1654).
- feat(search): optional user-config lexical alias source on both brain_search paths (Refs #995) (#1655).
- test(#1377): proof fixtures for weekly VACUUM deadline and hold release (report-only) (#1659).
- fix(watcher): isolate non-UUID realtime chunk ids by full stem + project (future ingest only) (#1661).
- fix(ci): run_with_deadline kills past-deadline suites again on macOS without waitid (#1669).
- fix(hooks): exclude retired chunks from ambient session-start and prompt context (#1670).

## [1.5.90] - 2026-11-04

### Backup
- The nightly Drive upload survives slow or contended uplinks (#1622). 8 MiB chunks with a fixed 120 s per-request deadline were aborted and resent from zero whenever a chunk needed longer (an 8 MiB PUT at ~45 KB/s needs ~180 s), so a slow night never confirmed a byte. Chunks are now 64 MiB, deadlines scale to a 32 KiB/s floor rate (bounded by the run deadline), and the receipt records progress every 30 s (confirmed bytes, rate, retries, probes, deadline hits). Probe-before-retransmit and resumability are unchanged.
- `python -m brainlayer.backup_daily --help` (or any unknown argument) prints usage and exits instead of starting a production backup (#1626, Fixes #1438). A bare invocation and the explicit `run` verb still run a backup; every production caller passes no arguments.

### Supervisor
- Periodic jobs keep running across Mac sleep (#1624). Deadlines now count time asleep, missed ticks coalesce into one run, and a delayed or blocked periodic run logs its reason. Process and cleanup deadlines stay monotonic.

### BrainBar
- A stale health verdict shows amber "Health check stale since HH:mm" instead of falling back to a red dot from old calendar history (#1624). A last-known red stays red; plan §3d (the dot is the outcome verdict) holds.

### MCP
- Same-length core tool labels steer deep, multi-source recall to the brain-worker sub-agent while `brain_store` stays direct (#1621; skill-creator w24 eval: 0/15 → 15/15 deep-recall delegation on Opus).

### Drive
- Offline `brainlayer backup auth --status` names the local reconnect deadline when consent expires within two days (#1623, Fixes #1364).
- A Google-rejected refresh (`invalid_grant`) names the known local consent deadline and says it is not Google's revocation time (#1627, Fixes #1343).

### T3 provenance
- `unavailable`, `schema_drift` and `linkage_invalid` runtime alarms log on change plus hourly reminders instead of CRITICAL every poll (~2.9k lines/day); every observation still raises (#1628, Fixes #1402).

### Tests
- Supervisor shutdown assertions carry the child's stderr tail (#1625, Fixes #1362).

## [1.5.89] - 2026-11-03

### Maintenance
- A busy nightly WAL checkpoint no longer fails the night (#1619). After bounded TRUNCATE retries stay busy (readers open), the step runs a PASSIVE checkpoint, reports "deferred" and exits 0; the 15-minute fallback truncates later. Real errors and a WAL above 4 GiB still fail.

### BrainBar
- The menu starts clean (#1615). A job alert superseded by a later success clears (a successful fallback checkpoint clears the busy alert); older timed alerts show dimmed as "Earlier: …" and do not count in the header. Alert times live in a sibling `job-alerts.times.json`; `job-alerts.json` stays a plain string map so older BrainBar builds read it unchanged. While a Drive upload is running, the database row says "Uploading to Drive since HH:mm".

### Bridge
- The MCP stdio bridge logs each reconnect to BrainBar, and a disconnect error says the request outcome is unknown (#1616, shipped in 1.5.88's tree).

## [1.5.88] - 2026-11-03

### Supervisor
- Policy-only launcher changes now reach already-consolidated Macs (#1617). `install.sh consolidate` journals a launcher plist transition when the only differences are `ProcessType`, `Nice`, `LowPriorityIO` or `SoftResourceLimits`, publishes it, restarts only the changed launcher, and records the new configuration hash. It runs before a keg upgrade continues in the same invocation. Any other launcher difference still refuses with "explicit replan required". It refuses without changes while the nightly calendar or a maintenance/backup lock is active.
- This delivers 1.5.87's maintenance change (#1612): on the next consolidate the maintenance launcher moves from `ProcessType=Background` to `Standard`, so the nightly Drive upload is no longer throttled.

### Tests
- Pre-push maps `wal_checkpoint.py` to its real tests instead of falling back to the full suite (#1614).

## [1.5.87] - 2026-11-03

### Maintenance
- The nightly Drive upload is no longer throttled (#1612). The maintenance LaunchAgent ran as `ProcessType=Background`, which clamps every thread to the lowest scheduling band and background-class networking; a 10.7 GB upload ran at about 0.1-0.3 MB/s. The template now uses `ProcessType=Standard`, keeping `Nice 10` and per-step low-priority disk I/O. On an already-consolidated Mac, the next `install.sh consolidate` applies it through the journaled policy reconcile (#1607).
- The backup receipt records the upload thread's real scheduler priority and QoS (`upload_thread_policy`), so a throttled upload is visible in the receipt instead of reading as "default".

## [1.5.86] - 2026-11-03

### BrainBar
- The menu and the dashboard tell one story (#1610).
  - The header says "All good" only when every check is green and there are no notes; otherwise "Memory safe · N things to look at".
  - One problem is one line: the nightly backup chain (snapshot alert, full-maintenance alert, the maintenance job failure and the snapshot diagnostic) becomes "Nightly backup failed N nights in a row — last: … · next try …".
  - At most 3 problem lines, then "+N more — Open Dashboard".
  - The Backups card shows an amber dot and "last run failed …" for a backup whose last nightly run failed, even while its verified copy is still inside the window, so the card and the menu agree.
  - The Today card shows a small "not measured yet today" instead of a large dash.
- The Drive "reconnect" warning clears as soon as Drive is re-approved (#1609). BrainBar re-reads Drive status when the token file changes and when the menu or dashboard opens; before, only every 30 minutes.

## [1.5.85] - 2026-11-02

### Supervisor
- Resource policy follows the keg. On an already-consolidated Mac, `install.sh consolidate` re-derives each component's `Nice`, `LowPriorityIO`, `ProcessType` and `SoftResourceLimits` from the new keg's launchd templates, through a journaled policy-reconcile transition. Before this, `supervisor/jobs.json` kept whatever policy it was first copied with, so plist policy changes never deployed.
  - Only those four keys change. Commands, environment, paths and schedules stay journal-pinned, and every existing custody check still refuses a manifest change the reconcile did not journal.
  - A crash at any point leaves the old or the new verifiable state. Recovery restarts the supervisor and requires readiness. A TLA+ model is checked with TLC in CI.
  - A component with no keg template keeps its pinned policy and is logged. Off macOS, pinned policies are kept, because the templates carry Darwin-only keys (#1607).
- This delivers 1.5.84's hotlane change (#1604). On the next install, the hotlane's `LowPriorityIO` flips to false, and the supervisor resets the inherited disk-I/O throttle, so a throttled reader no longer holds the SQLite write lock in front of foreground writers.

## [1.5.84] - 2026-11-02

The pipes rebuild. The BrainBar dot now answers one question, "is my memory at risk, or has a pipe stopped, right now?", from outcomes, not from job events. The rules behind it are model-checked.

### Health and the BrainBar dot
- The dot is the outcome verdict: a pure function of the measured facts (verified local and offsite copies, transcripts, Drive auth, integrity, ingest, embedding, stores, search, disk, monitor). A failed or skipped job alone is a note, never a red dot. Checks it cannot observe are unknown, not red (#1583, #1587, #1588).
- Backup receipts are authoritative. Revocation (`verification_error`, `drive_md5_match: false`) beats any persisted maximum, and a local copy removed after a verified upload still counts as safe (#1589, #1595). Abandoned and superseded snapshots do not count (#1596).
- Recovery needs evidence. Red on ingest, stores or search clears only after 2 consecutive observed good checks; an amber-age tick, an unobserved tick or a wake-grace tick does not count (#1600).
- Wake grace is one window per stall episode, never renewed, so a Mac that sleeps between short wakes can no longer keep a stalled embedder, a stuck store queue or a dead watcher green (#1602).
- BrainBar renders the verdict and publishes its UI status. Fresh verdicts drive the dot, while the Swift diagnostic rows stay visible (#1584, #1585).

### Backup and maintenance
- The nightly backup lifts the darwin-background CPU and disk throttling it inherited from launchd. Integrity was running 7.4× and zstd 14× slower under it, which caused the nightly timeouts. Receipts record the run's own `io_policy`, `iopol` and `darwin_background`, including on reuse (#1597).
- A verified archive with the same date is reused instead of failing the nightly with a target conflict (#1599).
- A bounded read-only snapshot primitive (`VACUUM INTO` from a read-only connection) is added, and rollback failures are reported (#1592, #1593).
- Weekly VACUUM is optional: below a 10% freelist it is not needed, and with no fresh snapshot it is deferred without raising a maintenance alert (#1582).

### Hotlane and embeddings
- A busy writer slot never discards computed vectors. They carry to the next cycle, and the `BusyError` contract is kept (#1594).
- CPU budget: the hotlane runs at nice 20 with torch threads capped (2 on M4, 1 on M1). The backlog yields under high load or a `hotlane.pause` marker without losing or duplicating vectors, keeps store-queue backpressure, and still gets one batch every 300 s. A rolling CPU producer and an installed-Mac ratchet row measure CPU only while the backlog runs (#1605).
- The hotlane is no longer disk-I/O throttled: `LowPriorityIO` is explicitly false, so the supervisor resets the inherited throttle. A throttled reader holding the write lock was blocking foreground writers (#1604).
- Merging into an existing chunk no longer deletes its vector unless the embedding input changed (#1590), and redelivering an identical original is idempotent (#1591).

### Doctor
- Supervisor children are recognized, and the source version checks run (#1581). A fresh embedding backlog no longer reads as a fatal lag (#1586).

### Verification
- A TLA+ model of backup → snapshot → VACUUM → dot, plus the hotlane's grace and hysteresis, is checked with TLC in CI. Every bug fixed tonight reproduces as a counterexample at its pre-fix commit, and the fixed code conforms to the model tick for tick across 3,000 simulated behaviours per check (#1601).
- Every bug fixed tonight is a lint rule or ratchet row, including a suite guard that stops tests writing the real home's job alerts or logs (#1603). The recovery oracle converges before its timeout (#1598).

## [1.5.83] - 2026-11-01

### Doctor
- `brainlayer doctor`'s index-completeness census reads the routing, FTS and vector tables from one read snapshot. Before, each statement took its own autocommit snapshot, so chunks committed mid-census showed up as false `index_completeness_aux_to_chunk` / `vector_parity_gap` findings. That stopped the 1.5.82 M4 install. The snapshot closes with SQL `ROLLBACK`, so Python 3.12+ `autocommit=True` callers are covered too (#1576).

### BrainBar
- While its window is open, BrainBar is a regular app: it shows in the Dock and in Cmd-Tab, and the Dock icon brings the window back, even when minimized. Closing the last window returns it to menu-bar only. `LSUIElement` stays on, so launch is unchanged.
- The window's size and position survive quit and relaunch. Geometry events that arrive after a quit or close begins no longer overwrite the saved frame (#1577).

### Backup
- The raw-reclaim compress path records per-chunk stage timings: raw read and hash, compress, archive write and full fsync, punch, block reduction, volume delta, free floor, and final hash. They go into the backup receipt as `chunk_stage_seconds`, and a progress line every 8 chunks means a run killed at its timeout still leaves partial timings. Measurement only: ordering, durability, chunk size and timeouts are unchanged. This is for the M1's nightly backup timeouts (#1578).

Release scope: Python package + Homebrew formula + BrainBar app 1.5.83 (cask)

## [1.5.82] - 2026-11-01

### Backup
- Choose how to prove a raw-snapshot reclaim from the staging volume's APFS snapshot inventory, probed once per backup. With no snapshot, the per-file block drop proves the reclaim even while unrelated writers allocate on the same volume. That was the cause of the 2026-10-31 and 2026-11-01 nightly failures on the M4: the blocks dropped by 64 MiB, while another process consumed about 250 MB during the check. With a snapshot, or if the probe is uncertain, the re-sampled volume check still applies. The no-block-release abort and the 2 GiB free-space floor are unchanged, and the receipt records `reclaim_basis` (#1574).

Release scope: Python package + Homebrew formula + BrainBar app 1.5.82 (cask)

## [1.5.81] - 2026-10-31

### Backup
- Re-check free space for up to 2 seconds before stopping a backup on the raw-reclaim check, instead of failing on one immediate reading. The block-release check, the 2 GiB free-space floor and a lasting shortfall still stop it. A stopped backup now reads "Latest backup stopped; last verified backup: <date>" with the measured detail, the same on the status line, the job alert and the Backups page (#1569).

### BrainBar
- Show "Retry failed steps" only when pressing it would retry something. Otherwise say in plain words what happens next and when ("runs again automatically tonight during scheduled maintenance, after 04:30"), keep a refused retry's reason, and show only progress while a retry runs (#1570).
- Associate the health-relay LaunchAgent with BrainBar, so it groups under the app in Login Items (#1568).

### Knowledge graph
- Add `python -m brainlayer.linkedin_import`, which imports a LinkedIn connections export as people and companies with `works_at` and `linkedin_connection` relations. It is copy-only unless run with `--live --confirm-sha256`, re-runs are idempotent, identity merges are guarded against joining different people, and output contains counts only (#1572).

### Tests
- Measure the recovery-oracle ack p95 over 100 real admissions instead of 6, so one slow fsync no longer fails CI; the 500 ms bounds are unchanged (#1571).

Release scope: Python package + Homebrew formula + BrainBar app 1.5.81 (cask)

## [1.5.80] - 2026-10-31

### Health relay
- Deliver durable, append-only alerts to the lead's inbox, with unique event identities and safe local writes. Missing or inaccessible sources no longer stop delivery or clear an incident, and stale or unverifiable bootstrap state is refused (#1541, #1542, #1543).
- Read M1 health over SSH, including badge, watcher and pause signals. Keep muted incidents open, handle deferred stale checks, harden configuration and probes, and reconcile migrated escalation identities (#1545, #1546, #1552).
- Escalate only user-visible outages that remain unacknowledged for over 30 minutes; backup alerts stay in the lead's collab (#1549, #1558).

### Nightly maintenance and recovery
- Retry failed daytime-safe steps under calendar custody, with confirmation and a clear outcome in BrainBar. Steps that need the nightly window remain deferred (#1555, #1556).

### WAL growth fixes
- Reclaim the weekly VACUUM's WAL before releasing the maintenance hold (#1554).
- Add an independent checkpoint fallback that releases calendar, backup and maintenance admission probes before SQLite work. Known limit: a WAL over 1 GB is alerted, not reclaimed, until the nightly run (#1565).
- Apply a shared WAL recycle policy to the supported Python writer paths and BrainBar: a 256 MB default recycle limit, with separate checkpoint defaults for Python and interactive BrainBar writes. The limit applies after WAL reset, not while readers pin the WAL; remaining raw writer paths are follow-up work (#1566).

### BrainBar panels that no longer mislead
- Clear a failed night after complete recovery, preserve distinct runner failures, and name the steps that actually failed or were deferred (#1550, #1553).
- Show one cause for an abandoned night without step details, and use one status truth across the icon, menu, Jobs row and Dashboard (#1559, #1560).
- Treat a routine partial vector scan as normal, explain when an unconfirmed BrainBar database blocks backup, and count saved memories rather than agent activity in the memory chart (#1562, #1563, #1564).

### Updater
- Preserve Gatekeeper approval across upgrades and check the linked BrainLayer keg rather than the first installed version (#1548, #1561).

### Backup
- Skip transcript sources that disappear during bundling (#1544).

### Tests
- Pin backup reclaim capability and bound producer completion on Darwin; make lock-recovery and recovery-oracle tests wait for the writes they own (#1547, #1551, #1557).

Release scope: Python package + Homebrew formula + BrainBar app 1.5.80 (cask)

## [1.5.79] - 2026-10-28

### Fixed
- Increase the supervisor guard timeout from 30 to 120 seconds and make it configurable (#1537).
- Preserve BrainBar legacy ownership when checking processes for termination (#1521).
- Tolerate torn process-list rows during activation scans (#1538).
- Emit one notification per incident (#1539).
- Tolerate host stalls with an absolute guard grace period (#1536).

Release scope: Python package + Homebrew formula + BrainBar app 1.5.79 (cask)

## [1.5.78] - 2026-10-27

### Fixed
- A killed or abandoned nightly run is recorded once as an abandoned nightly event, counts as a failure, and is retried by the next scheduled run. A torn journal line or alert-write failure no longer blocks a run (#1517).
- BrainBar shows a dot only when no verified Drive copy exists within 36 hours, backup age has been unknown for over 60 minutes, the daemon or watcher is killed (including a watcher silent for over 15 minutes), or maintenance fails on two consecutive nights. A single failure shows a dated notice without a dot. “No action needed” and the three-failures/14-days masking are removed (#1518).
- Blocking calendar I/O runs off the heartbeat loop, with a 120-second budget per operation so the guard can still kill and reap a stuck worker. Compute the backup budget before spawning its worker (#1519).
- Before each timeout kill, queue a best-effort `supervisor_guard_timeout` row (parent and birth identity, mode, heartbeat age, process groups and reason) for `guard-events.jsonl` in the supervisor and calendar custody directories; the write finishes after the kill and can be dropped, and never delays the kill. The supervisor also records overdue operations that suppress heartbeats (#1520).

Release scope: Python package + Homebrew formula + BrainBar app 1.5.78 (cask)

### Upgrade note
- The legacy-ownership kill check (#1521) is NOT included in 1.5.78. Machines using legacy ownership should wait for that fix.

## [1.5.77] - 2026-10-26

### Fixed
- Maintenance observability now validates the schedule plist and reports unusable schedules instead of treating them as healthy (#1510).
- Health checks determine supervisor ownership from migration state, and healing keeps paused workers from being stranded (#1511).
- BrainBar uses one status snapshot across its icon, menu, dashboard, jobs and backups. Dots indicate real failures; a deferred weekly maintenance stays neutral unless maintenance has failed on three consecutive scheduled runs or no weekly maintenance has succeeded for more than 14 days. High Contrast changes re-render the icon (#1512). A newer verified backup copy clears an earlier backup-failure streak (#1515).
- After a verified Drive upload, backup removes the local archive only when checksum and custody checks pass; older archives must also match the current file's MD5. A weekly pin is released the day after it was set (#1507).

### Tests
- Make the store busy-retry regression deterministic so lock contention is checked reliably (#1513).
- Isolate the disk-space purge-poll test's fake clock so other threads' sleeps cannot leak into it (#1516).

Release scope: Python package + Homebrew formula + BrainBar app 1.5.77 (cask)

## [1.5.76] - 2026-10-26

### Fixed
- Use `brainlayer disk --json` as the disk truth: it reports available space, free-now space, and purgeable-space estimates with their measurement source (#1503). Disk descriptions mark estimated values with `≈` (#1505).
- Backup preflight now escalates the macOS purge request from urgency 3 to urgency 4 when free-now space is short but Available is enough (#1503). Urgency 4 is unverified on a real purge; this release does not claim that runtime qualification.
- On staging volumes that support hole punching (APFS), backup now needs about 24 GB instead of about 34 GB for a 22 GB database; other filesystems keep the previous budget. Streaming compression reduces peak headroom while preserving the snapshot and verification gates (#1504).
- Avoid false backup aborts on short final reclaim chunks on APFS, while retaining per-file block and free-space-floor checks. Stop retrying Drive downloads after the local decoder closes its pipe (#1506, the release gate).

### Release scope
- Python package and Homebrew formula release only. Retain BrainBar 1.5.75 and its existing cask; the batch has no BrainBar source changes.

## [1.5.70] - 2026-10-22

### Fixed
- The supervisor keeps sending heartbeats while starting children or writing their custody records, so slow disk and process startup no longer make its own guard kill it and repeatedly restart the watcher. These operations run off the event loop with one custody writer; a stuck operation still stops heartbeats after its 120-second budget. Guard readiness now allows 30 seconds for interpreter startup and reports why it failed (#1464).
- A child whose start identity cannot be established is marked failed and its gated wrapper is reaped without running the command. The supervisor retries with backoff instead of crashing its heartbeat loop. Unexpected startup-task failures are also recorded per child (#1465).

### Tests and CI
- Concurrent pytest children use separate temporary directories for retention cleanup (#1452). The iCloud backup naming test freezes its clock (#1458), the BrainBar store-budget test checks the selected lock policy with debug-only hooks (#1461), and the activation lock test isolates its clock and lock doubles from other threads (#1463).
- The disposable recovery oracle adds interrupted-index, encoder failure/stall, drain-restart and bounded catch-up checks, with negative controls. The known MERGED stale-vector case remains a strict expected failure tracked in #1450; this adds coverage rather than fixing that defect (#1457).
- Swift CI has a 15-minute job budget, and changed-only pushes map workflow changes to their owning Python checks (#1462).

### Upgrade notes
- Formula-only supervisor hotfix: BrainLayer moves to 1.5.70; the BrainBar app and cask stay at 1.5.69.

## [1.5.69] - 2026-10-22

### Fixed
- The calendar runner keeps its watchdog heartbeat alive while a step finishes: receipt fsync, alert locking and macOS notifications run off the event loop, so slow completion work no longer makes the guard kill the runner and lose every later step after an index timeout. Completion writes finish even if cancellation arrives, and runner failures and cleanup errors still produce a summary (#1440).
- BrainBar's job-log links open the actual log for every calendar step, including index, backup snapshot, P0 counter, WAL checkpoint, decay and FTS repair. The log directory follows the calendar's `--logs-dir` from its LaunchAgent or supervisor configuration, with the usual logs directory as the fallback. A step that has not written its log yet says so (#1437).
- The Watcher chart shows "Updating…" when database activity makes flow counts temporarily unavailable but watcher health is still known. Stale or missing health stays amber or red with a matching status message; "Evidence unavailable" is reserved for an absence of watcher evidence (#1437).
- The Watcher chart's tooltip and accessibility text now include the reason and heartbeat age when the watcher needs attention or has stopped (#1442).

### Changed
- Developer tooling: app bundles, release extracts, build output and pytest temporary files now live under `<worktree>/.build.noindex/`, which Spotlight excludes and worktree removal cleans up. `docs.local/` is text-only, enforced by `scripts/check_stray_apps.py` (#1435).

### Upgrade notes
- Full release: the formula and the BrainBar app (cask) both move to 1.5.69.

## [1.5.68] - 2026-10-21

### Fixed
- BrainBar no longer shows a false red menu-bar dot on a healthy Mac. The health check now runs under the supervisor, but BrainBar only looked for its old LaunchAgent plist, so it reported "Badge freshness unknown: cadence unknown" and went red. BrainBar now reads the cadence from that plist first, then from the `com.brainlayer.health-check` entry in the supervisor's `jobs.json` (skipped when the entry is `enabled: false`). A cadence from 30 to 3600 seconds is used as-is; anything else falls back to 300 seconds. An unknown cadence on its own no longer raises an alarm; a badge older than twice the cadence still shows "Needs attention" and the red dot. The menu's status line is capped at 60 characters (was 72), and its tooltip keeps the full reason (#1433).
- BrainBar's Jobs page no longer paints healthy notes amber. Informational owner notes (for example `com.brainlayer.maintenance · last result: ok` or `Supervisor component … · PID …`) are plain neutral text. Failed jobs, configured-off jobs and runs that succeeded with warnings keep the amber ⚠, and a note nobody classified defaults to amber (#1430).
- When that night's weekly maintenance ran in place of the nightly and finished ok, the nightly row says "Nightly covered by weekly" and its last-run cell on Jobs and Backups reads "Covered by weekly · <time>". It used to say "Not yet run under the supervisor" or "No calendar run recorded". Coverage counts only for the same scheduled night, and only when the journal order shows the weekly after any nightly result. A missing scheduled date or missing order fails closed to the nightly's own result, and a later night with no nightly run goes back to the nightly's own failed or not-yet-run state. The weekly row now shows its own `last result: ok` (#1430, #1420).
- The Dashboard's job-status refresh works again. Its 60-second timer was created inside the view body and replaced on every re-render, so it never fired, and the Dashboard could keep showing a stale "things need you" from the sample taken when it opened. A single loop owned by the view's lifetime now samples on open and then every 60 seconds (#1430).
- The DB-copy dot and wording follow the copy's age. When observability reports backups `stale`, a verified offsite DB copy older than the freshness window turns the Dashboard's DB dot and the Backups recovery check red. The DB-copy codes read as sentences ("The database copy in Google Drive is older than N hours", "No verified database copy in Google Drive", …) instead of "DB backup error: …" (#1430; fixes #1420).
- BrainBar's local-copy listing recognizes zstd snapshots (`YYYY-MM-DD.db.zst`) as well as `.db` and legacy `.db.gz`. This Swift change merged with #1421 in 1.5.67, but 1.5.67 was formula-only, so this is the first BrainBar app that ships it (#1421).
- Health alerts for repeated scheduled-job failures give the real count. Nightly maintenance (`com.brainlayer.maintenance`) now reads `com.brainlayer.maintenance: failed N nights in a row; last exit E`. N is the actual streak, not the fixed heal-attempt constant (3), and the exit appears once; the text no longer says "crashloop". Other scheduled (interval) jobs report their real `N failed runs`. A run still in progress is never counted and never clears an alert, and a completed run behind a live one is counted exactly once. Alert and clear thresholds are unchanged, and so are daemon heal attempts and their wording. Alerts saved before this fix, which carry no count, keep their old text (#1427).
- T3 ingest: every exception text written to `t3-health.json` is now scrubbed with the at-rest secret scrubber (`scrub_for_storage`) before it is cut to 500 characters, matching what #1424 did for the indexing-failure record. This covers reader failures (`failures[].error`), the SQLite schema race (`failures[].sqlite_error`) and the source-selection failure (`failures[].message`). Codes, error types, alert reasons and the re-raised alarm are unchanged (#1429).

### Tests
- The recency checkpoint test in `tests/test_consumer_scoping.py` freezes its clock at 2026-06-15, so its May/June fixtures no longer swap order as the real date moves on. Test-only; no runtime change (#1432).

### Upgrade notes
- Full release: the formula and the BrainBar app (cask) both move to 1.5.68. The app is the first since 1.5.66.
- `BRAINLAYER_SUPERVISOR_JOBS_PATH` overrides where BrainBar reads the supervisor's `jobs.json` (default `~/.local/share/brainlayer/supervisor/jobs.json`). `BRAINLAYER_HEALTH_CHECK_PLIST_PATH` still names the LaunchAgent plist, which is checked first.

## [1.5.67] - 2026-10-20

### Changed
- DB snapshots are compressed with the zstd CLI (`zstd -3 -T0 --check`) to `YYYY-MM-DD.db.zst` instead of Python gzip -6 (about 12 s against 8+ minutes on the real 20.7 GB snapshot). Every reader (verify, restore, raw cleanup, resume, retention, observability, the weekly gate, eval snapshot materialization) accepts both `.db.zst` and legacy `.db.gz`. Local and opt-in Drive retention keep the newest verified copy of each format; on Drive only an object whose exact file ID has an `uploaded` + `verified` receipt counts as verified, and an unreadable log trashes nothing (#1421).

### Fixed
- The snapshot timeout is sized from the database (DB + WAL) at 200 s per GB, rounded up to five minutes, with a 45-minute floor and a 90-minute ceiling, instead of a flat 45 minutes that killed the 20.75 GB MBP snapshot on 2026-10-20. The backup's own deadline fires up to 60 s inside the calendar budget so its timeout receipt is written before the calendar stops the process group (#1419).
- Backup observability `freshness` covers the offsite DB copy as well as transcripts: a missing, unverified or stale DB copy makes `freshness` `stale`; its reason is reported in `error_type` when no more specific error is already set (#1417).
- The receipt journal reader skips and counts non-JSON lines. The known `drive upload progress: N/M bytes` lines are counted as noise, not uncertainty; a journal with no readable receipt fails with `BackupReceiptUnreadableError` and keeps the journal and all local snapshots (#1418).
- T3 ingest: a chunk or conversation id that the at-rest secret scrubber would rewrite (for example a thread named `...-mockapi-r1`) falls back to a stable `t3:sha256:<digest>:<i>` id (a digest of the thread id for `conversation_id`); the original `message_id`, `t3_thread_id` and `source_uri` stay in metadata, and ordinary ids are unchanged. This clears the `t3_ingest_unhealthy` `indexing_failure` (`chunk identifier requires redaction`) that had blocked T3 indexing since 2026-10-18. The `t3_indexing_failure` health record now carries the underlying `error`, scrubbed before truncation to 500 characters (#1424).

### Upgrade notes
- Formula-only release: the BrainBar app and its cask stay at 1.5.66. #1421 also widens BrainBar's local-copy listing to `.db.zst`; that Swift change is on main but NOT shipped, so until the next BrainBar release the app's local-copy listing does not recognize `.db.zst` snapshots.
- The tap formula gains `depends_on "zstd"`. If no zstd binary is found, the snapshot falls back to gzip and records `compressor_fallback` in its receipt.
- `BRAINLAYER_BACKUP_COMPRESSOR=gzip` rolls back to `.db.gz` output; any value other than `zstd` or `gzip` is refused.
- `BRAINLAYER_BACKUP_ZSTD_PATH` names the zstd binary (absolute and executable, or the run fails); otherwise `/opt/homebrew/bin/zstd`, `/usr/local/bin/zstd`, then `PATH` are tried, because launchd jobs run with `PATH=/usr/bin:/bin`.
- `BRAINLAYER_BACKUP_SNAPSHOT_TIMEOUT_SECONDS` keeps its meaning as an explicit override that replaces the sized default; leave it unset to get the size-derived budget. Either way the step is still clipped to finish before 06:00.
- Restore drill: decompress new snapshots with `zstd -d`; legacy `.db.gz` snapshots still use `gunzip`.
- A stale or unverified offsite DB copy now shows as stale `freshness` in observability, even when transcripts are covered.

## [1.5.66] - 2026-10-19

### Fixed
- Weekly VACUUM is gated on a fresh, restore-checked local snapshot instead of an offsite copy, and the Drive upload runs after maintenance; live scrubbing keeps its 24-hour offsite-backup contract (#1401).
- BrainBar shows verified idle transcript backups and pending DB uploads as neutral states across Backups, Dashboard and Jobs. A recent local snapshot cannot hide a stale offsite copy, and unrelated DB or weekly maintenance errors no longer turn covered transcripts red (#1406; closes #1396, #1398).
- The backup capacity gate is sized by the largest per-phase peak (raw copy, gzip, Drive download, restore) instead of a raw+restore sum; a lost-response retry gets its own explicit capacity check (#1408).
- A BrainBar "snapshot already in progress" rejection no longer releases or unlinks an earlier, still-running VACUUM INTO target; the run fails with `BrainBarSnapshotBusyError` and the next run reserves the surviving attempt's growth (#1411).
- Maintenance writer holds get a capped deadline (`BRAINLAYER_MAINTENANCE_HOLD_MAX_S`) that falls before the maintenance window closes, and supervisor ticks enforce overdue holds with generation-tied escalation. VACUUM starts only when its size-based estimate fits inside the hold, and is interrupted via SQLite progress callbacks before the hold cutoff; the non-interruptible final copy-back is bounded by the calendar cut and the supervisor watchdog (#1409).
- The calendar index pass backs off and retries on a lost writer-busy race inside its existing budget, resuming from its checkpoint, instead of exiting 75 `resident-writer-busy` (#1410).
- Isolated test HOMEs no longer accumulate copies of the vite-plus node runtime: a test HOME over 32 MiB fails its test, and pytest keeps temp trees only for failures (#1412).

### Changed
- Release the BrainLayer formula and BrainBar app together. This is the first BrainBar app build to ship the Settings watcher-heartbeat re-read from #1400 (merged in 1.5.65, which was formula-only).

### Upgrade notes
- Weekly calendar order is now index → backup-snapshot → weekly maintenance → … → backup upload.
- New environment knobs: `BRAINLAYER_BACKUP_LOCAL_MAX_AGE_HOURS` (local snapshot freshness for the weekly gate, default 6), `BRAINLAYER_MAINTENANCE_HOLD_MAX_S` (writer-hold cap, default 3600), `BRAINLAYER_VACUUM_SECONDS_PER_GB` (VACUUM duration estimate, default 30).
- Observability schema v2 adds optional `backups.local_snapshot` and `transcript_attempt` evidence; schema v1 is unchanged.
- BrainBar app: new neutral backup states (idle transcript coverage, pending upload).

## [1.5.65] - 2026-10-18

### Fixed
- A transient `ps` timeout or OS error during the BrainBar activation process scan defers activation instead of raising a false "activation monitor refused" alert. A scan alert is raised after 12 consecutive scan failures, provided both BrainBar jobs are unloaded at the time of the alert; the next successful scan clears it (#1399).
- A persistent non-scan `OSError` no longer flaps the activation alert (cleared after each successful scan, then re-reported every tick). A legacy `BrainBar activation monitor refused: [Errno …]` alert is cleared once, by a one-shot migration on the first successful scan; later errno failures stay raised (#1403).
- The `t3_runtime_linkage_pending` CRITICAL alarm is deduplicated: emission happens once per file/state transition with hourly reminders, backed by a durable per-file status in `t3-linkage-status/` beside the resolved DB. Every pending observation still raises and ingestion still defers (#1400).

### Documentation
- Health-check guide: an idle machine (no new transcripts, old last-store timestamp, flat drain totals with an empty queue) is healthy; the MCP canary passes with zero results. Adds regression tests and the 2026-10-18 idle-machine audit (#1397).

### Upgrade notes
- The BrainBar Settings watcher-heartbeat re-read from #1400 is on main but NOT shipped: it needs a BrainBar app release.
- Formula-only release: BrainBar and its cask remain at 1.5.61.

## [1.5.64] - 2026-10-18

### Added
- Add explicit MCP-verified BrainBar activation and supervisor recovery after deferred cask activation (#1392).

### Fixed
- Verify running BrainBar executable identity against the installed bundle and re-arm automatic activation after reinstall (#1393).

### Upgrade notes
- New `brainlayer brainbar activate` (MCP-verified, idempotent). The BrainLayer supervisor re-activates BrainBar automatically after a cask upgrade or reinstall (once per installed build; never while Homebrew is running; failure raises a job alert).
- Pairs with the upcoming tap cask change whose postflight no longer launches BrainBar (#1391), so macOS can carry the previous version's approval forward and upgrades need no click. Native no-consent/real-MCP acceptance remains outstanding; this release alone does not guarantee no consent prompt.
- Automatic activation requires the formula's always-on supervisor and both cask labels to be unloaded. Manual activation and `brainbar verify-running` provide explicit activation and installed-build verification after Homebrew returns.
- Formula-only release: BrainBar and its cask remain at 1.5.61.

## [1.5.63] - 2026-10-18

### Fixed
- Remove uncompressed VACUUM backup snapshots only after their gzip replacement is verified and durable (#1387).

### Upgrade notes
- After its `.db.gz` is durable (F_FULLFSYNC + directory fsync) and proven to match (own run: gzip size + SQLite header; leftovers from earlier runs: full SHA-256 of the decompressed gzip equals the raw), the uncompressed VACUUM snapshot is removed; otherwise both are kept and the receipt / capacity errors name the retained raw.
- Formula-only release: BrainBar and its cask remain at 1.5.61.

## [1.5.62] - 2026-10-18

### Fixed
- Resume stalled Drive backup uploads and preserve local snapshots until a newer backup is uploaded and verified (#1384).

### Upgrade notes
- The next backup run resumes a stalled upload of the same unchanged snapshot when it is less than 36 hours old (default); its upload session is saved privately with mode 0600.
- A stalled local backup is deleted only after a newer backup is uploaded and verified. If disk capacity is insufficient, the run fails with an explicit error naming the retained file instead of deleting it.
- Receipts carry `snapshot_created_at`; backup reuse and observability measure age from that snapshot time. Legacy receipts fall back to the filename date at UTC midnight.
- Formula-only release: BrainBar and its cask remain at 1.5.61.

## [1.5.61] - 2026-10-18

### Fixed
- Retry calendar watch status timeouts within the existing readiness deadline (#1361).
- Isolate Drive consent capture readers from installed production paths (#1369).
- Include calendar jobs in BrainBar health and retain measured failures while observability loads (#1353).
- Explain calendar maintenance substitutions and deferrals in BrainBar (#1354).
- Checkpoint bounded calendar index passes, yield to resident writers, and bound custody with stalled-yield escalation (#1371).
- Skip empty scrubbed chunks without aborting CLI or T3 indexing (#1372).
- Defer nightly Drive upload until maintenance finishes while preserving weekly daily-backup reuse (#1375).
- Keep BrainBar attention stable and explain index deferrals (#1376).
- Converge calendar backlog, isolate resource faults, version checkpoints, and escalate repeated deferrals (#1380).

### Changed
- Integrate the calendar, Drive capture, BrainBar health, maintenance, and index checkpoint fixes (#1373).
- Integrate calendar ordering, BrainBar attention, and index convergence with aligned regression tests (#1382).
- Release the BrainLayer formula and BrainBar app together.

### Upgrade notes
- The first calendar index pass after upgrading replays every file once while it builds the new checkpoint (`index-checkpoint-*.json` beside the DB); later passes resume. `INDEX_PIPELINE_VERSION` (currently 1) only gates resuming appended files and quarantine history.
- Calendar step order now differs on nightly and weekly nights: nightly maintenance precedes daily Drive upload; weekly nights retain the daily-backup receipt dependency.


### Operator-visible
- New index stdout/alarm tokens: `INDEX_CHECKPOINTED`, `INDEX_BACKLOG_STALLED`, `INDEX_FILE_QUARANTINED`, `INDEX_PASS_CAP_REACHED`, `INDEX_RESOURCE_FAILED` / `INDEX_RESOURCE_SKIPPED`.
- New tunables: `BRAINLAYER_INDEX_QUARANTINE_PASSES`, `BRAINLAYER_INDEX_STALL_PASSES`, and `BRAINLAYER_INDEX_CALENDAR_BUDGET_S`.
- Files can be quarantined after repeated deterministic failures. See [index convergence](docs/operations/index-convergence.md) for the procedure to clear quarantine and retry.

## [1.5.60] - 2026-10-18

### Fixed
- Exit the MCP stdio bridge when its parent exits (#1344).
- Stabilize backup identity resolution and expose read-only prior inventory (#1346).
- Bind backup retention to upload ancestry and preserve receipts (#1347).
- Surface expiring Drive consent in BrainBar before backups fail (#1348).
- Run SSH harness probes in the existing Aqua session, scope credentials and Cursor trust, validate receipt parents, and distinguish agy diagnostics from tool inventory (#1329, #1330, #1333, #1334).
- Report unavailable Codex App probes as skipped and clear inherited T3 attribution for Codex CLI probes (#1335, #1336).
- Restrict session cleanup to recorded owned MCP orphans without sending signals; add signal-path regression coverage (#1337, #1338).
- Keep replay fixture work and cleanup privately owned, with independent target-completion evidence (#1366).

### Changed
- Release the BrainLayer formula and BrainBar app together.

## [1.5.59] - 2026-10-16

### Fixed
- Let formula-upgrade consolidation proceed when the idle calendar has an empty custody journal with a drifted boot timestamp, while retaining custody locks and nonempty-record checks (#1320).
- Show the underlying cause when consolidation fails, alongside the rollback result (#1320).
- Isolate harness-probe configuration fixtures from the caller's ambient CODEX_HOME (#1324).
- Require both harness runtime test companions before narrowing test scope, and accept transport proof only for the canonical BrainBar socket (#1331).

### Changed
- Add a required live harness-probe deploy gate and release-receipt evidence for search, store, refind, ingest, and transport (#1322).
- Exercise the existing Codex app context and guard agy probes against authentication flows (#1323).
- Run T3 probes through an existing authenticated MCP caller context (#1327).
- Formula-only release; BrainBar remains at 1.5.57.

## [1.5.58] - 2026-10-15

### Fixed
- Avoid supervisor restart loops during formula upgrades when the migration receipt still names the previous keg; uncertain process identity stays critical and does not authorize a restart (#1316).
- Prevent supervisor-owned health checks from kickstarting their parent, preserving failed-heal attempts and retry backoff (#1317).

### Changed
- Require calendar idle before a formula upgrade, immediate consolidation afterward, and verification before the cask; document the expected warning and prior-keg rollback if the supervisor loops (#1318).
- Formula-only release; BrainBar remains at 1.5.57.

## [1.5.57] - 2026-10-15

### Fixed
- Installing or upgrading the BrainBar cask no longer force-kills the freshly started BrainBar app and daemon (EtanHey/homebrew-layers#123).
- Keep native tests from opening Finder or using the real desktop workspace (#1305).
- Recognize the consolidated supervisor in health checks; keep critical warnings while allowing known jobs to retain their restart limits and recovery progress (#1306, #1312).
- Show the supervisor's running, scheduled and failed jobs accurately in BrainBar (#1307).
- Keep chart tooltips inside compact cards without covering the next section (#1309).
- Skip supervisor-owned jobs during bulk restarts so package upgrades use the consolidation path (#1310).
- Show a brief Copied confirmation for Technical details (#1311).
- Identify the latest verified transcript archive as its Drive file ID (#1313).

### Changed
- Share dependency downloads across regression jobs while keeping installations and checks independent (#1308).
- Release the BrainLayer formula and BrainBar cask together.

## [1.5.56] - 2026-10-14

### Fixed
- Keep transport counts and alerts measurable when a receipt updates during a slow health scan (#1303).

### Changed
- Formula-only release; BrainBar remains at 1.5.55.

## [1.5.55] - 2026-10-14

### Added
- Supervisor and maintenance calendar launchers with shared process custody, watchdog health adapters, maintenance leases and component-aware CLI/jobs actions.
- Durable LaunchAgent consolidation and post-success revert, including archived preimages, disabled/load-state preservation, interrupted-transaction recovery and explicit installer activation/rollback commands.
- BrainBar transport counters, schema-v2 observability, historical failure latching and reconnect-deficit alerts for persistent clients left behind after restart.

### Fixed
- Pin legacy MCP/socat configurations to the reconnecting stdio bridge while preserving direct routes; report unsafe or foreign bridge targets.
- Make supervisor recovery, timeout diagnostics, legacy OFF routing and source-only pre-push sweep selection fail visibly instead of silently dropping state or coverage.
- Refuse unsafe/foreign transport receipts, preserve shutdown evidence on bounded SIGTERM/SIGINT, and ignore expired or previous-boot reconnect baselines without discarding valid request counters or incidents.
- Treat operator backup folders as inert; only exact journaled migration archive names trigger missing-journal custody refusal.
- Retry one search virtual-table constructor schema race (#1291).

### Changed
- Run every migration/revert crash shard in a separate required CI job while retaining complete local pre-push coverage.
- Release process: ship the Python formula and BrainBar cask together, verifying the formula keg's signatures before installing the BrainBar cask. Consolidation activation remains an explicit operator action.

## [1.5.48] - 2026-10-06

### Fixed
- Maintenance backup readiness and fleet quiescence safeguards (#1071, #1076).

### Changed
- Queue pre-push and release test suites through the shared heavy-suite lock (#1074).

## [1.5.47] - 2026-10-05
- BrainBar maintenance/job alerts: show each alert once per screen with its actual reason and Show log, clear it after a clean run, and use amber warnings for moderate warm-search latency (#1061, #1062).
- Backups: rename verification to Recovery checks, centre status dots, and consolidate the Config file path row (#1064).
- MCP correctness: preserve full expandable neighbor IDs and advertise only implemented recall modes (#1055).
- MCP output: retain both summary and preview in compact results; reduce framing while preserving canonical IDs, scores and dates, with the hybrid compact wire-size gate at or below pre-#1058 bytes (#1058, #1068).
- At-rest scrub reliability: settle process exit before writing, name refusal gates with value-free diagnostics, and preserve CLI exit status in the run wrapper (#1063).
- Credential scrubbing: extend provider and labeled-credential coverage while preserving non-credential identifiers (#1054).
- Test reliability: isolate backup poll-timing patches and check initial FTS telemetry independently of timing-ranked samples (#1051, #1052).

## [1.5.46] - 2026-10-01

- #1039: BrainBar’s Maintenance card renders a deliberate skip neutrally and unknown evidence as unknown.
- #1035: `brainlayer doctor` flags a shadowing CLI on PATH.
- #1041, #1045, #1048: Security: harden credential scrubbing and cloud-send redaction, add an at-rest scrub command, and scrub the canonical chunk-insert and hook write paths.

## [1.5.45] - 2026-09-30

- #1011: Collapsing Dashboard Details reflows the visible content without leaving unused space.
- #1013, #1014, #1024: One watcher-health model drives the Dashboard and Settings states, with malformed health details rejected instead of presented as truth.
- #1015, #1016: Backups show each installed schedule, last and next run, and the latest local database snapshot or transcript archive, with Reveal and Copy actions.
- #1018: BrainBar accepts the daemon inside its own running app bundle.
- #1019, #1025, #1026: Agent counts use the executable's kernel path and original argv[0] to exclude app helpers and non-session roles. Dashboard adds “Agent writes (24 h)” for agent `brain_store` writes only.
- #1020, #1021: BrainBar uses one window with a status item and a Dashboard-first sidebar; General settings are folded into that navigation.
- #1027, #1030: Database and transcript backups use a BrainLayer-owned Drive token; migration copies the legacy token without deleting it. `brainlayer backup auth` adds browser reconnect and JSON status for the seven-day consent, including a day-6 expiry warning.
- #1031: Backup and maintenance failures raise durable alerts with recovery notices; stale transcript-backup temporary bundles are cleaned only after the backup lock is free.
- #1028: BrainBar gains a bounded Google Drive access model over the backup-auth CLI, including consent timeout and stale-read guards.
- #1029: BrainBar shows Reconnect Google Drive on the Backups page and in the Dashboard attention banner.

## [1.5.44] - 2026-09-29

- #993: BrainBar diagnostics now use unified logging, while lifecycle events go to an owner-only `~/Library/Logs/BrainBar/lifecycle.log`. A rotating mode-0600 debug file is enabled only by `BRAINBAR_DEBUG_LOG=1`; request payloads are never logged.
- #994: Privacy: remaining evaluation and result exports are untracked, and personal entries are removed from shipped data.
- #997: Privacy: personal spelling variants are removed from BrainBar source; user-specific aliases load from optional local config.
- #998: Privacy: real people are removed from shipped defaults and LLM prompt examples; person seeds load from optional local config.
- #1002: Weekly maintenance waits a bounded time for an in-flight daily backup and reuses its verified result. Destructive steps start only inside the quiet window; scheduled FTS repair uses a read-only check, and every scheduled job's CLI has a guard test.
- #1004: The test gate has per-test and overall deadlines that name the stuck test. A concurrent search test no longer hangs in multi-threaded OpenBLAS matrix multiplication.
- #1005: BrainBarDaemon survives an MCP client closing before its connection is accepted; SIGPIPE is ignored and the failed response write is handled as EPIPE.
- #1008: Weekly backup rechecks for a completed, verified daily backup after acquiring the backup lock, preventing a duplicate multi-gigabyte copy.

## [1.5.43] - 2026-09-28

- #987: BrainBar Runtime "Agents" no longer counts app-bundled helpers, including the Codex Computer Use helper, as sessions. Bundle membership comes from the kernel's executable path (fixes #984).
- #988: Security: session-enrichment tool stats, model-authored dictionary keys (including NER relation properties), and nested or escaped JSON are secret-scrubbed before storage.

## [1.5.42] - 2026-09-28

- #974: Advanced → Embedding model now shows the configured model, live hotlane status and process memory instead of permanent "unavailable" rows.
- #976: Runtime Daemon and Last seen rows follow the live BrainLayer daemon across restarts.
- #977: Runtime agent activity counts real CLI sessions with a per-CLI breakdown instead of inflated process totals.
- #979: An unmeasurable daemon state reads neutral "Unknown — reason" instead of a false "Down".

## [1.5.41] - 2026-09-27

- #970: BrainBar no longer kills a healthy BrainLayer daemon when the Mac wakes, preventing MCP disconnects in every Claude seat. The watchdog uses a sleep-aware heartbeat and probes the socket before any restart.

## [1.5.40] - 2026-09-27

- #958: Scrub secrets before every cloud-LLM call and from LLM output before it is saved.
- #959: Recognize and redact Groq, Tailscale and Vercel key families.
- #960: Keep secret-label scanning linear-time on long inputs and redact secrets behind JSON-quoted labels.
- #961: Resolve overlapping secret findings efficiently so inputs with many redactions no longer stall scrubbing.
- #962: Scrub secrets before storage on every ingest path, including BrainBar store and its deferred queue, queue drain, index and digest, with shared tests keeping Swift and Python scrubbing in agreement.

## [1.5.39] - 2026-09-24

- #940: BrainBar's Jobs panel shows each job's real status, last run and next run, with a one-line cause on every "Needs attention" badge; a launchd counter reset reads "Awaiting next run".
- #941: BrainBar reports FTS5 and Trigram coverage against the indexes search actually routes to, and shows vector drain as "Not measured" instead of an estimate borrowed from enrichment.
- #943: When observability is stale, BrainBar keeps the last measured counts, labelled with their measurement time, and shows paused enrichment as "Enrichment paused · N queued".
- #944: Remove BrainBar's Search, Quick Capture and Knowledge Graph UI; MCP search and store are unchanged.
- #945: The Hotlane toggle in BrainBar Settings now actually enables and disables the `com.brainlayer.hotlane-brainbar` job.
- #946: Document which launchd job the Hotlane toggle controls.
- #947: Hotlane keeps its backlog cursor when a write hits a busy or locked database, and its schedule allows a 16-vector batch every 7 seconds.
- #948: Database backups ask macOS to purge purgeable space before the snapshot when raw free space is short, and refuse before writing if it still is not enough.
- #949: Remove the code left unused by the retired BrainBar Search, Quick Capture and Knowledge Graph UI.
- #950: The unit test suite can no longer reach the production BrainBar socket.
- #951: Database backups record a launchd stop and a lock refusal as themselves, and wait up to 30 seconds for purged space to be released.
- #952: A `brain_store` made while a backup snapshot is running is queued and saved once the backup finishes (`STORED (deferred)`), instead of failing.
- #953: Remove personal-data exports from the repository, and add a CI guard against adding them back.
- #954: A Drive backup upload that finished is no longer recorded as failed when the connection resets afterwards; interrupted uploads ask Drive for its offset before re-sending.
- #957: The embedding hotlane runs at Standard priority (Nice 10) instead of Background, which held embedding at scheduler priority 4; the hotlane plist is re-rendered only by `install.sh hotlane-brainbar`, not by a formula upgrade.

## [1.5.38] - 2026-09-23

- #939: Make the database backup's free-space check measure macOS available capacity with a protected raw floor, and give transcript backups a 2-hour limit.

## [1.5.37] - 2026-09-23

- #905: Quarantine the #891 socket-notification CI flake behind an opt-in test.
- #927: Keep long BrainBar Details facts visible in flat rows.
- #932: Add semantic BrainBar status tokens and enforce the 9 pt text floor.
- #934: Stabilize the BrainBar panel frame and scroll anchor across Details toggles.
- #936: Keep BrainBar dashboard card shapes steady across data states.
- #937: Open BrainBar Dashboard and Settings through URL routes.

## [1.5.36] - 2026-09-23

- #909: Expose backup import dependency failures in observability diagnostics.
- #910: Respect deliberately paused launchd jobs in health checks.
- #911: Accept an empty corpus in the canary health check.
- #912: Use the query-embedding interface for digest passage retrieval.
- #913: Keep BrainBar ingest charts readable at rest.
- #914: Declare required runtime dependencies and gate installed module and hook imports.
- #915: Contain inherited Git repository variables in hook tests.
- #916: Unify Settings inside the BrainBar panel.
- #917: Isolate Git reads when maintenance classifies installed code.
- #918: Preserve the SQLite error when an index dedupe savepoint disappears.
- #919: Retain BrainBar coverage fills while toggling Details.
- #920: Add the BrainBar Settings sidebar and section layout.
- #921: Restart loaded BrainLayer jobs after upgrade with verification.
- #922: Cap installed BrainLayer job logs.
- #923: Align BrainBar badge paths and confirm API-key replacement.
- #924: Report stalled drain progress without hiding a stale heartbeat.
- #925: Show measured search and store receipts in BrainBar.
- #926: Disclose Settings locality and model residency in BrainBar.
- #928: Harden BrainBar's first-run badge and API-key confirmation.
- #929: Self-heal failed BrainLayer jobs with badge escalation.
- #930: Preserve self-heal state across uncertain health checks.
- #931: Present honest signal-coverage counts in BrainBar.
- #933: Verify installed keg native-library loadability alongside signatures.

## [1.0.0] - 2026-02-19

### Added
- Initial open-source release as BrainLayer (formerly Zikaron)
- Semantic search across AI conversation history (sqlite-vec + bge-large-en-v1.5)
- 10-field LLM enrichment pipeline (Ollama / MLX backends)
- Brain graph visualization (HDBSCAN clustering + UMAP 3D layout)
- MCP server with 7 tools (+ 14 backward-compatible aliases) for Claude Code, Zed, Cursor
- Interactive setup wizard (`brainlayer init`)
- Centralized artifact storage (`~/.local/share/brainlayer/storage/`)
- Multi-source indexing: Claude Code, WhatsApp, YouTube, Markdown, Claude Desktop
- Communication style analysis pipeline
- Obsidian vault export
- FastAPI daemon with 25+ HTTP endpoints
- GitHub Actions CI/CD with PyPI publishing
- PII sanitization pipeline for safe cloud processing
- Source-aware enrichment thresholds
