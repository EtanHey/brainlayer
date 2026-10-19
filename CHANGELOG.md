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
