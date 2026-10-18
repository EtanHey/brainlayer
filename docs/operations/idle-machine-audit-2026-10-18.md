# Idle-machine health audit — 2026-10-18

Scope: base `02416a53`, Python producers and BrainBar consumers. Read-only runtime
samples on MBP and M1 at **18:24–18:27 UTC**. No production DB opened, launchctl
invoked, lifecycle action taken, or live state written. Eight installed Python
module SHA-256 hashes match this base on both Macs: health_check, doctor,
observability_surface, observability_backup, job_alerts, job_lifecycle_health,
service_supervisor and watcher. Swift conclusions are source review, not native
visual proof. `No` below means no matching reason in the sampled files; doctor
was deliberately not executed because its checks open a database.

## Current state and before/after

- M1: zero transcript/watcher flow; watcher polls **110 → 113**, no alert reasons;
  drain cycles **4763 → 4897**, zero drained writes; queue empty. Observability
  records zero CLI/subagent chunks in 24h and eight total stores (operational
  probes/backups). A completely zero-store window is covered by synthetic tests.
- M1's **only badge and job alert** is `maintenance-full`: “BrainLayer
  maintenance-weekly: quiet window missed”. Backups are fresh, retention PASS,
  seven archives in the last 30d. This scheduled-job failure stays red before
  and after this PR; no live alert was cleared.
- MBP: watcher polls **77 → 80**, no watcher alerts, drain cycles **3807 → 3924**.
  Badge/job reasons: backup-daily failure, activation monitor process-scan timeout,
  missed weekly window, plus T3 indexing failure in the badge. These failures
  remain unchanged; none derives merely from old transcript/store timestamps.
- Both supervisors report healthy watch/drain/hotlane processes and successful
  observability/T3/throughput runs. Both health-check components last exited 1;
  Tier-0 also last exited 1 and its durable log says `label_unloaded`, despite
  supervised health-check status and recent state files. This is an ownership
  check discrepancy, not inactivity; escalated separately to the lead.

## Check inventory

Paths in the Python rows are under `src/brainlayer/`; native rows under
`brain-bar/Sources/BrainBar/`. “Integrity” is neither activity nor heartbeat: it
measures existing data/configuration, and remains actionable on an idle machine.

| Check / attention trigger | File:line | Activity or liveness basis | Red on idle M1 today? | Fix / disposition |
|---|---|---|---|---|
| Supervisor unavailable; invalid health state; hold recovery failure | health_check.py:1335 | Liveness / control integrity | No matching badge reason | Keep failures |
| Persisted job alerts (any job, including notifications) | health_check.py:1360; job_alerts.py:61 | Actual producer failures, not time since a store | **Yes: maintenance-full** | Preserve; next successful job clears it |
| Badge publication/invalidation failure; state lock; check deadline | health_check.py:1274,1371,1385 | Liveness / I/O | No matching badge reason | Keep failures |
| Missing/malformed backup receipt | health_check.py:913 | Job execution / evidence integrity | No | Keep failures |
| No backup attempt for 36h / multiple nights | health_check.py:968,975 | Job liveness, not archive creation | No | Keep; fresh verified no-op accepted at :977 |
| Failed/unverified backup result | health_check.py:989 | Actual job failure | No | Keep red |
| T3 reader/index failure | health_check.py:992 | Actual adapter failure | No | Keep red; zero ingest is not failure |
| Hotlane absent / backlog disabled | health_check.py:1439,1447 | Process/configuration liveness | No matching badge reason; supervisor PID present | Keep failures |
| Missing-vector query failure / missing count rising or flat | health_check.py:1468,1496,1505,1518 | Existing unembedded work / query failure | No matching badge reason | Keep; zero missing vectors cannot stall |
| Canary retrieval/transport failure | health_check.py:1540,1552 | Request liveness | No; canary status `empty` | Empty successful results remain healthy |
| Unloaded/retired service; expired pause | health_check.py:1582,1590,1596 | Process/configuration liveness | No matching badge reason | Keep failures |
| Pending-store read failure / queue backlog | health_check.py:1629,1637 | Existing durable work / I/O | No; queue 0 | Keep failures |
| Watcher poll/offset stall with recent input | health_check.py:1667 | Liveness with work precondition | No | Old offsets alone do not fail; canonical-heartbeat policy belongs to watcher-alarm lane |
| Drain heartbeat/progress / flat drained_total | drain_liveness.py:49; health_check.py:1710 | Liveness / existing queued work | No | Flat write count only fails with backlog |
| Writer lock wedge | health_check.py:1732 | Blocked existing work / lock ownership | No matching badge reason | Keep failures |
| Job scan failure / repeated failed runs or heals | job_lifecycle_health.py:145,242; health_check.py:1825 | Job completion / runtime identity | Maintenance exit 1 recorded, not yet failed-heals escalation | Keep failures |
| Watcher coverage drop | watcher.py:403 | **Work**: accepted input without writes | No; input/output rates 0 | Idle does not start coverage timer |
| Watcher offset lag / ingestion errors / quarantine / DB probe | watcher.py:414,1748,1760 | Existing work / real ingestion or read failure | No | Keep failures |
| Throughput watchdog stalled counters | scripts/launchd/throughput-watchdog.py:619 | Liveness or pending work; no pending input gives idle | No; action `progress`, pending 0 | Drain cycles prove liveness without writes |
| Tier-0 missing label/state, stale state, slow check | scripts/tier0-watchdog.sh:144,169,176 | Health-job liveness | **Yes: label_unloaded** in log / supervisor failure | Separate ownership issue; not idle suppression |
| Fleet watchdog absent service / invalid pause state | scripts/launchd/fleet-watchdog.sh:53,137 | Service/control liveness | No current failure in supervisor status | Keep; not based on transcript age |
| Doctor CLI/version/Spotlight/MCP/DB config findings | doctor.py:710 | Configuration integrity | Not run (DB opens forbidden) | Keep; no activity clock |
| Doctor zero **total** corpus / FTS desync / DB unreadable | doctor.py:744,749,757 | Corpus integrity, not zero indexed **today** | Not run | Keep; do not equate daily zero with total-empty DB |
| Doctor unembedded rows / lexical gaps / phantom/orphan vectors / roundtrip | doctor.py:770,782,808,814,823,836,849 | Existing data integrity | Not run | Keep; no absence-of-new-data test |
| Doctor missing process / loaded label / deploy identity | doctor.py:858,879 | Process/runtime liveness | Not run | Keep failures |
| Doctor stalled / growing queue | doctor.py:958,980,988 | Existing queued work | Not run | Empty queue has no progress obligation |
| Observability stores, emitters, unknown-author census | observability_surface.py:132,158,176 | Activity metrics, **not health gates** | No; measured zero agent activity | Keep zero and old latest timestamps measured |
| Observability unreadable input / stale document | observability_surface.py:219; Dashboard/ObservabilityView.swift:380 | Measurement liveness / integrity | No; document refreshed at 18:20 UTC | Keep missing/stale producer evidence visible |
| Backup attempt freshness / daily backup errors / retention | observability_backup.py:317,337 | Job/evidence integrity | No freshness/retention failure; job alert overlays maintenance failure | Fresh verified no-op stays fresh; no fake upload |
| Transport missing/stale evidence; drops/I/O storm/reconnect deficit | observability_transport.py:103,258,271 | Request/receipt liveness and actual failures | No; measured transport | Idle zero counters never trigger storm |
| Dashboard indexed today / store count / last-store receipt | BrainBarWindowRootView.swift:591,1315,1405; BrainBarOperationReceipts.swift:30 | Activity display, **not red gates** | No activity-triggered reason in producer files | Zero/old remain informational; live pixels not inspected |
| Pipeline ingress and agent lanes | Dashboard/PipelineState.swift:432,774,882 | Activity display → idle | No source-level idle red | Already neutral idle |
| Watcher stopped/missing/stale heartbeat / reported errors | Dashboard/WatcherHealthStatus.swift:76,127 | Process/poll liveness | No source-derived failure from fresh heartbeat | Keep actual failures; no-flow is idle at PipelineState.swift:834 |
| Daemon down/unreadable process evidence | Dashboard/DaemonHealthMonitor.swift:434 | Process liveness | No native probe run | Keep failures; old last-answer text alone is not attention |
| Jobs exit failure/unloaded/paused/unknown ownership | BrainLayerLaunchdActivity.swift:285 | Job/process liveness | Maintenance job failure in durable files | Keep failures; configured-off/skipped states neutral |
| Weekly completion over 8d / absent completion | BrainLayerLaunchdActivity.swift:418 | Required scheduled work, not transcripts | Existing weekly failure; native verdict not sampled | Keep operational failure |
| Drive missing/revoked/expiring consent | BrainBarDriveAuth.swift:106 | External capability expiry, not user activity | Not independently probed | Keep actual expiry; no credential/token reads |
| Backup missing upload / zero archives in 30d | Dashboard/ObservabilityView.swift:480,518; BrainBarWindowRootView.swift:579 | **Activity-based red**, even after a verified no-op | **No today**; upload exists, 7 archives | **Follow-up #1396**: neutral proven-idle state, retain true history |
| Backup freshness wording claims both copies recent | BrainBarBackupChecks.swift:58 | Activity-age claim mislabels no-op attempt freshness | Not red today | **#1396**: describe verified no-op truthfully |
| Backup attention aggregation / menu badge | BrainLayerLaunchdActivity.swift:530; Dashboard/ObservabilityView.swift:62; BrainBarStatusPopoverController.swift:224 | Underlying status lines / health producer freshness | **Yes maintenance alert** in badge | Preserve actual failures; #1396 fixes backup activity lines |

## Changes and verification

No Python producer activity-only false red was found, so this PR changes no
runtime health policy. Six regression cases cover a 49h-old corpus, repeated idle
health ticks with an empty successful search, actual backup/job failures retained
in the badge, and verified no-op attempts without an upload or with >30d history.
The six tests pass on the baseline behavior. The first fixture run exposed a
WAL-copy setup mistake, corrected by staging committed DELETE-journal bytes; it
was not a production defect. Follow-up: [#1396](https://github.com/EtanHey/brainlayer/issues/1396).

Local targeted pytest: **477 passed**. Ruff checks pass. CI and native visual
verification remain separate review/deploy gates. No production data changed.

Surface sweep: Entry points — health CLI, observability job, watchdogs and doctor
reviewed; Clients — Dashboard, Jobs, Backups and menu consumers reviewed;
Providers — local Python/Swift, both installed Macs sampled; Contracts — JSON
health/badge/observability shapes unchanged; Reverse states — empty, no-op,
failed backup and failed job covered; Connection modes — local MBP and SSH M1
read-only, supervisor status only; Docs — health guide's obsolete empty-canary
failure claims corrected. No release/install or UI behavior change is claimed.
