# Backup → snapshot → VACUUM → dot: state diagram

Every red box sits on a transition TLC walked when a pre-fix switch was on. Each one is labelled
with the PR that removed it, and its counterexample is in [COUNTEREXAMPLES.md](COUNTEREXAMPLES.md).
The three `S1a-5` boxes are bugs this model found itself (F1–F3), fixed by S1a-5 (4e07651b). The
`S1a-6` box is renewable wake grace, found by the R1 review once the model took the real
`_graced_age` path, and fixed by S1a-6 (8eecb5ee). Neither fix is on main yet. Run `formal/backup-verdict/check.sh` to reproduce every box.

## 1. Receipts and the snapshot check (`BackupVerdict.tla`)

```mermaid
stateDiagram-v2
    classDef bug fill:#fdd,stroke:#c00,color:#900
    classDef finding fill:#ffe9c7,stroke:#c60,color:#740
    classDef ok fill:#e8f6ea,stroke:#2a7,color:#164

    [*] --> LocalVerified: backup-snapshot writes local_verified
    LocalVerified --> OffsiteVerified: backup-daily upload, Drive md5 match
    OffsiteVerified --> LocalRemoved: archive deleted after Drive verify
    LocalVerified --> Abandoned: local file missing or changed
    LocalVerified --> Superseded: newer snapshot, stall deleted
    LocalVerified --> Revoked: verification_error
    OffsiteVerified --> Revoked: md5 mismatch
    Revoked --> LocalReverified: local-only re-verify

    state "Counts as a verified copy" as Counts
    state "No verified copy: red" as NoCopy
    OffsiteVerified --> Counts
    LocalRemoved --> Counts: Drive copy still counts
    Abandoned --> NoCopy
    Superseded --> NoCopy
    Revoked --> NoCopy

    state "#1595: still labelled 'Local database backup'" as b1595
    state "#1596: abandoned still counted" as b1596a
    state "#1596: superseded stall comes back as the clock" as b1596s
    state "#1583 R1 B5: persisted max keeps the revoked copy" as b1583b5
    state "#1589 B5b: local re-verify restores the Drive copy" as b1589b5b
    state "#1589 B5c: filename-only revocation misses legacy state" as b1589b5c
    LocalRemoved --> b1595
    b1595 --> Counts
    Abandoned --> b1596a
    b1596a --> Counts
    Superseded --> b1596s
    b1596s --> Counts
    Revoked --> b1583b5
    b1583b5 --> Counts
    LocalReverified --> b1589b5b
    b1589b5b --> Counts
    Revoked --> b1589b5c
    b1589b5c --> Counts

    class b1595,b1596a,b1596s,b1583b5,b1589b5b,b1589b5c bug
    class Counts ok
```

## 2. One backup check across a monitoring gap (`BackupVerdict.tla`)

```mermaid
stateDiagram-v2
    classDef bug fill:#fdd,stroke:#c00,color:#900
    classDef limit fill:#eef,stroke:#669,color:#336

    [*] --> Green: readable tick, copy within red age
    Green --> Red: copy ages past red while the monitor runs
    Green --> GraceAmber: Mac asleep while the SAME copy ages past red
    GraceAmber --> Red: catch-up deadline passes (never renewed)
    GraceAmber --> Green: new verified receipt
    Red --> Green: new verified receipt (only way out)
    Green --> Red: readable tick, no verified copy

    state "#1583 R1 B3: every later gap renews grace" as b1583b3
    state "#1589 B5a: grace with no clock after revocation" as b1589b5a
    GraceAmber --> b1583b3
    b1583b3 --> GraceAmber
    Green --> b1589b5a: revoked at wake
    b1589b5a --> GraceAmber

    state "Receipts unreadable: last known copy keeps aging" as Unreadable
    Green --> Unreadable
    Unreadable --> Red: last known copy ages past red
    Unreadable --> Green: log readable again

    class b1583b3,b1589b5a bug
    class Unreadable limit
```

The blue box is a design limit, not a bug. A receipt appended while the log is unreadable cannot be
seen, so the dot trusts the last copy it knew until that copy ages out (case `limit-unreadable-log`).

## 3. Latched and hysteresis checks (`PipeVerdict.tla`)

```mermaid
stateDiagram-v2
    classDef bug fill:#fdd,stroke:#c00,color:#900
    classDef finding fill:#ffe9c7,stroke:#c60,color:#740

    [*] --> Green
    Green --> Red: failure observed
    Red --> Recovering: good observation 1 of 2
    Recovering --> Red: bad, unobserved or graced tick (count resets, #1595, S1a-5)
    Recovering --> Green: observed good tick 2 of 2
    Red --> Red: hysteresis check observes amber (stays red, S1a-5)
    Green --> Graced: stall past red right after a monitoring gap (one wake window)
    Graced --> Red: the episode's one window ends (S1a-6, never renewed)
    Graced --> Green: the stall clears (a new episode)
    state "S1a-6: every gap renews wake grace on a stopped embedder, stuck store or dead watcher" as s1a6
    Graced --> s1a6
    s1a6 --> Graced
    Red --> Green: drive_auth: observed ok. embedding: vectors written

    state "#1583 R1 B1: empty queue repays a failed store canary" as b1583b1
    state "#1583 R1 B2: tick with no heartbeat counts as good" as b1583b2
    state "#1583 R1 B4: refresh_failed clears lost Drive access" as b1583b4
    state "#1595: unknown tick keeps the recovery count" as b1595r
    state "#1587 B4: backlog rows age out, no vector written" as b1587b4
    state "#1587 B5: failed round trip leaves the old success" as b1587b5
    Red --> b1583b1
    b1583b1 --> Green
    Recovering --> b1583b2
    b1583b2 --> Green
    Red --> b1583b4
    b1583b4 --> Amber
    Recovering --> b1595r
    b1595r --> Green
    Red --> b1587b4
    b1587b4 --> Green
    Red --> b1587b5
    b1587b5 --> Green

    state "S1a-5 F1: one aging heartbeat steps red to amber" as f1
    state "S1a-5 F2: unobserved tick keeps the count" as f2
    state "S1a-5 F3: stale heartbeat seen in wake grace keeps the count" as f3
    Red --> f1
    f1 --> Amber
    Recovering --> f2
    f2 --> Green
    Recovering --> f3
    f3 --> Green

    class b1583b1,b1583b2,b1583b4,b1595r,b1587b4,b1587b5,f1,f2,f3,s1a6 bug
```

## 4. The dot (`DotTable.tla`, plan §3d)

```mermaid
stateDiagram-v2
    classDef bug fill:#fdd,stroke:#c00,color:#900
    state "Dot green: every check green or unknown" as DG
    state "Dot amber: worst check amber" as DA
    state "Dot red: some check red" as DR
    [*] --> DG
    DG --> DA: a check turns amber (a note)
    DA --> DR: a check turns red (its RED column)
    DR --> DA: that check cleared red by its own evidence
    DA --> DG: every check green
    state "Job alert (job-alerts.json): always a note" as JA
    DG --> JA: a job fails
    JA --> DG: dot unchanged, its own clean run clears the note
    state "#1583 S1: a job alert lit the badge" as b1583s1
    JA --> b1583s1
    b1583s1 --> DR
    class b1583s1 bug
```

### Every dot state (plan §3d), and where it is checked

| Check | GREEN | AMBER (note) | RED (dot) | What clears red | Checked in |
|---|---|---|---|---|---|
| `backup_local` "Database snapshot (verified)" | ≤ 30 h | 30–54 h | > 54 h, or none | a new verified receipt | `BackupVerdict` NoFalseGreen, RedHoldsUntilEvidence, TruthfulSnapshotLabel |
| `backup_offsite` | ≤ 30 h | 30–78 h | > 78 h | a new verified upload | `BackupVerdict` NoFalseGreen, RedHoldsUntilEvidence |
| `transcripts` | ≤ 30 h | 30–54 h | > 54 h | a new verified transcript backup | `DotTable` aging class (same `_verified_age` path as the two above, not separately modelled) |
| `drive_auth` | ok | refresh failed | consent needed | an observed `ok` | `PipeVerdict` Focus=drive |
| `integrity` | ok | — | corrupt | a clean result | `DotTable` two-state class |
| `ingest` | fresh | > 10 min | > 30 min | 2 consecutive good ticks, advancing heartbeat | `PipeVerdict` Focus=ingest (S1a-5 F1–F3; conformance vs real code) |
| `embedding` | progressing | — | not shrinking > 6 h | vectors written | `PipeVerdict` Focus=embedding (S1a-6 grace; conformance vs real code) |
| `stores` | ≤ 15 min, canary ok | 15–45 min or 1 fail | > 45 min or 3 fails | canary success, then 2 good ticks | `PipeVerdict` Focus=stores (S1a-6 grace; conformance vs real code) |
| `search` | ok | 1–2 fails | 3 fails | 2 consecutive good ticks | `DotTable` hysteresis class |
| `disk` | ≥ peak + 2× floor | < peak + 2× floor | < peak | space recovered | `DotTable` aging class |
| `monitor` | — | 3+ blind ticks | never | a good observation | `DotTable` MonitorNeverRed |
| job alerts | — | always a note | never | their own next clean run | `DotTable` NoStuckAlert, AlertClears |
