#if DEBUG
import Foundation

/// Stage 2's capture mode and the headless renders' states. `BRAINBAR_DESIGN_FIXTURE=<name>` (DEBUG builds only) drives
/// the real status item, dropdown and window from one named, invented state, with the design forced to `new`. It never
/// writes real state: in this mode the app starts no server, sampler, watcher or job, and reads no file.
enum BrainBarDesignFixture {
    static let environmentKey = "BRAINBAR_DESIGN_FIXTURE"

    /// The states the design is judged in (DESIGN-SPEC-R1 § 3.2's seven, comp 03's two faults, the Reconnect waiting
    /// line, and a failed upload on Backups). Earlier names: checking → not-confirmed, one-fault → drive-signed-out,
    /// several-faults → two-faults, jobs-failed-streak → nightly-failing; cant-tell is gone.
    enum Name: String, CaseIterable, Sendable {
        case calm
        case agentNotRead = "agent-not-read"
        case driveSignedOut = "drive-signed-out"
        case savesStuck = "saves-stuck"
        case meaningBehind = "meaning-behind"
        case nightlyFailing = "nightly-failing"
        case notConfirmed = "not-confirmed"
        case twoFaults = "two-faults"
        case reconnectWaiting = "reconnect-waiting"
        case backupsFailedUpload = "backups-failed-upload"
        /// L24: one amber notice that is not catching up (the copy on this Mac is late): "Check soon" and the amber dot.
        case backupLate = "backup-late"
        /// R1.13 (A2): BrainLayer 1.6.6's catch-up: the reader catching up (amber, `catch_up.state` catching_up, no
        /// `offset_lag`), 117 MB left, about 25 min to go; as in the installed app, no episode clock.
        case lagBehind = "lag-behind"
        /// R1.13 (A3): the producer says the lag stopped shrinking (`offset_lag` raised, `catch_up.state` stalled) and the
        /// reader is red.
        case lagStalled = "lag-stalled"
        /// R1.11 (B3): a failure only an agent can fix: search failed its last 3 checks.
        case searchFailing = "search-failing"
        case uploadProgress = "upload-progress"
        case uploadNoTotal = "upload-no-total"
        case uploadStalled = "upload-stalled"
        case verifying
        case uploadSnapshotRed = "upload-snapshot-red"
        case uploadDriveSignedOut = "upload-drive-signed-out"
        case uploadDriveStalled = "upload-drive-stalled"

        /// The window page the fixture opens on.
        var page: NDPage {
            switch self {
            case .backupsFailedUpload, .uploadNoTotal, .uploadStalled, .verifying, .uploadSnapshotRed, .uploadDriveStalled: .backups
            default: .overview
            }
        }
    }

    /// The fixture this process was asked for, or nil. Only names on the list count.
    static var requested: Name? {
        ProcessInfo.processInfo.environment[environmentKey].flatMap(Name.init(rawValue:))
    }

    /// F06: a capture keeps its own preferences (window frame, the Design control), in a suite of its own that is emptied
    /// at launch, so nothing it does reaches the installed app's preferences.
    static let captureSuiteName = "com.brainlayer.brainbar.design-capture"
    nonisolated(unsafe) static let capturePreferences: UserDefaults = {
        let defaults = UserDefaults(suiteName: captureSuiteName)!
        defaults.removePersistentDomain(forName: captureSuiteName)
        return defaults
    }()

    /// The real status item, dropdown and window, wired to the capture's own preferences.
    @MainActor static func makeCaptureSurfaces(runtime: BrainBarRuntime, activationPolicy: BrainBarWindowActivationPolicy?,
                                               statusBar: any BrainBarStatusItemHosting)
        -> (BrainBarDashboardPanelController, BrainBarStatusPopoverController) {
        let workspace = BrainBarRecordingWorkspace()
        let panel = BrainBarDashboardPanelController(
            workspace: workspace, runtime: runtime, activationPolicy: activationPolicy,
            frameStore: BrainBarWindowFrameStore(defaults: capturePreferences, key: BrainBarDashboardPanelController.frameDefaultsKey))
        // Fix 4: the capture draws the new design, so its Settings says New (the shipping default stays Classic).
        capturePreferences.set(BrainBarDesign.new.rawValue, forKey: BrainBarDesign.key)
        panel.designDefaults = capturePreferences
        panel.applyDesign()
        let status = BrainBarStatusPopoverController(workspace: workspace, runtime: runtime, dashboardPanelController: panel,
                                                     statusBar: statusBar, menuDefaults: capturePreferences)
        return (panel, status)
    }

    /// One "now" every fixture time derives from. Tests and renders use a fixed moment, so they draw the same picture;
    /// a capture uses its launch time (to the minute) in this Mac's zone, so absolute times on screen ("today at 12:09")
    /// agree with the menu-bar clock and with the ages beside them (review blocker 3: a frozen 08:00 read as 12 h off).
    static let now: Date = makeNow(capturing: requested != nil, wall: Date())

    static func makeNow(capturing: Bool, wall: Date,
                        pinned: String? = ProcessInfo.processInfo.environment["BRAINBAR_DESIGN_NOW"]) -> Date {
        guard capturing else { return fixedNow }
        // D7.7: `BRAINBAR_DESIGN_NOW=<ISO 8601>` pins every launch of a capture pass to one moment.
        let start = pinned.flatMap { ISO8601DateFormatter().date(from: $0) } ?? wall
        return Date(timeIntervalSinceReferenceDate: (start.timeIntervalSinceReferenceDate / 60).rounded(.down) * 60)
    }
    static let fixedNow: Date = ISO8601DateFormatter().date(from: "2026-11-15T08:00:00+02:00")!
    static var zone: TimeZone { requested == nil ? TimeZone(secondsFromGMT: 7200)! : .current }

    static func check(_ name: String, _ state: BrainBarHealthVerdict.State = .green, age: Double? = nil,
                      limit: Double? = nil, fails: Int? = nil) -> BrainBarHealthVerdict.Check {
        BrainBarHealthVerdict.Check(name: name, state: state, ageS: age, thresholdS: limit, note: "", unknownTicks: 0,
                                    blind: false, fails: fails)
    }

    /// The calm verdict every fixture starts from: every check green, ages well inside their limits.
    static func calmChecks() -> [BrainBarHealthVerdict.Check] {
        [
            check("ingest", age: 48, limit: 900), check("stores", age: 0, limit: 1800), check("search", fails: 0),
            check("embedding", age: 420, limit: 3600), check("backup_local", age: 4 * 3600, limit: 54 * 3600),
            check("backup_offsite", age: 3 * 3600, limit: 54 * 3600), check("transcripts", age: 4 * 3600, limit: 54 * 3600),
            check("drive_auth"), check("integrity"), check("disk"), check("monitor"),
        ]
    }

    static func capture(_ checks: [BrainBarHealthVerdict.Check], dot: BrainBarHealthVerdict.State = .green,
                        age: TimeInterval = 60) -> BrainBarMenuCapture {
        var c = BrainBarMenuCapture(now: now, timeZone: zone)
        c.verdict = BrainBarHealthVerdict(dot: dot, checks: checks, notes: [], evaluatedAt: now.addingTimeInterval(-age))
        c.verdictAwakeAgeS = age
        c.maintenanceNightFailures = 0
        c.backupDailyFailures = 0
        c.jsonlBackupFailures = 0
        c.backupDailyNextRunAt = now.addingTimeInterval(19 * 3600)
        c.keywordCoverage = 1
        return c
    }

    static func replacing(_ checks: [BrainBarHealthVerdict.Check], _ replacement: BrainBarHealthVerdict.Check) -> [BrainBarHealthVerdict.Check] {
        checks.map { $0.name == replacement.name ? replacement : $0 }
    }

    static func inputs(_ name: Name) -> NDInputs {
        var i = base()
        i.settingsBackups = BrainBarSettingsBackupSummary(
            file: URL(fileURLWithPath: "/fixture/settings/brainlayer.env"), count: 3)
        switch name {
        case .calm:
            break
        case .agentNotRead:
            var ingest = check("ingest", .red, age: 48, limit: 900)
            ingest.contextCodes = ["growth_not_indexed:cursor"]
            i.capture = capture(replacing(calmChecks(), ingest), dot: .red)
        case .driveSignedOut, .reconnectWaiting:
            i.capture = capture(replacing(calmChecks(), check("drive_auth", .red)), dot: .red)
            i.capture.driveAuthModel = .invalid
            i.capture = withDriveAge(i.capture, database: 27, conversations: 28)
            i.driveReconnecting = name == .reconnectWaiting
            i.backupHistory = history(driveStalledAfterDays: 1)
        case .savesStuck:
            i.capture = capture(replacing(calmChecks(), check("stores", .red, age: 2 * 3600, limit: 1800)), dot: .red)
            i.savesWaiting = 38
            i.oldestSaveQueuedAt = now.addingTimeInterval(-2 * 3600)
        case .meaningBehind:
            i.capture = capture(replacing(calmChecks(), check("embedding", .amber, age: 2 * 3600, limit: 3600)), dot: .amber)
        case .nightlyFailing:
            i.capture.maintenanceNightFailures = 3
            i.nightlyLastRun = .ran(now.addingTimeInterval(-3 * 86_400 - 4 * 3600))
            i.jobs = jobs(nightlyFailing: true)
        case .notConfirmed:
            i.capture = capture(calmChecks(), age: 2 * 3600)
        case .twoFaults:
            i.capture = capture(replacing(calmChecks(), check("drive_auth", .red)), dot: .red)
            i.capture.driveAuthModel = .invalid
            i.capture.maintenanceNightFailures = 3
            i.capture = withDriveAge(i.capture, database: 27, conversations: 28)
            i.backupHistory = history(driveStalledAfterDays: 1)
            i.nightlyLastRun = .ran(now.addingTimeInterval(-3 * 86_400 - 4 * 3600))
            i.jobs = jobs(nightlyFailing: true)
        case .backupsFailedUpload:
            // R1.2 change 13: a failed upload retried successfully the same night; the checks stay green.
            i.backupHistory = history(failedDriveDayAgo: 3, retried: true)
        case .backupLate:
            // R2 item 4: Drive's database copy is the Mac copy it uploads, so it is as old: the upload of the 30 h copy.
            i.capture = capture(replacing(replacing(calmChecks(), check("backup_local", .amber, age: 30 * 3600, limit: 54 * 3600)),
                                          check("backup_offsite", age: 30 * 3600, limit: 54 * 3600)), dot: .amber)
            i.newestLocalCopyAt = now.addingTimeInterval(-30 * 3600)
            i.localCopiesKept = 2  // a missed night: two copies on this Mac
            i.backupHistory = history(localNewestHoursAgo: 30, driveNewestHoursAgo: 30)
        case .lagBehind:
            i.capture = capture(replacing(calmChecks(), check("ingest", .amber, age: 40 * 60, limit: 900)), dot: .amber)
            i.capture.watcher = .init(alertReasons: [], lagBytes: 117 * 1_048_576, lastReadAt: now.addingTimeInterval(-30),
                                      catchUp: WatcherCatchUp(state: "catching_up", bytesBehind: 117 * 1_048_576, etaS: 25 * 60))
        case .lagStalled:
            i.capture = capture(replacing(calmChecks(), check("ingest", .red, age: 40 * 60, limit: 900)), dot: .red)
            i.capture.watcher = .init(alertReasons: ["offset_lag"], lagBytes: 117 * 1_048_576, lastReadAt: now.addingTimeInterval(-30),
                                      catchUp: WatcherCatchUp(state: "stalled", bytesBehind: 117 * 1_048_576))
        case .searchFailing:
            i.capture = capture(replacing(calmChecks(), check("search", .red, fails: 3)), dot: .red)
        case .uploadProgress, .uploadNoTotal, .uploadStalled, .verifying, .uploadSnapshotRed, .uploadDriveSignedOut, .uploadDriveStalled:
            let backups = backupFacts(name)
            if name == .uploadSnapshotRed {
                i.capture = capture(replacing(calmChecks(), check("backup_offsite", .red, age: 72 * 3600, limit: 54 * 3600)), dot: .red)
                i.backupHistory = history(driveNewestHoursAgo: 72)
                i.captureBackupChecks = BrainBarBackupChecks.derive(backups, now: now)
            } else if name == .uploadDriveStalled {
                i.capture = capture(replacing(calmChecks(), check("backup_offsite", .red, age: 72 * 3600, limit: 54 * 3600)), dot: .red)
                i.capture.driveAuthModel = .valid
                i.backupHistory = history(driveNewestHoursAgo: 72)
            } else if name == .uploadDriveSignedOut {
                i.capture = capture(replacing(calmChecks(), check("drive_auth", .red)), dot: .red)
                i.capture.driveAuthModel = .invalid
                i.capture = withDriveAge(i.capture, database: 27, conversations: 28)
                i.backupHistory = history(driveStalledAfterDays: 1)
            }
            i.backupProgress = BrainBarBackupProgress.read(backups, now: now)
            if name != .verifying { i.capture.upload = .init(startedAt: now.addingTimeInterval(-3600), live: true) }
        }
        return i
    }

    /// Synthetic receipts enter the same progress and Classic-check presenters as real observations.
    static func backupFacts(_ name: Name) -> ObservabilityDocument.Backups {
        let verifying = name == .verifying
        let stale = name == .uploadSnapshotRed
        let signedOut = name == .uploadDriveSignedOut
        let driveHours: Double = stale || name == .uploadDriveStalled ? 72 : signedOut ? 27 : 3
        let transcriptHours: Double = signedOut ? 28 : 4
        let local = ObservabilityDocument.LocalSnapshot(
            lastAt: now.addingTimeInterval(-4 * 3600), destination: "capture-copy.db.gz", localVerified: true,
            uploaded: verifying, uploadState: verifying ? "pending" : "in_progress", uploadError: nil,
            uploadProcessLive: true, uploadProcessStartedAt: now.addingTimeInterval(-3600),
            localIntegrity: "ok", localIntegrityPragma: "integrity_check", bytesConfirmed: 25,
            bytesTotal: name == .uploadNoTotal ? nil : 100, verificationPending: verifying,
            bytesConfirmedAt: now.addingTimeInterval(name == .uploadDriveStalled || name == .uploadStalled ? -21 * 60 : -60))
        return ObservabilityDocument.Backups(
            state: "measured", reason: "", inputs: [], freshness: stale ? "stale" : "fresh", thresholdHours: 36,
            retentionInvariant: "PASS", survivingArchives30D: 7, errorType: nil,
            lastVerifiedUpload: .init(at: now.addingTimeInterval(-transcriptHours * 3600), ageHours: transcriptHours, archiveId: "capture-archive", verified: true),
            dbSnapshot: .init(lastAt: now.addingTimeInterval(-driveHours * 3600), destination: "capture-previous.db.gz", verified: true),
            launchd: .init(label: "com.brainlayer.jsonl-backup", bootstrapped: true, disabledDirPresent: false),
            localSnapshot: local)
    }

    private static func withDriveAge(_ c: BrainBarMenuCapture, database: Double, conversations: Double) -> BrainBarMenuCapture {
        var c = c
        let checks = (c.verdict?.checks ?? []).map { k -> BrainBarHealthVerdict.Check in
            switch k.name {
            case "backup_offsite": check(k.name, age: database * 3600, limit: 54 * 3600)
            case "transcripts": check(k.name, age: conversations * 3600, limit: 54 * 3600)
            default: k
            }
        }
        c.verdict = BrainBarHealthVerdict(dot: c.verdict?.dot ?? .red, checks: checks, notes: [], evaluatedAt: c.verdict?.evaluatedAt ?? now)
        return c
    }


    /// Everything else a calm moment holds: the numbers on the comps, invented.
    static func base() -> NDInputs {
        var i = NDInputs(capture: capture(calmChecks()))
        i.memories = 297_412
        i.searchable = 0.9946
        i.savesWaiting = 0
        i.watcherHeartbeatAgeS = 48
        traffic(&i, at: now)
        i.jobs = jobs(nightlyFailing: false)
        i.backupHistory = history()
        i.nightlyLastRun = .ran(now.addingTimeInterval(-5 * 3600))
        i.weeklyLastRun = .ran(now.addingTimeInterval(-4 * 86_400))
        i.newestLocalCopyAt = now.addingTimeInterval(-(4 * 3600 + 2 * 60))
        i.localCopiesKept = 3
        i.canRevealLocalCopy = true
        return i
    }

    /// Review B2: every wave and the Agents column come from one invented half-minute series, so a longer range always
    /// holds at least what a shorter one does, and the Agents rows sum to the 24 h saves plus the 24 h reads.
    static func traffic(_ i: inout NDInputs, at clock: Date) {
        i.menuWave = NDFixtureTraffic.wave(minutes: 60, buckets: 24, now: clock)
        i.heroWaves = Dictionary(uniqueKeysWithValues: NDRange.allCases.map {
            ($0, NDFixtureTraffic.wave(minutes: $0.minutes, buckets: 60, now: clock))
        })
        // L5 ruling: the Agents rows split every memory in 24 h (saved by agents plus read from conversations), as the real
        // per-family count does; the fixture attributes all of it (the real source can leave some unattributed).
        let dayWave = NDFixtureTraffic.wave(minutes: 1440, buckets: 60, now: clock)
        let day = dayWave.savesTotal + dayWave.readsTotal
        let claude = day * 62 / 100, codex = day * 28 / 100
        i.agents = NDAgents(agents: [
            .init(family: .claude, sessions: 5, saves: claude), .init(family: .codex, sessions: 3, saves: codex),
            .init(family: .cursor, sessions: 0, saves: day - claude - codex), .init(family: .gemini, sessions: 0, saves: 0),
        ])
    }

    /// One capture tick (fix 5): the fixture clock moves one minute, the 1 h hero's bucket width, and every wave and
    /// total is read again off the same series. The 1 h hero shifts on every tick; the dropdown's hour (150 s buckets),
    /// 3 h and 24 h shift when their own bucket boundary passes, and their newest bar grows in between, as live data does.
    static func ticked(_ inputs: NDInputs, step: Int) -> NDInputs {
        var next = inputs
        guard inputs.menuWave?.readable != false, inputs.heroWaves.values.allSatisfy(\.readable) else { return next }
        traffic(&next, at: now.addingTimeInterval(Double(step) * 60))
        return next
    }

    /// A week of nightly attempts. Each lane's newest attempt sits at the age its row's gauge says (the Drive copy 3 h,
    /// the others 4 h; with Drive stalled, the Drive lanes stop at 21 h), so the rail and the number agree.
    static func history(failedDriveDayAgo: Int? = nil, driveStalledAfterDays: Int? = nil, retried: Bool = false,
                        localNewestHoursAgo: Double = 4, driveNewestHoursAgo: Double? = nil) -> NDBackupHistory {
        func nights(newestHoursAgo: Double, fail: Int? = nil) -> [NDAttempt] {
            (0...6).reversed().flatMap { day -> [NDAttempt] in
                let at = now.addingTimeInterval(-Double(day) * 86_400 - newestHoursAgo * 3600)
                guard now.timeIntervalSince(at) <= 144 * 3600 else { return [] }
                // A retry 46 minutes after a failed upload, the same night.
                if day == fail, retried { return [NDAttempt(at: at, ok: false), NDAttempt(at: at.addingTimeInterval(46 * 60), ok: true)] }
                return [NDAttempt(at: at, ok: day != fail)]
            }
        }
        let stalled = driveStalledAfterDays != nil
        return .read(NDBackupLanes(
            local: nights(newestHoursAgo: localNewestHoursAgo), drive: nights(newestHoursAgo: driveNewestHoursAgo ?? (stalled ? 27 : 3), fail: failedDriveDayAgo),
            conversations: nights(newestHoursAgo: stalled ? 28 : 4), integrity: nights(newestHoursAgo: 4)))
    }

    static func jobs(nightlyFailing: Bool) -> [NDJobInput] {
        func at(_ hours: Double) -> Date { now.addingTimeInterval(hours * 3600) }
        return [
            NDJobInput(job: .watch, enabled: true, continuous: true, lastRun: nil, nextRun: nil, loaded: true),
            NDJobInput(job: .hotlane, enabled: true, continuous: true, lastRun: nil, nextRun: nil, loaded: true),
            NDJobInput(job: .drain, enabled: true, continuous: false, lastRun: at(-0.05), nextRun: at(4 / 60), loaded: true, lastOutcome: .ok),
            NDJobInput(job: .walCheckpoint, enabled: true, continuous: false, lastRun: at(-0.4), nextRun: at(9 / 60), loaded: true, lastOutcome: .ok),
            NDJobInput(job: .backupDaily, enabled: true, continuous: false, lastRun: at(-4.1), nextRun: at(19), loaded: true, lastOutcome: .ok),
            NDJobInput(job: .jsonlBackup, enabled: true, continuous: false, lastRun: at(-4.15), nextRun: at(19), loaded: true, lastOutcome: .ok),
            NDJobInput(job: .maintenanceNightly, enabled: true, continuous: false,
                       lastRun: nightlyFailing ? at(-4.5) : at(-5), nextRun: at(19), loaded: true,
                       lastOutcome: nightlyFailing ? .failed : .ok),
            NDJobInput(job: .maintenanceWeekly, enabled: true, continuous: false, lastRun: at(-96), nextRun: at(72), loaded: true, lastOutcome: .ok),
            NDJobInput(job: .decay, enabled: true, continuous: false, lastRun: at(-18), nextRun: at(2), loaded: true, lastOutcome: .ok),
            NDJobInput(job: .repairFTS, enabled: true, continuous: false, lastRun: at(-144), nextRun: at(22), loaded: true, lastOutcome: .ok),
        ]
    }
}
/// The invented traffic behind every fixture wave: counts per half minute, deterministic in the half minute's absolute
/// index (no seed per range), with bursts of a few minutes. A wave sums it into its buckets on a fixed grid.
enum NDFixtureTraffic {
    static let half: Int = 30

    static func unit(_ x: Int, _ salt: UInt64) -> Double {
        var z = UInt64(bitPattern: Int64(x)) &+ salt &* 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return Double(z >> 11) / Double(UInt64(1) << 53)
    }

    static func counts(_ h: Int) -> NDWave.Bucket {
        let burst = unit(h >= 0 ? h / 16 : (h - 15) / 16, 1) < 0.18
        let reads = Int(7 + unit(h, 2) * 30 * (burst ? 1.6 : 0.7))
        let saves = unit(h, 3) < (burst ? 0.55 : 0.18) ? 1 + Int(unit(h, 4) * (burst ? 2.5 : 1.2)) : 0
        return NDWave.Bucket(saves: saves, reads: reads)
    }

    /// `buckets` buckets over `minutes`, on the grid of that bucket width, the newest one partly filled up to `now`.
    static func wave(minutes: Int, buckets n: Int, now: Date) -> NDWave {
        let width = minutes * 60 / n
        let t = now.timeIntervalSinceReferenceDate
        let end = Int((t / Double(width) - 1e-6).rounded(.up))
        let filledHalves = Int((t / Double(half) - 1e-6).rounded(.up))
        var saves: [Int] = [], reads: [Int] = []
        for k in 0..<n {
            let start = (end - n + k) * width / half
            var s = 0, r = 0
            for h in start..<max(start, min(start + width / half, filledHalves)) {
                let c = counts(h)
                s += c.saves
                r += c.reads
            }
            saves.append(s)
            reads.append(r)
        }
        var wave = NDWave(saves: saves, reads: reads, agentReads: saves.map { $0 * 2 }, agentReadsPeakPerMinute: 12, readable: true,
                          endsAt: Date(timeIntervalSinceReferenceDate: Double(end * width)))
        wave.sequence = end
        return wave
    }
}

#endif
