#if DEBUG
import Foundation

/// Stage 2's capture mode and the headless renders' states. `BRAINBAR_DESIGN_FIXTURE=<name>` (DEBUG builds only) drives
/// the real status item, dropdown and window from one named, invented state, with the design forced to `new`. It never
/// writes real state: in this mode the app starts no server, sampler, watcher or job, and reads no file.
enum BrainBarDesignFixture {
    static let environmentKey = "BRAINBAR_DESIGN_FIXTURE"

    enum Name: String, CaseIterable, Sendable {
        case calm
        /// The verdict is 18 minutes old: nothing it said can be confirmed ("Checking on your memory…").
        case checking
        /// A fresh verdict with checks it cannot tell yet ("Can't confirm backups yet").
        case cantTell = "cant-tell"
        case oneFault = "one-fault"
        case reconnectWaiting = "reconnect-waiting"
        case severalFaults = "several-faults"
        case backupsFailedUpload = "backups-failed-upload"
        case jobsFailedStreak = "jobs-failed-streak"

        /// The window page the fixture opens on.
        var page: NDPage {
            switch self {
            case .backupsFailedUpload: .backups
            case .jobsFailedStreak: .jobs
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
        panel.designDefaults = capturePreferences
        panel.applyDesign()
        let status = BrainBarStatusPopoverController(workspace: workspace, runtime: runtime, dashboardPanelController: panel,
                                                     statusBar: statusBar, menuDefaults: capturePreferences)
        return (panel, status)
    }

    /// A fixed moment, so every capture of a fixture draws the same picture.
    static let now: Date = ISO8601DateFormatter().date(from: "2026-11-15T08:00:00+02:00")!

    static func check(_ name: String, _ state: BrainBarHealthVerdict.State = .green, age: Double? = nil,
                      limit: Double? = nil, fails: Int? = nil) -> BrainBarHealthVerdict.Check {
        BrainBarHealthVerdict.Check(name: name, state: state, ageS: age, thresholdS: limit, note: "", unknownTicks: 0,
                                    blind: false, fails: fails)
    }

    /// The calm verdict every fixture starts from: every check green, ages well inside their limits.
    static func calmChecks() -> [BrainBarHealthVerdict.Check] {
        [
            check("ingest", age: 48, limit: 900), check("stores", age: 0, limit: 1800), check("search", fails: 0),
            check("embedding", age: 420, limit: 3600), check("backup_local", age: 4 * 3600, limit: 36 * 3600),
            check("backup_offsite", age: 3 * 3600, limit: 36 * 3600), check("transcripts", age: 4 * 3600, limit: 36 * 3600),
            check("drive_auth"), check("integrity"), check("disk"), check("monitor"),
        ]
    }

    static func capture(_ checks: [BrainBarHealthVerdict.Check], dot: BrainBarHealthVerdict.State = .green,
                        age: TimeInterval = 60) -> BrainBarMenuCapture {
        var c = BrainBarMenuCapture(now: now, timeZone: TimeZone(secondsFromGMT: 7200)!)
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
        switch name {
        case .calm:
            break
        case .cantTell:
            var checks = calmChecks()
            checks = replacing(checks, check("backup_offsite", .unknown))
            checks = replacing(checks, check("search", .unknown))
            checks = replacing(checks, check("stores", .unknown))
            i.capture = capture(checks, dot: .amber)
        case .checking:
            i.capture = capture(calmChecks(), age: 18 * 60)
        case .oneFault, .reconnectWaiting:
            i.capture = capture(replacing(calmChecks(), check("drive_auth", .red)), dot: .red)
            i.capture.driveAuthModel = .invalid
            i.capture = withDriveAge(i.capture, hours: 21)
            i.driveReconnecting = name == .reconnectWaiting
            i.backupHistory = history(driveStalledAfterDays: 1)
        case .severalFaults:
            i.capture = capture(replacing(calmChecks(), check("drive_auth", .red)), dot: .red)
            i.capture.driveAuthModel = .invalid
            i.capture.maintenanceNightFailures = 3
            i.capture = withDriveAge(i.capture, hours: 21)
            i.backupHistory = history(driveStalledAfterDays: 1)
            i.nightlyLastRun = .ran(now.addingTimeInterval(-3 * 86_400 - 4 * 3600))
        case .backupsFailedUpload:
            i.backupHistory = history(failedDriveDayAgo: 3)
        case .jobsFailedStreak:
            i.capture.maintenanceNightFailures = 3
            i.nightlyLastRun = .ran(now.addingTimeInterval(-3 * 86_400 - 4 * 3600))
            i.jobs = jobs(nightlyFailing: true)
        }
        return i
    }

    private static func withDriveAge(_ c: BrainBarMenuCapture, hours: Double) -> BrainBarMenuCapture {
        var c = c
        let checks = (c.verdict?.checks ?? []).map { k -> BrainBarHealthVerdict.Check in
            ["backup_offsite", "transcripts"].contains(k.name) ? check(k.name, age: hours * 3600, limit: 36 * 3600) : k
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
        i.menuWave = wave(24, seed: 7, end: now)
        i.heroWaves = Dictionary(uniqueKeysWithValues: NDRange.allCases.map { ($0, wave(60, seed: 11 + $0.minutes, end: now)) })
        i.agents = NDAgents(agents: [
            .init(family: .claude, sessions: 5, saves: 72), .init(family: .codex, sessions: 3, saves: 31),
            .init(family: .cursor, sessions: 0, saves: 8), .init(family: .gemini, sessions: 0, saves: 0),
        ])
        i.jobs = jobs(nightlyFailing: false)
        i.backupHistory = history()
        i.nightlyLastRun = .ran(now.addingTimeInterval(-5 * 3600))
        i.weeklyLastRun = .ran(now.addingTimeInterval(-4 * 86_400))
        i.newestLocalCopyAt = now.addingTimeInterval(-(4 * 3600 + 2 * 60))
        i.canRevealLocalCopy = true
        return i
    }

    /// Invented traffic with a fixed seed: steady reads, bursty saves, some empty buckets.
    static func wave(_ n: Int, seed: Int, end: Date) -> NDWave {
        var s = UInt32(truncatingIfNeeded: seed &* 2_654_435_761) | 1
        func next() -> Double { s = s &* 1_664_525 &+ 1_013_904_223; return Double(s) / Double(UInt32.max) }
        var saves: [Int] = [], reads: [Int] = [], burst = 0
        for _ in 0..<n {
            if next() < 0.12 { burst = 3 + Int(next() * 4) }
            reads.append(Int(14 + next() * 60 * (burst > 0 ? 1.6 : 0.7)))
            saves.append(next() < 0.5 ? Int(ceil(next() * (burst > 0 ? 5 : 2))) : 0)
            if burst > 0 { burst -= 1 }
        }
        return NDWave(saves: saves, reads: reads, readable: true, endsAt: end)
    }

    /// A week of nightly attempts. Each lane's newest attempt sits at the age its row's gauge says (the Drive copy 3 h,
    /// the others 4 h; with Drive stalled, the Drive lanes stop at 21 h), so the rail and the number agree.
    static func history(failedDriveDayAgo: Int? = nil, driveStalledAfterDays: Int? = nil) -> NDBackupHistory {
        func nights(newestHoursAgo: Double, fail: Int? = nil) -> [NDAttempt] {
            (0...6).reversed().compactMap { day in
                let at = now.addingTimeInterval(-Double(day) * 86_400 - newestHoursAgo * 3600)
                return now.timeIntervalSince(at) <= 144 * 3600 ? NDAttempt(at: at, ok: day != fail) : nil
            }
        }
        let stalled = driveStalledAfterDays != nil
        return .read(NDBackupLanes(
            local: nights(newestHoursAgo: 4), drive: nights(newestHoursAgo: stalled ? 21 : 3, fail: failedDriveDayAgo),
            conversations: nights(newestHoursAgo: stalled ? 21 : 4), integrity: nights(newestHoursAgo: 4)))
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
#endif
