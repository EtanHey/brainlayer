import Foundation

struct BrainLayerLaunchdJobObservation: Equatable, Sendable {
    let loadState: BrainLayerLaunchdLoadState
    let runs: Int?
    let lastExitCode: Int32?
    let lastRunAt: Date?
    let nextRunAt: Date?
    let isContinuous: Bool

    static func stateOnly(_ state: BrainLayerLaunchdLoadState) -> Self {
        Self(
            loadState: state,
            runs: nil,
            lastExitCode: nil,
            lastRunAt: nil,
            nextRunAt: nil,
            isContinuous: false
        )
    }
}

enum BrainLayerLaunchdGroupHealth: Equatable, Sendable {
    case healthy
    case awaitingRun
    case unhealthy
    /// The watcher's health cannot be determined (#966): an honest "can't tell", not a failure.
    case unknown

    var title: String {
        switch self {
        case .healthy: "Healthy"
        case .awaitingRun: "Awaiting next run"
        case .unhealthy: "Needs attention"
        case .unknown: "Status unknown"
        }
    }
}

struct BrainLayerLaunchdGroupStatus: Equatable, Sendable {
    let health: BrainLayerLaunchdGroupHealth
    let attentionReason: String?
    let lastRunText: String
    let nextRunText: String
}

enum BrainLayerLaunchdJobGroup: String, CaseIterable, Identifiable, Sendable {
    case ingest
    case backups
    case maintenance

    var id: String { rawValue }

    var title: String {
        switch self {
        case .ingest: "Ingest"
        case .backups: "Backups"
        case .maintenance: "Maintenance"
        }
    }

    var summary: String {
        switch self {
        case .ingest: "Watches coding transcripts and indexes new memory."
        case .backups: "Copies the database and transcript archives."
        case .maintenance: "Runs nightly and weekly database housekeeping."
        }
    }

    var jobs: [BrainLayerLaunchdJob] {
        switch self {
        case .ingest: [.watch, .index]
        case .backups: [.backupDaily, .jsonlBackup]
        case .maintenance: [.maintenanceNightly, .maintenanceWeekly]
        }
    }

    static let advancedJobs: [BrainLayerLaunchdJob] = [
        .walCheckpoint,
        .repairFTS,
        .decay,
        .drain,
        .hotlane,
    ]

    func status(
        settings: [BrainLayerLaunchdJob: BrainLayerLaunchdJobSetting],
        observations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation],
        formatDate: (Date) -> String,
        watcher: WatcherHealthStatus? = nil,
        now: Date = Date()
    ) -> BrainLayerLaunchdGroupStatus {
        // Ingest owns the watcher, so its health is the one WatcherHealthStatus (#966): the same
        // reason the Dashboard and footer show, not a second launchd-only verdict.
        let ingestWatcher = self == .ingest ? watcher : nil
        let watcherAttention = ingestWatcher?.needsAttention == true ? ingestWatcher?.reasonText(now: now) : nil
        let watcherUnknown: String? = if case .unknown = ingestWatcher { ingestWatcher?.reasonText(now: now) } else { nil }
        let jobReason = jobs.compactMap { job -> String? in
            guard settings[job]?.enabled == true else { return "\(job.humanGroupLabel) is disabled." }
            guard let observation = observations[job] else { return "\(job.humanGroupLabel) status is unavailable." }
            switch observation.loadState {
            case .running:
                return nil
            case .loaded:
                if let code = observation.lastExitCode, code != 0 {
                    let when = observation.lastRunAt.map { " at \(formatDate($0))" } ?? ""
                    return "\(job.humanGroupLabel) last run exited \(code)\(when)."
                }
                return nil // launchd resets run counters when a job is reloaded.
            case .unloaded:
                return "\(job.humanGroupLabel) is unloaded."
            case .unknown, .probeError:
                return "\(job.humanGroupLabel) status is unavailable."
            }
        }.first
        let reason = watcherAttention ?? jobReason
        let awaitingRun = jobs.contains { job in
            guard let observation = observations[job] else { return false }
            return observation.loadState == .loaded && observation.runs == 0 && observation.lastExitCode == nil
        }
        return BrainLayerLaunchdGroupStatus(
            health: reason != nil ? .unhealthy : watcherUnknown != nil ? .unknown : awaitingRun ? .awaitingRun : .healthy,
            attentionReason: reason ?? watcherUnknown,
            lastRunText: jobs.map { job in
                "\(job.humanGroupLabel) \(observations[job]?.lastRunAt.map(formatDate) ?? "No run recorded")"
            }.joined(separator: " · "),
            nextRunText: jobs.map { job in
                let observation = observations[job]
                let value = observation?.isContinuous == true
                    ? "Continuous"
                    : observation?.nextRunAt.map(formatDate) ?? "Unavailable"
                return "\(job.humanGroupLabel) \(value)"
            }.joined(separator: " · ")
        )
    }
}

enum BrainBarSettingsPresentation {
    static let defaultAdvancedExpanded = false

    static func visibleAdvancedJobs(isExpanded: Bool) -> [BrainLayerLaunchdJob] {
        isExpanded ? BrainLayerLaunchdJobGroup.advancedJobs : []
    }
}

private extension BrainLayerLaunchdJob {
    var humanGroupLabel: String {
        switch self {
        case .watch: "Watcher"
        case .index: "Index"
        case .backupDaily: "Database"
        case .jsonlBackup: "Transcripts"
        case .maintenanceNightly: "Nightly"
        case .maintenanceWeekly: "Weekly"
        default: title
        }
    }
}

extension WatcherLaunchdEvidence {
    /// Settings' view of `com.brainlayer.watch`: the configured setting plus the launchd observation.
    init(setting: BrainLayerLaunchdJobSetting?, loadState: BrainLayerLaunchdLoadState?) {
        if setting?.enabled == false {
            self = .notRunning("the Watcher job is disabled in Settings")
            return
        }
        switch loadState {
        case .running:
            self = .running
        case .loaded:
            self = .notRunning("com.brainlayer.watch is loaded but not running")
        case .unloaded:
            self = .notRunning("com.brainlayer.watch is not loaded")
        case let .probeError(detail):
            self = .unavailable(detail)
        case .unknown, nil:
            self = .unavailable("launchd status for com.brainlayer.watch is unknown")
        }
    }
}

/// #1029 review B1: the Backups page's one verdict. The group badge, the Google Drive card and the
/// backup status lines all come from it, so the page can never say "Backups can't upload" beside
/// "Healthy".
struct BrainBarBackupsHealth: Equatable, Sendable {
    enum Badge: Equatable, Sendable {
        case healthy, awaitingRun, expiring, attention, unknown

        var title: String {
            switch self {
            case .healthy: "Healthy"
            case .awaitingRun: "Awaiting next run"
            case .expiring: "Drive access expiring"
            case .attention: "Needs attention"
            case .unknown: "Status unknown"
            }
        }
    }

    let badge: Badge
    /// The most severe reason on the page, shown under the badge. Nil when there is none.
    let reason: String?

    static func derive(
        job: BrainLayerLaunchdGroupStatus,
        drive: DriveAuthPresentation?,
        status: ObservabilityBackupStatus?,
        statusUnavailableReason: String
    ) -> Self {
        let badge: Badge = switch job.health {
        case .healthy: .healthy
        case .awaitingRun: .awaitingRun
        case .unhealthy: .attention
        case .unknown: .unknown
        }
        return .init(badge: badge, reason: job.attentionReason)
    }
}
