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

    var title: String {
        switch self {
        case .healthy: "Healthy"
        case .awaitingRun: "Awaiting next run"
        case .unhealthy: "Needs attention"
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
        formatDate: (Date) -> String
    ) -> BrainLayerLaunchdGroupStatus {
        let reason = jobs.compactMap { job -> String? in
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
        let awaitingRun = jobs.contains { job in
            guard let observation = observations[job] else { return false }
            return observation.loadState == .loaded && observation.runs == 0 && observation.lastExitCode == nil
        }
        return BrainLayerLaunchdGroupStatus(
            health: reason != nil ? .unhealthy : awaitingRun ? .awaitingRun : .healthy,
            attentionReason: reason,
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
