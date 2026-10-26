import Foundation

/// One thing the status surfaces report, with a stable identity so two inputs that describe the
/// same fault (a job alert and the calendar's failed run) collapse into one row.
struct BrainBarStatusIssue: Equatable, Sendable, Identifiable {
    enum Component: String, Equatable, Sendable {
        case watcher, database, transcripts, backups, maintenance, ingest, configuration, drive, health, jobs
    }

    enum Severity: Equatable, Sendable {
        /// The menu-bar dot. Reserved for the lead's dot policy (S5, 2026-10-26).
        case attention
        /// Visible neutral text; never a dot.
        case notice
    }

    let id: String
    let component: Component
    let severity: Severity
    let text: String
    var jobs: Set<BrainLayerLaunchdJob> = []
    /// How a job group shows this issue when it is only a notice.
    var noticeHealth: BrainLayerLaunchdGroupHealth = .notice
    /// The job-alert key behind the issue, when a job alert reported it.
    var alertKey: String? = nil
}

struct BrainBarStatusInputs: Sendable {
    enum Backups: Sendable {
        case notRead
        case unavailable(String)
        case measured(ObservabilityDocument.Backups)
    }

    var now: Date
    var calendar: Calendar = .current
    var badge: BadgeStatePresentation? = nil
    /// The live job-alert file; nil when it could not be read.
    var jobAlerts: BrainBarJobAlerts? = nil
    var watcher: WatcherHealthStatus? = nil
    var watcherHeartbeatAt: Date? = nil
    var settings: [BrainLayerLaunchdJob: BrainLayerLaunchdJobSetting] = [:]
    /// Nil until the first launchd/calendar sample completes.
    var observations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation]? = nil
    var backups: Backups = .notRead
    var drive: DriveAuthPresentation? = nil
    var maintenance: BrainLayerMaintenanceEvidence = .unread
}

/// The one status verdict every BrainBar surface renders (S5): menu-bar icon dot, menu text,
/// Dashboard headline and attention list, Jobs group badges and the Backups badge.
struct BrainBarStatusSnapshot: Equatable, Sendable {
    let issues: [BrainBarStatusIssue]
    let isChecking: Bool

    var attention: [BrainBarStatusIssue] { issues.filter { $0.severity == .attention } }
    var notices: [BrainBarStatusIssue] { issues.filter { $0.severity == .notice } }
    var needsAttention: Bool { !attention.isEmpty }

    static let checking = Self(issues: [], isChecking: true)

    static func derive(_ inputs: BrainBarStatusInputs) -> Self {
        Self(issues: [], isChecking: false)
    }

    var verdict: BrainBarStatusVerdict {
        .init(attention: attention.map(\.text), notices: notices.map(\.text), isChecking: isChecking)
    }

    var dashboardStatus: BrainBarOnePageStatus { verdict.dashboardStatus }

    func project(_ status: BrainLayerLaunchdGroupStatus, group _: BrainLayerLaunchdJobGroup) -> BrainLayerLaunchdGroupStatus {
        status
    }

    func project(_ health: BrainBarBackupsHealth) -> BrainBarBackupsHealth {
        health
    }
}
