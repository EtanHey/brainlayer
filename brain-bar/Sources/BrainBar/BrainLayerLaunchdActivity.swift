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
    /// A maintenance safety gate deferred the last run (exit 75): the gate did its job. Neutral.
    case skipped

    var title: String {
        switch self {
        case .healthy: "Healthy"
        case .awaitingRun: "Awaiting next run"
        case .unhealthy: "Needs attention"
        case .unknown: "Status unknown"
        case .skipped: "Skipped"
        }
    }
}

/// What the maintenance jobs recorded beyond launchd: the weekly's last COMPLETED full pass (the
/// #1015 reader) and the newest record each job's last run printed to its StandardOutPath.
struct BrainLayerMaintenanceEvidence: Equatable, Sendable {
    enum WeeklyCompletion: Equatable, Sendable {
        case completed(Date)
        /// The history is readable and holds no completed full pass.
        case noneRecorded
        /// Missing, unreadable, not UTF-8, or no parseable row at all.
        case unread
    }

    /// Only the NEWEST stdout record, never an older one searched for.
    enum RunRecord: Equatable, Sendable {
        /// A complete `{"status": "aborted", "reason": …}` line; `writtenAt` is the log's mtime.
        case aborted(reason: String, writtenAt: Date)
        /// A complete record that is not an abort, e.g. a successful run's `{"status": "ok"}`.
        case notAnAbort(writtenAt: Date)
        case unavailable(String)
    }

    let weeklyCompletion: WeeklyCompletion
    let runRecords: [BrainLayerLaunchdJob: RunRecord]

    static let unread = Self(weeklyCompletion: .unread, runRecords: [:])
}

/// `src/brainlayer/maintenance.py` exit codes, as the Maintenance card reads them.
enum BrainLayerMaintenanceExit {
    /// `MaintenanceAbort`'s default code: a gate deferral, but also real failures (#1040).
    static let deferred: Int32 = 75
    /// The weekly VACUUM was skipped because the fresh backup failed.
    static let vacuumSkipped: Int32 = 76
    /// A weekly pass older than this, with only skips since, is attention.
    static let staleCompletion: TimeInterval = 8 * 86_400
    /// The abort line is printed when the run ends: after the run-start receipt, and at most the
    /// maintenance lock wait (`MAINTENANCE_LOCK_TIMEOUT_SECONDS`, 4 h) plus the gates later.
    static let recordClockSlack: TimeInterval = 5
    static let recordLatestAfterStart: TimeInterval = 4 * 3_600 + 15 * 60

    /// The deliberate deferrals `maintenance.py` raises with exit 75 before any service is quiesced
    /// (`_check_quiet_window`, `_check_idle`, `_check_lsof_clean`), matched whole and
    /// case-insensitively. Every other 75 reason is attention: this is an allowlist, not a blacklist.
    static let deliberateDeferrals = [
        #"^outside quiet window: now=\S+ start_hour=\d+ duration_minutes=\d+$"#,
        #"^recent queue write activity: \d+ file\(s\) modified recently$"#,
        #"^queue depth growing: before=\d+ after=\d+$"#,
        #"^unexpected writer holds brainlayer db: pid=\d+ command=.+ fd=\S+$"#,
    ]

    static func isDeliberateDeferral(_ reason: String) -> Bool {
        !reason.localizedCaseInsensitiveContains("failed to resume") && deliberateDeferrals.contains {
            reason.range(of: $0, options: [.regularExpression, .caseInsensitive]) != nil
        }
    }

    enum Verdict: Equatable {
        /// A verified, allowlisted gate deferral, in human words.
        case deferred(String)
        /// This run's own abort reason, not on the allowlist.
        case failed(String)
        /// No verified record for this run; the string says why.
        case unexplained(String)
    }

    /// Exit 75's meaning, from the newest stdout record only when it is this run's own abort.
    static func verdict(_ record: BrainLayerMaintenanceEvidence.RunRecord?, lastRunAt: Date?) -> Verdict {
        guard let lastRunAt else { return .unexplained("no run start is on record") }
        switch record {
        case nil: return .unexplained("its stdout log was not read")
        case let .unavailable(why): return .unexplained(why)
        case .notAnAbort: return .unexplained("the newest stdout record is not an abort")
        case let .aborted(reason, _):
            guard let reason = thisRunsReason(record, lastRunAt: lastRunAt) else {
                return .unexplained("the newest stdout record is not from this run")
            }
            return isDeliberateDeferral(reason) ? .deferred(humanDeferral(reason)) : .failed(reason)
        }
    }

    /// The abort reason, only when the record was written during the run that started at `lastRunAt`.
    static func thisRunsReason(_ record: BrainLayerMaintenanceEvidence.RunRecord?, lastRunAt: Date?) -> String? {
        guard let lastRunAt, case let .aborted(reason, writtenAt) = record else { return nil }
        let delay = writtenAt.timeIntervalSince(lastRunAt)
        return delay >= -recordClockSlack && delay <= recordLatestAfterStart ? reason : nil
    }

    /// `outside quiet window: now=… start_hour=4 duration_minutes=120` → `outside the 04:00–06:00 quiet window`.
    static func humanDeferral(_ reason: String) -> String {
        guard reason.lowercased().hasPrefix("outside quiet window") else { return reason }
        func value(_ key: String) -> Int? {
            reason.lowercased().split(separator: " ").first { $0.hasPrefix("\(key)=") }.flatMap { Int($0.dropFirst(key.count + 1)) }
        }
        guard let hour = value("start_hour"), let minutes = value("duration_minutes") else {
            return "outside the maintenance quiet window"
        }
        let end = (hour * 60 + minutes) % (24 * 60)
        return String(format: "outside the %02d:00–%02d:%02d quiet window", hour, end / 60, end % 60)
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
        maintenance: BrainLayerMaintenanceEvidence = .unread,
        now: Date = Date()
    ) -> BrainLayerLaunchdGroupStatus {
        // Ingest owns the watcher, so its health is the one WatcherHealthStatus (#966): the same
        // reason the Dashboard and footer show, not a second launchd-only verdict.
        let ingestWatcher = self == .ingest ? watcher : nil
        let watcherAttention = ingestWatcher?.needsAttention == true ? ingestWatcher?.reasonText(now: now) : nil
        let watcherUnknown: String? = if case .unknown = ingestWatcher { ingestWatcher?.reasonText(now: now) } else { nil }
        // Exit 75 in Maintenance: Skipped only for a verified deferral; unknown or attention otherwise.
        var deferrals: [BrainLayerLaunchdJob: MaintenanceDeferral] = [:]
        if self == .maintenance {
            for job in jobs {
                guard settings[job]?.enabled == true, let observation = observations[job], observation.loadState == .loaded,
                      observation.lastExitCode == BrainLayerMaintenanceExit.deferred else { continue }
                deferrals[job] = deferral(job, observation: observation, maintenance: maintenance, formatDate: formatDate, now: now)
            }
        }
        let jobReason = jobs.compactMap { job -> String? in
            guard settings[job]?.enabled == true else { return "\(job.humanGroupLabel) is disabled." }
            guard let observation = observations[job] else { return "\(job.humanGroupLabel) status is unavailable." }
            switch observation.loadState {
            case .running:
                return nil
            case .loaded:
                if let code = observation.lastExitCode, code != 0 {
                    let when = observation.lastRunAt.map { " at \(formatDate($0))" } ?? ""
                    if let deferral = deferrals[job] {
                        if case let .attention(text) = deferral { return text }
                        return nil
                    }
                    if self == .maintenance, code == BrainLayerMaintenanceExit.vacuumSkipped {
                        let detail = BrainLayerMaintenanceExit.thisRunsReason(
                            maintenance.runRecords[job], lastRunAt: observation.lastRunAt
                        ).map { reason in
                            let trimmed = reason.hasSuffix("; VACUUM skipped") ? String(reason.dropLast("; VACUUM skipped".count)) : reason
                            return " (\(trimmed))"
                        } ?? ""
                        return "\(job.humanGroupLabel) VACUUM skipped\(when): the fresh backup failed\(detail)."
                    }
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
        let deferralUnknown = jobs.lazy.compactMap { job -> String? in
            if case let .unknown(text) = deferrals[job] { text } else { nil }
        }.first
        let skips = jobs.compactMap { job -> String? in
            if case let .skipped(text) = deferrals[job] { text } else { nil }
        }
        let unknownReason = watcherUnknown ?? deferralUnknown
        let awaitingRun = jobs.contains { job in
            guard let observation = observations[job] else { return false }
            return observation.loadState == .loaded && observation.runs == 0 && observation.lastExitCode == nil
        }
        let health: BrainLayerLaunchdGroupHealth = reason != nil ? .unhealthy
            : unknownReason != nil ? .unknown
            : !skips.isEmpty ? .skipped
            : awaitingRun ? .awaitingRun : .healthy
        return BrainLayerLaunchdGroupStatus(
            health: health,
            attentionReason: reason ?? unknownReason ?? (skips.isEmpty ? nil : skips.joined(separator: " ")),
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

/// One maintenance job's exit 75, as the card shows it.
private enum MaintenanceDeferral {
    case attention(String)
    case unknown(String)
    case skipped(String)
}

private extension BrainLayerLaunchdJobGroup {
    func deferral(
        _ job: BrainLayerLaunchdJob,
        observation: BrainLayerLaunchdJobObservation,
        maintenance: BrainLayerMaintenanceEvidence,
        formatDate: (Date) -> String,
        now: Date
    ) -> MaintenanceDeferral {
        let label = job.humanGroupLabel
        let when = observation.lastRunAt.map { " at \(formatDate($0))" } ?? ""
        let verdict = BrainLayerMaintenanceExit.verdict(maintenance.runRecords[job], lastRunAt: observation.lastRunAt)
        if case let .failed(reason) = verdict {
            return .attention("\(label) last run exited 75\(when): \(reason)")
        }
        // Rule 3: skips, explained or not, must not hide a weekly pass that never completes.
        if job == .maintenanceWeekly {
            let lastRun = switch verdict {
            case let .deferred(text): "last run skipped: \(text)."
            case let .unexplained(why): "last run exited 75, reason unavailable (\(why))."
            case .failed: ""
            }
            switch maintenance.weeklyCompletion {
            case let .completed(at) where now.timeIntervalSince(at) > BrainLayerMaintenanceExit.staleCompletion:
                return .attention("Weekly maintenance hasn't completed since \(formatDate(at)); \(lastRun)")
            case .noneRecorded:
                return .attention("Weekly maintenance has no completed pass on record; \(lastRun)")
            case .completed, .unread:
                break
            }
        }
        switch verdict {
        case let .unexplained(why):
            return .unknown("\(label) exited 75\(when); reason unavailable (\(why)).")
        case let .deferred(text):
            // Without the completion history, a weekly skip cannot be told apart from one that
            // hides a pass that stopped completing.
            if job == .maintenanceWeekly, maintenance.weeklyCompletion == .unread {
                return .unknown("\(label) skipped\(when): \(text); its last completed pass could not be read.")
            }
            return .skipped("\(label) skipped\(when): \(text).")
        case let .failed(reason):
            return .attention("\(label) last run exited 75\(when): \(reason)")
        }
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
        // Every red line the page shows, most severe first: the failed job's own alert, Drive
        // access, the launchd job, then the other backup diagnostics.
        let red = [
            status.flatMap { $0.errorIsJobAlert ? $0.attentionLine?.text : nil },
            drive?.tone == .attention ? drive?.line : nil,
            job.health == .unhealthy ? job.attentionReason : nil,
            status.map { $0.attentionLine?.text } ?? statusUnavailableReason,
        ].compactMap { $0 }
        if let reason = red.first { return .init(badge: .attention, reason: reason) }
        switch drive?.tone {
        case .expiring: return .init(badge: .expiring, reason: drive?.line)
        case .unknown: return .init(badge: .unknown, reason: drive?.line)
        case .attention, .connected, nil: break
        }
        return switch job.health {
        case .healthy: .init(badge: .healthy, reason: nil)
        case .awaitingRun: .init(badge: .awaitingRun, reason: nil)
        // Unreachable for Backups (only the Maintenance group skips); a deferral is not a failure.
        case .skipped: .init(badge: .healthy, reason: nil)
        case .unknown: .init(badge: .unknown, reason: job.attentionReason)
        case .unhealthy: .init(badge: .attention, reason: job.attentionReason)
        }
    }
}
