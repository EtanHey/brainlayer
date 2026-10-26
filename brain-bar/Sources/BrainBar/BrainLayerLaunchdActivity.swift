import Foundation

/// How a job or group note is drawn: `info` is neutral text, `warning` the amber ⚠ (2026-10-21).
/// Explicit, never inferred from the note's words.
enum BrainLayerLaunchdNoteTone: Equatable, Sendable {
    case info
    case warning
}

struct BrainLayerLaunchdJobObservation: Equatable, Sendable {
    let loadState: BrainLayerLaunchdLoadState
    let runs: Int?
    let lastExitCode: Int32?
    let lastRunAt: Date?
    let nextRunAt: Date?
    let isContinuous: Bool
    var pid: Int32? = nil
    var ownership: BrainBarJobOwnership = .legacy
    var ownerNote: String? = nil
    /// Owner notes are informational ("last result: ok", "PID …"); only a failed result is a warning.
    var ownerNoteTone: BrainLayerLaunchdNoteTone = .info
    /// The nightly did not run because the weekly ran in its place and finished ok, at this time.
    var calendarCoveredByWeeklyAt: Date? = nil
    var calendarStartedAt: Date? = nil
    var calendarDisabled = false
    var calendarDeferralRecord: BrainLayerMaintenanceEvidence.RunRecord? = nil
    var calendarMissedWindowReason: String? = nil
    var calendarIndexDeferralReason: String? = nil
    /// S5: failed calendar runs of this step in a row, newest first (deferrals neither count nor reset).
    var calendarConsecutiveFailures: Int? = nil
    /// S5: the schedule or parent probe could not be read; the known run above still stands.
    var calendarScheduleUncertainty: String? = nil
    /// S5: the installed calendar LaunchAgent exists but is not a usable schedule (malformed config).
    var calendarScheduleProblem: String? = nil

    var backupScheduleLine: ObservabilityStatusLine? {
        guard ownership != .legacy else { return nil }
        if ownership == .supervisor, loadState == .loaded, lastExitCode == 75 {
            if let calendarIndexDeferralReason {
                return .init(text: "Deferred — " + calendarIndexDeferralReason, tone: .neutral)
            }
            if case let .deferred(reason) = calendarDeferralVerdict {
                return .init(text: "Skipped — " + reason, tone: .neutral)
            }
            return .init(text: calendarStartedAt == nil ? "Deferred — not run; check its calendar window"
                : "Deferred — exit 75; no verified deliberate deferral", tone: .neutral)
        }
        switch loadState {
        case .loaded, .running:
            return .init(text: ownerNote ?? "Maintenance calendar", tone: .green)
        case let .ownerUnknown(reason), let .probeError(reason):
            return .init(text: reason, tone: .neutral)
        case let .failed(reason): return .init(text: reason, tone: .red)
        case .paused: return .init(text: "Maintenance calendar paused", tone: .neutral)
        case .unknown, .unloaded: return .init(text: "Maintenance calendar status unavailable", tone: .neutral)
        }
    }

    /// The last-run cell on Jobs and Backups: a covered nightly names the weekly that covered it.
    func lastRunText(formatDate: (Date) -> String, never: String) -> String {
        if let calendarCoveredByWeeklyAt { return "Covered by weekly · " + formatDate(calendarCoveredByWeeklyAt) }
        return lastRunAt.map(formatDate) ?? never
    }

    var calendarDeferralVerdict: BrainLayerMaintenanceExit.Verdict {
        if let calendarMissedWindowReason { return .failed(calendarMissedWindowReason) }
        if calendarDisabled { return .deferred("disabled") }
        return BrainLayerMaintenanceExit.verdict(calendarDeferralRecord, lastRunAt: calendarStartedAt)
    }

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
    case checking
    case healthy
    case awaitingRun
    case unhealthy
    /// The watcher's health cannot be determined (#966): an honest "can't tell", not a failure.
    case unknown
    /// A maintenance safety gate deferred the last run (exit 75): the gate did its job. Neutral.
    case skipped
    case deferred
    /// S5: a known non-urgent fault (a single failed run that retries, an unreadable schedule).
    /// Visible text, never the dot.
    case notice

    var title: String {
        switch self {
        case .checking: "Checking…"
        case .healthy: "Healthy"
        case .awaitingRun: "Awaiting next run"
        case .unhealthy: "Needs attention"
        case .unknown: "Status unknown"
        case .skipped: "Skipped"
        case .deferred: "Deferred"
        case .notice: "No action needed"
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
        /// A complete record that is not an abort, e.g. a successful run's `{"status": "ok"}`, with
        /// the `warnings` a successful run may carry (#1061: e.g. search latency above target).
        case notAnAbort(writtenAt: Date, warnings: [String] = [])
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
    /// (`_check_quiet_window`, `_check_idle`, `_check_lsof_clean`, a refused window-bound hold), matched whole and
    /// case-insensitively. Every other 75 reason is attention: this is an allowlist, not a blacklist.
    static let deliberateDeferrals = [
        #"^outside quiet window: now=\S+ start_hour=\d+ duration_minutes=\d+$"#,
        #"^recent queue write activity: \d+ file\(s\) modified recently$"#,
        #"^queue depth growing: before=\d+ after=\d+$"#,
        #"^unexpected writer holds brainlayer db: pid=\d+ command=.+ fd=\S+$"#,
        #"^quiet window leaves no time for a bounded maintenance hold$"#,
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
        case .aborted:
            guard let reason = thisRunsReason(record, lastRunAt: lastRunAt) else {
                return .unexplained("the newest stdout record is not from this run")
            }
            return isDeliberateDeferral(reason) ? .deferred(humanDeferral(reason)) : .failed(reason)
        }
    }

    /// The abort reason, only when the record was written during the run that started at `lastRunAt`.
    static func thisRunsReason(_ record: BrainLayerMaintenanceEvidence.RunRecord?, lastRunAt: Date?) -> String? {
        guard let lastRunAt, case let .aborted(reason, writtenAt) = record else { return nil }
        return isThisRun(writtenAt, lastRunAt: lastRunAt) ? reason : nil
    }

    /// A successful run's warnings, only when the record was written during the run that started at
    /// `lastRunAt`; a later run without warnings writes a newer record, which clears them.
    static func thisRunsWarnings(_ record: BrainLayerMaintenanceEvidence.RunRecord?, lastRunAt: Date?) -> [String] {
        guard let lastRunAt, case let .notAnAbort(writtenAt, warnings) = record,
              isThisRun(writtenAt, lastRunAt: lastRunAt) else { return [] }
        return warnings
    }

    private static func isThisRun(_ writtenAt: Date, lastRunAt: Date) -> Bool {
        let delay = writtenAt.timeIntervalSince(lastRunAt)
        return delay >= -recordClockSlack && delay <= recordLatestAfterStart
    }

    /// `post-maintenance search latency above target: 62.0ms > 50.0ms` → `search slower than target
    /// (62 ms)`; any other warning is shown as written.
    static func humanWarning(_ warning: String) -> String {
        let pattern = #"search latency above target: ([0-9.]+)\s*ms"#
        guard let match = warning.range(of: pattern, options: [.regularExpression, .caseInsensitive]),
              let value = Double(warning[match].replacingOccurrences(
                  of: #"^.*: ([0-9.]+)\s*ms$"#, with: "$1", options: .regularExpression
              )) else { return warning }
        // Int(_:) traps outside its range: anything not a sane, finite latency is shown as written.
        guard value.isFinite, value >= 0, value < 1_000_000 else { return warning }
        return "search slower than target (\(Int(value.rounded())) ms)"
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
    /// Desired Ingest configuration or maintenance warnings; never changes runtime `health`.
    var note: String?
    /// Defaults to `warning`, so a note nobody classified is never silently drawn neutral.
    var noteTone: BrainLayerLaunchdNoteTone = .warning
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

    func disabledJobReason(settings: [BrainLayerLaunchdJob: BrainLayerLaunchdJobSetting]) -> String? {
        jobs.first { settings[$0]?.enabled != true }.map { "\($0.humanGroupLabel) is disabled." }
    }

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
        do {
            for job in jobs {
                guard settings[job]?.enabled != false,
                      let observation = observations[job], (self == .maintenance || observation.ownership == .supervisor), observation.loadState == .loaded,
                      observation.lastExitCode == BrainLayerMaintenanceExit.deferred else { continue }
                deferrals[job] = deferral(job, observation: observation, maintenance: maintenance, formatDate: formatDate, now: now)
            }
        }
        let jobReason = jobs.compactMap { job -> String? in
            // Desired OFF is neutral; sampled operational disable is attention only for desired ON.
            guard settings[job]?.enabled != false else { return nil }
            if settings[job]?.enabled != false, observations[job]?.ownership == .supervisor,
               observations[job]?.calendarDisabled == true { return "\(job.humanGroupLabel) is disabled." }
            // The unified watcher verdict already accounts for missing/error launchd evidence.
            if job == .watch, ingestWatcher != nil { return nil }
            // Ingest configuration is desired state; toggling it does not stop loaded jobs.
            guard let observation = observations[job] else { return "\(job.humanGroupLabel) status is unavailable." }
            switch observation.loadState {
            case .running:
                return nil
            case .loaded:
                if let code = observation.lastExitCode, code != 0 {
                    let when = observation.lastRunAt.map { " at \(formatDate($0))" } ?? ""
                    if let deferral = deferrals[job] {
                        if case let .attention(text) = deferral, observation.ownership != .supervisor { return text }
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
            case .ownerUnknown:
                return nil
            case let .failed(reason):
                return reason
            case .paused:
                return "\(job.humanGroupLabel) is paused under the supervisor."
            case .unknown, .probeError:
                return "\(job.humanGroupLabel) status is unavailable."
            }
        }.first
        let calendarAttention = jobs.compactMap { job -> String? in
            guard observations[job]?.ownership == .supervisor, case let .attention(text) = deferrals[job] else { return nil }
            return text
        }.first
        let reason = watcherAttention ?? jobReason
        let deferralUnknown = jobs.lazy.compactMap { job -> String? in
            if case let .unknown(text) = deferrals[job] { text } else { nil }
        }.first
        let skips = jobs.compactMap { job -> String? in
            if case let .skipped(text) = deferrals[job] { text } else { nil }
        }
        let ownerUnknown = jobs.compactMap { job -> String? in
            if case let .ownerUnknown(reason) = observations[job]?.loadState { return reason }
            return nil
        }.first
        let unknownReason = ownerUnknown ?? watcherUnknown ?? deferralUnknown
        let awaitingRun = jobs.contains { job in
            guard settings[job]?.enabled != false, let observation = observations[job],
                  observation.calendarCoveredByWeeklyAt == nil else { return false }
            return observation.loadState == .loaded && observation.runs == 0 && observation.lastExitCode == nil
        }
        let configuredOff = jobs.filter { settings[$0]?.enabled == false }
        let health: BrainLayerLaunchdGroupHealth = reason != nil ? .unhealthy
            : calendarAttention != nil ? .deferred
            : unknownReason != nil ? .unknown
            : !skips.isEmpty || configuredOff.count == jobs.count ? .skipped
            : awaitingRun ? .awaitingRun : .healthy
        // N1 (lead addendum 2026-10-04): a clean exit whose own record carries warnings.
        let warningNotes = self != .maintenance ? [] : jobs.compactMap { job -> String? in
            guard let observation = observations[job], observation.lastExitCode == 0 else { return nil }
            let warnings = BrainLayerMaintenanceExit.thisRunsWarnings(maintenance.runRecords[job], lastRunAt: observation.lastRunAt)
            guard !warnings.isEmpty else { return nil }
            return "\(job.humanGroupLabel) last run OK · "
                + warnings.map(BrainLayerMaintenanceExit.humanWarning).joined(separator: " · ")
        }
        let ownerNotes = jobs.compactMap { observations[$0]?.ownerNote }.reduce(into: [String]()) {
            if !$0.contains($1) { $0.append($1) }
        }
        let ownerNoteTone: BrainLayerLaunchdNoteTone = jobs.contains {
            observations[$0]?.ownerNote != nil && observations[$0]?.ownerNoteTone == .warning
        } ? .warning : .info
        let configurationNote = configuredOff.isEmpty ? nil
            : "Configured OFF: \(configuredOff.map(\.humanGroupLabel).joined(separator: ", "))."
        // Configuration and run warnings are warnings; owner notes carry their own tone.
        let note: String?
        let noteTone: BrainLayerLaunchdNoteTone
        if let configurationNote { (note, noteTone) = (configurationNote, .warning) }
        else if !warningNotes.isEmpty { (note, noteTone) = (warningNotes.joined(separator: " · "), .warning) }
        else if !ownerNotes.isEmpty { (note, noteTone) = (ownerNotes.joined(separator: " · "), ownerNoteTone) }
        else { (note, noteTone) = (nil, .info) }
        return BrainLayerLaunchdGroupStatus(
            health: health,
            attentionReason: reason ?? calendarAttention ?? unknownReason ?? (skips.isEmpty ? nil : skips.joined(separator: " ")),
            lastRunText: jobs.map { job in
                let never = observations[job]?.ownership == .supervisor ? "Not yet run under the supervisor" : "No run recorded"
                return "\(job.humanGroupLabel) \(observations[job]?.lastRunText(formatDate: formatDate, never: never) ?? never)"
            }.joined(separator: " · "),
            nextRunText: jobs.map { job in
                let observation = observations[job]
                let value = observation?.isContinuous == true
                    ? "Continuous"
                    : observation?.nextRunAt.map(formatDate) ?? "Unavailable"
                return "\(job.humanGroupLabel) \(value)"
            }.joined(separator: " · "),
            note: note,
            noteTone: noteTone
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
        if let reason = observation.calendarIndexDeferralReason {
            return .skipped("Deferred — " + reason)
        }
        let label = job.humanGroupLabel
        let when = observation.lastRunAt.map { " at \(formatDate($0))" } ?? ""
        let verdict = observation.ownership == .supervisor ? observation.calendarDeferralVerdict
            : BrainLayerMaintenanceExit.verdict(maintenance.runRecords[job], lastRunAt: observation.lastRunAt)
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
            return observation.ownership == .supervisor
                ? .attention("\(label) deferred\(when); reason unavailable (\(why)).")
                : .unknown("\(label) exited 75\(when); reason unavailable (\(why)).")
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
    /// Settings' runtime evidence comes from launchd. Configuration expresses desired state;
    /// saving it does not stop the process, and cannot replace an unknown/error observation.
    init(setting _: BrainLayerLaunchdJobSetting?, loadState: BrainLayerLaunchdLoadState?) {
        switch loadState {
        case .running:
            self = .running
        case .loaded:
            self = .notRunning("com.brainlayer.watch is loaded but not running")
        case .unloaded:
            self = .notRunning("com.brainlayer.watch is not loaded")
        case let .ownerUnknown(detail):
            self = .ownerUnknown(detail)
        case .paused:
            self = .notRunning("Supervisor watcher is paused")
        case let .failed(detail):
            self = .notRunning("Supervisor watcher failed: \(detail)")
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
        case checking, healthy, awaitingRun, expiring, attention, unknown, skipped, deferred, idle, queued, uploading

        var title: String {
            switch self {
            case .checking: "Checking…"
            case .healthy: "Healthy"
            case .idle: "Idle — no new transcripts"
            case .queued: "Upload queued"
            case .uploading: "Upload in progress"
            case .awaitingRun: "Awaiting next run"
            case .expiring: "Drive access expiring"
            case .attention: "Needs attention"
            case .unknown: "Status unknown"
            case .skipped: "Skipped"
            case .deferred: "Deferred"
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
        statusUnavailableReason: String,
        attentionSentence: String? = nil,
        isCheckingStatus: Bool = false
    ) -> Self {
        // Every red line the page shows, most severe first: the failed job's own alert, Drive
        // access, the launchd job, then the other backup diagnostics. `attentionSentence` is the
        // same red line in the Backups page's plain words (E1); it never adds or removes one.
        let statusAttention = status.map { status in
            status.attentionLine.map { attentionSentence ?? $0.text }
        }
        let red = [
            status.flatMap { $0.errorIsJobAlert ? $0.attentionLine?.text : nil },
            drive?.tone == .attention ? drive?.line : nil,
            job.health == .unhealthy ? job.attentionReason : nil,
            statusAttention ?? (isCheckingStatus ? nil : statusUnavailableReason),
        ].compactMap { $0 }
        if let reason = red.first { return .init(badge: .attention, reason: reason) }
        if isCheckingStatus || job.health == .checking { return .init(badge: .checking, reason: nil) }
        switch drive?.tone {
        case .expiring: return .init(badge: .expiring, reason: drive?.line)
        case .unknown: return .init(badge: .unknown, reason: drive?.line)
        case .attention, .connected, nil: break
        }
        if job.health == .healthy || job.health == .awaitingRun {
            if status?.uploadUnconfirmed == true { return .init(badge: .unknown, reason: status?.snapshotUpload?.text) }
            if status?.uploadInProgress == true { return .init(badge: status?.uploadQueued == true ? .queued : .uploading, reason: status?.snapshotUpload?.text) }
            if status?.upload.tone == .neutral { return .init(badge: .idle, reason: status?.upload.text) }
        }
        return switch job.health {
        case .checking: .init(badge: .checking, reason: nil)
        case .healthy: .init(badge: .healthy, reason: nil)
        case .awaitingRun: .init(badge: .awaitingRun, reason: nil)
        // A positively verified calendar gate is neutral across all job groups.
        case .skipped: .init(badge: .skipped, reason: job.attentionReason)
        case .deferred: .init(badge: .deferred, reason: job.attentionReason)
        case .unknown, .notice: .init(badge: .unknown, reason: job.attentionReason)
        case .unhealthy: .init(badge: .attention, reason: job.attentionReason)
        }
    }
}
