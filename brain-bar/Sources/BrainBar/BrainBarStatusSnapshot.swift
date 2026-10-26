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
    var severity: Severity
    var text: String
    var jobs: Set<BrainLayerLaunchdJob> = []
    /// How a job group shows this issue when it is only a notice.
    var noticeHealth: BrainLayerLaunchdGroupHealth = .notice
    /// The job-alert key behind the issue, when a job alert reported it.
    var alertKey: String? = nil
    /// That job alert's reason exactly as the job wrote it.
    var alertReason: String? = nil
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
    /// The backups section is measured but its envelope is stale or future-dated.
    var backupsNotice: String? = nil
    var drive: DriveAuthPresentation? = nil
    var maintenance: BrainLayerMaintenanceEvidence = .unread
}

/// The one status verdict every BrainBar surface renders (S5, R2 §5.A): the menu-bar icon dot,
/// the menu text, the Dashboard headline and attention list, Jobs group badges and the Backups
/// badge. A pure reducer: the same inputs give the same verdict everywhere.
///
/// Dot policy (lead ruling, 2026-10-26), and nothing else lights the dot:
/// - watcher heartbeat stale > 300 s while the watcher should be running;
/// - the verified DB or transcript copy older than the backups threshold (36 h);
/// - a backup that FAILED with no verified copy newer than the failure;
/// - a required job failing on two consecutive scheduled runs;
/// - malformed or unreadable required configuration (e.g. a broken calendar LaunchAgent);
/// - Drive authorization lost.
/// A single weekly-maintenance exit 76 with a newer verified DB copy and a pending next run is a
/// neutral "deferred" notice; it returns to a dot on a second consecutive failure, or when the last
/// completed weekly pass is more than 8 days old. Unknown/checking is neutral text, never a dot.
struct BrainBarStatusSnapshot: Equatable, Sendable {
    let issues: [BrainBarStatusIssue]
    let isChecking: Bool

    var attention: [BrainBarStatusIssue] { issues.filter { $0.severity == .attention } }
    var notices: [BrainBarStatusIssue] { issues.filter { $0.severity == .notice } }
    var needsAttention: Bool { !attention.isEmpty }

    static let checking = Self(issues: [], isChecking: true)

    /// Health-check codes that ARE the dot policy's "malformed or unreadable required configuration".
    /// Every other health-check code is visible neutral text (lead ruling: the dot list is exhaustive).
    static let configurationBadgeCodes: Set<String> = ["job_scan_failed", "retired_service_config"]
    /// Job alerts that report a recovery or a deferral, not a failure.
    static let informationalAlertKeys: Set<String> = ["fleet-watchdog-hold", "mcp-transport", "brainbar-activation-brew"]
    static let requiredCalendarJobs: [BrainLayerLaunchdJob] = [.index, .backupDaily, .jsonlBackup, .maintenanceNightly, .maintenanceWeekly]

    var verdict: BrainBarStatusVerdict {
        .init(attention: attention.map(\.text), notices: notices.map(\.text), isChecking: isChecking)
    }

    var dashboardStatus: BrainBarOnePageStatus { verdict.dashboardStatus }

    // MARK: projections

    /// A Jobs group as the snapshot sees it: Needs attention only for a dot issue on the group's
    /// jobs; anything else the raw status flagged is shown as neutral text.
    func project(_ status: BrainLayerLaunchdGroupStatus, group: BrainLayerLaunchdJobGroup) -> BrainLayerLaunchdGroupStatus {
        guard status.health != .checking else { return status }
        let touching = issues.filter { !$0.jobs.isDisjoint(with: group.jobs) }
        func with(_ health: BrainLayerLaunchdGroupHealth, _ reason: String?) -> BrainLayerLaunchdGroupStatus {
            .init(health: health, attentionReason: reason, lastRunText: status.lastRunText,
                  nextRunText: status.nextRunText, note: status.note, noteTone: status.noteTone)
        }
        if let issue = touching.first(where: { $0.severity == .attention }) { return with(.unhealthy, issue.text) }
        // Only a raw "Needs attention" is re-labelled; Unknown and Deferred already are neutral.
        guard [.unhealthy, .deferred, .unknown].contains(status.health) else { return status }
        let health = status.health == .unhealthy ? touching.first?.noticeHealth ?? .notice : status.health
        return with(health, touching.first?.text ?? status.attentionReason)
    }

    /// The Backups badge as the snapshot sees it: Needs attention only for a backup or Drive dot.
    func project(_ health: BrainBarBackupsHealth) -> BrainBarBackupsHealth {
        let backupJobs: Set<BrainLayerLaunchdJob> = [.backupDaily, .jsonlBackup]
        // Job alerts ride on the Backups page (observability carries them in its backups section).
        let relevant = issues.filter {
            [.database, .transcripts, .backups, .drive].contains($0.component) || !$0.jobs.isDisjoint(with: backupJobs)
                || $0.alertReason != nil
        }
        if let issue = relevant.first(where: { $0.severity == .attention }) {
            return .init(badge: .attention, reason: health.badge == .attention ? health.reason ?? issue.text : issue.text)
        }
        guard health.badge == .attention else { return health }
        guard let issue = relevant.first else { return .init(badge: .unknown, reason: health.reason) }
        let badge: BrainBarBackupsHealth.Badge = switch issue.noticeHealth {
        case .deferred: .deferred
        case .unknown, .checking: .unknown
        default: .notice
        }
        return .init(badge: badge, reason: issue.text)
    }

    /// Whether the snapshot rates this job alert as neutral text (e.g. a deferred weekly VACUUM):
    /// then the Backups page must not carry it as a red backup alert.
    func isNoticeAlert(reason: String) -> Bool {
        issues.first { $0.alertReason == reason }?.severity == .notice
    }

    // MARK: reducer

    static func derive(_ inputs: BrainBarStatusInputs) -> Self {
        var builder = Builder()
        let now = inputs.now
        let observations = inputs.observations
        func desired(_ job: BrainLayerLaunchdJob) -> Bool { inputs.settings[job]?.enabled != false }
        let formatDate = DashboardMetricFormatter.jobDateTimeString

        // 1. Watcher: the same WatcherHealthStatus the Dashboard, footer and Ingest render.
        let paused = observations?[.watch]?.loadState == .paused
        let watcherShouldRun = desired(.watch) && !paused
        switch inputs.watcher {
        case nil, .running?:
            break
        case let .degraded(_, heartbeatAt)?:
            let stale = now.timeIntervalSince(heartbeatAt) > WatcherHealthStatus.staleHeartbeatSeconds
            let text = inputs.watcher?.reasonText(now: now) ?? "Watcher needs attention"
            builder.add(.init(id: stale ? "watcher.heartbeat" : "watcher.alerts", component: .watcher,
                              severity: stale && watcherShouldRun ? .attention : .notice, text: text, jobs: [.watch]))
        case let .stopped(reason)?:
            let stale = inputs.watcherHeartbeatAt.map { now.timeIntervalSince($0) > WatcherHealthStatus.staleHeartbeatSeconds } ?? true
            builder.add(.init(id: "watcher.heartbeat", component: .watcher,
                              severity: stale && watcherShouldRun ? .attention : .notice, text: reason, jobs: [.watch]))
        case let .unknown(reason)?:
            builder.add(.init(id: "watcher.unknown", component: .watcher, severity: .notice, text: reason, jobs: [.watch],
                              noticeHealth: .unknown))
        }

        // 2. Backups: verified copy ages against the threshold, and failures newer than every copy.
        var dbCopy: Date?
        var transcriptCopy: Date?
        switch inputs.backups {
        case .notRead:
            break
        case let .unavailable(reason):
            builder.add(.init(id: "backups.unmeasured", component: .backups, severity: .notice, text: reason,
                              noticeHealth: .unknown))
        case let .measured(backups):
            let threshold = (backups.thresholdHours.flatMap { $0 > 0 ? $0 : nil } ?? 36) * 3600
            dbCopy = backups.dbSnapshot.flatMap { $0.verified ? $0.lastAt : nil }
            transcriptCopy = [
                backups.lastVerifiedUpload.flatMap { $0.verified ? $0.at : nil },
                backups.transcriptAttempt.flatMap { $0.verified && $0.status == "no-op" ? $0.at : nil },
            ].compactMap { $0 }.max()
            for (id, label, copy, component, job) in [
                ("database", "Database", dbCopy, BrainBarStatusIssue.Component.database, BrainLayerLaunchdJob.backupDaily),
                ("transcripts", "Transcripts", transcriptCopy, .transcripts, .jsonlBackup),
            ] {
                guard let copy else {
                    // Backup history survives a supervisor migration: no verified copy at all is a
                    // copy older than any threshold, not a first run (the producer calls it stale too).
                    builder.add(.init(id: "\(id).copy", component: component, severity: .attention,
                                      text: "\(label): no verified copy on record", jobs: [job]))
                    continue
                }
                let age = now.timeIntervalSince(copy)
                if age < -300 {
                    builder.add(.init(id: "\(id).copy", component: component, severity: .notice,
                                      text: "\(label): newest verified copy is dated in the future", jobs: [job]))
                } else if age > threshold {
                    builder.add(.init(id: "\(id).copy", component: component, severity: .attention,
                                      text: "\(label): newest verified copy is \(Int(age / 3600)) h old (threshold \(Int(threshold / 3600)) h)",
                                      jobs: [job]))
                }
            }
            if let notice = inputs.backupsNotice {
                builder.add(.init(id: "backups.status", component: .backups, severity: .notice, text: notice,
                                  noticeHealth: .unknown))
            }
            if let error = backups.errorType, !error.hasPrefix(BrainBarJobAlerts.observabilityPrefix) {
                let text = ObservabilityPresentation.backupStatus(for: backups, now: now).error?.text ?? "Backup failed: \(error)"
                if error == "drive_credentials_restored_backup_pending" {
                    builder.add(.init(id: "backups.error", component: .backups, severity: .notice, text: text))
                } else if !error.hasPrefix("db_snapshot_") || error.hasPrefix("db_snapshot_unverified") {
                    // The producer reports the newest attempt's failure: nothing verified is newer.
                    builder.add(.init(id: "backups.error", component: .backups, severity: .attention, text: text))
                }
            }
        }

        // 3. Drive authorization.
        switch inputs.drive?.tone {
        case .attention?:
            builder.add(.init(id: "drive.auth", component: .drive, severity: .attention, text: inputs.drive!.line))
        case .expiring?:
            builder.add(.init(id: "drive.auth", component: .drive, severity: .notice, text: inputs.drive!.line))
        case .unknown?:
            builder.add(.init(id: "drive.auth", component: .drive, severity: .notice, text: inputs.drive!.line,
                              noticeHealth: .unknown))
        case .connected?, nil:
            break
        }

        // 4. Configuration the calendar needs.
        if let problem = observations?.values.compactMap(\.calendarScheduleProblem).sorted().first {
            builder.add(.init(id: "config.maintenance-plist", component: .configuration, severity: .attention,
                              text: "Maintenance calendar configuration is broken: \(problem)",
                              jobs: Set(requiredCalendarJobs)))
        }
        if let uncertainty = requiredCalendarJobs.lazy.compactMap({ observations?[$0]?.calendarScheduleUncertainty }).first {
            builder.add(.init(id: "jobs.schedule", component: .jobs, severity: .notice,
                              text: "\(uncertainty); last results are still shown", noticeHealth: .unknown))
        }

        // 5. Calendar job results.
        for (job, copy) in [(BrainLayerLaunchdJob.backupDaily, dbCopy), (.jsonlBackup, transcriptCopy)] where desired(job) {
            guard let observation = observations?[job], observation.isFailedRun else { continue }
            let newer = copy.flatMap { copy in observation.lastRunAt.map { copy > $0 } } ?? false
            let streak = observation.calendarConsecutiveFailures ?? 1
            let severity: BrainBarStatusIssue.Severity = !newer || streak >= 2 ? .attention : .notice
            let base = observation.failureText(job: job, formatDate: formatDate)
            let text = streak >= 2 ? "\(base) (failed on \(streak) consecutive scheduled runs)"
                : newer ? "\(base); a newer verified copy exists" : base
            builder.add(.init(id: "job.\(job.rawValue)", component: .backups, severity: severity, text: text, jobs: [job]))
        }
        let weeklyStale: Bool = switch inputs.maintenance.weeklyCompletion {
        case let .completed(at): now.timeIntervalSince(at) > BrainLayerMaintenanceExit.staleCompletion
        case .noneRecorded: true
        case .unread: false
        }
        for job in [BrainLayerLaunchdJob.index, .maintenanceNightly, .maintenanceWeekly] where desired(job) {
            guard let observation = observations?[job] else { continue }
            let component: BrainBarStatusIssue.Component = job == .index ? .ingest : .maintenance
            let when = observation.lastRunAt.map { " at \(formatDate($0))" } ?? ""
            if job == .maintenanceWeekly, observation.lastExitCode == BrainLayerMaintenanceExit.deferred,
               observation.loadState == .loaded, weeklyStale {
                builder.add(.init(id: "job.\(job.rawValue)", component: component, severity: .attention,
                                  text: "Weekly maintenance has not completed in over 8 days; last run deferred\(when)", jobs: [job]))
                continue
            }
            guard observation.isFailedRun else { continue }
            let streak = observation.calendarConsecutiveFailures ?? 1
            if job == .maintenanceWeekly, observation.lastExitCode == BrainLayerMaintenanceExit.vacuumSkipped {
                let newerCopy = dbCopy.flatMap { copy in observation.lastRunAt.map { copy > $0 } } ?? false
                let next = observation.nextRunAt.flatMap { $0 > now ? $0 : nil }
                if streak < 2, newerCopy, let next, !weeklyStale {
                    builder.add(.init(id: "job.\(job.rawValue)", component: component, severity: .notice,
                                      text: "Weekly maintenance deferred: retries \(retryPhrase(next, now: now, calendar: inputs.calendar))",
                                      jobs: [job], noticeHealth: .deferred))
                    continue
                }
                let why = streak >= 2 ? "failed on \(streak) consecutive scheduled runs"
                    : weeklyStale ? "has not completed in over 8 days"
                    : !newerCopy ? "skipped VACUUM and no verified database copy is newer"
                    : "skipped VACUUM and no retry is scheduled"
                builder.add(.init(id: "job.\(job.rawValue)", component: component, severity: .attention,
                                  text: "Weekly maintenance \(why) (last: exit 76\(when): local verified snapshot unavailable)",
                                  jobs: [job]))
                continue
            }
            let base = observation.failureText(job: job, formatDate: formatDate)
            builder.add(streak >= 2
                ? .init(id: "job.\(job.rawValue)", component: component, severity: .attention,
                        text: "\(base) (failed on \(streak) consecutive scheduled runs)", jobs: [job])
                : .init(id: "job.\(job.rawValue)", component: component, severity: .notice,
                        text: "\(base); it retries at its next scheduled run", jobs: [job]))
        }

        // 6. Job alerts: the live file names every failing job, including keys the badge lacks.
        var alerts = inputs.jobAlerts?.active ?? [:]
        if inputs.jobAlerts == nil, let badge = inputs.badge, badge.activeMessages.count == badge.activeCodes.count {
            // The live file is unreadable: the badge's own job-alert issues are the best evidence.
            for (code, message) in zip(badge.activeCodes, badge.activeMessages) where code.hasPrefix("job_alert_") {
                alerts[String(code.dropFirst("job_alert_".count))] = message
            }
        } else if inputs.jobAlerts == nil, let badge = inputs.badge, badge.badgeOn {
            for code in badge.activeCodes where code.hasPrefix("job_alert_") {
                alerts[String(code.dropFirst("job_alert_".count))] = badge.reason
            }
        }
        if inputs.jobAlerts == nil, alerts.isEmpty, case let .measured(backups) = inputs.backups,
           let error = backups.errorType, error.hasPrefix(BrainBarJobAlerts.observabilityPrefix) {
            let reason = String(error.dropFirst(BrainBarJobAlerts.observabilityPrefix.count))
            alerts[inferredAlertKey(reason: reason)] = reason
        }
        for (key, reason) in alerts.sorted(by: { $0.key < $1.key }) {
            switch alertClass(key: key) {
            case let .maintenance(job):
                if let job, builder.attachAlert(key, reason: reason, toJob: job) { continue }
                builder.add(.init(id: job.map { "job.\($0.rawValue)" } ?? "alert.\(key)", component: .maintenance,
                                  severity: .attention, text: reason, jobs: job.map { [$0] } ?? [], alertKey: key,
                                  alertReason: reason))
            case .backup:
                builder.add(.init(id: "alert.\(key)", component: .backups, severity: .attention, text: reason,
                                  jobs: [.backupDaily], alertKey: key, alertReason: reason))
            case .drive:
                let lost = reason.localizedCaseInsensitiveContains("expired") || reason.localizedCaseInsensitiveContains("revoked")
                builder.add(.init(id: "drive.auth", component: .drive, severity: lost ? .attention : .notice,
                                  text: reason, alertKey: key, alertReason: reason))
            case .informational:
                builder.add(.init(id: "alert.\(key)", component: .health, severity: .notice, text: reason, alertKey: key,
                                  alertReason: reason))
            case .unknown:
                // A job reported a failure nobody has classified: never hide it behind a guess.
                builder.add(.init(id: "alert.\(key)", component: .jobs, severity: .attention, text: reason, alertKey: key,
                                  alertReason: reason))
            }
        }

        // 7. The health check's badge. Job alerts and watcher stalls have their own rules above.
        if let badge = inputs.badge {
            if badge.activeCodes.isEmpty {
                if badge.badgeOn || !badge.reason.isEmpty {
                    builder.add(.init(id: "health.badge", component: .health, severity: .notice,
                                      text: "Health check: \(badge.reason)"))
                }
            } else {
                for (index, code) in badge.activeCodes.enumerated()
                where !code.hasPrefix("job_alert_") && code != "watcher_stalled" {
                    let message = index < badge.activeMessages.count ? badge.activeMessages[index] : badge.reason
                    if configurationBadgeCodes.contains(code) {
                        let maintenancePlist = message.contains("com.brainlayer.maintenance.plist")
                        builder.add(.init(id: maintenancePlist ? "config.maintenance-plist" : "config.\(code)",
                                          component: .configuration, severity: .attention, text: message,
                                          jobs: maintenancePlist ? Set(requiredCalendarJobs) : []))
                    } else {
                        builder.add(.init(id: "health.\(code)", component: .health, severity: .notice, text: message))
                    }
                }
            }
        }

        // 8. Anything else a Jobs group flags, as neutral text on that group.
        if let observations {
            for group in BrainLayerLaunchdJobGroup.allCases {
                let raw = group.status(settings: inputs.settings, observations: observations, formatDate: formatDate,
                                       watcher: inputs.watcher, maintenance: inputs.maintenance, now: now)
                let jobs = Set(group.jobs)
                guard !builder.issues.contains(where: { !$0.jobs.isDisjoint(with: jobs) }) else { continue }
                switch raw.health {
                case .unhealthy, .deferred, .unknown:
                    builder.add(.init(id: "group.\(group.rawValue)", component: .jobs, severity: .notice,
                                      text: raw.attentionReason ?? "\(group.title): \(raw.health.title)", jobs: jobs,
                                      noticeHealth: raw.health == .unhealthy ? .notice : raw.health))
                case .awaitingRun:
                    let next = group.jobs.compactMap { observations[$0]?.nextRunAt }.filter { $0 > now }.min()
                    builder.add(.init(id: "group.\(group.rawValue)", component: .jobs, severity: .notice,
                                      text: "\(group.title): awaiting first scheduled run"
                                          + (next.map { " (\(retryPhrase($0, now: now, calendar: inputs.calendar)))" } ?? ""),
                                      jobs: jobs, noticeHealth: .awaitingRun))
                case .checking, .healthy, .skipped, .notice:
                    break
                }
            }
        }

        let nothingRead = inputs.badge == nil && inputs.watcher == nil && observations == nil
        let backupsUnread = if case .notRead = inputs.backups { true } else { false }
        return Self(issues: builder.issues, isChecking: nothingRead && backupsUnread && builder.issues.isEmpty)
    }

    /// "tonight 03:15", "today 15:00", "tomorrow 15:00", else the date.
    static func retryPhrase(_ date: Date, now: Date, calendar: Calendar) -> String {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "HH:mm"
        let time = formatter.string(from: date)
        let hour = calendar.component(.hour, from: date)
        if calendar.isDate(date, inSameDayAs: now) { return (hour >= 18 ? "tonight " : "today ") + time }
        if let tomorrow = calendar.date(byAdding: .day, value: 1, to: now), calendar.isDate(date, inSameDayAs: tomorrow) {
            return (hour < 6 ? "tonight " : "tomorrow ") + time
        }
        formatter.dateFormat = "MMM d HH:mm"
        return "on " + formatter.string(from: date)
    }

    enum AlertClass: Equatable {
        case maintenance(BrainLayerLaunchdJob?)
        case backup, drive, informational, unknown
    }

    static func alertClass(key: String) -> AlertClass {
        switch key {
        case "maintenance-full": return .maintenance(.maintenanceWeekly)
        case "maintenance-light": return .maintenance(.maintenanceNightly)
        case "drive-consent": return .drive
        case let key where informationalAlertKeys.contains(key): return .informational
        case let key where key.hasPrefix("maintenance-"): return .maintenance(nil)
        case let key where key.hasPrefix("backup") || key.hasPrefix("jsonl"): return .backup
        default: return .unknown
        }
    }

    /// The job key behind an alert reason when only `observability.json`'s copy was readable.
    static func inferredAlertKey(reason: String) -> String {
        for mode in ["full", "light", "burn"] where reason.hasPrefix("BrainLayer \(mode) maintenance") {
            return "maintenance-\(mode)"
        }
        return "unknown"
    }

    private struct Builder {
        var issues: [BrainBarStatusIssue] = []

        /// The first report of an identity keeps its place; a later dot report escalates it.
        mutating func add(_ issue: BrainBarStatusIssue) {
            guard let index = issues.firstIndex(where: { $0.id == issue.id }) else { issues.append(issue); return }
            issues[index].jobs.formUnion(issue.jobs)
            if issues[index].alertKey == nil { issues[index].alertKey = issue.alertKey }
            if issues[index].alertReason == nil { issues[index].alertReason = issue.alertReason }
            if issue.severity == .attention, issues[index].severity == .notice {
                issues[index].severity = .attention
                issues[index].text = issue.text
            }
        }

        /// A job alert for a job the calendar already reported: the same issue, now with Show log.
        mutating func attachAlert(_ key: String, reason: String, toJob job: BrainLayerLaunchdJob) -> Bool {
            guard let index = issues.firstIndex(where: { $0.id == "job.\(job.rawValue)" }) else { return false }
            issues[index].alertKey = key
            issues[index].alertReason = reason
            return true
        }
    }
}

extension BrainLayerLaunchdJobObservation {
    /// The failed run in the job's own words: the calendar's reason, else its exit and time.
    func failureText(job: BrainLayerLaunchdJob, formatDate: (Date) -> String) -> String {
        if case let .failed(reason) = loadState { return reason }
        let when = lastRunAt.map { " at \(formatDate($0))" } ?? ""
        return "\(job.title) last run exited \(lastExitCode.map(String.init) ?? "nonzero")\(when)"
    }

    /// The run that last finished failed: a calendar failure, or a launchd job's nonzero exit
    /// other than the exit-75 deferral (whose own verdict lives on the Maintenance card).
    var isFailedRun: Bool {
        switch loadState {
        case .failed: return calendarCoveredByWeeklyAt == nil
        case .loaded: return calendarCoveredByWeeklyAt == nil
            && (lastExitCode.map { $0 != 0 && $0 != BrainLayerMaintenanceExit.deferred } ?? false)
        default: return false
        }
    }
}
