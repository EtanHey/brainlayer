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
    var driveTone: DriveAuthPresentation.Tone? = nil
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
    var vectorCheck: BrainBarVectorCheck? = nil
    /// R2-3: how the live job-alert file read went.
    enum JobAlertsState: Equatable, Sendable {
        /// The file was read; `jobAlerts` is that map.
        case readable
        /// No file exists (no job has ever reported): genuine first-run absence.
        case absent
        /// Not read yet.
        case unread
        /// The file exists but could not be read or parsed; `jobAlerts` is the last verified map.
        case unavailable(String)
    }

    /// The last VERIFIED live job-alert map; nil when none has been read.
    var jobAlerts: BrainBarJobAlerts? = nil
    /// Nil derives it: `.readable` when `jobAlerts` is set, else `.unread`.
    var jobAlertsState: JobAlertsState? = nil
    var watcher: WatcherHealthStatus? = nil
    var watcherHeartbeatAt: Date? = nil
    /// The watcher's own structured report (DB probe, ingest alerts), when readable.
    var watcherFile: WatcherHealthFile? = nil
    var settings: [BrainLayerLaunchdJob: BrainLayerLaunchdJobSetting] = [:]
    /// Nil until the first launchd/calendar sample completes.
    var observations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation]? = nil
    var backups: Backups = .notRead
    /// The backups section is measured but its envelope is stale or future-dated.
    var backupsNotice: String? = nil
    var drive: DriveAuthPresentation? = nil
    var maintenance: BrainLayerMaintenanceEvidence = .unread
    /// How long `observability.json` has been continuously unreadable while the app sampled it.
    var observabilityUnreadableFor: TimeInterval? = nil
    /// The app sampler's period (L3 measures unreadability in these).
    var samplingPeriod: TimeInterval = 30
}

/// The one status verdict every BrainBar surface renders (S5, R2 §5.A): the menu-bar icon dot,
/// the menu text, the Dashboard headline and attention list, Jobs group badges and the Backups
/// badge. A pure reducer: the same inputs give the same verdict everywhere.
///
/// Etan's dot rule (2026-10-27): no verified Drive copy within 36 h, an unclean daemon/
/// watcher exit until six healthy hours, or two consecutive failed/abandoned maintenance runs.
/// Lead r1: an unprovable backup age for >60 min is at risk; a supposedly running watcher
/// without a heartbeat for >15 min is not capturing. Future receipts cannot prove backup safety.
/// Other faults remain visible notices on the same snapshot.
struct BrainBarStatusSnapshot: Equatable, Sendable {
    let issues: [BrainBarStatusIssue]
    let isChecking: Bool

    var attention: [BrainBarStatusIssue] { issues.filter { $0.severity == .attention } }
    var notices: [BrainBarStatusIssue] { issues.filter { $0.severity == .notice } }
    var needsAttention: Bool { !attention.isEmpty }

    static let checking = Self(issues: [], isChecking: true)

    /// Job alerts that report a recovery or a deferral, not a failure.
    /// Job alerts whose key itself reports a recovery or a deferral, never an active failure.
    /// Active incidents remain visible until their producer clears them.
    static let informationalAlertKeys: Set<String> = ["fleet-watchdog-hold", "brainbar-activation-brew"]
    /// Known advisory health-check codes.
    static let neutralBadgeCodes: Set<String> = ["queue_backed_up", "missing_embeddings_scan_incomplete"]
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
        if let notice = touching.first, status.health == .healthy || status.health == .skipped || status.health == .awaitingRun {
            return with(notice.noticeHealth, notice.text)
        }
        guard [.unhealthy, .deferred, .unknown].contains(status.health) else { return status }
        let health = status.health == .unhealthy ? touching.first?.noticeHealth ?? .notice : status.health
        return with(health, touching.first?.text ?? status.attentionReason)
    }

    /// The Backups badge renders the snapshot's receipt-age and maintenance verdict.
    func project(_ health: BrainBarBackupsHealth) -> BrainBarBackupsHealth {
        let backupJobs: Set<BrainLayerLaunchdJob> = [.backupDaily, .jsonlBackup]
        // Job alerts ride on the Backups page (observability carries them in its backups section).
        let relevant = issues.filter {
            [.database, .transcripts, .backups, .drive].contains($0.component) || !$0.jobs.isDisjoint(with: backupJobs)
                || $0.alertReason != nil
        }
        // B4: the badge and its primary reason are the snapshot's own typed issue.
        if let issue = relevant.first(where: { $0.severity == .attention }) { return .init(badge: .attention, reason: issue.text) }
        if let failure = relevant.first(where: {
            $0.severity == .notice && $0.noticeHealth == .notice
                && $0.driveTone != .expiring
                && ([.backups, .drive].contains($0.component) || !$0.jobs.isDisjoint(with: backupJobs)
                    || $0.alertKey == "unknown")
        }) {
            return .init(badge: .notice, reason: failure.text)
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
        switch inputs.watcher {
        case nil, .running?:
            break
        case let .degraded(_, heartbeatAt)?:
            let stale = now.timeIntervalSince(heartbeatAt) > WatcherHealthStatus.staleHeartbeatSeconds
            let text = inputs.watcher?.reasonText(now: now) ?? "Watcher needs attention"
            // Keep the watcher's structured faults visible independently of the kill evidence.
            let critical = inputs.watcherFile.map { $0.dbProbeFailed || $0.alertReasons.contains("coverage_drop") } ?? false
            builder.add(.init(id: stale ? "watcher.heartbeat" : critical ? "watcher.ingest" : "watcher.alerts", component: .watcher,
                              severity: .notice, text: text, jobs: [.watch]))
        case let .stopped(reason)?:
            builder.add(.init(id: "watcher.heartbeat", component: .watcher,
                              severity: .notice, text: reason, jobs: [.watch]))
        case let .unknown(reason)?:
            builder.add(.init(id: "watcher.unknown", component: .watcher, severity: .notice, text: reason, jobs: [.watch],
                              noticeHealth: .unknown))
        }

        if let problem = observations?[.watch]?.killCheckUnavailable {
            builder.add(.init(id: "process.kill-check", component: .watcher, severity: .notice,
                              text: "Kill check unavailable: " + problem, jobs: [.watch], noticeHealth: .unknown))
        }
        if observations?[.watch]?.loadState == .running, desired(.watch),
           let heartbeat = inputs.watcherHeartbeatAt, now.timeIntervalSince(heartbeat) > 15 * 60 {
            builder.add(.init(id: "watcher.capture", component: .watcher, severity: .attention,
                              text: "Watcher isn't capturing; last heartbeat at " + absoluteDate(heartbeat, calendar: inputs.calendar),
                              jobs: [.watch]))
        }

        if let incident = observations?[.watch]?.uncleanExit {
            let since = observations?[.watch]?.healthySince
            let recovered = since.map { now.timeIntervalSince($0) >= 6 * 3600 && $0 <= now } ?? false
            if !recovered {
                builder.add(.init(id: "process.unclean-exit", component: .watcher, severity: .attention,
                                  text: incident + "; clears after 6 h continuously healthy", jobs: [.watch]))
            }
        }

        // 2. Backups: verified copy ages against the threshold, and failures newer than every copy.
        var dbCopy: Date?
        var transcriptCopy: Date?
        switch inputs.backups {
        case .notRead:
            break
        case let .unavailable(reason):
            let atRisk = (inputs.observabilityUnreadableFor ?? 0) > 60 * 60
            builder.add(.init(id: "backups.unmeasured", component: .backups, severity: atRisk ? .attention : .notice,
                              text: atRisk ? "BrainLayer backup safety can't be proven: receipt age unavailable for over 60 minutes; " + reason : reason,
                              jobs: [.backupDaily, .jsonlBackup], noticeHealth: .unknown))
        case let .measured(backups):
            let threshold: TimeInterval = 36 * 3600
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
                    builder.add(.init(id: "\(id).copy", component: component, severity: .attention,
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
                let text = ObservabilityPresentation.backupStatus(for: backups, now: now, calendar: inputs.calendar).error?.text ?? "Backup failed: \(error)"
                if error == "drive_credentials_restored_backup_pending" {
                    builder.add(.init(id: "backups.error", component: .backups, severity: .notice, text: text))
                } else if !error.hasPrefix("db_snapshot_") || error.hasPrefix("db_snapshot_unverified") {
                    // The producer reports the newest attempt's failure: nothing verified is newer.
                    builder.add(.init(id: "backups.error", component: .backups, severity: .notice, text: text))
                }
            }
        }

        // 3. Drive authorization.
        switch inputs.drive?.tone {
        case .attention?:
            builder.add(.init(id: "drive.auth", component: .drive, severity: .notice, text: inputs.drive!.line))
        case .expiring?:
            builder.add(.init(id: "drive.auth", component: .drive, severity: .notice, text: inputs.drive!.line,
                              driveTone: .expiring))
        case .unknown?:
            builder.add(.init(id: "drive.auth", component: .drive, severity: .notice, text: inputs.drive!.line,
                              noticeHealth: .unknown))
        case .connected?, nil:
            break
        }

        // 4. Configuration the calendar needs.
        if let problem = observations?.values.compactMap(\.calendarScheduleProblem).sorted().first {
            builder.add(.init(id: "config.maintenance-plist", component: .configuration, severity: .notice,
                              text: "Maintenance calendar configuration is broken: \(problem)",
                              jobs: Set(requiredCalendarJobs)))
        }
        if let uncertainty = requiredCalendarJobs.lazy.compactMap({ observations?[$0]?.calendarScheduleUncertainty }).first {
            builder.add(.init(id: "jobs.schedule", component: .jobs, severity: .notice,
                              text: "\(uncertainty); last results are still shown", noticeHealth: .unknown))
        }

        // 5. Calendar job results. A future-dated copy is clock skew, never a recovery.
        // S11: a newer verified copy resolves the failure; a historical streak never overrides it.
        func trusted(_ copy: Date?) -> Date? { copy.flatMap { $0.timeIntervalSince(now) <= 300 ? $0 : nil } }
        for (job, copy) in [(BrainLayerLaunchdJob.backupDaily, trusted(dbCopy)), (.jsonlBackup, trusted(transcriptCopy))]
        where desired(job) {
            guard let observation = observations?[job], observation.isFailedRun else { continue }
            let newer = copy.flatMap { copy in observation.lastRunAt.map { copy > $0 } } ?? false
            let streak = observation.calendarConsecutiveFailures ?? 1
            let base = observation.failureText(job: job, formatDate: formatDate)
            let text = newer ? "\(base); recovered through a newer verified copy"
                : streak >= 2 ? "\(base) (failed on \(streak) consecutive scheduled runs)" : base
            builder.add(.init(id: "job.\(job.rawValue)", component: .backups, severity: .notice,
                              text: text, jobs: [job]))
        }
        // Index failures stay visible; maintenance owns the failure-streak dot.
        if desired(.index), let observation = observations?[.index] {
            if observation.isFailedRun {
                let streak = observation.calendarConsecutiveFailures ?? 1
                let base = observation.failureText(job: .index, formatDate: formatDate)
                builder.add(streak >= 2
                    ? .init(id: "job.index", component: .ingest, severity: .notice,
                            text: "\(base) (failed on \(streak) consecutive scheduled runs)", jobs: [.index])
                    : .init(id: "job.index", component: .ingest, severity: .notice,
                            text: "\(base); it retries at its next scheduled run", jobs: [.index]))
            } else if let earned = observation.earnedStreakThroughDeferral(threshold: 2) {
                builder.add(.init(id: "job.index", component: .ingest, severity: .notice,
                                  text: "Index failed on \(earned) consecutive scheduled runs; the latest run was deferred (exit 75)",
                                  jobs: [.index]))
            }
        }
        // One outcome per scheduled night, independent of which maintenance plan it selected.
        let maintenanceJobs: [BrainLayerLaunchdJob] = [.maintenanceNightly, .maintenanceWeekly]
        let night = maintenanceJobs.compactMap { observations?[$0]?.calendarNightOutcome }.first
        if let night, night.failed, maintenanceJobs.contains(where: desired) {
            let next = maintenanceJobs.compactMap { observations?[$0]?.nextRunAt }.filter { $0 > now }.min()
            let retry = next.map { "next retry " + absoluteDate($0, calendar: inputs.calendar) } ?? "next retry time unavailable"
            let labels = ["jsonl-backup": "Transcript backup", "backup-daily": "Database backup",
                "backup-snapshot": "Database snapshot", "index": "Transcript indexing",
                "maintenance-nightly": "Nightly maintenance", "maintenance-weekly": "Weekly maintenance",
                "wal-checkpoint": "Database checkpoint", "repair-fts": "Search index repair",
                "decay": "Memory decay", "p0-counter": "Memory counts"]
            var outcome: String
            func stepLabel(_ step: String) -> String { "\(labels[step] ?? step) (step \(step))" }
            func stepText(_ step: String, includeOutcome: Bool) -> String {
                let problem = night.stepProblems[step]
                let action = includeOutcome ? " failed" : ""
                let label = labels[step] ?? (includeOutcome ? "Nightly maintenance" : step)
                return "\(label)\(action) (step \(step)) at \(absoluteDate(problem?.at ?? night.at, calendar: inputs.calendar))"
            }
            if night.failedSteps.contains(where: { night.stepProblems[$0]?.cutOff == true }) {
                // These are detection times. Group equal displayed timestamps once, while
                // retaining each failed step's own finish time in mixed outcomes.
                var cutOffs: [String: [String]] = [:]
                var times: [String] = []
                var problems: [String] = []
                for step in night.failedSteps {
                    if let problem = night.stepProblems[step], problem.cutOff {
                        let time = absoluteDate(problem.at, calendar: inputs.calendar)
                        if cutOffs[time] == nil { times.append(time) }
                        cutOffs[time, default: []].append(stepLabel(step))
                    } else {
                        problems.append(stepText(step, includeOutcome: true))
                    }
                }
                outcome = (times.map { "\(cutOffs[$0, default: []].joined(separator: ", ")) found cut off at \($0)" } + problems)
                    .joined(separator: "; ")
            } else if night.failedSteps.count == 1, let step = night.failedSteps.first {
                outcome = stepText(step, includeOutcome: true)
            } else if !night.failedSteps.isEmpty {
                let steps = night.failedSteps.map { stepText($0, includeOutcome: false) }.joined(separator: ", ")
                outcome = "Nightly tasks failed: \(steps)"
            } else {
                // Each real cause once, at its own time: an abandonment that named no step, then
                // any runner error or independent unknown failure appended below.
                outcome = night.abandonmentFailureAt.map {
                    "Nightly maintenance was abandoned at \(absoluteDate($0, calendar: inputs.calendar))"
                } ?? ""
                if outcome.isEmpty, night.runnerErrorAt == nil, night.unknownFailureAt == nil {
                    outcome = "\(night.abandoned ? "Nightly maintenance was abandoned" : "Scheduled maintenance failed") at \(absoluteDate(night.at, calendar: inputs.calendar))"
                }
            }
            if let at = night.runnerErrorAt {
                outcome += (outcome.isEmpty ? "" : "; ") + "Maintenance runner failed at \(absoluteDate(at, calendar: inputs.calendar))"
            } else if let at = night.unknownFailureAt {
                outcome += (outcome.isEmpty ? "" : "; ") + "Scheduled maintenance reported an unknown failure at \(absoluteDate(at, calendar: inputs.calendar))"
            }
            let count = night.consecutiveFailures >= 2 ? "; nightly runs have had problems \(night.consecutiveFailures) nights in a row" : ""
            builder.add(.init(id: "maintenance.night", component: .maintenance,
                severity: night.consecutiveFailures >= 2 ? .attention : .notice,
                text: "\(outcome)\(count) — \(retry)",
                jobs: Set(maintenanceJobs)))
        } else {
            // Legacy per-step failures stay visible, but can never earn a nightly-run dot.
            for job in maintenanceJobs where desired(job) {
                guard let observation = observations?[job], observation.isFailedRun else { continue }
                builder.add(maintenanceIssue(job, observation: observation, now: now, calendar: inputs.calendar))
            }
        }
        // 6. Job alerts: the live file names every failing job, including keys the badge lacks.
        // R2-3: `jobAlerts` is the last VERIFIED map. An unavailable read keeps its unresolved
        // criticals (an unknown read is never a recovery); only a readable map can clear them.
        let alertsState = inputs.jobAlertsState ?? (inputs.jobAlerts == nil ? .unread : .readable)
        var alerts = inputs.jobAlerts?.active ?? [:]
        if case let .unavailable(reason) = alertsState {
            builder.add(.init(id: "jobs.alerts-unavailable", component: .health, severity: .notice,
                              text: "Job alerts could not be read (\(reason)); last known alerts are kept",
                              noticeHealth: .unknown))
        }
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
                // An alert alone does not establish two failed runs.
                builder.add(.init(id: job.map { "job.\($0.rawValue)" } ?? "alert.\(key)", component: .maintenance,
                                  severity: .notice, text: reason + "; failure time unavailable; "
                                      + (job.flatMap { observations?[$0]?.nextRunAt }
                                          .map { "next retry " + absoluteDate($0, calendar: inputs.calendar) }
                                          ?? "next retry time unavailable"), jobs: job.map { [$0] } ?? [],
                                  alertKey: key, alertReason: reason))
            case let .backup(job):
                // One failed run is one issue: the alert joins its job's run issue (Show log keeps its key).
                var text = reason
                // Only the generic alert takes the measured sentence; a distinct live diagnosis stays as is.
                if ["backup-daily", "backup-snapshot"].contains(key), reason == "Database backup failed; check the backup log",
                   case let .measured(backups) = inputs.backups,
                   backups.errorType?.hasPrefix("raw_reclaim_unconfirmed:") == true {
                    text = ObservabilityPresentation.backupStatus(for: backups, now: now, calendar: inputs.calendar).error?.text ?? reason
                    // Join the effective alert, including a badge fallback, once per attempt.
                    builder.issues.removeAll { $0.id == "backups.error" }
                }
                if let job, builder.attachAlert(key, reason: reason, toJob: job, text: text == reason ? nil : text) { continue }
                builder.add(.init(id: job.map { "job.\($0.rawValue)" } ?? "alert.\(key)", component: .backups,
                                  severity: .notice, text: text, jobs: job.map { [$0] } ?? [],
                                  alertKey: key, alertReason: reason))
            case .drive:
                // Drive authorization remains visible independently of backup receipt age.
                builder.add(.init(id: "drive.auth", component: .drive, severity: .notice,
                                  text: reason, alertKey: key, alertReason: reason))
            case .informational:
                builder.add(.init(id: "alert.\(key)", component: .health, severity: .notice, text: reason, alertKey: key,
                                  alertReason: reason))
            case .unknown:
                // A job reported a failure nobody has classified: never hide it behind a guess.
                builder.add(.init(id: "alert.\(key)", component: .jobs, severity: .notice, text: reason, alertKey: key,
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
                for (index, code) in badge.activeCodes.enumerated() where !code.hasPrefix("job_alert_") {
                    let message = index < badge.activeMessages.count ? badge.activeMessages[index] : badge.reason
                    let severity = index < badge.activeSeverities.count ? badge.activeSeverities[index] : "critical"
                    if code == "watcher_stalled" {
                        // Deduped into the watcher issue; only a running watcher with a fresh heartbeat
                        // is local evidence of recovery.
                        if inputs.watcher?.isRunning == true { continue }
                        builder.add(.init(id: "watcher.heartbeat", component: .watcher,
                                          severity: .notice, text: message, jobs: [.watch]))
                    } else if code == "missing_embeddings_scan_incomplete" {
                        if inputs.badge?.vectorScan?.routinePartial == true, let check = inputs.vectorCheck {
                            if check.isAwaitingFirstResult || check.isFresh(now: now) { continue }
                            builder.add(.init(id: "health.vector-coverage", component: .health, severity: .notice,
                                              text: BrainBarVectorCheck.unverified))
                            continue
                        }
                        let scanReason = inputs.badge?.vectorScan?.reason
                        let detail: String
                        switch scanReason {
                        case "sql_error": detail = "The database query failed."
                        case "unexpected": detail = "The background scan stopped unexpectedly."
                        case let reason? where !reason.isEmpty && !["instruction_limit", "time_limit", "page_limit", "window_end"].contains(reason):
                            detail = "The background scan stopped: " + reason.replacingOccurrences(of: "_", with: " ") + "."
                        default: detail = "The background scan could not finish."
                        }
                        builder.add(.init(id: "health.vector-coverage", component: .health, severity: .notice,
                            text: "The vector check could not finish. The current backlog is not verified by this check. " + detail))
                    } else if neutralBadgeCodes.contains(code) || severity == "warning" {
                        builder.add(.init(id: "health.\(code)", component: .health, severity: .notice, text: message))
                    } else if code == "job_scan_failed", message.contains("com.brainlayer.maintenance.plist") {
                        builder.add(.init(id: "config.maintenance-plist", component: .configuration, severity: .notice,
                                          text: message, jobs: Set(requiredCalendarJobs)))
                    } else {
                        // Retain all other criticals as notices under the three-condition dot rule.
                        builder.add(.init(id: "health.\(code)", component: code == "job_scan_failed" || code == "retired_service_config"
                                              ? .configuration : .health, severity: .notice, text: message))
                    }
                }
            }
        }

        if let check = inputs.vectorCheck, !check.isAwaitingFirstResult, !check.isFresh(now: now) {
            builder.add(.init(id: "health.vector-coverage", component: .health, severity: .notice,
                              text: BrainBarVectorCheck.unverified))
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
                case .skipped:
                    // A skipped run is not green: the Dashboard and the menu say why, as the row does.
                    // A group configured OFF carries no reason; its row says so in its own note.
                    guard let reason = raw.attentionReason else { break }
                    builder.add(.init(id: "group.\(group.rawValue)", component: .jobs, severity: .notice,
                                      text: reason, jobs: jobs, noticeHealth: .skipped))
                case .checking, .healthy, .notice:
                    break
                }
            }
        }

        // B5: every required source must have been read before anything can be "All good".
        let backupsUnread = if case .notRead = inputs.backups { true } else { false }
        let alertsIncomplete = switch alertsState {
        case .readable, .absent: false
        case .unread, .unavailable: true
        }
        let incomplete = inputs.badge == nil || inputs.watcher == nil || observations == nil || backupsUnread || alertsIncomplete
        // Backup job failures lead the notices even when other sources are unavailable (B2).
        let backupJobs: Set<BrainLayerLaunchdJob> = [.backupDaily, .jsonlBackup]
        let backupNotices = builder.issues.filter { $0.severity == .notice && $0.id.hasPrefix("job.") && !$0.jobs.isDisjoint(with: backupJobs) }
        let otherIssues = builder.issues.filter { !backupNotices.contains($0) }
        return Self(issues: backupNotices + otherIssues, isChecking: incomplete)
    }

    /// One failure is explicit, but only the second consecutive failure gives a dot.
    static func maintenanceIssue(_ job: BrainLayerLaunchdJob, observation: BrainLayerLaunchdJobObservation,
                                 now: Date, calendar: Calendar) -> BrainBarStatusIssue {
        let abandoned = observation.calendarRunStatus == "abandoned"
        let label = abandoned ? "Scheduled" : job == .maintenanceWeekly ? "Weekly" : "Nightly"
        let when = observation.lastRunAt.map { absoluteDate($0, calendar: calendar) } ?? "time unavailable"
        let outcome = abandoned ? "abandoned" : "failed"
        let retry = observation.nextRunAt.flatMap { $0 > now ? $0 : nil }
            .map { "next retry " + absoluteDate($0, calendar: calendar) } ?? "next retry time unavailable"
        return .init(id: abandoned ? "maintenance.abandoned-run" : "job.\(job.rawValue)", component: .maintenance,
                     severity: .notice, text: "\(label) maintenance \(outcome) at \(when) — \(retry)", jobs: [job])
    }

    static func absoluteDate(_ date: Date, calendar: Calendar) -> String {
        let formatter = DateFormatter()
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm z"
        return formatter.string(from: date)
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
        case backup(BrainLayerLaunchdJob?)
        case drive, informational, unknown
    }

    static func alertClass(key: String) -> AlertClass {
        switch key {
        case "maintenance-full": return .maintenance(.maintenanceWeekly)
        case "maintenance-light": return .maintenance(.maintenanceNightly)
        case "drive-consent": return .drive
        case let key where informationalAlertKeys.contains(key): return .informational
        case let key where key.hasPrefix("maintenance-"): return .maintenance(nil)
        case "backup-daily", "backup-upload-abandoned": return .backup(.backupDaily)
        case "jsonl-backup": return .backup(.jsonlBackup)
        case let key where key.hasPrefix("backup") || key.hasPrefix("jsonl"): return .backup(nil)
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
        mutating func attachAlert(_ key: String, reason: String, toJob job: BrainLayerLaunchdJob, text: String? = nil) -> Bool {
            guard let index = issues.firstIndex(where: { $0.id == "job.\(job.rawValue)"
                || (["maintenance.abandoned-run", "maintenance.night"].contains($0.id) && $0.jobs.contains(job)) }) else { return false }
            issues[index].alertKey = key
            issues[index].alertReason = reason
            if let text { issues[index].text = text }
            return true
        }
    }
}

extension BrainLayerLaunchdJobObservation {
    /// R2-2: the failure streak still outstanding behind a latest exit-75 deferral, when it has
    /// reached `threshold`. Only a proven success (the history resets the streak) or a successful
    /// weekly that covered this nightly resolves it; a deferral neither extends nor clears it.
    func earnedStreakThroughDeferral(threshold: Int) -> Int? {
        guard calendarCoveredByWeeklyAt == nil, loadState == .loaded,
              lastExitCode == BrainLayerMaintenanceExit.deferred,
              let streak = calendarConsecutiveFailures, streak >= threshold else { return nil }
        return streak
    }

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
