import Foundation

/// Value-only adapter; deliberately has no production caller until menu PR 2b.
/// The caller samples the two private clocks alongside publishedStatus: awake age via
/// sleepHistory.elapsed(since: verdict.evaluatedAt, now: now), catch-up via
/// wakeCatchUp.isPending(uptime: awakeUptime). Never substitute wall time for awake age.
enum BrainBarMenuLiveCapture {
    static func capture(
        status: BrainBarStatusSnapshot,
        verdictAwakeAgeS: TimeInterval?,
        wakeCatchUpPending: Bool,
        driveAuth: DriveAuthStatus?,
        backups: BrainBarStatusInputs.Backups,
        backupsNotice: String?,
        watcher: WatcherHealthFileRead?,
        watcherAwakeAgeS: TimeInterval?,
        observations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation]?,
        maintenanceNightStreakKnown: Bool,
        stats: DashboardStats? = nil,
        now: Date, timeZone: TimeZone
    ) -> BrainBarMenuCapture {
        var c = BrainBarMenuCapture(now: now, timeZone: timeZone)
        c.verdict = status.healthVerdict ?? status.staleHealthVerdict
        c.verdictAwakeAgeS = verdictAwakeAgeS ?? .infinity
        c.isChecking = status.isChecking
        c.wakeCatchUpPending = wakeCatchUpPending
        c.driveAuthModel = driveAuth?.state
        c.backupsMeasured = false
        if case let .measured(value) = backups {
            c.backupsMeasured = value.state == "measured" && backupsNotice == nil
            if let upload = value.localSnapshot, upload.uploadState == "in_progress",
               let start = upload.uploadProcessStartedAt, let live = upload.uploadProcessLive {
                c.upload = .init(startedAt: start, live: live)
            }
        }
        c.watcher = nil
        if case let .readable(file) = watcher {
            c.watcher = .init(readable: true,
                fresh: watcherAwakeAgeS.map { (0.0...WatcherHealthStatus.staleHeartbeatSeconds).contains($0) } ?? false,
                alertReasons: file.alertReasons, dbProbeFailed: file.dbProbeFailed)
        }
        c.backupDailyFailures = observations?[.backupDaily]?.calendarConsecutiveFailures
        c.backupDailyNextRunAt = observations?[.backupDaily]?.nextRunAt
        c.jsonlBackupFailures = observations?[.jsonlBackup]?.calendarConsecutiveFailures
        let night = [BrainLayerLaunchdJob.maintenanceNightly, .maintenanceWeekly]
            .compactMap { observations?[$0]?.calendarNightOutcome }.first
        c.maintenanceNightFailures = maintenanceNightStreakKnown && night?.streakComplete == true
            ? night?.consecutiveFailures : nil
        c.keywordCoverage = stats.flatMap { $0.signalCoverageIsAvailable ? $0.ftsCoveragePercent / 100.0 : nil }
        // Not carried yet: lastKnownBackupAt, offsiteRecovering, lagEpisodeStart.
        return c
    }
}
