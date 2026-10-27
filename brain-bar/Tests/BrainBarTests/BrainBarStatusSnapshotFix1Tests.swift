import XCTest
@testable import BrainBar

/// Shared-source and recovery regressions, updated for Etan's 2026-10-27 dot rule.
@MainActor
final class BrainBarStatusSnapshotFix1Tests: XCTestCase {
    typealias T = BrainBarStatusSnapshotTests
    nonisolated static let day: TimeInterval = 86_400

    nonisolated static func weekly(_ code: Int32, consecutive: Int, next: Date? = T.nextRun) -> BrainLayerLaunchdJobObservation {
        T.failed(.maintenanceWeekly, code: code, at: ISO8601DateFormatter().date(from: "2026-10-26T02:00:16Z")!,
                 consecutive: consecutive, next: next)
    }

    nonisolated static func completed(daysAgo: Double) -> BrainLayerMaintenanceEvidence {
        .init(weeklyCompletion: .completed(T.now.addingTimeInterval(-daysAgo * day)), runRecords: [:])
    }

    // MARK: maintenance failure streaks

    func testTodaysTwoConsecutiveExit76FailuresGiveADot() {
        var inputs = T.healthy()
        // Oct 25 + Oct 26 exit 76, last successful weekly Oct 21 (5 days).
        inputs.observations?[.maintenanceWeekly] = Self.weekly(76, consecutive: 2)
        inputs.maintenance = Self.completed(daysAgo: 5)
        inputs.jobAlerts = BrainBarJobAlerts(active: ["maintenance-full":
            "BrainLayer full maintenance failed: tonight's local verified snapshot unavailable; VACUUM skipped."])
        let (snapshot, surfaces) = T.assertAgree(inputs, dot: true, "today")
        let notice = snapshot.attention.filter { $0.jobs.contains(.maintenanceWeekly) }
        XCTAssertEqual(notice.count, 1, "\(snapshot.notices)")
        XCTAssertEqual(notice.first?.text.hasPrefix("Weekly maintenance failed at 2026-10-26 04:00"), true, notice.first?.text ?? "")
        XCTAssertEqual(surfaces.jobs[.maintenance]?.health, .unhealthy)
    }

    func testMaintenanceDotsOnTheSecondConsecutiveFailure() {
        for (job, code) in [(BrainLayerLaunchdJob.maintenanceWeekly, Int32(76)), (.maintenanceWeekly, 1), (.maintenanceNightly, 1)] {
            var inputs = T.healthy()
            inputs.maintenance = Self.completed(daysAgo: 3)
            inputs.observations?[job] = T.failed(job, code: code, consecutive: 2)
            T.assertAgree(inputs, dot: true, "\(job) exit \(code) ×2")
            inputs.observations?[job] = T.failed(job, code: code, consecutive: 3)
            T.assertAgree(inputs, dot: true, "\(job) exit \(code) ×3")
        }
    }

    func testOldWeeklyCompletionAloneDoesNotGiveADot() {
        var inputs = T.healthy() // latest weekly run is exit 0
        inputs.maintenance = Self.completed(daysAgo: 14)
        T.assertAgree(inputs, dot: false, "exactly 14 days")
        inputs.maintenance = .init(weeklyCompletion: .completed(T.now.addingTimeInterval(-14 * Self.day - 1)), runRecords: [:])
        T.assertAgree(inputs, dot: false, "old completion, latest exit 0")
        for unknown in [BrainLayerMaintenanceEvidence.unread, .init(weeklyCompletion: .noneRecorded, runRecords: [:])] {
            inputs.maintenance = unknown
            T.assertAgree(inputs, dot: false, "unknown completion age is not a measured > 14-day age")
        }
    }

    func testMissingNextRunIsUncertaintyNotAMaintenanceEmergency() {
        var inputs = T.healthy()
        inputs.observations?[.maintenanceWeekly] = Self.weekly(76, consecutive: 1, next: nil)
        let (snapshot, _) = T.assertAgree(inputs, dot: false, "no next run")
        XCTAssertTrue(snapshot.notices.contains { $0.text.hasPrefix("Weekly maintenance failed at 2026-10-26 04:00 GMT+2 — next retry time unavailable") },
                      "\(snapshot.notices)")
    }

    func testMaintenanceAlertWithoutCalendarEvidenceIsNeutral() {
        var inputs = T.healthy()
        inputs.observations?[.maintenanceWeekly] = nil
        inputs.jobAlerts = BrainBarJobAlerts(active: ["maintenance-full": "BrainLayer full maintenance failed: VACUUM skipped."])
        T.assertAgree(inputs, dot: false, "alert, no observation")
        inputs.jobAlerts = nil
        inputs.badge = .init(badgeOn: true, reason: "BrainLayer light maintenance failed", activeCodes: ["job_alert_maintenance-light"],
                             activeMessages: ["BrainLayer light maintenance failed"], activeSeverities: ["critical"])
        T.assertAgree(inputs, dot: false, "badge maintenance alert, alert file unreadable")
    }

    // MARK: other criticals remain visible notices

    func testHealthCodesRemainVisibleWithoutADot() {
        for code in ["brain_search_canary_failed", "job_state_unknown", "drain_no_progress", "supervisor_control_unavailable",
                     "a_code_nobody_has_seen_yet"] {
            var inputs = T.healthy()
            inputs.badge = .init(badgeOn: true, reason: code, activeCodes: [code], activeMessages: ["\(code) message"],
                                 activeSeverities: ["critical"])
            T.assertAgree(inputs, dot: false, code)
        }
        for (code, severity) in [("queue_backed_up", "warning"), ("missing_embeddings_scan_incomplete", "warning"),
                                 ("queue_backed_up", "critical"), ("missing_embeddings_scan_incomplete", "critical"),
                                 ("missing_embeddings_backlog", "warning")] {
            var inputs = T.healthy()
            inputs.badge = .init(badgeOn: severity == "critical", reason: code, activeCodes: [code],
                                 activeMessages: ["\(code) message"], activeSeverities: [severity])
            T.assertAgree(inputs, dot: false, "\(code) \(severity)")
        }
    }

    func testWatcherStalledCriticalSurvivesUnlessLocalEvidenceShowsRecovery() {
        var inputs = T.healthy()
        inputs.badge = .init(badgeOn: true, reason: "watcher stalled", activeCodes: ["watcher_stalled"],
                             activeMessages: ["watcher stalled"], activeSeverities: ["critical"])
        T.assertAgree(inputs, dot: false, "fresh local heartbeat proves recovery")
        inputs.watcher = .unknown(reason: "Watcher is running, but its health file is missing.")
        inputs.watcherHeartbeatAt = nil
        let (snapshot, _) = T.assertAgree(inputs, dot: false, "no local recovery evidence")
        XCTAssertEqual(snapshot.notices.filter { $0.id == "watcher.heartbeat" }.count, 1, "\(snapshot.attention)")
    }

    func testFreshHeartbeatDoesNotHideADatabaseProbeFailureOrAnIngestStall() {
        let now = T.now
        for (file, dot, name) in [
            (WatcherHealthFile(updatedAt: now.addingTimeInterval(-30), pollCount: 72, dbProbeFailed: true), false, "db probe"),
            (WatcherHealthFile(updatedAt: now.addingTimeInterval(-30), pollCount: 72, alertReasons: ["coverage_drop"]), false, "ingest stall"),
            (WatcherHealthFile(updatedAt: now.addingTimeInterval(-30), pollCount: 72, alertReasons: ["offset_lag"],
                               maxOffsetLagBytes: 4_000_000), false, "lag"),
            (WatcherHealthFile(updatedAt: now.addingTimeInterval(-30), pollCount: 72, alertReasons: ["file_ingestion_failure"],
                               fileIngestionFailureCount: 2), false, "two failed files"),
        ] {
            var inputs = T.healthy()
            inputs.watcher = WatcherHealthStatus.derive(launchd: .running, file: .readable(file), now: now)
            inputs.watcherHeartbeatAt = file.updatedAt
            inputs.watcherFile = file
            T.assertAgree(inputs, dot: dot, name)
        }
    }

    func testTransportAlertAndRecoveryRemainVisibleWithoutADot() {
        var inputs = T.healthy()
        inputs.jobAlerts = BrainBarJobAlerts(active: ["mcp-transport": "BrainBar MCP transport: event buffer overflow; storm counts unmeasurable"])
        T.assertAgree(inputs, dot: false, "active MCP transport incident")
        inputs.jobAlerts = BrainBarJobAlerts(active: ["fleet-watchdog-hold": "BrainLayer recovered a stale fleet-watchdog hold; supervisor restored"])
        T.assertAgree(inputs, dot: false, "recovered hold")
    }

    func testDriveConsentAlertIsTypedByTheAuthResultNotItsWording() {
        var inputs = T.healthy()
        inputs.jobAlerts = BrainBarJobAlerts(active: ["drive-consent": "Drive access expires at 2026-10-28T00:00:00Z: Reconnect in BrainBar"])
        inputs.drive = .init(line: "1 thing needs you: reconnect Drive by Oct 28", detail: nil, tone: .expiring,
                             showsReconnect: true, reconnectEnabled: true, needsAttention: true)
        T.assertAgree(inputs, dot: false, "typed expiring")
        inputs.jobAlerts = BrainBarJobAlerts(active: ["drive-consent": "Drive consent is gone"])
        inputs.drive = .init(line: "Drive access expired or was revoked. Backups can't upload.", detail: nil, tone: .attention,
                             showsReconnect: true, reconnectEnabled: true, needsAttention: true)
        T.assertAgree(inputs, dot: false, "typed lost, whatever the wording")
    }

    // MARK: future receipts are not recovery

    func testAFutureDatedCopyNeverDemotesARealBackupFailure() {
        var inputs = T.healthy()
        inputs.observations?[.backupDaily] = T.failed(.backupDaily, code: 1, at: T.hoursAgo(4))
        inputs.backups = .measured(T.backups(db: T.now.addingTimeInterval(3_600)))
        T.assertAgree(inputs, dot: true, "future DB copy is not a usable verified receipt")
    }

    // MARK: an unreadable status file

    func testUnreadableStatusFileRemainsANotice() {
        for (seconds, dot) in [(59.0, false), (60.0, false), (61.0, false)] {
            var inputs = T.healthy()
            inputs.backups = .unavailable("observability.json is unreadable")
            inputs.observabilityUnreadableFor = seconds
            inputs.samplingPeriod = 30
            let (snapshot, _) = T.assertAgree(inputs, dot: dot, "\(seconds) s unreadable")
            if dot { XCTAssertTrue(snapshot.attention.contains { $0.text.contains("status file") }, "\(snapshot.attention)") }
        }
        var unmeasured = T.healthy()
        unmeasured.backups = .unavailable("Backup status is unmeasurable — no LaunchAgents")
        unmeasured.observabilityUnreadableFor = nil
        T.assertAgree(unmeasured, dot: false, "readable but unmeasured is a separate neutral state")
    }

    // MARK: B5 — partial input is never All good

    func testPartialInputIsCheckingInEveryCompletionOrder() {
        let full = T.healthy()
        var partials: [BrainBarStatusInputs] = []
        var noBadge = full; noBadge.badge = nil; partials.append(noBadge)
        var noJobs = full; noJobs.observations = nil; partials.append(noJobs)
        var noBackups = full; noBackups.backups = .notRead; partials.append(noBackups)
        var noWatcher = full; noWatcher.watcher = nil; noWatcher.watcherHeartbeatAt = nil; partials.append(noWatcher)
        partials.append(BrainBarStatusInputs(now: T.now, calendar: T.calendar, watcher: .running(heartbeatAt: T.now)))
        for (index, inputs) in partials.enumerated() {
            let snapshot = BrainBarStatusSnapshot.derive(inputs)
            XCTAssertTrue(snapshot.isChecking, "partial \(index)")
            XCTAssertNotEqual(snapshot.dashboardStatus.headline, "All good", "partial \(index)")
            XCTAssertNotEqual(BrainBarStatusPopoverController.statusLineTitle(for: snapshot.verdict), "Nothing needs attention",
                              "partial \(index)")
        }
    }

    // MARK: R2-3 — an unknown alert read keeps the last known critical

    func testUnreadAlertSourceIsNeverAllGood() {
        var inputs = T.healthy()
        inputs.jobAlerts = nil
        let snapshot = BrainBarStatusSnapshot.derive(inputs)
        XCTAssertTrue(snapshot.isChecking)
        XCTAssertNotEqual(snapshot.dashboardStatus.headline, "All good")
    }

    func testUnavailableAlertReadKeepsTheLastVerifiedCritical() {
        var inputs = T.healthy()
        inputs.jobAlerts = BrainBarJobAlerts(active: ["mcp-transport": "BrainBar MCP transport: event buffer overflow"])
        inputs.jobAlertsState = .unavailable("job-alerts.json is unreadable or malformed")
        let snapshot = BrainBarStatusSnapshot.derive(inputs)
        XCTAssertFalse(snapshot.needsAttention, "\(snapshot.issues)")
        XCTAssertTrue(snapshot.isChecking, "incomplete evidence")
        XCTAssertTrue(snapshot.notices.contains { $0.text.contains("job-alerts.json") }, "\(snapshot.notices)")
    }

    // MARK: Macroscope Medium — one failed run is one issue

    func testBackupRunAndItsAlertAreOneIssueWithShowLog() {
        var inputs = T.healthy()
        inputs.observations?[.backupDaily] = T.failed(.backupDaily, code: 1, at: T.hoursAgo(1))
        inputs.backups = .measured(T.backups(db: T.hoursAgo(6), transcripts: T.hoursAgo(12)))
        inputs.jobAlerts = BrainBarJobAlerts(active: ["backup-daily": "BrainLayer daily backup failed"])
        let snapshot = BrainBarStatusSnapshot.derive(inputs)
        XCTAssertEqual(snapshot.notices.count, 1, "\(snapshot.attention)")
        XCTAssertEqual(snapshot.notices.first?.alertKey, "backup-daily")
        XCTAssertEqual(snapshot.dashboardStatus.headline, "Status notice")
        inputs.observations?[.jsonlBackup] = T.failed(.jsonlBackup, code: 1, at: T.hoursAgo(1))
        inputs.jobAlerts = BrainBarJobAlerts(active: ["backup-daily": "BrainLayer daily backup failed",
                                                      "jsonl-backup": "Transcript backup failed"])
        let both = BrainBarStatusSnapshot.derive(inputs)
        XCTAssertEqual(both.notices.count, 2, "\(both.attention)")
        XCTAssertEqual(Set(both.notices.compactMap(\.alertKey)), ["backup-daily", "jsonl-backup"])
    }

    // MARK: model: one publication, one cadence, mixed Backups reason

    private final class Clock: @unchecked Sendable {
        var now: Date
        init(_ now: Date) { self.now = now }
    }

    private func model(clock: Clock, observability: @escaping @Sendable () -> ObservabilityReadResult,
                       badgeURL: URL? = nil, alertsURL: URL? = nil, watcherReads: FixCounter = FixCounter(),
                       launchdReads: FixCounter = FixCounter(),
                       observations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation] = T.healthy().observations!) -> BrainBarSettingsViewModel {
        var config = BrainLayerConfig.defaultConfig
        for job in BrainLayerLaunchdJob.allCases { config.launchdJobs[job] = .init(enabled: true, loadState: .loaded) }
        let heartbeat = clock
        return BrainBarSettingsViewModel(
            workspace: BrainBarRecordingWorkspace(),
            store: BrainLayerConfigStore(configURL: URL(fileURLWithPath: "/fixture/config"),
                                         loadDocumentOverride: { BrainLayerEnvDocument(config: config) }),
            launchdStatusProvider: CountingActivity(observations: observations, reads: launchdReads),
            embeddingResidencyProbe: StaticBrainBarEmbeddingResidencyProbe(state: .unmeasurable),
            initialLaunchdObservations: observations, refreshStatusOnLoad: false, now: { clock.now },
            observabilityURL: URL(fileURLWithPath: "/fixture/observability.json"),
            observabilityRead: { _ in observability() },
            watcherHealthURL: URL(fileURLWithPath: "/fixture/watcher-health.json"),
            watcherHealthRead: { _ in
                watcherReads.value += 1
                return .readable(.init(updatedAt: heartbeat.now.addingTimeInterval(-20), pollCount: 50))
            },
            initialMaintenanceEvidence: Self.completed(daysAgo: 2),
            healthBadgeURL: badgeURL, jobAlertsURL: alertsURL, statusSamplingPeriod: 30)
    }

    private func document(_ backups: ObservabilityDocument.Backups, at date: Date) throws -> ObservabilityReadResult {
        guard case let .readable(base) = BrainBarDashboardFixture.healthyObservabilityResult else {
            throw XCTSkip("fixture must be readable")
        }
        return .readable(base.replacingBackupsForStatusTests(backups, generatedAt: date))
    }

    func testUnreadableStatusFileNoticeClearsOnRecoveryThroughTheModel() async throws {
        let clock = Clock(T.now)
        let readable = try document(T.backups(), at: T.now)
        final class Result: @unchecked Sendable { var value: ObservabilityReadResult = .unreadable("Unexpected character") }
        let result = Result()
        let model = model(clock: clock, observability: { result.value })
        model.applyHealthBadge(.init(badgeOn: false, reason: "", activeCodes: []), alerts: BrainBarJobAlerts(active: [:]))
        for (offset, dot) in [(0.0, false), (30.0, false), (60.0, false), (61.0, false)] {
            clock.now = T.now.addingTimeInterval(offset)
            await model.sampleStatusTick(includeLaunchd: false)
            XCTAssertEqual(model.publishedStatus.needsAttention, dot, "t+\(offset)s: \(model.publishedStatus.issues)")
        }
        result.value = readable
        clock.now = T.now.addingTimeInterval(90)
        await model.sampleStatusTick(includeLaunchd: false)
        XCTAssertFalse(model.publishedStatus.needsAttention, "recovery clears the episode: \(model.publishedStatus.issues)")
        result.value = .unreadable("again")
        clock.now = T.now.addingTimeInterval(120)
        await model.sampleStatusTick(includeLaunchd: false)
        XCTAssertFalse(model.publishedStatus.needsAttention, "a new episode starts its own clock")
    }

    func testOneTickReadsEachSourceOnceAndLaunchdOnItsSlowerSubCadence() async throws {
        let clock = Clock(T.now)
        let readable = try document(T.backups(), at: T.now)
        let watcherReads = FixCounter(), launchdReads = FixCounter()
        let model = model(clock: clock, observability: { readable }, watcherReads: watcherReads, launchdReads: launchdReads)
        let ticks = FixCounter()
        await model.monitorStatus(launchdEvery: 2) {
            ticks.value += 1
            if ticks.value == 4 { throw CancellationError() }
        }
        XCTAssertEqual(watcherReads.value, 4, "one watcher read per tick")
        XCTAssertEqual(launchdReads.value, 2, "launchd on every second tick")
    }

    func testNewAlertReachesPublishedNoticeWithNoWindowAndNoMenuOpen() async throws {
        let root = URL(fileURLWithPath: brainBarTestTemporaryPath(), isDirectory: true)
            .appendingPathComponent("brainbar-s5-fix1-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let alerts = root.appendingPathComponent("job-alerts.json")
        try Data("{}".utf8).write(to: alerts)
        let clock = Clock(T.now)
        let readable = try document(T.backups(), at: T.now)
        let model = model(clock: clock, observability: { readable }, badgeURL: root.appendingPathComponent("badge-state.json"),
                          alertsURL: alerts)
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        runtime.jobsModel = model
        runtime.driveAuth.setForPreview(status: .init(state: .valid, reason: nil, expiresAt: T.now.addingTimeInterval(30 * Self.day)))
        let workspace = BrainBarRecordingWorkspace()
        let controller = BrainBarStatusPopoverController(workspace: workspace, runtime: runtime,
            dashboardPanelController: BrainBarDashboardPanelController(workspace: workspace, runtime: runtime),
            statusBar: TestStatusItemHost())
        defer { controller.stop() }
        await model.sampleStatusTick(includeLaunchd: true)
        XCTAssertFalse(controller.statusDotForTesting, "\(model.publishedStatus.issues)")
        try Data(#"{"backup-upload-abandoned": "Local DB snapshot invalid; upload abandoned"}"#.utf8).write(to: alerts)
        await model.sampleStatusTick(includeLaunchd: false)
        XCTAssertFalse(controller.statusDotForTesting)
        XCTAssertTrue(model.publishedStatus.notices.contains { $0.alertKey == "backup-upload-abandoned" })
        try Data("{}".utf8).write(to: alerts)
        await model.sampleStatusTick(includeLaunchd: false)
        XCTAssertFalse(controller.statusDotForTesting)
    }

    func testLosingTheAlertSourceKeepsAKnownCriticalUntilAReadableEmptyMap() async throws {
        let root = URL(fileURLWithPath: brainBarTestTemporaryPath(), isDirectory: true)
            .appendingPathComponent("brainbar-s5-fix2-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let alerts = root.appendingPathComponent("job-alerts.json")
        try Data(#"{"mcp-transport": "BrainBar MCP transport: event buffer overflow"}"#.utf8).write(to: alerts)
        let clock = Clock(T.now)
        let readable = try document(T.backups(), at: T.now)
        let model = model(clock: clock, observability: { readable }, badgeURL: root.appendingPathComponent("badge-state.json"),
                          alertsURL: alerts)
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        runtime.jobsModel = model
        runtime.driveAuth.setForPreview(status: .init(state: .valid, reason: nil, expiresAt: T.now.addingTimeInterval(30 * Self.day)))
        await model.sampleStatusTick(includeLaunchd: true)
        XCTAssertFalse(model.publishedStatus.needsAttention, "\(model.publishedStatus.issues)")
        try Data("{not json".utf8).write(to: alerts)
        await model.sampleStatusTick(includeLaunchd: false)
        XCTAssertFalse(model.publishedStatus.needsAttention)
        XCTAssertTrue(model.publishedStatus.notices.contains { $0.alertKey == "mcp-transport" })
        XCTAssertNotEqual(model.publishedStatus.dashboardStatus.headline, "All good")
        try Data("{}".utf8).write(to: alerts)
        await model.sampleStatusTick(includeLaunchd: false)
        XCTAssertFalse(model.publishedStatus.needsAttention, "a readable empty map is the recovery: \(model.publishedStatus.issues)")
    }

    func testEverySurfaceReadsTheSamePublishedSnapshot() async throws {
        let clock = Clock(T.now)
        let readable = try document(T.backups(), at: T.now)
        let model = model(clock: clock, observability: { readable })
        await model.sampleStatusTick(includeLaunchd: true)
        let published = model.publishedStatus
        clock.now = T.now.addingTimeInterval(10 * Self.day) // surfaces must not re-derive with their own clock
        XCTAssertEqual(model.publishedStatus, published)
        XCTAssertEqual(model.groupStatus(.backups).health, published.project(
            BrainLayerLaunchdGroupStatus(health: .healthy, attentionReason: nil, lastRunText: "", nextRunText: ""), group: .backups).health)
    }

    func testBackupsBadgeNamesTheSnapshotsAttentionNotANeutralRawFailure() async throws {
        // DB backup failed once 12 h ago; a verified DB copy exists from 6 h ago; transcripts are 37 h old.
        let clock = Clock(T.now)
        var observations = T.healthy().observations!
        observations[.backupDaily] = T.failed(.backupDaily, code: 1, at: T.hoursAgo(12))
        let readable = try document(T.backups(db: T.hoursAgo(6), transcripts: T.hoursAgo(37)), at: T.now)
        let model = model(clock: clock, observability: { readable }, observations: observations)
        model.applyHealthBadge(.init(badgeOn: false, reason: "", activeCodes: []), alerts: BrainBarJobAlerts(active: [:]))
        await model.sampleStatusTick(includeLaunchd: true)
        let attention = try XCTUnwrap(model.publishedStatus.attention.first)
        XCTAssertEqual(attention.component, .transcripts)
        let health = model.backupsHealth(drive: nil)
        XCTAssertEqual(health.badge, .attention)
        XCTAssertEqual(health.reason, attention.text, "the neutral DB failure must not speak for the badge")
    }
}

private final class FixCounter: @unchecked Sendable {
    var value = 0
}

private struct CountingActivity: BrainLayerLaunchdStatusSampling, @unchecked Sendable {
    let observations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation]
    let reads: FixCounter

    func sample() -> [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] { observations.mapValues(\.loadState) }

    func sampleActivity() -> [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation] {
        reads.value += 1
        return observations
    }
}
