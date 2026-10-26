@testable import BrainBar
import XCTest

@MainActor
final class BrainBarSettingsPendingStateTests: XCTestCase {
    private var healthyStates: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] {
        Dictionary(uniqueKeysWithValues: BrainLayerLaunchdJob.allCases.map { ($0, .running) })
    }

    func testFirstSamplesStayNeutralUntilEachProviderCompletes() async throws {
        let launchd = PendingLaunchdSamples([healthyStates])
        let reads = PendingSettingsSamples([BrainBarDashboardFixture.healthyObservabilityResult])
        defer { launchd.releaseAll(); reads.releaseAll() }
        let (root, model) = try makeModel(launchd: launchd, reads: reads, refreshOnLoad: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for group in BrainLayerLaunchdJobGroup.allCases {
            XCTAssertEqual(model.groupStatus(group).health, .checking)
            XCTAssertNil(model.groupStatus(group).attentionReason)
        }
        XCTAssertEqual(model.footerPresentation.state, .checking)
        XCTAssertEqual(model.launchdStatusTitle(.hotlane), "Checking…")
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .checking)
        XCTAssertNil(model.backupStatusReason)
        XCTAssertTrue(model.isRefreshingObservabilityStatus)

        try await waitUntil { launchd.calls == 1 && reads.calls == 1 }
        launchd.release(0)
        try await waitUntil { !model.isRefreshingLaunchdStatus }
        XCTAssertEqual(model.groupStatus(.maintenance).health, .healthy)
        XCTAssertEqual(model.launchdStatusTitle(.hotlane), "Running")
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .checking)
        reads.release(0)
        try await waitUntil { !model.isRefreshingObservabilityStatus }
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .healthy)
        XCTAssertNil(model.backupStatusReason)
    }

    func testCompletedEmptyAndUnreadableSamplesBecomeRealErrors() async throws {
        let launchd = PendingLaunchdSamples([[:]])
        let reads = PendingSettingsSamples<ObservabilityReadResult>([.unreadable("Read failed")])
        defer { launchd.releaseAll(); reads.releaseAll() }
        let (root, model) = try makeModel(launchd: launchd, reads: reads)
        defer { try? FileManager.default.removeItem(at: root) }
        model.refreshLaunchdStatus()
        launchd.release(0)
        try await waitUntil { !model.isRefreshingLaunchdStatus }
        XCTAssertTrue(model.hasCompletedLaunchdSample)
        // S5 dot policy: an empty sample is unmeasured — visible "Status unknown", never the dot.
        XCTAssertEqual(model.groupStatus(.maintenance).health, .unknown)
        XCTAssertEqual(model.groupStatus(.maintenance).attentionReason, "Nightly status is unavailable.")
        XCTAssertEqual(model.launchdStatusTitle(.hotlane), "Unknown")
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .checking,
                       "An unmeasured job is not a failure; the backup checks are still pending")
        reads.release(0)
        try await waitUntil { !model.isRefreshingObservabilityStatus }
        XCTAssertEqual(model.backupStatusReason, "Read failed")
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .unknown)
        XCTAssertFalse(model.publishedStatus.needsAttention)
    }

    func testKnownErrorsRemainVisibleThroughoutRefreshAndReload() async throws {
        let failedStates = healthyStates.merging([.maintenanceNightly: .unloaded, .backupDaily: .unloaded]) { _, new in new }
        let launchd = PendingLaunchdSamples([failedStates])
        let reads = PendingSettingsSamples<ObservabilityReadResult>([.unreadable("Still failed")])
        defer { launchd.releaseAll(); reads.releaseAll() }
        let (root, model) = try makeModel(launchd: launchd, reads: reads, initialStates: failedStates,
                                          initialResult: .unreadable("Still failed"))
        defer { try? FileManager.default.removeItem(at: root) }
        let before = model.groupStatus(.maintenance)
        model.refreshLaunchdStatus()
        XCTAssertTrue(model.reloadConfigFromDisk(preservingDrafts: true))
        try await waitUntil { launchd.calls == 1 && reads.calls == 1 }
        XCTAssertEqual(model.groupStatus(.maintenance), before)
        XCTAssertEqual(model.backupStatusReason, "Still failed")
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .unknown, "S5: unreadable is neutral, still visible")
        XCTAssertEqual(model.launchdStatusTitle(.backupDaily), "Unloaded")
        launchd.release(0)
        reads.release(0)
        try await waitUntil { !model.isRefreshingLaunchdStatus && !model.isRefreshingObservabilityStatus }
        XCTAssertEqual(model.groupStatus(.maintenance), before)
        XCTAssertEqual(model.backupStatusReason, "Still failed")
    }

    func testCompletedBackupReadFailuresStayRedBesideHealthyJobs() async throws {
        let fixtureURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("tests/fixtures/observability/golden/missing-launchd-dev.json")
        let results: [ObservabilityReadResult] = [.unreadable("Read failed"), ObservabilityReader.read(url: fixtureURL)]
        for result in results {
            let launchd = PendingLaunchdSamples([healthyStates])
            let reads = PendingSettingsSamples([result])
            defer { launchd.releaseAll(); reads.releaseAll() }
            let (root, model) = try makeModel(launchd: launchd, reads: reads, initialStates: healthyStates)
            defer { try? FileManager.default.removeItem(at: root) }
            XCTAssertEqual(model.backupsHealth(drive: nil).badge, .checking)
            reads.release(0)
            try await waitUntil { !model.isRefreshingObservabilityStatus }
            XCTAssertEqual(model.groupStatus(.backups).health, .healthy)
            XCTAssertTrue(model.hasCompletedObservabilityRead)
            XCTAssertEqual(model.backupsHealth(drive: nil).badge, .unknown, "S5: unmeasured is neutral, still visible")
            XCTAssertNotNil(model.backupStatusReason)
            XCTAssertEqual(model.backupsHealth(drive: nil).reason, model.backupStatusReason)
        }
    }

    func testRecreatedModelStartsCheckingWhileSharedModelRetainsMeasuredFailure() async throws {
        let launchd = PendingLaunchdSamples([healthyStates])
        let reads = PendingSettingsSamples<ObservabilityReadResult>([.unreadable("Measured failure")])
        let freshLaunchd = PendingLaunchdSamples([healthyStates])
        let freshReads = PendingSettingsSamples([BrainBarDashboardFixture.healthyObservabilityResult])
        defer { launchd.releaseAll(); reads.releaseAll(); freshLaunchd.releaseAll(); freshReads.releaseAll() }
        let (root, shared) = try makeModel(launchd: launchd, reads: reads, initialStates: healthyStates,
                                           initialResult: .unreadable("Measured failure"))
        let (freshRoot, fresh) = try makeModel(launchd: freshLaunchd, reads: freshReads, refreshOnLoad: true)
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: freshRoot) }
        XCTAssertEqual(shared.backupsHealth(drive: nil).badge, .unknown)
        XCTAssertEqual(fresh.backupsHealth(drive: nil).badge, .checking)
        freshLaunchd.release(0)
        freshReads.release(0)
        try await waitUntil { !fresh.isRefreshingLaunchdStatus && !fresh.isRefreshingObservabilityStatus }
        XCTAssertEqual(fresh.backupsHealth(drive: nil).badge, .healthy)
        XCTAssertEqual(shared.backupsHealth(drive: nil).badge, .unknown)
    }

    func testHealthyStateSurvivesReentryThenPublishesFailureAndRecovery() async throws {
        let failedStates: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] = [:]
        let launchd = PendingLaunchdSamples([failedStates, healthyStates])
        let reads = PendingSettingsSamples<ObservabilityReadResult>([
            .unreadable("New failure"), BrainBarDashboardFixture.healthyObservabilityResult,
        ])
        defer { launchd.releaseAll(); reads.releaseAll() }
        let (root, model) = try makeModel(launchd: launchd, reads: reads, initialStates: healthyStates,
                                          initialResult: BrainBarDashboardFixture.healthyObservabilityResult)
        defer { try? FileManager.default.removeItem(at: root) }
        let navigation = BrainBarSettingsNavigation(selected: .jobs)
        navigation.select(.backups)
        navigation.select(.advanced)
        // The view's activationRevision handler reloads and refreshes this shared model.
        XCTAssertTrue(model.reloadConfigFromDisk(preservingDrafts: true))
        model.refreshLaunchdStatus()
        try await waitUntil { launchd.calls == 1 && reads.calls == 1 }
        XCTAssertEqual(model.groupStatus(.maintenance).health, .healthy)
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .healthy)
        launchd.release(0)
        reads.release(0)
        try await waitUntil { !model.isRefreshingLaunchdStatus && !model.isRefreshingObservabilityStatus }
        XCTAssertEqual(model.groupStatus(.maintenance).health, .unknown, "S5: an empty sample is unmeasured")
        XCTAssertEqual(model.backupStatusReason, "New failure")
        XCTAssertEqual(model.launchdStatusTitle(.hotlane), "Unknown", "An empty completed sample must clear prior active state")

        model.refreshAllStatus()
        try await waitUntil { launchd.calls == 2 && reads.calls == 2 }
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .unknown)
        XCTAssertEqual(model.backupStatusReason, "New failure")
        launchd.release(1)
        reads.release(1)
        try await waitUntil { !model.isRefreshingLaunchdStatus && !model.isRefreshingObservabilityStatus }
        XCTAssertEqual(model.groupStatus(.maintenance).health, .healthy)
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .healthy)
        XCTAssertNil(model.backupStatusReason)
    }

    func testOverlappingLaunchdRefreshesOnlyPublishNewestSampleInEitherCompletionOrder() async throws {
        for newestFinishesFirst in [false, true] {
            let launchd = PendingLaunchdSamples([[:], healthyStates])
            let reads = PendingSettingsSamples([BrainBarDashboardFixture.healthyObservabilityResult])
            defer { launchd.releaseAll(); reads.releaseAll() }
            let (root, model) = try makeModel(launchd: launchd, reads: reads)
            defer { try? FileManager.default.removeItem(at: root) }
            reads.release(0)
            model.refreshLaunchdStatus()
            try await waitUntil { launchd.calls == 1 }
            model.refreshLaunchdStatus()
            try await waitUntil { launchd.calls == 2 }
            if newestFinishesFirst {
                launchd.release(1)
                try await waitUntil { !model.isRefreshingLaunchdStatus }
                launchd.release(0)
            } else {
                launchd.release(0)
                try await waitUntil { launchd.completed == 1 }
                // Let the obsolete task resume on the main actor before checking pending state.
                try await Task.sleep(for: .milliseconds(50))
                XCTAssertTrue(model.isRefreshingLaunchdStatus)
                XCTAssertFalse(model.hasCompletedLaunchdSample)
                XCTAssertEqual(model.groupStatus(.maintenance).health, .checking)
                launchd.release(1)
            }
            try await waitUntil { !model.isRefreshingLaunchdStatus && launchd.completed == 2 }
            try await Task.sleep(for: .milliseconds(50))
            XCTAssertEqual(model.groupStatus(.maintenance).health, .healthy)
            XCTAssertEqual(model.launchdStatusTitle(.hotlane), "Running")
        }
    }

    func testPendingBackupChecksCannotHideMeasuredAlertsOrDriveFailures() {
        let job = BrainLayerLaunchdGroupStatus(health: .checking, attentionReason: nil, lastRunText: "", nextRunText: "")
        let alert = ObservabilityPresentation.backupStatus(for: {
            guard case let .readable(document) = BrainBarDashboardFixture.maintenanceAlertObservabilityResult else {
                fatalError("Expected readable fixture")
            }
            return document.backups
        }())
        XCTAssertEqual(BrainBarBackupsHealth.derive(job: job, drive: nil, status: alert,
                                                    statusUnavailableReason: "unused", isCheckingStatus: true).badge, .attention)
        let drive = DriveAuthPresentation.derive(status: .init(state: .missing, reason: nil, expiresAt: nil), isReconnecting: false,
                                                 lastOutcome: nil, now: Date(), formatDate: { _ in "" })
        XCTAssertEqual(BrainBarBackupsHealth.derive(job: job, drive: drive, status: nil,
                                                    statusUnavailableReason: "unused", isCheckingStatus: true).badge, .attention)
    }

    func testConfiguredOffWatcherStaysCheckingBeforeTheFirstLaunchdSample() throws {
        let launchd = PendingLaunchdSamples([healthyStates])
        let reads = PendingSettingsSamples([BrainBarDashboardFixture.healthyObservabilityResult])
        defer { launchd.releaseAll(); reads.releaseAll() }
        let (root, model) = try makeModel(launchd: launchd, reads: reads)
        defer { try? FileManager.default.removeItem(at: root) }
        model.setJob(.watch, enabled: false)
        XCTAssertEqual(model.footerPresentation.state, .watcher(model.watcherStatus))
        XCTAssertEqual(model.groupStatus(.ingest).health, .checking)
        XCTAssertNil(model.groupStatus(.ingest).attentionReason)
        XCTAssertEqual(model.groupStatus(.ingest).lastRunText, "Checking…")
        XCTAssertEqual(model.groupStatus(.ingest).nextRunText, "Checking…")
        if case .unknown = model.watcherStatus {} else {
            XCTFail("Configured OFF without a sample is unknown runtime, never an invented stop")
        }
    }

    func testMeasuredWatcherFailureWinsInBothIndependentReaderOrders() async throws {
        let now = Date(timeIntervalSince1970: 1_791_500_000)
        let health = WatcherHealthFileRead.readable(WatcherHealthFile(
            updatedAt: now, pollCount: 72, alertReasons: ["file_ingestion_failure"], dbProbeFailed: true,
            fileIngestionFailureCount: 2, earliestFileIngestionFailureAt: now.addingTimeInterval(-60)
        ))
        for watcherFirst in [true, false] {
            let launchd = PendingLaunchdSamples([healthyStates])
            let reads = PendingSettingsSamples([BrainBarDashboardFixture.healthyObservabilityResult])
            let watcherReads = PendingSettingsSamples([health])
            defer { launchd.releaseAll(); reads.releaseAll(); watcherReads.releaseAll() }
            let (root, model) = try makeModel(launchd: launchd, reads: reads, refreshOnLoad: true,
                                            watcherReads: watcherReads, now: now)
            defer { try? FileManager.default.removeItem(at: root) }
            try await waitUntil { launchd.calls == 1 && watcherReads.calls == 1 }
            XCTAssertEqual(model.groupStatus(.ingest).health, .checking)
            XCTAssertEqual(model.footerPresentation.state, .checking)
            if !watcherFirst {
                launchd.release(0)
                try await waitUntil { !model.isRefreshingLaunchdStatus }
            }
            watcherReads.release(0)
            try await waitUntil { model.watcherHealth == health }
            XCTAssertTrue(model.watcherStatus.needsAttention)
            let reason = try XCTUnwrap(model.watcherStatus.reasonText(now: now))
            XCTAssertTrue(reason.contains("2 transcript files could not be ingested"))
            // L2: a fresh heartbeat is not recovery from a DB-read failure.
            XCTAssertEqual(model.groupStatus(.ingest).health, .unhealthy)
            XCTAssertTrue(model.publishedStatus.needsAttention)
            XCTAssertEqual(model.groupStatus(.ingest).attentionReason, reason)
            XCTAssertEqual(model.footerPresentation.state, .watcher(model.watcherStatus))
            XCTAssertEqual(model.footerPresentation.detail, reason)
            if watcherFirst {
                XCTAssertTrue(model.isRefreshingLaunchdStatus)
                XCTAssertFalse(model.hasCompletedLaunchdSample)
                XCTAssertEqual(model.groupStatus(.maintenance).health, .checking)
                launchd.release(0)
                try await waitUntil { !model.isRefreshingLaunchdStatus }
                XCTAssertEqual(model.groupStatus(.ingest).attentionReason, reason)
                XCTAssertEqual(model.footerPresentation.detail, reason)
            }
        }
    }

    func testInjectedWatcherAttentionWinsWhileHealthyAndUnknownStayChecking() throws {
        let now = Date(timeIntervalSince1970: 1_791_500_000)
        let degraded = WatcherHealthFileRead.readable(WatcherHealthFile(updatedAt: now, pollCount: 72, dbProbeFailed: true))
        let controls: [WatcherHealthFileRead?] = [nil, .missing(path: "/fixture/watcher-health.json"),
                                                .readable(WatcherHealthFile(updatedAt: now, pollCount: 72))]
        for health in [degraded] + controls {
            let launchd = PendingLaunchdSamples([healthyStates])
            let reads = PendingSettingsSamples([BrainBarDashboardFixture.healthyObservabilityResult])
            defer { launchd.releaseAll(); reads.releaseAll() }
            let (root, model) = try makeModel(launchd: launchd, reads: reads, initialWatcherHealth: health, now: now)
            defer { try? FileManager.default.removeItem(at: root) }
            XCTAssertFalse(model.hasCompletedLaunchdSample)
            let failed = health == degraded
            XCTAssertEqual(model.watcherStatus.needsAttention, failed)
            XCTAssertEqual(model.groupStatus(.ingest).health, failed ? .unhealthy : .checking, "L2: DB-read failure")
            XCTAssertEqual(model.footerPresentation.state, failed ? .watcher(model.watcherStatus) : .checking)
            XCTAssertEqual(model.groupStatus(.ingest).attentionReason, failed ? model.watcherStatus.reasonText(now: now) : nil)
            XCTAssertEqual(model.footerPresentation.detail, failed ? model.watcherStatus.reasonText(now: now) : nil)
        }
    }

    func testPartiallyConfiguredOffGroupsStayCheckingUntilFirstLaunchdSample() async throws {
        for group in BrainLayerLaunchdJobGroup.allCases {
            let launchd = PendingLaunchdSamples([healthyStates])
            let reads = PendingSettingsSamples([BrainBarDashboardFixture.healthyObservabilityResult])
            defer { launchd.releaseAll(); reads.releaseAll() }
            let (root, model) = try makeModel(
                launchd: launchd, reads: reads, refreshOnLoad: true,
                initialWatcherHealth: BrainBarDashboardFixture.healthyWatcherHealth,
                now: BrainBarDashboardFixture.fetchedAt
            )
            defer { try? FileManager.default.removeItem(at: root) }
            let disabled = try XCTUnwrap(group.jobs.last)
            model.setJob(disabled, enabled: false)
            try await waitUntil { launchd.calls == 1 }
            XCTAssertFalse(model.hasCompletedLaunchdSample)
            let pending = model.groupStatus(group)
            XCTAssertEqual(pending.health, .checking)
            XCTAssertNil(pending.attentionReason)
            let configurationNote: String = switch group {
            case .ingest: "Configured OFF: Index."
            case .backups: "Configured OFF: Transcripts."
            case .maintenance: "Configured OFF: Weekly."
            }
            XCTAssertEqual(pending.lastRunText, "Checking…")
            XCTAssertEqual(pending.nextRunText, "Checking…")

            launchd.release(0)
            try await waitUntil { !model.isRefreshingLaunchdStatus }
            XCTAssertEqual(model.groupStatus(group).health, .healthy)
            XCTAssertNil(model.groupStatus(group).attentionReason)
            XCTAssertEqual(model.groupStatus(group).note, configurationNote)
            model.setJob(disabled, enabled: true)
            XCTAssertTrue(model.isGroupEnabled(group))
            XCTAssertNil(model.groupStatus(group).note)
            XCTAssertNotEqual(model.groupStatus(group).health, .checking)
            XCTAssertFalse(model.groupStatus(group).attentionReason?.contains("is disabled") ?? false)
        }
    }

    private var ingestConfigurations: [(watch: Bool, index: Bool, note: String?)] {
        [(true, true, nil),
         (false, false, "Configured OFF: Watcher, Index."),
         (false, true, "Configured OFF: Watcher."),
         (true, false, "Configured OFF: Index.")]
    }

    func testConfiguredIngestPendingTruthInBothIndependentReaderOrders() async throws {
        let now = BrainBarDashboardFixture.fetchedAt
        let healthy = BrainBarDashboardFixture.healthyWatcherHealth
        let degraded = WatcherHealthFileRead.readable(WatcherHealthFile(
            updatedAt: now, pollCount: 72, dbProbeFailed: true
        ))
        let stale = WatcherHealthFileRead.readable(WatcherHealthFile(
            updatedAt: now.addingTimeInterval(-900), pollCount: 72
        ))
        var sampledStates = healthyStates
        sampledStates[.index] = .loaded // Scheduled and loaded, not proof of execution.
        for configuration in ingestConfigurations {
            for health in [healthy, degraded, stale] {
                for watcherFirst in [true, false] {
                    let launchd = PendingLaunchdSamples([sampledStates])
                    let reads = PendingSettingsSamples([BrainBarDashboardFixture.healthyObservabilityResult])
                    let watcherReads = PendingSettingsSamples([health])
                    defer { launchd.releaseAll(); reads.releaseAll(); watcherReads.releaseAll() }
                    let (root, model) = try makeModel(
                        launchd: launchd, reads: reads, refreshOnLoad: true,
                        watcherReads: watcherReads, now: now,
                        watchEnabled: configuration.watch, indexEnabled: configuration.index
                    )
                    defer { try? FileManager.default.removeItem(at: root) }
                    try await waitUntil { launchd.calls == 1 && watcherReads.calls == 1 }
                    assertPendingConfiguredIngest(model, configuration: configuration)
                    XCTAssertFalse(model.hasCompletedLaunchdSample)
                    if !watcherFirst {
                        launchd.release(0)
                        try await waitUntil { model.hasCompletedLaunchdSample }
                        XCTAssertEqual(model.groupStatus(.ingest).health, .unknown)
                        XCTAssertEqual(model.footerPresentation.state, .watcher(model.watcherStatus))
                        XCTAssertEqual(model.footerPresentation.detail,
                                       "Watcher is running, but its health file has not been read yet.")
                    }
                    watcherReads.release(0)
                    try await waitUntil { model.watcherHealth == health }
                    let attention = health == degraded || (!watcherFirst && health == stale)
                    let reason = model.watcherStatus.reasonText(now: now)
                    if watcherFirst && !attention {
                        assertPendingConfiguredIngest(model, configuration: configuration)
                    } else {
                        // S5 + L2: a stale heartbeat or a DB-read failure on a desired-ON watcher is the dot.
                        let dot = configuration.watch
                        XCTAssertEqual(model.groupStatus(.ingest).health,
                                       attention ? (dot ? .unhealthy : .notice) : sampledIngestHealth(configuration))
                        XCTAssertEqual(model.groupStatus(.ingest).attentionReason, reason)
                        XCTAssertEqual(model.footerPresentation.state, .watcher(model.watcherStatus))
                        XCTAssertEqual(model.footerPresentation.detail, reason)
                    }
                    XCTAssertEqual(model.groupStatus(.ingest).note,
                                   watcherFirst && !attention ? nil : configuration.note)
                    if watcherFirst {
                        XCTAssertFalse(model.hasCompletedLaunchdSample)
                        launchd.release(0)
                        try await waitUntil { model.hasCompletedLaunchdSample }
                    }
                    let completedAttention = health != healthy
                    let completedReason = model.watcherStatus.reasonText(now: now)
                    XCTAssertEqual(model.watcherStatus.needsAttention, completedAttention)
                    let completedDot = configuration.watch
                    XCTAssertEqual(model.groupStatus(.ingest).health,
                                   completedAttention ? (completedDot ? .unhealthy : .notice) : sampledIngestHealth(configuration))
                    XCTAssertEqual(model.groupStatus(.ingest).attentionReason, completedReason)
                    XCTAssertEqual(model.footerPresentation.state, .watcher(model.watcherStatus))
                    XCTAssertEqual(model.footerPresentation.detail, completedReason)
                    XCTAssertEqual(model.groupStatus(.ingest).note, configuration.note)
                    XCTAssertEqual(model.config.launchdJobs[.watch]?.enabled, configuration.watch)
                    XCTAssertEqual(model.config.launchdJobs[.index]?.enabled, configuration.index)
                    XCTAssertEqual(model.launchdStatusTitle(.index), "Loaded")
                    let runtime = model.watcherStatus
                    model.setGroup(.ingest, enabled: false)
                    XCTAssertEqual(model.watcherStatus, runtime)
                    XCTAssertEqual(model.groupStatus(.ingest).health, completedAttention ? .notice : .skipped,
                                   "a configured-OFF watcher is not a dot")
                    XCTAssertEqual(model.groupStatus(.ingest).note, "Configured OFF: Watcher, Index.")
                    model.setGroup(.ingest, enabled: true)
                    XCTAssertEqual(model.watcherStatus, runtime)
                    XCTAssertEqual(model.groupStatus(.ingest).health,
                                   completedAttention ? .unhealthy : .healthy)
                    XCTAssertEqual(model.groupStatus(.ingest).attentionReason, completedReason)
                    XCTAssertNil(model.groupStatus(.ingest).note)
                }
            }
        }
    }

    func testPendingIngestOffOnAndMixedConfigurationKeepsMeasuredControls() throws {
        let now = BrainBarDashboardFixture.fetchedAt
        let healths: [WatcherHealthFileRead?] = [
            nil, .missing(path: "/fixture/health.json"), BrainBarDashboardFixture.healthyWatcherHealth,
            .readable(WatcherHealthFile(updatedAt: now.addingTimeInterval(-900), pollCount: 72)),
            .readable(WatcherHealthFile(updatedAt: now, pollCount: 72, dbProbeFailed: true)),
        ]
        for configuration in ingestConfigurations {
            for (position, health) in healths.enumerated() {
                let launchd = PendingLaunchdSamples([healthyStates])
                let reads = PendingSettingsSamples([BrainBarDashboardFixture.healthyObservabilityResult])
                defer { launchd.releaseAll(); reads.releaseAll() }
                let (root, model) = try makeModel(
                    launchd: launchd, reads: reads, initialWatcherHealth: health, now: now,
                    watchEnabled: configuration.watch, indexEnabled: configuration.index
                )
                defer { try? FileManager.default.removeItem(at: root) }
                let attention = position == healths.count - 1
                XCTAssertEqual(model.watcherStatus.needsAttention, attention)
                XCTAssertFalse(model.hasCompletedLaunchdSample)
                assertPendingConfiguredIngest(model, configuration: configuration)
                model.setGroup(.ingest, enabled: false)
                assertPendingConfiguredIngest(model, configuration: ingestConfigurations[1])
                model.setGroup(.ingest, enabled: true)
                XCTAssertTrue(model.isGroupEnabled(.ingest))
                assertPendingConfiguredIngest(model, configuration: ingestConfigurations[0])
                XCTAssertFalse(model.hasCompletedLaunchdSample)
            }
        }
    }

    func testConfiguredIngestOnlyPublishesNewestLaunchdSampleInEitherOrder() async throws {
        let now = BrainBarDashboardFixture.fetchedAt
        for configuration in ingestConfigurations {
            for newestFirst in [false, true] {
                var sampledStates = healthyStates
                sampledStates[.index] = .loaded
                let launchd = PendingLaunchdSamples([[:], sampledStates])
                let reads = PendingSettingsSamples([BrainBarDashboardFixture.healthyObservabilityResult])
                defer { launchd.releaseAll(); reads.releaseAll() }
                let (root, model) = try makeModel(
                    launchd: launchd, reads: reads,
                    initialWatcherHealth: BrainBarDashboardFixture.healthyWatcherHealth, now: now,
                    watchEnabled: configuration.watch, indexEnabled: configuration.index
                )
                defer { try? FileManager.default.removeItem(at: root) }
                model.refreshLaunchdStatus()
                try await waitUntil { launchd.calls == 1 }
                model.refreshLaunchdStatus()
                try await waitUntil { launchd.calls == 2 }
                if newestFirst {
                    launchd.release(1)
                    try await waitUntil { model.hasCompletedLaunchdSample }
                    launchd.release(0)
                } else {
                    launchd.release(0)
                    try await waitUntil { launchd.completed == 1 }
                    try await Task.sleep(for: .milliseconds(50))
                    XCTAssertFalse(model.hasCompletedLaunchdSample)
                    XCTAssertTrue(model.isRefreshingLaunchdStatus, "The stale sample must not finish the newest request")
                    XCTAssertTrue(model.launchdObservations.isEmpty, "The stale sample must not publish raw observations")
                    assertPendingConfiguredIngest(model, configuration: configuration)
                    launchd.release(1)
                }
                try await waitUntil { !model.isRefreshingLaunchdStatus && launchd.completed == 2 }
                try await Task.sleep(for: .milliseconds(50))
                XCTAssertTrue(model.hasCompletedLaunchdSample)
                XCTAssertEqual(model.config.launchdJobs[.watch]?.loadState, .running)
                XCTAssertEqual(model.config.launchdJobs[.index]?.loadState, .loaded)
                XCTAssertEqual(model.launchdObservations[.watch]?.loadState, .running)
                XCTAssertEqual(model.launchdObservations[.index]?.loadState, .loaded)
                XCTAssertEqual(model.groupStatus(.ingest).health, sampledIngestHealth(configuration))
                XCTAssertEqual(model.footerPresentation.state, .watcher(model.watcherStatus))
                XCTAssertEqual(model.groupStatus(.ingest).note, configuration.note)
            }
        }
    }

    private func sampledIngestHealth(_ configuration: (watch: Bool, index: Bool, note: String?)) -> BrainLayerLaunchdGroupHealth {
        configuration.watch || configuration.index ? .healthy : .skipped
    }

    private func assertPendingConfiguredIngest(
        _ model: BrainBarSettingsViewModel,
        configuration: (watch: Bool, index: Bool, note: String?),
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertFalse(model.hasCompletedLaunchdSample, file: file, line: line)
        let status = model.groupStatus(.ingest)
        if model.watcherStatus.needsAttention {
            // S5 + L2: these pending fixtures carry a fresh heartbeat but `dbProbeFailed` — a DB-read
            // failure is the dot while the watcher should run, neutral text when it is configured OFF.
            XCTAssertEqual(status.health, configuration.watch ? .unhealthy : .notice, file: file, line: line)
            XCTAssertEqual(status.attentionReason, model.watcherStatus.reasonText(now: BrainBarDashboardFixture.fetchedAt),
                           file: file, line: line)
            XCTAssertEqual(status.note, configuration.note, file: file, line: line)
        } else {
            XCTAssertEqual(status.health, .checking, file: file, line: line)
            XCTAssertNil(status.attentionReason, file: file, line: line)
            XCTAssertEqual(status.lastRunText, "Checking…", file: file, line: line)
            XCTAssertEqual(status.nextRunText, "Checking…", file: file, line: line)
            XCTAssertNil(status.note, file: file, line: line)
            if case .stopped = model.watcherStatus {
                XCTFail("Desired OFF must not manufacture stopped runtime before a sample", file: file, line: line)
            }
        }
        if configuration.watch && !model.watcherStatus.needsAttention {
            XCTAssertEqual(model.footerPresentation.state, .checking, file: file, line: line)
            XCTAssertNil(model.footerPresentation.detail, file: file, line: line)
        } else {
            XCTAssertEqual(model.footerPresentation.state, .watcher(model.watcherStatus), file: file, line: line)
            XCTAssertEqual(model.footerPresentation.detail, model.watcherStatus.reasonText(now: BrainBarDashboardFixture.fetchedAt),
                           file: file, line: line)
        }
    }

    func testOpenSettingsRereadsHeartbeatAfterSupervisorRestartWithoutAnotherActivation() async throws {
        let now = Date(timeIntervalSince1970: 1_791_500_000)
        let old = WatcherHealthFileRead.readable(.init(updatedAt: now.addingTimeInterval(-3600), pollCount: 400))
        let fresh = WatcherHealthFileRead.readable(.init(updatedAt: now, pollCount: 1))
        let launchd = PendingLaunchdSamples([healthyStates])
        let reads = PendingSettingsSamples([BrainBarDashboardFixture.healthyObservabilityResult])
        let watcherReads = PendingSettingsSamples([old, fresh])
        defer { launchd.releaseAll(); reads.releaseAll(); watcherReads.releaseAll() }
        let (root, model) = try makeModel(launchd: launchd, reads: reads, initialStates: healthyStates,
                                        watcherReads: watcherReads, initialWatcherHealth: old, now: now)
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertTrue(model.watcherStatus.needsAttention)
        let gate = WatcherHealthPollGate()
        let task = Task { await model.monitorWatcherHealth { try await gate.next() } }
        defer { task.cancel() }
        watcherReads.release(0)
        try await waitUntil { watcherReads.completed == 1 }
        // Python has restarted and polls again; no launchd or DB refresh is requested.
        await gate.advance()
        watcherReads.release(1)
        try await waitUntil { model.watcherHealth == fresh }
        XCTAssertEqual(model.watcherStatus, .running(heartbeatAt: now))
        XCTAssertEqual(launchd.calls, 0)
        task.cancel()
        await gate.finish()
        await task.value
    }

    private func makeModel(
        launchd: PendingLaunchdSamples, reads: PendingSettingsSamples<ObservabilityReadResult>,
        initialStates: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] = [:],
        initialResult: ObservabilityReadResult? = nil,
        refreshOnLoad: Bool = false,
        watcherReads: PendingSettingsSamples<WatcherHealthFileRead>? = nil,
        initialWatcherHealth: WatcherHealthFileRead? = nil,
        now: Date = BrainBarDashboardFixture.fetchedAt,
        watchEnabled: Bool = true,
        indexEnabled: Bool = true
    ) throws -> (URL, BrainBarSettingsViewModel) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("settings-pending-\(UUID().uuidString)")
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        var config = BrainLayerConfig.defaultConfig
        config.launchdJobs[.watch]?.enabled = watchEnabled
        config.launchdJobs[.index]?.enabled = indexEnabled
        try store.save(config)
        return (root, BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(), store: store, launchdStatusProvider: launchd,
            embeddingResidencyProbe: StaticBrainBarEmbeddingResidencyProbe(state: .unmeasurable),
            initialLaunchdStates: initialStates, refreshStatusOnLoad: refreshOnLoad,
            now: { now },
            observabilityURL: root.appendingPathComponent("observability.json"), initialObservabilityResult: initialResult,
            observabilityRead: { _ in await Task.detached { reads.sample() }.value },
            watcherHealthURL: watcherReads == nil ? nil : root.appendingPathComponent("watcher-health.json"),
            initialWatcherHealth: initialWatcherHealth,
            watcherHealthRead: { _ in watcherReads?.sample() ?? .missing(path: "/fixture/watcher-health.json") }
        ))
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        for _ in 0 ..< 200 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Timed out waiting for gated sample")
    }
}

private final class PendingLaunchdSamples: BrainLayerLaunchdStatusSampling, @unchecked Sendable {
    private let samples: PendingSettingsSamples<[BrainLayerLaunchdJob: BrainLayerLaunchdLoadState]>
    init(_ values: [[BrainLayerLaunchdJob: BrainLayerLaunchdLoadState]]) {
        samples = .init(values)
    }

    var calls: Int {
        samples.calls
    }

    var completed: Int {
        samples.completed
    }

    func sample() -> [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] {
        samples.sample()
    }

    func release(_ index: Int) {
        samples.release(index)
    }

    func releaseAll() {
        samples.releaseAll()
    }
}

private final class PendingSettingsSamples<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    private var completedCount = 0
    private let values: [Value]
    private let gates: [DispatchSemaphore]
    init(_ values: [Value]) {
        self.values = values
        gates = values.map { _ in DispatchSemaphore(value: 0) }
    }

    var calls: Int {
        lock.withLock { count }
    }

    var completed: Int {
        lock.withLock { completedCount }
    }

    func sample() -> Value {
        let index = lock.withLock { let index = count; count += 1; return index }
        gates[index].wait()
        lock.withLock { completedCount += 1 }
        return values[index]
    }

    func release(_ index: Int) {
        gates[index].signal()
    }

    func releaseAll() {
        gates.forEach { $0.signal() }
    }
}

private actor WatcherHealthPollGate {
    private var continuation: CheckedContinuation<Void, Error>?
    private var released = false
    private var finished = false
    func next() async throws {
        if finished { throw CancellationError() }
        if released { released = false; return }
        try await withCheckedThrowingContinuation { continuation = $0 }
    }
    func advance() {
        if let pending = continuation { continuation = nil; pending.resume() }
        else { released = true }
    }
    func finish() {
        finished = true
        continuation?.resume(throwing: CancellationError())
        continuation = nil
    }
}
