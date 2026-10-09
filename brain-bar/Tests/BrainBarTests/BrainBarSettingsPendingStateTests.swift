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
        XCTAssertEqual(model.groupStatus(.maintenance).health, .unhealthy)
        XCTAssertEqual(model.launchdStatusTitle(.hotlane), "Unknown")
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .attention,
                       "A completed job failure stays visible even while backup checks are pending")
        reads.release(0)
        try await waitUntil { !model.isRefreshingObservabilityStatus }
        XCTAssertEqual(model.backupStatusReason, "Read failed")
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .attention)
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
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .attention)
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
            XCTAssertEqual(model.backupsHealth(drive: nil).badge, .attention)
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
        XCTAssertEqual(shared.backupsHealth(drive: nil).badge, .attention)
        XCTAssertEqual(fresh.backupsHealth(drive: nil).badge, .checking)
        freshLaunchd.release(0)
        freshReads.release(0)
        try await waitUntil { !fresh.isRefreshingLaunchdStatus && !fresh.isRefreshingObservabilityStatus }
        XCTAssertEqual(fresh.backupsHealth(drive: nil).badge, .healthy)
        XCTAssertEqual(shared.backupsHealth(drive: nil).badge, .attention)
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
        XCTAssertEqual(model.groupStatus(.maintenance).health, .unhealthy)
        XCTAssertEqual(model.backupStatusReason, "New failure")
        XCTAssertEqual(model.launchdStatusTitle(.hotlane), "Unknown", "An empty completed sample must clear prior active state")

        model.refreshAllStatus()
        try await waitUntil { launchd.calls == 2 && reads.calls == 2 }
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .attention)
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

    func testDisabledWatcherIsVisibleEvenBeforeTheFirstLaunchdSample() throws {
        let launchd = PendingLaunchdSamples([healthyStates])
        let reads = PendingSettingsSamples([BrainBarDashboardFixture.healthyObservabilityResult])
        defer { launchd.releaseAll(); reads.releaseAll() }
        let (root, model) = try makeModel(launchd: launchd, reads: reads)
        defer { try? FileManager.default.removeItem(at: root) }
        model.setJob(.watch, enabled: false)
        XCTAssertEqual(model.footerPresentation.state, .watcher(model.watcherStatus))
        XCTAssertEqual(model.groupStatus(.ingest).health, .unhealthy)
    }

    private func makeModel(
        launchd: PendingLaunchdSamples, reads: PendingSettingsSamples<ObservabilityReadResult>,
        initialStates: [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] = [:],
        initialResult: ObservabilityReadResult? = nil,
        refreshOnLoad: Bool = false
    ) throws -> (URL, BrainBarSettingsViewModel) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("settings-pending-\(UUID().uuidString)")
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        return (root, BrainBarSettingsViewModel(
            store: store, launchdStatusProvider: launchd,
            embeddingResidencyProbe: StaticBrainBarEmbeddingResidencyProbe(state: .unmeasurable),
            initialLaunchdStates: initialStates, refreshStatusOnLoad: refreshOnLoad,
            observabilityURL: root.appendingPathComponent("observability.json"), initialObservabilityResult: initialResult,
            observabilityRead: { _ in await Task.detached { reads.sample() }.value }
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
