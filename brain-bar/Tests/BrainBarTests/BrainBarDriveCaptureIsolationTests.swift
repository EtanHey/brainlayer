import XCTest
@testable import BrainBar

@MainActor
final class BrainBarDriveCaptureIsolationTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_792_300_000)
    private final class Reads: @unchecked Sendable {
        private let lock = NSLock()
        private var values: Set<String> = []
        func record(_ name: String) { lock.withLock { _ = values.insert(name) } }
        var names: Set<String> { lock.withLock { values } }
    }

    func test_settings_refresh_rejects_live_sampler_before_sampling() async {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let reads = Reads()
        let fixture = BrainBarDriveConsentCaptureFixture(directory: directory, now: now, initialEmbeddingProcess: .stopped,
                                                        didRead: { reads.record($0) })
        let probe = Mirror(reflecting: fixture.settings).children.first { $0.label == "embeddingResidencyProbe" }?.value
        let isStatic = probe is StaticBrainBarEmbeddingResidencyProbe
        XCTAssertTrue(isStatic, "Unsafe capture default; RED must stop before invoking any process sampler")
        guard isStatic else { return }
        await settle { !fixture.settings.isRefreshingObservabilityStatus }
        fixture.settings.refreshAllStatus()
        await settle {
            !fixture.settings.isRefreshingLaunchdStatus && !fixture.settings.isRefreshingObservabilityStatus
                && fixture.settings.embeddingProcess == .unmeasurable
                && fixture.settings.watcherHealth != nil && fixture.settings.backupSchedules.count == 3
        }
        XCTAssertEqual(fixture.settings.configFileURL, directory.appendingPathComponent("unused.env"))
        XCTAssertTrue(fixture.settings.configReadSucceeded)
        XCTAssertTrue(Set(["config", "observability", "watcher", "backup-file", "backup-list"]).isSubset(of: reads.names))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }

    func test_dashboard_refresh_stays_private_and_preserves_synthetic_stats() async {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let reads = Reads()
        let fixture = BrainBarDriveConsentCaptureFixture(directory: directory, now: now, didRead: { reads.record($0) })
        await settle { !fixture.settings.isRefreshingObservabilityStatus }
        let isPrivate = fixture.collector.databasePathForObservability == directory.appendingPathComponent("unused.db").path
        XCTAssertTrue(isPrivate, "RED must stop before a collector with unowned/default paths is refreshed")
        guard isPrivate else { return }
        fixture.collector.requestRefresh(force: true, trigger: .tabSwitch)
        await fixture.collector.waitForFixtureRefresh()
        XCTAssertNil(fixture.collector.lastFetchError)
        XCTAssertEqual(fixture.collector.stats.chunkCount, BrainBarDashboardFixture.stats.chunkCount)
        fixture.collector.selectTimeframe(windowMinutes: 180, isLive: false)
        await fixture.collector.waitForFixtureRefresh()
        XCTAssertNotNil(fixture.collector.windowedBucketsError)
        XCTAssertTrue(Set(["stats", "coverage", "window", "agents"]).isSubset(of: reads.names))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
        fixture.collector.stop()
    }

    func test_receipts_and_cadence_do_not_resolve_installed_files() async {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let fixture = BrainBarDriveConsentCaptureFixture(directory: directory, now: now)
        await settle { !fixture.settings.isRefreshingObservabilityStatus }
        XCTAssertFalse(fixture.receipts.isPersistent)
        XCTAssertEqual(fixture.cadence.interval, 300)
        XCTAssertNil(fixture.cadence.assumption)
        let receipt = BrainBarOperationReceipt(kind: .search, durationMillis: 142, count: 10, recordedAt: now)
        fixture.receipts.record(receipt)
        fixture.receipts.reload()
        XCTAssertEqual(fixture.receipts.search, receipt)
        _ = fixture.dashboard(panelState: BrainBarDashboardPanelState())
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }

    private func settle(_ complete: () -> Bool) async {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !complete(), ContinuousClock.now < deadline { await Task.yield() }
        XCTAssertTrue(complete(), "Owned synthetic refresh did not complete")
    }
}
