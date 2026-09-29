import XCTest
@testable import BrainBar

/// #966: the Dashboard said "Watcher flow needs attention" while Settings said "Watcher running".
/// The hero's `.stalled` fired on BrainBar's replay debt (deferred stores), which is not
/// watcher work, and no reason was shown anywhere.
@MainActor
final class BrainBarWatcherTruthSurfaceTests: XCTestCase {
    private let now = BrainBarDashboardFixture.fetchedAt

    private func healthyBackups() -> ObservabilityBackupStatus {
        let green = ObservabilityStatusLine(text: "verified", tone: .green)
        return .init(upload: green, snapshot: green, job: green, freshness: green, retention: green, archives: green, error: nil)
    }

    private func stats(
        replayDebt: Int,
        process: WatcherProcessProbeResult = .running(pid: 4242),
        health: WatcherHealthFileRead? = BrainBarDashboardFixture.healthyWatcherHealth
    ) -> DashboardStats {
        DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 120,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 0],
            recentAgentWriteBuckets: [0, 0, 0, 0],
            recentWatcherWriteBuckets: [0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0],
            activityWindowMinutes: 30,
            bucketCount: 4,
            pendingStoreQueueDepth: replayDebt,
            pendingStoreFlushQueueDepth: replayDebt,
            watcherHealth: health,
            watcherProcessProbeResult: process,
            watcherRecentDistinctChunkCount: 0,
            watcherFlowReadability: .readable
        )
    }

    func testReplayDebtNoLongerImplicatesTheWatcher() {
        let stats = stats(replayDebt: 5)
        XCTAssertGreaterThan(stats.replayDebtBreakdown.deduplicatedTotal, 0, "fixture must carry replay debt")
        let flow = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
        XCTAssertNotEqual(flow.lane(for: .jsonlWatcher).statusText, "STALLED")
        let hero = BrainBarHeroPresentation.derive(flow: flow, stats: stats, backupTruth: .measured(healthyBackups()))
        XCTAssertFalse(
            hero.healthReason.localizedCaseInsensitiveContains("watcher flow needs attention"),
            "replay debt is BrainBar's deferred-store queue, not watcher work: \(hero.healthReason)"
        )
        XCTAssertNotEqual(hero.healthVerdict, "Needs attention", hero.healthReason)
        XCTAssertEqual(hero.healthVerdict, "Healthy", "a healthy, idle watcher with replay debt is healthy: \(hero.healthReason)")
        XCTAssertEqual(flow.watcherStatus, .running(heartbeatAt: now.addingTimeInterval(-70)))
    }

    /// Every surface renders the same status with the same reason line (#966 acceptance:
    /// "the footer never contradicts the dashboard").
    func testEverySurfaceShowsTheSameWatcherTruth() {
        let firstFailure = now.addingTimeInterval(-7_200)
        let degraded = WatcherHealthFileRead.readable(WatcherHealthFile(
            updatedAt: now.addingTimeInterval(-70),
            pollCount: 72,
            alertReasons: ["file_ingestion_failure"],
            fileIngestionFailureCount: 2,
            earliestFileIngestionFailureAt: firstFailure
        ))
        let cases: [(String, WatcherProcessProbeResult, WatcherHealthFileRead?, BrainLayerLaunchdLoadState, String)] = [
            ("running", .running(pid: 1), BrainBarDashboardFixture.healthyWatcherHealth, .running, "Watcher running"),
            ("degraded", .running(pid: 1), degraded, .running, "Watcher needs attention"),
            ("stopped", .absent, BrainBarDashboardFixture.healthyWatcherHealth, .unloaded, "Watcher stopped"),
            ("unknown", .running(pid: 1), .missing(path: "/data/watcher-health.json"), .running, "Watcher status unknown"),
        ]
        for (name, process, health, loadState, title) in cases {
            let stats = stats(replayDebt: 0, process: process, health: health)
            let flow = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
            let hero = BrainBarHeroPresentation.derive(flow: flow, stats: stats, backupTruth: .measured(healthyBackups()))
            let lane = flow.lane(for: .jsonlWatcher)

            var config = BrainLayerConfig.defaultConfig
            config.launchdJobs[.watch] = BrainLayerLaunchdJobSetting(enabled: true, loadState: loadState)
            let settingsStatus = WatcherHealthStatus.derive(
                launchd: WatcherLaunchdEvidence(setting: config.launchdJobs[.watch], loadState: loadState),
                file: health,
                now: now
            )
            let footer = BrainBarSettingsFooterPresentation(config: config, watcher: settingsStatus, now: now)
            let ingest = BrainLayerLaunchdJobGroup.ingest.status(
                settings: config.launchdJobs,
                observations: [.watch: .stateOnly(loadState), .index: .stateOnly(.loaded)],
                formatDate: { _ in "t" },
                watcher: settingsStatus,
                now: now
            )

            XCTAssertEqual(flow.watcherStatus.title, title, name)
            XCTAssertEqual(footer.state.title, title, "\(name): footer must match the dashboard")
            let reason = flow.watcherStatusReason
            // Settings observes launchd itself (loaded/unloaded/disabled) while the Dashboard probes
            // the process, so a stopped job may carry a more specific launchd detail in Settings.
            // The verdict and every other reason are identical.
            if name == "stopped" {
                XCTAssertTrue(reason?.hasPrefix("Watcher is not running") == true, reason ?? "nil")
                XCTAssertTrue(footer.detail?.hasPrefix("Watcher is not running") == true, footer.detail ?? "nil")
            } else {
                XCTAssertEqual(footer.detail, reason, "\(name): footer reason")
            }
            XCTAssertEqual(ingest.attentionReason, footer.detail, "\(name): Settings Ingest and footer share one reason")
            if let reason {
                XCTAssertEqual(lane.lastEventText, reason, "\(name): lane reason")
                XCTAssertEqual(hero.healthReason, reason, "\(name): hero reason")
            } else {
                XCTAssertEqual(hero.healthVerdict, "Healthy", name)
                XCTAssertEqual(ingest.health, .healthy, name)
            }
        }
    }

    func testDegradedReasonNamesWhatSinceWhenAndWhatToDo() {
        let health = WatcherHealthFileRead.readable(WatcherHealthFile(
            updatedAt: now.addingTimeInterval(-70),
            pollCount: 72,
            alertReasons: ["file_ingestion_failure"],
            fileIngestionFailureCount: 2,
            earliestFileIngestionFailureAt: now.addingTimeInterval(-7_200)
        ))
        let flow = DashboardFlowSummary.derive(daemon: nil, stats: stats(replayDebt: 0, health: health), now: now)
        XCTAssertEqual(flow.lane(for: .jsonlWatcher).statusText, "NEEDS ATTENTION")
        XCTAssertEqual(
            flow.watcherStatusReason,
            "2 transcript files could not be ingested · since 2h ago · See file_ingestion_failures in watcher-health.json"
        )
    }

    func testUnknownIngestIsNotReportedAsAFailure() {
        var config = BrainLayerConfig.defaultConfig
        config.launchdJobs[.watch] = BrainLayerLaunchdJobSetting(enabled: true, loadState: .running)
        let unknown = WatcherHealthStatus.unknown(reason: "Watcher is running, but its health file is missing at /x.")
        let ingest = BrainLayerLaunchdJobGroup.ingest.status(
            settings: config.launchdJobs,
            observations: [.watch: .stateOnly(.running), .index: .stateOnly(.loaded)],
            formatDate: { _ in "t" },
            watcher: unknown,
            now: now
        )
        XCTAssertEqual(ingest.health, .unknown)
        XCTAssertEqual(ingest.health.title, "Status unknown")
        XCTAssertEqual(ingest.attentionReason, "Watcher is running, but its health file is missing at /x.")
    }

    func testSettingsViewModelFooterReadsTheSameHealthFile() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("watcher-health.json")
        let store = BrainLayerConfigStore(configURL: directory.appendingPathComponent("brainlayer.env"))

        // launchd says running, but the health file is missing: honest unknown, never "Watcher running".
        let viewModel = BrainBarSettingsViewModel(
            store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [.watch: .running]),
            initialLaunchdStates: [.watch: .running],
            refreshStatusOnLoad: false,
            now: { [now] in now },
            watcherHealthURL: url
        )
        viewModel.refreshWatcherHealth()
        for _ in 0..<200 where viewModel.watcherHealth == nil { try await Task.sleep(nanoseconds: 10_000_000) }
        XCTAssertEqual(viewModel.watcherHealth, .missing(path: url.path))
        XCTAssertEqual(viewModel.footerPresentation.state.title, "Watcher status unknown")
        XCTAssertTrue(viewModel.footerPresentation.detail?.contains(url.path) == true)
    }
}
