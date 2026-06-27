import AppKit
import XCTest
@testable import BrainBar
@testable import BrainBarLifecycle

final class DashboardTests: XCTestCase {
    private final class WatchdogTestState: @unchecked Sendable {
        private let lock = NSLock()
        private var observedPIDs: [pid_t] = [111]
        private var signals: [(pid_t, Int32)] = []
        private var relaunchCount = 0

        func processProvider() -> [pid_t] {
            lock.lock()
            defer { lock.unlock() }
            return observedPIDs
        }

        func recordTermination(pid: pid_t, signal: Int32) {
            lock.lock()
            signals.append((pid, signal))
            observedPIDs = [222]
            lock.unlock()
        }

        func recordRelaunch() {
            lock.lock()
            relaunchCount += 1
            lock.unlock()
        }

        func snapshot() -> (signals: [(pid_t, Int32)], relaunchCount: Int) {
            lock.lock()
            defer { lock.unlock() }
            return (signals, relaunchCount)
        }
    }

    private var db: BrainDatabase!
    private var tempDBPath: String!
    private var fallbackReplayRoot: URL!
    private var restoreFallbackReplayRoot: (() -> Void)!
    private static let fractionalTimestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXXXX"
        return formatter
    }()

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-dashboard-\(UUID().uuidString).db"
        fallbackReplayRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("brainbar-dashboard-fallback-root-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: fallbackReplayRoot, withIntermediateDirectories: true)
        restoreFallbackReplayRoot = setDashboardFallbackReplayGitsRoot(fallbackReplayRoot)
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        restoreFallbackReplayRoot?()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        if let fallbackReplayRoot {
            try? FileManager.default.removeItem(at: fallbackReplayRoot)
        }
        super.tearDown()
    }

    func testDashboardStatsSummarizesChunkAndEnrichmentCounts() throws {
        try db.insertChunk(
            id: "dash-1",
            content: "Fresh chunk waiting for enrichment",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )
        try db.insertChunk(
            id: "dash-2",
            content: "Already enriched chunk",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 7
        )
        db.exec("UPDATE chunks SET enriched_at = datetime('now'), enrich_status = 'success' WHERE id = 'dash-2'")

        let stats = try db.dashboardStats(activityWindowMinutes: 30, bucketCount: 6)

        XCTAssertEqual(stats.chunkCount, 2)
        XCTAssertEqual(stats.enrichedChunkCount, 1)
        XCTAssertEqual(stats.pendingEnrichmentCount, 1)
        XCTAssertEqual(stats.enrichmentPercent, 50.0, accuracy: 0.001)
        XCTAssertEqual(stats.recentActivityBuckets.count, 6)
        XCTAssertEqual(stats.recentEnrichmentBuckets.count, 6)
        XCTAssertGreaterThanOrEqual(stats.recentActivityBuckets.reduce(0, +), 2)
        XCTAssertGreaterThanOrEqual(stats.recentEnrichmentBuckets.reduce(0, +), 1)
        XCTAssertGreaterThan(stats.databaseSizeBytes, 0)
    }

    func testDashboardStatsReportsPerSignalCoverageAndBacklogs() throws {
        for id in ["signal-1", "signal-2", "signal-3", "signal-archived"] {
            try db.insertChunk(
                id: id,
                content: "Signal coverage fixture \(id)",
                sessionId: "dashboard",
                project: "brainlayer",
                contentType: "assistant_text",
                importance: 5
            )
        }
        db.exec("UPDATE chunks SET archived = 1, archived_at = '2026-06-19T00:00:00Z', status = 'archived' WHERE id = 'signal-archived'")
        db.exec("CREATE TABLE IF NOT EXISTS chunk_vectors_rowids(id TEXT PRIMARY KEY)")
        db.exec("INSERT INTO chunk_vectors_rowids(id) VALUES ('signal-1'), ('signal-2')")
        db.exec("DELETE FROM chunks_fts WHERE chunk_id = 'signal-3'")
        db.exec("DELETE FROM chunks_fts_trigram WHERE chunk_id = 'signal-2'")

        let stats = try db.dashboardStats(activityWindowMinutes: 30, bucketCount: 6)

        XCTAssertEqual(stats.chunkCount, 4)
        XCTAssertEqual(stats.signalEligibleChunkCount, 3)
        XCTAssertEqual(stats.vectorIndexedChunkCount, 2)
        XCTAssertEqual(stats.ftsIndexedChunkCount, 2)
        XCTAssertEqual(stats.trigramIndexedChunkCount, 2)
        XCTAssertEqual(stats.vectorBacklogCount, 1)
        XCTAssertEqual(stats.ftsBacklogCount, 1)
        XCTAssertEqual(stats.trigramBacklogCount, 1)
        XCTAssertEqual(stats.vectorCoveragePercent, 66.666, accuracy: 0.01)
        XCTAssertEqual(stats.ftsCoveragePercent, 66.666, accuracy: 0.01)
        XCTAssertEqual(stats.trigramCoveragePercent, 66.666, accuracy: 0.01)
    }

    func testDashboardStatsTreatsLegacyFtsTableWithoutChunkIdAsUnknownCoverage() throws {
        try db.insertChunk(
            id: "legacy-fts-1",
            content: "Legacy FTS fixture",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )
        db.exec("DROP TABLE chunks_fts")
        db.exec("CREATE VIRTUAL TABLE chunks_fts USING fts5(content)")
        db.exec("INSERT INTO chunks_fts(content) VALUES ('Legacy FTS fixture')")

        let stats = try db.dashboardStats(activityWindowMinutes: 30, bucketCount: 6)

        XCTAssertEqual(stats.signalEligibleChunkCount, 1)
        XCTAssertEqual(stats.ftsIndexedChunkCount, 0)
        XCTAssertEqual(stats.ftsBacklogCount, 1)
    }

    func testDashboardStatsReadsWatcherHealthSnapshot() throws {
        let healthURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("watcher-health-\(UUID().uuidString).json")
        try """
        {
          "alerting": true,
          "files_tracked": 7,
          "max_offset_lag_bytes": 2097152,
          "active_jsonl_entries_per_minute": 44.0,
          "db_realtime_inserts_per_minute": 3.0
        }
        """.write(to: healthURL, atomically: true, encoding: .utf8)
        let restoreHealthPath = setDashboardWatcherHealthPath(healthURL)
        defer {
            restoreHealthPath()
            try? FileManager.default.removeItem(at: healthURL)
        }

        let stats = try db.dashboardStats(activityWindowMinutes: 30, bucketCount: 6)

        XCTAssertEqual(stats.watcherHealth?.filesTracked, 7)
        XCTAssertEqual(stats.watcherHealth?.alerting, true)
        XCTAssertEqual(stats.watcherHealth?.summaryText, "lag 2 MB")
    }

    func testDashboardStatsRecentEnrichmentCountSharesBucketSource() {
        let stats = DashboardStats(
            chunkCount: 12,
            enrichedChunkCount: 8,
            pendingEnrichmentCount: 4,
            enrichmentPercent: 66.7,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [1, 0, 2, 0],
            recentEnrichmentBuckets: [0, 2, 0, 3]
        )

        XCTAssertEqual(stats.recentEnrichmentCount, stats.recentEnrichmentBuckets.reduce(0, +))
    }

    func testDashboardStatsTrailingFiveMinuteCountsUseTrueFiveMinuteWindow() throws {
        let offsets: [(String, TimeInterval)] = [
            ("dash-enrich-2m", -120),
            ("dash-enrich-4m", -240),
            ("dash-enrich-6m", -360),
            ("dash-enrich-45m", -2_700),
        ]
        for (id, offset) in offsets {
            try db.insertChunk(
                id: id,
                content: "Enrichment fixture \(id)",
                sessionId: "dashboard",
                project: "brainlayer",
                contentType: "assistant_text",
                importance: 5
            )
            db.exec("""
                UPDATE chunks
                SET created_at = datetime('now', '\(Int(offset)) seconds'),
                    enriched_at = datetime('now', '\(Int(offset)) seconds'),
                    enrich_status = 'success'
                WHERE id = '\(id)'
            """)
        }

        let stats = try db.dashboardStats(activityWindowMinutes: 60, bucketCount: 12)

        XCTAssertEqual(stats.recentWriteFiveMinuteCount, 2)
        XCTAssertEqual(stats.recentEnrichmentFiveMinuteCount, 2)
        XCTAssertEqual(stats.recentEnrichmentBuckets.reduce(0, +), 4)
    }

    func testDashboardStatsBucketsEventsOnFixedWallClockWindowEndingNow() throws {
        let fixtures: [(id: String, offset: TimeInterval)] = [
            ("dash-wallclock-52m", -52 * 60),
            ("dash-wallclock-27m-a", -27 * 60),
            ("dash-wallclock-27m-b", -27 * 60),
            ("dash-wallclock-4m", -4 * 60),
            ("dash-wallclock-outside", -65 * 60),
        ]
        for fixture in fixtures {
            try db.insertChunk(
                id: fixture.id,
                content: "Wall-clock enrichment fixture \(fixture.id)",
                sessionId: "dashboard",
                project: "brainlayer",
                contentType: "assistant_text",
                importance: 5
            )
            db.exec("""
                UPDATE chunks
                SET created_at = datetime('now', '\(Int(fixture.offset)) seconds'),
                    enriched_at = datetime('now', '\(Int(fixture.offset)) seconds'),
                    enrich_status = 'success'
                WHERE id = '\(fixture.id)'
            """)
        }

        let stats = try db.dashboardStats(activityWindowMinutes: 60, bucketCount: 12)

        XCTAssertEqual(stats.recentEnrichmentBuckets, [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1])
        XCTAssertEqual(stats.recentEnrichmentFiveMinuteCount, 1)
        XCTAssertEqual(stats.recentEnrichmentCount, 4)
    }

    func testDashboardStatsSplitsLiveWritesIntoAgentStoresAndJSONLWatcherPaths() throws {
        let fixtures: [(id: String, source: String, offset: TimeInterval)] = [
            ("agent-27m", "mcp", -27 * 60),
            ("agent-manual-4m", "manual", -4 * 60),
            ("agent-precompact-4m", "precompact-hook", -4 * 60),
            ("quick-capture-4m", "quick-capture", -4 * 60),
            ("watcher-52m", "realtime_watcher", -52 * 60),
            ("watcher-27m", "realtime_watcher", -27 * 60),
            ("watcher-4m", "realtime", -4 * 60),
            ("claude-code-4m", "claude_code", -4 * 60),
            ("digest-4m", "digest", -4 * 60),
            ("youtube-4m", "youtube", -4 * 60),
            ("whatsapp-4m", "whatsapp", -4 * 60),
        ]
        for fixture in fixtures {
            _ = try db.store(
                content: "Write source split fixture \(fixture.id)",
                tags: ["dashboard"],
                importance: 5,
                source: fixture.source,
                chunkID: fixture.id
            )
            db.exec("""
                UPDATE chunks
                SET created_at = datetime('now', '\(Int(fixture.offset)) seconds')
                WHERE id = '\(fixture.id)'
            """)
        }

        let stats = try db.dashboardStats(activityWindowMinutes: 60, bucketCount: 12)

        XCTAssertEqual(stats.recentAgentWriteBuckets, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3])
        XCTAssertEqual(stats.recentWatcherWriteBuckets, [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        XCTAssertEqual(stats.recentActivityBuckets, [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 8])
    }

    func testDashboardFlowLabelsWriteSeriesBySourcePath() {
        let stats = DashboardStats(
            chunkCount: 9,
            enrichedChunkCount: 0,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 0,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 0,
            recentActivityBuckets: [1, 2, 3],
            recentAgentWriteBuckets: [1, 0, 1],
            recentWatcherWriteBuckets: [0, 2, 1],
            recentEnrichmentBuckets: [0, 0, 0],
            activityWindowMinutes: 15
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: Date(timeIntervalSince1970: 1_764_236_400))

        XCTAssertEqual(summary.ingress.primarySeriesLabel, "Agent stores")
        XCTAssertEqual(summary.ingress.secondarySeriesLabel, "JSONL watcher")
        XCTAssertNil(summary.ingress.tertiarySeriesLabel)
        XCTAssertTrue(summary.ingress.tertiaryValues.isEmpty)
    }

    func testDashboardFlowUsesDistinctV1WriteSeriesPalette() {
        let stats = DashboardStats(
            chunkCount: 9,
            enrichedChunkCount: 0,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 0,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 0,
            recentActivityBuckets: [1, 2, 3],
            recentAgentWriteBuckets: [1, 0, 1],
            recentWatcherWriteBuckets: [0, 2, 1],
            recentEnrichmentBuckets: [0, 0, 0],
            activityWindowMinutes: 15
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: Date(timeIntervalSince1970: 1_764_236_400))

        XCTAssertEqual(summary.ingress.accentColor, BrainBarDesignTokens.Colors.seriesAgent)
        XCTAssertEqual(summary.ingress.secondaryAccentColor, BrainBarDesignTokens.Colors.seriesWatcher)
        XCTAssertNil(summary.ingress.tertiaryAccentColor)
    }

    func testDashboardStatsComputesVectorETAFromBacklogAndNetDrain() {
        let stats = DashboardStats(
            chunkCount: 10_105,
            enrichedChunkCount: 0,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 0,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [10],
            recentEnrichmentBuckets: [65],
            activityWindowMinutes: 1,
            bucketCount: 1,
            signalEligibleChunkCount: 10_105,
            vectorIndexedChunkCount: 0
        )

        XCTAssertEqual(stats.vectorNetDrainRatePerHour, 3_300, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(stats.vectorBacklogETAHours), 10_105.0 / 3_300.0, accuracy: 0.001)
    }

    func testDashboardStatsSamplesPendingStoreQueueBeforeReadTransaction() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/BrainDatabase.swift")
        let methodRange = try XCTUnwrap(source.range(of: "func dashboardStats("))
        let methodSource = source[methodRange.lowerBound...]
        let queueSnapshotRange = try XCTUnwrap(methodSource.range(of: "let pendingStoreFlushQueue = pendingStoreQueueSnapshot()"))
        let transactionRange = try XCTUnwrap(methodSource.range(of: "try withReadTransaction"))

        XCTAssertLessThan(
            queueSnapshotRange.lowerBound,
            transactionRange.lowerBound,
            "dashboardStats must not acquire the pending-store file lock while holding the SQLite read transaction."
        )
    }

    func testBrainBarHeaderRefreshControlsObserveStatsCollector() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")

        XCTAssertTrue(
            source.contains("private struct BrainBarHeaderRefreshControls: View"),
            "Refresh UI should live in a child view so it can observe StatsCollector directly."
        )
        XCTAssertTrue(
            source.contains("@ObservedObject private var collector: StatsCollector")
                || source.contains("@ObservedObject var collector: StatsCollector"),
            "Refresh UI must observe StatsCollector so spinner/disabled state update without unrelated header redraws."
        )
    }

    func testStatsCollectorDoesNotKeepUnusedDatabaseConnection() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/Dashboard/StatsCollector.swift")

        XCTAssertFalse(
            source.contains("private let database: BrainDatabase"),
            "StatsCollector refreshes through short-lived background handles and should not keep an unused DB connection open."
        )
        XCTAssertFalse(
            source.contains("database.close()"),
            "Removing the retained DB handle also removes the matching close call."
        )
    }

    func testDashboardShowsLoadingUntilFirstStatsFetchCompletes() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")

        XCTAssertTrue(source.contains("private struct BrainBarDashboardContent: View"))
        XCTAssertTrue(source.contains("@ObservedObject var collector: StatsCollector"))
        XCTAssertTrue(source.contains("BrainBarDashboardContent("))
        XCTAssertTrue(source.contains("collector.lastDataFetchedAt == nil"))
        XCTAssertTrue(source.contains("Connecting to daemon and loading dashboard data"))
    }

    func testDashboardRendersPerSignalCoverageSection() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")

        XCTAssertTrue(source.contains("BrainBarSignalCoveragePanel"))
        XCTAssertTrue(source.contains("see under the hood"))
        XCTAssertTrue(source.contains("@State private var signalCoverageExpanded"))
        XCTAssertTrue(source.contains("@State private var vectorSignalDetailExpanded"))
        XCTAssertFalse(source.contains("BrainBarSectionLabel(\"Signal Coverage\")"))
        XCTAssertFalse(source.contains("Coverage is counted per retrieval signal"))
        XCTAssertTrue(source.contains("Vector"))
        XCTAssertTrue(source.contains("FTS5"))
        XCTAssertTrue(source.contains("Trigram"))
        XCTAssertTrue(source.contains("signalEligibleChunkCount"))
        XCTAssertTrue(source.contains("vectorBacklogCount"))
        XCTAssertTrue(source.contains("ftsBacklogCount"))
        XCTAssertTrue(source.contains("trigramBacklogCount"))
    }

    func testVectorSignalDetailMountsAtRootToEscapePipelineAndScrollClips() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")
        let bodyRange = try XCTUnwrap(source.range(of: "var body: some View"))
        let pipelinePanelRange = try XCTUnwrap(source.range(of: "private func pipelinePanel"))
        // Redesign (feat/brainbar-dashboard-redesign): the old single `writesCard`
        // helper was split into `writeCardsBand` (+ per-series card helpers). The
        // band helper is the first function after `pipelinePanel`, so it remains
        // the correct lower bound for slicing the pipeline-panel source.
        let writesCardRange = try XCTUnwrap(source[pipelinePanelRange.upperBound...].range(of: "private func writeCardsBand"))
        let bodySource = String(source[bodyRange.lowerBound..<pipelinePanelRange.lowerBound])
        let pipelinePanelSource = String(source[pipelinePanelRange.lowerBound..<writesCardRange.lowerBound])
        let signalColumnRange = try XCTUnwrap(source.range(of: "private func signalColumn"))
        let coverageModelRange = try XCTUnwrap(source[signalColumnRange.upperBound...].range(of: "private struct BrainBarSignalCoverage"))
        let signalColumnSource = String(source[signalColumnRange.lowerBound..<coverageModelRange.lowerBound])

        XCTAssertFalse(
            source.contains("overlayPreferenceValue(VectorRowAnchorKey.self)"),
            "Vector detail should not depend on a preference anchor that can fail to resolve while the selected row styling still changes."
        )
        XCTAssertFalse(
            source.contains("anchorPreference(key: VectorRowAnchorKey.self"),
            "The Vector row click path should mount the detail directly instead of only emitting an anchor preference."
        )
        XCTAssertTrue(
            bodySource.contains(".coordinateSpace(name: BrainBarVectorSignalCoordinateSpace.root)"),
            "The root GeometryReader should define the coordinate space used to position the unclipped Vector detail."
        )
        XCTAssertTrue(
            bodySource.contains(".overlay(alignment: .topLeading)"),
            "The Vector detail must be hoisted to a root overlay so the pipeline panel and ScrollView cannot clip it."
        )
        XCTAssertTrue(
            bodySource.contains("BrainBarVectorSignalDetail(signal: vectorSignal, compact: layout.compactCards)"),
            "The root overlay should own the Vector detail float."
        )
        XCTAssertTrue(
            bodySource.contains(".zIndex(vectorSignalDetailExpanded ? 30 : 0)"),
            "The hoisted Vector detail needs a high zIndex above every sibling pipeline card."
        )
        XCTAssertFalse(
            pipelinePanelSource.contains("BrainBarVectorSignalDetail(signal: vectorSignal, compact: layout.compactCards)"),
            "The Vector detail cannot remain inside the pipeline panel overlay because that panel clips the float."
        )
        XCTAssertFalse(
            pipelinePanelSource.contains(".zIndex(vectorSignalDetailExpanded ? 1 : 0)"),
            "The old in-panel zIndex workaround should be removed once the float lives at root level."
        )
        XCTAssertFalse(
            pipelinePanelSource.contains(".padding(.top, max(0, 20 - layout.gridSpacing))"),
            "The old in-panel spacing workaround should be removed once the float lives at root level."
        )
        XCTAssertFalse(
            signalColumnSource.contains("BrainBarVectorSignalDetail"),
            "The Vector detail cannot live in the signal column overlay because later pipeline siblings paint over that layer."
        )
        XCTAssertTrue(
            signalColumnSource.contains("BrainBarVectorSignalRootFrameKey"),
            "The Vector signal row should emit a root-space frame for the unclipped overlay."
        )
    }

    func testVectorSignalDetailUsesDerivedZLiftNotHeldState() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")

        XCTAssertFalse(source.contains("@State private var isFloatLifted"))
        XCTAssertFalse(source.contains("DispatchQueue.main.asyncAfter(deadline: .now() + 0.27)"))
        XCTAssertTrue(source.contains(".zIndex(vectorSignalDetailExpanded ? 30 : 0)"))
    }

    func testDashboardCardTopHighlightIsClippedInsideRoundedCorners() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")
        let styleRange = try XCTUnwrap(source.range(of: "private struct BrainBarDashboardCardStyle"))
        let nextStructRange = try XCTUnwrap(source[styleRange.upperBound...].range(of: "private struct BrainBarFlowStatusPill"))
        let styleSource = String(source[styleRange.lowerBound..<nextStructRange.lowerBound])

        XCTAssertFalse(
            styleSource.contains(".overlay(alignment: .top) {\n                Rectangle()"),
            "A full-width top Rectangle creates the flat divider hat across rounded card corners."
        )
        XCTAssertTrue(
            styleSource.contains(".strokeBorder(Color.brainBarBorderSoft, lineWidth: 1)"),
            "The card border should use strokeBorder so it remains correct after clipping."
        )
        XCTAssertTrue(
            styleSource.contains(".padding(.horizontal, max(10, cornerRadius * 0.70))\n            }\n            .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))\n            .shadow"),
            "The shared card top highlight should be inset clear of rounded corners."
        )
    }

    func testBrainBarHeaderExposesRestartAndQuitControls() throws {
        let rootSource = try brainBarSourceFile("Sources/BrainBar/BrainBarWindowRootView.swift")
        let processSource = try brainBarSourceFile("Sources/BrainBar/BrainBarProcessControl.swift")

        XCTAssertTrue(rootSource.contains("BrainBarAppControlMenu()"))
        XCTAssertTrue(rootSource.contains("Restart BrainBar"))
        XCTAssertTrue(rootSource.contains("Quit BrainBar"))
        XCTAssertTrue(processSource.contains("static func restart"))
        XCTAssertTrue(processSource.contains("static func quit"))
        XCTAssertTrue(processSource.contains("BrainBarRestartHandoff.markRestartingProcess()"))
        XCTAssertTrue(processSource.contains("BrainBarRestartHandoff.clear()"))
        XCTAssertTrue(processSource.contains("FileManager.default.fileExists"))
        XCTAssertTrue(processSource.contains("URL(fileURLWithPath: \"/usr/bin/open\")"))
        XCTAssertFalse(processSource.contains("URL(fileURLWithPath: \"/bin/sh\")"))
    }

    @MainActor
    func testDashboardPanelUsesKeyWindowContractAndSettingsDismissSuppression() throws {
        let controller = BrainBarDashboardPanelController(runtime: BrainBarRuntime())
        let panel = controller.panelForTesting
        let panelSource = try brainBarSourceFile("Sources/BrainBar/BrainBarDashboardPanelController.swift")
        let settingsSource = try brainBarSourceFile("Sources/BrainBar/BrainBarSettingsActions.swift")

        XCTAssertTrue(panel.canBecomeKey)
        XCTAssertFalse(panel.canBecomeMain)
        XCTAssertFalse(panel.becomesKeyOnlyIfNeeded)
        XCTAssertTrue(panelSource.contains("func windowWillClose(_ notification: Notification)"))
        XCTAssertTrue(panelSource.contains("BrainBarSettingsActions.suppressDashboardResignDismiss"))
        XCTAssertTrue(settingsSource.contains("private(set) static var suppressDashboardResignDismiss"))
        XCTAssertTrue(settingsSource.contains("suppressDashboardResignDismiss = true"))
        XCTAssertTrue(settingsSource.contains("suppressDashboardResignDismiss = false"))
    }

    func testRestartHandoffAllowsOnlyMatchingFreshExistingInstance() throws {
        let markerPath = NSTemporaryDirectory() + "brainbar-restart-handoff-\(UUID().uuidString)"
        let timestamp = Date(timeIntervalSince1970: 1_000)

        BrainBarRestartHandoff.markRestartingProcess(pid: 42, at: timestamp, path: markerPath)

        XCTAssertFalse(
            BrainBarRestartHandoff.consumeIfMatches(
                existingPID: 41,
                now: Date(timeIntervalSince1970: 1_005),
                path: markerPath
            )
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: markerPath))
        XCTAssertTrue(
            BrainBarRestartHandoff.consumeIfMatches(
                existingPID: 42,
                now: Date(timeIntervalSince1970: 1_005),
                path: markerPath
            )
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: markerPath))

        BrainBarRestartHandoff.markRestartingProcess(pid: 42, at: timestamp, path: markerPath)
        XCTAssertFalse(
            BrainBarRestartHandoff.consumeIfMatches(
                existingPID: 42,
                now: Date(timeIntervalSince1970: 1_020),
                maxAge: 10,
                path: markerPath
            )
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: markerPath))
    }

    func testLegacyPopoverSparklineUsesLastFetchAnchor() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/Dashboard/StatusPopoverView.swift")

        XCTAssertTrue(source.contains("fetchedAt: collector.lastDataFetchedAt ?? Date()"))
    }

    func testBrainBarLifecycleWatchdogWiresUIAndDaemonHeartbeats() throws {
        let watchdog = try brainBarSourceFile("Sources/BrainBarLifecycle/BrainBarLifecycleWatchdog.swift")
        let app = try brainBarSourceFile("Sources/BrainBar/BrainBarApp.swift")
        let server = try brainBarSourceFile("Sources/BrainBar/BrainBarServer.swift")
        let daemon = try brainBarSourceFile("Sources/BrainBarDaemon/BrainBarDaemonMain.swift")

        XCTAssertTrue(watchdog.contains("brainbar-ui.heartbeat"))
        XCTAssertTrue(watchdog.contains("brainbar-daemon.heartbeat"))
        XCTAssertTrue(watchdog.contains("SIGTERM"))
        XCTAssertTrue(watchdog.contains("SIGKILL"))
        XCTAssertTrue(watchdog.contains("launchctl"))
        XCTAssertTrue(app.contains("startUIHeartbeat"))
        XCTAssertTrue(app.contains("startDaemonWatchdog"))
        XCTAssertTrue(server.contains("startDaemonHeartbeatOnQueue"))
        XCTAssertTrue(daemon.contains("startUIWatchdog"))
    }

    func testWatchdogPolicyMarksMissingAndStaleHeartbeatUnhealthy() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-watchdog-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let missing = directory.appendingPathComponent("missing.heartbeat").path
        XCTAssertTrue(BrainBarLifecycleWatchdog.isHeartbeatStale(atPath: missing, now: Date(), timeout: 5))

        let heartbeat = directory.appendingPathComponent("present.heartbeat")
        FileManager.default.createFile(atPath: heartbeat.path, contents: Data("ok".utf8))
        let staleDate = Date(timeIntervalSince1970: 100)
        try FileManager.default.setAttributes([.modificationDate: staleDate], ofItemAtPath: heartbeat.path)

        XCTAssertTrue(
            BrainBarLifecycleWatchdog.isHeartbeatStale(
                atPath: heartbeat.path,
                now: staleDate.addingTimeInterval(6),
                timeout: 5
            )
        )
        XCTAssertFalse(
            BrainBarLifecycleWatchdog.isHeartbeatStale(
                atPath: heartbeat.path,
                now: staleDate.addingTimeInterval(4),
                timeout: 5
            )
        )
    }

    func testWatchdogTerminatesOnlyOriginallyStaleProcessBeforeRelaunch() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-watchdog-restart-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let heartbeatPath = directory.appendingPathComponent("missing.heartbeat").path
        let relaunched = expectation(description: "watchdog relaunches stale process")
        let state = WatchdogTestState()

        let watchdog = BrainBarLifecycleWatchdog(
            configuration: .init(
                watchedName: "TestBrainBar",
                heartbeatPath: heartbeatPath,
                staleTimeout: 0.01,
                checkInterval: 0.01,
                terminateGraceInterval: 0.02,
                relaunchCommand: .openBundle("/tmp/TestBrainBar.app")
            ),
            processProvider: {
                state.processProvider()
            },
            terminateProcess: { pid, signal in
                state.recordTermination(pid: pid, signal: signal)
            },
            relaunch: { _ in
                state.recordRelaunch()
                relaunched.fulfill()
            }
        )

        watchdog.start()
        wait(for: [relaunched], timeout: 1)
        watchdog.stop()

        let snapshot = state.snapshot()

        XCTAssertEqual(snapshot.signals.map(\.0), [111, 111])
        XCTAssertEqual(snapshot.signals.map(\.1), [SIGTERM, SIGKILL])
        XCTAssertEqual(snapshot.relaunchCount, 1)
    }

    func testDashboardLatestEventFallbackUsesSQLMaxWithoutTextOrderingLimit() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/BrainDatabase.swift")
        let methodRange = try XCTUnwrap(source.range(of: "private func latestIndexedTimestampEpoch("))
        let nextMethodRange = try XCTUnwrap(source[methodRange.upperBound...].range(of: "private func latestIndexedEpochSince("))
        let methodSource = source[methodRange.lowerBound..<nextMethodRange.lowerBound]

        XCTAssertTrue(
            methodSource.contains("SELECT MAX(event_epoch)"),
            "Fallback latest-event lookup should compute the max normalized epoch in SQL."
        )
        XCTAssertFalse(
            methodSource.contains("ORDER BY \\(column) DESC"),
            "Fallback must not text-sort raw timestamp strings before comparing normalized epochs."
        )
        XCTAssertFalse(
            methodSource.contains("LIMIT 1000"),
            "Fallback must not cap rows before finding the max normalized epoch."
        )
    }

    func testDashboardStatsCountsTerminalEnrichmentStatusesAsCovered() throws {
        try db.insertChunk(
            id: "dash-success",
            content: "Successfully enriched chunk",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 7
        )
        try db.insertChunk(
            id: "dash-duplicate",
            content: "Terminal duplicate marker",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )
        try db.insertChunk(
            id: "dash-noise",
            content: "Terminal noise marker",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )
        db.exec("""
            UPDATE chunks
            SET enriched_at = datetime('now', '-5 minutes'),
                enrich_status = 'success'
            WHERE id = 'dash-success'
        """)
        db.exec("""
            UPDATE chunks
            SET enriched_at = 'skipped:duplicate',
                enrich_status = 'duplicate'
            WHERE id = 'dash-duplicate'
        """)
        db.exec("""
            UPDATE chunks
            SET enriched_at = NULL,
                enrich_status = 'noise'
            WHERE id = 'dash-noise'
        """)

        let stats = try db.dashboardStats(activityWindowMinutes: 30, bucketCount: 6)

        XCTAssertEqual(stats.enrichedChunkCount, 3)
        XCTAssertEqual(stats.pendingEnrichmentCount, 0)
        XCTAssertEqual(stats.enrichmentPercent, 100.0, accuracy: 0.001)
        XCTAssertEqual(stats.recentEnrichmentBuckets.reduce(0, +), 1)
        let lastEnrichedAt = try XCTUnwrap(stats.lastEnrichedAt)
        XCTAssertLessThan(abs(lastEnrichedAt.timeIntervalSinceNow + 300), 10)
    }

    func testDashboardStatsReturnsZeroPercentForEmptyDatabase() throws {
        let stats = try db.dashboardStats(activityWindowMinutes: 15, bucketCount: 4)

        XCTAssertEqual(stats.chunkCount, 0)
        XCTAssertEqual(stats.enrichedChunkCount, 0)
        XCTAssertEqual(stats.pendingEnrichmentCount, 0)
        XCTAssertEqual(stats.enrichmentPercent, 0.0, accuracy: 0.001)
        XCTAssertEqual(stats.recentActivityBuckets, [0, 0, 0, 0])
        XCTAssertEqual(stats.recentEnrichmentBuckets, [0, 0, 0, 0])
    }

    func testEnrichmentStatsCountsAnyTerminalStatusAsCovered() throws {
        try db.insertChunk(
            id: "stats-success",
            content: "Successfully enriched stats chunk",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 7
        )
        try db.insertChunk(
            id: "stats-duplicate",
            content: "Duplicate terminal stats chunk",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )
        try db.insertChunk(
            id: "stats-noise",
            content: "Noise terminal stats chunk",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )
        try db.insertChunk(
            id: "stats-pending",
            content: "Pending eligible stats chunk with enough text to meet the eligibility threshold",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )
        db.exec("UPDATE chunks SET enriched_at = datetime('now'), enrich_status = 'success' WHERE id = 'stats-success'")
        db.exec("UPDATE chunks SET enriched_at = 'skipped:duplicate', enrich_status = 'duplicate' WHERE id = 'stats-duplicate'")
        db.exec("UPDATE chunks SET enriched_at = NULL, enrich_status = 'noise' WHERE id = 'stats-noise'")

        let summary = try db.enrichmentStats()

        XCTAssertEqual(summary.totalChunks, 4)
        XCTAssertEqual(summary.enriched, 3)
        XCTAssertEqual(summary.unenrichedEligible, 1)
        XCTAssertEqual(summary.skippedTooShort, 0)
        XCTAssertEqual(
            summary.totalChunks,
            summary.enriched + summary.unenrichedEligible + summary.skippedTooShort
        )
        XCTAssertEqual(summary.enrichedPercentText, "75.0%")
    }

    func testDashboardStatsReadsPendingStoreQueueDepthAndOldestEntry() throws {
        let queuePath = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pending-stores-dashboard-\(UUID().uuidString).jsonl")
        let restoreQueuePath = setDashboardPendingStoreQueuePath(queuePath)
        defer {
            restoreQueuePath()
            try? FileManager.default.removeItem(at: queuePath)
        }

        let older = ISO8601DateFormatter().string(from: Date(timeIntervalSince1970: 1_000))
        let newer = ISO8601DateFormatter().string(from: Date(timeIntervalSince1970: 1_060))
        let payload = """
        {"content":"old queued item","tags":["queue"],"importance":5,"source":"mcp","queued_at":"\(older)"}
        {"content":"new queued item","tags":["queue"],"importance":5,"source":"mcp","queued_at":"\(newer)"}

        """
        try payload.write(to: queuePath, atomically: true, encoding: .utf8)

        let stats = try db.dashboardStats(activityWindowMinutes: 15, bucketCount: 4)

        XCTAssertEqual(stats.pendingStoreQueueDepth, 2)
        XCTAssertEqual(stats.pendingStoreFlushQueueDepth, 2)
        XCTAssertEqual(stats.pendingStoreOldestQueuedAt, Date(timeIntervalSince1970: 1_000))
        XCTAssertEqual(stats.pendingStoreFlushRatePerMinute, 0)
    }

    func testDashboardStatsIncludesDocsLocalFallbackReplayDebt() throws {
        let queuePath = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pending-stores-dashboard-empty-\(UUID().uuidString).jsonl")
        let gitsRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("fallback-gits-\(UUID().uuidString)", isDirectory: true)
        let fallbackPath = gitsRoot
            .appendingPathComponent("brainlayer", isDirectory: true)
            .appendingPathComponent("docs.local", isDirectory: true)
            .appendingPathComponent("decisions", isDirectory: true)
            .appendingPathComponent("pending.md")
        let restoreQueuePath = setDashboardPendingStoreQueuePath(queuePath)
        let restoreFallbackRoot = setDashboardFallbackReplayGitsRoot(gitsRoot)
        defer {
            restoreQueuePath()
            restoreFallbackRoot()
            try? FileManager.default.removeItem(at: queuePath)
            try? FileManager.default.removeItem(at: gitsRoot)
        }
        try FileManager.default.createDirectory(
            at: fallbackPath.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try """
        ---
        intended_brain_store: true
        importance: 10
        tags: [fallback-replay]
        timestamp: 1970-01-01T00:20:00Z
        retry_attempted: true
        queued_chunk_id: fallback-queued
        chunk_id:
        ---
        queued fallback body
        """.write(to: fallbackPath, atomically: true, encoding: .utf8)

        let stats = try db.dashboardStats(activityWindowMinutes: 15, bucketCount: 4)
        let summary = DashboardFlowSummary.derive(
            daemon: nil,
            stats: stats,
            now: Date(timeIntervalSince1970: 1_300)
        )
        let agentStores = summary.lane(for: .agentStores)

        XCTAssertEqual(stats.pendingStoreQueueDepth, 1)
        XCTAssertEqual(stats.pendingStoreFlushQueueDepth, 0)
        XCTAssertEqual(stats.pendingStoreOldestQueuedAt, Date(timeIntervalSince1970: 1_200))
        XCTAssertEqual(agentStores.status, .queued)
        XCTAssertEqual(agentStores.statusText, "1 agent store queued for replay")
    }

    func testDashboardStatsIncludesStaleFallbackReplayChunkIDDebt() throws {
        let stalePath = fallbackReplayRoot
            .appendingPathComponent("brainlayer", isDirectory: true)
            .appendingPathComponent("docs.local", isDirectory: true)
            .appendingPathComponent("decisions", isDirectory: true)
            .appendingPathComponent("stale.md")
        let storedPath = fallbackReplayRoot
            .appendingPathComponent("brainlayer", isDirectory: true)
            .appendingPathComponent("docs.local", isDirectory: true)
            .appendingPathComponent("decisions", isDirectory: true)
            .appendingPathComponent("stored.md")
        try FileManager.default.createDirectory(
            at: stalePath.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try """
        ---
        intended_brain_store: true
        importance: 10
        timestamp: 1970-01-01T00:24:00Z
        chunk_id: fallback-stale-edited-body
        ---
        edited fallback body
        """.write(to: stalePath, atomically: true, encoding: .utf8)
        try """
        ---
        intended_brain_store: true
        importance: 10
        timestamp: 1970-01-01T00:25:00Z
        chunk_id: fallback-bf4489e94479ad32
        ---
        stored fallback body
        """.write(to: storedPath, atomically: true, encoding: .utf8)

        let stats = try db.dashboardStats(activityWindowMinutes: 15, bucketCount: 4)

        XCTAssertEqual(stats.pendingStoreQueueDepth, 1)
        XCTAssertEqual(stats.pendingStoreFlushQueueDepth, 0)
        XCTAssertEqual(stats.pendingStoreOldestQueuedAt, Date(timeIntervalSince1970: 1_440))
    }

    func testDashboardStatsAcceptsStoredScopedFallbackChunkIDWhenProjectIsPersisted() throws {
        let storedPath = fallbackReplayRoot
            .appendingPathComponent("brainlayer", isDirectory: true)
            .appendingPathComponent("docs.local", isDirectory: true)
            .appendingPathComponent("decisions", isDirectory: true)
            .appendingPathComponent("scoped.md")
        try FileManager.default.createDirectory(
            at: storedPath.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try """
        ---
        intended_brain_store: true
        project: systems
        importance: 10
        timestamp: 1970-01-01T00:26:00Z
        chunk_id: fallback-f7c15c956dca2e7e
        ---
        scoped fallback body
        """.write(to: storedPath, atomically: true, encoding: .utf8)

        let stats = try db.dashboardStats(activityWindowMinutes: 15, bucketCount: 4)

        XCTAssertEqual(stats.pendingStoreQueueDepth, 0)
        XCTAssertEqual(stats.pendingStoreFlushQueueDepth, 0)
    }

    func testDashboardStatsAcceptsStoredFallbackChunkIDWithNullTimestamp() throws {
        let storedPath = fallbackReplayRoot
            .appendingPathComponent("brainlayer", isDirectory: true)
            .appendingPathComponent("docs.local", isDirectory: true)
            .appendingPathComponent("decisions", isDirectory: true)
            .appendingPathComponent("null-timestamp.md")
        try FileManager.default.createDirectory(
            at: storedPath.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try """
        ---
        intended_brain_store: true
        importance: 10
        timestamp: null
        chunk_id: fallback-c99da95d9166ba2e
        ---
        null timestamp fallback body
        """.write(to: storedPath, atomically: true, encoding: .utf8)

        let stats = try db.dashboardStats(activityWindowMinutes: 15, bucketCount: 4)

        XCTAssertEqual(stats.pendingStoreQueueDepth, 0)
        XCTAssertEqual(stats.pendingStoreFlushQueueDepth, 0)
    }

    func testDashboardStatsAcceptsStoredFallbackChunkIDWithTildeTimestamp() throws {
        let storedPath = fallbackReplayRoot
            .appendingPathComponent("brainlayer", isDirectory: true)
            .appendingPathComponent("docs.local", isDirectory: true)
            .appendingPathComponent("decisions", isDirectory: true)
            .appendingPathComponent("tilde-timestamp.md")
        try FileManager.default.createDirectory(
            at: storedPath.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try """
        ---
        intended_brain_store: true
        importance: 10
        timestamp: ~
        chunk_id: fallback-edf2bca7f184d086
        ---
        tilde timestamp fallback body
        """.write(to: storedPath, atomically: true, encoding: .utf8)

        let stats = try db.dashboardStats(activityWindowMinutes: 15, bucketCount: 4)

        XCTAssertEqual(stats.pendingStoreQueueDepth, 0)
        XCTAssertEqual(stats.pendingStoreFlushQueueDepth, 0)
    }

    func testDashboardStatsIncludesLegacyFallbackReplayDebtWithoutFrontmatter() throws {
        let legacyPath = fallbackReplayRoot
            .appendingPathComponent("brainlayer", isDirectory: true)
            .appendingPathComponent("docs.local", isDirectory: true)
            .appendingPathComponent("brain-store-fallback", isDirectory: true)
            .appendingPathComponent("2026-05-29-gen10-boot", isDirectory: true)
            .appendingPathComponent("pending-stores.md")
        try FileManager.default.createDirectory(
            at: legacyPath.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try "legacy fallback body\n".write(to: legacyPath, atomically: true, encoding: .utf8)

        let stats = try db.dashboardStats(activityWindowMinutes: 15, bucketCount: 4)

        XCTAssertEqual(stats.pendingStoreQueueDepth, 1)
        XCTAssertEqual(stats.pendingStoreFlushQueueDepth, 0)
        XCTAssertEqual(stats.pendingStoreOldestQueuedAt, ISO8601DateFormatter().date(from: "2026-05-29T00:00:00Z"))
    }

    func testDashboardStatsIncludesParsedLegacyFallbackReplayDebtWithoutIntentFlag() throws {
        let legacyPath = fallbackReplayRoot
            .appendingPathComponent("brainlayer", isDirectory: true)
            .appendingPathComponent("docs.local", isDirectory: true)
            .appendingPathComponent("brain-store-fallback", isDirectory: true)
            .appendingPathComponent("parsed-legacy.md")
        try FileManager.default.createDirectory(
            at: legacyPath.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try """
        ---
        memory_type: note
        timestamp: 2024-01-15T10:30:45.123456+00:00
        chunk_id:
        ---
        parsed legacy fallback body
        """.write(to: legacyPath, atomically: true, encoding: .utf8)
        let expected = try XCTUnwrap(Self.fractionalTimestampFormatter.date(from: "2024-01-15T10:30:45.123456+00:00"))

        let stats = try db.dashboardStats(activityWindowMinutes: 15, bucketCount: 4)

        XCTAssertEqual(stats.pendingStoreQueueDepth, 1)
        XCTAssertEqual(stats.pendingStoreFlushQueueDepth, 0)
        XCTAssertEqual(
            try XCTUnwrap(stats.pendingStoreOldestQueuedAt).timeIntervalSince1970,
            expected.timeIntervalSince1970,
            accuracy: 0.001
        )
    }

    func testSparklineChartPresentationCarriesBucketsAndVoiceOverMetadata() {
        let now = Date(timeIntervalSince1970: 1_764_236_400)
        let presentation = SparklineChartPresentation(
            label: "Recent activity sparkline",
            values: [0, 2, 5, 3],
            activityWindowMinutes: 20,
            fetchedAt: now
        )

        XCTAssertEqual(presentation.points.map(\.bucket), [0, 1, 2, 3])
        XCTAssertEqual(presentation.points.map(\.value), [0, 2, 5, 3])
        XCTAssertEqual(presentation.accessibilityLabel, "Recent activity sparkline")
        XCTAssertEqual(presentation.accessibilityValue, "latest bucket count 3, trending down")
        XCTAssertEqual(
            presentation.bucketLabel(for: 0),
            "\(Self.shortTime(now.addingTimeInterval(-20 * 60)))-\(Self.shortTime(now.addingTimeInterval(-15 * 60)))"
        )
        XCTAssertEqual(
            presentation.bucketLabel(for: 3),
            "\(Self.shortTime(now.addingTimeInterval(-5 * 60)))-\(Self.shortTime(now))"
        )
        XCTAssertEqual(presentation.relativeBucketLabel(for: 2), "10m-5m ago")
    }

    func testSparklineChartPresentationNamesHoverBucketsByRecency() {
        let now = Date(timeIntervalSince1970: 1_764_236_400)
        let presentation = SparklineChartPresentation(
            label: "Recent activity sparkline",
            values: [0, 2, 5, 3],
            activityWindowMinutes: 20,
            fetchedAt: now
        )

        XCTAssertEqual(presentation.bucketRecencyLabel(for: 3), "now")
        XCTAssertEqual(presentation.bucketRecencyLabel(for: 2), "5m ago")
        XCTAssertEqual(presentation.bucketRecencyLabel(for: 0), "15m ago")
    }

    func testSparklineChartPresentationLabelsPartialMinuteBucketsLikeDatabase() {
        let now = Date(timeIntervalSince1970: 1_764_236_400)
        let presentation = SparklineChartPresentation(
            label: "Recent activity sparkline",
            values: Array(repeating: 0, count: 12),
            activityWindowMinutes: 31,
            fetchedAt: now
        )

        let bucketWidth = Double(31 * 60) / 12
        XCTAssertEqual(
            presentation.bucketLabel(for: 0),
            "\(Self.shortTime(now.addingTimeInterval(-31 * 60)))-\(Self.shortTime(now.addingTimeInterval(-(31 * 60 - bucketWidth))))"
        )
        XCTAssertEqual(
            presentation.bucketLabel(for: 11),
            "\(Self.shortTime(now.addingTimeInterval(-bucketWidth)))-\(Self.shortTime(now))"
        )
    }

    func testSparklineChartPresentationLabelsLongBucketsInHours() {
        let now = Date(timeIntervalSince1970: 1_764_236_400)
        let presentation = SparklineChartPresentation(
            label: "JSONL watcher over Last 24h",
            values: Array(repeating: 0, count: 12),
            activityWindowMinutes: 1_440,
            fetchedAt: now
        )

        XCTAssertEqual(presentation.relativeBucketLabel(for: 8), "8h-6h ago")
        XCTAssertEqual(presentation.relativeBucketLabel(for: 11), "last 2h")
    }

    func testSparklinePresentationMarksLatestNonZeroBucketEvenWhenPeakCrushesScale() {
        let presentation = SparklineChartPresentation(
            label: "JSONL watcher over Last 24h",
            values: [0, 0, 280, 590, 496, 763, 8_284, 2_065, 42, 0, 0, 70],
            activityWindowMinutes: 1_440
        )

        XCTAssertEqual(
            presentation.visiblePointMarkers(for: .primary, compact: false).map(\.bucket),
            [11]
        )
        XCTAssertEqual(
            presentation.visiblePointMarkers(for: .primary, compact: false).map(\.value),
            [70]
        )
    }

    func testSparklineChartPresentationAnchorsPointsToWallClockBucketMidpoints() {
        let now = Date(timeIntervalSince1970: 1_764_236_400)
        let presentation = SparklineChartPresentation(
            label: "Recent enrichment sparkline",
            values: [0, 3, 0, 1],
            activityWindowMinutes: 20,
            fetchedAt: now
        )

        XCTAssertEqual(presentation.points.map(\.bucket), [0, 1, 2, 3])
        XCTAssertEqual(presentation.points.map(\.value), [0, 3, 0, 1])
        XCTAssertEqual(presentation.points.map(\.timestamp), [
            now.addingTimeInterval(-17.5 * 60),
            now.addingTimeInterval(-12.5 * 60),
            now.addingTimeInterval(-7.5 * 60),
            now.addingTimeInterval(-2.5 * 60),
        ])
        XCTAssertEqual(presentation.xAxisDomainStart, now.addingTimeInterval(-20 * 60))
        XCTAssertEqual(presentation.xAxisDomainEnd, now)
    }

    func testSparklinePresentationOmitsZeroWriteSourcesButKeepsDimmedLegendEntries() {
        let presentation = SparklineChartPresentation(
            label: "Writes over 30m",
            values: [0, 1, 0],
            secondaryValues: [0, 0, 0],
            primarySeriesLabel: "Agent stores",
            secondarySeriesLabel: "JSONL watcher",
            activityWindowMinutes: 30,
            fetchedAt: Date(timeIntervalSince1970: 1_764_236_400)
        )

        XCTAssertEqual(presentation.visibleSeriesLabels, ["Agent stores"])
        XCTAssertEqual(presentation.legendEntries.map(\.label), ["Agent stores", "JSONL watcher"])
        XCTAssertEqual(presentation.legendEntries.map(\.isActive), [true, false])
    }

    func testSparklinePresentationShowsListeningCaptionForColdWriteStart() {
        let presentation = SparklineChartPresentation(
            label: "Writes over 30m",
            values: [0, 0, 0],
            secondaryValues: [0, 0, 0],
            primarySeriesLabel: "Agent stores",
            secondarySeriesLabel: "JSONL watcher",
            activityWindowMinutes: 30,
            fetchedAt: Date(timeIntervalSince1970: 1_764_236_400)
        )

        XCTAssertTrue(presentation.showsListeningForWritesCaption)
        XCTAssertEqual(presentation.visibleSeriesLabels, [])
        XCTAssertTrue(presentation.legendEntries.allSatisfy { !$0.isActive })
    }

    func testSparklinePresentationClassifiesDensityAndUsesTightTicksForSmallPeaks() {
        let sparse = SparklineChartPresentation(
            label: "Enrichments over 30m",
            values: [0, 0, 3, 0, 0],
            secondaryValues: [1, 1, 0, 1, 1],
            activityWindowMinutes: 30,
            fetchedAt: Date(timeIntervalSince1970: 1_764_236_400)
        )

        XCTAssertEqual(sparse.nonZeroFraction(.primary), 0.2, accuracy: 0.001)
        XCTAssertFalse(sparse.isDense(.primary))
        XCTAssertTrue(sparse.isDense(.secondary))
        XCTAssertEqual(sparse.axisMax, 5)
        XCTAssertEqual(sparse.tightAxisMax, 3)
        XCTAssertEqual(sparse.tightYAxisTicks, [0, 2, 3])

        let larger = SparklineChartPresentation(
            label: "Writes over 30m",
            values: [0, 9],
            activityWindowMinutes: 30,
            fetchedAt: Date(timeIntervalSince1970: 1_764_236_400)
        )

        XCTAssertEqual(larger.tightAxisMax, larger.axisMax)
        XCTAssertEqual(larger.axisMax, 10)
    }

    func testSparklineCompactMarkersPreferLatestPointForSparseSeries() {
        let sparse = SparklineChartPresentation(
            label: "Sparse writes over 30m",
            values: [0, 1, 0, 1],
            activityWindowMinutes: 30,
            fetchedAt: Date(timeIntervalSince1970: 1_764_236_400)
        )

        XCTAssertEqual(sparse.visiblePointMarkers(for: .primary, compact: false).map(\.bucket), [1, 3])
        XCTAssertEqual(sparse.visiblePointMarkers(for: .primary, compact: true).map(\.bucket), [3])
    }

    func testSparklineNonCompactMarkersKeepLatestEndpointWhenLatestBucketIsZero() {
        let presentation = SparklineChartPresentation(
            label: "Writes over 30m",
            values: [3, 2, 0],
            activityWindowMinutes: 30,
            fetchedAt: Date(timeIntervalSince1970: 1_764_236_400)
        )

        XCTAssertEqual(presentation.visiblePointMarkers(for: .primary, compact: false).map(\.bucket), [2])
    }

    func testSparklineTooltipPlacementClampsHorizontally() {
        let container = CGSize(width: 160, height: 100)
        let tooltip = SparklineTooltipPlacement.tooltipSize(in: container)

        let left = SparklineTooltipPlacement.position(
            near: CGPoint(x: 0, y: 80),
            anchorY: 80,
            hoveredX: 0,
            containerBounds: container,
            tooltipSize: tooltip
        )
        let right = SparklineTooltipPlacement.position(
            near: CGPoint(x: 220, y: 80),
            anchorY: 80,
            hoveredX: 220,
            containerBounds: container,
            tooltipSize: tooltip
        )

        XCTAssertGreaterThanOrEqual(left.x - tooltip.width / 2, 8)
        XCTAssertLessThanOrEqual(right.x + tooltip.width / 2, container.width - 8)
    }

    func testSparklineTooltipPlacementAnchorsToPointAndFallsBackSidewaysNearTopSpike() {
        let container = CGSize(width: 420, height: 96)
        let tooltip = SparklineTooltipPlacement.tooltipSize(in: container)

        let topSpike = SparklineTooltipPlacement.position(
            near: CGPoint(x: 35, y: 12),
            anchorY: 12,
            hoveredX: 35,
            containerBounds: container,
            tooltipSize: tooltip
        )
        let roomy = SparklineTooltipPlacement.position(
            near: CGPoint(x: 220, y: 86),
            anchorY: 86,
            hoveredX: 220,
            containerBounds: container,
            tooltipSize: tooltip
        )

        XCTAssertGreaterThan(topSpike.x, 35, "Top spikes should side-place instead of pinning the tooltip to the bottom.")
        XCTAssertLessThanOrEqual(topSpike.y + tooltip.height / 2, container.height - 8)
        XCTAssertLessThan(roomy.y, 86, "Roomy lower points should place the tooltip above the on-curve anchor.")
    }

    // Degenerate case: a top-edge spike in a container so narrow that NEITHER side
    // candidate fits at the anchor column. The card must still clear the dot by
    // pinning to the edge with more room — never land on the anchor column and
    // overlap the curve. (Reviewer M1, PR #526.)
    func testSparklineTooltipPlacementSidePinsToEdgeWhenNoSideRoomAtAnchor() {
        // 300 wide so minX (138) < maxX (162): the edge is distinct from the
        // horizontally-clamped anchor column, yet the tooltip (260 wide) is wide
        // enough that NEITHER side candidate fits at the anchor.
        let container = CGSize(width: 300, height: 96)
        let tooltip = SparklineTooltipPlacement.tooltipSize(in: container)
        let halfWidth = tooltip.width / 2
        let minX = 8 + halfWidth
        let maxX = max(container.width - 8 - halfWidth, minX)
        XCTAssertLessThan(minX, maxX, "Test geometry must keep the edges distinct from the anchor column.")

        // Top spike right-of-center: rightX and leftX both fall outside [minX, maxX],
        // so the SIDE fallback must pin to the roomier (left) edge, not the anchor column.
        let anchorX: CGFloat = 175
        let topSpike = SparklineTooltipPlacement.position(
            near: CGPoint(x: anchorX, y: 8),
            anchorY: 8,
            hoveredX: anchorX,
            containerBounds: container,
            tooltipSize: tooltip
        )
        let clampedAnchorColumn = min(max(anchorX, minX), maxX)
        XCTAssertEqual(topSpike.x, minX, accuracy: 0.5,
                       "No side room at the anchor column should pin to the roomier (left) edge.")
        XCTAssertNotEqual(topSpike.x, clampedAnchorColumn, accuracy: 0.5,
                          "The card must not sit on the anchor column overlapping the curve.")
    }

    // A sparse series with a spike gets the lifted soft floor (so its zeros read as a
    // designed band), but a series with NO data at all stays on the true zero baseline
    // so it never implies a phantom band of activity. (Reviewer Codex P2, PR #526.)
    func testSparkBaselineUsesSoftFloorOnlyForSparseSeriesWithData() {
        // Sparse but has a spike -> soft floor.
        XCTAssertTrue(SparklineBaseline.usesSoftFloor(isDense: false, nonZeroFraction: 0.2))
        // Completely empty -> true baseline (no soft floor).
        XCTAssertFalse(SparklineBaseline.usesSoftFloor(isDense: false, nonZeroFraction: 0.0))
        // Dense -> true baseline.
        XCTAssertFalse(SparklineBaseline.usesSoftFloor(isDense: true, nonZeroFraction: 0.8))
    }

    // The hover indicator must ride the PLOTTED series with the visible spike at the
    // hovered bucket, not always the primary — otherwise a Writes chart fed only by
    // the watcher lane (primary all-zero) anchors the dot/tooltip to the baseline.
    // (Reviewers: Cursor Bugbot Medium / Codex P2, PR #526.)
    func testHoverAnchorPicksDominantPlottedSeries() {
        // Only secondary plotted, secondary has the value at this bucket.
        XCTAssertEqual(
            SparklineHoverAnchor.dominantRole(
                plotted: [.secondary],
                valueAtBucket: [.secondary: 4]
            ),
            .secondary
        )
        // Both plotted; secondary spikes while primary is zero -> ride secondary.
        XCTAssertEqual(
            SparklineHoverAnchor.dominantRole(
                plotted: [.primary, .secondary],
                valueAtBucket: [.primary: 0, .secondary: 5]
            ),
            .secondary
        )
        // Tie favors primary (declaration order).
        XCTAssertEqual(
            SparklineHoverAnchor.dominantRole(
                plotted: [.primary, .secondary],
                valueAtBucket: [.primary: 3, .secondary: 3]
            ),
            .primary
        )
        // Primary dominates.
        XCTAssertEqual(
            SparklineHoverAnchor.dominantRole(
                plotted: [.primary, .secondary, .tertiary],
                valueAtBucket: [.primary: 6, .secondary: 2, .tertiary: 1]
            ),
            .primary
        )
        // Nothing plotted -> safe fallback to primary.
        XCTAssertEqual(
            SparklineHoverAnchor.dominantRole(plotted: [], valueAtBucket: [:]),
            .primary
        )
    }

    func testSparklineTooltipPlacementFlipsBelowNearTopEdge() {
        let container = CGSize(width: 260, height: 120)
        let tooltip = SparklineTooltipPlacement.tooltipSize(in: container)

        let top = SparklineTooltipPlacement.position(
            near: CGPoint(x: 130, y: 4),
            anchorY: 4,
            hoveredX: 130,
            containerBounds: container,
            tooltipSize: tooltip
        )
        let lower = SparklineTooltipPlacement.position(
            near: CGPoint(x: 130, y: 90),
            anchorY: 90,
            hoveredX: 130,
            containerBounds: container,
            tooltipSize: tooltip
        )

        XCTAssertGreaterThan(top.y, 4)
        XCTAssertLessThan(lower.y, 90)
    }

    func testSparklineTooltipUsesRoundedUpMetricsAndSingleOpaqueBackground() throws {
        let compactContainer = CGSize(width: 260, height: 90)
        let tooltip = SparklineTooltipPlacement.tooltipSize(in: compactContainer, seriesCount: 3)
        let source = try brainBarSourceFile("Sources/BrainBar/Dashboard/SparklineRenderer.swift")

        XCTAssertEqual(tooltip.height, 104)
        XCTAssertTrue(source.contains("private static let headerHeight: CGFloat = 24"))
        XCTAssertTrue(source.contains("private static let dividerBlock: CGFloat = 14"))
        XCTAssertTrue(source.contains("private static let rowHeight: CGFloat = 16"))
        XCTAssertFalse(source.contains(".background(.regularMaterial"))
        XCTAssertTrue(source.contains(".fill(Color.brainBarBackgroundRaised)"))
    }

    func testSparklineRendererSimplifiesStructuredTooltipHelpers() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/Dashboard/SparklineRenderer.swift")

        XCTAssertFalse(source.contains("func tooltipText(forBucket"))
        XCTAssertTrue(source.contains("let clampedBucket = min(max(hoveredBucket, 0), max(presentation.values.count - 1, 0))"))
        XCTAssertTrue(source.contains("private var plotMax: Int { compact ? presentation.maxValue : presentation.tightAxisMax }"))
        XCTAssertFalse(source.contains("line 523"))
        XCTAssertFalse(source.contains("(G3)"))
        XCTAssertFalse(source.contains("(G4)"))
        XCTAssertFalse(source.contains("(G5)"))
        XCTAssertFalse(source.contains("(I1)"))
    }

    func testDashboardMetricFormatterRequiresAbsoluteLastEventText() {
        let now = Date(timeIntervalSince1970: 1_764_236_400)
        let lastEvent = now.addingTimeInterval(-125)

        XCTAssertEqual(
            DashboardMetricFormatter.lastEventString(lastEventAt: lastEvent, now: now),
            "\(Self.absoluteTime(lastEvent)) (2m ago)"
        )
    }

    func testDashboardMetricFormatterDoesNotOverstateRelativeBoundaries() {
        let now = Date(timeIntervalSince1970: 1_764_236_400)

        XCTAssertEqual(
            DashboardMetricFormatter.relativeEventString(
                lastEventAt: now.addingTimeInterval(-60.1),
                now: now
            ),
            "1m ago"
        )
        XCTAssertEqual(
            DashboardMetricFormatter.relativeEventString(
                lastEventAt: now.addingTimeInterval(-3600.1),
                now: now
            ),
            "1h ago"
        )
    }

    func testSparklineRendererCompactClassificationMatchesEndpointAndChartPadding() {
        XCTAssertTrue(SparklineRenderer.isCompact(size: NSSize(width: 52, height: 116)))
        XCTAssertTrue(SparklineRenderer.isCompact(size: NSSize(width: 300, height: 20)))
        XCTAssertFalse(SparklineRenderer.isCompact(size: NSSize(width: 53, height: 21)))
    }

    func testSparklineChartOmitsCurrentValueEndpointAnnotation() throws {
        let source = try brainBarSourceFile("Sources/BrainBar/Dashboard/SparklineRenderer.swift")

        XCTAssertFalse(
            source.contains("currentValueLabelPosition"),
            "Dashboard sparklines should keep endpoint markers but not render the floating current-value pill."
        )
        XCTAssertFalse(
            source.contains("Text(\"\\(last.value)\")"),
            "The confusing numeric endpoint pill should be removed from Writes and Enrichments charts."
        )
    }

    func testDashboardStatsCountsRecentISO8601Timestamps() throws {
        try db.insertChunk(
            id: "dash-iso",
            content: "Recent chunk written by Python with ISO timestamp",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 6
        )
        db.exec("""
            UPDATE chunks
            SET created_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = 'dash-iso'
        """)

        let stats = try db.dashboardStats(activityWindowMinutes: 5, bucketCount: 5)

        XCTAssertEqual(stats.recentActivityBuckets.reduce(0, +), 1)
    }

    func testDashboardStatsCountsRecentNaiveLocalTimestamps() throws {
        try db.insertChunk(
            id: "dash-local-naive",
            content: "Recent chunk written with a local wall-clock timestamp",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 6
        )

        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
        let now = Date()

        db.exec("""
            UPDATE chunks
            SET created_at = '\(formatter.string(from: now))'
            WHERE id = 'dash-local-naive'
        """)

        let stats = try db.dashboardStats(activityWindowMinutes: 5, bucketCount: 5)

        XCTAssertEqual(stats.recentActivityBuckets.reduce(0, +), 1)
        let lastWriteAt = try XCTUnwrap(stats.lastWriteAt)
        XCTAssertLessThan(abs(lastWriteAt.timeIntervalSince(now)), 5)
    }

    func testDashboardStatsCountsRecentOffsetTimestampCrossingUTCDateBoundary() throws {
        try db.insertChunk(
            id: "dash-offset-crossing",
            content: "Recent chunk written with an offset timestamp on a different UTC date",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 6
        )

        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        let utcTimeZone = TimeZone(secondsFromGMT: 0)!
        let recent = Date().addingTimeInterval(-60)
        let utcHour = Calendar(identifier: .gregorian)
            .dateComponents(in: utcTimeZone, from: recent)
            .hour ?? 0
        let crossingOffsetHours = utcHour >= 10 ? 14 : -12
        formatter.timeZone = TimeZone(secondsFromGMT: crossingOffsetHours * 3600)
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssXXXXX"

        let utcDay = DateFormatter()
        utcDay.locale = Locale(identifier: "en_US_POSIX")
        utcDay.timeZone = utcTimeZone
        utcDay.dateFormat = "yyyy-MM-dd"
        let offsetDay = DateFormatter()
        offsetDay.locale = Locale(identifier: "en_US_POSIX")
        offsetDay.timeZone = formatter.timeZone
        offsetDay.dateFormat = "yyyy-MM-dd"
        XCTAssertNotEqual(utcDay.string(from: recent), offsetDay.string(from: recent))

        db.exec("""
            UPDATE chunks
            SET created_at = '\(formatter.string(from: recent))'
            WHERE id = 'dash-offset-crossing'
        """)

        let stats = try db.dashboardStats(activityWindowMinutes: 5, bucketCount: 5)

        XCTAssertEqual(stats.recentActivityBuckets.reduce(0, +), 1)
    }

    func testDashboardStatsTracksRecentEnrichmentSeparatelyFromIncomingWrites() throws {
        try db.insertChunk(
            id: "dash-enrichment-only",
            content: "Older chunk enriched just now",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 6
        )
        db.exec("""
            UPDATE chunks
            SET created_at = datetime('now', '-45 minutes'),
                enriched_at = datetime('now'),
                enrich_status = 'success'
            WHERE id = 'dash-enrichment-only'
        """)

        let stats = try db.dashboardStats(activityWindowMinutes: 30, bucketCount: 6)

        XCTAssertEqual(stats.recentActivityBuckets.reduce(0, +), 0)
        XCTAssertEqual(stats.recentEnrichmentBuckets.reduce(0, +), 1)
        XCTAssertGreaterThan(stats.enrichmentRatePerMinute, 0)
    }

    func testDashboardStatsCurrentEnrichmentRateDropsToZeroWhenPipelineIsIdle() throws {
        try db.insertChunk(
            id: "dash-stale-enrichment",
            content: "Enriched earlier but not currently moving",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 6
        )
        db.exec("""
            UPDATE chunks
            SET enriched_at = datetime('now', '-6 minutes'),
                enrich_status = 'success'
            WHERE id = 'dash-stale-enrichment'
        """)

        let stats = try db.dashboardStats(activityWindowMinutes: 30, bucketCount: 6)

        XCTAssertEqual(stats.enrichmentRatePerMinute, 0, accuracy: 0.001)
        XCTAssertEqual(stats.recentEnrichmentBuckets.reduce(0, +), 1)
    }

    func testDashboardStatsTreatsNinetySecondStallAsIdle() throws {
        try db.insertChunk(
            id: "dash-minute-stall",
            content: "Recently enriched but already stalled past the 60s live window",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 6
        )
        db.exec("""
            UPDATE chunks
            SET enriched_at = datetime('now', '-90 seconds'),
                enrich_status = 'success'
            WHERE id = 'dash-minute-stall'
        """)

        let stats = try db.dashboardStats(activityWindowMinutes: 30, bucketCount: 6)

        XCTAssertEqual(stats.enrichmentRatePerMinute, 0, accuracy: 0.001)
        XCTAssertEqual(stats.recentEnrichmentBuckets.reduce(0, +), 1)
    }

    func testDashboardStatsCarriesExplicitWindowMetadataAndLastEventTimestamps() throws {
        try db.insertChunk(
            id: "dash-explicit-times",
            content: "Fresh write with a slightly older enrichment event",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 6
        )
        db.exec("""
            UPDATE chunks
            SET created_at = datetime('now'),
                enriched_at = datetime('now', '-90 seconds'),
                enrich_status = 'success'
            WHERE id = 'dash-explicit-times'
        """)

        let stats = try db.dashboardStats(activityWindowMinutes: 60, bucketCount: 12)

        XCTAssertEqual(stats.activityWindowMinutes, 60)
        XCTAssertEqual(stats.bucketCount, 12)
        XCTAssertEqual(stats.liveWindowMinutes, 1)
        XCTAssertNotNil(stats.lastWriteAt)
        XCTAssertNotNil(stats.lastEnrichedAt)
        let lastWriteAt = try XCTUnwrap(stats.lastWriteAt)
        let lastEnrichedAt = try XCTUnwrap(stats.lastEnrichedAt)
        XCTAssertGreaterThan(lastWriteAt, lastEnrichedAt)
        XCTAssertEqual(stats.enrichmentRatePerMinute, 0, accuracy: 0.001)
        XCTAssertEqual(stats.recentEnrichmentBuckets.reduce(0, +), 1)
    }

    func testDashboardDataVersionChangesAfterExternalWrite() throws {
        let baseline = try db.dataVersion()
        let writer = BrainDatabase(path: tempDBPath)
        defer { writer.close() }

        try writer.insertChunk(
            id: "dash-version",
            content: "External writer changed the database",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )

        XCTAssertGreaterThan(try db.dataVersion(), baseline)
    }

    @MainActor
    func testStatsCollectorRefreshesAfterDatabaseWriteNotification() async throws {
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            statsRefreshCoalesceInterval: 0.05
        )
        defer { collector.stop() }

        collector.start()
        try await waitForCollector(collector) { !$0.isRefreshing && $0.lastDataFetchedAt != nil }
        XCTAssertEqual(collector.stats.chunkCount, 0)

        let writer = BrainDatabase(path: tempDBPath)
        defer { writer.close() }

        try writer.insertChunk(
            id: "dash-late-write",
            content: "Inserted after collector start",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )

        let deadline = Date().addingTimeInterval(2.0)
        while collector.stats.chunkCount == 0 && Date() < deadline {
            try await Task.sleep(for: .milliseconds(50))
        }

        XCTAssertEqual(collector.stats.chunkCount, 1)
    }

    @MainActor
    func testReadOnlyStatsCollectorReopensAfterDaemonCreatesDatabase() async throws {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")

        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            databaseOpenConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
        defer { collector.stop() }

        collector.refresh(force: true)
        try await waitForCollector(collector) { !$0.isRefreshing }
        XCTAssertEqual(collector.stats.chunkCount, 0)

        let writer = BrainDatabase(path: tempDBPath)
        defer { writer.close() }
        try writer.insertChunk(
            id: "dash-readonly-late-db",
            content: "Daemon created the database after UI startup",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )

        collector.refresh(force: true)
        try await waitForCollector(collector) { $0.stats.chunkCount == 1 }

        XCTAssertEqual(collector.stats.chunkCount, 1)
    }

    @MainActor
    func testStatsCollectorSubscribesToBrainBusWithoutPollingDelay() async throws {
        let eventSource = RecordingBrainBusEventSource()
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            brainBusEvents: eventSource
        )
        defer { collector.stop() }

        let startedAt = DispatchTime.now()
        collector.start()
        let elapsedMillis = Double(DispatchTime.now().uptimeNanoseconds - startedAt.uptimeNanoseconds) / 1_000_000

        XCTAssertEqual(eventSource.streamRequestCount, 1)
        XCTAssertLessThan(elapsedMillis, 1_000)
        try await waitForCollector(collector) { !$0.isRefreshing && $0.lastDataFetchedAt != nil }
    }

    @MainActor
    func testStatsCollectorRefreshesWithBrainBusEvents() async throws {
        try db.insertChunk(
            id: "dash-preexisting",
            content: "Inserted before collector start",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )

        let eventSource = RecordingBrainBusEventSource()
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            brainBusEvents: eventSource
        )
        defer { collector.stop() }

        collector.start()

        let deadline = Date().addingTimeInterval(2.0)
        while collector.stats.chunkCount == 0 && Date() < deadline {
            try await Task.sleep(for: .milliseconds(50))
        }

        XCTAssertEqual(eventSource.streamRequestCount, 1)
        XCTAssertEqual(collector.stats.chunkCount, 1)
    }

    @MainActor
    func testBrainBusEventsUpdateHeartbeatAndScheduleCoalescedStatsRefresh() async throws {
        let eventSource = RecordingBrainBusEventSource()
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            statsRefreshCoalesceInterval: 0.05,
            autoRefreshInterval: 60,
            brainBusEvents: eventSource
        )
        defer { collector.stop() }

        collector.start()
        try await waitForCollector(collector) { $0.lastDataFetchedAt != nil }
        let fetchedAt = try XCTUnwrap(collector.lastDataFetchedAt)

        eventSource.publish(.lastChunkID("heartbeat-only"))
        try await waitForCollector(collector) { $0.heartbeat.lastEvent?.lastChunkID == "heartbeat-only" }
        try await waitForCollector(collector) { $0.lastDataFetchedAt != fetchedAt }

        XCTAssertNotEqual(collector.lastDataFetchedAt, fetchedAt)
    }

    @MainActor
    func testHeartbeatMarksStatsSnapshotPendingDuringCoalescedRefresh() async throws {
        let eventSource = RecordingBrainBusEventSource()
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            statsRefreshCoalesceInterval: 60,
            autoRefreshInterval: 60,
            brainBusEvents: eventSource
        )
        defer { collector.stop() }

        collector.start()
        try await waitForCollector(collector) { $0.lastDataFetchedAt != nil }
        let fetchedAt = try XCTUnwrap(collector.lastDataFetchedAt)

        eventSource.publish(.lastChunkID("pending-stats-refresh"))
        try await waitForCollector(collector) { $0.heartbeat.lastEvent?.lastChunkID == "pending-stats-refresh" }

        XCTAssertEqual(collector.lastDataFetchedAt, fetchedAt)
        XCTAssertTrue(collector.hasPendingStatsRefresh)
        XCTAssertTrue(collector.isHeartbeatAheadOfStats)
    }

    @MainActor
    func testBrainBusDataEventRefreshesEnrichmentBucketsWithoutLongCoalesceDelay() async throws {
        let eventSource = RecordingBrainBusEventSource()
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            statsRefreshCoalesceInterval: 60,
            autoRefreshInterval: 60,
            brainBusEvents: eventSource
        )
        defer { collector.stop() }

        collector.start()
        try await waitForCollector(collector) { $0.lastDataFetchedAt != nil }
        XCTAssertEqual(collector.stats.recentEnrichmentBuckets.reduce(0, +), 0)

        try db.insertChunk(
            id: "brain-bus-live-enrichment",
            content: "BrainBus enrichment events should refresh the dashboard chart promptly",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )
        db.exec("""
            UPDATE chunks
            SET enriched_at = datetime('now'),
                enrich_status = 'success'
            WHERE id = 'brain-bus-live-enrichment'
        """)

        eventSource.publish(.enrichStatus("running"))

        try await waitForCollector(collector, timeout: 0.5) {
            $0.stats.recentEnrichmentBuckets.reduce(0, +) == 1
        }

        XCTAssertEqual(collector.stats.recentEnrichmentBuckets.reduce(0, +), 1)
    }

    @MainActor
    func testManualRefreshRepopulatesRecentEnrichmentBuckets() async throws {
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            statsRefreshCoalesceInterval: 60,
            autoRefreshInterval: 60
        )
        defer { collector.stop() }

        collector.refresh(force: true)
        try await waitForCollector(collector) { !$0.isRefreshing }
        XCTAssertEqual(collector.stats.recentEnrichmentBuckets.reduce(0, +), 0)

        try db.insertChunk(
            id: "manual-refresh-enrichment",
            content: "Manual refresh should repopulate the enrichment sparkline buckets",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )
        db.exec("""
            UPDATE chunks
            SET created_at = datetime('now', '-45 minutes'),
                enriched_at = datetime('now'),
                enrich_status = 'success'
            WHERE id = 'manual-refresh-enrichment'
        """)

        collector.manualRefresh()
        try await waitForCollector(collector) { $0.stats.recentEnrichmentBuckets.reduce(0, +) == 1 }

        XCTAssertEqual(collector.stats.recentEnrichmentCount, 1)
        XCTAssertEqual(collector.stats.recentEnrichmentBuckets.reduce(0, +), 1)
    }

    @MainActor
    func testStatsCollectorComputesPendingStoreFlushRateFromDepthDrops() async throws {
        let queuePath = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pending-stores-collector-\(UUID().uuidString).jsonl")
        let restoreQueuePath = setDashboardPendingStoreQueuePath(queuePath)
        defer {
            restoreQueuePath()
            try? FileManager.default.removeItem(at: queuePath)
        }

        try writeDashboardPendingStoreQueue(count: 3, to: queuePath)

        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        defer { collector.stop() }

        collector.refresh(force: true)
        try await waitForCollector(collector) { $0.stats.pendingStoreQueueDepth == 3 }
        XCTAssertEqual(collector.stats.pendingStoreQueueDepth, 3)
        XCTAssertEqual(collector.stats.pendingStoreFlushRatePerMinute, 0)

        try writeDashboardPendingStoreQueue(count: 1, to: queuePath)
        collector.refresh(force: true)
        try await waitForCollector(collector) {
            $0.stats.pendingStoreQueueDepth == 1 && $0.stats.pendingStoreFlushRatePerMinute == 2
        }

        XCTAssertEqual(collector.stats.pendingStoreQueueDepth, 1)
        XCTAssertEqual(collector.stats.pendingStoreFlushRatePerMinute, 2)
    }

    @MainActor
    func testStatsCollectorDoesNotTreatFallbackReplayDebtDropAsStoreFlush() async throws {
        let queuePath = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pending-stores-collector-empty-\(UUID().uuidString).jsonl")
        let restoreQueuePath = setDashboardPendingStoreQueuePath(queuePath)
        let fallbackPath = fallbackReplayRoot
            .appendingPathComponent("brainlayer", isDirectory: true)
            .appendingPathComponent("docs.local", isDirectory: true)
            .appendingPathComponent("decisions", isDirectory: true)
            .appendingPathComponent("collector-pending.md")
        defer {
            restoreQueuePath()
            try? FileManager.default.removeItem(at: queuePath)
        }
        try FileManager.default.createDirectory(
            at: fallbackPath.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try """
        ---
        intended_brain_store: true
        importance: 10
        timestamp: 1970-01-01T00:27:00Z
        chunk_id:
        ---
        collector fallback body
        """.write(to: fallbackPath, atomically: true, encoding: .utf8)

        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        defer { collector.stop() }

        collector.refresh(force: true)
        try await waitForCollector(collector) { $0.stats.pendingStoreQueueDepth == 1 }
        XCTAssertEqual(collector.stats.pendingStoreFlushQueueDepth, 0)
        XCTAssertEqual(collector.stats.pendingStoreFlushRatePerMinute, 0)

        try FileManager.default.removeItem(at: fallbackPath)
        collector.refresh(force: true)
        try await waitForCollector(collector) { $0.stats.pendingStoreQueueDepth == 0 }

        XCTAssertEqual(collector.stats.pendingStoreFlushQueueDepth, 0)
        XCTAssertEqual(collector.stats.pendingStoreFlushRatePerMinute, 0)
    }

    @MainActor
    func testStatsCollectorDoesNotResampleAgentActivityOnEveryMutationRefresh() throws {
        let sampleCounter = AgentActivitySampleCounter()
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            agentActivityMonitor: AgentActivityMonitor(snapshotProvider: {
                sampleCounter.snapshot()
            })
        )
        defer { collector.stop() }

        collector.refresh(force: true)
        collector.refresh(force: false)
        collector.refresh(force: false)

        XCTAssertEqual(sampleCounter.count, 1)
    }

    @MainActor
    func testStatsCollectorCoalescesRapidMutationStatsRefreshes() async throws {
        let queuePath = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pending-stores-coalesce-\(UUID().uuidString).jsonl")
        let restoreQueuePath = setDashboardPendingStoreQueuePath(queuePath)
        defer {
            restoreQueuePath()
            try? FileManager.default.removeItem(at: queuePath)
        }

        try writeDashboardPendingStoreQueue(count: 3, to: queuePath)
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            statsRefreshCoalesceInterval: 0.05
        )
        defer { collector.stop() }

        collector.refresh(force: true)
        try await waitForCollector(collector) { $0.stats.pendingStoreQueueDepth == 3 }
        XCTAssertEqual(collector.stats.pendingStoreQueueDepth, 3)

        try writeDashboardPendingStoreQueue(count: 2, to: queuePath)
        collector.refresh(force: false)
        try await waitForCollector(collector) { $0.stats.pendingStoreQueueDepth == 2 }
        XCTAssertEqual(collector.stats.pendingStoreQueueDepth, 2)

        try writeDashboardPendingStoreQueue(count: 1, to: queuePath)
        collector.refresh(force: false)

        XCTAssertEqual(collector.stats.pendingStoreQueueDepth, 2)

        try await waitForCollector(collector) { $0.stats.pendingStoreQueueDepth == 1 }
        XCTAssertEqual(collector.stats.pendingStoreQueueDepth, 1)
    }

    @MainActor
    func testStatsCollectorAutoRefreshesWithoutBrainBusEvents() async throws {
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            statsRefreshCoalesceInterval: 60,
            autoRefreshInterval: 0.05,
            brainBusEvents: nil
        )
        defer { collector.stop() }

        collector.start()
        XCTAssertEqual(collector.stats.chunkCount, 0)

        try db.insertChunk(
            id: "auto-refresh-after-silence",
            content: "Fallback refresh should pick this up",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )

        try await waitForCollector(collector) { $0.stats.chunkCount == 1 }

        XCTAssertEqual(collector.stats.chunkCount, 1)
    }

    @MainActor
    func testStatsCollectorRetriesAfterFailedNonForcedRefresh() async throws {
        let brokenPath = NSTemporaryDirectory() + "brainbar-dashboard-broken-\(UUID().uuidString).db"
        try FileManager.default.createDirectory(
            atPath: brokenPath,
            withIntermediateDirectories: false
        )
        defer {
            try? FileManager.default.removeItem(atPath: brokenPath)
            try? FileManager.default.removeItem(atPath: brokenPath + "-wal")
            try? FileManager.default.removeItem(atPath: brokenPath + "-shm")
        }

        let collector = StatsCollector(
            dbPath: brokenPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier),
            statsRefreshCoalesceInterval: 60
        )
        defer { collector.stop() }

        collector.refresh(force: false)
        try await waitForCollector(collector) { !$0.isRefreshing }
        XCTAssertEqual(collector.stats.chunkCount, 0)

        try FileManager.default.removeItem(atPath: brokenPath)
        let repairedDB = BrainDatabase(path: brokenPath)
        try repairedDB.insertChunk(
            id: "recovered-dashboard",
            content: "Recovered dashboard refresh",
            sessionId: "dashboard",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 5
        )
        repairedDB.close()

        collector.refresh(force: false)
        try await waitForCollector(collector) { $0.stats.chunkCount == 1 }

        XCTAssertEqual(collector.stats.chunkCount, 1)
    }

    func test_DaemonHealthMonitor_returns_non_nil_when_PID_provided() throws {
        let monitor = DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)

        let snapshot = try XCTUnwrap(monitor.sample())

        XCTAssertEqual(snapshot.pid, ProcessInfo.processInfo.processIdentifier)
        XCTAssertTrue(snapshot.isResponsive)
    }

    @MainActor
    func testMakeUIStatsCollectorUsesDiscoveredDaemonPIDWhenProvided() throws {
        let collector = BrainBarAppSupport.makeUIStatsCollector(
            dbPath: tempDBPath,
            brainBusEvents: nil,
            daemonPIDProvider: { ProcessInfo.processInfo.processIdentifier }
        )
        defer { collector.stop() }

        collector.refresh(force: true)

        let snapshot = try XCTUnwrap(collector.daemon)
        XCTAssertEqual(snapshot.pid, ProcessInfo.processInfo.processIdentifier)
    }

    func testDaemonPIDFileIgnoresPIDForNonDaemonProcess() throws {
        let pidFile = URL(fileURLWithPath: tempDBPath).deletingLastPathComponent()
            .appendingPathComponent("brainbar-daemon-\(UUID().uuidString).pid")
        try "\(ProcessInfo.processInfo.processIdentifier)\n".write(
            to: pidFile,
            atomically: true,
            encoding: .utf8
        )
        defer { try? FileManager.default.removeItem(at: pidFile) }

        XCTAssertNil(BrainBarAppSupport.daemonPIDFromFile(pidFile.path))
    }

    func test_DashboardFlowSummary_renders_lanes_with_nil_daemon_snapshot() {
        let now = Date()
        let stats = DashboardStats(
            chunkCount: 12,
            enrichedChunkCount: 9,
            pendingEnrichmentCount: 3,
            enrichmentPercent: 75,
            enrichmentRatePerMinute: 1.5,
            databaseSizeBytes: 8192,
            recentActivityBuckets: [0, 1, 2, 0],
            recentEnrichmentBuckets: [0, 0, 1, 1],
            activityWindowMinutes: 30,
            bucketCount: 4,
            liveWindowMinutes: 1,
            lastWriteAt: now.addingTimeInterval(-15),
            lastEnrichedAt: now.addingTimeInterval(-20)
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)

        XCTAssertEqual(summary.ingress.status, .live)
        XCTAssertEqual(summary.enrichment.status, .live)
        XCTAssertNotEqual(summary.queue.status, .unavailable)
        XCTAssertEqual(summary.ingress.volumeText, "3 in 30m")
        XCTAssertEqual(summary.enrichment.volumeText, "2 in 30m")
    }

    func testPipelineStateTreatsMissingDaemonSnapshotAsIdleWhenDatabaseIsSettled() {
        let stats = DashboardStats(
            chunkCount: 10,
            enrichedChunkCount: 10,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4096,
            recentActivityBuckets: [0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0]
        )

        let state = PipelineState.derive(daemon: nil, stats: stats)

        XCTAssertEqual(state, .idle)
    }

    func testPipelineStateDerivesDatabaseActivityWhenDaemonIsUnhealthy() {
        let stats = DashboardStats(
            chunkCount: 20,
            enrichedChunkCount: 10,
            pendingEnrichmentCount: 10,
            enrichmentPercent: 50,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 8192,
            recentActivityBuckets: [0, 2, 3, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0]
        )
        let daemon = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: false,
            rssBytes: 1_024 * 1_024,
            uptime: 120,
            openConnections: 0,
            lastSeenAt: Date()
        )

        let state = PipelineState.derive(daemon: daemon, stats: stats)

        XCTAssertEqual(state, .indexing)
    }

    func testPipelineStateReportsIndexingForRecentWriteBurst() {
        let stats = DashboardStats(
            chunkCount: 20,
            enrichedChunkCount: 20,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 2.5,
            databaseSizeBytes: 8192,
            recentActivityBuckets: [0, 0, 4, 6],
            recentEnrichmentBuckets: [0, 0, 1, 2]
        )
        let daemon = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: true,
            rssBytes: 1_024 * 1_024,
            uptime: 120,
            openConnections: 2,
            lastSeenAt: Date()
        )

        let state = PipelineState.derive(daemon: daemon, stats: stats)

        XCTAssertEqual(state, .indexing)
    }

    func testPipelineStateReportsEnrichingWhenBacklogExistsWithoutFreshWrites() {
        let stats = DashboardStats(
            chunkCount: 20,
            enrichedChunkCount: 12,
            pendingEnrichmentCount: 8,
            enrichmentPercent: 60,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 8192,
            recentActivityBuckets: [0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0]
        )
        let daemon = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: true,
            rssBytes: 1_024 * 1_024,
            uptime: 120,
            openConnections: 2,
            lastSeenAt: Date()
        )

        let state = PipelineState.derive(daemon: daemon, stats: stats)

        XCTAssertEqual(state, .enriching)
    }

    func testPipelineStateReportsIdleForHealthySettledSystem() {
        let stats = DashboardStats(
            chunkCount: 20,
            enrichedChunkCount: 20,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 8192,
            recentActivityBuckets: [0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0]
        )
        let daemon = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: true,
            rssBytes: 1_024 * 1_024,
            uptime: 120,
            openConnections: 1,
            lastSeenAt: Date()
        )

        let state = PipelineState.derive(daemon: daemon, stats: stats)

        XCTAssertEqual(state, .idle)
    }

    @MainActor
    func testStatusPopoverViewIsAppKitViewController() {
        let collector = StatsCollector(
            dbPath: tempDBPath,
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        defer { collector.stop() }

        let viewController = StatusPopoverView(collector: collector)

        XCTAssertFalse(viewController.isViewLoaded)
        _ = viewController.view
        XCTAssertTrue(viewController.isViewLoaded)
    }

    @MainActor
    private func waitForCollector(
        _ collector: StatsCollector,
        timeout: TimeInterval = 2.0,
        until predicate: (StatsCollector) -> Bool
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !predicate(collector), Date() < deadline {
            try await Task.sleep(for: .milliseconds(50))
        }
    }

    private static func absoluteTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: date)
    }

    private static func shortTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        return formatter.string(from: date)
    }
}

private final class RecordingBrainBusEventSource: BrainBusEventSource, @unchecked Sendable {
    private let lock = NSLock()
    private var requests = 0
    private var continuation: AsyncStream<BrainBusEvent>.Continuation?

    var streamRequestCount: Int {
        lock.withLock { requests }
    }

    func events() -> AsyncStream<BrainBusEvent> {
        lock.withLock {
            requests += 1
        }
        return AsyncStream { continuation in
            lock.withLock {
                self.continuation = continuation
            }
        }
    }

    func publish(_ event: BrainBusEvent) {
        lock.withLock { continuation }?.yield(event)
    }
}

private final class AgentActivitySampleCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var samples = 0

    deinit {}

    var count: Int {
        lock.withLock { samples }
    }

    func snapshot() -> String {
        lock.withLock {
            samples += 1
        }
        return ""
    }
}

private func setDashboardPendingStoreQueuePath(_ path: URL) -> () -> Void {
    let previous = ProcessInfo.processInfo.environment["BRAINBAR_PENDING_STORES_PATH"]
    setenv("BRAINBAR_PENDING_STORES_PATH", path.path, 1)
    return {
        if let previous {
            setenv("BRAINBAR_PENDING_STORES_PATH", previous, 1)
        } else {
            unsetenv("BRAINBAR_PENDING_STORES_PATH")
        }
    }
}

private func setDashboardWatcherHealthPath(_ path: URL) -> () -> Void {
    let previous = ProcessInfo.processInfo.environment["BRAINLAYER_WATCHER_HEALTH_PATH"]
    setenv("BRAINLAYER_WATCHER_HEALTH_PATH", path.path, 1)
    return {
        if let previous {
            setenv("BRAINLAYER_WATCHER_HEALTH_PATH", previous, 1)
        } else {
            unsetenv("BRAINLAYER_WATCHER_HEALTH_PATH")
        }
    }
}

private func setDashboardFallbackReplayGitsRoot(_ path: URL) -> () -> Void {
    let previous = ProcessInfo.processInfo.environment["BRAINBAR_FALLBACK_REPLAY_GITS_ROOT"]
    setenv("BRAINBAR_FALLBACK_REPLAY_GITS_ROOT", path.path, 1)
    return {
        if let previous {
            setenv("BRAINBAR_FALLBACK_REPLAY_GITS_ROOT", previous, 1)
        } else {
            unsetenv("BRAINBAR_FALLBACK_REPLAY_GITS_ROOT")
        }
    }
}

private func brainBarSourceFile(_ relativePath: String, testFilePath: StaticString = #filePath) throws -> String {
    let packageRoot = URL(fileURLWithPath: "\(testFilePath)")
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let sourceURL = packageRoot.appendingPathComponent(relativePath)
    return try String(contentsOf: sourceURL, encoding: .utf8)
}

private func writeDashboardPendingStoreQueue(count: Int, to path: URL) throws {
    let queuedAt = ISO8601DateFormatter().string(from: Date())
    let lines = (0..<count).map { index in
        """
        {"content":"queued item \(index)","tags":["queue"],"importance":5,"source":"mcp","queued_at":"\(queuedAt)"}
        """
    }
    try lines.joined(separator: "\n").appending("\n").write(to: path, atomically: true, encoding: .utf8)
}
