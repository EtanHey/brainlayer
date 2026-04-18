import AppKit
import XCTest
@testable import BrainBar

final class DashboardTests: XCTestCase {
    private var db: BrainDatabase!
    private var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-dashboard-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
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
        db.exec("UPDATE chunks SET enriched_at = datetime('now') WHERE id = 'dash-2'")

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

    func testDashboardStatsReturnsZeroPercentForEmptyDatabase() throws {
        let stats = try db.dashboardStats(activityWindowMinutes: 15, bucketCount: 4)

        XCTAssertEqual(stats.chunkCount, 0)
        XCTAssertEqual(stats.enrichedChunkCount, 0)
        XCTAssertEqual(stats.pendingEnrichmentCount, 0)
        XCTAssertEqual(stats.enrichmentPercent, 0.0, accuracy: 0.001)
        XCTAssertEqual(stats.recentActivityBuckets, [0, 0, 0, 0])
        XCTAssertEqual(stats.recentEnrichmentBuckets, [0, 0, 0, 0])
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
                enriched_at = datetime('now')
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
            SET enriched_at = datetime('now', '-6 minutes')
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
            SET enriched_at = datetime('now', '-90 seconds')
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
                enriched_at = datetime('now', '-90 seconds')
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
            daemonMonitor: DaemonHealthMonitor(targetPID: ProcessInfo.processInfo.processIdentifier)
        )
        defer { collector.stop() }

        collector.start()
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

    func testPipelineStateTreatsMissingDaemonSnapshotAsDegraded() {
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

        XCTAssertEqual(state, .degraded)
    }

    func testPipelineStatePrefersDegradedOverBusyWhenDaemonIsUnhealthy() {
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

        XCTAssertEqual(state, .degraded)
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
}
