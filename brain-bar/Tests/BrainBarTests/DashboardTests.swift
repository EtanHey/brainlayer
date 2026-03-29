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
        XCTAssertGreaterThanOrEqual(stats.recentActivityBuckets.reduce(0, +), 2)
        XCTAssertGreaterThan(stats.databaseSizeBytes, 0)
    }

    func testDashboardStatsReturnsZeroPercentForEmptyDatabase() throws {
        let stats = try db.dashboardStats(activityWindowMinutes: 15, bucketCount: 4)

        XCTAssertEqual(stats.chunkCount, 0)
        XCTAssertEqual(stats.enrichedChunkCount, 0)
        XCTAssertEqual(stats.pendingEnrichmentCount, 0)
        XCTAssertEqual(stats.enrichmentPercent, 0.0, accuracy: 0.001)
        XCTAssertEqual(stats.recentActivityBuckets, [0, 0, 0, 0])
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

    func testPipelineStateIsOfflineWhenDaemonSnapshotMissing() {
        let stats = DashboardStats(
            chunkCount: 10,
            enrichedChunkCount: 10,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            databaseSizeBytes: 4096,
            recentActivityBuckets: [0, 0, 0, 0]
        )

        let state = PipelineState.derive(daemon: nil, stats: stats)

        XCTAssertEqual(state, .offline)
    }

    func testPipelineStatePrefersDegradedOverBusyWhenDaemonIsUnhealthy() {
        let stats = DashboardStats(
            chunkCount: 20,
            enrichedChunkCount: 10,
            pendingEnrichmentCount: 10,
            enrichmentPercent: 50,
            databaseSizeBytes: 8192,
            recentActivityBuckets: [0, 2, 3, 0]
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
            databaseSizeBytes: 8192,
            recentActivityBuckets: [0, 0, 4, 6]
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
            databaseSizeBytes: 8192,
            recentActivityBuckets: [0, 0, 0, 0]
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
            databaseSizeBytes: 8192,
            recentActivityBuckets: [0, 0, 0, 0]
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
}
