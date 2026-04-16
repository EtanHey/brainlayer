import XCTest
@testable import BrainBar

final class BrainBarUXLogicTests: XCTestCase {
    func testPipelineIndicatorsCanShowIndexingAndEnrichingLiveAtTheSameTime() {
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 100,
            pendingEnrichmentCount: 20,
            enrichmentPercent: 83.3,
            enrichmentRatePerMinute: 24,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 2, 4],
            recentEnrichmentBuckets: [0, 0, 0, 1, 3]
        )
        let daemon = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: true,
            rssBytes: 1_024,
            uptime: 60,
            openConnections: 1,
            lastSeenAt: Date()
        )

        let indicators = PipelineIndicators.derive(daemon: daemon, stats: stats)

        XCTAssertEqual(indicators.indexing.status, .live)
        XCTAssertEqual(indicators.enriching.status, .live)
    }

    func testPipelineIndicatorsShowQueuedEnrichmentWithoutRecentCompletions() {
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 100,
            pendingEnrichmentCount: 20,
            enrichmentPercent: 83.3,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0, 0]
        )
        let daemon = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: true,
            rssBytes: 1_024,
            uptime: 60,
            openConnections: 1,
            lastSeenAt: Date()
        )

        let indicators = PipelineIndicators.derive(daemon: daemon, stats: stats)

        XCTAssertEqual(indicators.indexing.status, .idle)
        XCTAssertEqual(indicators.enriching.status, .queued)
    }

    func testDashboardMetricFormatterUsesChunksPerSecond() {
        XCTAssertEqual(
            DashboardMetricFormatter.speedString(ratePerMinute: 22.2),
            "0.37/s"
        )
    }

    func testIncomingRelationDisplayPutsEntityNameBeforeRelationVerb() {
        let relation = EntityCard.Relation(
            relationType: "coaches",
            targetName: "coachClaude",
            direction: "incoming"
        )

        XCTAssertEqual(relation.displayText, "coachClaude coaches")
    }

    func testOutgoingRelationDisplayKeepsRelationVerbBeforeEntityName() {
        let relation = EntityCard.Relation(
            relationType: "owns",
            targetName: "brainlayer",
            direction: "outgoing"
        )

        XCTAssertEqual(relation.displayText, "owns brainlayer")
    }
}
