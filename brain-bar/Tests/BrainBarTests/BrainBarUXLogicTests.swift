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

    func testDashboardMetricFormatterUsesPerSecondForFastRates() {
        XCTAssertEqual(
            DashboardMetricFormatter.speedString(ratePerMinute: 90),
            "1.5/s"
        )
    }

    func testDashboardMetricFormatterUsesPerMinuteForSubSecondRates() {
        XCTAssertEqual(
            DashboardMetricFormatter.speedString(ratePerMinute: 18),
            "18/min"
        )
    }

    func testDashboardMetricFormatterUsesPerHourForVerySlowRates() {
        XCTAssertEqual(
            DashboardMetricFormatter.speedString(ratePerMinute: 0.5),
            "30/hr"
        )
    }

    func testPipelineActivityTracksSplitIndexingFromEnrichment() {
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 100,
            pendingEnrichmentCount: 20,
            enrichmentPercent: 83.3,
            enrichmentRatePerMinute: 18,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6],
            recentEnrichmentBuckets: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3]
        )
        let daemon = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: true,
            rssBytes: 1_024,
            uptime: 60,
            openConnections: 1,
            lastSeenAt: Date()
        )

        let tracks = PipelineActivityTracks.derive(daemon: daemon, stats: stats)

        XCTAssertEqual(tracks.indexing.symbolName, "server.rack")
        XCTAssertEqual(tracks.indexing.rateText, "2/min")
        XCTAssertEqual(tracks.enriching.symbolName, "sparkles")
        XCTAssertEqual(tracks.enriching.rateText, "18/min")
        XCTAssertEqual(tracks.enriching.status, .live)
        XCTAssertEqual(tracks.indexing.status, .live)
    }

    func testPipelineActivityTracksShowQueuedEnrichmentWithNoRecentThroughput() {
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 90,
            pendingEnrichmentCount: 30,
            enrichmentPercent: 75,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: Array(repeating: 0, count: 12),
            recentEnrichmentBuckets: Array(repeating: 0, count: 12)
        )
        let daemon = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: true,
            rssBytes: 1_024,
            uptime: 60,
            openConnections: 1,
            lastSeenAt: Date()
        )

        let tracks = PipelineActivityTracks.derive(daemon: daemon, stats: stats)

        XCTAssertEqual(tracks.enriching.status, .queued)
        XCTAssertEqual(tracks.enriching.rateText, "queued")
        XCTAssertEqual(tracks.indexing.rateText, "idle")
    }

    func testPipelineActivityTracksUseRecentEnrichmentThroughputWhenInstantaneousRateDropsToZero() {
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 100,
            pendingEnrichmentCount: 20,
            enrichmentPercent: 83.3,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: Array(repeating: 0, count: 12),
            recentEnrichmentBuckets: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6]
        )
        let daemon = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: true,
            rssBytes: 1_024,
            uptime: 60,
            openConnections: 1,
            lastSeenAt: Date()
        )

        let tracks = PipelineActivityTracks.derive(daemon: daemon, stats: stats)

        XCTAssertEqual(tracks.enriching.status, .live)
        XCTAssertEqual(tracks.enriching.rateText, "2/min")
    }

    func testDashboardMetricFormatterSummarizesSecondaryRateUnits() {
        XCTAssertEqual(
            DashboardMetricFormatter.rateDetailString(ratePerMinute: 18),
            "0.3/s"
        )
    }

    func testDashboardMetricFormatterGuardsAgainstNonFiniteRates() {
        XCTAssertEqual(
            DashboardMetricFormatter.speedString(ratePerMinute: .infinity),
            "0/min"
        )
        XCTAssertEqual(
            DashboardMetricFormatter.rateDetailString(ratePerMinute: .infinity),
            "0/s"
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
