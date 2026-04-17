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

    func testDashboardMetricFormatterUsesChunksPerMinute() {
        XCTAssertEqual(
            DashboardMetricFormatter.speedString(ratePerMinute: 22.2),
            "22.2/min"
        )
    }

    func testDashboardMetricFormatterMakesIndexingLabelExplicit() {
        XCTAssertEqual(
            DashboardMetricFormatter.indexingString(
                recentActivityBuckets: [0, 0, 6, 9],
                activityWindowMinutes: 30
            ),
            "0.5/min"
        )
    }

    func testDashboardMetricFormatterSummarizesRecentWritesWithoutRepeatingRateUnits() {
        XCTAssertEqual(
            DashboardMetricFormatter.activitySummaryString(
                recentActivityBuckets: [0, 0, 6, 9],
                activityWindowMinutes: 30
            ),
            "15 in 30m"
        )
    }

    func testDashboardMetricFormatterReportsApproximateLastCompletionAge() {
        XCTAssertEqual(
            DashboardMetricFormatter.lastCompletionString(
                recentEnrichmentBuckets: [0, 0, 0, 0, 1, 2],
                activityWindowMinutes: 30
            ),
            "Just now"
        )
        XCTAssertEqual(
            DashboardMetricFormatter.lastCompletionString(
                recentEnrichmentBuckets: [0, 1, 0, 0, 0, 0],
                activityWindowMinutes: 30
            ),
            "20m ago"
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

    func testLivePulseTriggersWhenSparklineBucketsChange() {
        XCTAssertTrue(
            BrainBarLivePulse.shouldPulse(
                previous: [0, 0, 0, 0, 0, 0],
                current: [0, 0, 0, 0, 1, 0]
            )
        )
    }

    func testLivePulseDoesNotTriggerWhenSparklineBucketsStayTheSame() {
        XCTAssertFalse(
            BrainBarLivePulse.shouldPulse(
                previous: [0, 0, 0, 1, 2, 3],
                current: [0, 0, 0, 1, 2, 3]
            )
        )
    }

    func testLivePresentationUsesExplicitActiveAndIdleStatusText() {
        let activeStats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 100,
            pendingEnrichmentCount: 20,
            enrichmentPercent: 83.3,
            enrichmentRatePerMinute: 24,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 2, 4],
            recentEnrichmentBuckets: [0, 0, 0, 1, 3]
        )
        let idleStats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 120,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0, 0]
        )

        XCTAssertEqual(
            BrainBarLivePresentation.derive(stats: activeStats).statusText,
            "Live enrichment stream"
        )
        XCTAssertEqual(
            BrainBarLivePresentation.derive(stats: idleStats).statusText,
            "Idle — no enrichment in last 60s"
        )
    }

    func testDashboardLayoutCompactsForShortDashboardHeights() {
        let layout = BrainBarDashboardLayout(containerSize: CGSize(width: 900, height: 500))

        XCTAssertEqual(layout.outerPadding, 14)
        XCTAssertLessThan(layout.scale, 1)
    }

    func testDashboardLayoutUsesCompactTokensForNarrowWindowWidths() {
        let layout = BrainBarDashboardLayout(containerSize: CGSize(width: 820, height: 640))

        XCTAssertEqual(layout.metricValueFontSize, 20)
        XCTAssertEqual(layout.sparklineWidth, 280)
    }
}
