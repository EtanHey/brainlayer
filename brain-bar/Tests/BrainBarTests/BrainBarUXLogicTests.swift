import XCTest
@testable import BrainBar

final class BrainBarUXLogicTests: XCTestCase {
    func testPipelineIndicatorsCanShowIndexingAndEnrichingLiveAtTheSameTime() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 100,
            pendingEnrichmentCount: 20,
            enrichmentPercent: 83.3,
            enrichmentRatePerMinute: 24,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 2, 4],
            recentEnrichmentBuckets: [0, 0, 0, 1, 3],
            lastWriteAt: now.addingTimeInterval(-10),
            lastEnrichedAt: now.addingTimeInterval(-6)
        )
        let daemon = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: true,
            rssBytes: 1_024,
            uptime: 60,
            openConnections: 1,
            lastSeenAt: now
        )

        let indicators = PipelineIndicators.derive(daemon: daemon, stats: stats, now: now)

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
        let now = Date(timeIntervalSince1970: 1_000_000)

        XCTAssertEqual(
            DashboardMetricFormatter.lastCompletionString(
                lastEventAt: now.addingTimeInterval(-30),
                activityWindowMinutes: 60,
                now: now
            ),
            "Just now"
        )
        XCTAssertEqual(
            DashboardMetricFormatter.lastCompletionString(
                lastEventAt: now.addingTimeInterval(-90),
                activityWindowMinutes: 60,
                now: now
            ),
            "2m ago"
        )
    }

    func testDashboardFlowSummaryCanShowIngressAndEnrichmentLiveTogether() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 100,
            pendingEnrichmentCount: 20,
            enrichmentPercent: 83.3,
            enrichmentRatePerMinute: 24,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 2, 4],
            recentEnrichmentBuckets: [0, 0, 0, 1, 3],
            activityWindowMinutes: 60,
            bucketCount: 5,
            liveWindowMinutes: 1,
            lastWriteAt: now.addingTimeInterval(-15),
            lastEnrichedAt: now.addingTimeInterval(-10)
        )
        let daemon = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: true,
            rssBytes: 1_024,
            uptime: 60,
            openConnections: 1,
            lastSeenAt: now
        )

        let summary = DashboardFlowSummary.derive(daemon: daemon, stats: stats, now: now)

        XCTAssertEqual(summary.ingress.status, .live)
        XCTAssertEqual(summary.queue.status, .stable)
        XCTAssertEqual(summary.enrichment.status, .live)
        XCTAssertEqual(summary.windowLabel, "Last 1h")
    }

    func testDashboardFlowSummaryKeepsRecentEnrichmentDistinctFromLiveNow() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 118,
            pendingEnrichmentCount: 2,
            enrichmentPercent: 98.3,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 1, 0, 0],
            activityWindowMinutes: 60,
            bucketCount: 5,
            liveWindowMinutes: 1,
            lastWriteAt: now.addingTimeInterval(-600),
            lastEnrichedAt: now.addingTimeInterval(-90)
        )
        let daemon = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: true,
            rssBytes: 1_024,
            uptime: 60,
            openConnections: 1,
            lastSeenAt: now
        )

        let summary = DashboardFlowSummary.derive(daemon: daemon, stats: stats, now: now)

        XCTAssertEqual(summary.ingress.status, .idle)
        XCTAssertEqual(summary.queue.status, .draining)
        XCTAssertEqual(summary.enrichment.status, .recent)
        XCTAssertEqual(summary.enrichment.lastEventText, "2m ago")
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
        let now = Date(timeIntervalSince1970: 1_000_000)
        let activeStats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 100,
            pendingEnrichmentCount: 20,
            enrichmentPercent: 83.3,
            enrichmentRatePerMinute: 24,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 2, 4],
            recentEnrichmentBuckets: [0, 0, 0, 1, 3],
            lastEnrichedAt: now.addingTimeInterval(-8)
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
            BrainBarLivePresentation.derive(stats: activeStats, now: now).statusText,
            "Enrichments in the last 60s"
        )
        XCTAssertEqual(
            BrainBarLivePresentation.derive(stats: idleStats, now: now).statusText,
            "No enrichments in the last 60s"
        )
    }

    func testDashboardLayoutStacksFlowCardsOnNarrowWidths() {
        let layout = BrainBarDashboardLayout(containerSize: CGSize(width: 820, height: 640))

        XCTAssertEqual(layout.chartColumns, 1)
        XCTAssertEqual(layout.overviewMetricColumns, 2)
        XCTAssertEqual(layout.diagnosticColumns, 1)
        XCTAssertTrue(layout.usesQueueRail)
    }

    func testDashboardLayoutUsesSideBySideDiagnosticsInMediumWindows() {
        let layout = BrainBarDashboardLayout(containerSize: CGSize(width: 1_020, height: 640))

        XCTAssertEqual(layout.chartColumns, 2)
        XCTAssertEqual(layout.overviewMetricColumns, 4)
        XCTAssertEqual(layout.diagnosticColumns, 2)
        XCTAssertTrue(layout.usesQueueRail)
    }

    func testDashboardLayoutExpandsToThreeColumnsWhenSpaceAllows() {
        let layout = BrainBarDashboardLayout(containerSize: CGSize(width: 1_340, height: 760))

        XCTAssertEqual(layout.chartColumns, 2)
        XCTAssertEqual(layout.overviewMetricColumns, 4)
        XCTAssertEqual(layout.diagnosticColumns, 2)
        XCTAssertTrue(layout.usesQueueRail)
    }
}
