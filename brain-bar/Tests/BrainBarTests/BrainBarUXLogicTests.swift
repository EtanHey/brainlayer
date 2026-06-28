import XCTest
import AppKit
import SwiftUI
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
            "\(Self.absoluteTime(now.addingTimeInterval(-30))) (Just now)"
        )
        XCTAssertEqual(
            DashboardMetricFormatter.lastCompletionString(
                lastEventAt: now.addingTimeInterval(-90),
                activityWindowMinutes: 60,
                now: now
            ),
            "\(Self.absoluteTime(now.addingTimeInterval(-90))) (1m ago)"
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

    func testJsonlWatcherLaneDoesNotBorrowLiveStatusFromAgentStores() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 120,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 3],
            recentAgentWriteBuckets: [0, 0, 0, 3],
            recentWatcherWriteBuckets: [0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0],
            activityWindowMinutes: 30,
            bucketCount: 4,
            liveWindowMinutes: 1,
            lastWriteAt: now.addingTimeInterval(-15),
            watcherHealth: DashboardStats.WatcherHealth(
                alerting: false,
                filesTracked: 12,
                maxOffsetLagBytes: 0,
                activeEntriesPerMinute: 0,
                realtimeInsertsPerMinute: 0,
                updatedAt: now
            )
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
        let watcherLane = summary.lane(for: .jsonlWatcher)

        XCTAssertEqual(summary.ingress.status, .live)
        XCTAssertEqual(watcherLane.status, .idle)
        XCTAssertEqual(watcherLane.statusText, "No JSONL watcher writes")
        XCTAssertEqual(watcherLane.lastEventText, "No watcher writes in Last 30m")
    }

    func testAgentStoresLaneDoesNotBorrowLiveStatusFromJsonlWatcher() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 120,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 3],
            recentAgentWriteBuckets: [0, 0, 0, 0],
            recentWatcherWriteBuckets: [0, 0, 0, 3],
            recentEnrichmentBuckets: [0, 0, 0, 0],
            activityWindowMinutes: 30,
            bucketCount: 4,
            liveWindowMinutes: 1,
            lastWriteAt: now.addingTimeInterval(-15)
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
        let agentLane = summary.lane(for: .agentStores)

        XCTAssertEqual(summary.ingress.status, .live)
        XCTAssertEqual(agentLane.status, .idle)
        XCTAssertEqual(agentLane.statusText, "No agent-store writes")
        XCTAssertEqual(agentLane.lastEventText, "No agent-store writes in Last 30m")
    }

    func testAgentStoresLaneShowsPendingReplayDebtWhenCommittedGraphIsZero() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
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
            activityWindowMinutes: 60,
            bucketCount: 4,
            liveWindowMinutes: 1,
            pendingStoreQueueDepth: 5,
            pendingStoreOldestQueuedAt: now.addingTimeInterval(-120),
            pendingStoreFlushRatePerMinute: 0
        )

        let agentLane = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now).lane(for: .agentStores)

        XCTAssertEqual(agentLane.status, .queued)
        XCTAssertEqual(agentLane.statusText, "5 agent stores queued for replay")
        XCTAssertEqual(agentLane.volumeText, "0 in 1h, 5 queued")
        XCTAssertEqual(agentLane.lastEventText, "5 agent stores queued for replay")
    }

    func testAgentStoresLaneShowsQueuedReplayDebtBeforeLiveWrites() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 120,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 2],
            recentAgentWriteBuckets: [0, 0, 0, 2],
            recentWatcherWriteBuckets: [0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0],
            activityWindowMinutes: 60,
            bucketCount: 4,
            liveWindowMinutes: 1,
            pendingStoreQueueDepth: 5,
            pendingStoreOldestQueuedAt: now.addingTimeInterval(-120),
            pendingStoreFlushRatePerMinute: 0
        )

        let agentLane = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now).lane(for: .agentStores)

        XCTAssertEqual(agentLane.status, .queued)
        XCTAssertEqual(agentLane.statusText, "5 agent stores queued for replay")
        XCTAssertEqual(agentLane.volumeText, "2 in 1h, 5 queued")
        XCTAssertEqual(agentLane.lastEventText, "5 agent stores queued for replay; 2 committed in Last 1h")
    }

    func testJsonlWatcherLaneMarksFlatGraphBrokenWhenHealthSeesActiveJsonl() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
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
            watcherHealth: DashboardStats.WatcherHealth(
                alerting: false,
                filesTracked: 12,
                maxOffsetLagBytes: 0,
                activeEntriesPerMinute: 8.5,
                realtimeInsertsPerMinute: 0,
                updatedAt: now
            )
        )

        let watcherLane = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
            .lane(for: .jsonlWatcher)

        XCTAssertEqual(watcherLane.status, .unavailable)
        XCTAssertEqual(watcherLane.statusText, "Watcher stale: JSONL activity is not landing")
    }

    func testJsonlWatcherLaneMarksStaleWatcherHealthUnavailable() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
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
            watcherHealth: DashboardStats.WatcherHealth(
                alerting: false,
                filesTracked: 12,
                maxOffsetLagBytes: 0,
                activeEntriesPerMinute: 0,
                realtimeInsertsPerMinute: 0,
                updatedAt: now.addingTimeInterval(-601)
            )
        )

        let watcherLane = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
            .lane(for: .jsonlWatcher)

        XCTAssertEqual(watcherLane.status, .unavailable)
        XCTAssertEqual(watcherLane.statusText, "Watcher health stale")
    }

    func testJsonlWatcherLaneMarksMissingWatcherHealthUnavailable() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
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
            bucketCount: 4
        )

        let watcherLane = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
            .lane(for: .jsonlWatcher)

        XCTAssertEqual(watcherLane.status, .unavailable)
        XCTAssertEqual(watcherLane.statusText, "Watcher health unavailable")
    }

    func testJsonlWatcherLaneMarksTimestamplessWatcherHealthUnavailable() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
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
            watcherHealth: DashboardStats.WatcherHealth(
                alerting: false,
                filesTracked: 12,
                maxOffsetLagBytes: 0,
                activeEntriesPerMinute: 0,
                realtimeInsertsPerMinute: 0
            )
        )

        let watcherLane = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
            .lane(for: .jsonlWatcher)

        XCTAssertEqual(watcherLane.status, .unavailable)
        XCTAssertEqual(watcherLane.statusText, "Watcher health stale")
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
        XCTAssertEqual(summary.enrichment.lastEventText, "\(Self.absoluteTime(now.addingTimeInterval(-90))) (1m ago)")
    }

    func testDashboardFlowSummaryLabelsRightEdgeEnrichmentBurstAsBacklogDrain() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 100_000,
            enrichedChunkCount: 20_000,
            pendingEnrichmentCount: 80_000,
            enrichmentPercent: 20,
            enrichmentRatePerMinute: 34.25,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 1],
            recentEnrichmentBuckets: [0, 0, 0, 2_055],
            activityWindowMinutes: 60,
            bucketCount: 4,
            liveWindowMinutes: 1,
            lastEnrichedAt: now.addingTimeInterval(-10)
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)

        XCTAssertEqual(summary.enrichment.sparklineLabel, "Enrichment completions over Last 1h")
        XCTAssertEqual(summary.enrichment.latestBucketName, "latest enrichment bucket")
        XCTAssertEqual(summary.enrichment.statusText, "Backlog drain burst: 2055 enriched in latest 15m")
        XCTAssertEqual(summary.enrichment.volumeText, "2055 in 1h")
    }

    func testDashboardQueueSummaryReportsActiveDrainingForSmallFreshStoreQueue() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 120,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0],
            pendingStoreQueueDepth: 6,
            pendingStoreOldestQueuedAt: now.addingTimeInterval(-2),
            pendingStoreFlushRatePerMinute: 60
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)

        XCTAssertEqual(summary.queue.storeHealth, .activeDraining)
        XCTAssertEqual(summary.queue.storeHealthText, "active draining")
        XCTAssertEqual(summary.queue.storeDepthText, "6 queued")
        XCTAssertEqual(summary.queue.storeOldestAgeText, "oldest 2s")
        XCTAssertEqual(summary.queue.storeFlushRateText, "60/min")
        XCTAssertEqual(summary.queue.title, "Queue active draining")
    }

    func testDashboardQueueSummaryEscalatesToBacklogAccumulatingByDepthOrAge() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 120,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0],
            pendingStoreQueueDepth: 50,
            pendingStoreOldestQueuedAt: now.addingTimeInterval(-12),
            pendingStoreFlushRatePerMinute: 5
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)

        XCTAssertEqual(summary.queue.storeHealth, .backlogAccumulating)
        XCTAssertEqual(summary.queue.storeHealthText, "backlog accumulating")
        XCTAssertEqual(summary.queue.title, "Queue backlog accumulating")
    }

    func testDashboardQueueSummaryEscalatesToWriterStuckByDepthOrAge() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 120,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0],
            pendingStoreQueueDepth: 7,
            pendingStoreOldestQueuedAt: now.addingTimeInterval(-300),
            pendingStoreFlushRatePerMinute: 0
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)

        XCTAssertEqual(summary.queue.storeHealth, .writerStuck)
        XCTAssertEqual(summary.queue.storeHealthText, "writer stuck - investigate")
        XCTAssertEqual(summary.queue.title, "Q: writer stuck - investigate")
    }

    @MainActor
    func testRendersWriterStuckQueueLabelQAImage() throws {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 120,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0],
            pendingStoreQueueDepth: 7,
            pendingStoreOldestQueuedAt: now.addingTimeInterval(-300),
            pendingStoreFlushRatePerMinute: 0
        )
        let title = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now).queue.title
        let view = Text(title)
            .font(.system(size: 18, weight: .semibold))
            .padding(.horizontal, 18)
            .padding(.vertical, 12)
            .background(.thinMaterial, in: Capsule())
            .frame(width: 360, height: 80)

        try renderPNG(view, name: "bug1-writer-stuck-label.png")
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
    }

    func testDashboardLayoutStacksPipelineChartsInMediumWindows() {
        let layout = BrainBarDashboardLayout(containerSize: CGSize(width: 1_020, height: 640))

        XCTAssertEqual(layout.chartColumns, 1)
        XCTAssertEqual(layout.overviewMetricColumns, 4)
        XCTAssertEqual(layout.diagnosticColumns, 2)
    }

    func testDashboardLayoutExpandsToThreeColumnsWhenSpaceAllows() {
        let layout = BrainBarDashboardLayout(containerSize: CGSize(width: 1_340, height: 760))

        XCTAssertEqual(layout.chartColumns, 2)
        XCTAssertEqual(layout.overviewMetricColumns, 4)
        XCTAssertEqual(layout.diagnosticColumns, 2)
    }

    private static func absoluteTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: date)
    }
}

@MainActor
private func renderPNG<V: View>(_ view: V, name: String) throws {
    let renderer = ImageRenderer(content: view)
    renderer.scale = 2
    guard let image = renderer.nsImage,
          let tiff = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiff),
          let png = bitmap.representation(using: .png, properties: [:]) else {
        XCTFail("Expected renderer to produce a PNG")
        return
    }

    let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("docs.local/wave3-qa/\(name)")
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try png.write(to: url)
    XCTAssertGreaterThan(png.count, 1_000)
}
