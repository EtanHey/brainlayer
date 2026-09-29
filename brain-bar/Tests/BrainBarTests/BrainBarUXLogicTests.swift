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

    func testPipelineStateIsIdleForPendingBacklogWhenEnrichmentIsOff() {
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

        let state = PipelineState.derive(daemon: daemon, stats: stats)

        XCTAssertEqual(state, .idle)
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

    func testPipelinePulseGateIgnoresLiveBucketChangesWhileWindowedTimeframeIsSelected() {
        XCTAssertTrue(
            BrainBarPipelinePulseGate.shouldPulse(
                previous: [0, 0, 0],
                current: [0, 0, 1],
                timeframe: .live
            )
        )
        XCTAssertFalse(
            BrainBarPipelinePulseGate.shouldPulse(
                previous: [0, 0, 0],
                current: [0, 0, 1],
                timeframe: .threeHour
            )
        )
        XCTAssertFalse(
            BrainBarPipelinePulseGate.shouldPulse(
                previous: [0, 0, 0],
                current: [0, 0, 1],
                timeframe: .day
            )
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
            watcherHealth: WatcherHealthFileRead.readable(WatcherHealthFile(updatedAt: now, pollCount: 1, alertReasons: [], maxOffsetLagBytes: 0)),
            watcherProcessProbeResult: .running(pid: 4242)
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
        let watcherLane = summary.lane(for: .jsonlWatcher)

        XCTAssertEqual(summary.ingress.status, .live)
        XCTAssertEqual(watcherLane.status, .idle)
        XCTAssertEqual(watcherLane.statusText, "RUNNING · NO RECENT FLOW")
        XCTAssertEqual(watcherLane.lastEventText, "No watcher-ingested chunks in Last 30m")
    }

    func testWatcherStateDoesNotBorrowDaemonPIDWhenProbeEvidenceIsMissing() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 1,
            enrichedChunkCount: 1,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [1],
            recentWatcherWriteBuckets: [1],
            recentEnrichmentBuckets: [1],
            activityWindowMinutes: 60,
            bucketCount: 1,
            watcherProcessProbeResult: nil,
            watcherRecentDistinctChunkCount: 1,
            watcherFlowReadability: .readable
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

        XCTAssertEqual(summary.watcherFlowState, .unknown)
        XCTAssertEqual(summary.lane(for: .jsonlWatcher).statusText, "UNKNOWN")
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
        XCTAssertEqual(agentLane.statusText, "No agent-origin chunks")
        XCTAssertEqual(agentLane.lastEventText, "No agent-origin chunks in Last 30m")
    }

    func testWatcherOnlyBurstShowsAllCommitsWhileAgentLaneStaysIdle() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 120,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 460],
            recentAgentWriteBuckets: [0, 0, 0, 0],
            recentWatcherWriteBuckets: [0, 0, 0, 454],
            recentEnrichmentBuckets: [0, 0, 0, 0],
            activityWindowMinutes: 60,
            bucketCount: 4,
            liveWindowMinutes: 1,
            lastWriteAt: now.addingTimeInterval(-15),
            watcherHealth: WatcherHealthFileRead.readable(WatcherHealthFile(updatedAt: now, pollCount: 1, alertReasons: [], maxOffsetLagBytes: 0)),
            watcherProcessProbeResult: .running(pid: 4242)
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
        let allCommitsLane = summary.lane(for: .allCommits)
        let agentLane = summary.lane(for: .agentStores)
        let watcherLane = summary.lane(for: .jsonlWatcher)

        XCTAssertEqual(allCommitsLane.status, .live)
        XCTAssertEqual(allCommitsLane.values, [0, 0, 0, 460])
        XCTAssertEqual(allCommitsLane.volumeText, "460 in 1h")
        XCTAssertEqual(agentLane.status, .idle)
        XCTAssertEqual(agentLane.statusText, "No agent-origin chunks")
        XCTAssertEqual(watcherLane.status, .live)
    }

    func testFallbackReplayDebtDoesNotForceAgentLaneQueuedWhileWatcherCommitsAreLive() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let stats = DashboardStats(
            chunkCount: 120,
            enrichedChunkCount: 120,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 100,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 4_096,
            recentActivityBuckets: [0, 0, 0, 12],
            recentAgentWriteBuckets: [0, 0, 0, 0],
            recentWatcherWriteBuckets: [0, 0, 0, 12],
            recentEnrichmentBuckets: [0, 0, 0, 0],
            activityWindowMinutes: 60,
            bucketCount: 4,
            liveWindowMinutes: 1,
            lastWriteAt: now.addingTimeInterval(-15),
            pendingStoreQueueDepth: 3,
            pendingStoreFlushQueueDepth: 0,
            pendingStoreOldestQueuedAt: now.addingTimeInterval(-120),
            pendingStoreFlushRatePerMinute: 0,
            watcherHealth: WatcherHealthFileRead.readable(WatcherHealthFile(updatedAt: now, pollCount: 1, alertReasons: [], maxOffsetLagBytes: 0)),
            watcherProcessProbeResult: .running(pid: 4242)
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
        let agentLane = summary.lane(for: .agentStores)

        XCTAssertEqual(summary.lane(for: .allCommits).status, .live)
        XCTAssertEqual(summary.lane(for: .jsonlWatcher).status, .live)
        XCTAssertEqual(agentLane.status, .idle)
        XCTAssertEqual(agentLane.statusText, "No agent-origin chunks")
        XCTAssertEqual(summary.queue.storeReplayDebtDepth, 3)
        XCTAssertEqual(summary.queue.storeDepthText, "3 replay debt")
    }

    func testAgentFlushQueueDoesNotMaskCommittedAgentStores() {
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
            pendingStoreFlushQueueDepth: 5,
            pendingStoreOldestQueuedAt: now.addingTimeInterval(-120),
            pendingStoreFlushRatePerMinute: 0
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
        let agentLane = summary.lane(for: .agentStores)

        XCTAssertEqual(agentLane.status, .live)
        XCTAssertEqual(agentLane.statusText, "Agent-origin chunks landing now")
        XCTAssertEqual(agentLane.volumeText, "2 in 1h")
        XCTAssertEqual(agentLane.lastEventText, "2 agent-origin chunks in latest source-time bucket")
        XCTAssertEqual(summary.queue.storeDepthText, "5 queued")
    }

    func testPendingStoresStayOutOfAgentOriginChartWhenCommittedGraphIsZero() {
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
            pendingStoreFlushQueueDepth: 5,
            pendingStoreOldestQueuedAt: now.addingTimeInterval(-120),
            pendingStoreFlushRatePerMinute: 0
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
        let agentLane = summary.lane(for: .agentStores)

        XCTAssertEqual(agentLane.status, .idle)
        XCTAssertEqual(agentLane.statusText, "No agent-origin chunks")
        XCTAssertEqual(agentLane.volumeText, "0 in 1h")
        XCTAssertEqual(agentLane.lastEventText, "No agent-origin chunks in Last 1h")
        XCTAssertEqual(summary.queue.storeDepthText, "5 queued")
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
            pendingStoreFlushQueueDepth: 5,
            pendingStoreOldestQueuedAt: now.addingTimeInterval(-120),
            pendingStoreFlushRatePerMinute: 0
        )

        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
        let agentLane = summary.lane(for: .agentStores)

        XCTAssertEqual(agentLane.status, .live)
        XCTAssertEqual(agentLane.statusText, "Agent-origin chunks landing now")
        XCTAssertEqual(agentLane.volumeText, "2 in 1h")
        XCTAssertEqual(agentLane.lastEventText, "2 agent-origin chunks in latest source-time bucket")
        XCTAssertEqual(summary.queue.storeDepthText, "5 queued")
    }

    func testJsonlWatcherLaneDoesNotTreatHistoricalActivityMarkerAsLiveFlowTruth() {
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
            watcherHealth: WatcherHealthFileRead.readable(WatcherHealthFile(updatedAt: now, pollCount: 1, alertReasons: [], maxOffsetLagBytes: 0)),
            watcherProcessProbeResult: .running(pid: 4242)
        )

        let watcherLane = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
            .lane(for: .jsonlWatcher)

        XCTAssertEqual(watcherLane.status, .idle)
        XCTAssertEqual(watcherLane.statusText, "RUNNING · NO RECENT FLOW")
    }

    func testJsonlWatcherLaneReportsAStaleHeartbeatWithItsReason() {
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
            watcherHealth: WatcherHealthFileRead.readable(WatcherHealthFile(updatedAt: now.addingTimeInterval(-601), pollCount: 1, alertReasons: [], maxOffsetLagBytes: 0)),
            watcherProcessProbeResult: .running(pid: 4242)
        )

        let watcherLane = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
            .lane(for: .jsonlWatcher)

        // #966: watcher-health.json is the canonical liveness surface (AGENTS.md). A heartbeat ten
        // minutes old is several missed ~60-95 s polls, so it is shown as a real problem with its
        // reason, no longer ignored as a "historical marker".
        XCTAssertEqual(watcherLane.status, .idle)
        XCTAssertEqual(watcherLane.statusText, "NEEDS ATTENTION")
        XCTAssertTrue(
            watcherLane.lastEventText.hasPrefix("Watcher heartbeat stopped updating · since 10m ago"),
            watcherLane.lastEventText
        )
    }

    func testJsonlWatcherLaneReportsStoppedFromAbsentProcessWithoutHistoricalMarker() {
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
            watcherProcessProbeResult: .absent
        )

        let watcherLane = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
            .lane(for: .jsonlWatcher)

        XCTAssertEqual(watcherLane.status, .unavailable)
        XCTAssertEqual(watcherLane.statusText, "STOPPED")
    }

    func testUnreadableSourceSeriesNeverPresentZeroBucketsAsMeasuredActivity() {
        let stats = DashboardStats(
            chunkCount: 0,
            enrichedChunkCount: 0,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 0,
            enrichmentRatePerMinute: 0,
            databaseSizeBytes: 0,
            recentActivityBuckets: [0, 0, 0, 0],
            recentEnrichmentBuckets: [0, 0, 0, 0],
            activityWindowMinutes: 30,
            bucketCount: 4
        )
        let summary = DashboardFlowSummary.derive(daemon: nil, stats: stats)

        for series in [PipelineSeries.agentStores, .jsonlWatcher] {
            let lane = summary.lane(for: series)
            let presentation = BrainBarIngestSeriesPresentation(lane: lane)
            let disclosure = BrainBarDashboardChartDisclosure(
                series: series,
                lane: lane,
                timeframe: .live
            )

            XCTAssertEqual(lane.status, .unavailable)
            XCTAssertEqual(presentation.metricText, "Unavailable")
            XCTAssertFalse(presentation.showsSparkline)
            XCTAssertFalse(disclosure.accessibilitySummary.contains("Count: 0"))
            XCTAssertEqual(disclosure.tooltipDisclosure, "Evidence unavailable")
        }
    }

    func testWatcherFlowStateMatrixUsesProcessRecentDistinctFlowAndPendingEvidence() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let running = WatcherProcessProbeResult.running(pid: 4242)
        let historicalMarker = WatcherHealthFileRead.readable(WatcherHealthFile(updatedAt: now, pollCount: 1, alertReasons: [], maxOffsetLagBytes: 0))
        let alertingHistoricalMarker = WatcherHealthFileRead.readable(WatcherHealthFile(updatedAt: now.addingTimeInterval(-3_600), pollCount: 1, alertReasons: ["coverage_drop"], maxOffsetLagBytes: 10_000))

        func watcherStatusText(
            process: WatcherProcessProbeResult,
            recentDistinctChunks: [Int],
            pendingDepth: Int,
            historicalHealth: WatcherHealthFileRead?,
            recentFlowReadable: Bool = true
        ) -> String {
            let stats = DashboardStats(
                chunkCount: 120,
                enrichedChunkCount: 120,
                pendingEnrichmentCount: 0,
                enrichmentPercent: 100,
                enrichmentRatePerMinute: 0,
                databaseSizeBytes: 4_096,
                recentActivityBuckets: recentDistinctChunks,
                recentAgentWriteBuckets: Array(repeating: 0, count: recentDistinctChunks.count),
                recentWatcherWriteBuckets: recentDistinctChunks,
                recentEnrichmentBuckets: Array(repeating: 0, count: recentDistinctChunks.count),
                activityWindowMinutes: 1,
                bucketCount: recentDistinctChunks.count,
                liveWindowMinutes: 1,
                pendingStoreQueueDepth: pendingDepth,
                pendingStoreFlushQueueDepth: pendingDepth,
                watcherHealth: historicalHealth,
                watcherProcessProbeResult: process,
                watcherRecentDistinctChunkCount: recentDistinctChunks.reduce(0, +),
                watcherFlowReadability: recentFlowReadable ? .readable : .unreadable("fixture")
            )
            return DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
                .lane(for: .jsonlWatcher)
                .statusText
        }

        XCTAssertEqual(
            watcherStatusText(
                process: running,
                recentDistinctChunks: [0, 0, 0, 1],
                pendingDepth: 0,
                historicalHealth: historicalMarker
            ),
            "FLOWING"
        )
        XCTAssertEqual(
            watcherStatusText(
                process: running,
                recentDistinctChunks: [0, 0, 0, 0],
                pendingDepth: 2,
                historicalHealth: historicalMarker
            ),
            "RUNNING · NO RECENT FLOW",
            "#966: replay debt is BrainBar's deferred-store queue, not watcher work"
        )
        XCTAssertEqual(
            watcherStatusText(
                process: running,
                recentDistinctChunks: [0, 0, 0, 0],
                pendingDepth: 0,
                historicalHealth: historicalMarker
            ),
            "RUNNING · NO RECENT FLOW"
        )
        XCTAssertEqual(
            watcherStatusText(
                process: .absent,
                recentDistinctChunks: [0, 0, 0, 0],
                pendingDepth: 0,
                historicalHealth: historicalMarker
            ),
            "STOPPED"
        )
        XCTAssertEqual(
            watcherStatusText(
                process: running,
                recentDistinctChunks: [0, 0, 0, 1],
                pendingDepth: 0,
                historicalHealth: alertingHistoricalMarker
            ),
            "NEEDS ATTENTION",
            "#966: watcher-health.json is the canonical liveness surface; an hour-old heartbeat is a real problem even while chunks still land."
        )
        XCTAssertEqual(
            watcherStatusText(
                process: running,
                recentDistinctChunks: [0, 0, 0, 0],
                pendingDepth: 0,
                historicalHealth: historicalMarker,
                recentFlowReadable: false
            ),
            "RUNNING · FLOW UNVERIFIED"
        )
        XCTAssertEqual(
            watcherStatusText(
                process: .failure("launchctl timed out"),
                recentDistinctChunks: [0, 0, 0, 0],
                pendingDepth: 0,
                historicalHealth: nil,
                recentFlowReadable: false
            ),
            "UNKNOWN",
            "Ambiguous process-probe failure plus unavailable flow evidence must fail closed, not claim OFFLINE."
        )
    }

    func testJsonlWatcherLaneCallsATimestamplessHealthFileUnknown() {
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
            watcherHealth: WatcherHealthFileRead.unreadable(path: "fixture/watcher-health.json", reason: "no parseable updated_at"),
            watcherProcessProbeResult: .running(pid: 4242)
        )

        let watcherLane = DashboardFlowSummary.derive(daemon: nil, stats: stats, now: now)
            .lane(for: .jsonlWatcher)

        // A health file without updated_at cannot prove liveness: honest unknown, never running.
        XCTAssertEqual(watcherLane.status, .idle)
        XCTAssertEqual(watcherLane.statusText, "UNKNOWN")
        XCTAssertTrue(watcherLane.lastEventText.contains("no parseable updated_at"), watcherLane.lastEventText)
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

        XCTAssertEqual(summary.enrichment.sparklineLabel, "Successful enrichment completions over Last 1h")
        XCTAssertEqual(summary.enrichment.latestBucketName, "latest successful-enrichment bucket")
        let formattedCount = DashboardMetricFormatter.integerString(2_055)
        XCTAssertEqual(summary.enrichment.statusText, "Backlog drain burst: \(formattedCount) enriched in latest 15m")
        XCTAssertEqual(summary.enrichment.volumeText, "\(formattedCount) in 1h")
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
