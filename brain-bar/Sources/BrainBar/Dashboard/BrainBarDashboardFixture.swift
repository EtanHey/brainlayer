#if DEBUG
import AppKit
import Foundation

/// Deterministic, clock-independent fixture data for rendering the full BrainBar
/// dashboard to a PNG without any live collectors, database, daemon, or clock.
///
/// Why this exists: BrainBar is an `LSUIElement` menu-bar app, so `computer-use`
/// reports it "not_installed" and full-screen `screencapture` grabs the wrong
/// window — there is no reliable way to visually verify its UI. This fixture +
/// the snapshot tests in `BrainBarDashboardSnapshotTests` let ANY agent render a
/// byte-stable PNG of the real dashboard views and `Read` it to verify the UI.
///
/// Determinism rules (keep these intact):
/// - Every `Date?` is `nil` EXCEPT `fetchedAt`, which only ever renders through
///   `absoluteTimeString` (an absolute, not relative, format). Relative "Xm ago"
///   strings are the only clock-dependent text in the dashboard, and they are
///   produced solely from the `lastWriteAt` / `lastEnrichedAt` /
///   `pendingStoreOldestQueuedAt` dates — keeping those `nil` makes the render
///   independent of the wall clock.
/// - No randomness; all counts/buckets are literals.
/// - Renders pair this data with `accessibilityReduceMotion = true` so SwiftUI
///   animations resolve to their final state immediately.
@MainActor
enum BrainBarDashboardFixture {
    enum OperatorState: CaseIterable, Equatable {
        case loading
        case coverageLoading
        case loadingCards
        case live
        case stale
        case error
        case empty
        case unavailable
        case partialReplayDebt
        case watcherOffline
        case watcherUnknown
        case watcherRunningNoRecentFlow
        case watcherStalledWithPendingWork
        case queueDraining
        case queueBacklogged
    }

    /// Fixed "data fetched at" instant. Renders only via `absoluteTimeString`.
    /// 2023-11-14 22:13:20 UTC — an arbitrary but constant epoch.
    static let fetchedAt = Date(timeIntervalSince1970: 1_700_000_000)
    static let coverageDBError = "open(\"/Users/fixture/.local/share/brainlayer/fixture.db\", 14)"

    private static let readableReplayDebt = BrainDatabase.ReplayDebtBreakdown(
        pendingStores: .init(
            source: .pendingStores,
            snapshot: .init(
                depth: 320,
                oldestQueuedAt: nil,
                identityKeys: ["shared-pending-queue"]
            ),
            readability: .readable
        ),
        durableQueue: .init(
            source: .durableQueue,
            snapshot: .init(
                depth: 18,
                oldestQueuedAt: nil,
                identityKeys: ["shared-pending-queue", "shared-queue-fallback"]
            ),
            readability: .readable
        ),
        repositoryFallback: .init(
            source: .repositoryFallback,
            snapshot: .init(
                depth: 7,
                oldestQueuedAt: nil,
                identityKeys: ["shared-queue-fallback"]
            ),
            readability: .readable
        )
    )

    private static let partialReplayDebt = BrainDatabase.ReplayDebtBreakdown(
        pendingStores: readableReplayDebt.pendingStores,
        durableQueue: .init(
            source: .durableQueue,
            snapshot: readableReplayDebt.durableQueue.snapshot,
            readability: .unreadable("fixture queue directory could not be read completely")
        ),
        repositoryFallback: readableReplayDebt.repositoryFallback
    )

    private static let emptyReplayDebt = BrainDatabase.ReplayDebtBreakdown(
        pendingStores: .init(
            source: .pendingStores,
            snapshot: .init(depth: 0, oldestQueuedAt: nil, identityKeys: []),
            readability: .readable
        ),
        durableQueue: .init(
            source: .durableQueue,
            snapshot: .init(depth: 0, oldestQueuedAt: nil, identityKeys: []),
            readability: .readable
        ),
        repositoryFallback: .init(
            source: .repositoryFallback,
            snapshot: .init(depth: 0, oldestQueuedAt: nil, identityKeys: []),
            readability: .readable
        )
    )

    static let stats = makeStats(replayDebtBreakdown: readableReplayDebt)
    static let loadingStats = makeStats(replayDebtBreakdown: readableReplayDebt, coverageAvailable: false)
    static let vectorAt100Stats = makeStats(
        replayDebtBreakdown: readableReplayDebt,
        vectorIndexedChunkCount: 297_412
    )
    static let readableObservabilityResult = makeObservabilityResult(stats: stats)
    static let staleObservabilityResult = makeObservabilityResult(
        stats: stats, generatedAt: fetchedAt.addingTimeInterval(-901)
    )
    static let growingQueueStats = makeStats(
        replayDebtBreakdown: readableReplayDebt,
        recentEnrichmentBuckets: Array(repeating: 0, count: 12),
        recentEnrichmentFiveMinuteCount: 0,
        lastWriteAt: fetchedAt
    )
    static let emptyObservabilityResult = makeObservabilityResult(stats: emptyStats)

    private static func makeObservabilityResult(
        stats: DashboardStats, generatedAt: Date = fetchedAt
    ) -> ObservabilityReadResult {
        .readable(
        ObservabilityDocument(
            schemaVersion: 1,
            generatedAt: generatedAt,
            dbPath: "/fixture/brainlayer.db",
            windowHours: 24,
            stores: .init(
                state: "measured",
                reason: "",
                inputs: [],
                totalChunks: stats.chunkCount,
                inWindow: .init(
                    count: stats.recentActivityBuckets.reduce(0, +),
                    byHour: [.init(
                        hour: fetchedAt,
                        count: stats.recentActivityBuckets.reduce(0, +)
                    )]
                )
            ),
            emitters: .init(
                state: "measured",
                reason: "",
                inputs: [],
                byEmitter: [.init(emitter: "mcp", countInWindow: stats.recentAgentWriteCount)],
                bySourceClass: [.init(sourceClass: "claude_code", count: stats.chunkCount == 0 ? 0 : 180_000, inWindow: stats.recentAgentWriteCount)],
                hiddenFromDefaultSearch: 0
            ),
            authorUnknown: .init(
                state: "measured",
                reason: "",
                inputs: [],
                neverClassified: .init(count: 0, share: 0),
                classifiedUnknown: .init(count: 0, share: 0)
            ),
            backups: .init(
                state: "measured",
                reason: "",
                inputs: [],
                freshness: "fresh",
                thresholdHours: 36,
                retentionInvariant: "PASS",
                survivingArchives30D: 3,
                errorType: nil,
                lastVerifiedUpload: .init(
                    at: fetchedAt.addingTimeInterval(-3_600),
                    ageHours: 1,
                    archiveId: "transcripts-verified",
                    verified: true
                ),
                dbSnapshot: .init(
                    lastAt: fetchedAt.addingTimeInterval(-7_200),
                    destination: "brainlayer-verified.db.gz",
                    verified: true
                ),
                launchd: .init(
                    label: "com.brainlayer.jsonl-backup",
                    bootstrapped: true,
                    disabledDirPresent: false
                )
            )
        )
        )
    }
    static let partialReplayDebtStats = makeStats(replayDebtBreakdown: partialReplayDebt)
    static let watcherOfflineStats = makeStats(
        replayDebtBreakdown: readableReplayDebt,
        watcherProcessProbeResult: .absent,
        watcherRecentDistinctChunkCount: 0
    )
    static let watcherUnknownStats = makeStats(
        replayDebtBreakdown: readableReplayDebt,
        watcherProcessProbeResult: .failure("fixture watcher process probe failed"),
        watcherRecentDistinctChunkCount: 0
    )
    static let watcherRunningNoRecentFlowStats = makeStats(
        replayDebtBreakdown: emptyReplayDebt,
        watcherProcessProbeResult: .running(pid: 4242),
        watcherRecentDistinctChunkCount: 0
    )
    static let watcherStalledWithPendingWorkStats = makeStats(
        replayDebtBreakdown: readableReplayDebt,
        watcherProcessProbeResult: .running(pid: 4242),
        watcherRecentDistinctChunkCount: 0
    )
    static let queueDrainingStats = makeStats(
        replayDebtBreakdown: readableReplayDebt,
        watcherProcessProbeResult: .running(pid: 4242),
        watcherRecentDistinctChunkCount: 0,
        recentEnrichmentBuckets: [4, 6, 3, 7, 5, 8, 6, 9, 7, 5, 8, 6]
    )
    static let queueBackloggedStats = makeStats(
        replayDebtBreakdown: readableReplayDebt,
        watcherProcessProbeResult: .running(pid: 4242),
        watcherRecentDistinctChunkCount: 0,
        recentEnrichmentBuckets: Array(repeating: 0, count: 12),
        recentEnrichmentFiveMinuteCount: 0
    )
    static let emptyStats = makeStats(
        replayDebtBreakdown: emptyReplayDebt,
        watcherRecentDistinctChunkCount: 0,
        zeroFlow: true
    )
    static let unavailableStats = makeStats(
        replayDebtBreakdown: readableReplayDebt,
        coverageAvailable: false,
        watcherProcessProbeResult: .failure("fixture watcher unavailable"),
        agentWriteReadability: .unreadable("fixture agent flow unavailable"),
        watcherFlowReadability: .unreadable("fixture watcher flow unavailable")
    )

    static func makeStats(activityWindowMinutes: Int) -> DashboardStats {
        makeStats(
            replayDebtBreakdown: readableReplayDebt,
            activityWindowMinutes: activityWindowMinutes
        )
    }

    private static func makeStats(
        replayDebtBreakdown: BrainDatabase.ReplayDebtBreakdown,
        coverageAvailable: Bool = true,
        activityWindowMinutes: Int = 60,
        watcherProcessProbeResult: WatcherProcessProbeResult = .running(pid: 4242),
        watcherRecentDistinctChunkCount: Int = 14,
        vectorIndexedChunkCount: Int = 240_100,
        recentEnrichmentBuckets: [Int] = [4, 6, 3, 7, 5, 8, 6, 9, 7, 5, 8, 6],
        recentEnrichmentFiveMinuteCount: Int = 22,
        lastWriteAt: Date? = nil,
        zeroFlow: Bool = false,
        agentWriteReadability: MetricEvidenceReadability = .readable,
        watcherFlowReadability: MetricEvidenceReadability = .readable,
    ) -> DashboardStats {
        let windowScale = max(activityWindowMinutes / 60, 1)
        let watcherBuckets = watcherRecentDistinctChunkCount == 0
            ? Array(repeating: 0, count: 12)
            : [1, 0, 2, 1, 0, 3, 1, 2, 0, 1, 2, 1].map { $0 * windowScale }
        return DashboardStats(
            chunkCount: zeroFlow ? 0 : 297_412,
            enrichedChunkCount: zeroFlow ? 0 : 188_204,
            failedEnrichmentCount: zeroFlow ? 0 : 1_204,
            skippedEnrichmentCount: zeroFlow ? 0 : 2_104,
            pendingEnrichmentCount: zeroFlow ? 0 : 12_840,
            enrichmentPercent: zeroFlow ? 0 : 63.3,
            enrichmentRatePerMinute: zeroFlow ? 0 : 11.4,
            databaseSizeBytes: zeroFlow ? 0 : 8_120_000_000,
            recentActivityBuckets: (zeroFlow ? Array(repeating: 0, count: 12) : [3, 5, 2, 8, 6, 4, 9, 7, 5, 6, 8, 4]).map { $0 * windowScale },
            recentAgentWriteBuckets: (zeroFlow ? Array(repeating: 0, count: 12) : [1, 2, 0, 3, 2, 1, 4, 3, 1, 2, 3, 1]).map { $0 * windowScale },
            agentWriteReadability: agentWriteReadability,
            recentWatcherWriteBuckets: watcherBuckets,
            recentEnrichmentBuckets: (zeroFlow ? Array(repeating: 0, count: 12) : recentEnrichmentBuckets).map { $0 * windowScale },
            recentWriteFiveMinuteCount: zeroFlow ? 0 : 18,
            recentEnrichmentFiveMinuteCount: zeroFlow ? 0 : recentEnrichmentFiveMinuteCount,
            activityWindowMinutes: activityWindowMinutes,
            bucketCount: 12,
            liveWindowMinutes: 1,
            lastWriteAt: lastWriteAt,
            lastEnrichedAt: nil,
            signalEligibleChunkCount: coverageAvailable && !zeroFlow ? 297_412 : 0,
            vectorIndexedChunkCount: coverageAvailable && !zeroFlow ? vectorIndexedChunkCount : 0,
            ftsIndexedChunkCount: coverageAvailable && !zeroFlow ? 296_980 : 0,
            trigramIndexedChunkCount: coverageAvailable && !zeroFlow ? 210_540 : 0,
            signalCoverageIsAvailable: coverageAvailable,
            pendingStoreQueueDepth: replayDebtBreakdown.deduplicatedTotal,
            pendingStoreFlushQueueDepth: replayDebtBreakdown.pendingStores.snapshot.depth,
            pendingStoreOldestQueuedAt: nil,
            pendingStoreFlushRatePerMinute: 45,
            watcherHealth: DashboardStats.WatcherHealth(
                alerting: false,
                filesTracked: 14,
                maxOffsetLagBytes: 2_048,
                activeEntriesPerMinute: 12.5,
                realtimeInsertsPerMinute: 9.0,
                updatedAt: fetchedAt
            ),
            replayDebtBreakdown: replayDebtBreakdown,
            watcherProcessProbeResult: watcherProcessProbeResult,
            watcherRecentDistinctChunkCount: watcherRecentDistinctChunkCount,
            watcherFlowReadability: watcherFlowReadability
        )
    }

    static let daemon = DaemonHealthSnapshot(
        pid: 4242,
        isResponsive: true,
        rssBytes: 268_435_456,
        uptime: 18_000,
        openConnections: 3,
        lastSeenAt: fetchedAt
    )

    static let agentActivity = AgentActivitySnapshot(
        presences: [
            AgentPresence(family: .claude, count: 2),
            AgentPresence(family: .codex, count: 1),
            AgentPresence(family: .cursor, count: 0),
            AgentPresence(family: .gemini, count: 1),
        ]
    )

    static func makeReceiptStore(_ operatorState: OperatorState) -> BrainBarOperationReceipts {
        let store = BrainBarOperationReceipts()
        if operatorState == .live || operatorState == .error {
            let failed = operatorState == .error
            store.record(.init(kind: .search, durationMillis: 142, count: failed ? nil : 10, failed: failed, recordedAt: fetchedAt))
            store.record(.init(kind: .ingest, durationMillis: 1_200, count: failed ? nil : 1, failed: failed, recordedAt: fetchedAt))
        }
        return store
    }

    static var state: PipelineState {
        PipelineState.derive(daemon: daemon, stats: stats)
    }

    /// A `StatsCollector` pre-loaded with the fixture state and no live wiring
    /// (no DB, no observers, no timers — `start()` is never called).
    static func makeCollector(
        _ operatorState: OperatorState = .live,
        agentActivity: AgentActivitySnapshot = BrainBarDashboardFixture.agentActivity
    ) -> StatsCollector {
        let fixtureStats: DashboardStats
        switch operatorState {
        case .loading, .coverageLoading:
            fixtureStats = loadingStats
        case .partialReplayDebt:
            fixtureStats = partialReplayDebtStats
        case .watcherOffline:
            fixtureStats = watcherOfflineStats
        case .watcherUnknown:
            fixtureStats = watcherUnknownStats
        case .watcherRunningNoRecentFlow:
            fixtureStats = watcherRunningNoRecentFlowStats
        case .watcherStalledWithPendingWork:
            fixtureStats = watcherStalledWithPendingWorkStats
        case .queueDraining:
            fixtureStats = queueDrainingStats
        case .queueBacklogged:
            fixtureStats = queueBackloggedStats
        case .empty:
            fixtureStats = emptyStats
        case .loadingCards, .unavailable:
            fixtureStats = unavailableStats
        case .live, .stale, .error:
            fixtureStats = stats
        }
        let freshness: SnapshotFreshnessState
        let lastDataFetchedAt: Date?
        let lastFetchError: String?
        switch operatorState {
        case .loading:
            freshness = .loading
            lastDataFetchedAt = nil
            lastFetchError = nil
        case .live,
             .coverageLoading,
             .watcherOffline,
             .watcherUnknown,
             .watcherRunningNoRecentFlow,
             .watcherStalledWithPendingWork,
             .queueDraining,
             .queueBacklogged,
             .empty,
             .loadingCards:
            freshness = .live(ageSeconds: 0)
            lastDataFetchedAt = fetchedAt
            lastFetchError = nil
        case .stale:
            freshness = .stale(ageSeconds: 61)
            lastDataFetchedAt = fetchedAt
            lastFetchError = nil
        case .error:
            freshness = .error(message: "Fixture fetch failed", lastSuccessAgeSeconds: 15)
            lastDataFetchedAt = fetchedAt
            lastFetchError = "Fixture fetch failed"
        case .unavailable:
            freshness = .error(message: coverageDBError, lastSuccessAgeSeconds: nil)
            lastDataFetchedAt = nil
            lastFetchError = coverageDBError
        case .partialReplayDebt:
            freshness = .live(ageSeconds: 0)
            lastDataFetchedAt = fetchedAt
            lastFetchError = nil
        }

        return StatsCollector.fixture(
            stats: fixtureStats,
            daemon: daemon,
            agentActivity: agentActivity,
            state: PipelineState.derive(daemon: daemon, stats: fixtureStats),
            heartbeat: .empty,
            lastDataFetchedAt: lastDataFetchedAt,
            lastFetchError: lastFetchError,
            snapshotFreshnessState: freshness
        )
    }

    static func makeCollector(stats: DashboardStats) -> StatsCollector {
        StatsCollector.fixture(
            stats: stats,
            daemon: daemon,
            agentActivity: agentActivity,
            state: PipelineState.derive(daemon: daemon, stats: stats),
            heartbeat: .empty,
            lastDataFetchedAt: fetchedAt,
            snapshotFreshnessState: .live(ageSeconds: 0)
        )
    }
}
#endif
