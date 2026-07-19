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
        case live
        case stale
        case error
        case partialReplayDebt
        case watcherOffline
        case watcherUnknown
        case watcherRunningNoRecentFlow
        case watcherStalledWithPendingWork
    }

    /// Fixed "data fetched at" instant. Renders only via `absoluteTimeString`.
    /// 2023-11-14 22:13:20 UTC — an arbitrary but constant epoch.
    static let fetchedAt = Date(timeIntervalSince1970: 1_700_000_000)

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

    static func makeStats(activityWindowMinutes: Int) -> DashboardStats {
        makeStats(
            replayDebtBreakdown: readableReplayDebt,
            activityWindowMinutes: activityWindowMinutes
        )
    }

    private static func makeStats(
        replayDebtBreakdown: BrainDatabase.ReplayDebtBreakdown,
        activityWindowMinutes: Int = 60,
        watcherProcessProbeResult: WatcherProcessProbeResult = .running(pid: 4242),
        watcherRecentDistinctChunkCount: Int = 14
    ) -> DashboardStats {
        let windowScale = max(activityWindowMinutes / 60, 1)
        return DashboardStats(
            chunkCount: 297_412,
            enrichedChunkCount: 188_204,
            failedEnrichmentCount: 1_204,
            skippedEnrichmentCount: 2_104,
            pendingEnrichmentCount: 12_840,
            enrichmentPercent: 63.3,
            enrichmentRatePerMinute: 11.4,
            databaseSizeBytes: 8_120_000_000,
            recentActivityBuckets: [3, 5, 2, 8, 6, 4, 9, 7, 5, 6, 8, 4].map { $0 * windowScale },
            recentAgentWriteBuckets: [3, 5, 2, 8, 6, 4, 9, 7, 5, 6, 8, 4].map { $0 * windowScale },
            recentWatcherWriteBuckets: [1, 0, 2, 1, 0, 3, 1, 2, 0, 1, 2, 1].map { $0 * windowScale },
            recentEnrichmentBuckets: [4, 6, 3, 7, 5, 8, 6, 9, 7, 5, 8, 6].map { $0 * windowScale },
            recentWriteFiveMinuteCount: 18,
            recentEnrichmentFiveMinuteCount: 22,
            activityWindowMinutes: activityWindowMinutes,
            bucketCount: 12,
            liveWindowMinutes: 1,
            lastWriteAt: nil,
            lastEnrichedAt: nil,
            signalEligibleChunkCount: 297_412,
            vectorIndexedChunkCount: 240_100,
            ftsIndexedChunkCount: 296_980,
            trigramIndexedChunkCount: 210_540,
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
            watcherFlowReadability: .readable
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

    static var state: PipelineState {
        PipelineState.derive(daemon: daemon, stats: stats)
    }

    /// A `StatsCollector` pre-loaded with the fixture state and no live wiring
    /// (no DB, no observers, no timers — `start()` is never called).
    static func makeCollector(_ operatorState: OperatorState = .live) -> StatsCollector {
        let fixtureStats: DashboardStats
        switch operatorState {
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
        case .loading, .live, .stale, .error:
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
             .watcherOffline,
             .watcherUnknown,
             .watcherRunningNoRecentFlow,
             .watcherStalledWithPendingWork:
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
}
#endif
