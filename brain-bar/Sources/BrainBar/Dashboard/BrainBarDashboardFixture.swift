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
    /// Fixed "data fetched at" instant. Renders only via `absoluteTimeString`.
    /// 2023-11-14 22:13:20 UTC — an arbitrary but constant epoch.
    static let fetchedAt = Date(timeIntervalSince1970: 1_700_000_000)

    static let stats = DashboardStats(
        chunkCount: 297_412,
        enrichedChunkCount: 188_204,
        pendingEnrichmentCount: 12_840,
        enrichmentPercent: 63.3,
        enrichmentRatePerMinute: 11.4,
        databaseSizeBytes: 8_120_000_000,
        recentActivityBuckets: [3, 5, 2, 8, 6, 4, 9, 7, 5, 6, 8, 4],
        recentAgentWriteBuckets: [3, 5, 2, 8, 6, 4, 9, 7, 5, 6, 8, 4],
        recentWatcherWriteBuckets: [1, 0, 2, 1, 0, 3, 1, 2, 0, 1, 2, 1],
        recentEnrichmentBuckets: [4, 6, 3, 7, 5, 8, 6, 9, 7, 5, 8, 6],
        recentWriteFiveMinuteCount: 18,
        recentEnrichmentFiveMinuteCount: 22,
        activityWindowMinutes: 60,
        bucketCount: 12,
        liveWindowMinutes: 1,
        lastWriteAt: nil,
        lastEnrichedAt: nil,
        signalEligibleChunkCount: 297_412,
        vectorIndexedChunkCount: 240_100,
        ftsIndexedChunkCount: 296_980,
        trigramIndexedChunkCount: 210_540,
        pendingStoreQueueDepth: 320,
        pendingStoreOldestQueuedAt: nil,
        pendingStoreFlushRatePerMinute: 45,
        watcherHealth: DashboardStats.WatcherHealth(
            alerting: false,
            filesTracked: 14,
            maxOffsetLagBytes: 2_048,
            activeEntriesPerMinute: 12.5,
            realtimeInsertsPerMinute: 9.0,
            updatedAt: fetchedAt
        )
    )

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
    static func makeCollector() -> StatsCollector {
        StatsCollector.fixture(
            stats: stats,
            daemon: daemon,
            agentActivity: agentActivity,
            state: state,
            heartbeat: .empty,
            lastDataFetchedAt: fetchedAt
        )
    }
}
#endif
