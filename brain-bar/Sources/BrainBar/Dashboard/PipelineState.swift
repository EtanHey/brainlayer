import AppKit
import Foundation

typealias DashboardStats = BrainDatabase.DashboardStats

enum MetricClock: String, Sendable, Equatable {
    case sourceTime
    case ingestTime
}

enum MetricCardinality: String, Sendable, Equatable {
    case chunkRows
    case distinctChunkIDs
    case events
}

struct MetricContract: Sendable, Equatable {
    let clock: MetricClock
    let cardinality: MetricCardinality
}

enum DashboardMetricContract {
    static let chunkRows = MetricContract(clock: .sourceTime, cardinality: .chunkRows)
    static let agentOriginChunks = MetricContract(clock: .sourceTime, cardinality: .chunkRows)
    static let watcherIngestedChunks = MetricContract(clock: .ingestTime, cardinality: .distinctChunkIDs)
}

enum MetricEvidenceReadability: Sendable, Equatable {
    case readable
    case unreadable(String)

    var isReadable: Bool {
        if case .readable = self { return true }
        return false
    }
}

enum WatcherProcessProbeResult: Sendable, Equatable {
    case running(pid: pid_t)
    case absent
    case failure(String)
}

enum WatcherFlowState: Sendable, Equatable {
    case flowing
    case stalled
    case runningNoRecentFlow
    case offline
    case runningFlowUnverified
    case unknown

    static func derive(
        process: WatcherProcessProbeResult,
        recentDistinctChunkCount: Int,
        recentFlowReadable: Bool,
        pendingWorkCount: Int
    ) -> WatcherFlowState {
        switch process {
        case .failure:
            return .unknown
        case .absent:
            return .offline
        case .running:
            guard recentFlowReadable else { return .runningFlowUnverified }
            if recentDistinctChunkCount > 0 { return .flowing }
            if pendingWorkCount > 0 { return .stalled }
            return .runningNoRecentFlow
        }
    }

    var label: String {
        switch self {
        case .flowing: return "FLOWING"
        case .stalled: return "STALLED"
        case .runningNoRecentFlow: return "RUNNING · NO RECENT FLOW"
        case .offline: return "OFFLINE"
        case .runningFlowUnverified: return "RUNNING · FLOW UNVERIFIED"
        case .unknown: return "UNKNOWN"
        }
    }
}

enum SnapshotFreshnessState: Sendable, Equatable {
    case loading
    case live(ageSeconds: Int)
    case stale(ageSeconds: Int)
    case error(message: String, lastSuccessAgeSeconds: Int?)

    static func derive(
        lastSuccessAt: Date?,
        isRefreshing: Bool,
        lastFetchError: String?,
        now: Date,
        snapshotFreshnessThreshold: TimeInterval
    ) -> SnapshotFreshnessState {
        let ageSeconds = lastSuccessAt.map { max(0, Int(now.timeIntervalSince($0))) }
        if let lastFetchError {
            return .error(message: lastFetchError, lastSuccessAgeSeconds: ageSeconds)
        }
        if lastSuccessAt == nil && isRefreshing {
            return .loading
        }
        guard let lastSuccessAt else {
            return .loading
        }
        let age = max(0, now.timeIntervalSince(lastSuccessAt))
        if age > snapshotFreshnessThreshold {
            return .stale(ageSeconds: Int(age))
        }
        return .live(ageSeconds: Int(age))
    }

    var isLoading: Bool {
        if case .loading = self { return true }
        return false
    }

    var label: String {
        switch self {
        case .loading: return "LOADING"
        case .live: return "LIVE"
        case .stale: return "STALE"
        case .error: return "ERROR"
        }
    }

    var ageSeconds: Int? {
        switch self {
        case .loading:
            return nil
        case .live(let ageSeconds), .stale(let ageSeconds):
            return ageSeconds
        case .error(_, let lastSuccessAgeSeconds):
            return lastSuccessAgeSeconds
        }
    }
}

struct DaemonHealthSnapshot: Sendable, Equatable {
    let pid: pid_t
    let isResponsive: Bool
    let rssBytes: UInt64
    let uptime: TimeInterval
    let openConnections: Int
    let lastSeenAt: Date
}

enum PipelineIndicatorStatus: Sendable, Equatable {
    case live
    case queued
    case idle
    case unavailable

    var label: String {
        switch self {
        case .live:
            return "live"
        case .queued:
            return "queued"
        case .idle:
            return "idle"
        case .unavailable:
            return "offline"
        }
    }

    var color: NSColor {
        stateTheme.theme.color
    }

    var stateTheme: BrainBarStateTheme {
        switch self {
        case .live:
            return .active
        case .queued:
            return .loading
        case .idle:
            return .idle
        case .unavailable:
            return .error
        }
    }
}

struct PipelineIndicator: Sendable, Equatable {
    let name: String
    let status: PipelineIndicatorStatus
}

struct PipelineIndicators: Sendable, Equatable {
    let indexing: PipelineIndicator
    let enriching: PipelineIndicator

    static func derive(daemon: DaemonHealthSnapshot?, stats: DashboardStats, now: Date = Date()) -> PipelineIndicators {
        let summary = DashboardFlowSummary.derive(daemon: daemon, stats: stats, now: now)

        let indexingStatus: PipelineIndicatorStatus
        let enrichingStatus: PipelineIndicatorStatus

        if summary.isUnavailable {
            indexingStatus = .unavailable
            enrichingStatus = .unavailable
        } else {
            indexingStatus = switch summary.ingress.status {
            case .live:
                .live
            case .recent where summary.queue.status == .growing:
                .queued
            default:
                .idle
            }

            enrichingStatus = switch summary.enrichment.status {
            case .live:
                .live
            case .recent where summary.queue.status == .draining:
                .live
            case .queued:
                .queued
            default:
                stats.pendingEnrichmentCount > 0 ? .queued : .idle
            }
        }

        return PipelineIndicators(
            indexing: PipelineIndicator(name: "Indexing", status: indexingStatus),
            enriching: PipelineIndicator(name: "Enriching", status: enrichingStatus)
        )
    }
}

enum DashboardFlowLaneStatus: String, Sendable, Equatable {
    case live
    case recent
    case draining
    case queued
    case idle
    case unavailable

    var label: String {
        switch self {
        case .live:
            return "live"
        case .recent:
            return "recent"
        case .draining:
            return "draining"
        case .queued:
            return "queued"
        case .idle:
            return "idle"
        case .unavailable:
            return "offline"
        }
    }
}

enum DashboardQueueStatus: String, Sendable, Equatable {
    case empty
    case stable
    case growing
    case draining
    case backlogged
    case unavailable

    var label: String {
        switch self {
        case .empty:
            return "empty"
        case .stable:
            return "stable"
        case .growing:
            return "growing"
        case .draining:
            return "draining"
        case .backlogged:
            return "backlogged"
        case .unavailable:
            return "offline"
        }
    }
}

enum DashboardStoreQueueHealth: String, Sendable, Equatable {
    case empty
    case activeDraining
    case backlogAccumulating
    case writerStuck

    var label: String {
        switch self {
        case .empty:
            return "empty"
        case .activeDraining:
            return "active draining"
        case .backlogAccumulating:
            return "backlog accumulating"
        case .writerStuck:
            return "writer stuck - investigate"
        }
    }

    var color: NSColor {
        switch self {
        case .empty:
            return BrainBarStateTheme.empty.theme.color
        case .activeDraining:
            return BrainBarStateTheme.active.theme.color
        case .backlogAccumulating:
            return BrainBarStateTheme.degraded.theme.color
        case .writerStuck:
            return BrainBarStateTheme.error.theme.color
        }
    }
}

struct DashboardFlowLane: Sendable, Equatable {
    let name: String
    let status: DashboardFlowLaneStatus
    let statusText: String
    let windowLabel: String
    let activityWindowMinutes: Int
    let rateText: String
    let volumeText: String
    let lastEventText: String
    let values: [Int]
    let sparklineLabel: String
    let latestBucketName: String
    let accentColor: NSColor
    let primarySeriesLabel: String?
    let secondaryValues: [Int]
    let secondarySeriesLabel: String?
    let secondaryAccentColor: NSColor?
    let tertiaryValues: [Int]
    let tertiarySeriesLabel: String?
    let tertiaryAccentColor: NSColor?
}

struct DashboardQueueSummary: Sendable, Equatable {
    let status: DashboardQueueStatus
    let backlogCount: Int
    let storeHealth: DashboardStoreQueueHealth
    let storeHealthText: String
    let storeDepth: Int
    let storeFlushDepth: Int
    let storeReplayDebtDepth: Int
    let storeOldestAgeSeconds: Int?
    let storeFlushRatePerMinute: Double
    let storeDepthText: String
    let storeOldestAgeText: String
    let storeFlushRateText: String
    let title: String
    let detail: String
}

struct DashboardFlowSummary: Sendable, Equatable {
    let headline: String
    let detail: String
    let windowLabel: String
    let allCommits: DashboardFlowLane
    let ingress: DashboardFlowLane
    let queue: DashboardQueueSummary
    let enrichment: DashboardFlowLane
    let watcherFlowState: WatcherFlowState
    let watcherHealth: DashboardStats.WatcherHealth?
    let watcherHealthIsFresh: Bool

    var isUnavailable: Bool {
        ingress.status == .unavailable || enrichment.status == .unavailable || queue.status == .unavailable
    }

    static func derive(daemon: DaemonHealthSnapshot?, stats: DashboardStats, now: Date = Date()) -> DashboardFlowSummary {
        let windowLabel = DashboardMetricFormatter.windowLabel(minutes: stats.activityWindowMinutes)
        let allCommitsColor = BrainBarDesignTokens.Colors.accentBright
        let agentStoresColor = BrainBarDesignTokens.Colors.seriesAgent
        let jsonlWatcherColor = BrainBarDesignTokens.Colors.seriesWatcher
        let enrichmentColor = BrainBarStateTheme.active.theme.color

        let writesLive = stats.eventIsLive(stats.lastWriteAt, now: now)
        let enrichmentsLive = stats.eventIsLive(stats.lastEnrichedAt, now: now)
        let backlogCount = stats.pendingEnrichmentCount
        let storeOldestAgeSeconds = stats.pendingStoreOldestQueuedAt.map { max(0, Int(now.timeIntervalSince($0).rounded())) }
        let storeHealth = storeQueueHealth(depth: stats.pendingStoreQueueDepth, oldestAgeSeconds: storeOldestAgeSeconds)
        let watcherProcess = stats.watcherProcessProbeResult
            ?? daemon.map { WatcherProcessProbeResult.running(pid: $0.pid) }
            ?? .absent
        let watcherFlowState = WatcherFlowState.derive(
            process: watcherProcess,
            recentDistinctChunkCount: stats.watcherRecentDistinctChunkCount,
            recentFlowReadable: stats.watcherFlowReadability.isReadable,
            pendingWorkCount: stats.replayDebtBreakdown.deduplicatedTotal
        )

        let ingressStatus: DashboardFlowLaneStatus
        if writesLive {
            ingressStatus = .live
        } else if stats.recentWriteCount > 0 {
            ingressStatus = .recent
        } else {
            ingressStatus = .idle
        }

        let enrichmentStatus: DashboardFlowLaneStatus
        if enrichmentsLive {
            enrichmentStatus = .live
        } else if backlogCount > 0 {
            enrichmentStatus = stats.recentEnrichmentCount > 0 ? .recent : .queued
        } else if stats.recentEnrichmentCount > 0 {
            enrichmentStatus = .recent
        } else {
            enrichmentStatus = .idle
        }

        let queueStatus: DashboardQueueStatus
        if backlogCount == 0 {
            queueStatus = (stats.recentWriteCount > 0 || stats.recentEnrichmentCount > 0) ? .stable : .empty
        } else if writesLive && !enrichmentsLive {
            queueStatus = .growing
        } else if enrichmentsLive && !writesLive {
            queueStatus = .draining
        } else if writesLive && enrichmentsLive {
            queueStatus = .stable
        } else if stats.recentEnrichmentCount > 0 {
            queueStatus = .draining
        } else {
            queueStatus = .backlogged
        }

        let headline: String
        let detail: String

        if ingressStatus == .live && queueStatus == .stable && enrichmentStatus == .live {
            headline = "Writes are landing and enrichments are shipping"
            detail = "\(stats.recentWriteCount) writes and \(stats.recentEnrichmentCount) enrichments in \(windowLabel.lowercased())."
        } else if ingressStatus == .live && queueStatus == .growing {
            headline = "Writes are outrunning enrichments"
            detail = "\(backlogCount) chunks are waiting while ingress is still active."
        } else if backlogCount > 0 &&
            (queueStatus == .draining || enrichmentStatus == .draining || enrichmentStatus == .live) {
            headline = "Enrichment is draining backlog"
            detail = "\(backlogCount) chunks remain queued, and completions are still moving."
        } else if queueStatus == .backlogged || enrichmentStatus == .queued {
            headline = "Backlog is waiting for enrichment"
            detail = "\(backlogCount) chunks are queued with no enrichment in the live window."
        } else if ingressStatus == .recent || enrichmentStatus == .recent {
            headline = "The flow is cooling down"
            detail = "Live activity is quiet, but recent movement is still visible in \(windowLabel.lowercased())."
        } else {
            headline = "The flow is idle"
            detail = "No writes or enrichments landed in \(windowLabel.lowercased())."
        }

        let enrichmentStatusText = enrichmentStatusText(
            status: enrichmentStatus,
            stats: stats,
            windowLabel: windowLabel
        )

        return DashboardFlowSummary(
            headline: headline,
            detail: detail,
            windowLabel: windowLabel,
            allCommits: DashboardFlowLane(
                name: "All commits",
                status: ingressStatus,
                statusText: allCommitsStatusText(
                    status: ingressStatus,
                    totalEvents: stats.recentWriteCount,
                    windowLabel: windowLabel
                ),
                windowLabel: windowLabel,
                activityWindowMinutes: stats.activityWindowMinutes,
                rateText: DashboardMetricFormatter.rateString(
                    totalEvents: stats.recentWriteCount,
                    activityWindowMinutes: stats.activityWindowMinutes
                ),
                volumeText: DashboardMetricFormatter.activitySummaryString(
                    totalEvents: stats.recentWriteCount,
                    activityWindowMinutes: stats.activityWindowMinutes
                ),
                lastEventText: allCommitsLastEventText(
                    status: ingressStatus,
                    totalEvents: stats.recentWriteCount,
                    latestBucketCount: stats.recentActivityBuckets.last ?? 0,
                    windowLabel: windowLabel
                ),
                values: stats.recentActivityBuckets,
                sparklineLabel: "All committed chunks over \(windowLabel)",
                latestBucketName: "latest commit bucket",
                accentColor: allCommitsColor,
                primarySeriesLabel: nil,
                secondaryValues: [],
                secondarySeriesLabel: nil,
                secondaryAccentColor: nil,
                tertiaryValues: [],
                tertiarySeriesLabel: nil,
                tertiaryAccentColor: nil
            ),
            ingress: DashboardFlowLane(
                name: "Writes",
                status: ingressStatus,
                statusText: ingressStatus == .live ? "Ingress live now" : (ingressStatus == .recent ? "Recent writes in \(windowLabel.lowercased())" : "No recent writes"),
                windowLabel: windowLabel,
                activityWindowMinutes: stats.activityWindowMinutes,
                rateText: DashboardMetricFormatter.rateString(
                    totalEvents: stats.recentWriteCount,
                    activityWindowMinutes: stats.activityWindowMinutes
                ),
                volumeText: DashboardMetricFormatter.activitySummaryString(
                    totalEvents: stats.recentWriteCount,
                    activityWindowMinutes: stats.activityWindowMinutes
                ),
                lastEventText: DashboardMetricFormatter.lastEventString(
                    lastEventAt: stats.lastWriteAt,
                    activityWindowMinutes: stats.activityWindowMinutes,
                    now: now
                ),
                values: stats.recentAgentWriteBuckets,
                sparklineLabel: "Writes over \(windowLabel)",
                latestBucketName: "latest write bucket",
                accentColor: agentStoresColor,
                primarySeriesLabel: "Agent MCP stores",
                secondaryValues: stats.recentWatcherWriteBuckets,
                secondarySeriesLabel: "JSONL watcher",
                secondaryAccentColor: jsonlWatcherColor,
                tertiaryValues: [],
                tertiarySeriesLabel: nil,
                tertiaryAccentColor: nil
            ),
            queue: DashboardQueueSummary(
                status: queueStatus,
                backlogCount: backlogCount,
                storeHealth: storeHealth,
                storeHealthText: storeHealth.label,
                storeDepth: stats.pendingStoreQueueDepth,
                storeFlushDepth: stats.pendingStoreFlushQueueDepth,
                storeReplayDebtDepth: stats.pendingStoreReplayDebtDepth,
                storeOldestAgeSeconds: storeOldestAgeSeconds,
                storeFlushRatePerMinute: stats.pendingStoreFlushRatePerMinute,
                storeDepthText: storeDepthText(
                    totalDepth: stats.pendingStoreQueueDepth,
                    flushDepth: stats.pendingStoreFlushQueueDepth,
                    replayDebtDepth: stats.pendingStoreReplayDebtDepth
                ),
                storeOldestAgeText: storeOldestAgeText(storeOldestAgeSeconds),
                storeFlushRateText: DashboardMetricFormatter.speedString(
                    ratePerMinute: stats.pendingStoreFlushRatePerMinute
                ),
                title: queueTitle(
                    status: queueStatus,
                    backlogCount: backlogCount,
                    storeHealth: storeHealth
                ),
                detail: queueDetail(
                    status: queueStatus,
                    backlogCount: backlogCount,
                    storeHealth: storeHealth,
                    storeDepth: stats.pendingStoreQueueDepth,
                    storeFlushDepth: stats.pendingStoreFlushQueueDepth,
                    storeReplayDebtDepth: stats.pendingStoreReplayDebtDepth,
                    storeOldestAgeSeconds: storeOldestAgeSeconds,
                    storeFlushRatePerMinute: stats.pendingStoreFlushRatePerMinute,
                    stats: stats,
                    windowLabel: windowLabel
                )
            ),
            enrichment: DashboardFlowLane(
                name: "Enrichments",
                status: enrichmentStatus,
                statusText: enrichmentStatusText,
                windowLabel: windowLabel,
                activityWindowMinutes: stats.activityWindowMinutes,
                rateText: DashboardMetricFormatter.rateString(
                    totalEvents: stats.recentEnrichmentCount,
                    activityWindowMinutes: stats.activityWindowMinutes
                ),
                volumeText: DashboardMetricFormatter.activitySummaryString(
                    totalEvents: stats.recentEnrichmentCount,
                    activityWindowMinutes: stats.activityWindowMinutes
                ),
                lastEventText: DashboardMetricFormatter.lastEventString(
                    lastEventAt: stats.lastEnrichedAt,
                    activityWindowMinutes: stats.activityWindowMinutes,
                    now: now
                ),
                values: stats.recentEnrichmentBuckets,
                sparklineLabel: "Enrichment completions over \(windowLabel)",
                latestBucketName: "latest enrichment bucket",
                accentColor: enrichmentColor,
                primarySeriesLabel: "Enrichments",
                secondaryValues: [],
                secondarySeriesLabel: nil,
                secondaryAccentColor: nil,
                tertiaryValues: [],
                tertiarySeriesLabel: nil,
                tertiaryAccentColor: nil
            ),
            watcherFlowState: watcherFlowState,
            watcherHealth: stats.watcherHealth,
            watcherHealthIsFresh: stats.watcherHealth?.isFresh(now: now) ?? false
        )
    }

    private static func enrichmentStatusText(
        status: DashboardFlowLaneStatus,
        stats: DashboardStats,
        windowLabel: String
    ) -> String {
        if let burstText = enrichmentBurstText(stats: stats) {
            return burstText
        }

        switch status {
        case .live:
            return "Enrichments live now"
        case .draining:
            return "Recent enrichments are draining backlog"
        case .queued:
            return "Backlog is queued without live enrichments"
        case .recent:
            return "Recent enrichments in \(windowLabel.lowercased())"
        case .idle:
            return "No recent enrichments"
        case .unavailable:
            return "Unavailable"
        }
    }

    private static func allCommitsStatusText(
        status: DashboardFlowLaneStatus,
        totalEvents: Int,
        windowLabel: String
    ) -> String {
        switch status {
        case .live:
            return "All commits live now"
        case .recent:
            return "Recent commits in \(windowLabel.lowercased())"
        case .idle:
            return totalEvents == 0 ? "No committed chunks" : "Commits idle"
        case .queued:
            return "Commits queued"
        case .draining:
            return "Commits draining"
        case .unavailable:
            return "Commits unavailable"
        }
    }

    private static func allCommitsLastEventText(
        status: DashboardFlowLaneStatus,
        totalEvents: Int,
        latestBucketCount: Int,
        windowLabel: String
    ) -> String {
        if totalEvents == 0 {
            return "No committed chunks in \(windowLabel)"
        }
        if latestBucketCount > 0 || status == .live {
            return "\(latestBucketCount) committed chunks in latest bucket"
        }
        return "\(totalEvents) committed chunks in \(windowLabel)"
    }

    private static func enrichmentBurstText(stats: DashboardStats) -> String? {
        guard stats.pendingEnrichmentCount > 0,
              let latestBucketCount = stats.recentEnrichmentBuckets.last,
              latestBucketCount >= 25 else {
            return nil
        }

        let earlierBucketTotal = stats.recentEnrichmentBuckets.dropLast().reduce(0, +)
        guard latestBucketCount >= max(earlierBucketTotal * 2, 25) else {
            return nil
        }

        let bucketMinutes = max(1, stats.activityWindowMinutes / max(stats.bucketCount, 1))
        let bucketLabel = DashboardMetricFormatter.shortWindowLabel(minutes: bucketMinutes)
        return "Backlog drain burst: \(latestBucketCount) enriched in latest \(bucketLabel)"
    }

    private static func storeQueueHealth(depth: Int, oldestAgeSeconds: Int?) -> DashboardStoreQueueHealth {
        guard depth > 0 else { return .empty }
        let oldest = oldestAgeSeconds ?? 0
        if depth >= 500 || oldest >= 300 {
            return .writerStuck
        }
        if depth >= 50 || oldest >= 30 {
            return .backlogAccumulating
        }
        return .activeDraining
    }

    private static func storeDepthText(totalDepth: Int, flushDepth: Int, replayDebtDepth: Int) -> String {
        if flushDepth > 0, replayDebtDepth > 0 {
            let flush = flushDepth == 1 ? "1 queued" : "\(flushDepth) queued"
            let replay = replayDebtDepth == 1 ? "1 replay debt" : "\(replayDebtDepth) replay debt"
            return "\(flush), \(replay)"
        }
        if replayDebtDepth > 0 {
            return replayDebtDepth == 1 ? "1 replay debt" : "\(replayDebtDepth) replay debt"
        }
        return totalDepth == 1 ? "1 queued" : "\(totalDepth) queued"
    }

    private static func storeOldestAgeText(_ oldestAgeSeconds: Int?) -> String {
        guard let seconds = oldestAgeSeconds else { return "oldest unknown" }
        if seconds < 60 {
            return "oldest \(seconds)s"
        }
        let minutes = Int((Double(seconds) / 60.0).rounded())
        return "oldest \(minutes)m"
    }

    private static func queueTitle(
        status: DashboardQueueStatus,
        backlogCount: Int,
        storeHealth: DashboardStoreQueueHealth
    ) -> String {
        if storeHealth == .writerStuck {
            return "Q: \(storeHealth.label)"
        }
        if storeHealth != .empty {
            return "Queue \(storeHealth.label)"
        }
        switch status {
        case .empty:
            return "Queue empty"
        case .stable:
            return backlogCount == 0 ? "Flow balanced" : "Queue stable"
        case .growing:
            return "Queue growing"
        case .draining:
            return "Queue draining"
        case .backlogged:
            return "Queue backlogged"
        case .unavailable:
            return "Queue unavailable"
        }
    }

    private static func queueDetail(
        status: DashboardQueueStatus,
        backlogCount: Int,
        storeHealth: DashboardStoreQueueHealth,
        storeDepth: Int,
        storeFlushDepth: Int,
        storeReplayDebtDepth: Int,
        storeOldestAgeSeconds: Int?,
        storeFlushRatePerMinute: Double,
        stats: DashboardStats,
        windowLabel: String
    ) -> String {
        if storeHealth != .empty {
            let breakdown = stats.replayDebtBreakdown.detailText
            let oldest = storeOldestAgeText(storeOldestAgeSeconds)
            guard storeFlushDepth > 0 else {
                return "\(breakdown); \(oldest)."
            }
            let rate = DashboardMetricFormatter.speedString(ratePerMinute: storeFlushRatePerMinute)
            return "\(breakdown); \(oldest); pending-store flush draining \(rate) over 60s."
        }

        if stats.replayDebtBreakdown.isPartial {
            return "\(stats.replayDebtBreakdown.detailText)."
        }

        switch status {
        case .empty:
            return "No chunks are waiting for enrichment."
        case .stable:
            return backlogCount == 0
                ? "Ingress and enrichment stayed balanced across \(windowLabel.lowercased())."
                : "\(backlogCount) chunks queued while ingress and enrichment remain balanced."
        case .growing:
            return "\(backlogCount) chunks are accumulating faster than enrichments are landing."
        case .draining:
            return "\(backlogCount) chunks remain queued, but enrichments are still landing."
        case .backlogged:
            return "\(backlogCount) chunks are queued with no enrichments in the live window."
        case .unavailable:
            return "Queue state cannot be trusted until the daemon comes back."
        }
    }
}

/// The independent flow series the redesigned dashboard plots as separately
/// scaled cards. `allCommits` is the raw committed chunk rate from `chunks`;
/// `agentStores` and `jsonlWatcher` remain source-specific slices so watcher
/// peaks do not crush the agent series flat. `enrichment` reuses the existing
/// enrichment lane.
enum PipelineSeries: String, Sendable, Equatable, CaseIterable, Identifiable {
    case allCommits
    case agentStores
    case jsonlWatcher
    case enrichment

    var id: String { rawValue }
}

extension DashboardFlowSummary {
    /// A single-series lane for one `PipelineSeries`, used by the redesigned
    /// per-series cards. Each lane carries ONLY its own `values` (empty
    /// secondary/tertiary), so `SparklineChartPresentation.maxValue` auto-fits
    /// the chart to that one series — the scale-disconnect fix is free.
    ///
    /// `ingress` is intentionally NOT removed: legacy callers (status popover,
    /// diagnostics "Writes" row) still read the combined lane.
    func lane(for series: PipelineSeries) -> DashboardFlowLane {
        switch series {
        case .allCommits:
            return allCommits
        case .enrichment:
            // Reuse the existing enrichment lane verbatim (keeps its
            // sparklineReferenceValue benchmark behaviour downstream).
            return enrichment
        case .agentStores:
            let agentValues = ingress.values
            let agentTotal = agentValues.reduce(0, +)
            let pendingFlushDepth = queue.storeFlushDepth
            let agentStatus = agentStoreStatus(values: agentValues, pendingFlushDepth: pendingFlushDepth)
            return DashboardFlowLane(
                name: "Agent MCP stores",
                status: agentStatus,
                statusText: agentStoreStatusText(
                    status: agentStatus,
                    totalEvents: agentTotal,
                    pendingFlushDepth: pendingFlushDepth,
                    windowLabel: ingress.windowLabel
                ),
                windowLabel: ingress.windowLabel,
                activityWindowMinutes: ingress.activityWindowMinutes,
                rateText: DashboardMetricFormatter.rateString(
                    totalEvents: agentTotal,
                    activityWindowMinutes: ingress.activityWindowMinutes
                ),
                volumeText: agentStoreVolumeText(
                    totalEvents: agentTotal,
                    pendingFlushDepth: pendingFlushDepth,
                    activityWindowMinutes: ingress.activityWindowMinutes
                ),
                lastEventText: agentStoreLastEventText(
                    status: agentStatus,
                    totalEvents: agentTotal,
                    pendingFlushDepth: pendingFlushDepth,
                    latestBucketCount: agentValues.last ?? 0,
                    windowLabel: ingress.windowLabel
                ),
                values: agentValues,
                sparklineLabel: "Agent MCP stores over \(ingress.windowLabel)",
                latestBucketName: "latest agent MCP store bucket",
                accentColor: BrainBarDesignTokens.Colors.seriesAgent,
                primarySeriesLabel: nil,
                secondaryValues: [],
                secondarySeriesLabel: nil,
                secondaryAccentColor: nil,
                tertiaryValues: [],
                tertiarySeriesLabel: nil,
                tertiaryAccentColor: nil
            )
        case .jsonlWatcher:
            let watcherValues = ingress.secondaryValues
            let watcherTotal = watcherValues.reduce(0, +)
            let watcherStatus = jsonlWatcherStatus(flowState: watcherFlowState)
            return DashboardFlowLane(
                name: "JSONL watcher",
                status: watcherStatus,
                statusText: watcherFlowState.label,
                windowLabel: ingress.windowLabel,
                activityWindowMinutes: ingress.activityWindowMinutes,
                rateText: DashboardMetricFormatter.rateString(
                    totalEvents: watcherTotal,
                    activityWindowMinutes: ingress.activityWindowMinutes
                ),
                volumeText: DashboardMetricFormatter.activitySummaryString(
                    totalEvents: watcherTotal,
                    activityWindowMinutes: ingress.activityWindowMinutes
                ),
                lastEventText: jsonlWatcherLastEventText(
                    status: watcherStatus,
                    totalEvents: watcherTotal,
                    latestBucketCount: watcherValues.last ?? 0,
                    windowLabel: ingress.windowLabel
                ),
                values: watcherValues,
                sparklineLabel: "JSONL watcher over \(ingress.windowLabel)",
                latestBucketName: "latest watcher bucket",
                accentColor: BrainBarDesignTokens.Colors.seriesWatcher,
                primarySeriesLabel: nil,
                secondaryValues: [],
                secondarySeriesLabel: nil,
                secondaryAccentColor: nil,
                tertiaryValues: [],
                tertiarySeriesLabel: nil,
                tertiaryAccentColor: nil
            )
        }
    }

    private func agentStoreStatus(values: [Int], pendingFlushDepth: Int) -> DashboardFlowLaneStatus {
        let totalEvents = values.reduce(0, +)
        if (values.last ?? 0) > 0 {
            return .live
        }
        if totalEvents > 0 {
            return .recent
        }
        if pendingFlushDepth > 0 {
            return .queued
        }
        return .idle
    }

    private func agentStoreFlushQueueText(_ depth: Int) -> String {
        depth == 1 ? "1 agent MCP store flush queued" : "\(depth) agent MCP stores flush queued"
    }

    private func shortAgentStoreFlushQueueText(_ depth: Int) -> String {
        depth == 1 ? "1 flush queued" : "\(depth) flush queued"
    }

    private func agentStoreVolumeText(
        totalEvents: Int,
        pendingFlushDepth: Int,
        activityWindowMinutes: Int
    ) -> String {
        let committed = DashboardMetricFormatter.activitySummaryString(
            totalEvents: totalEvents,
            activityWindowMinutes: activityWindowMinutes
        )
        guard pendingFlushDepth > 0 else { return committed }
        return "\(committed), \(shortAgentStoreFlushQueueText(pendingFlushDepth))"
    }

    private func agentStoreStatusText(
        status: DashboardFlowLaneStatus,
        totalEvents: Int,
        pendingFlushDepth: Int,
        windowLabel: String
    ) -> String {
        if status == .queued, pendingFlushDepth > 0 {
            return agentStoreFlushQueueText(pendingFlushDepth)
        }

        switch status {
        case .live:
            return "Agent MCP stores live now"
        case .recent:
            return "Recent agent MCP stores in \(windowLabel.lowercased())"
        case .idle:
            return totalEvents == 0 ? "No agent MCP stores" : "Agent MCP stores idle"
        case .queued:
            return "Agent MCP stores queued"
        case .draining:
            return "Agent MCP stores draining"
        case .unavailable:
            return "Agent MCP stores unavailable"
        }
    }

    private func agentStoreLastEventText(
        status: DashboardFlowLaneStatus,
        totalEvents: Int,
        pendingFlushDepth: Int,
        latestBucketCount: Int,
        windowLabel: String
    ) -> String {
        if pendingFlushDepth > 0 && totalEvents == 0 {
            return agentStoreFlushQueueText(pendingFlushDepth)
        }
        if totalEvents == 0 {
            return "No agent MCP stores in \(windowLabel)"
        }
        let queuedSuffix = pendingFlushDepth > 0 ? "; \(shortAgentStoreFlushQueueText(pendingFlushDepth))" : ""
        if latestBucketCount > 0 || status == .live {
            return "\(latestBucketCount) agent MCP stores in latest bucket\(queuedSuffix)"
        }
        return "\(totalEvents) agent MCP stores in \(windowLabel)\(queuedSuffix)"
    }

    private func jsonlWatcherStatus(flowState: WatcherFlowState) -> DashboardFlowLaneStatus {
        switch flowState {
        case .flowing:
            return .live
        case .stalled:
            return .queued
        case .runningNoRecentFlow:
            return .idle
        case .offline, .runningFlowUnverified, .unknown:
            return .unavailable
        }
    }

    private func jsonlWatcherLastEventText(
        status: DashboardFlowLaneStatus,
        totalEvents: Int,
        latestBucketCount: Int,
        windowLabel: String
    ) -> String {
        if totalEvents == 0 {
            return "No watcher writes in \(windowLabel)"
        }
        if latestBucketCount > 0 || status == .live {
            return "\(latestBucketCount) watcher writes in latest bucket"
        }
        return "\(totalEvents) watcher writes in \(windowLabel)"
    }
}

enum PipelineState: String, Sendable, Equatable {
    case degraded
    case indexing
    case enriching
    case idle

    static func derive(daemon: DaemonHealthSnapshot?, stats: DashboardStats, now: Date = Date()) -> PipelineState {
        let summary = DashboardFlowSummary.derive(daemon: daemon, stats: stats, now: now)
        if summary.isUnavailable {
            return .degraded
        }
        if summary.ingress.status == .live || summary.ingress.status == .recent || summary.queue.status == .growing {
            return .indexing
        }
        if summary.enrichment.status == .live ||
            summary.enrichment.status == .queued ||
            (summary.enrichment.status == .recent && summary.queue.status == .draining) ||
            summary.queue.status == .draining ||
            summary.queue.status == .backlogged {
            return .enriching
        }
        return .idle
    }

    var label: String {
        switch self {
        case .degraded: return "Degraded"
        case .indexing: return "Indexing"
        case .enriching: return "Enriching"
        case .idle: return "Idle"
        }
    }

    var symbolName: String {
        switch self {
        case .degraded: return "exclamationmark.triangle.fill"
        case .indexing: return "waveform.path.ecg"
        case .enriching: return "sparkles"
        case .idle: return "checkmark.circle.fill"
        }
    }

    var color: NSColor {
        stateTheme.theme.color
    }

    var stateTheme: BrainBarStateTheme {
        switch self {
        case .degraded:
            return .degraded
        case .indexing:
            return .loading
        case .enriching:
            return .active
        case .idle:
            return .idle
        }
    }
}
