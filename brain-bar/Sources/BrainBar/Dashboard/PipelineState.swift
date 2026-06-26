import AppKit
import Foundation

typealias DashboardStats = BrainDatabase.DashboardStats

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
    let ingress: DashboardFlowLane
    let queue: DashboardQueueSummary
    let enrichment: DashboardFlowLane
    let watcherHealth: DashboardStats.WatcherHealth?
    let watcherHealthIsFresh: Bool

    var isUnavailable: Bool {
        ingress.status == .unavailable || enrichment.status == .unavailable || queue.status == .unavailable
    }

    static func derive(daemon: DaemonHealthSnapshot?, stats: DashboardStats, now: Date = Date()) -> DashboardFlowSummary {
        let windowLabel = DashboardMetricFormatter.windowLabel(minutes: stats.activityWindowMinutes)
        let agentStoresColor = BrainBarDesignTokens.Colors.seriesAgent
        let jsonlWatcherColor = BrainBarDesignTokens.Colors.seriesWatcher
        let enrichmentColor = BrainBarStateTheme.active.theme.color

        let writesLive = stats.eventIsLive(stats.lastWriteAt, now: now)
        let enrichmentsLive = stats.eventIsLive(stats.lastEnrichedAt, now: now)
        let backlogCount = stats.pendingEnrichmentCount
        let storeOldestAgeSeconds = stats.pendingStoreOldestQueuedAt.map { max(0, Int(now.timeIntervalSince($0).rounded())) }
        let storeHealth = storeQueueHealth(depth: stats.pendingStoreQueueDepth, oldestAgeSeconds: storeOldestAgeSeconds)

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
                primarySeriesLabel: "Agent stores",
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
                storeOldestAgeSeconds: storeOldestAgeSeconds,
                storeFlushRatePerMinute: stats.pendingStoreFlushRatePerMinute,
                storeDepthText: storeDepthText(stats.pendingStoreQueueDepth),
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

    private static func storeDepthText(_ depth: Int) -> String {
        depth == 1 ? "1 queued" : "\(depth) queued"
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
        storeOldestAgeSeconds: Int?,
        storeFlushRatePerMinute: Double,
        stats: DashboardStats,
        windowLabel: String
    ) -> String {
        if storeHealth != .empty {
            let depth = storeDepthText(storeDepth)
            let oldest = storeOldestAgeText(storeOldestAgeSeconds)
            let rate = DashboardMetricFormatter.speedString(ratePerMinute: storeFlushRatePerMinute)
            return "\(depth), \(oldest), draining \(rate) over 60s."
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

/// The three independent flow series the redesigned dashboard plots as
/// separately-scaled cards. `agentStores` and `jsonlWatcher` were previously
/// co-normalized into a single `ingress` lane (the watcher peak crushed the
/// agent series flat). `enrichment` reuses the existing enrichment lane.
enum PipelineSeries: String, Sendable, Equatable, CaseIterable, Identifiable {
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
        case .enrichment:
            // Reuse the existing enrichment lane verbatim (keeps its
            // sparklineReferenceValue benchmark behaviour downstream).
            return enrichment
        case .agentStores:
            let agentValues = ingress.values
            let agentTotal = agentValues.reduce(0, +)
            let agentStatus = agentStoreStatus(values: agentValues)
            return DashboardFlowLane(
                name: "Agent stores",
                status: agentStatus,
                statusText: agentStoreStatusText(
                    status: agentStatus,
                    totalEvents: agentTotal,
                    windowLabel: ingress.windowLabel
                ),
                windowLabel: ingress.windowLabel,
                activityWindowMinutes: ingress.activityWindowMinutes,
                rateText: DashboardMetricFormatter.rateString(
                    totalEvents: agentTotal,
                    activityWindowMinutes: ingress.activityWindowMinutes
                ),
                volumeText: DashboardMetricFormatter.activitySummaryString(
                    totalEvents: agentTotal,
                    activityWindowMinutes: ingress.activityWindowMinutes
                ),
                lastEventText: agentStoreLastEventText(
                    status: agentStatus,
                    totalEvents: agentTotal,
                    latestBucketCount: agentValues.last ?? 0,
                    windowLabel: ingress.windowLabel
                ),
                values: agentValues,
                sparklineLabel: "Agent stores over \(ingress.windowLabel)",
                latestBucketName: "latest agent-store bucket",
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
            let watcherStatus = jsonlWatcherStatus(
                values: watcherValues,
                health: watcherHealth,
                healthIsFresh: watcherHealthIsFresh
            )
            let watcherStatusText = jsonlWatcherStatusText(
                status: watcherStatus,
                totalEvents: watcherTotal,
                windowLabel: ingress.windowLabel
            )
            return DashboardFlowLane(
                name: "JSONL watcher",
                status: watcherStatus,
                statusText: watcherStatusText,
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

    private func agentStoreStatus(values: [Int]) -> DashboardFlowLaneStatus {
        let totalEvents = values.reduce(0, +)
        if (values.last ?? 0) > 0 {
            return .live
        }
        if totalEvents > 0 {
            return .recent
        }
        return .idle
    }

    private func agentStoreStatusText(
        status: DashboardFlowLaneStatus,
        totalEvents: Int,
        windowLabel: String
    ) -> String {
        switch status {
        case .live:
            return "Agent stores live now"
        case .recent:
            return "Recent agent-store writes in \(windowLabel.lowercased())"
        case .idle:
            return totalEvents == 0 ? "No agent-store writes" : "Agent stores idle"
        case .queued:
            return "Agent stores queued"
        case .draining:
            return "Agent stores draining"
        case .unavailable:
            return "Agent stores unavailable"
        }
    }

    private func agentStoreLastEventText(
        status: DashboardFlowLaneStatus,
        totalEvents: Int,
        latestBucketCount: Int,
        windowLabel: String
    ) -> String {
        if totalEvents == 0 {
            return "No agent-store writes in \(windowLabel)"
        }
        if latestBucketCount > 0 || status == .live {
            return "\(latestBucketCount) agent-store writes in latest bucket"
        }
        return "\(totalEvents) agent-store writes in \(windowLabel)"
    }

    private func jsonlWatcherStatus(
        values: [Int],
        health: DashboardStats.WatcherHealth?,
        healthIsFresh: Bool
    ) -> DashboardFlowLaneStatus {
        let totalEvents = values.reduce(0, +)
        guard health != nil else {
            return .unavailable
        }
        if !healthIsFresh {
            return .unavailable
        }
        if health?.alerting == true {
            return .unavailable
        }
        if totalEvents == 0,
           let health,
           health.activeEntriesPerMinute > 0,
           health.realtimeInsertsPerMinute <= 0 {
            return .unavailable
        }
        if (values.last ?? 0) > 0 {
            return .live
        }
        if totalEvents > 0 {
            return .recent
        }
        return .idle
    }

    private func jsonlWatcherStatusText(
        status: DashboardFlowLaneStatus,
        totalEvents: Int,
        windowLabel: String
    ) -> String {
        switch status {
        case .live:
            return "JSONL watcher live now"
        case .recent:
            return "Recent JSONL watcher writes in \(windowLabel.lowercased())"
        case .unavailable:
            if watcherHealth == nil {
                return "Watcher health unavailable"
            }
            if !watcherHealthIsFresh {
                return "Watcher health stale"
            }
            if watcherHealth?.alerting == true {
                return "Watcher coverage alert"
            }
            return "Watcher stale: JSONL activity is not landing"
        case .idle:
            return totalEvents == 0 ? "No JSONL watcher writes" : "JSONL watcher idle"
        case .queued:
            return "JSONL watcher queued"
        case .draining:
            return "JSONL watcher draining"
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
