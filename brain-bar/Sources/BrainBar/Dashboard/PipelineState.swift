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
        switch self {
        case .live:
            return .systemGreen
        case .queued:
            return .systemOrange
        case .idle:
            return .secondaryLabelColor
        case .unavailable:
            return .systemRed
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

    static func derive(daemon: DaemonHealthSnapshot?, stats: DashboardStats) -> PipelineIndicators {
        let indexingStatus: PipelineIndicatorStatus
        let enrichingStatus: PipelineIndicatorStatus

        if daemon?.isResponsive != true {
            indexingStatus = .unavailable
            enrichingStatus = .unavailable
        } else {
            let recentWrites = stats.recentActivityBuckets.suffix(2).reduce(0, +)
            let recentEnrichments = stats.recentEnrichmentBuckets.suffix(2).reduce(0, +)

            indexingStatus = recentWrites > 0 ? .live : .idle
            if recentEnrichments > 0 {
                enrichingStatus = .live
            } else if stats.pendingEnrichmentCount > 0 {
                enrichingStatus = .queued
            } else {
                enrichingStatus = .idle
            }
        }

        return PipelineIndicators(
            indexing: PipelineIndicator(name: "Indexing", status: indexingStatus),
            enriching: PipelineIndicator(name: "Enriching", status: enrichingStatus)
        )
    }
}

struct PipelineActivityTrack: Sendable, Equatable {
    let name: String
    let symbolName: String
    let status: PipelineIndicatorStatus
    let rateText: String
    let detailText: String
    let values: [Int]
}

struct PipelineActivityTracks: Sendable, Equatable {
    let indexing: PipelineActivityTrack
    let enriching: PipelineActivityTrack

    static func derive(
        daemon: DaemonHealthSnapshot?,
        stats: DashboardStats,
        activityWindowMinutes: Double = 30,
        trailingBucketCount: Int = 2
    ) -> PipelineActivityTracks {
        let indicators = PipelineIndicators.derive(daemon: daemon, stats: stats)
        let activityWindowLabel = windowLabel(activityWindowMinutes)

        let indexingRatePerMinute = recentRatePerMinute(
            values: stats.recentActivityBuckets,
            activityWindowMinutes: activityWindowMinutes,
            trailingBucketCount: trailingBucketCount
        )
        let recentWrites = stats.recentActivityBuckets.reduce(0, +)
        let indexingRateText = recentWrites > 0
            ? DashboardMetricFormatter.speedString(ratePerMinute: indexingRatePerMinute)
            : "idle"
        let indexingDetailText = recentWrites > 0
            ? "\(recentWrites) chunks in last \(activityWindowLabel)"
            : "No new chunks in last \(activityWindowLabel)"

        let recentCompletions = stats.recentEnrichmentBuckets.reduce(0, +)
        let enrichmentDisplayRatePerMinute = displayedRatePerMinute(
            primaryRatePerMinute: stats.enrichmentRatePerMinute,
            values: stats.recentEnrichmentBuckets,
            activityWindowMinutes: activityWindowMinutes,
            trailingBucketCount: trailingBucketCount
        )
        let enrichingRateText: String
        if enrichmentDisplayRatePerMinute > 0 {
            enrichingRateText = DashboardMetricFormatter.speedString(ratePerMinute: enrichmentDisplayRatePerMinute)
        } else if indicators.enriching.status == .queued {
            enrichingRateText = "queued"
        } else {
            enrichingRateText = "idle"
        }

        let enrichingDetailText: String
        if stats.pendingEnrichmentCount > 0, recentCompletions > 0 {
            enrichingDetailText = "\(stats.pendingEnrichmentCount) pending · \(recentCompletions) done in last \(activityWindowLabel)"
        } else if stats.pendingEnrichmentCount > 0 {
            enrichingDetailText = "\(stats.pendingEnrichmentCount) pending"
        } else if recentCompletions > 0 {
            enrichingDetailText = "\(recentCompletions) done in last \(activityWindowLabel)"
        } else {
            enrichingDetailText = "No completions in last \(activityWindowLabel)"
        }

        return PipelineActivityTracks(
            indexing: PipelineActivityTrack(
                name: "Indexing",
                symbolName: "server.rack",
                status: indicators.indexing.status,
                rateText: indexingRateText,
                detailText: indexingDetailText,
                values: stats.recentActivityBuckets
            ),
            enriching: PipelineActivityTrack(
                name: "Enriching",
                symbolName: "sparkles",
                status: indicators.enriching.status,
                rateText: enrichingRateText,
                detailText: enrichingDetailText,
                values: stats.recentEnrichmentBuckets
            )
        )
    }

    static func displayedRatePerMinute(
        primaryRatePerMinute: Double,
        values: [Int],
        activityWindowMinutes: Double = 30,
        trailingBucketCount: Int = 2
    ) -> Double {
        let instantaneousRate = max(primaryRatePerMinute, 0)
        guard instantaneousRate == 0 else { return instantaneousRate }
        return recentRatePerMinute(
            values: values,
            activityWindowMinutes: activityWindowMinutes,
            trailingBucketCount: trailingBucketCount
        )
    }

    static func recentRatePerMinute(
        values: [Int],
        activityWindowMinutes: Double,
        trailingBucketCount: Int
    ) -> Double {
        guard !values.isEmpty, activityWindowMinutes > 0 else { return 0 }
        let recentBucketCount = max(1, min(trailingBucketCount, values.count))
        let recentCount = values.suffix(recentBucketCount).reduce(0, +)
        let bucketWidthMinutes = activityWindowMinutes / Double(values.count)
        let recentWindowMinutes = bucketWidthMinutes * Double(recentBucketCount)
        guard recentWindowMinutes > 0 else { return 0 }
        return Double(recentCount) / recentWindowMinutes
    }

    private static func windowLabel(_ activityWindowMinutes: Double) -> String {
        guard activityWindowMinutes.isFinite, activityWindowMinutes > 0 else {
            return "0m"
        }
        let roundedMinutes = activityWindowMinutes.rounded()
        if roundedMinutes == activityWindowMinutes {
            return "\(Int(roundedMinutes))m"
        }
        return "\(String(format: "%.1f", activityWindowMinutes))m"
    }
}

enum PipelineState: String, Sendable, Equatable {
    case degraded
    case indexing
    case enriching
    case idle

    static func derive(daemon: DaemonHealthSnapshot?, stats: DashboardStats) -> PipelineState {
        guard let daemon else { return .degraded }
        guard daemon.isResponsive else { return .degraded }

        let recentWrites = stats.recentActivityBuckets.reduce(0, +)
        if recentWrites > 0 {
            return .indexing
        }
        if stats.pendingEnrichmentCount > 0 {
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
        switch self {
        case .degraded:
            return .systemOrange
        case .indexing:
            return .systemBlue
        case .enriching:
            return .systemPurple
        case .idle:
            return .systemGreen
        }
    }
}
