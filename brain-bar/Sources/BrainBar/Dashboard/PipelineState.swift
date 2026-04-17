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
